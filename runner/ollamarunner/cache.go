package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// Maintain a minimum amount of host RAM after prefix-cache load succeeds.
//
// Percentage-based thresholds were too aggressive on 8 GB Apple Silicon
// systems and caused successful fast-path loads to be immediately discarded,
// forcing prompt processing back onto slower unpinned execution.
//
// A fixed absolute floor preserves zero-copy fallback for loads that fail
// (ENOMEM) while keeping small-to-medium prefixes pinned when enough
// working-set headroom remains.
const minPrefixCacheHeadroomBytes = 300 << 20

type InputCache struct {
	// context window size (per slot)
	numCtx int32

	// does the cache store data or do we need to always send the full input?
	// note that when enabled is false the underlying cache may either be nil
	// or a non-nil dummy that doesn't actually store anything
	enabled bool

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	cache kvcache.Cache

	prefixCache *PrefixCache
	modelPath   string
	archParams  ArchParams
}

func NewInputCache(model model.Model, kvCacheType string, kvSize int32, numSlots int, batchSize int, multiUserCache bool, modelPath string, archParams ArchParams, prefixCache *PrefixCache) (*InputCache, error) {
	numCtx := kvSize / int32(numSlots)

	if int(numCtx) < batchSize {
		return nil, fmt.Errorf("kv size must be at least as large as batch size * parallel (kv: %v batch: %v parallel: %v)", kvSize, batchSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{Id: i}
	}

	cache := model.Config().Cache
	if cache != nil {
		cache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), numSlots, int(numCtx), batchSize)
	}

	return &InputCache{
		numCtx:         numCtx,
		enabled:        cache != nil,
		slots:          slots,
		multiUserCache: multiUserCache,
		cache:          cache,
		prefixCache:    prefixCache,
		modelPath:      modelPath,
		archParams:     archParams,
	}, nil
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch s {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	default:
		return ml.DTypeF16
	}
}

func (c *InputCache) Close() {
	if c != nil && c.cache != nil {
		c.cache.Close()
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and processBatch

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []*input.Input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time

	// SavePrefixSize tracks a pending prefix save once the first full prefix has been computed.
	SavePrefixSize int32
}

func (c *InputCache) LoadCacheSlot(prompt []*input.Input, cachePrompt bool, numKeep int32) (*InputCacheSlot, []*input.Input, error) {
	var slot *InputCacheSlot
	var numPast int32
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it keeps the footprint of the cache small, which improves throughput.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache.
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.SavePrefixSize = 0

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if c.cache != nil {
		if numPast > 0 {
			// Recurrent caches use checkpoints to pick a safe resume position.
			if cc, ok := c.cache.(kvcache.CheckpointCache); ok {
				if restored, ok := cc.PrepareRestore(slot.Id, numPast); ok {
					numPast = restored
				} else {
					numPast = 0
				}
			} else if !c.cache.CanResume(slot.Id, numPast) {
				numPast = 0
			}
		}

		err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
		if err != nil {
			// Some models don't support partial erasure
			err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			if err != nil {
				return nil, nil, err
			}
			numPast = 0
		}

		// Only attempt a prefix-cache restore when we have a clean slate
		// (no prior state in the slot). Loading a prefix over existing KV
		// data would create a broken prefix/mutable boundary where tokens
		// overlap. The slot must be empty before we call LoadPrefixData.
		if c.prefixCache != nil && numKeep > 0 && numPast == 0 && len(prompt) >= int(numKeep) {
			key := c.prefixCache.Key(c.modelPath, c.archParams, prefixInputTokens(prompt[:int(numKeep)]), numKeep)
			keys, vals, hit, err := c.prefixCache.Load(key)
			if err == nil && hit {
				if available, ok := availablePrefixCacheHeadroomBytes(); ok && available < minPrefixCacheHeadroomBytes {
					slog.Info("prefix cache skipped: insufficient headroom",
						"available_gb", fmt.Sprintf("%.2f", float64(available)/float64(1<<30)),
						"min_headroom_gb", fmt.Sprintf("%.2f", float64(minPrefixCacheHeadroomBytes)/float64(1<<30)))
				} else if loadErr := c.cache.LoadPrefixData(slot.Id, numKeep, keys, vals); loadErr == nil {
					slog.Info(fmt.Sprintf("prefix cache hit: skipping %d tokens", numKeep))
					numPast = numKeep
				}
			}
			if numPast == 0 {
				slot.SavePrefixSize = numKeep
			}
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", int32(len(prompt))-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func prefixInputTokens(inputs []*input.Input) []int32 {
	tokens := make([]int32, len(inputs))
	for i, inp := range inputs {
		tokens[i] = inp.Token
	}
	return tokens
}

func (c *InputCache) findLongestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == int32(len(longestSlot.Inputs)) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]*input.Input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []*input.Input, b []*input.Input) int32 {
	var count int32

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i].Token != b[i].Token || a[i].MultimodalHash != b[i].MultimodalHash {
			break
		}

		count++
	}

	return count
}

// ShiftDiscard computes how many inputs can be discarded from the cache. Inputs in the same batch
// are discarded together.
func (c *InputCache) ShiftDiscard(inputs []*input.Input, numKeep int32) int32 {
	targetFree := max((c.numCtx-numKeep)/2, 1)
	currentFree := c.numCtx - int32(len(inputs))

	var discard, sameBatch int32
	for _, input := range inputs[numKeep:] {
		if sameBatch <= 0 && currentFree >= targetFree {
			break
		}

		sameBatch--
		currentFree++
		discard++

		if input.SameBatch > 0 {
			sameBatch = int32(input.SameBatch)
		}
	}

	return discard
}

// Frees up space in the KV cache by deleting the oldest half of history and shifting
// the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int32) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := int32(len(slot.Inputs))
	discard := c.ShiftDiscard(slot.Inputs, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if c.cache != nil {
		c.cache.SetPrefixSize(slot.Id, numKeep)
		if err := c.cache.Remove(slot.Id, numKeep, numKeep+discard); err != nil {
			return fmt.Errorf("kv cache shift failed: %w", err)
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
