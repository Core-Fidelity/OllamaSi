package qwen3next

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

var (
	_ kvcache.Cache           = (*HybridCache)(nil)
	_ kvcache.CheckpointCache = (*HybridCache)(nil)
)

// HybridCache adapts the shared recurrent cache base for Qwen3-Next naming.
type HybridCache struct {
	*kvcache.Recurrent

	// isRecurrent[i] is true if layer i is a recurrent (SSM) layer,
	// false if it is a full attention layer.
	isRecurrent []bool
}

func NewHybridCache(shift func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error), convDim, convChannels, deltaStateSize int, isRecurrent []bool) *HybridCache {
	base := kvcache.NewRecurrentCache(kvcache.RecurrentConfig{
		Shift:               shift,
		ConvDim:             convDim,
		ConvChannels:        convChannels,
		RecurrentStateSize:  deltaStateSize,
		CheckpointLogPrefix: "qwen3next",
	})
	return &HybridCache{Recurrent: base, isRecurrent: isRecurrent}
}

// DeltaState returns the delta state for current batch sequences as
// [headVDim, headVDim*numVHeads, nSeqs].
func (c *HybridCache) DeltaState(ctx ml.Context, layer int, headVDim, numVHeads int) (ml.Tensor, error) {
	return c.RecurrentState(ctx, layer, headVDim, headVDim*numVHeads)
}

// UpdateDeltaState writes a new delta state for current batch sequences.
func (c *HybridCache) UpdateDeltaState(ctx ml.Context, layer int, newState ml.Tensor) {
	c.UpdateRecurrentState(ctx, layer, newState)
}

func (c *HybridCache) seqTokens() int {
	return c.SeqTokens()
}

func (c *HybridCache) numSeqs() int {
	return c.NumSeqs()
}

// IsRecurrentLayer returns true if layer i is a recurrent (SSM) layer.
func (c *HybridCache) IsRecurrentLayer(i int) bool {
	if c.isRecurrent == nil || i >= len(c.isRecurrent) {
		return false
	}
	return c.isRecurrent[i]
}

// SavePrefixData saves both causal K/V data (for attention layers) and
// recurrent state (conv + delta, for recurrent layers) into a combined
// per-layer format. For each layer, the data is:
//   - Attention layer: keys = K rows, vals = V rows (position-indexed)
//   - Recurrent layer: keys = conv state slot row, vals = delta state slot row (fixed-size)
func (c *HybridCache) SavePrefixData(seqId int, numKeep int32) (keys [][]byte, vals [][]byte, err error) {
	if numKeep <= 0 || c.isRecurrent == nil {
		return nil, nil, nil
	}

	// Get the causal K/V data (attention layers have data, recurrent layers will be nil).
	causalKeys, causalVals, err := c.Recurrent.KV().SavePrefixData(seqId, numKeep)
	if err != nil {
		return nil, nil, err
	}

	// Save recurrent state (conv + delta) for recurrent layers.
	recurrentKeys, recurrentVals, err := c.Recurrent.SaveRecurrentState(seqId)
	if err != nil {
		return nil, nil, err
	}

	// Determine total layers.
	numLayers := len(c.isRecurrent)
	if len(causalKeys) > numLayers {
		numLayers = len(causalKeys)
	}
	if len(recurrentKeys) > numLayers {
		numLayers = len(recurrentKeys)
	}

	keys = make([][]byte, numLayers)
	vals = make([][]byte, numLayers)

	for layer := 0; layer < numLayers; layer++ {
		isRecurrent := layer < len(c.isRecurrent) && c.isRecurrent[layer]

		if isRecurrent {
			// Recurrent layer: use conv/delta state.
			if layer < len(recurrentKeys) {
				keys[layer] = recurrentKeys[layer]
				vals[layer] = recurrentVals[layer]
			}
		} else {
			// Attention layer: use causal K/V data.
			if layer < len(causalKeys) {
				keys[layer] = causalKeys[layer]
				vals[layer] = causalVals[layer]
			}
		}
	}

	return keys, vals, nil
}

// LoadPrefixData restores causal K/V for attention layers and recurrent state
// for recurrent layers.
func (c *HybridCache) LoadPrefixData(seqId int, numKeep int32, keys [][]byte, vals [][]byte) error {
	if numKeep <= 0 || c.isRecurrent == nil {
		return nil
	}
	if len(keys) != len(vals) {
		return nil // let the causal LoadPrefixData validate
	}
	if len(keys) == 0 {
		return nil
	}

	// Split keys/vals into causal and recurrent per-layer slices.
	numLayers := len(c.isRecurrent)
	if len(keys) < numLayers {
		numLayers = len(keys)
	}

	causalKeys := make([][]byte, 0)
	causalVals := make([][]byte, 0)
	recurrentKeys := make([][]byte, 0)
	recurrentVals := make([][]byte, 0)

	causalLayerCount := 0
	recurrentLayerCount := 0

	for layer := 0; layer < numLayers; layer++ {
		isRecurrent := c.isRecurrent[layer]
		if isRecurrent {
			recurrentLayerCount++
		} else {
			causalLayerCount++
		}
	}

	// Build index-mapped slices. Causal K/V uses a dense layer index (0, 1, ... for
	// attention layers only), while recurrent state uses a dense recurrent layer index.
	causalKeys = make([][]byte, causalLayerCount)
	causalVals = make([][]byte, causalLayerCount)
	recurrentKeys = make([][]byte, recurrentLayerCount)
	recurrentVals = make([][]byte, recurrentLayerCount)

	ci := 0
	ri := 0
	for layer := 0; layer < numLayers; layer++ {
		if c.isRecurrent[layer] {
			recurrentKeys[ri] = keys[layer]
			recurrentVals[ri] = vals[layer]
			ri++
		} else {
			causalKeys[ci] = keys[layer]
			causalVals[ci] = vals[layer]
			ci++
		}
	}

	// Load causal K/V for attention layers.
	if err := c.Recurrent.KV().LoadPrefixData(seqId, numKeep, causalKeys, causalVals); err != nil {
		return err
	}

	// Load recurrent state for recurrent layers.
	if err := c.Recurrent.LoadRecurrentState(seqId, recurrentKeys, recurrentVals); err != nil {
		return err
	}

	return nil
}

// Keep qwen3next behavior for partial mid-sequence removals.
func (c *HybridCache) Remove(seq int, beginIndex, endIndex int32) error {
	if beginIndex > 0 && endIndex != math.MaxInt32 {
		return kvcache.ErrNotSupported
	}
	return c.Recurrent.Remove(seq, beginIndex, endIndex)
}