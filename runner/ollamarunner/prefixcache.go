package ollamarunner

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	prefixCacheMagic    = "OLKVPFX1"
	prefixCacheMagicV2  = "OLKVPFX2"

	// Layer type constants for the V2 format
	layerTypeCausal    = 0
	layerTypeRecurrent = 1
)

type ArchParams struct {
	NumLayers  int
	NumKVHeads int
	HeadDim    int
	DType      int
	RopeFreq   float32

	// IsRecurrent indicates which layers are recurrent (true) vs attention (false).
	// nil means "all layers are attention" (V1 format compatibility).
	//
	// The cache key includes the recurrent layout so that hybrid models produce
	// different entries from pure-attention models with identical arch params.
	// Without this, a hybrid and a pure-attention model sharing the same
	// NumLayers/NumKVHeads would collide on the same prefix-cache key and
	// load incompatible KV data.
	IsRecurrent []bool
}

func (a ArchParams) numRecurrentLayers() int {
	if a.IsRecurrent == nil {
		return 0
	}
	n := 0
	for _, r := range a.IsRecurrent {
		if r {
			n++
		}
	}
	return n
}

type PrefixCache struct {
	dir     string
	maxSize int64
}

func NewPrefixCache(dir string, maxSize int64) (*PrefixCache, error) {
	if maxSize <= 0 {
		maxSize = 4 << 30
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &PrefixCache{dir: dir, maxSize: maxSize}, nil
}

func (pc *PrefixCache) Key(modelPath string, arch ArchParams, tokens []int32, numKeep int32) string {
	if int(numKeep) < len(tokens) {
		tokens = tokens[:numKeep]
	}

	var tokenBytes bytes.Buffer
	_ = binary.Write(&tokenBytes, binary.LittleEndian, tokens)

	archString := fmt.Sprintf(
		"layers=%d,heads=%d,headdim=%d,dtype=%d,rope=%.4f",
		arch.NumLayers, arch.NumKVHeads, arch.HeadDim, arch.DType, arch.RopeFreq,
	)

	// Include recurrent layer pattern in the key so that hybrid models
	// produce different keys from pure-attention models with the same arch params.
	if arch.IsRecurrent != nil {
		var sb bytes.Buffer
		for _, r := range arch.IsRecurrent {
			if r {
				sb.WriteByte('R')
			} else {
				sb.WriteByte('A')
			}
		}
		archString += ",layout=" + sb.String()
	}

	sum := sha256.Sum256([]byte(modelPath + "|" + archString + "|" + tokenBytes.String()))
	return fmt.Sprintf("%x", sum[:])
}

func (pc *PrefixCache) Save(key string, keys [][]byte, vals [][]byte, numKeep int32, arch ArchParams) error {
	if len(keys) != len(vals) {
		return fmt.Errorf("prefix cache key/value layer mismatch: %d != %d", len(keys), len(vals))
	}
	if len(keys) == 0 || numKeep <= 0 {
		return nil
	}

	if arch.IsRecurrent != nil {
		return pc.saveV2(key, keys, vals, numKeep, arch)
	}

	return pc.saveV1(key, keys, vals, numKeep)
}

// saveV1 writes the original V1 format (all layers are causal, uniform row sizes).
func (pc *PrefixCache) saveV1(key string, keys [][]byte, vals [][]byte, numKeep int32) error {
	// Validate uniform row sizes.
	keyRowBytes := len(keys[0]) / int(numKeep)
	valRowBytes := len(vals[0]) / int(numKeep)
	for layer := range keys {
		if len(keys[layer]) != keyRowBytes*int(numKeep) {
			return fmt.Errorf("prefix cache key layer %d size mismatch", layer)
		}
		if len(vals[layer]) != valRowBytes*int(numKeep) {
			return fmt.Errorf("prefix cache value layer %d size mismatch", layer)
		}
	}

	path := pc.pathForKey(key)
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	cleanup := func(err error) error {
		f.Close()
		_ = os.Remove(tmpPath)
		return err
	}

	if _, err := f.Write([]byte(prefixCacheMagic)); err != nil {
		return cleanup(err)
	}

	keyHash := sha256.Sum256([]byte(key))
	header := []uint32{
		uint32(len(keys)),
		uint32(numKeep),
		uint32(keyRowBytes),
		uint32(valRowBytes),
	}
	for _, v := range header {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return cleanup(err)
		}
	}
	if _, err := f.Write(keyHash[:]); err != nil {
		return cleanup(err)
	}

	for layer := range keys {
		if _, err := f.Write(keys[layer]); err != nil {
			return cleanup(err)
		}
		if _, err := f.Write(vals[layer]); err != nil {
			return cleanup(err)
		}
	}

	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return os.Rename(tmpPath, path)
}

// saveV2 writes the V2 format that supports mixed causal and recurrent layers.
//
// Format:
//
//	magic       [8]byte  "OLKVPFX2"
//	keyHash     [32]byte sha256(key)
//	numLayers   uint32
//	numKeep     uint32
//	for each layer:
//	  layerType   uint8  (0=causal, 1=recurrent)
//	  keyLen      uint32 length of key data in bytes
//	  valLen      uint32 length of val data in bytes
//	  keyData     [keyLen]byte
//	  valData     [valLen]byte
func (pc *PrefixCache) saveV2(key string, keys [][]byte, vals [][]byte, numKeep int32, arch ArchParams) error {
	path := pc.pathForKey(key)
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	cleanup := func(err error) error {
		f.Close()
		_ = os.Remove(tmpPath)
		return err
	}

	if _, err := f.Write([]byte(prefixCacheMagicV2)); err != nil {
		return cleanup(err)
	}

	keyHash := sha256.Sum256([]byte(key))
	if _, err := f.Write(keyHash[:]); err != nil {
		return cleanup(err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint32(len(keys))); err != nil {
		return cleanup(err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(numKeep)); err != nil {
		return cleanup(err)
	}

	for layer := range keys {
		layerType := uint8(layerTypeCausal)
		if arch.IsRecurrent != nil && layer < len(arch.IsRecurrent) && arch.IsRecurrent[layer] {
			layerType = uint8(layerTypeRecurrent)
		}

		if err := binary.Write(f, binary.LittleEndian, layerType); err != nil {
			return cleanup(err)
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(len(keys[layer]))); err != nil {
			return cleanup(err)
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(len(vals[layer]))); err != nil {
			return cleanup(err)
		}
		if len(keys[layer]) > 0 {
			if _, err := f.Write(keys[layer]); err != nil {
				return cleanup(err)
			}
		}
		if len(vals[layer]) > 0 {
			if _, err := f.Write(vals[layer]); err != nil {
				return cleanup(err)
			}
		}
	}

	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return os.Rename(tmpPath, path)
}

func (pc *PrefixCache) Load(key string) (keys [][]byte, vals [][]byte, hit bool, err error) {
	data, err := os.ReadFile(pc.pathForKey(key))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil, false, nil
		}
		return nil, nil, false, nil
	}

	if len(data) < 8 {
		return nil, nil, false, nil
	}

	magic := string(data[:8])
	switch magic {
	case prefixCacheMagicV2:
		return pc.loadV2(data, key)
	case prefixCacheMagic:
		return pc.loadV1(data, key)
	default:
		return nil, nil, false, nil
	}
}

func (pc *PrefixCache) loadV2(data []byte, key string) (keys [][]byte, vals [][]byte, hit bool, err error) {
	const headerSize = 8 + 32 + 4 + 4 // magic + keyHash + numLayers + numKeep
	if len(data) < headerSize {
		return nil, nil, false, nil
	}

	var expectedHash [32]byte
	copy(expectedHash[:], data[8:40])
	if expectedHash != sha256.Sum256([]byte(key)) {
		return nil, nil, false, nil
	}

	numLayers := binary.LittleEndian.Uint32(data[40:44])
	numKeep := binary.LittleEndian.Uint32(data[44:48])

	offset := headerSize
	keys = make([][]byte, numLayers)
	vals = make([][]byte, numLayers)

	for layer := uint32(0); layer < numLayers; layer++ {
		if offset+1+4+4 > len(data) {
			return nil, nil, false, nil
		}
		_ = data[offset] // layerType — reader doesn't need it for restoration
		offset++

		keyLen := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4
		valLen := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4

		if offset+int(keyLen)+int(valLen) > len(data) {
			return nil, nil, false, nil
		}

		keys[layer] = append([]byte(nil), data[offset:offset+int(keyLen)]...)
		offset += int(keyLen)
		vals[layer] = append([]byte(nil), data[offset:offset+int(valLen)]...)
		offset += int(valLen)
	}

	// Keep numKeep accessible via unused variable check
	_ = numKeep

	now := time.Now()
	_ = os.Chtimes(pc.pathForKey(key), now, now)
	return keys, vals, true, nil
}

func (pc *PrefixCache) loadV1(data []byte, key string) (keys [][]byte, vals [][]byte, hit bool, err error) {
	const headerSize = 8 + 4 + 4 + 4 + 4 + 32
	if len(data) < headerSize || string(data[:8]) != prefixCacheMagic {
		return nil, nil, false, nil
	}

	numLayers := binary.LittleEndian.Uint32(data[8:12])
	numTokens := binary.LittleEndian.Uint32(data[12:16])
	keyRowBytes := binary.LittleEndian.Uint32(data[16:20])
	valRowBytes := binary.LittleEndian.Uint32(data[20:24])

	var expectedHash [32]byte
	copy(expectedHash[:], data[24:56])
	if expectedHash != sha256.Sum256([]byte(key)) {
		return nil, nil, false, nil
	}

	offset := headerSize
	keys = make([][]byte, int(numLayers))
	vals = make([][]byte, int(numLayers))
	for layer := 0; layer < int(numLayers); layer++ {
		keySize := int(keyRowBytes) * int(numTokens)
		valSize := int(valRowBytes) * int(numTokens)
		if offset+keySize+valSize > len(data) {
			return nil, nil, false, nil
		}

		keys[layer] = append([]byte(nil), data[offset:offset+keySize]...)
		offset += keySize
		vals[layer] = append([]byte(nil), data[offset:offset+valSize]...)
		offset += valSize
	}

	now := time.Now()
	_ = os.Chtimes(pc.pathForKey(key), now, now)
	return keys, vals, true, nil
}

func (pc *PrefixCache) Evict() error {
	type fileInfo struct {
		path string
		size int64
		time int64
	}

	entries, err := os.ReadDir(pc.dir)
	if err != nil {
		return err
	}

	var totalSize int64
	var files []fileInfo
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".kvc" {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		totalSize += info.Size()
		files = append(files, fileInfo{
			path: filepath.Join(pc.dir, entry.Name()),
			size: info.Size(),
			time: info.ModTime().UnixNano(),
		})
	}

	sort.Slice(files, func(i, j int) bool { return files[i].time < files[j].time })
	for _, file := range files {
		if totalSize < pc.maxSize {
			break
		}
		if err := os.Remove(file.path); err != nil && !os.IsNotExist(err) {
			return err
		}
		totalSize -= file.size
	}

	return nil
}

func (pc *PrefixCache) pathForKey(key string) string {
	return filepath.Join(pc.dir, key+".kvc")
}