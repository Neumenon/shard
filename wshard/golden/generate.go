// generate.go — Produces golden .wshard files for cross-language testing.
//
// Usage:
//
//	go run generate.go [output_dir]
//
// Writes simple_episode.wshard, dtype_zoo.wshard, per_block_compressed.wshard,
// and golden_hashes.json into output_dir (default: current directory).
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"

	"github.com/cespare/xxhash/v2"
	"github.com/klauspost/compress/zstd"
)

// ============================================================
// Shard v2 binary constants (duplicated from shard package so
// this generator has zero intra-repo deps).
// ============================================================

var shardMagic = [4]byte{'S', 'H', 'R', 'D'}

const (
	shardVersion2       = 0x02
	shardRoleWShard     = 0x05
	headerSize          = 64
	indexEntrySize      = 48
	flagLittleEndian    = 0x0002
	flagHasChecksums    = 0x0020
	flagHasContentTypes = 0x0080
	compressNone        = 0
	compressZstd        = 1
	alignNone           = 0
	align64             = 64
	entryFlagCompressed = 0x0001
	entryFlagZstd       = 0x0002
	contentTypeJSON     = 2
)

var crc32cTable = crc32.MakeTable(crc32.Castagnoli)

func crc32c(data []byte) uint32 { return crc32.Checksum(data, crc32cTable) }

func xxh64(s string) uint64 { return xxhash.Sum64String(s) }

// ============================================================
// Low-level shard v2 writer (standalone, no package import)
// ============================================================

type entry struct {
	name        string
	data        []byte // on-disk bytes (possibly compressed)
	origSize    int
	checksum    uint32
	flags       uint16 // entry flags (compression bits)
	contentType uint16
}

type shardWriter struct {
	entries []entry
	align   uint8
	compDef uint8
}

func (w *shardWriter) addEntry(name string, data []byte, ct uint16) {
	w.entries = append(w.entries, entry{
		name:        name,
		data:        data,
		origSize:    len(data),
		checksum:    crc32c(data),
		flags:       0,
		contentType: ct,
	})
}

func (w *shardWriter) addEntryCompressed(name string, data []byte, ct uint16) {
	origSize := len(data)
	checksum := crc32c(data)

	enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	if err != nil {
		panic(err)
	}
	compressed := enc.EncodeAll(data, nil)
	enc.Close()

	// Only use compressed version if it saves space.
	if len(compressed) < origSize {
		w.entries = append(w.entries, entry{
			name:        name,
			data:        compressed,
			origSize:    origSize,
			checksum:    checksum,
			flags:       entryFlagCompressed | entryFlagZstd,
			contentType: ct,
		})
	} else {
		w.entries = append(w.entries, entry{
			name:        name,
			data:        data,
			origSize:    origSize,
			checksum:    checksum,
			flags:       0,
			contentType: ct,
		})
	}
}

func (w *shardWriter) writeTo(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	entryCount := uint32(len(w.entries))
	indexSize := int64(entryCount) * int64(indexEntrySize)

	// Build string table (concatenated WITHOUT null terminators, matching Python/TS).
	var stringTable []byte
	nameOffsets := make([]uint32, len(w.entries))
	for i, e := range w.entries {
		nameOffsets[i] = uint32(len(stringTable))
		stringTable = append(stringTable, []byte(e.name)...)
	}

	stringTableOffset := int64(headerSize) + indexSize
	dataSectionOffset := stringTableOffset + int64(len(stringTable))

	// Align data section.
	if w.align > 0 {
		a := int64(w.align)
		dataSectionOffset = (dataSectionOffset + a - 1) & ^(a - 1)
	}

	// Compute per-entry offsets in final file.
	cur := dataSectionOffset
	dataOffsets := make([]uint64, len(w.entries))
	for i, e := range w.entries {
		if w.align > 0 {
			a := int64(w.align)
			cur = (cur + a - 1) & ^(a - 1)
		}
		dataOffsets[i] = uint64(cur)
		cur += int64(len(e.data))
	}
	totalSize := uint64(cur)

	// Determine flags.
	flags := uint16(flagLittleEndian | flagHasChecksums)
	for _, e := range w.entries {
		if e.contentType != 0 {
			flags |= flagHasContentTypes
			break
		}
	}

	// --- Write header (64 bytes) ---
	hdr := make([]byte, headerSize)
	copy(hdr[0:4], shardMagic[:])
	hdr[4] = shardVersion2
	hdr[5] = shardRoleWShard
	binary.LittleEndian.PutUint16(hdr[6:8], flags)
	hdr[8] = w.align
	hdr[9] = w.compDef
	binary.LittleEndian.PutUint16(hdr[10:12], indexEntrySize)
	binary.LittleEndian.PutUint32(hdr[12:16], entryCount)
	binary.LittleEndian.PutUint64(hdr[16:24], uint64(stringTableOffset))
	binary.LittleEndian.PutUint64(hdr[24:32], uint64(dataSectionOffset))
	binary.LittleEndian.PutUint64(hdr[32:40], 0) // no schema
	binary.LittleEndian.PutUint64(hdr[40:48], totalSize)
	// bytes 48-63 reserved (zero)
	if _, err := f.Write(hdr); err != nil {
		return err
	}

	// --- Write index entries (48 bytes each) ---
	for i, e := range w.entries {
		ie := make([]byte, indexEntrySize)
		binary.LittleEndian.PutUint64(ie[0:8], xxh64(e.name))
		binary.LittleEndian.PutUint32(ie[8:12], nameOffsets[i])
		binary.LittleEndian.PutUint16(ie[12:14], uint16(len(e.name)))
		binary.LittleEndian.PutUint16(ie[14:16], e.flags)
		binary.LittleEndian.PutUint64(ie[16:24], dataOffsets[i])
		binary.LittleEndian.PutUint64(ie[24:32], uint64(len(e.data)))
		binary.LittleEndian.PutUint64(ie[32:40], uint64(e.origSize))
		binary.LittleEndian.PutUint32(ie[40:44], e.checksum)
		binary.LittleEndian.PutUint32(ie[44:48], uint32(e.contentType)) // Reserved = content type
		if _, err := f.Write(ie); err != nil {
			return err
		}
	}

	// --- Write string table ---
	if _, err := f.Write(stringTable); err != nil {
		return err
	}

	// --- Write padding to reach data section ---
	padLen := int64(dataSectionOffset) - stringTableOffset - int64(len(stringTable))
	if padLen > 0 {
		if _, err := f.Write(make([]byte, padLen)); err != nil {
			return err
		}
	}

	// --- Write data (with alignment padding between entries) ---
	written := dataSectionOffset
	for i, e := range w.entries {
		// Pad to aligned offset.
		padNeeded := int64(dataOffsets[i]) - written
		if padNeeded > 0 {
			if _, err := f.Write(make([]byte, padNeeded)); err != nil {
				return err
			}
			written += padNeeded
		}
		if _, err := f.Write(e.data); err != nil {
			return err
		}
		written += int64(len(e.data))
	}

	return nil
}

// ============================================================
// Helper: encode float32 slice to LE bytes
// ============================================================

func f32Bytes(vals []float32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func f64Bytes(vals []float64) []byte {
	buf := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return buf
}

func i32Bytes(vals []int32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	return buf
}

func u8Bytes(vals []uint8) []byte {
	return vals
}

func u16Bytes(vals []uint16) []byte {
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	return buf
}

func boolBytes(vals []bool) []byte {
	buf := make([]byte, len(vals))
	for i, v := range vals {
		if v {
			buf[i] = 1
		}
	}
	return buf
}

// ============================================================
// Golden file 1: simple_episode.wshard
// ============================================================

func genSimpleEpisode(dir string) error {
	const T = 10
	w := &shardWriter{align: align64, compDef: compressNone}

	// meta/wshard
	metaWShard, _ := json.Marshal(map[string]any{
		"version":  1,
		"timebase": map[string]any{"type": "ticks", "tick_hz": 30.0},
	})
	w.addEntry("meta/wshard", metaWShard, contentTypeJSON)

	// meta/episode — keys must match Python convention exactly
	metaEp, _ := json.Marshal(map[string]any{
		"episode_id":     "golden_simple",
		"env_id":         "TestEnv-v1",
		"length_T":       T,
		"timestep_range": [2]int{0, T},
		"timebase":       map[string]any{"type": "ticks", "dt_ns": 33333333},
	})
	w.addEntry("meta/episode", metaEp, contentTypeJSON)

	// meta/channels — must be {"channels": [...]} list format matching Python
	chanJSON, _ := json.Marshal(map[string]any{
		"channels": []map[string]any{
			{"id": "state", "dtype": "f32", "shape": []int{4}, "signal_block": "signal/state"},
			{"id": "ctrl", "dtype": "f32", "shape": []int{2}, "signal_block": "action/ctrl"},
		},
	})
	w.addEntry("meta/channels", chanJSON, contentTypeJSON)

	// signal/state — sequential 0.0..39.0
	state := make([]float32, T*4)
	for i := range state {
		state[i] = float32(i)
	}
	w.addEntry("signal/state", f32Bytes(state), 0)

	// action/ctrl — 0.1, 0.2 repeated
	ctrl := make([]float32, T*2)
	for i := range ctrl {
		if i%2 == 0 {
			ctrl[i] = 0.1
		} else {
			ctrl[i] = 0.2
		}
	}
	w.addEntry("action/ctrl", f32Bytes(ctrl), 0)

	// reward — alternating 1.0, 0.5
	rewards := make([]float32, T)
	for i := range rewards {
		if i%2 == 0 {
			rewards[i] = 1.0
		} else {
			rewards[i] = 0.5
		}
	}
	w.addEntry("reward", f32Bytes(rewards), 0)

	// done — all false except last
	done := make([]bool, T)
	done[T-1] = true
	w.addEntry("done", boolBytes(done), 0)

	return w.writeTo(filepath.Join(dir, "simple_episode.wshard"))
}

// ============================================================
// Golden file 2: dtype_zoo.wshard
// ============================================================

func genDtypeZoo(dir string) error {
	const T = 4
	w := &shardWriter{align: align64, compDef: compressNone}

	// meta/wshard
	metaWShard, _ := json.Marshal(map[string]any{
		"version":  1,
		"timebase": map[string]any{"type": "ticks", "tick_hz": 1.0},
	})
	w.addEntry("meta/wshard", metaWShard, contentTypeJSON)

	// meta/episode
	metaEp, _ := json.Marshal(map[string]any{
		"episode_id":     "golden_dtypes",
		"env_id":         "DtypeTestEnv-v0",
		"length_T":       T,
		"timestep_range": [2]int{0, T},
		"timebase":       map[string]any{"type": "ticks", "dt_ns": 1000000000},
	})
	w.addEntry("meta/episode", metaEp, contentTypeJSON)

	// meta/channels — list format matching Python
	chanJSON, _ := json.Marshal(map[string]any{
		"channels": []map[string]any{
			{"id": "f32_ch", "dtype": "f32", "shape": []int{1}, "signal_block": "signal/f32_ch"},
			{"id": "f64_ch", "dtype": "f64", "shape": []int{1}, "signal_block": "signal/f64_ch"},
			{"id": "i32_ch", "dtype": "i32", "shape": []int{1}, "signal_block": "signal/i32_ch"},
			{"id": "u8_ch", "dtype": "u8", "shape": []int{1}, "signal_block": "signal/u8_ch"},
			{"id": "bf16_ch", "dtype": "bf16", "shape": []int{1}, "signal_block": "signal/bf16_ch"},
		},
	})
	w.addEntry("meta/channels", chanJSON, contentTypeJSON)

	// signal/f32_ch — [1.0, 2.0, 3.0, 4.0]
	w.addEntry("signal/f32_ch", f32Bytes([]float32{1.0, 2.0, 3.0, 4.0}), 0)

	// signal/f64_ch — [10.0, 20.0, 30.0, 40.0]
	w.addEntry("signal/f64_ch", f64Bytes([]float64{10.0, 20.0, 30.0, 40.0}), 0)

	// signal/i32_ch — [-1, 0, 1, 2147483647]
	w.addEntry("signal/i32_ch", i32Bytes([]int32{-1, 0, 1, math.MaxInt32}), 0)

	// signal/u8_ch — [0, 127, 255, 42]
	w.addEntry("signal/u8_ch", u8Bytes([]uint8{0, 127, 255, 42}), 0)

	// signal/bf16_ch — raw uint16 LE: 0x3F80=1.0, 0x4000=2.0, 0x4040=3.0, 0x4080=4.0
	w.addEntry("signal/bf16_ch", u16Bytes([]uint16{0x3F80, 0x4000, 0x4040, 0x4080}), 0)

	// reward — all zeros (not the focus of this test)
	w.addEntry("reward", f32Bytes(make([]float32, T)), 0)

	// done — all false except last
	done := make([]bool, T)
	done[T-1] = true
	w.addEntry("done", boolBytes(done), 0)

	return w.writeTo(filepath.Join(dir, "dtype_zoo.wshard"))
}

// ============================================================
// Golden file 3: per_block_compressed.wshard
// ============================================================

func genCompressed(dir string) error {
	const T = 100
	w := &shardWriter{align: align64, compDef: compressZstd}

	// meta/wshard
	metaWShard, _ := json.Marshal(map[string]any{
		"version":  1,
		"timebase": map[string]any{"type": "ticks", "tick_hz": 60.0},
	})
	w.addEntry("meta/wshard", metaWShard, contentTypeJSON)

	// meta/episode
	metaEp, _ := json.Marshal(map[string]any{
		"episode_id":     "golden_compressed",
		"env_id":         "CompressTestEnv-v0",
		"length_T":       T,
		"timestep_range": [2]int{0, T},
		"timebase":       map[string]any{"type": "ticks", "dt_ns": 16666667},
	})
	w.addEntry("meta/episode", metaEp, contentTypeJSON)

	// meta/channels
	chanJSON, _ := json.Marshal(map[string]any{
		"channels": []map[string]any{
			{"id": "obs", "dtype": "f32", "shape": []int{8}, "signal_block": "signal/obs"},
		},
	})
	w.addEntry("meta/channels", chanJSON, contentTypeJSON)

	// signal/obs — sequential floats (very compressible)
	obs := make([]float32, T*8)
	for i := range obs {
		obs[i] = float32(i) * 0.01
	}
	obsData := f32Bytes(obs)
	w.addEntryCompressed("signal/obs", obsData, 0)

	// reward — sequential (compressible)
	rewards := make([]float32, T)
	for i := range rewards {
		rewards[i] = float32(i) * 0.1
	}
	rewardData := f32Bytes(rewards)
	w.addEntryCompressed("reward", rewardData, 0)

	// done — all false except last
	done := make([]bool, T)
	done[T-1] = true
	w.addEntry("done", boolBytes(done), 0)

	return w.writeTo(filepath.Join(dir, "per_block_compressed.wshard"))
}

// ============================================================
// Golden hashes JSON
// ============================================================

func genHashes(dir string) error {
	// CRC32C of "hello"
	crc32cHello := crc32c([]byte("hello"))

	// xxHash64 of "signal/obs" and "meta/manifest"
	xxhSignalObs := xxh64("signal/obs")
	xxhMetaManifest := xxh64("meta/manifest")

	hashes := map[string]any{
		"crc32c_hello":         fmt.Sprintf("0x%08x", crc32cHello),
		"xxhash64_signal_obs":  fmt.Sprintf("0x%016x", xxhSignalObs),
		"xxhash64_meta_manifest": fmt.Sprintf("0x%016x", xxhMetaManifest),
		"dtype_sizes": map[string]int{
			"f32":  4,
			"f64":  8,
			"bf16": 2,
			"f16":  2,
			"i32":  4,
			"i64":  8,
			"i16":  2,
			"i8":   1,
			"u8":   1,
			"u16":  2,
			"u32":  4,
			"u64":  8,
			"bool": 1,
		},
	}

	data, err := json.MarshalIndent(hashes, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(filepath.Join(dir, "golden_hashes.json"), data, 0644)
}

// ============================================================
// Main
// ============================================================

func main() {
	dir := "."
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}

	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", dir, err)
		os.Exit(1)
	}

	generators := []struct {
		name string
		fn   func(string) error
	}{
		{"simple_episode.wshard", genSimpleEpisode},
		{"dtype_zoo.wshard", genDtypeZoo},
		{"per_block_compressed.wshard", genCompressed},
		{"golden_hashes.json", genHashes},
	}

	for _, g := range generators {
		fmt.Printf("generating %s ... ", g.name)
		if err := g.fn(dir); err != nil {
			fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("ok")
	}

	fmt.Printf("\nAll golden files written to %s\n", dir)
}
