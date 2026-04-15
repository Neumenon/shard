// +build ignore

// generate_golden.go produces deterministic golden shard files for cross-language testing.
//
// Run: go run ./testdata/generate_golden.go
//
// Produces:
//   testdata/golden_basic.shard       - 3 uncompressed entries, align 64, MoSH role
//   testdata/golden_types.shard       - entries with different content types
//   testdata/golden_align16.shard     - alignment 16
//   testdata/golden_noalign.shard     - no alignment
//   testdata/golden_manifest.json     - JSON manifest of all expected values
package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/phenomenon0/Agent-GO/cowrie/ucodec"
)

// GoldenManifest describes all golden files and their expected contents.
type GoldenManifest struct {
	Files []GoldenFile `json:"files"`
}

type GoldenFile struct {
	Filename  string        `json:"filename"`
	SHA256    string        `json:"sha256"`
	Header    GoldenHeader  `json:"header"`
	Entries   []GoldenEntry `json:"entries"`
}

type GoldenHeader struct {
	Version            int    `json:"version"`
	Role               int    `json:"role"`
	Flags              int    `json:"flags"`
	Alignment          int    `json:"alignment"`
	CompressionDefault int    `json:"compression_default"`
	EntryCount         int    `json:"entry_count"`
	IndexEntrySize     int    `json:"index_entry_size"`
	StringTableOffset  uint64 `json:"string_table_offset"`
	DataSectionOffset  uint64 `json:"data_section_offset"`
	SchemaOffset       uint64 `json:"schema_offset"`
	TotalFileSize      uint64 `json:"total_file_size"`
}

type GoldenEntry struct {
	Name        string `json:"name"`
	NameHash    uint64 `json:"name_hash"`
	ContentType int    `json:"content_type"`
	OrigSize    uint64 `json:"orig_size"`
	DiskSize    uint64 `json:"disk_size"`
	Checksum    uint32 `json:"checksum"`
	Compressed  bool   `json:"compressed"`
	DataSHA256  string `json:"data_sha256"`
}

func main() {
	dir := filepath.Dir(os.Args[0])
	if dir == "" || dir == "." {
		dir = "."
	}
	// Always write to the testdata directory relative to where the source lives
	outDir := "testdata"
	if _, err := os.Stat(outDir); err != nil {
		// Try current directory
		outDir = "testdata"
		if _, err := os.Stat(outDir); err != nil {
			outDir = "."
		}
	}

	var manifest GoldenManifest

	// === golden_basic.shard ===
	{
		path := filepath.Join(outDir, "golden_basic.shard")
		w, err := ucodec.NewShardWriter(path, ucodec.ShardRoleMoSH)
		must(err)

		// Entry 1: simple text
		data1 := []byte("hello world")
		must(w.WriteEntryTyped("greeting", data1, ucodec.ContentTypeText))

		// Entry 2: binary blob (deterministic pattern)
		data2 := make([]byte, 1024)
		for i := range data2 {
			data2[i] = byte(i % 256)
		}
		must(w.WriteEntryTyped("pattern/1k", data2, ucodec.ContentTypeBlob))

		// Entry 3: JSON
		data3 := []byte(`{"model":"qwen2.5","params":7000000000,"quantization":"Q4_K_M"}`)
		must(w.WriteEntryTyped("metadata/model", data3, ucodec.ContentTypeJSON))

		must(w.Close())

		gf := readGoldenFile(path)
		manifest.Files = append(manifest.Files, gf)
	}

	// === golden_types.shard ===
	{
		path := filepath.Join(outDir, "golden_types.shard")
		w, err := ucodec.NewShardWriter(path, ucodec.ShardRoleSample)
		must(err)

		// Various content types
		must(w.WriteEntryTyped("tensor/weights", makePattern(4096, 0xAA), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("config.json", []byte(`{"layers":32,"hidden":4096}`), ucodec.ContentTypeJSON))
		must(w.WriteEntryTyped("config.glyph", []byte("layers=32\nhidden=4096\n"), ucodec.ContentTypeGLYPH))
		must(w.WriteEntryTyped("readme.txt", []byte("This is a sample shard for testing."), ucodec.ContentTypeText))
		must(w.WriteEntryTyped("image/thumbnail", makePattern(256, 0x42), ucodec.ContentTypeImage))

		must(w.Close())

		gf := readGoldenFile(path)
		manifest.Files = append(manifest.Files, gf)
	}

	// === golden_align16.shard ===
	{
		path := filepath.Join(outDir, "golden_align16.shard")
		w, err := ucodec.NewShardWriter(path, ucodec.ShardRoleMoSH)
		must(err)
		must(w.SetAlignment(ucodec.Align16))

		must(w.WriteEntryTyped("a", []byte("short"), ucodec.ContentTypeText))
		must(w.WriteEntryTyped("b", makePattern(100, 0x55), ucodec.ContentTypeBlob))

		must(w.Close())

		gf := readGoldenFile(path)
		manifest.Files = append(manifest.Files, gf)
	}

	// === golden_noalign.shard ===
	{
		path := filepath.Join(outDir, "golden_noalign.shard")
		w, err := ucodec.NewShardWriter(path, ucodec.ShardRoleMoSH)
		must(err)
		must(w.SetAlignment(ucodec.AlignNone))

		must(w.WriteEntryTyped("x", []byte{0xDE, 0xAD, 0xBE, 0xEF}, ucodec.ContentTypeBlob))
		must(w.WriteEntryTyped("y", []byte{0xCA, 0xFE, 0xBA, 0xBE, 0x00, 0x01, 0x02, 0x03}, ucodec.ContentTypeBlob))

		must(w.Close())

		gf := readGoldenFile(path)
		manifest.Files = append(manifest.Files, gf)
	}

	// === golden_wshard.shard (WShard role) ===
	{
		path := filepath.Join(outDir, "golden_wshard.shard")
		w, err := ucodec.NewShardWriter(path, ucodec.ShardRoleWShard)
		must(err)

		must(w.WriteEntryTyped("signal/imu", makePattern(600, 0x11), ucodec.ContentTypeBlob))
		must(w.WriteEntryTyped("omen/imu/mlp_v1", makePattern(600, 0x22), ucodec.ContentTypeBlob))
		must(w.WriteEntryTyped("residual/imu/sign2nddiff", makePattern(75, 0x33), ucodec.ContentTypeBlob))

		must(w.Close())

		gf := readGoldenFile(path)
		manifest.Files = append(manifest.Files, gf)
	}

	// === golden_hierarchical.shard (deep paths) ===
	{
		path := filepath.Join(outDir, "golden_hierarchical.shard")
		w, err := ucodec.NewShardWriter(path, ucodec.ShardRoleMoSH)
		must(err)

		must(w.WriteEntryTyped("layer.0/attention/q_proj/weight", makePattern(512, 0x01), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/attention/k_proj/weight", makePattern(512, 0x02), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/attention/v_proj/weight", makePattern(512, 0x03), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/attention/o_proj/weight", makePattern(512, 0x04), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/ffn/gate/weight", makePattern(2048, 0x05), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/ffn/up/weight", makePattern(2048, 0x06), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/ffn/down/weight", makePattern(2048, 0x07), ucodec.ContentTypeTensor))
		must(w.WriteEntryTyped("layer.0/norm", makePattern(128, 0x08), ucodec.ContentTypeTensor))

		must(w.Close())

		gf := readGoldenFile(path)
		manifest.Files = append(manifest.Files, gf)
	}

	// Write manifest
	manifestPath := filepath.Join(outDir, "golden_manifest.json")
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	must(err)
	must(os.WriteFile(manifestPath, manifestData, 0644))

	fmt.Printf("Generated %d golden files + manifest in %s\n", len(manifest.Files), outDir)
	for _, f := range manifest.Files {
		fmt.Printf("  %s: %d entries, %d bytes\n", f.Filename, len(f.Entries), f.Header.TotalFileSize)
	}
}

func makePattern(size int, seed byte) []byte {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte((int(seed) + i) % 256)
	}
	return data
}

func readGoldenFile(path string) GoldenFile {
	// Read the file bytes for SHA256
	fileData, err := os.ReadFile(path)
	must(err)
	fileHash := sha256.Sum256(fileData)

	// Open with reader
	r, err := ucodec.OpenShard(path)
	must(err)
	defer r.Close()

	h := r.Header()
	gf := GoldenFile{
		Filename: filepath.Base(path),
		SHA256:   fmt.Sprintf("%x", fileHash),
		Header: GoldenHeader{
			Version:            int(h.Version),
			Role:               int(h.Role),
			Flags:              int(h.Flags),
			Alignment:          int(h.Alignment),
			CompressionDefault: int(h.CompressionDefault),
			EntryCount:         int(h.EntryCount),
			IndexEntrySize:     int(h.IndexEntrySize),
			StringTableOffset:  h.StringTableOffset,
			DataSectionOffset:  h.DataSectionOffset,
			SchemaOffset:       h.SchemaOffset,
			TotalFileSize:      h.TotalFileSize,
		},
	}

	for i := 0; i < r.EntryCount(); i++ {
		info := r.GetEntryInfo(i)
		data, err := r.ReadEntry(i)
		must(err)

		dataHash := sha256.Sum256(data)
		ge := GoldenEntry{
			Name:        r.EntryName(i),
			NameHash:    info.NameHash,
			ContentType: int(info.Reserved & 0xFFFF),
			OrigSize:    info.OrigSize,
			DiskSize:    info.DiskSize,
			Checksum:    info.Checksum,
			Compressed:  info.Flags&ucodec.EntryFlagCompressed != 0,
			DataSHA256:  fmt.Sprintf("%x", dataHash),
		}
		gf.Entries = append(gf.Entries, ge)
	}

	return gf
}

func must(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: %v\n", err)
		os.Exit(1)
	}
}
