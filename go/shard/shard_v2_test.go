package shard

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestShardV2HeaderRoundtrip(t *testing.T) {
	h := NewShardV2Header(ShardRoleMoSH)
	h.EntryCount = 42
	h.Alignment = Align64

	var buf bytes.Buffer
	if err := WriteShardV2Header(&buf, h); err != nil {
		t.Fatalf("write header: %v", err)
	}
	if buf.Len() != 64 {
		t.Fatalf("expected 64 bytes, got %d", buf.Len())
	}

	h2, err := ReadShardV2Header(&buf)
	if err != nil {
		t.Fatalf("read header: %v", err)
	}

	if h2.Version != 0x02 {
		t.Errorf("version: %d", h2.Version)
	}
	if h2.Role != ShardRoleMoSH {
		t.Errorf("role: %d", h2.Role)
	}
	if h2.EntryCount != 42 {
		t.Errorf("entry count: %d", h2.EntryCount)
	}
	if h2.Alignment != Align64 {
		t.Errorf("alignment: %d", h2.Alignment)
	}
}

func TestShardV2WriteRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.shard")

	w, err := NewShardV2Writer(path, ShardRoleMoSH)
	if err != nil {
		t.Fatalf("create writer: %v", err)
	}

	w.SetAlignment(Align64)
	w.SetCompression(CompressZstd)

	if err := w.WriteEntry("config", []byte(`{"model":"test"}`)); err != nil {
		t.Fatalf("write config: %v", err)
	}

	bigData := make([]byte, 4096)
	for i := range bigData {
		bigData[i] = byte(i % 256)
	}
	if err := w.WriteEntryCompressed("weights", bigData); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	if err := w.WriteEntryTyped("metadata", []byte(`{"version":1}`), ContentTypeJSON); err != nil {
		t.Fatalf("write metadata: %v", err)
	}

	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Read back
	r, err := OpenShardV2(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	if r.EntryCount() != 3 {
		t.Fatalf("expected 3 entries, got %d", r.EntryCount())
	}

	// O(1) lookup
	idx := r.Lookup("config")
	if idx < 0 {
		t.Fatal("config not found")
	}
	data, err := r.ReadEntry(idx)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if string(data) != `{"model":"test"}` {
		t.Errorf("config data: %q", data)
	}

	// Compressed entry
	idx = r.Lookup("weights")
	if idx < 0 {
		t.Fatal("weights not found")
	}
	info := r.GetEntryInfo(idx)
	if !info.IsCompressed() {
		t.Error("expected weights to be compressed")
	}
	data, err = r.ReadEntry(idx)
	if err != nil {
		t.Fatalf("read weights: %v", err)
	}
	if len(data) != 4096 {
		t.Errorf("expected 4096 bytes, got %d", len(data))
	}
	for i, b := range data {
		if b != byte(i%256) {
			t.Fatalf("data mismatch at %d: got %d, want %d", i, b, byte(i%256))
		}
	}

	// Content type
	idx = r.Lookup("metadata")
	if idx < 0 {
		t.Fatal("metadata not found")
	}
	info = r.GetEntryInfo(idx)
	if info.ContentType() != ContentTypeJSON {
		t.Errorf("expected JSON content type, got %d", info.ContentType())
	}
}

func TestShardV2ChecksumMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.shard")

	w, _ := NewShardV2Writer(path, ShardRoleMoSH)
	w.WriteEntry("data", []byte("hello world"))
	w.Close()

	// Corrupt the data section
	raw, _ := os.ReadFile(path)
	if len(raw) > 100 {
		raw[len(raw)-5] ^= 0xFF
		os.WriteFile(path, raw, 0o644)
	}

	r, err := OpenShardV2(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	_, err = r.ReadEntry(0)
	if err == nil {
		t.Error("expected checksum error")
	}
}

func TestShardV2StreamWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream.shard")

	sw, err := NewShardV2StreamWriter(path, ShardRoleMoSH, 100)
	if err != nil {
		t.Fatalf("create stream writer: %v", err)
	}

	sw.SetAlignment(Align64)
	if err := sw.BeginData(); err != nil {
		t.Fatalf("begin data: %v", err)
	}

	for i := 0; i < 50; i++ {
		data := make([]byte, 128)
		for j := range data {
			data[j] = byte(i)
		}
		if err := sw.WriteEntryCompressed("entry_"+string(rune('A'+i%26)), data); err != nil {
			t.Fatalf("write entry %d: %v", i, err)
		}
	}

	if err := sw.Finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	// Read back
	r, err := OpenShardV2(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	if r.EntryCount() != 50 {
		t.Errorf("expected 50 entries, got %d", r.EntryCount())
	}
}

func TestShardV2MetadataRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meta.shard")

	w, _ := NewShardV2Writer(path, ShardRoleMoSH)
	meta := &ShardMetadata{
		SchemaVersion: "shard-v2.1",
		Producer:      "test",
		Description:   "unit test shard",
		Tags:          []string{"test", "v2"},
	}
	w.SetMetadata(meta)
	w.WriteEntry("data", []byte("test"))
	w.Close()

	r, _ := OpenShardV2(path)
	defer r.Close()

	restored, err := r.ReadMetadata()
	if err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	if restored == nil {
		t.Fatal("metadata is nil")
	}
	if restored.Producer != "test" {
		t.Errorf("producer: %q", restored.Producer)
	}
	if restored.Description != "unit test shard" {
		t.Errorf("description: %q", restored.Description)
	}
	if len(restored.Tags) != 2 {
		t.Errorf("tags: %v", restored.Tags)
	}
}

func TestShardV2LookupMissing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lookup.shard")

	w, _ := NewShardV2Writer(path, ShardRoleMoSH)
	w.WriteEntry("exists", []byte("yes"))
	w.Close()

	r, _ := OpenShardV2(path)
	defer r.Close()

	if r.Lookup("exists") < 0 {
		t.Error("expected to find 'exists'")
	}
	if r.Lookup("missing") >= 0 {
		t.Error("expected -1 for missing entry")
	}
}

func TestShardV2ListPrefix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prefix.shard")

	w, _ := NewShardV2Writer(path, ShardRoleMoSH)
	w.WriteEntry("layer0/weight", []byte("w0"))
	w.WriteEntry("layer0/bias", []byte("b0"))
	w.WriteEntry("layer1/weight", []byte("w1"))
	w.WriteEntry("config", []byte("cfg"))
	w.Close()

	r, _ := OpenShardV2(path)
	defer r.Close()

	matches := r.ListPrefix("layer0/")
	if len(matches) != 2 {
		t.Errorf("expected 2 matches for layer0/, got %d: %v", len(matches), matches)
	}

	all := r.EntryNames()
	if len(all) != 4 {
		t.Errorf("expected 4 names, got %d", len(all))
	}
}
