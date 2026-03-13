package shard

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestColumnShardRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.cshard")

	// Write
	w, err := NewColumnShardWriter(path, 4) // small row group for testing
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("id", ColTypeInt64, false, ColEncDelta)
	w.AddColumn("label", ColTypeBool, false, ColEncRLE)
	w.AddColumn("score", ColTypeFloat32, false, ColEncByteStreamSplit)
	w.AddColumn("name", ColTypeString, false, ColEncDictionary)

	rows := []map[string]any{
		{"id": int64(1), "label": true, "score": float32(0.95), "name": "alice"},
		{"id": int64(2), "label": false, "score": float32(0.42), "name": "bob"},
		{"id": int64(3), "label": true, "score": float32(0.88), "name": "alice"},
		{"id": int64(4), "label": false, "score": float32(0.11), "name": "charlie"},
		{"id": int64(5), "label": true, "score": float32(0.99), "name": "bob"},
		{"id": int64(6), "label": true, "score": float32(0.77), "name": "alice"},
		{"id": int64(7), "label": false, "score": float32(0.33), "name": "dave"},
		{"id": int64(8), "label": true, "score": float32(0.65), "name": "eve"},
		{"id": int64(9), "label": false, "score": float32(0.50), "name": "bob"},
		{"id": int64(10), "label": true, "score": float32(0.81), "name": "alice"},
	}

	if err := w.WriteRows(rows); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify file exists and has content
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() == 0 {
		t.Fatal("output file is empty")
	}
	t.Logf("ColumnShard file: %d bytes", info.Size())

	// Read
	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	// Check schema
	schema := r.Schema()
	if schema.RowCount != 10 {
		t.Errorf("RowCount = %d, want 10", schema.RowCount)
	}
	// 10 rows / 4 per group = 3 groups (4, 4, 2)
	if schema.RowGroupCount != 3 {
		t.Errorf("RowGroupCount = %d, want 3", schema.RowGroupCount)
	}
	if len(schema.Columns) != 4 {
		t.Errorf("Columns = %d, want 4", len(schema.Columns))
	}

	// Read int64 column
	ids, _, err := r.ReadInt64Column("id")
	if err != nil {
		t.Fatal("ReadInt64Column:", err)
	}
	if len(ids) != 10 {
		t.Errorf("id column: got %d values, want 10", len(ids))
	}
	for i, want := range []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10} {
		if ids[i] != want {
			t.Errorf("id[%d] = %d, want %d", i, ids[i], want)
		}
	}

	// Read bool column
	labels, _, err := r.ReadBoolColumn("label")
	if err != nil {
		t.Fatal("ReadBoolColumn:", err)
	}
	if len(labels) != 10 {
		t.Errorf("label column: got %d values, want 10", len(labels))
	}
	wantLabels := []bool{true, false, true, false, true, true, false, true, false, true}
	for i, want := range wantLabels {
		if labels[i] != want {
			t.Errorf("label[%d] = %v, want %v", i, labels[i], want)
		}
	}

	// Read float32 column
	scores, _, err := r.ReadFloat32Column("score")
	if err != nil {
		t.Fatal("ReadFloat32Column:", err)
	}
	if len(scores) != 10 {
		t.Errorf("score column: got %d values, want 10", len(scores))
	}
	wantScores := []float32{0.95, 0.42, 0.88, 0.11, 0.99, 0.77, 0.33, 0.65, 0.50, 0.81}
	for i, want := range wantScores {
		if scores[i] != want {
			t.Errorf("score[%d] = %f, want %f", i, scores[i], want)
		}
	}

	// Read string column
	names, _, err := r.ReadStringColumn("name")
	if err != nil {
		t.Fatal("ReadStringColumn:", err)
	}
	if len(names) != 10 {
		t.Errorf("name column: got %d values, want 10", len(names))
	}
	wantNames := []string{"alice", "bob", "alice", "charlie", "bob", "alice", "dave", "eve", "bob", "alice"}
	for i, want := range wantNames {
		if names[i] != want {
			t.Errorf("name[%d] = %q, want %q", i, names[i], want)
		}
	}

	// Verify stats exist
	idStats := r.ColumnStats("id", 0)
	if idStats == nil {
		t.Error("expected stats for id column, row group 0")
	} else {
		t.Logf("id rg0 stats: count=%d, min=%v, max=%v", idStats.Count, idStats.Min, idStats.Max)
	}
}

func TestColumnShardNullable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nullable.cshard")

	w, err := NewColumnShardWriter(path, 100)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("id", ColTypeInt64, false, ColEncPlain)
	w.AddColumn("value", ColTypeInt64, true, ColEncPlain)

	rows := []map[string]any{
		{"id": int64(1), "value": int64(100)},
		{"id": int64(2)}, // value is null
		{"id": int64(3), "value": int64(300)},
	}

	if err := w.WriteRows(rows); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	values, valid, err := r.ReadInt64Column("value")
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 3 {
		t.Fatalf("got %d values, want 3", len(values))
	}
	if valid == nil {
		t.Fatal("expected validity bitmap for nullable column")
	}
	if !valid[0] || valid[1] || !valid[2] {
		t.Errorf("valid = %v, want [true, false, true]", valid)
	}
	if values[0] != 100 || values[2] != 300 {
		t.Errorf("values = %v, want [100, _, 300]", values)
	}
}

func TestColumnEncodings(t *testing.T) {
	t.Run("PlainInt64", func(t *testing.T) {
		input := []int64{10, -20, 30, 0, math.MaxInt64}
		encoded := EncodePlainInt64(input)
		decoded := DecodePlainInt64(encoded, len(input))
		for i := range input {
			if decoded[i] != input[i] {
				t.Errorf("[%d] = %d, want %d", i, decoded[i], input[i])
			}
		}
	})

	t.Run("DeltaInt64", func(t *testing.T) {
		input := []int64{100, 103, 107, 108, 115}
		encoded := EncodeDeltaInt64(input)
		decoded := DecodeDeltaInt64(encoded, len(input))
		for i := range input {
			if decoded[i] != input[i] {
				t.Errorf("[%d] = %d, want %d", i, decoded[i], input[i])
			}
		}
		t.Logf("Delta: %d values -> %d bytes (plain would be %d)", len(input), len(encoded), len(input)*8)
	})

	t.Run("RLEBool", func(t *testing.T) {
		input := []bool{true, true, true, false, false, true}
		encoded := EncodeRLEBool(input)
		decoded := DecodeRLEBool(encoded, len(input))
		for i := range input {
			if decoded[i] != input[i] {
				t.Errorf("[%d] = %v, want %v", i, decoded[i], input[i])
			}
		}
	})

	t.Run("ByteStreamSplit32", func(t *testing.T) {
		input := []float32{1.5, 2.5, 3.5, 4.5}
		encoded := EncodeByteStreamSplit32(input)
		decoded := DecodeByteStreamSplit32(encoded, len(input))
		for i := range input {
			if decoded[i] != input[i] {
				t.Errorf("[%d] = %f, want %f", i, decoded[i], input[i])
			}
		}
	})

	t.Run("ByteStreamSplit64", func(t *testing.T) {
		input := []float64{1.5, 2.5, 3.5, 4.5}
		encoded := EncodeByteStreamSplit64(input)
		decoded := DecodeByteStreamSplit64(encoded, len(input))
		for i := range input {
			if decoded[i] != input[i] {
				t.Errorf("[%d] = %f, want %f", i, decoded[i], input[i])
			}
		}
	})

	t.Run("Dictionary", func(t *testing.T) {
		input := []string{"cat", "dog", "cat", "cat", "bird", "dog"}
		encoded := EncodeDictionary(input)
		decoded := DecodeDictionary(encoded, len(input))
		for i := range input {
			if decoded[i] != input[i] {
				t.Errorf("[%d] = %q, want %q", i, decoded[i], input[i])
			}
		}
	})

	t.Run("NullBitmap", func(t *testing.T) {
		valid := []bool{true, false, true, true, false, true, false, false, true}
		packed := PackNullBitmap(valid)
		unpacked := UnpackNullBitmap(packed, len(valid))
		for i := range valid {
			if unpacked[i] != valid[i] {
				t.Errorf("[%d] = %v, want %v", i, unpacked[i], valid[i])
			}
		}
	})
}

