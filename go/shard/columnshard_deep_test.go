package shard

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// =============================================================================
// ENCODING EDGE CASES
// =============================================================================

func TestDeltaInt64_EdgeCases(t *testing.T) {
	cases := []struct {
		name   string
		values []int64
	}{
		{"single value", []int64{42}},
		{"two values", []int64{10, 20}},
		{"all same", []int64{7, 7, 7, 7, 7}},
		{"ascending", []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
		{"descending", []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}},
		{"negative deltas", []int64{100, 50, 25, 12, 6, 3}},
		{"alternating sign", []int64{-1, 1, -1, 1, -1, 1}},
		{"large values", []int64{math.MaxInt64 - 10, math.MaxInt64 - 5, math.MaxInt64}},
		{"large negative", []int64{math.MinInt64, math.MinInt64 + 5, math.MinInt64 + 15}},
		{"zeros", []int64{0, 0, 0, 0}},
		{"mixed magnitude", []int64{0, 1000000, -1000000, 1, -1}},
		{"timestamps monotonic", generateTimestamps(100)},
		{"large gap then small", []int64{0, 1000000000, 1000000001, 1000000002}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := EncodeDeltaInt64(tc.values)
			decoded := DecodeDeltaInt64(encoded, len(tc.values))

			if len(decoded) != len(tc.values) {
				t.Fatalf("length mismatch: got %d, want %d", len(decoded), len(tc.values))
			}
			for i := range tc.values {
				if decoded[i] != tc.values[i] {
					t.Errorf("[%d] = %d, want %d", i, decoded[i], tc.values[i])
				}
			}

			// Compression ratio for monotonic data
			plainSize := len(tc.values) * 8
			ratio := float64(len(encoded)) / float64(plainSize)
			t.Logf("%d values: %d bytes delta vs %d bytes plain (%.1f%%)", len(tc.values), len(encoded), plainSize, ratio*100)
		})
	}
}

func TestRLEBool_EdgeCases(t *testing.T) {
	cases := []struct {
		name   string
		values []bool
	}{
		{"empty", []bool{}},
		{"single true", []bool{true}},
		{"single false", []bool{false}},
		{"all true", repeat(true, 1000)},
		{"all false", repeat(false, 1000)},
		{"alternating", alternating(100)},
		{"long run then short", append(repeat(true, 500), repeat(false, 3)...)},
		{"one hot", append(append(repeat(false, 499), true), repeat(false, 500)...)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := EncodeRLEBool(tc.values)
			decoded := DecodeRLEBool(encoded, len(tc.values))

			if len(decoded) != len(tc.values) {
				t.Fatalf("length mismatch: got %d, want %d", len(decoded), len(tc.values))
			}
			for i := range tc.values {
				if decoded[i] != tc.values[i] {
					t.Errorf("[%d] = %v, want %v", i, decoded[i], tc.values[i])
				}
			}

			plainSize := (len(tc.values) + 7) / 8 // bit-packed
			t.Logf("%d values: %d bytes RLE vs %d bytes packed (%.1f%%)", len(tc.values), len(encoded), plainSize, safeDivPercent(len(encoded), plainSize))
		})
	}
}

func TestDictionary_EdgeCases(t *testing.T) {
	cases := []struct {
		name   string
		values []string
	}{
		{"single value repeated", repeatStr("hello", 100)},
		{"all unique", uniqueStrings(100)},
		{"empty strings", repeatStr("", 50)},
		{"256 unique (u8 boundary)", uniqueStrings(256)},
		{"257 unique (u16 boundary)", uniqueStrings(257)},
		{"mixed empty and non-empty", []string{"", "a", "", "b", "", "a", "b", ""}},
		{"long strings", repeatStr("this_is_a_reasonably_long_string_for_dictionary_testing_purposes", 20)},
		{"unicode", []string{"日本語", "中文", "한국어", "日本語", "中文"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := EncodeDictionary(tc.values)
			decoded := DecodeDictionary(encoded, len(tc.values))

			if len(decoded) != len(tc.values) {
				t.Fatalf("length mismatch: got %d, want %d", len(decoded), len(tc.values))
			}
			for i := range tc.values {
				if decoded[i] != tc.values[i] {
					t.Errorf("[%d] = %q, want %q", i, decoded[i], tc.values[i])
				}
			}

			plainSize := 0
			for _, s := range tc.values {
				plainSize += 4 + len(s)
			}
			t.Logf("%d values: %d bytes dict vs %d bytes plain (%.1f%%)", len(tc.values), len(encoded), plainSize, safeDivPercent(len(encoded), plainSize))
		})
	}
}

func TestByteStreamSplit_EdgeCases(t *testing.T) {
	t.Run("float32 special values", func(t *testing.T) {
		values := []float32{
			0, -0,
			float32(math.Inf(1)), float32(math.Inf(-1)),
			float32(math.NaN()),
			math.MaxFloat32, math.SmallestNonzeroFloat32,
			1.0, -1.0,
		}
		encoded := EncodeByteStreamSplit32(values)
		decoded := DecodeByteStreamSplit32(encoded, len(values))

		for i := range values {
			if math.IsNaN(float64(values[i])) {
				if !math.IsNaN(float64(decoded[i])) {
					t.Errorf("[%d] expected NaN", i)
				}
			} else if decoded[i] != values[i] {
				t.Errorf("[%d] = %v, want %v", i, decoded[i], values[i])
			}
		}
	})

	t.Run("float64 special values", func(t *testing.T) {
		values := []float64{
			0, -0,
			math.Inf(1), math.Inf(-1),
			math.NaN(),
			math.MaxFloat64, math.SmallestNonzeroFloat64,
		}
		encoded := EncodeByteStreamSplit64(values)
		decoded := DecodeByteStreamSplit64(encoded, len(values))

		for i := range values {
			if math.IsNaN(values[i]) {
				if !math.IsNaN(decoded[i]) {
					t.Errorf("[%d] expected NaN", i)
				}
			} else if decoded[i] != values[i] {
				t.Errorf("[%d] = %v, want %v", i, decoded[i], values[i])
			}
		}
	})

	t.Run("float32 large array", func(t *testing.T) {
		n := 65536
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(i) * 0.001
		}
		encoded := EncodeByteStreamSplit32(values)
		decoded := DecodeByteStreamSplit32(encoded, n)

		for i := range values {
			if decoded[i] != values[i] {
				t.Fatalf("[%d] = %v, want %v", i, decoded[i], values[i])
			}
		}
		t.Logf("%d float32s: %d bytes split vs %d bytes plain", n, len(encoded), n*4)
	})
}

func TestPlainStrings_EdgeCases(t *testing.T) {
	cases := []struct {
		name   string
		values []string
	}{
		{"empty slice", []string{}},
		{"single empty string", []string{""}},
		{"nulls inside", []string{"a\x00b", "c\x00d"}},
		{"very long", []string{string(make([]byte, 100000))}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := EncodePlainStrings(tc.values)
			decoded := DecodePlainStrings(encoded, len(tc.values))

			if len(decoded) != len(tc.values) {
				t.Fatalf("length mismatch: got %d, want %d", len(decoded), len(tc.values))
			}
			for i := range tc.values {
				if decoded[i] != tc.values[i] {
					t.Errorf("[%d] length %d != %d", i, len(decoded[i]), len(tc.values[i]))
				}
			}
		})
	}
}

func TestNullBitmap_EdgeCases(t *testing.T) {
	cases := []struct {
		name  string
		valid []bool
	}{
		{"empty", []bool{}},
		{"single true", []bool{true}},
		{"single false", []bool{false}},
		{"exactly 8", []bool{true, false, true, false, true, false, true, false}},
		{"exactly 16", repeat(true, 16)},
		{"9 values (crosses byte boundary)", []bool{true, true, true, true, true, true, true, true, false}},
		{"all null 100", repeat(false, 100)},
		{"all valid 100", repeat(true, 100)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			packed := PackNullBitmap(tc.valid)
			unpacked := UnpackNullBitmap(packed, len(tc.valid))

			if len(unpacked) != len(tc.valid) {
				t.Fatalf("length mismatch: got %d, want %d", len(unpacked), len(tc.valid))
			}
			for i := range tc.valid {
				if unpacked[i] != tc.valid[i] {
					t.Errorf("[%d] = %v, want %v", i, unpacked[i], tc.valid[i])
				}
			}

			expectedBytes := (len(tc.valid) + 7) / 8
			if len(packed) != expectedBytes {
				t.Errorf("packed size = %d, want %d", len(packed), expectedBytes)
			}
		})
	}
}

func TestColumnChunkMarshalUnmarshal(t *testing.T) {
	t.Run("non-nullable", func(t *testing.T) {
		data := EncodePlainInt64([]int64{10, 20, 30})
		blob := MarshalColumnChunk(ColEncPlain, 3, false, nil, data)

		chunk, err := UnmarshalColumnChunk(blob, false)
		if err != nil {
			t.Fatal(err)
		}
		if chunk.Encoding != ColEncPlain {
			t.Errorf("encoding = %d, want %d", chunk.Encoding, ColEncPlain)
		}
		if chunk.Count != 3 {
			t.Errorf("count = %d, want 3", chunk.Count)
		}
		if chunk.Nullable {
			t.Error("expected non-nullable")
		}

		values := DecodePlainInt64(chunk.Data, 3)
		for i, want := range []int64{10, 20, 30} {
			if values[i] != want {
				t.Errorf("[%d] = %d, want %d", i, values[i], want)
			}
		}
	})

	t.Run("nullable with nulls", func(t *testing.T) {
		valid := []bool{true, false, true}
		data := EncodePlainInt64([]int64{10, 0, 30})
		blob := MarshalColumnChunk(ColEncPlain, 3, true, valid, data)

		chunk, err := UnmarshalColumnChunk(blob, true)
		if err != nil {
			t.Fatal(err)
		}
		if !chunk.Nullable {
			t.Error("expected nullable")
		}
		if len(chunk.Valid) != 3 {
			t.Fatalf("valid len = %d, want 3", len(chunk.Valid))
		}
		if !chunk.Valid[0] || chunk.Valid[1] || !chunk.Valid[2] {
			t.Errorf("valid = %v, want [true, false, true]", chunk.Valid)
		}
	})
}

// =============================================================================
// COLUMNSHARD INTEGRATION TESTS
// =============================================================================

func TestColumnShard_LargeDataset(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "large.cshard")

	const numRows = 100_000
	const rowGroupSize = 16384

	w, err := NewColumnShardWriter(path, rowGroupSize)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("id", ColTypeInt64, false, ColEncDelta)
	w.AddColumn("value", ColTypeFloat32, false, ColEncByteStreamSplit)
	w.AddColumn("category", ColTypeString, false, ColEncDictionary)
	w.AddColumn("flag", ColTypeBool, false, ColEncRLE)

	categories := []string{"cat_a", "cat_b", "cat_c", "cat_d", "cat_e"}
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < numRows; i++ {
		row := map[string]any{
			"id":       int64(i),
			"value":    float32(rng.Float64() * 100),
			"category": categories[rng.Intn(len(categories))],
			"flag":     rng.Intn(2) == 0,
		}
		if err := w.WriteRow(row); err != nil {
			t.Fatal(err)
		}
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("100K rows, 4 cols: %d bytes (%.1f bytes/row)", info.Size(), float64(info.Size())/numRows)

	// Read back
	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	// Verify schema
	schema := r.Schema()
	if schema.RowCount != numRows {
		t.Errorf("RowCount = %d, want %d", schema.RowCount, numRows)
	}
	expectedGroups := (numRows + rowGroupSize - 1) / rowGroupSize
	if int(schema.RowGroupCount) != expectedGroups {
		t.Errorf("RowGroupCount = %d, want %d", schema.RowGroupCount, expectedGroups)
	}

	// Read all columns
	ids, _, err := r.ReadInt64Column("id")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != numRows {
		t.Fatalf("id column: got %d, want %d", len(ids), numRows)
	}
	// Verify monotonic IDs
	for i := 0; i < numRows; i++ {
		if ids[i] != int64(i) {
			t.Fatalf("id[%d] = %d, want %d", i, ids[i], i)
			break
		}
	}

	values, _, err := r.ReadFloat32Column("value")
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != numRows {
		t.Fatalf("value column: got %d, want %d", len(values), numRows)
	}

	cats, _, err := r.ReadStringColumn("category")
	if err != nil {
		t.Fatal(err)
	}
	if len(cats) != numRows {
		t.Fatalf("category column: got %d, want %d", len(cats), numRows)
	}
	// Verify all categories are valid
	validCats := map[string]bool{"cat_a": true, "cat_b": true, "cat_c": true, "cat_d": true, "cat_e": true}
	for i, c := range cats {
		if !validCats[c] {
			t.Fatalf("category[%d] = %q, not a valid category", i, c)
		}
	}

	flags, _, err := r.ReadBoolColumn("flag")
	if err != nil {
		t.Fatal(err)
	}
	if len(flags) != numRows {
		t.Fatalf("flag column: got %d, want %d", len(flags), numRows)
	}
}

func TestColumnShard_PredicatePushdown(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pushdown.cshard")

	const rowGroupSize = 100

	w, err := NewColumnShardWriter(path, rowGroupSize)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("id", ColTypeInt64, false, ColEncDelta)
	w.AddColumn("value", ColTypeInt64, false, ColEncPlain)

	// Write 500 rows: IDs 0-499, values = id * 10
	// Row groups: [0-99], [100-199], [200-299], [300-399], [400-499]
	for i := 0; i < 500; i++ {
		if err := w.WriteRow(map[string]any{"id": int64(i), "value": int64(i * 10)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if r.RowGroupCount() != 5 {
		t.Fatalf("expected 5 row groups, got %d", r.RowGroupCount())
	}

	// Predicate: value > 3500 (should skip row groups 0-2, partially scan 3, fully scan 4)
	threshold := int64(3500)
	skipped := 0
	scanned := 0
	matchCount := 0

	for rg := 0; rg < r.RowGroupCount(); rg++ {
		stats := r.ColumnStats("value", rg)
		if stats == nil {
			t.Fatalf("no stats for value rg %d", rg)
		}

		// Check if max value in this row group is <= threshold
		maxVal, ok := stats.Max.(float64) // JSON numbers unmarshal as float64
		if !ok {
			t.Fatalf("stats.Max type = %T, want float64 (json number)", stats.Max)
		}

		if int64(maxVal) <= threshold {
			skipped++
			continue
		}
		scanned++

		// Read the actual values for this row group
		chunk, err := r.ReadColumnChunk("value", rg)
		if err != nil {
			t.Fatal(err)
		}
		values := DecodePlainInt64(chunk.Data, int(chunk.Count))
		for _, v := range values {
			if v > threshold {
				matchCount++
			}
		}
	}

	t.Logf("Predicate value > %d: skipped %d row groups, scanned %d, found %d matches", threshold, skipped, scanned, matchCount)

	// Row groups: [0-990], [1000-1990], [2000-2990], [3000-3990], [4000-4990]
	// value > 3500 means id > 350, so RGs 0-2 should be skipped (max values 990, 1990, 2990)
	// RG 3 max = 3990 > 3500: scan it
	// RG 4 max = 4990 > 3500: scan it
	if skipped != 3 {
		t.Errorf("expected 3 skipped row groups, got %d", skipped)
	}
	if scanned != 2 {
		t.Errorf("expected 2 scanned row groups, got %d", scanned)
	}
	// Matches: values 3510, 3520, ..., 4990 => IDs 351-499 => 149 matches
	if matchCount != 149 {
		t.Errorf("expected 149 matches, got %d", matchCount)
	}
}

func TestColumnShard_ProjectionPushdown(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "projection.cshard")

	w, err := NewColumnShardWriter(path, 100)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("col_a", ColTypeInt64, false, ColEncPlain)
	w.AddColumn("col_b", ColTypeFloat32, false, ColEncPlain)
	w.AddColumn("col_c", ColTypeString, false, ColEncPlain)
	w.AddColumn("col_d", ColTypeBool, false, ColEncRLE)

	for i := 0; i < 50; i++ {
		if err := w.WriteRow(map[string]any{
			"col_a": int64(i),
			"col_b": float32(i) * 0.5,
			"col_c": fmt.Sprintf("row_%d", i),
			"col_d": i%2 == 0,
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	// Only read col_a — col_b, col_c, col_d should never be touched
	values, _, err := r.ReadInt64Column("col_a")
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 50 {
		t.Fatalf("got %d values, want 50", len(values))
	}
	for i, v := range values {
		if v != int64(i) {
			t.Errorf("[%d] = %d, want %d", i, v, i)
		}
	}

	// Verify non-existent column returns error
	_, _, err = r.ReadInt64Column("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent column")
	}
}

func TestColumnShard_AllNullable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "allnull.cshard")

	w, err := NewColumnShardWriter(path, 10)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("id", ColTypeInt64, false, ColEncPlain)
	w.AddColumn("opt_int", ColTypeInt64, true, ColEncPlain)
	w.AddColumn("opt_str", ColTypeString, true, ColEncDictionary)
	w.AddColumn("opt_float", ColTypeFloat32, true, ColEncByteStreamSplit)

	// Mix of present and null values
	rows := []map[string]any{
		{"id": int64(1), "opt_int": int64(100), "opt_str": "hello", "opt_float": float32(1.5)},
		{"id": int64(2)}, // all optional cols null
		{"id": int64(3), "opt_int": int64(300)},                       // str and float null
		{"id": int64(4), "opt_str": "world"},                          // int and float null
		{"id": int64(5), "opt_float": float32(5.5)},                   // int and str null
		{"id": int64(6), "opt_int": int64(600), "opt_str": "!", "opt_float": float32(6.5)},
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

	// Check int nullable
	intVals, intValid, err := r.ReadInt64Column("opt_int")
	if err != nil {
		t.Fatal(err)
	}
	wantIntValid := []bool{true, false, true, false, false, true}
	for i, want := range wantIntValid {
		if intValid[i] != want {
			t.Errorf("opt_int valid[%d] = %v, want %v", i, intValid[i], want)
		}
	}
	// Check non-null values
	if intVals[0] != 100 {
		t.Errorf("opt_int[0] = %d, want 100", intVals[0])
	}
	if intVals[2] != 300 {
		t.Errorf("opt_int[2] = %d, want 300", intVals[2])
	}
	if intVals[5] != 600 {
		t.Errorf("opt_int[5] = %d, want 600", intVals[5])
	}

	// Check string nullable
	strVals, strValid, err := r.ReadStringColumn("opt_str")
	if err != nil {
		t.Fatal(err)
	}
	wantStrValid := []bool{true, false, false, true, false, true}
	for i, want := range wantStrValid {
		if strValid[i] != want {
			t.Errorf("opt_str valid[%d] = %v, want %v", i, strValid[i], want)
		}
	}
	if strVals[0] != "hello" {
		t.Errorf("opt_str[0] = %q, want %q", strVals[0], "hello")
	}
	if strVals[3] != "world" {
		t.Errorf("opt_str[3] = %q, want %q", strVals[3], "world")
	}

	// Check float nullable
	floatVals, floatValid, err := r.ReadFloat32Column("opt_float")
	if err != nil {
		t.Fatal(err)
	}
	wantFloatValid := []bool{true, false, false, false, true, true}
	for i, want := range wantFloatValid {
		if floatValid[i] != want {
			t.Errorf("opt_float valid[%d] = %v, want %v", i, floatValid[i], want)
		}
	}
	if floatVals[0] != 1.5 {
		t.Errorf("opt_float[0] = %f, want 1.5", floatVals[0])
	}
	if floatVals[4] != 5.5 {
		t.Errorf("opt_float[4] = %f, want 5.5", floatVals[4])
	}
}

func TestColumnShard_StatsAccuracy(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "stats.cshard")

	w, err := NewColumnShardWriter(path, 5)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("val", ColTypeInt64, false, ColEncPlain)

	// Row group 0: [10, 20, 30, 40, 50] → min=10, max=50
	// Row group 1: [-5, -3, 0, 7, 100]  → min=-5, max=100
	// Row group 2: [42, 42]              → min=42, max=42
	allValues := []int64{10, 20, 30, 40, 50, -5, -3, 0, 7, 100, 42, 42}
	for _, v := range allValues {
		if err := w.WriteRow(map[string]any{"val": v}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	expectations := []struct {
		rg       int
		count    uint32
		minVal   int64
		maxVal   int64
	}{
		{0, 5, 10, 50},
		{1, 5, -5, 100},
		{2, 2, 42, 42},
	}

	for _, exp := range expectations {
		stats := r.ColumnStats("val", exp.rg)
		if stats == nil {
			t.Fatalf("no stats for rg %d", exp.rg)
		}
		if stats.Count != exp.count {
			t.Errorf("rg %d: count = %d, want %d", exp.rg, stats.Count, exp.count)
		}
		// JSON unmarshals numbers as float64
		if gotMin := int64(stats.Min.(float64)); gotMin != exp.minVal {
			t.Errorf("rg %d: min = %d, want %d", exp.rg, gotMin, exp.minVal)
		}
		if gotMax := int64(stats.Max.(float64)); gotMax != exp.maxVal {
			t.Errorf("rg %d: max = %d, want %d", exp.rg, gotMax, exp.maxVal)
		}
	}
}

func TestColumnShard_EmptyRowGroups(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.cshard")

	w, err := NewColumnShardWriter(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	w.AddColumn("id", ColTypeInt64, false, ColEncPlain)

	// Close without writing any rows
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if r.RowCount() != 0 {
		t.Errorf("RowCount = %d, want 0", r.RowCount())
	}
	if r.RowGroupCount() != 0 {
		t.Errorf("RowGroupCount = %d, want 0", r.RowGroupCount())
	}
}

func TestColumnShard_SingleRowPerGroup(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "single.cshard")

	w, err := NewColumnShardWriter(path, 1)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("x", ColTypeInt64, false, ColEncDelta)

	for i := 0; i < 5; i++ {
		if err := w.WriteRow(map[string]any{"x": int64(i * 100)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if r.RowGroupCount() != 5 {
		t.Errorf("RowGroupCount = %d, want 5", r.RowGroupCount())
	}

	vals, _, err := r.ReadInt64Column("x")
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int64{0, 100, 200, 300, 400} {
		if vals[i] != want {
			t.Errorf("x[%d] = %d, want %d", i, vals[i], want)
		}
	}
}

func TestColumnShard_MixedEncodings(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mixed.cshard")

	w, err := NewColumnShardWriter(path, 50)
	if err != nil {
		t.Fatal(err)
	}

	// Each column uses a different encoding
	w.AddColumn("delta_col", ColTypeInt64, false, ColEncDelta)
	w.AddColumn("plain_col", ColTypeInt64, false, ColEncPlain)
	w.AddColumn("rle_col", ColTypeBool, false, ColEncRLE)
	w.AddColumn("bss_col", ColTypeFloat32, false, ColEncByteStreamSplit)
	w.AddColumn("dict_col", ColTypeString, false, ColEncDictionary)

	for i := 0; i < 200; i++ {
		if err := w.WriteRow(map[string]any{
			"delta_col": int64(i),
			"plain_col": int64(i * i),
			"rle_col":   i%10 < 5,
			"bss_col":   float32(i) * 0.1,
			"dict_col":  fmt.Sprintf("group_%d", i%7),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	// Verify all columns roundtrip
	deltaVals, _, _ := r.ReadInt64Column("delta_col")
	plainVals, _, _ := r.ReadInt64Column("plain_col")
	rleVals, _, _ := r.ReadBoolColumn("rle_col")
	bssVals, _, _ := r.ReadFloat32Column("bss_col")
	dictVals, _, _ := r.ReadStringColumn("dict_col")

	for i := 0; i < 200; i++ {
		if deltaVals[i] != int64(i) {
			t.Errorf("delta[%d] = %d", i, deltaVals[i])
		}
		if plainVals[i] != int64(i*i) {
			t.Errorf("plain[%d] = %d", i, plainVals[i])
		}
		if rleVals[i] != (i%10 < 5) {
			t.Errorf("rle[%d] = %v", i, rleVals[i])
		}
		if bssVals[i] != float32(i)*0.1 {
			t.Errorf("bss[%d] = %f", i, bssVals[i])
		}
		want := fmt.Sprintf("group_%d", i%7)
		if dictVals[i] != want {
			t.Errorf("dict[%d] = %q, want %q", i, dictVals[i], want)
		}
	}
}

func TestColumnShard_Int32Column(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "int32.cshard")

	w, err := NewColumnShardWriter(path, 50)
	if err != nil {
		t.Fatal(err)
	}

	w.AddColumn("x", ColTypeInt32, false, ColEncPlain)
	w.AddColumn("y", ColTypeInt32, false, ColEncDelta)

	for i := 0; i < 100; i++ {
		if err := w.WriteRow(map[string]any{
			"x": int32(i * 3),
			"y": int32(1000 - i),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenColumnShard(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	xVals, _, err := r.ReadInt32Column("x")
	if err != nil {
		t.Fatal(err)
	}
	yVals, _, err := r.ReadInt32Column("y")
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 100; i++ {
		if xVals[i] != int32(i*3) {
			t.Errorf("x[%d] = %d, want %d", i, xVals[i], i*3)
		}
		if yVals[i] != int32(1000-i) {
			t.Errorf("y[%d] = %d, want %d", i, yVals[i], 1000-i)
		}
	}
}

// =============================================================================
// HELPERS
// =============================================================================

func generateTimestamps(n int) []int64 {
	ts := make([]int64, n)
	base := int64(1700000000000000000) // ~2023 in nanoseconds
	for i := range ts {
		base += int64(1000000 + i*100) // ~1ms + small increment
		ts[i] = base
	}
	return ts
}

func repeat[T any](val T, n int) []T {
	s := make([]T, n)
	for i := range s {
		s[i] = val
	}
	return s
}

func repeatStr(val string, n int) []string {
	s := make([]string, n)
	for i := range s {
		s[i] = val
	}
	return s
}

func uniqueStrings(n int) []string {
	s := make([]string, n)
	for i := range s {
		s[i] = fmt.Sprintf("unique_%06d", i)
	}
	return s
}

func alternating(n int) []bool {
	s := make([]bool, n)
	for i := range s {
		s[i] = i%2 == 0
	}
	return s
}

func safeDivPercent(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}
