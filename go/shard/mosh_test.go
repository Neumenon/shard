package shard

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ============================================================
// Lifecycle test (Class 2 shape): create → write → read → verify
// ============================================================

func TestMoSHLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lifecycle.mosh")

	// CREATE + POPULATE
	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatalf("create writer: %v", err)
	}
	w.SetProducer("test-harness")
	w.SetDescription("lifecycle test")
	w.SetSourceURI("https://example.com/model")
	w.AddTag("test")
	w.AddTag("lifecycle")

	// Write tensors of varying dtypes
	if err := w.AddTensorFloat32("layer.0.weight", []float32{1, 2, 3, 4}, 2, 2); err != nil {
		t.Fatalf("add f32: %v", err)
	}
	if err := w.AddTensorFloat64("layer.0.bias", []float64{0.1, 0.2}); err != nil {
		t.Fatalf("add f64: %v", err)
	}
	if err := w.AddTensorInt8("layer.1.quant", []int8{-10, 20, -30, 40, 50, -60}, 2, 3); err != nil {
		t.Fatalf("add i8: %v", err)
	}
	if err := w.AddTensorInt32("layer.1.ids", []int32{100, 200, 300}); err != nil {
		t.Fatalf("add i32: %v", err)
	}
	if err := w.AddTensorInt64("embedding.offsets", []int64{0, 1024, 2048}); err != nil {
		t.Fatalf("add i64: %v", err)
	}

	w.AddMetadata(map[string]any{
		"architecture": "llama",
		"hidden_size":  4096,
		"num_layers":   32,
	})

	if w.TensorCount() != 5 {
		t.Errorf("writer TensorCount = %d, want 5", w.TensorCount())
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}

	// READ + VERIFY
	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	// Role
	if r.Header().Role != ShardRoleMoSH {
		t.Errorf("role = 0x%02x, want 0x%02x (MoSH)", uint8(r.Header().Role), uint8(ShardRoleMoSH))
	}

	// Tensor count excludes reserved entries
	if r.TensorCount() != 5 {
		t.Errorf("reader TensorCount = %d, want 5", r.TensorCount())
	}

	// TensorNames excludes __metadata__
	names := r.TensorNames()
	if len(names) != 5 {
		t.Errorf("len(TensorNames) = %d, want 5", len(names))
	}
	for _, name := range names {
		if strings.HasPrefix(name, "__") {
			t.Errorf("TensorNames includes reserved entry %q", name)
		}
	}

	// Verify each tensor's dtype, shape, and data
	t.Run("f32", func(t *testing.T) {
		tensor, err := r.GetTensor("layer.0.weight")
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		if tensor.DType != DTypeFloat32 {
			t.Errorf("dtype = %s, want float32", DTypeName(tensor.DType))
		}
		if len(tensor.Dims) != 2 || tensor.Dims[0] != 2 || tensor.Dims[1] != 2 {
			t.Errorf("dims = %v, want [2,2]", tensor.Dims)
		}
		data := tensor.AsFloat32()
		expected := []float32{1, 2, 3, 4}
		if len(data) != len(expected) {
			t.Fatalf("len = %d, want %d", len(data), len(expected))
		}
		for i := range expected {
			if data[i] != expected[i] {
				t.Errorf("data[%d] = %v, want %v", i, data[i], expected[i])
			}
		}
	})

	t.Run("f64", func(t *testing.T) {
		tensor, err := r.GetTensor("layer.0.bias")
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		if tensor.DType != DTypeFloat64 {
			t.Errorf("dtype = %s, want float64", DTypeName(tensor.DType))
		}
		data := tensor.AsFloat64()
		if data[0] != 0.1 || data[1] != 0.2 {
			t.Errorf("data = %v, want [0.1, 0.2]", data)
		}
	})

	t.Run("i8", func(t *testing.T) {
		tensor, err := r.GetTensor("layer.1.quant")
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		if tensor.DType != DTypeInt8 {
			t.Errorf("dtype = %s, want int8", DTypeName(tensor.DType))
		}
		if len(tensor.Dims) != 2 || tensor.Dims[0] != 2 || tensor.Dims[1] != 3 {
			t.Errorf("dims = %v, want [2,3]", tensor.Dims)
		}
	})

	t.Run("i32", func(t *testing.T) {
		tensor, err := r.GetTensor("layer.1.ids")
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		got := tensor.AsInt32()
		if got[0] != 100 || got[1] != 200 || got[2] != 300 {
			t.Errorf("data = %v, want [100,200,300]", got)
		}
	})

	t.Run("i64", func(t *testing.T) {
		tensor, err := r.GetTensor("embedding.offsets")
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		got := tensor.AsInt64()
		if got[0] != 0 || got[1] != 1024 || got[2] != 2048 {
			t.Errorf("data = %v, want [0,1024,2048]", got)
		}
	})

	// Metadata roundtrip
	var meta map[string]any
	if err := r.GetMetadata(&meta); err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	if meta["architecture"] != "llama" {
		t.Errorf("architecture = %v, want llama", meta["architecture"])
	}
	if meta["hidden_size"] != float64(4096) {
		t.Errorf("hidden_size = %v, want 4096", meta["hidden_size"])
	}
	if meta["num_layers"] != float64(32) {
		t.Errorf("num_layers = %v, want 32", meta["num_layers"])
	}

	// HasTensor
	if !r.HasTensor("layer.0.weight") {
		t.Error("HasTensor(layer.0.weight) = false, want true")
	}
	if r.HasTensor("nonexistent") {
		t.Error("HasTensor(nonexistent) = true, want false")
	}
}

// ============================================================
// Invariant 2: encode(decode) == encode (canonical roundtrip via shard)
// ============================================================

func TestMoSHCanonicalRoundtrip(t *testing.T) {
	// Write a shard, read it, write it again — tensor data must be identical.
	path1 := filepath.Join(t.TempDir(), "canon1.mosh")
	path2 := filepath.Join(t.TempDir(), "canon2.mosh")

	data := []float32{1.0, -2.5, 3.14, 0.0}

	// Write first
	w1, err := NewMoSHWriter(path1)
	if err != nil {
		t.Fatal(err)
	}
	if err := w1.AddTensorFloat32("t", data, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := w1.Close(); err != nil {
		t.Fatal(err)
	}

	// Read back
	r1, err := OpenMoSH(path1)
	if err != nil {
		t.Fatal(err)
	}
	tensor, err := r1.GetTensor("t")
	if err != nil {
		t.Fatal(err)
	}
	r1.Close()

	// Write again from decoded tensor
	w2, err := NewMoSHWriter(path2)
	if err != nil {
		t.Fatal(err)
	}
	if err := w2.AddTensor("t", tensor); err != nil {
		t.Fatal(err)
	}
	if err := w2.Close(); err != nil {
		t.Fatal(err)
	}

	// Read back again and verify data identity
	r2, err := OpenMoSH(path2)
	if err != nil {
		t.Fatal(err)
	}
	defer r2.Close()

	tensor2, err := r2.GetTensor("t")
	if err != nil {
		t.Fatal(err)
	}

	got := tensor2.AsFloat32()
	for i, v := range data {
		if got[i] != v {
			t.Errorf("roundtrip data[%d] = %v, want %v", i, got[i], v)
		}
	}
}

// ============================================================
// Compression roundtrip
// ============================================================

func TestMoSHCompressedRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compressed.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	w.SetCompression(CompressZstd)

	// Large tensor to trigger actual compression (minCompressSize threshold)
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}
	if err := w.AddTensorCompressed("big", NewTensorV1Float32(data, 1000)); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	tensor, err := r.GetTensor("big")
	if err != nil {
		t.Fatalf("get tensor: %v", err)
	}
	got := tensor.AsFloat32()
	if len(got) != 1000 {
		t.Fatalf("len = %d, want 1000", len(got))
	}
	// Verify every element — not just spot check
	for i, v := range got {
		if v != float32(i) {
			t.Errorf("data[%d] = %v, want %v", i, v, float32(i))
			break
		}
	}
}

// ============================================================
// Class 14: Metadata survives roundtrip with typed struct
// ============================================================

func TestMoSHMetadataStructRoundtrip(t *testing.T) {
	type ModelMeta struct {
		Architecture string  `json:"architecture"`
		HiddenSize   int     `json:"hidden_size"`
		VocabSize    int     `json:"vocab_size"`
		RopeTheta    float64 `json:"rope_theta"`
		QuantType    string  `json:"quant_type"`
	}

	path := filepath.Join(t.TempDir(), "structmeta.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	w.AddMetadata(ModelMeta{
		Architecture: "llama-3",
		HiddenSize:   4096,
		VocabSize:    128256,
		RopeTheta:    500000.0,
		QuantType:    "q4_k_m",
	})
	if err := w.AddTensorFloat32("dummy", []float32{0}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	var meta ModelMeta
	if err := r.GetMetadata(&meta); err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	if meta.Architecture != "llama-3" {
		t.Errorf("architecture = %q, want llama-3", meta.Architecture)
	}
	if meta.HiddenSize != 4096 {
		t.Errorf("hidden_size = %d, want 4096", meta.HiddenSize)
	}
	if meta.VocabSize != 128256 {
		t.Errorf("vocab_size = %d, want 128256", meta.VocabSize)
	}
	if meta.RopeTheta != 500000.0 {
		t.Errorf("rope_theta = %v, want 500000.0", meta.RopeTheta)
	}
	if meta.QuantType != "q4_k_m" {
		t.Errorf("quant_type = %q, want q4_k_m", meta.QuantType)
	}
}

// ============================================================
// Error paths — specific error verification (Rule 4)
// ============================================================

func TestMoSHGetMetadataWithoutMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nometa.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("t", []float32{1}); err != nil {
		t.Fatal(err)
	}
	// No AddMetadata call
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	var meta map[string]any
	err = r.GetMetadata(&meta)
	if err == nil {
		t.Fatal("GetMetadata on shard without metadata should return error")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want 'not found' error", err.Error())
	}
}

func TestMoSHGetTensorNotFound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nope.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("exists", []float32{1}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	_, err = r.GetTensor("does_not_exist")
	if err == nil {
		t.Fatal("GetTensor for nonexistent name should return error")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want 'not found' error", err.Error())
	}
}

func TestMoSHLoadTensorMappedNotFound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mapped_nf.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("t", []float32{1}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	_, err = r.LoadTensorMapped("missing")
	if err == nil {
		t.Fatal("LoadTensorMapped for nonexistent name should return error")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want 'not found' error", err.Error())
	}
}

func TestMoSHRoleValidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wrongrole.shard")

	// Write a WShard
	w, err := NewShardV2Writer(path, ShardRoleWShard)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteEntry("dummy", []byte("data")); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Try to open as MoSH
	_, err = OpenMoSH(path)
	if err == nil {
		t.Fatal("OpenMoSH on WShard should fail")
	}
	if !strings.Contains(err.Error(), "MoSH") {
		t.Errorf("error = %q, want mention of MoSH role", err.Error())
	}
}

func TestMoSHOpenNonexistentFile(t *testing.T) {
	_, err := OpenMoSH("/nonexistent/path/model.mosh")
	if err == nil {
		t.Fatal("OpenMoSH on nonexistent file should fail")
	}
}

// ============================================================
// Quantized tensor lifecycle (Class 2 shape)
// ============================================================

func TestMoSHQuantizedTensorLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quant.mosh")

	// Write quantized data + scales
	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	data := []int8{10, 20, 30, 40}
	scales := []float32{0.1, 0.5}
	if err := w.AddQuantizedTensor("weight", data, scales, 2, 2); err != nil {
		t.Fatal(err)
	}
	if w.TensorCount() != 2 { // data + scales
		t.Errorf("TensorCount = %d, want 2", w.TensorCount())
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Read back
	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	// Verify data tensor
	tensor, err := r.GetTensor("weight")
	if err != nil {
		t.Fatalf("get weight: %v", err)
	}
	if tensor.DType != DTypeInt8 {
		t.Errorf("weight dtype = %s, want int8", DTypeName(tensor.DType))
	}
	i8data := tensor.AsInt8()
	if i8data[0] != 10 || i8data[3] != 40 {
		t.Errorf("weight data = %v", i8data)
	}

	// Verify scales tensor
	sTensor, err := r.GetTensor("weight.scales")
	if err != nil {
		t.Fatalf("get scales: %v", err)
	}
	sData := sTensor.AsFloat32()
	if len(sData) != 2 || sData[0] != 0.1 || sData[1] != 0.5 {
		t.Errorf("scales = %v, want [0.1, 0.5]", sData)
	}

	// Dequantize and verify
	deq, err := tensor.ToFloat32QuantizedPerRow(sData)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	// Row 0: [10*0.1, 20*0.1] = [1.0, 2.0]
	// Row 1: [30*0.5, 40*0.5] = [15.0, 20.0]
	expected := []float32{1.0, 2.0, 15.0, 20.0}
	for i, v := range expected {
		if math.Abs(float64(deq[i]-v)) > 1e-6 {
			t.Errorf("deq[%d] = %v, want %v", i, deq[i], v)
		}
	}
}

// ============================================================
// InstantLoader (header-only scan)
// ============================================================

func TestMoSHInstantLoaderLayout(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loader.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("attn.q", []float32{1, 2, 3, 4, 5, 6}, 2, 3); err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorInt8("attn.k", []int8{1, 2, 3, 4}, 4); err != nil {
		t.Fatal(err)
	}
	w.AddMetadata(map[string]any{"model": "test"})
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	loader, err := NewInstantLoader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loader.Close()

	layout := loader.Layout()
	if len(layout.Layers) != 2 {
		t.Errorf("Layout layers = %d, want 2 (should exclude __metadata__)", len(layout.Layers))
	}

	// Verify attn.q layout info
	info, ok := loader.GetLayerInfo("attn.q")
	if !ok {
		t.Fatal("attn.q not in layout")
	}
	if info.DType != DTypeFloat32 {
		t.Errorf("attn.q dtype = %s, want float32", DTypeName(info.DType))
	}
	if len(info.Shape) != 2 || info.Shape[0] != 2 || info.Shape[1] != 3 {
		t.Errorf("attn.q shape = %v, want [2,3]", info.Shape)
	}
	if info.Offset <= 0 {
		t.Errorf("attn.q offset = %d, want > 0", info.Offset)
	}
	if info.Size <= 0 {
		t.Errorf("attn.q size = %d, want > 0", info.Size)
	}

	// Verify attn.k layout info
	kInfo, ok := loader.GetLayerInfo("attn.k")
	if !ok {
		t.Fatal("attn.k not in layout")
	}
	if kInfo.DType != DTypeInt8 {
		t.Errorf("attn.k dtype = %s, want int8", DTypeName(kInfo.DType))
	}

	// LayerNames — sorted
	names := loader.LayerNames()
	if len(names) != 2 {
		t.Fatalf("LayerNames = %v", names)
	}
	if names[0] != "attn.k" || names[1] != "attn.q" {
		t.Errorf("LayerNames = %v, want [attn.k, attn.q] (sorted)", names)
	}

	// Missing layer
	_, ok = loader.GetLayerInfo("nonexistent")
	if ok {
		t.Error("GetLayerInfo for nonexistent should return false")
	}

	// LoadTensor via loader
	tensor, err := loader.LoadTensor("attn.q")
	if err != nil {
		t.Fatalf("LoadTensor: %v", err)
	}
	got := tensor.AsFloat32()
	if got[0] != 1.0 || got[5] != 6.0 {
		t.Errorf("data = %v", got)
	}
}

// ============================================================
// Mmap access
// ============================================================

func TestMoSHMmapAccess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mmap.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("t1", []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if err := r.EnableMmap(); err != nil {
		t.Fatalf("EnableMmap: %v", err)
	}

	mapped, err := r.LoadTensorMapped("t1")
	if err != nil {
		t.Fatalf("LoadTensorMapped: %v", err)
	}
	if mapped.DType != DTypeFloat32 {
		t.Errorf("dtype = %s, want float32", DTypeName(mapped.DType))
	}
	if len(mapped.Dims) != 1 || mapped.Dims[0] != 3 {
		t.Errorf("dims = %v, want [3]", mapped.Dims)
	}
	expectedBytes := int(TensorV1DataSize(DTypeFloat32, []uint64{3}))
	if len(mapped.Data) != expectedBytes {
		t.Errorf("data len = %d, want %d", len(mapped.Data), expectedBytes)
	}
}

// ============================================================
// AddTensorRaw — preserving native formats
// ============================================================

func TestMoSHAddTensorRawF16(t *testing.T) {
	path := filepath.Join(t.TempDir(), "raw.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}

	// Raw float16 data: 0x3C00=1.0, 0x4000=2.0
	raw := []byte{0x00, 0x3C, 0x00, 0x40}
	if err := w.AddTensorRaw("f16", raw, DTypeFloat16, 2); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	tensor, err := r.GetTensor("f16")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if tensor.DType != DTypeFloat16 {
		t.Errorf("dtype = %s, want float16", DTypeName(tensor.DType))
	}
	f32 := tensor.ToFloat32()
	if f32[0] != 1.0 || f32[1] != 2.0 {
		t.Errorf("f16→f32 = %v, want [1.0, 2.0]", f32)
	}
}

// ============================================================
// GetTensorByIndex
// ============================================================

func TestMoSHGetTensorByIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "byindex.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("a", []float32{1}); err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("b", []float32{2}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	t0, err := r.GetTensorByIndex(0)
	if err != nil {
		t.Fatalf("index 0: %v", err)
	}
	if t0.AsFloat32()[0] != 1 {
		t.Errorf("tensor[0] = %v, want 1", t0.AsFloat32()[0])
	}

	t1, err := r.GetTensorByIndex(1)
	if err != nil {
		t.Fatalf("index 1: %v", err)
	}
	if t1.AsFloat32()[0] != 2 {
		t.Errorf("tensor[1] = %v, want 2", t1.AsFloat32()[0])
	}
}

// ============================================================
// Scale test: 1M elements
// ============================================================

func TestMoSHLargeTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}

	n := 1_000_000
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i) * 0.001
	}
	if err := w.AddTensorFloat32("big", data, uint64(n)); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	tensor, err := r.GetTensor("big")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	got := tensor.AsFloat32()
	if len(got) != n {
		t.Fatalf("len = %d, want %d", len(got), n)
	}

	// Verify first, middle, and last elements — not just spot check
	if got[0] != 0.0 {
		t.Errorf("got[0] = %v, want 0.0", got[0])
	}
	mid := n / 2
	expectedMid := float32(mid) * 0.001
	if math.Abs(float64(got[mid]-expectedMid)) > 0.01 {
		t.Errorf("got[%d] = %v, want ~%v", mid, got[mid], expectedMid)
	}
	last := n - 1
	expectedLast := float32(last) * 0.001
	if math.Abs(float64(got[last]-expectedLast)) > 0.01 {
		t.Errorf("got[%d] = %v, want ~%v", last, got[last], expectedLast)
	}
}

// ============================================================
// Empty shard — metadata only
// ============================================================

func TestMoSHEmptyShard(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	w.AddMetadata(map[string]any{"model": "empty"})
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if r.TensorCount() != 0 {
		t.Errorf("TensorCount = %d, want 0", r.TensorCount())
	}
	names := r.TensorNames()
	if len(names) != 0 {
		t.Errorf("TensorNames = %v, want []", names)
	}

	var meta map[string]any
	if err := r.GetMetadata(&meta); err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	if meta["model"] != "empty" {
		t.Errorf("metadata model = %v, want 'empty'", meta["model"])
	}
}

// ============================================================
// NaN/Inf in MoSH roundtrip (Invariant 6 at MoSH layer)
// ============================================================

func TestMoSHNaNInfRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "naninf.mosh")

	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("special", []float32{nan, pinf, ninf, 1.0}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenMoSH(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	tensor, err := r.GetTensor("special")
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.AsFloat32()
	if !math.IsNaN(float64(got[0])) {
		t.Errorf("got[0] = %v, want NaN", got[0])
	}
	if !math.IsInf(float64(got[1]), 1) {
		t.Errorf("got[1] = %v, want +Inf", got[1])
	}
	if !math.IsInf(float64(got[2]), -1) {
		t.Errorf("got[2] = %v, want -Inf", got[2])
	}
	if got[3] != 1.0 {
		t.Errorf("got[3] = %v, want 1.0", got[3])
	}
}

// ============================================================
// File existence verification
// ============================================================

func TestMoSHFileCreated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "exists.mosh")

	w, err := NewMoSHWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensorFloat32("t", []float32{1}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file should exist: %v", err)
	}
	if info.Size() == 0 {
		t.Error("file size = 0, want > 0")
	}
}

// ============================================================
// isReservedEntry helper
// ============================================================

func TestIsReservedEntry(t *testing.T) {
	reserved := []string{"__metadata__", "__schema__", "__any__"}
	for _, name := range reserved {
		if !isReservedEntry(name) {
			t.Errorf("%q should be reserved", name)
		}
	}
	notReserved := []string{"metadata", "__half", "layer.0.weight", "__", "___"}
	for _, name := range notReserved {
		if isReservedEntry(name) {
			t.Errorf("%q should not be reserved", name)
		}
	}
}
