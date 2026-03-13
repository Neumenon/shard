// mosh.go - MoSH (Model Shard) profile for shard v2.
//
// MoSH stores model weights keyed by layer name for fast, mmap-friendly access.
// This is the standalone version (no cowrie dependency) that uses JSON for metadata.
//
// Features:
//   - Shard v2 format with content type hints
//   - 64-byte aligned tensor data for SIMD/AVX-512 efficiency
//   - Optional per-tensor compression (zstd/lz4)
//   - CRC32C checksums for integrity verification
//   - Header-only scanning for fast layout discovery
//   - Zero-copy mmap access for efficient tensor loading
//   - Embedded metadata support (__metadata__ entry)
package shard

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// MetadataEntryName is the reserved entry name for model metadata.
const MetadataEntryName = "__metadata__"

// ============================================================
// MoSHWriter
// ============================================================

// MoSHWriter writes MoSH shard files with model weights.
type MoSHWriter struct {
	sw          *ShardV2Writer
	shardMeta   *ShardMetadata
	modelMeta   any
	tensorCount int
}

// NewMoSHWriter creates a new MoSH writer with 64-byte alignment.
func NewMoSHWriter(path string) (*MoSHWriter, error) {
	sw, err := NewShardV2Writer(path, ShardRoleMoSH)
	if err != nil {
		return nil, err
	}
	sw.SetAlignment(Align64)

	meta := NewShardMetadata()
	meta.Producer = "mosh-writer"

	return &MoSHWriter{
		sw:        sw,
		shardMeta: meta,
	}, nil
}

// SetAlignment sets data alignment (0, 16, 32, or 64 bytes).
func (w *MoSHWriter) SetAlignment(align uint8) error {
	return w.sw.SetAlignment(align)
}

// SetCompression sets the default compression type.
func (w *MoSHWriter) SetCompression(comp uint8) {
	w.sw.SetCompression(comp)
}

// SetProducer sets the producer name in shard metadata.
func (w *MoSHWriter) SetProducer(producer string) {
	w.shardMeta.Producer = producer
}

// SetDescription sets the description in shard metadata.
func (w *MoSHWriter) SetDescription(desc string) {
	w.shardMeta.Description = desc
}

// SetSourceURI sets the source URI in shard metadata.
func (w *MoSHWriter) SetSourceURI(uri string) {
	w.shardMeta.SourceURI = uri
}

// AddTag adds a tag to shard metadata.
func (w *MoSHWriter) AddTag(tag string) {
	w.shardMeta.AddTag(tag)
}

// AddTensor adds a tensor to the shard.
func (w *MoSHWriter) AddTensor(name string, t *TensorV1) error {
	blob, err := EncodeTensorV1(t)
	if err != nil {
		return fmt.Errorf("encode tensor %q: %w", name, err)
	}
	if err := w.sw.WriteEntryTyped(name, blob, ContentTypeTensor); err != nil {
		return err
	}
	w.tensorCount++
	return nil
}

// AddTensorCompressed adds a compressed tensor.
func (w *MoSHWriter) AddTensorCompressed(name string, t *TensorV1) error {
	blob, err := EncodeTensorV1(t)
	if err != nil {
		return fmt.Errorf("encode tensor %q: %w", name, err)
	}
	if err := w.sw.writeEntryFull(name, blob, true, w.sw.header.CompressionDefault, ContentTypeTensor); err != nil {
		return err
	}
	w.tensorCount++
	return nil
}

// AddTensorFloat32 adds a float32 tensor.
func (w *MoSHWriter) AddTensorFloat32(name string, data []float32, dims ...uint64) error {
	return w.AddTensor(name, NewTensorV1Float32(data, dims...))
}

// AddTensorFloat64 adds a float64 tensor.
func (w *MoSHWriter) AddTensorFloat64(name string, data []float64, dims ...uint64) error {
	return w.AddTensor(name, NewTensorV1Float64(data, dims...))
}

// AddTensorInt8 adds an int8 tensor.
func (w *MoSHWriter) AddTensorInt8(name string, data []int8, dims ...uint64) error {
	return w.AddTensor(name, NewTensorV1Int8(data, dims...))
}

// AddTensorInt32 adds an int32 tensor.
func (w *MoSHWriter) AddTensorInt32(name string, data []int32, dims ...uint64) error {
	return w.AddTensor(name, NewTensorV1Int32(data, dims...))
}

// AddTensorInt64 adds an int64 tensor.
func (w *MoSHWriter) AddTensorInt64(name string, data []int64, dims ...uint64) error {
	return w.AddTensor(name, NewTensorV1Int64(data, dims...))
}

// AddQuantizedTensor adds a quantized int8 tensor with per-channel scales.
// The scales are stored as a separate tensor named "{name}.scales".
func (w *MoSHWriter) AddQuantizedTensor(name string, data []int8, scales []float32, dims ...uint64) error {
	if err := w.AddTensorInt8(name, data, dims...); err != nil {
		return err
	}
	return w.AddTensorFloat32(name+".scales", scales)
}

// AddTensorRaw adds a tensor with raw bytes and explicit dtype.
func (w *MoSHWriter) AddTensorRaw(name string, data []byte, dtype DType, dims ...uint64) error {
	return w.AddTensor(name, &TensorV1{DType: dtype, Dims: dims, Data: data})
}

// AddMetadata sets the model metadata to be embedded as JSON.
func (w *MoSHWriter) AddMetadata(metadata any) {
	w.modelMeta = metadata
}

// Close finalizes the shard file, writing metadata if set.
func (w *MoSHWriter) Close() error {
	if w.modelMeta != nil {
		data, err := json.Marshal(w.modelMeta)
		if err != nil {
			return fmt.Errorf("encode metadata: %w", err)
		}
		if err := w.sw.WriteEntryTyped(MetadataEntryName, data, ContentTypeJSON); err != nil {
			return fmt.Errorf("write metadata: %w", err)
		}
	}

	w.sw.SetMetadata(w.shardMeta)
	return w.sw.Close()
}

// TensorCount returns the number of tensors written so far.
func (w *MoSHWriter) TensorCount() int {
	return w.tensorCount
}

// ============================================================
// MoSHReader
// ============================================================

// MoSHReader reads MoSH shard files.
type MoSHReader struct {
	sr *ShardV2Reader
}

// OpenMoSH opens a MoSH shard file for reading.
// Accepts MoSH (0x01) and Unknown (0x00) roles for compatibility.
func OpenMoSH(path string) (*MoSHReader, error) {
	sr, err := OpenShardV2(path)
	if err != nil {
		return nil, err
	}

	switch sr.header.Role {
	case ShardRoleMoSH, ShardRoleUnknown:
		// OK
	default:
		sr.Close()
		return nil, fmt.Errorf("expected MoSH role (0x01), got %s (0x%02x)", sr.header.Role, uint8(sr.header.Role))
	}

	return &MoSHReader{sr: sr}, nil
}

// TensorNames returns all tensor names (excluding reserved __ entries).
func (r *MoSHReader) TensorNames() []string {
	allNames := r.sr.EntryNames()
	names := make([]string, 0, len(allNames))
	for _, name := range allNames {
		if !isReservedEntry(name) {
			names = append(names, name)
		}
	}
	return names
}

// TensorCount returns the number of tensors (excluding reserved entries).
func (r *MoSHReader) TensorCount() int {
	count := 0
	for i := 0; i < r.sr.EntryCount(); i++ {
		if !isReservedEntry(r.sr.EntryName(i)) {
			count++
		}
	}
	return count
}

// HasTensor checks if a tensor exists by name.
func (r *MoSHReader) HasTensor(name string) bool {
	return r.sr.Lookup(name) >= 0
}

// GetTensor reads a tensor by name.
func (r *MoSHReader) GetTensor(name string) (*TensorV1, error) {
	data, err := r.sr.ReadEntryByName(name)
	if err != nil {
		return nil, err
	}
	return DecodeTensorV1(data)
}

// GetTensorByIndex reads a tensor by index.
func (r *MoSHReader) GetTensorByIndex(i int) (*TensorV1, error) {
	data, err := r.sr.ReadEntry(i)
	if err != nil {
		return nil, err
	}
	return DecodeTensorV1(data)
}

// LoadTensorMapped returns a zero-copy view of tensor data if mmap is enabled.
func (r *MoSHReader) LoadTensorMapped(name string) (*MappedTensor, error) {
	i := r.sr.Lookup(name)
	if i < 0 {
		return nil, fmt.Errorf("%w: %q", ErrEntryNotFound, name)
	}

	data, err := r.sr.ReadEntry(i)
	if err != nil {
		return nil, err
	}

	t, err := DecodeTensorV1(data)
	if err != nil {
		return nil, err
	}

	return &MappedTensor{
		DType: t.DType,
		Dims:  t.Dims,
		Data:  t.Data,
	}, nil
}

// GetMetadata reads the embedded JSON metadata from __metadata__.
func (r *MoSHReader) GetMetadata(target any) error {
	if r.sr.Lookup(MetadataEntryName) < 0 {
		return fmt.Errorf("%w: %s", ErrEntryNotFound, MetadataEntryName)
	}
	data, err := r.sr.ReadEntryByName(MetadataEntryName)
	if err != nil {
		return fmt.Errorf("read metadata: %w", err)
	}
	return json.Unmarshal(data, target)
}

// EnableMmap enables memory-mapped access for zero-copy tensor loading.
func (r *MoSHReader) EnableMmap() error {
	return r.sr.EnableMmap()
}

// Close closes the reader.
func (r *MoSHReader) Close() error {
	return r.sr.Close()
}

// Header returns the underlying shard v2 header.
func (r *MoSHReader) Header() *ShardV2Header {
	return r.sr.Header()
}

// ============================================================
// InstantLoader - mmap-based weight loading
// ============================================================

// LayerInfo contains metadata about a tensor layer.
type LayerInfo struct {
	Name   string   // Layer name
	Offset int64    // Byte offset in shard file
	Size   int64    // Size on disk (may be compressed)
	DType  DType    // Data type
	Shape  []uint64 // Tensor dimensions
}

// WeightLayout maps layer paths to tensor metadata.
type WeightLayout struct {
	Layers map[string]LayerInfo
}

// InstantLoader provides mmap-based weight access for fast LLM loading.
type InstantLoader struct {
	reader *MoSHReader
	layout *WeightLayout
}

// NewInstantLoader creates a new instant loader for a MoSH file.
// Builds the layout using header-only reads (no tensor data loaded).
func NewInstantLoader(path string) (*InstantLoader, error) {
	reader, err := OpenMoSH(path)
	if err != nil {
		return nil, err
	}

	// Enable mmap (optional — fallback to read if unavailable)
	_ = reader.EnableMmap()

	layout, err := buildWeightLayout(reader)
	if err != nil {
		reader.Close()
		return nil, fmt.Errorf("build layout: %w", err)
	}

	return &InstantLoader{
		reader: reader,
		layout: layout,
	}, nil
}

func buildWeightLayout(r *MoSHReader) (*WeightLayout, error) {
	layout := &WeightLayout{Layers: make(map[string]LayerInfo)}
	for i := 0; i < r.sr.EntryCount(); i++ {
		name := r.sr.EntryName(i)
		if isReservedEntry(name) {
			continue
		}
		info := r.sr.GetEntryInfo(i)
		if info == nil {
			continue
		}
		headerData, err := r.sr.ReadEntryPrefix(i, MaxTensorV1HeaderSize)
		if err != nil {
			continue
		}
		dtype, dims, _, err := DecodeTensorV1Header(headerData)
		if err != nil {
			continue
		}
		layout.Layers[name] = LayerInfo{
			Name:   name,
			Offset: int64(info.DataOffset),
			Size:   int64(info.DiskSize),
			DType:  dtype,
			Shape:  dims,
		}
	}
	return layout, nil
}

// Layout returns the weight layout.
func (l *InstantLoader) Layout() *WeightLayout {
	return l.layout
}

// LayerNames returns all layer names sorted alphabetically.
func (l *InstantLoader) LayerNames() []string {
	names := make([]string, 0, len(l.layout.Layers))
	for name := range l.layout.Layers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// GetLayerInfo returns metadata for a specific layer.
func (l *InstantLoader) GetLayerInfo(name string) (LayerInfo, bool) {
	info, ok := l.layout.Layers[name]
	return info, ok
}

// LoadTensor loads a tensor by name.
func (l *InstantLoader) LoadTensor(name string) (*TensorV1, error) {
	return l.reader.GetTensor(name)
}

// LoadTensorMapped returns a zero-copy view of tensor data.
func (l *InstantLoader) LoadTensorMapped(name string) (*MappedTensor, error) {
	return l.reader.LoadTensorMapped(name)
}

// Close closes the loader.
func (l *InstantLoader) Close() error {
	return l.reader.Close()
}

// ============================================================
// Helpers
// ============================================================

// isReservedEntry returns true if the entry name is reserved (e.g., "__metadata__").
func isReservedEntry(name string) bool {
	return len(name) >= 4 && strings.HasPrefix(name, "__") && strings.HasSuffix(name, "__")
}

// Compression constants re-exported for convenience.
const (
	MoSHCompressNone = CompressNone
	MoSHCompressZstd = CompressZstd
	MoSHCompressLZ4  = CompressLZ4
)

// Alignment values re-exported for convenience.
const (
	MoSHAlignNone = AlignNone
	MoSHAlign16   = Align16
	MoSHAlign32   = Align32
	MoSHAlign64   = Align64
)
