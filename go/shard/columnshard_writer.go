package shard

import (
	"encoding/binary"
	"fmt"
	"math"
)

// ColumnShardWriter builds a ColumnShard (.cshard) file.
//
// Usage:
//
//	w, _ := NewColumnShardWriter("data.cshard", 65536)
//	w.AddColumn("user_id", ColTypeInt64, false, ColEncDelta)
//	w.AddColumn("label", ColTypeBool, false, ColEncRLE)
//	w.AddColumn("embedding", ColTypeFloat32, false, ColEncByteStreamSplit)
//
//	w.WriteRow(map[string]any{"user_id": int64(1), "label": true, "embedding": []float32{0.1, 0.2}})
//	// ... more rows ...
//
//	w.Close() // flushes final row group, writes schema, finalizes shard
type ColumnShardWriter struct {
	path         string
	shard        *ShardWriter
	schema       *ColumnShardSchema
	rowGroupSize int

	// Per-column buffers for current row group
	colBuffers map[string]*columnBuffer
	colOrder   []string // column insertion order

	// Current row group state
	currentRowGroup int
	rowsInGroup     int
}

// columnBuffer accumulates values for one column in the current row group.
type columnBuffer struct {
	def    *ColumnShardColDef
	values []any  // raw values (typed)
	valid  []bool // validity (for nullable columns)
}

// NewColumnShardWriter creates a new ColumnShard writer.
func NewColumnShardWriter(path string, rowGroupSize int) (*ColumnShardWriter, error) {
	if rowGroupSize <= 0 {
		rowGroupSize = 65536
	}

	shard, err := NewShardWriter(path, ShardRoleColumn)
	if err != nil {
		return nil, err
	}
	shard.SetCompression(CompressZstd)

	return &ColumnShardWriter{
		path:         path,
		shard:        shard,
		schema:       NewColumnShardSchema(uint32(rowGroupSize)),
		rowGroupSize: rowGroupSize,
		colBuffers:   make(map[string]*columnBuffer),
	}, nil
}

// AddColumn defines a column. Must be called before any WriteRow calls.
func (w *ColumnShardWriter) AddColumn(name string, typ ColType, nullable bool, encoding byte) {
	col := w.schema.AddColumn(name, typ, nullable, encoding)
	w.colBuffers[name] = &columnBuffer{
		def:    col,
		values: make([]any, 0, w.rowGroupSize),
	}
	if nullable {
		w.colBuffers[name].valid = make([]bool, 0, w.rowGroupSize)
	}
	w.colOrder = append(w.colOrder, name)
}

// WriteRow writes a single row. Values are keyed by column name.
// Missing keys for nullable columns are treated as null.
// Missing keys for non-nullable columns cause an error.
func (w *ColumnShardWriter) WriteRow(row map[string]any) error {
	for _, name := range w.colOrder {
		buf := w.colBuffers[name]
		v, ok := row[name]
		if !ok {
			if !buf.def.Nullable {
				return fmt.Errorf("columnshard: missing non-nullable column %q", name)
			}
			buf.values = append(buf.values, nil)
			buf.valid = append(buf.valid, false)
		} else {
			buf.values = append(buf.values, v)
			if buf.def.Nullable {
				buf.valid = append(buf.valid, true)
			}
		}
	}

	w.rowsInGroup++
	w.schema.RowCount++

	if w.rowsInGroup >= w.rowGroupSize {
		return w.flushRowGroup()
	}
	return nil
}

// WriteRows writes multiple rows.
func (w *ColumnShardWriter) WriteRows(rows []map[string]any) error {
	for _, row := range rows {
		if err := w.WriteRow(row); err != nil {
			return err
		}
	}
	return nil
}

// Close flushes the final row group, writes the schema, and closes the shard.
func (w *ColumnShardWriter) Close() error {
	// Flush remaining rows
	if w.rowsInGroup > 0 {
		if err := w.flushRowGroup(); err != nil {
			return err
		}
	}

	// Write schema entry
	schemaBytes, err := w.schema.Marshal()
	if err != nil {
		return fmt.Errorf("columnshard: marshal schema: %w", err)
	}
	if err := w.shard.WriteEntryTyped(SchemaEntryName, schemaBytes, ContentTypeCowrie); err != nil {
		return err
	}

	return w.shard.Close()
}

// flushRowGroup encodes all column buffers and writes them as shard entries.
func (w *ColumnShardWriter) flushRowGroup() error {
	count := uint32(w.rowsInGroup)
	rgIdx := w.currentRowGroup

	for _, name := range w.colOrder {
		buf := w.colBuffers[name]

		// Encode the column data
		encodedData, stats, err := w.encodeColumn(buf, count)
		if err != nil {
			return fmt.Errorf("columnshard: encode column %q rg %d: %w", name, rgIdx, err)
		}

		// Marshal the column chunk (header + bitmap + data)
		chunk := MarshalColumnChunk(buf.def.Encoding, count, buf.def.Nullable, buf.valid, encodedData)

		// Write to shard
		entryName := ColumnChunkEntryName(rgIdx, name)
		if err := w.shard.WriteEntryCompressed(entryName, chunk); err != nil {
			return err
		}

		// Record stats
		buf.def.Stats = append(buf.def.Stats, stats)

		// Reset buffer
		buf.values = buf.values[:0]
		if buf.def.Nullable {
			buf.valid = buf.valid[:0]
		}
	}

	w.currentRowGroup++
	w.schema.RowGroupCount++
	w.rowsInGroup = 0
	return nil
}

// encodeColumn encodes column values using the column's encoding and computes stats.
func (w *ColumnShardWriter) encodeColumn(buf *columnBuffer, count uint32) ([]byte, *ColumnChunkStats, error) {
	stats := &ColumnChunkStats{Count: count}

	// Count nulls
	if buf.def.Nullable {
		for _, v := range buf.valid {
			if !v {
				stats.NullCount++
			}
		}
	}

	switch buf.def.Type {
	case ColTypeBool:
		return w.encodeBoolColumn(buf, stats)
	case ColTypeInt32:
		return w.encodeInt32Column(buf, stats)
	case ColTypeInt64, ColTypeDatetime:
		return w.encodeInt64Column(buf, stats)
	case ColTypeUint64:
		return w.encodeUint64Column(buf, stats)
	case ColTypeFloat32:
		return w.encodeFloat32Column(buf, stats)
	case ColTypeFloat64:
		return w.encodeFloat64Column(buf, stats)
	case ColTypeString:
		return w.encodeStringColumn(buf, stats)
	default:
		return w.encodePlainGeneric(buf, stats)
	}
}

func (w *ColumnShardWriter) encodeBoolColumn(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]bool, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, false)
			continue
		}
		b, ok := v.(bool)
		if !ok {
			return nil, nil, fmt.Errorf("expected bool, got %T", v)
		}
		values = append(values, b)
	}

	// Stats
	hasTrue, hasFalse := false, false
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		if v {
			hasTrue = true
		} else {
			hasFalse = true
		}
	}
	if hasFalse || !hasTrue {
		stats.Min = false
	}
	if hasTrue {
		stats.Max = true
		if !hasFalse {
			stats.Min = true
		}
	}

	return EncodeRLEBool(values), stats, nil
}

func (w *ColumnShardWriter) encodeInt32Column(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]int32, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, 0)
			continue
		}
		var iv int32
		switch x := v.(type) {
		case int32:
			iv = x
		case int:
			iv = int32(x)
		case int64:
			iv = int32(x)
		default:
			return nil, nil, fmt.Errorf("expected int32, got %T", v)
		}
		values = append(values, iv)
	}

	// Stats
	minV, maxV := int32(math.MaxInt32), int32(math.MinInt32)
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if minV <= maxV {
		stats.Min = minV
		stats.Max = maxV
	}

	var data []byte
	switch buf.def.Encoding {
	case ColEncDelta:
		// Promote to int64 for delta encoding
		i64 := make([]int64, len(values))
		for j, v := range values {
			i64[j] = int64(v)
		}
		data = EncodeDeltaInt64(i64)
	default:
		data = EncodePlainInt32(values)
	}
	return data, stats, nil
}

func (w *ColumnShardWriter) encodeInt64Column(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]int64, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, 0)
			continue
		}
		var iv int64
		switch x := v.(type) {
		case int64:
			iv = x
		case int:
			iv = int64(x)
		case int32:
			iv = int64(x)
		default:
			return nil, nil, fmt.Errorf("expected int64, got %T", v)
		}
		values = append(values, iv)
	}

	// Stats
	minV, maxV := int64(math.MaxInt64), int64(math.MinInt64)
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if minV <= maxV {
		stats.Min = minV
		stats.Max = maxV
	}

	var data []byte
	switch buf.def.Encoding {
	case ColEncDelta:
		data = EncodeDeltaInt64(values)
	default:
		data = EncodePlainInt64(values)
	}
	return data, stats, nil
}

func (w *ColumnShardWriter) encodeUint64Column(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]uint64, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, 0)
			continue
		}
		var uv uint64
		switch x := v.(type) {
		case uint64:
			uv = x
		case uint:
			uv = uint64(x)
		case int64:
			uv = uint64(x)
		case int:
			uv = uint64(x)
		default:
			return nil, nil, fmt.Errorf("expected uint64, got %T", v)
		}
		values = append(values, uv)
	}

	// Stats
	minV, maxV := uint64(math.MaxUint64), uint64(0)
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if minV <= maxV {
		stats.Min = minV
		stats.Max = maxV
	}

	return EncodePlainUint64(values), stats, nil
}

func (w *ColumnShardWriter) encodeFloat32Column(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]float32, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, 0)
			continue
		}
		switch x := v.(type) {
		case float32:
			values = append(values, x)
		case float64:
			values = append(values, float32(x))
		case []float32:
			// Flatten embedding arrays — write raw bytes
			values = append(values, x...)
		default:
			return nil, nil, fmt.Errorf("expected float32, got %T", v)
		}
	}

	// Stats
	minV, maxV := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	for i, v := range values {
		if buf.def.Nullable && i < len(buf.valid) && !buf.valid[i] {
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if minV <= maxV {
		stats.Min = minV
		stats.Max = maxV
	}

	var data []byte
	switch buf.def.Encoding {
	case ColEncByteStreamSplit:
		data = EncodeByteStreamSplit32(values)
	default:
		data = EncodePlainFloat32(values)
	}
	return data, stats, nil
}

func (w *ColumnShardWriter) encodeFloat64Column(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]float64, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, 0)
			continue
		}
		switch x := v.(type) {
		case float64:
			values = append(values, x)
		case float32:
			values = append(values, float64(x))
		default:
			return nil, nil, fmt.Errorf("expected float64, got %T", v)
		}
	}

	// Stats
	minV, maxV := math.MaxFloat64, -math.MaxFloat64
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if minV <= maxV {
		stats.Min = minV
		stats.Max = maxV
	}

	var data []byte
	switch buf.def.Encoding {
	case ColEncByteStreamSplit:
		data = EncodeByteStreamSplit64(values)
	default:
		data = EncodePlainFloat64(values)
	}
	return data, stats, nil
}

func (w *ColumnShardWriter) encodeStringColumn(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	values := make([]string, 0, len(buf.values))
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			values = append(values, "")
			continue
		}
		s, ok := v.(string)
		if !ok {
			return nil, nil, fmt.Errorf("expected string, got %T", v)
		}
		values = append(values, s)
	}

	// Stats: min/max lexicographic, truncated to 64 bytes
	var minS, maxS string
	first := true
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		if first {
			minS = v
			maxS = v
			first = false
		} else {
			if v < minS {
				minS = v
			}
			if v > maxS {
				maxS = v
			}
		}
	}
	if !first {
		if len(minS) > 64 {
			minS = minS[:64]
		}
		if len(maxS) > 64 {
			maxS = maxS[:64]
		}
		stats.Min = minS
		stats.Max = maxS
	}

	// Count distinct for dictionary decision
	distinct := make(map[string]struct{})
	for i, v := range values {
		if buf.def.Nullable && !buf.valid[i] {
			continue
		}
		distinct[v] = struct{}{}
	}
	stats.DistinctCount = uint32(len(distinct))

	var data []byte
	switch buf.def.Encoding {
	case ColEncDictionary:
		data = EncodeDictionary(values)
	default:
		data = EncodePlainStrings(values)
	}
	return data, stats, nil
}

// encodePlainGeneric handles types we don't have specialized encoders for yet.
func (w *ColumnShardWriter) encodePlainGeneric(buf *columnBuffer, stats *ColumnChunkStats) ([]byte, *ColumnChunkStats, error) {
	width := buf.def.Type.Width()
	if width < 0 {
		return nil, nil, fmt.Errorf("columnshard: unsupported variable-width type %v for plain encoding", buf.def.Type)
	}

	data := make([]byte, len(buf.values)*width)
	for i, v := range buf.values {
		if buf.def.Nullable && !buf.valid[i] {
			continue // leave zeros
		}
		off := i * width
		switch width {
		case 1:
			switch x := v.(type) {
			case byte:
				data[off] = x
			case int8:
				data[off] = byte(x)
			}
		case 2:
			switch x := v.(type) {
			case int16:
				binary.LittleEndian.PutUint16(data[off:], uint16(x))
			case uint16:
				binary.LittleEndian.PutUint16(data[off:], x)
			}
		case 16:
			// UUID
			if b, ok := v.([]byte); ok && len(b) == 16 {
				copy(data[off:], b)
			}
		}
	}
	return data, stats, nil
}
