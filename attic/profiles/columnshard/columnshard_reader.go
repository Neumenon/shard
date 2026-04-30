package shard

import "fmt"

// ColumnShardReader reads columnar data from a ColumnShard (.cshard) file.
//
// Usage:
//
//	r, _ := OpenColumnShard("data.cshard")
//	defer r.Close()
//
//	schema := r.Schema()
//	ages, valid, _ := r.ReadInt64Column("age")
//	labels, _, _ := r.ReadBoolColumn("label")
//
//	// With predicate pushdown:
//	for rg := 0; rg < r.RowGroupCount(); rg++ {
//	    stats := r.ColumnStats("age", rg)
//	    if stats.Max.(int64) <= 30 { continue } // skip row group
//	    chunk, _ := r.ReadColumnChunk("age", rg)
//	    // process chunk...
//	}
type ColumnShardReader struct {
	shard  *ShardReader
	schema *ColumnShardSchema
}

// OpenColumnShard opens a .cshard file for reading.
func OpenColumnShard(path string) (*ColumnShardReader, error) {
	shard, err := OpenShard(path)
	if err != nil {
		return nil, err
	}

	// Read schema entry
	schemaData, err := shard.ReadEntryByName(SchemaEntryName)
	if err != nil {
		shard.Close()
		return nil, fmt.Errorf("columnshard: read schema: %w", err)
	}

	schema, err := UnmarshalColumnShardSchema(schemaData)
	if err != nil {
		shard.Close()
		return nil, fmt.Errorf("columnshard: parse schema: %w", err)
	}

	return &ColumnShardReader{
		shard:  shard,
		schema: schema,
	}, nil
}

// Schema returns the column schema.
func (r *ColumnShardReader) Schema() *ColumnShardSchema {
	return r.schema
}

// RowGroupCount returns the number of row groups.
func (r *ColumnShardReader) RowGroupCount() int {
	return int(r.schema.RowGroupCount)
}

// RowCount returns total row count.
func (r *ColumnShardReader) RowCount() uint64 {
	return r.schema.RowCount
}

// ColumnNames returns all column names.
func (r *ColumnShardReader) ColumnNames() []string {
	names := make([]string, len(r.schema.Columns))
	for i, c := range r.schema.Columns {
		names[i] = c.Name
	}
	return names
}

// ColumnStats returns statistics for a column in a specific row group.
func (r *ColumnShardReader) ColumnStats(column string, rowGroup int) *ColumnChunkStats {
	col := r.schema.FindColumn(column)
	if col == nil || rowGroup >= len(col.Stats) {
		return nil
	}
	return col.Stats[rowGroup]
}

// ReadRawChunk reads the raw (decompressed) entry bytes for a column chunk.
func (r *ColumnShardReader) ReadRawChunk(column string, rowGroup int) ([]byte, error) {
	entryName := ColumnChunkEntryName(rowGroup, column)
	return r.shard.ReadEntryByName(entryName)
}

// ReadColumnChunk reads and parses a column chunk from a specific row group.
func (r *ColumnShardReader) ReadColumnChunk(column string, rowGroup int) (*ColumnChunk, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	raw, err := r.ReadRawChunk(column, rowGroup)
	if err != nil {
		return nil, err
	}

	return UnmarshalColumnChunk(raw, col.Nullable)
}

// ReadInt64Column reads an entire int64 column across all row groups.
// Returns values and validity bitmap (nil if non-nullable).
func (r *ColumnShardReader) ReadInt64Column(column string) ([]int64, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]int64, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		var values []int64
		switch chunk.Encoding {
		case ColEncDelta:
			values = DecodeDeltaInt64(chunk.Data, int(chunk.Count))
		default:
			values = DecodePlainInt64(chunk.Data, int(chunk.Count))
		}

		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// ReadFloat32Column reads an entire float32 column across all row groups.
func (r *ColumnShardReader) ReadFloat32Column(column string) ([]float32, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]float32, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		var values []float32
		switch chunk.Encoding {
		case ColEncByteStreamSplit:
			values = DecodeByteStreamSplit32(chunk.Data, int(chunk.Count))
		default:
			values = DecodePlainFloat32(chunk.Data, int(chunk.Count))
		}

		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// ReadFloat64Column reads an entire float64 column across all row groups.
func (r *ColumnShardReader) ReadFloat64Column(column string) ([]float64, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]float64, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		var values []float64
		switch chunk.Encoding {
		case ColEncByteStreamSplit:
			values = DecodeByteStreamSplit64(chunk.Data, int(chunk.Count))
		default:
			values = DecodePlainFloat64(chunk.Data, int(chunk.Count))
		}

		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// ReadBoolColumn reads an entire bool column across all row groups.
func (r *ColumnShardReader) ReadBoolColumn(column string) ([]bool, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]bool, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		values := DecodeRLEBool(chunk.Data, int(chunk.Count))
		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// ReadStringColumn reads an entire string column across all row groups.
func (r *ColumnShardReader) ReadStringColumn(column string) ([]string, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]string, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		var values []string
		switch chunk.Encoding {
		case ColEncDictionary:
			values = DecodeDictionary(chunk.Data, int(chunk.Count))
		default:
			values = DecodePlainStrings(chunk.Data, int(chunk.Count))
		}

		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// ReadUint64Column reads an entire uint64 column across all row groups.
func (r *ColumnShardReader) ReadUint64Column(column string) ([]uint64, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]uint64, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		values := DecodePlainUint64(chunk.Data, int(chunk.Count))
		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// ReadInt32Column reads an entire int32 column across all row groups.
func (r *ColumnShardReader) ReadInt32Column(column string) ([]int32, []bool, error) {
	col := r.schema.FindColumn(column)
	if col == nil {
		return nil, nil, fmt.Errorf("columnshard: column %q not found", column)
	}

	result := make([]int32, 0, r.schema.RowCount)
	var allValid []bool
	if col.Nullable {
		allValid = make([]bool, 0, r.schema.RowCount)
	}

	for rg := 0; rg < int(r.schema.RowGroupCount); rg++ {
		chunk, err := r.ReadColumnChunk(column, rg)
		if err != nil {
			return nil, nil, err
		}

		var values []int32
		switch chunk.Encoding {
		case ColEncDelta:
			i64 := DecodeDeltaInt64(chunk.Data, int(chunk.Count))
			values = make([]int32, len(i64))
			for j, v := range i64 {
				values[j] = int32(v)
			}
		default:
			values = DecodePlainInt32(chunk.Data, int(chunk.Count))
		}

		result = append(result, values...)
		if col.Nullable {
			allValid = append(allValid, chunk.Valid...)
		}
	}

	return result, allValid, nil
}

// Close closes the underlying shard reader.
func (r *ColumnShardReader) Close() error {
	return r.shard.Close()
}
