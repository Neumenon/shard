package shard

import "encoding/json"

// ColumnShardSchema is the schema section for a ColumnShard (.cshard) file.
// It is stored as a JSON-encoded Shard entry named "__schema__".
type ColumnShardSchema struct {
	Version       int                   `json:"version"`        // Schema version (1)
	RowCount      uint64                `json:"row_count"`      // Total rows across all row groups
	RowGroupSize  uint32                `json:"row_group_size"` // Target rows per group (last may be smaller)
	RowGroupCount uint32                `json:"row_group_count"`
	Columns       []*ColumnShardColDef  `json:"columns"`
}

// ColumnShardColDef defines a single column in the schema.
type ColumnShardColDef struct {
	Name     string              `json:"name"`               // Column path, e.g. "age" or "user.address.city"
	Type     ColType             `json:"type"`               // Column type code
	Nullable bool                `json:"nullable"`
	Encoding byte                `json:"encoding"`           // Default encoding for this column
	Stats    []*ColumnChunkStats `json:"stats"`              // One per row group
}

// ColumnChunkStats holds per-row-group per-column statistics.
type ColumnChunkStats struct {
	Count         uint32  `json:"count"`                    // Rows in this chunk
	NullCount     uint32  `json:"null_count"`
	Min           any     `json:"min,omitempty"`            // Typed min value
	Max           any     `json:"max,omitempty"`            // Typed max value
	DistinctCount uint32  `json:"distinct_count,omitempty"` // 0 if unknown
}

// NewColumnShardSchema creates a new schema with defaults.
func NewColumnShardSchema(rowGroupSize uint32) *ColumnShardSchema {
	if rowGroupSize == 0 {
		rowGroupSize = 65536
	}
	return &ColumnShardSchema{
		Version:      1,
		RowGroupSize: rowGroupSize,
		Columns:      make([]*ColumnShardColDef, 0),
	}
}

// AddColumn adds a column definition.
func (s *ColumnShardSchema) AddColumn(name string, typ ColType, nullable bool, encoding byte) *ColumnShardColDef {
	col := &ColumnShardColDef{
		Name:     name,
		Type:     typ,
		Nullable: nullable,
		Encoding: encoding,
		Stats:    make([]*ColumnChunkStats, 0),
	}
	s.Columns = append(s.Columns, col)
	return col
}

// FindColumn finds a column by name.
func (s *ColumnShardSchema) FindColumn(name string) *ColumnShardColDef {
	for _, c := range s.Columns {
		if c.Name == name {
			return c
		}
	}
	return nil
}

// Marshal serializes the schema to JSON.
func (s *ColumnShardSchema) Marshal() ([]byte, error) {
	return json.Marshal(s)
}

// UnmarshalColumnShardSchema parses a schema from JSON bytes.
func UnmarshalColumnShardSchema(data []byte) (*ColumnShardSchema, error) {
	s := &ColumnShardSchema{}
	if err := json.Unmarshal(data, s); err != nil {
		return nil, err
	}
	return s, nil
}

// EntryName returns the Shard entry name for a column chunk.
// Format: "{row_group}:{column_path}"
func ColumnChunkEntryName(rowGroup int, column string) string {
	// Use simple integer formatting — no zero-padding needed for hash lookup
	buf := make([]byte, 0, 20+len(column))
	buf = appendInt(buf, rowGroup)
	buf = append(buf, ':')
	buf = append(buf, column...)
	return string(buf)
}

// appendInt appends an integer as ASCII digits.
func appendInt(buf []byte, v int) []byte {
	if v == 0 {
		return append(buf, '0')
	}
	if v < 0 {
		buf = append(buf, '-')
		v = -v
	}
	// Write digits in reverse, then flip
	start := len(buf)
	for v > 0 {
		buf = append(buf, byte('0'+v%10))
		v /= 10
	}
	// Reverse the digits
	for i, j := start, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	return buf
}

// SchemaEntryName is the reserved entry name for the schema.
const SchemaEntryName = "__schema__"
