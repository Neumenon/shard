package shard

import (
	"encoding/json"
	"time"
)

// ShardMetadata represents the standardized schema JSON structure.
type ShardMetadata struct {
	// Schema identification
	SchemaVersion string `json:"schema_version,omitempty"` // e.g., "shard-v2.1"
	SchemaURI     string `json:"schema_uri,omitempty"`     // URL to schema definition

	// Provenance
	CreatedAt time.Time `json:"created_at,omitempty"`
	SourceURI string    `json:"source_uri,omitempty"`
	Producer  string    `json:"producer,omitempty"`

	// Shard-level metadata
	Description string         `json:"description,omitempty"`
	Tags        []string       `json:"tags,omitempty"`
	Extra       map[string]any `json:"extra,omitempty"`

	// Per-entry metadata (keyed by entry name)
	EntryMetadata map[string]*EntryMeta `json:"entry_metadata,omitempty"`
}

// EntryMeta holds per-entry metadata.
type EntryMeta struct {
	ContentType string         `json:"content_type,omitempty"` // MIME-like type
	Tags        []string       `json:"tags,omitempty"`
	Description string         `json:"description,omitempty"`
	Extra       map[string]any `json:"extra,omitempty"`
}

// NewShardMetadata creates a new metadata instance with defaults.
func NewShardMetadata() *ShardMetadata {
	return &ShardMetadata{
		SchemaVersion: "shard-v2.1",
		CreatedAt:     time.Now().UTC(),
		EntryMetadata: make(map[string]*EntryMeta),
	}
}

// SetEntryMeta sets metadata for an entry.
func (m *ShardMetadata) SetEntryMeta(name string, meta *EntryMeta) {
	if m.EntryMetadata == nil {
		m.EntryMetadata = make(map[string]*EntryMeta)
	}
	m.EntryMetadata[name] = meta
}

// GetEntryMeta gets metadata for an entry.
func (m *ShardMetadata) GetEntryMeta(name string) *EntryMeta {
	if m.EntryMetadata == nil {
		return nil
	}
	return m.EntryMetadata[name]
}

// AddTag adds a shard-level tag.
func (m *ShardMetadata) AddTag(tag string) {
	for _, t := range m.Tags {
		if t == tag {
			return
		}
	}
	m.Tags = append(m.Tags, tag)
}

// Marshal serializes to JSON.
func (m *ShardMetadata) Marshal() ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// Unmarshal deserializes from JSON.
func (m *ShardMetadata) Unmarshal(data []byte) error {
	return json.Unmarshal(data, m)
}

// ParseShardMetadata parses metadata from JSON bytes.
func ParseShardMetadata(data []byte) (*ShardMetadata, error) {
	m := &ShardMetadata{}
	if err := m.Unmarshal(data); err != nil {
		return nil, err
	}
	return m, nil
}
