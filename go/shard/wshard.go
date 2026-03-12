// wshard.go — Typed API for WShard (World-model episode) shard files.
//
// A WShard stores a single RL episode: observations, actions, rewards,
// terminations, plus metadata (env, timebase, channel definitions).
//
// Layout inside the shard (all entries are path-separated):
//
//	meta/wshard       — JSON: WShardMetaBlock (format version, timebase)
//	meta/episode      — JSON: WShardEpisodeMeta (id, env_id, length_t, ...)
//	meta/channels     — JSON: map[string]WShardChannelDef (name→dtype/shape/modality)
//	signal/<name>     — raw bytes for observation channel <name>
//	action/<name>     — raw bytes for action channel <name>
//	reward            — []float32 LE
//	done              — []uint8 (0/1 booleans)
package shard

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// ============================================================
// Public types
// ============================================================

// WShardEpisode is the top-level structure for a single episode.
type WShardEpisode struct {
	ID            string
	EnvID         string
	LengthT       int
	Timebase      WShardTimebase
	Observations  map[string]*WShardChannel
	Actions       map[string]*WShardChannel
	Rewards       []float32
	Terminations  []bool
	ChunkIndex    *int // nil if not chunked
	TotalChunks   *int
	TimestepRange [2]int
	Metadata      map[string]any
}

// WShardChannel holds a single named signal channel.
type WShardChannel struct {
	Name     string
	DType    string
	Shape    []int
	Data     []byte
	Modality string // empty if N/A
}

// WShardTimebase describes the episode's time axis.
type WShardTimebase struct {
	Type   string  // "ticks" or "timestamps_ns"
	TickHz float64 // meaningful only when Type == "ticks"
}

// ============================================================
// Internal JSON structs for meta serialization
// ============================================================

type wshardMetaBlock struct {
	Version  int              `json:"version"`
	Timebase wshardTimebaseJ  `json:"timebase"`
}

type wshardTimebaseJ struct {
	Type   string  `json:"type"`
	TickHz float64 `json:"tick_hz,omitempty"`
}

type wshardEpisodeMeta struct {
	ID            string         `json:"id"`
	EnvID         string         `json:"env_id"`
	LengthT       int            `json:"length_t"`
	ChunkIndex    *int           `json:"chunk_index,omitempty"`
	TotalChunks   *int           `json:"total_chunks,omitempty"`
	TimestepRange [2]int         `json:"timestep_range"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type wshardChannelDef struct {
	DType    string `json:"dtype"`
	Shape    []int  `json:"shape"`
	Modality string `json:"modality,omitempty"`
}

// ============================================================
// Writer
// ============================================================

// CreateWShard writes a WShardEpisode to a new shard file at path.
func CreateWShard(path string, ep *WShardEpisode) error {
	w, err := NewShardV2Writer(path, ShardRoleWShard)
	if err != nil {
		return fmt.Errorf("wshard: create writer: %w", err)
	}

	// --- meta/wshard ---
	metaBlock := wshardMetaBlock{
		Version: 1,
		Timebase: wshardTimebaseJ{
			Type:   ep.Timebase.Type,
			TickHz: ep.Timebase.TickHz,
		},
	}
	metaJSON, err := json.Marshal(metaBlock)
	if err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: marshal meta/wshard: %w", err)
	}
	if err := w.WriteEntryTyped("meta/wshard", metaJSON, ContentTypeJSON); err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: write meta/wshard: %w", err)
	}

	// --- meta/episode ---
	epMeta := wshardEpisodeMeta{
		ID:            ep.ID,
		EnvID:         ep.EnvID,
		LengthT:       ep.LengthT,
		ChunkIndex:    ep.ChunkIndex,
		TotalChunks:   ep.TotalChunks,
		TimestepRange: ep.TimestepRange,
		Metadata:      ep.Metadata,
	}
	epJSON, err := json.Marshal(epMeta)
	if err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: marshal meta/episode: %w", err)
	}
	if err := w.WriteEntryTyped("meta/episode", epJSON, ContentTypeJSON); err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: write meta/episode: %w", err)
	}

	// --- meta/channels ---
	chanDefs := make(map[string]wshardChannelDef)
	for name, ch := range ep.Observations {
		chanDefs["signal/"+name] = wshardChannelDef{
			DType:    ch.DType,
			Shape:    ch.Shape,
			Modality: ch.Modality,
		}
	}
	for name, ch := range ep.Actions {
		chanDefs["action/"+name] = wshardChannelDef{
			DType:    ch.DType,
			Shape:    ch.Shape,
			Modality: ch.Modality,
		}
	}
	chanJSON, err := json.Marshal(chanDefs)
	if err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: marshal meta/channels: %w", err)
	}
	if err := w.WriteEntryTyped("meta/channels", chanJSON, ContentTypeJSON); err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: write meta/channels: %w", err)
	}

	// --- signal/* ---
	for name, ch := range ep.Observations {
		if err := w.WriteEntry("signal/"+name, ch.Data); err != nil {
			w.Close()
			os.Remove(path)
			return fmt.Errorf("wshard: write signal/%s: %w", name, err)
		}
	}

	// --- action/* ---
	for name, ch := range ep.Actions {
		if err := w.WriteEntry("action/"+name, ch.Data); err != nil {
			w.Close()
			os.Remove(path)
			return fmt.Errorf("wshard: write action/%s: %w", name, err)
		}
	}

	// --- reward ---
	rewardBuf := make([]byte, len(ep.Rewards)*4)
	for i, r := range ep.Rewards {
		binary.LittleEndian.PutUint32(rewardBuf[i*4:], math.Float32bits(r))
	}
	if err := w.WriteEntry("reward", rewardBuf); err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: write reward: %w", err)
	}

	// --- done ---
	doneBuf := make([]byte, len(ep.Terminations))
	for i, t := range ep.Terminations {
		if t {
			doneBuf[i] = 1
		}
	}
	if err := w.WriteEntry("done", doneBuf); err != nil {
		w.Close()
		os.Remove(path)
		return fmt.Errorf("wshard: write done: %w", err)
	}

	return w.Close()
}

// ============================================================
// Reader
// ============================================================

// OpenWShard reads a WShard file and returns the decoded episode.
func OpenWShard(path string) (*WShardEpisode, error) {
	r, err := OpenShardV2(path)
	if err != nil {
		return nil, fmt.Errorf("wshard: open: %w", err)
	}
	defer r.Close()

	// Verify role
	if r.Header().Role != ShardRoleWShard {
		return nil, fmt.Errorf("wshard: expected role WShard (0x%02x), got %s (0x%02x)",
			uint8(ShardRoleWShard), r.Header().Role, uint8(r.Header().Role))
	}

	// --- meta/wshard ---
	metaRaw, err := r.ReadEntryByName("meta/wshard")
	if err != nil {
		return nil, fmt.Errorf("wshard: read meta/wshard: %w", err)
	}
	var metaBlock wshardMetaBlock
	if err := json.Unmarshal(metaRaw, &metaBlock); err != nil {
		return nil, fmt.Errorf("wshard: parse meta/wshard: %w", err)
	}

	// --- meta/episode ---
	epRaw, err := r.ReadEntryByName("meta/episode")
	if err != nil {
		return nil, fmt.Errorf("wshard: read meta/episode: %w", err)
	}
	var epMeta wshardEpisodeMeta
	if err := json.Unmarshal(epRaw, &epMeta); err != nil {
		return nil, fmt.Errorf("wshard: parse meta/episode: %w", err)
	}

	// --- meta/channels ---
	chanRaw, err := r.ReadEntryByName("meta/channels")
	if err != nil {
		return nil, fmt.Errorf("wshard: read meta/channels: %w", err)
	}
	var chanDefs map[string]wshardChannelDef
	if err := json.Unmarshal(chanRaw, &chanDefs); err != nil {
		return nil, fmt.Errorf("wshard: parse meta/channels: %w", err)
	}

	// Build observations and actions from channel defs + data
	observations := make(map[string]*WShardChannel)
	actions := make(map[string]*WShardChannel)

	for fullName, def := range chanDefs {
		data, err := r.ReadEntryByName(fullName)
		if err != nil {
			return nil, fmt.Errorf("wshard: read %s: %w", fullName, err)
		}

		parts := SplitPath(fullName)
		if len(parts) != 2 {
			return nil, fmt.Errorf("wshard: unexpected channel path %q", fullName)
		}
		prefix, name := parts[0], parts[1]

		ch := &WShardChannel{
			Name:     name,
			DType:    def.DType,
			Shape:    def.Shape,
			Data:     data,
			Modality: def.Modality,
		}

		switch prefix {
		case "signal":
			observations[name] = ch
		case "action":
			actions[name] = ch
		default:
			return nil, fmt.Errorf("wshard: unknown channel prefix %q", prefix)
		}
	}

	// --- reward ---
	rewardRaw, err := r.ReadEntryByName("reward")
	if err != nil {
		return nil, fmt.Errorf("wshard: read reward: %w", err)
	}
	if len(rewardRaw)%4 != 0 {
		return nil, fmt.Errorf("wshard: reward data length %d not multiple of 4", len(rewardRaw))
	}
	rewards := make([]float32, len(rewardRaw)/4)
	for i := range rewards {
		rewards[i] = math.Float32frombits(binary.LittleEndian.Uint32(rewardRaw[i*4:]))
	}

	// --- done ---
	doneRaw, err := r.ReadEntryByName("done")
	if err != nil {
		return nil, fmt.Errorf("wshard: read done: %w", err)
	}
	terminations := make([]bool, len(doneRaw))
	for i, b := range doneRaw {
		terminations[i] = b != 0
	}

	return &WShardEpisode{
		ID:            epMeta.ID,
		EnvID:         epMeta.EnvID,
		LengthT:       epMeta.LengthT,
		Timebase: WShardTimebase{
			Type:   metaBlock.Timebase.Type,
			TickHz: metaBlock.Timebase.TickHz,
		},
		Observations:  observations,
		Actions:       actions,
		Rewards:       rewards,
		Terminations:  terminations,
		ChunkIndex:    epMeta.ChunkIndex,
		TotalChunks:   epMeta.TotalChunks,
		TimestepRange: epMeta.TimestepRange,
		Metadata:      epMeta.Metadata,
	}, nil
}

// ============================================================
// Helpers
// ============================================================

// dtypeSizeBytes returns the byte size for a dtype string.
// Returns 0 for unknown dtypes.
func dtypeSizeBytes(dtype string) int {
	switch dtype {
	case "bool":
		return 1
	case "uint8", "int8":
		return 1
	case "uint16", "int16", "float16", "bfloat16":
		return 2
	case "uint32", "int32", "float32":
		return 4
	case "uint64", "int64", "float64":
		return 8
	default:
		return 0
	}
}
