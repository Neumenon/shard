package shard

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWShardRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.wshard")

	chunkIdx := 0
	totalChunks := 3

	original := &WShardEpisode{
		ID:      "ep-001",
		EnvID:   "cartpole-v1",
		LengthT: 4,
		Timebase: WShardTimebase{
			Type:   "ticks",
			TickHz: 60.0,
		},
		Observations: map[string]*WShardChannel{
			"rgb": {
				Name:     "rgb",
				DType:    "uint8",
				Shape:    []int{84, 84, 3},
				Data:     make([]byte, 4*84*84*3), // 4 timesteps
				Modality: "visual",
			},
			"velocity": {
				Name:  "velocity",
				DType: "float32",
				Shape: []int{3},
				Data:  make([]byte, 4*3*4), // 4 timesteps * 3 floats * 4 bytes
			},
		},
		Actions: map[string]*WShardChannel{
			"discrete": {
				Name:  "discrete",
				DType: "int32",
				Shape: []int{1},
				Data:  make([]byte, 4*1*4), // 4 timesteps * 1 int * 4 bytes
			},
		},
		Rewards:       []float32{1.0, 0.5, -0.1, 2.0},
		Terminations:  []bool{false, false, false, true},
		ChunkIndex:    &chunkIdx,
		TotalChunks:   &totalChunks,
		TimestepRange: [2]int{0, 3},
		Metadata: map[string]any{
			"seed": float64(42), // JSON numbers decode as float64
		},
	}

	// Fill observation data with non-zero pattern so we can verify
	for i := range original.Observations["rgb"].Data {
		original.Observations["rgb"].Data[i] = byte(i % 256)
	}
	for i := range original.Observations["velocity"].Data {
		original.Observations["velocity"].Data[i] = byte((i * 7) % 256)
	}
	for i := range original.Actions["discrete"].Data {
		original.Actions["discrete"].Data[i] = byte((i * 13) % 256)
	}

	// Write
	if err := CreateWShard(path, original); err != nil {
		t.Fatalf("CreateWShard: %v", err)
	}

	// Verify file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("written file is empty")
	}

	// Read back
	got, err := OpenWShard(path)
	if err != nil {
		t.Fatalf("OpenWShard: %v", err)
	}

	// Verify scalar fields
	if got.ID != original.ID {
		t.Errorf("ID: got %q, want %q", got.ID, original.ID)
	}
	if got.EnvID != original.EnvID {
		t.Errorf("EnvID: got %q, want %q", got.EnvID, original.EnvID)
	}
	if got.LengthT != original.LengthT {
		t.Errorf("LengthT: got %d, want %d", got.LengthT, original.LengthT)
	}
	if got.Timebase.Type != original.Timebase.Type {
		t.Errorf("Timebase.Type: got %q, want %q", got.Timebase.Type, original.Timebase.Type)
	}
	if got.Timebase.TickHz != original.Timebase.TickHz {
		t.Errorf("Timebase.TickHz: got %f, want %f", got.Timebase.TickHz, original.Timebase.TickHz)
	}
	if got.TimestepRange != original.TimestepRange {
		t.Errorf("TimestepRange: got %v, want %v", got.TimestepRange, original.TimestepRange)
	}

	// Chunk fields
	if got.ChunkIndex == nil {
		t.Fatal("ChunkIndex is nil, want non-nil")
	}
	if *got.ChunkIndex != *original.ChunkIndex {
		t.Errorf("ChunkIndex: got %d, want %d", *got.ChunkIndex, *original.ChunkIndex)
	}
	if got.TotalChunks == nil {
		t.Fatal("TotalChunks is nil, want non-nil")
	}
	if *got.TotalChunks != *original.TotalChunks {
		t.Errorf("TotalChunks: got %d, want %d", *got.TotalChunks, *original.TotalChunks)
	}

	// Metadata
	if !reflect.DeepEqual(got.Metadata, original.Metadata) {
		t.Errorf("Metadata: got %v, want %v", got.Metadata, original.Metadata)
	}

	// Rewards
	if !reflect.DeepEqual(got.Rewards, original.Rewards) {
		t.Errorf("Rewards: got %v, want %v", got.Rewards, original.Rewards)
	}

	// Terminations
	if !reflect.DeepEqual(got.Terminations, original.Terminations) {
		t.Errorf("Terminations: got %v, want %v", got.Terminations, original.Terminations)
	}

	// Observations
	if len(got.Observations) != len(original.Observations) {
		t.Fatalf("Observations count: got %d, want %d", len(got.Observations), len(original.Observations))
	}
	for name, origCh := range original.Observations {
		gotCh, ok := got.Observations[name]
		if !ok {
			t.Fatalf("missing observation %q", name)
		}
		if gotCh.Name != origCh.Name {
			t.Errorf("obs %s Name: got %q, want %q", name, gotCh.Name, origCh.Name)
		}
		if gotCh.DType != origCh.DType {
			t.Errorf("obs %s DType: got %q, want %q", name, gotCh.DType, origCh.DType)
		}
		if !reflect.DeepEqual(gotCh.Shape, origCh.Shape) {
			t.Errorf("obs %s Shape: got %v, want %v", name, gotCh.Shape, origCh.Shape)
		}
		if gotCh.Modality != origCh.Modality {
			t.Errorf("obs %s Modality: got %q, want %q", name, gotCh.Modality, origCh.Modality)
		}
		if len(gotCh.Data) != len(origCh.Data) {
			t.Errorf("obs %s Data length: got %d, want %d", name, len(gotCh.Data), len(origCh.Data))
		} else {
			for i := range origCh.Data {
				if gotCh.Data[i] != origCh.Data[i] {
					t.Errorf("obs %s Data[%d]: got %d, want %d", name, i, gotCh.Data[i], origCh.Data[i])
					break
				}
			}
		}
	}

	// Actions
	if len(got.Actions) != len(original.Actions) {
		t.Fatalf("Actions count: got %d, want %d", len(got.Actions), len(original.Actions))
	}
	for name, origCh := range original.Actions {
		gotCh, ok := got.Actions[name]
		if !ok {
			t.Fatalf("missing action %q", name)
		}
		if gotCh.Name != origCh.Name {
			t.Errorf("act %s Name: got %q, want %q", name, gotCh.Name, origCh.Name)
		}
		if gotCh.DType != origCh.DType {
			t.Errorf("act %s DType: got %q, want %q", name, gotCh.DType, origCh.DType)
		}
		if !reflect.DeepEqual(gotCh.Shape, origCh.Shape) {
			t.Errorf("act %s Shape: got %v, want %v", name, gotCh.Shape, origCh.Shape)
		}
		if len(gotCh.Data) != len(origCh.Data) {
			t.Errorf("act %s Data length: got %d, want %d", name, len(gotCh.Data), len(origCh.Data))
		} else {
			for i := range origCh.Data {
				if gotCh.Data[i] != origCh.Data[i] {
					t.Errorf("act %s Data[%d]: got %d, want %d", name, i, gotCh.Data[i], origCh.Data[i])
					break
				}
			}
		}
	}
}

func TestDTypeSizeBytes(t *testing.T) {
	cases := []struct {
		dtype string
		want  int
	}{
		{"bool", 1},
		{"uint8", 1},
		{"int8", 1},
		{"uint16", 2},
		{"int16", 2},
		{"float16", 2},
		{"bfloat16", 2},
		{"uint32", 4},
		{"int32", 4},
		{"float32", 4},
		{"uint64", 8},
		{"int64", 8},
		{"float64", 8},
	}

	for _, tc := range cases {
		got := dtypeSizeBytes(tc.dtype)
		if got != tc.want {
			t.Errorf("dtypeSizeBytes(%q) = %d, want %d", tc.dtype, got, tc.want)
		}
	}

	// Unknown dtype should return 0
	if got := dtypeSizeBytes("complex128"); got != 0 {
		t.Errorf("dtypeSizeBytes(\"complex128\") = %d, want 0", got)
	}
}
