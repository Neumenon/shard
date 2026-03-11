package shard

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Suite 7: Performance Cliff Detection for shard format.

func TestPerfCliff_ManyEntries(t *testing.T) {
	sizes := []int{100, 1000, 5000}
	var lastDuration time.Duration

	for _, size := range sizes {
		t.Run(fmt.Sprintf("entries_%d", size), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test.shard")

			w, err := NewShardV2Writer(path, ShardRoleMoSH)
			if err != nil {
				t.Fatalf("NewShardV2Writer failed: %v", err)
			}

			for i := 0; i < size; i++ {
				name := fmt.Sprintf("entry_%05d", i)
				data := make([]byte, 64)
				if err := w.WriteEntry(name, data); err != nil {
					t.Fatalf("WriteEntry %d failed: %v", i, err)
				}
			}

			start := time.Now()
			if err := w.Close(); err != nil {
				t.Fatalf("Close failed: %v", err)
			}

			r, err := OpenShardV2(path)
			if err != nil {
				t.Fatalf("OpenShardV2 failed: %v", err)
			}
			defer r.Close()

			if r.EntryCount() != size {
				t.Errorf("entry count mismatch: got %d, want %d", r.EntryCount(), size)
			}

			for i := 0; i < r.EntryCount(); i++ {
				_, err := r.ReadEntry(i)
				if err != nil {
					t.Fatalf("ReadEntry %d failed: %v", i, err)
				}
			}
			dur := time.Since(start)

			if lastDuration > 0 && dur > lastDuration*20 {
				t.Errorf("possible quadratic blowup at size=%d: took %v, previous %v",
					size, dur, lastDuration)
			}
			lastDuration = dur
		})
	}
}

func TestPerfCliff_LargeEntry(t *testing.T) {
	sizes := []int{1024, 1024 * 1024, 10 * 1024 * 1024}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test.shard")

			w, err := NewShardV2Writer(path, ShardRoleMoSH)
			if err != nil {
				t.Fatalf("NewShardV2Writer failed: %v", err)
			}

			data := make([]byte, size)
			for i := range data {
				data[i] = byte(i & 0xFF)
			}
			if err := w.WriteEntry("large", data); err != nil {
				t.Fatalf("WriteEntry failed: %v", err)
			}

			start := time.Now()
			if err := w.Close(); err != nil {
				t.Fatalf("Close failed: %v", err)
			}

			r, err := OpenShardV2(path)
			if err != nil {
				t.Fatalf("OpenShardV2 failed: %v", err)
			}
			defer r.Close()

			entry, err := r.ReadEntry(0)
			if err != nil {
				t.Fatalf("ReadEntry failed: %v", err)
			}
			if len(entry) != size {
				t.Errorf("entry size mismatch: got %d, want %d", len(entry), size)
			}
			dur := time.Since(start)

			if dur > 10*time.Second {
				t.Errorf("large entry size=%d took %v, exceeds 10s threshold", size, dur)
			}

			os.Remove(path)
		})
	}
}
