# Shard

Binary container format for episodic data. 64-byte header, O(1) name lookup, per-entry compression, CRC32C checksums, five native language implementations.

```
any_file.shard
├── Header (64 bytes)     Magic "SHRD", version, role byte, alignment, compression default
├── Index  (N × 48 bytes) xxHash64 name → offset, size, compression flags, CRC32C
├── String Table           Entry names (UTF-8, null-terminated)
└── Data Blocks            Aligned to 16/32/64 bytes, independently compressed
```

Shard is the container. **Profiles** — concrete schemas for what the entries mean — live downstream of core. The role byte identifies the profile; readers that understand a given role validate it, others treat the file as a generic named-entry archive.

The flagship downstream profile is [**WShard**](wshard/) — episodic world-model data (signal / action / omen / uncert / residual lanes).

## Roles

| Role | Profile | Status |
|------|---------|--------|
| 0x04 | Manifest | Core — multi-file coordination |
| 0x05 | [WShard](wshard/) | Active downstream profile |
| 0x01 | ModelShard (`.mosh`) | Parked — see `attic/profiles/mosh/` |
| 0x02 | SampleShard (`.smpl`) | Parked — see `attic/profiles/sampleshard/` |
| 0x08 | ColumnShard (`.cshard`) | Parked — see `attic/profiles/columnshard/` |

Parked profiles still decode (the container reader does not care about role semantics) but are not part of the lead surface or default build.

## Languages

Five independent, idiomatic implementations of the core container. All produce byte-identical output verified by golden file tests.

| Language | Location | Status |
|----------|----------|--------|
| Go | [`go/`](go/) | Production |
| Rust | [`rs/`](rs/) | Production |
| Python | [`py/`](py/) | Production |
| C | [`c/`](c/) | Production |
| TypeScript | [`ts/`](ts/) | Production |

## Install

```bash
# Go
import "github.com/Neumenon/shard/go/shard"

# Rust
[dependencies]
shard = { git = "https://github.com/Neumenon/shard.git", path = "rs" }

# Python
cd py && pip install -e .

# TypeScript
cd ts && npm install

# C
cd c && make
```

## Quick Start (Go)

```go
import "github.com/Neumenon/shard/go/shard"

// Write a generic named-entry archive
w, _ := shard.NewShardStreamWriter("data.shard", shard.ShardOptions{
    Alignment:   64,
    Compression: shard.CompressionZstd,
    MaxEntries:  1000,
})
w.WriteEntry("entry/0", payloadBytes, shard.ContentTypeRaw)
w.Finalize()

// Read
r, _ := shard.OpenShard("data.shard")
data, _ := r.ReadEntry("entry/0")
```

## Quick Start (Python)

```python
from shard_format import ShardWriter, ShardReader

# Write
with ShardWriter("data.shard", alignment=64) as w:
    w.write_entry("entry/0", payload_bytes, compression="zstd")

# Read
reader = ShardReader("data.shard")
data = reader.read_entry("entry/0")
```

## Key Properties

**O(1) name lookup.** xxHash64 of the entry name maps directly to a 48-byte index entry. Finding one entry in a 400-entry file costs one hash + one 48-byte read.

**Per-entry compression.** Each entry independently chooses none/zstd/lz4. JSON configs compress 10:1. Pre-quantized payloads stay raw. The reader auto-detects from entry flags.

**CRC32C checksums.** Hardware-accelerated (SSE4.2/ARM CRC32) integrity verification on every entry. Checksum mismatch is a hard error, not a warning.

**Aligned mmap.** Data blocks start at 32/64-byte boundaries. `mmap()` + pointer cast gives AVX-aligned arrays with zero copies.

**Streaming writes.** Pre-reserve header space, write entries sequentially, finalize header at the end. Memory stays constant regardless of entry count.

**Security limits.** Four hard limits enforced on read: MAX_ENTRY_COUNT (10M), MAX_INDEX_SIZE (1GB), MAX_STRING_TABLE_SIZE (100MB), MAX_DECOMPRESS_SIZE (1GB). No code execution risk.

---

## Downstream profiles

### WShard

**The file format for world model episode data.** One file = one episode. Cross-language. Zero-copy reads.

```
episode.wshard
├── meta/episode     → {"episode_id": "ep_001", "env_id": "Manip-v2", "length_T": 500}
├── signal/rgb       → [500, 84, 84, 3] uint8     (zstd compressed)
├── signal/joint_pos → [500, 7] float32            (32-byte aligned)
├── action/ctrl      → [500, 7] float32
├── omen/rgb/dreamer → [500, 84, 84, 3] uint8      (model predictions)
├── uncert/rgb/std   → [500, 84, 84, 3] float32    (uncertainty estimates)
├── reward           → [500] float32
└── done             → [500] bool
```

Semantic lane prefixes give meaning to raw data:

| Prefix | Purpose |
|--------|---------|
| `signal/` | Ground truth observations |
| `action/` | Agent actions |
| `omen/` | Model predictions (dreamed trajectories) |
| `uncert/` | Uncertainty estimates (ensemble variance, entropy) |
| `residual/` | Compressed residuals (sign-2nd-diff encoding) |
| `meta/` | JSON metadata |
| `time/` | Timestamps |

**Key capabilities:**
- Crash-safe streaming writes via `.partial` file pattern
- Chunked episodes with manifest-tracked continuity
- Per-block compression (zstd video, raw scalars)
- Format conversion from DreamerV3 NPZ, Minari, D4RL
- 13 data types including bf16

**Implementations:** Python ([`wshard/py/`](wshard/py/)), TypeScript ([`wshard/js/`](wshard/js/)), Go ([`go/shard/`](go/shard/)).

See [`wshard/README.md`](wshard/README.md) for full API docs. See [`wshard/docs/DEEP_DIVE.md`](wshard/docs/DEEP_DIVE.md) for the byte-level format spec.

### Manifest

Multi-file coordination — references to other shard files with range keys. Chunked WShard episodes, split datasets, file groupings. Part of core (all 5 languages).

---

## Directory Structure

```
shard/
├── go/                Go core
├── rs/                Rust core
├── py/                Python core
├── c/                 C core
├── ts/                TypeScript core
├── wshard/            WShard profile (Python + TypeScript + golden tests)
│   ├── py/
│   ├── js/
│   ├── golden/        Cross-language golden test fixtures
│   └── docs/
├── attic/profiles/    Parked profiles (mosh, sampleshard, columnshard)
├── ucodec/            Golden test data + safety test fixtures
└── explainer.html     Interactive format explainer
```

## Cross-Language Parity

All implementations agree on:
- **CRC32C polynomial:** Castagnoli (`0x82F63B78`)
- **Name hashing:** xxHash64 (seed 0)
- **Alignment:** Configurable 0/16/32/64 bytes
- **Compression threshold:** Only compress entries > 256 bytes, only keep if ratio < 0.9
- **Security limits:** Identical across all 5 languages

Verified by golden files and a shared manifest (`testdata/golden_manifest.json`).

```
CRC32C("hello")                = 0x9a71bb4c
xxHash64("signal/obs")         = 0x86f8c8413116a0ae
xxHash64("meta/manifest")      = 0x9a191dcd325813d3
```

## Docs

| Document | Description |
|----------|-------------|
| [WShard Deep Dive](wshard/docs/DEEP_DIVE.md) | Byte-level format spec + cross-language interop |

## License

MIT
