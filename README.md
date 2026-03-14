# Shard

Binary container format for ML data. 64-byte header, O(1) name lookup, per-entry compression, CRC32C checksums, five native language implementations.

```
any_file.shard
├── Header (64 bytes)     Magic "SHRD", version, role byte, alignment, compression default
├── Index  (N × 48 bytes) xxHash64 name → offset, size, compression flags, CRC32C
├── String Table           Entry names (UTF-8, null-terminated)
└── Data Blocks            Aligned to 16/32/64 bytes, independently compressed
```

## Profiles

One container format, one reader, one toolchain. The **role byte** in the header tells you what's inside.

| Role | Profile | Extension | What It Stores |
|------|---------|-----------|----------------|
| 0x01 | [ModelShard](#modelshard) | `.mosh` | Model weights keyed by layer name |
| 0x02 | [SampleShard](#sampleshard) | `.smpl` | Training samples keyed by sample ID |
| 0x03 | [GemmShard](#gemmshard) | `.gemm` | Pre-packed BLIS matrix panels for distributed GEMM |
| 0x04 | [Manifest](#manifest) | `.manifest` | Multi-file coordination — chunked episodes, split models |
| 0x05 | [WShard](#wshard) | `.wshard` | World model episodes — signal/omen/residual traces |
| 0x06 | [GraphShard](#graphshard) | `.gshard` | GNN datasets — node/edge tables, features, splits |
| 0x07 | [TraceShard](#traceshard) | `.tshard` | Model introspection — MoE routing, attention patterns |
| 0x08 | [ColumnShard](#columnshard) | `.cshard` | Columnar tabular data with row group stats |

## Languages

Five independent, idiomatic implementations. All produce byte-identical output verified by golden file tests.

| Language | Location | LOC | Status |
|----------|----------|-----|--------|
| Go | [`go/`](go/) | ~1,575 | Production |
| Rust | [`rs/`](rs/) | ~1,949 | Production |
| Python | [`py/`](py/) | ~1,096 | Production |
| C | [`c/`](c/) | ~1,832 | Production |
| TypeScript | [`ts/`](ts/) | ~1,455 | Production |

Total: **840+ tests** across all five languages.

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

// Write
w, _ := shard.NewShardStreamWriter("model.mosh", shard.ShardOptions{
    Role:        shard.RoleModelShard,
    Alignment:   64,
    Compression: shard.CompressionZstd,
    MaxEntries:  1000,
})
w.WriteEntry("layer.0.weight", weightBytes, shard.ContentTypeRaw)
w.Finalize()

// Read
r, _ := shard.OpenShard("model.mosh")
data, _ := r.ReadEntry("layer.0.weight")
```

## Quick Start (Python)

```python
from shard_v2 import ShardWriter, ShardReader

# Write
with ShardWriter("model.mosh", role=0x01, alignment=64) as w:
    w.write_entry("layer.0.weight", weight_bytes, compression="zstd")

# Read
reader = ShardReader("model.mosh")
data = reader.read_entry("layer.0.weight")
```

## Key Properties

**O(1) name lookup.** xxHash64 of the entry name maps directly to a 48-byte index entry. Finding one tensor in a 400-entry model file costs one hash + one 48-byte read.

**Per-entry compression.** Each entry independently chooses none/zstd/lz4. JSON configs compress 10:1. Quantized weights stay raw. The reader auto-detects from entry flags.

**CRC32C checksums.** Hardware-accelerated (SSE4.2/ARM CRC32) integrity verification on every entry. Checksum mismatch is a hard error, not a warning.

**GPU-aligned mmap.** Data blocks start at 32/64-byte boundaries. `mmap()` + pointer cast gives AVX-aligned arrays with zero copies. Direct DMA to GPU without intermediate buffers.

**Streaming writes.** Pre-reserve header space, write entries sequentially, finalize header at the end. Memory stays constant regardless of entry count.

**Security limits.** Four hard limits enforced on read: MAX_ENTRY_COUNT (10M), MAX_INDEX_SIZE (1GB), MAX_STRING_TABLE_SIZE (100MB), MAX_DECOMPRESS_SIZE (1GB). No code execution risk.

---

## Profile Details

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
- DeepData bridge for trajectory similarity search
- 13 data types including bf16

**Implementations:** Python ([`wshard/py/`](wshard/py/)), TypeScript ([`wshard/js/`](wshard/js/)), Go ([`go/shard/`](go/shard/)). 164 cross-language tests.

**World model targets:** DreamerV3/V4, V-JEPA 2, Genie 3, NVIDIA Cosmos, VERSES AXIOM. No standard trajectory format exists — WShard fills the gap with native support for paired observation/prediction data.

See [`wshard/README.md`](wshard/README.md) for full API docs. See [`wshard/docs/DEEP_DIVE.md`](wshard/docs/DEEP_DIVE.md) for the byte-level format spec.

---

### ModelShard

Model weights keyed by layer name. Quantization-aware (INT8/Q4_K/Q5_K/Q6_K with per-row scales), delta encoding (sparse, scale, LoRA A*B), content-addressed via SHA256.

```
llama2-7b.mosh
├── model.layers.0.self_attn.q_proj.weight  → [4096, 4096] q4_k
├── model.layers.0.self_attn.q_proj.scales  → [4096] float16
├── model.layers.0.self_attn.k_proj.weight  → [4096, 4096] q4_k
└── ...
```

**Key capabilities:**
- mmap + zero-copy to GPU (64-byte aligned)
- Per-entry AES-256-GCM encryption with HKDF key derivation
- LoRA adapters stored as first-class deltas (not full layers)
- Streaming index for large models over WAN/object stores

**Implementation:** Go ([`go/shard/`](go/shard/)), Python ([`mosh/`](mosh/) — in progress).

---

### SampleShard

Pre-tokenized training samples keyed by sample ID. PyTorch DataLoader integration. HuggingFace dataset import.

```
train.smpl
├── sample/000000  → tokenized record
├── sample/000001  → tokenized record
└── ...            → O(1) random access to any sample
```

**Implementations:** Python ([`sampleshard/py/`](sampleshard/py/)), TypeScript ([`sampleshard/js/`](sampleshard/js/)).

---

### GemmShard

Pre-packed BLIS matrix panels for distributed GEMM. Tiles keyed by coordinate (row_block, col_block). Content-addressed via SHA256 for worker deduplication and caching.

**Implementation:** Go ([`go/shard/`](go/shard/)).

---

### Manifest

Multi-file coordination. References to other shard files with range keys. Used for:
- Chunked WShard episodes (1000-timestep chunks with continuity validation)
- Split model weights across multiple `.mosh` files
- Dataset indices spanning many `.smpl` files

**Implementation:** All 5 languages (part of core Shard).

---

### GraphShard

GNN datasets: node tables, edge tables, feature tensors, train/val/test splits. CSR adjacency stored as native entries. Scene graphs for spatial intelligence (World Labs, Luma), factor graphs for active inference (VERSES AXIOM, pymdp).

**Implementation:** Go ([`go/shard/`](go/shard/)).

---

### TraceShard

Sparse model introspection data: MoE expert routing decisions, attention patterns, activation constraints. Sequence-keyed events for offline analysis.

**Status:** Defined (spec complete, implementation pending).

---

### ColumnShard

Columnar tabular data with row group statistics and predicate pushdown. Column chunks with per-column encoding headers and min/max/null stats.

**Implementation:** Go ([`go/shard/`](go/shard/)).

---

## Directory Structure

```
shard/
├── go/              Go core + ColumnShard + WShard + ModelShard + GemmShard
├── rs/              Rust core
├── py/              Python core
├── c/               C core
├── ts/              TypeScript core
├── wshard/          WShard profile (Python + TypeScript + golden tests)
│   ├── py/          Python wshard package
│   ├── js/          TypeScript @wshard/core package
│   ├── golden/      Cross-language golden test fixtures
│   └── docs/        DEEP_DIVE.md, MARKETING_BRIEF.md
├── sampleshard/     SampleShard profile (Python + TypeScript)
│   ├── py/
│   └── js/
├── mosh/            ModelShard profile (in progress)
├── ucodec/          Golden test data + safety test fixtures
└── explainer.html   Interactive format explainer
```

## Cross-Language Parity

All implementations agree on:
- **CRC32C polynomial:** Castagnoli (`0x82F63B78`)
- **Name hashing:** xxHash64 (seed 0)
- **Alignment:** Configurable 0/16/32/64 bytes
- **Compression threshold:** Only compress entries > 256 bytes, only keep if ratio < 0.9
- **Security limits:** Identical across all 5 languages

Verified by 6 golden files and a shared manifest (`ucodec/testdata/golden_manifest.json`).

```
CRC32C("hello")                = 0x9a71bb4c
xxHash64("signal/obs")         = 0x86f8c8413116a0ae
xxHash64("meta/manifest")      = 0x9a191dcd325813d3
```

## Docs

| Document | Description |
|----------|-------------|
| [WShard Deep Dive](wshard/docs/DEEP_DIVE.md) | Byte-level format spec, cross-language interop, market position |
| [WShard Marketing Brief](wshard/docs/MARKETING_BRIEF.md) | Positioning, messaging, competitive landscape |
| [Filetype Taxonomy](../docs/specs/FILETYPES.md) | Complete profile registry, decision tree |
| [World Model Market Guide](../docs/market/WORLD_MODEL_MARKET_GUIDE.md) | WShard across 5 world model categories |
| [ModelShard Competitive Analysis](../docs/market/MODELSHARD_COMPETITIVE_ANALYSIS.md) | Weight format comparison matrix |
| [Shard Market Analysis](../docs/shard-market/00-market-analysis.md) | SAM sizing, cost impact by scale |

## License

MIT
