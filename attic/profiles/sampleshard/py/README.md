# SampleShard Python — Parked

> **This profile is parked.** It is not built or tested by default.
> See [`../README.md`](../README.md) for the parked-profile notice.

Python implementation of the SampleShard format for storing training samples.

## Installation

```bash
pip install sampleshard

# With optional compression support
pip install sampleshard[compression]

# With optional xxhash support (faster hashing)
pip install sampleshard[hash]

# All optional dependencies
pip install sampleshard[all]
```

## Quick Start

```python
from sampleshard import SampleShardWriter, SampleShardReader

# Writing samples
with SampleShardWriter("train.smpl") as w:
    w.add_sample(1, {"input": [1, 2, 3], "label": 0})
    w.add_sample(2, {"input": [4, 5, 6], "label": 1})
    w.add_sample(3, {"input": [7, 8, 9], "label": 2})

# Reading samples
with SampleShardReader("train.smpl") as r:
    # Get sample count
    print(f"Total samples: {r.sample_count()}")
    
    # Random access by ID
    sample = r.get_sample(1)
    print(sample)  # {"input": [1, 2, 3], "label": 0}
    
    # Check if sample exists
    if r.has_sample(2):
        print("Sample 2 exists!")
    
    # Iterate all samples
    for sample_id, sample in r:
        print(f"Sample {sample_id}: {sample}")
    
    # Batch access
    batch = r.get_batch([1, 2, 3])
    range_batch = r.get_batch_by_range(0, 10)
```

## Features

- **Fast random access** by sample ID (O(1) lookup)
- **Deterministic iteration** order (ascending by sample ID)
- **Duplicate rejection**: Writers reject duplicate sample IDs
- **Metadata-safe**: Reserved entries (starting with `__`) excluded from sample counts
- **Memory-mapped access** for zero-copy reads
- **Optional compression** (zstd, lz4) with correct CRC on decompressed data
- **CRC32C checksums** for data integrity (requires `crc32c` library)
- **Structured metadata** at schema section with dataset profiles

## Dataset Metadata

SampleShard embeds structured metadata alongside data:

```python
from sampleshard import SampleShardWriter, SampleShardReader

# Writing with metadata
with SampleShardWriter("train.smpl") as w:
    w.set_sample_profile(
        dataset_name="imagenet-train",
        sample_id_type="uint64",
    )
    w.add_sample(1, {"input": [1, 2, 3], "label": 0})
    w.add_sample(2, {"input": [4, 5, 6], "label": 1})
    # sample_count auto-populated on close

# Reading metadata without decoding samples
with SampleShardReader("train.smpl") as r:
    profile = r.sample_profile()
    print(profile.dataset_name)    # "imagenet-train"
    print(profile.sample_count)    # 2
    print(profile.sample_id_type)  # "uint64"

    meta = r.read_metadata()
    print(meta.schema_version)     # "shard-v2.1"
    print(meta.profile)            # "sampleshard.v1"
```

Per-entry metadata can track codec provenance, shapes, stats, and content hashes — enabling dataset catalogs that query thousands of shards without opening data sections.

## File Format

SampleShard uses the `.smpl` extension and the Shard binary format:

- 64-byte header with magic bytes `SHRD`
- Role byte = 0x02 (Sample)
- 48-byte index entries with xxHash64 name hashes
- Cowrie-encoded sample data (auto-detected on read)
- CRC32C checksums on decompressed data per entry
- Optional JSON metadata at `schema_offset` with profile and per-entry metadata

## Interoperability

SampleShard files are byte-identical across all implementations:
- **Go**: `go/shard` — open via `shard.OpenShard()` with role 0x02
- **Python**: `sampleshard.SampleShardReader()`
- **TypeScript**: `@sampleshard/core`

Cross-language guarantees:
- Identical iteration order (ascending sample ID)
- Identical CRC32C verification (on decompressed data)
- Identical compression flag interpretation (Go bit-flag encoding)
- Identical `ListChildren` output (bare path components)
- Identical metadata serialization (snake_case JSON keys)

## License

MIT
