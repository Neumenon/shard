# SampleShard Python

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
- **Deterministic iteration** order
- **Metadata-safe**: Reserved entries (starting with `__`) excluded from sample counts
- **Memory-mapped access** for zero-copy reads
- **Optional compression** (zstd, lz4)
- **CRC32C checksums** for data integrity

## File Format

SampleShard uses the `.smpl` extension and the Shard v2 binary format:

- 64-byte header with magic bytes `SHRD`
- Role byte = 0x02 (Sample)
- 48-byte index entries with xxHash64 name hashes
- JSON-encoded sample data
- CRC32C checksums per entry

## Interoperability

SampleShard files created with Python can be read by:
- Go: `agentscope/cowrie/ucodec.OpenSampleShard()`
- TypeScript: `@sampleshard/core`

## License

MIT
