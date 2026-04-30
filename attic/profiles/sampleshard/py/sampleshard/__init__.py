"""
SampleShard - Training sample storage format.

SampleShard stores training samples keyed by uint64 sample ID. Each sample is
a JSON-serializable value that can contain tensors, metadata, and labels.

File extension: .smpl

Key features:
- Fast random access by sample ID (O(1) lookup)
- Deterministic iteration (ascending sample ID order)
- Metadata-safe: reserved entries (__schema__, etc.) excluded from sample counts
- Memory-mapped access for zero-copy reads
- Optional compression (zstd, lz4)

Iteration order:
    Iteration is always in ascending sample ID order, regardless of write order.
    This ensures cross-language consistency (Go, Python, TypeScript).

Example:
    >>> from sampleshard import SampleShardWriter, SampleShardReader
    >>>
    >>> # Writing
    >>> with SampleShardWriter("train.smpl") as w:
    ...     w.add_sample(1, {"input": [1, 2, 3], "target": 0})
    ...     w.add_sample(2, {"input": [4, 5, 6], "target": 1})
    >>>
    >>> # Reading
    >>> with SampleShardReader("train.smpl") as r:
    ...     print(r.sample_count())  # 2
    ...     sample = r.get_sample(1)
    ...     for sample_id, sample in r:  # ascending ID order
    ...         print(sample_id, sample)
"""

from .reader import SampleShardReader
from .writer import SampleShardWriter
from .types import (
    ShardHeader,
    ShardRole,
    IndexEntry,
    ShardMetadata,
    EntryMeta,
    SampleProfile,
    ManifestProfile,
    ManifestFileRef,
    # Content types
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_TENSOR,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_COWRIE,
    CONTENT_TYPE_GLYPH,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_AUDIO,
    CONTENT_TYPE_VIDEO,
    CONTENT_TYPE_PROTO,
    CONTENT_TYPE_BLOB,
    CONTENT_TYPE_USER_BASE,
    SHARD_FLAG_HAS_SCHEMA,
    SHARD_FLAG_HAS_CONTENT_TYPES,
    PROFILE_SAMPLESHARD_V1,
    PROFILE_MANIFEST_V1,
    # Content type helper
    content_type_name,
    # Path helpers
    PATH_SEPARATOR,
    split_path,
    join_path,
    path_prefix,
    path_parent,
    path_base,
)
from .bridges import (
    from_hf_dataset,
    from_parquet,
    from_image_folder,
    from_csv,
    from_jsonl,
    from_iterable,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "SampleShardReader",
    "SampleShardWriter",
    "ShardHeader",
    "ShardRole",
    "IndexEntry",
    "ShardMetadata",
    "EntryMeta",
    "SampleProfile",
    "ManifestProfile",
    "ManifestFileRef",
    # Content types
    "CONTENT_TYPE_UNKNOWN",
    "CONTENT_TYPE_TENSOR",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_COWRIE",
    "CONTENT_TYPE_GLYPH",
    "CONTENT_TYPE_TEXT",
    "CONTENT_TYPE_IMAGE",
    "CONTENT_TYPE_AUDIO",
    "CONTENT_TYPE_VIDEO",
    "CONTENT_TYPE_PROTO",
    "CONTENT_TYPE_BLOB",
    "CONTENT_TYPE_USER_BASE",
    "SHARD_FLAG_HAS_SCHEMA",
    "SHARD_FLAG_HAS_CONTENT_TYPES",
    "PROFILE_SAMPLESHARD_V1",
    "PROFILE_MANIFEST_V1",
    "content_type_name",
    # Path helpers
    "PATH_SEPARATOR",
    "split_path",
    "join_path",
    "path_prefix",
    "path_parent",
    "path_base",
    # Bridges
    "from_hf_dataset",
    "from_parquet",
    "from_image_folder",
    "from_csv",
    "from_jsonl",
    "from_iterable",
]
