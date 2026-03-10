"""
SampleShard type definitions.

This module defines the core data structures for the SampleShard format:
- ShardHeader: 64-byte v2 header
- IndexEntry: 48-byte index entry
- ShardRole: Role enumeration
- Content type constants (v2.1)
- Path helpers for hierarchical names
"""

import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

# Magic bytes
SHARD_MAGIC = b"SHRD"

# Version
SHARD_VERSION_2 = 0x02

# Header size
SHARD_V2_HEADER_SIZE = 64

# Index entry size
SHARD_V2_INDEX_ENTRY_SIZE = 48

# Alignment values
ALIGN_NONE = 0
ALIGN_16 = 16
ALIGN_32 = 32
ALIGN_64 = 64

# Compression types
COMPRESS_NONE = 0x00
COMPRESS_ZSTD = 0x01
COMPRESS_LZ4 = 0x02

# Content types (entry.reserved[0:2]) - Shard v2.1
CONTENT_TYPE_UNKNOWN = 0x0000
CONTENT_TYPE_TENSOR = 0x0001   # TensorV1 encoded tensor
CONTENT_TYPE_JSON = 0x0002     # Standard JSON
CONTENT_TYPE_COWRIE = 0x0003   # Cowrie binary format
CONTENT_TYPE_GLYPH = 0x0004   # GLYPH text format
CONTENT_TYPE_TEXT = 0x0005     # Plain text (UTF-8)
CONTENT_TYPE_IMAGE = 0x0006   # Image (PNG, JPEG, etc.)
CONTENT_TYPE_AUDIO = 0x0007   # Audio (WAV, MP3, etc.)
CONTENT_TYPE_VIDEO = 0x0008   # Video (MP4, WebM, etc.)
CONTENT_TYPE_PROTO = 0x0009   # Protocol Buffers
CONTENT_TYPE_BLOB = 0x000A    # Opaque binary blob
CONTENT_TYPE_USER_BASE = 0x8000

# Header flag for content types
SHARD_FLAG_HAS_SCHEMA = 0x0010
SHARD_FLAG_HAS_CONTENT_TYPES = 0x0080

PROFILE_SAMPLESHARD_V1 = "sampleshard.v1"
PROFILE_MANIFEST_V1 = "manifest.v1"


class ShardRole(IntEnum):
    """Shard role/profile type."""

    UNKNOWN = 0x00
    UMSH = 0x01  # Model weights
    SAMPLE = 0x02  # Training samples (SampleShard)
    GEMM_PANEL = 0x03  # GEMM panels
    MANIFEST = 0x04  # Multi-file manifest
    WSHARD = 0x05  # W-SHARD: World-model episode data


@dataclass
class EntryMeta:
    """Per-entry metadata stored in shard JSON metadata."""

    content_type: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    codec: str = ""
    codec_version: str = ""
    schema_fingerprint: str = ""
    semantic_type: str = ""
    canonical_hash: str = ""
    base_hash: str = ""
    row_count: int = 0
    shape: List[int] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.content_type:
            data["content_type"] = self.content_type
        if self.tags:
            data["tags"] = self.tags
        if self.description:
            data["description"] = self.description
        if self.extra:
            data["extra"] = self.extra
        if self.codec:
            data["codec"] = self.codec
        if self.codec_version:
            data["codec_version"] = self.codec_version
        if self.schema_fingerprint:
            data["schema_fingerprint"] = self.schema_fingerprint
        if self.semantic_type:
            data["semantic_type"] = self.semantic_type
        if self.canonical_hash:
            data["canonical_hash"] = self.canonical_hash
        if self.base_hash:
            data["base_hash"] = self.base_hash
        if self.row_count:
            data["row_count"] = self.row_count
        if self.shape:
            data["shape"] = self.shape
        if self.stats:
            data["stats"] = self.stats
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntryMeta":
        return cls(
            content_type=data.get("content_type", ""),
            tags=list(data.get("tags", [])),
            description=data.get("description", ""),
            extra=dict(data.get("extra", {})),
            codec=data.get("codec", ""),
            codec_version=data.get("codec_version", ""),
            schema_fingerprint=data.get("schema_fingerprint", ""),
            semantic_type=data.get("semantic_type", ""),
            canonical_hash=data.get("canonical_hash", ""),
            base_hash=data.get("base_hash", ""),
            row_count=int(data.get("row_count", 0)),
            shape=list(data.get("shape", [])),
            stats=dict(data.get("stats", {})),
        )


@dataclass
class SampleProfile:
    """Standardized dataset profile for SampleShard metadata."""

    dataset_name: str = ""
    sample_id_type: str = ""
    key_encoding: str = ""
    sample_count: int = 0
    dataset_schema: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, Any] = field(default_factory=dict)
    label_map: Dict[str, Any] = field(default_factory=dict)
    feature_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.dataset_name:
            data["dataset_name"] = self.dataset_name
        if self.sample_id_type:
            data["sample_id_type"] = self.sample_id_type
        if self.key_encoding:
            data["key_encoding"] = self.key_encoding
        if self.sample_count:
            data["sample_count"] = self.sample_count
        if self.dataset_schema:
            data["dataset_schema"] = self.dataset_schema
        if self.splits:
            data["splits"] = self.splits
        if self.label_map:
            data["label_map"] = self.label_map
        if self.feature_stats:
            data["feature_stats"] = self.feature_stats
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleProfile":
        return cls(
            dataset_name=data.get("dataset_name", ""),
            sample_id_type=data.get("sample_id_type", ""),
            key_encoding=data.get("key_encoding", ""),
            sample_count=int(data.get("sample_count", 0)),
            dataset_schema=dict(data.get("dataset_schema", {})),
            splits=dict(data.get("splits", {})),
            label_map=dict(data.get("label_map", {})),
            feature_stats=dict(data.get("feature_stats", {})),
        )


@dataclass
class ManifestFileRef:
    """Single file reference within a manifest profile."""

    uri: str = ""
    sha256: str = ""
    role: str = ""
    profile: str = ""
    start_key: str = ""
    end_key: str = ""
    entry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.uri:
            data["uri"] = self.uri
        if self.sha256:
            data["sha256"] = self.sha256
        if self.role:
            data["role"] = self.role
        if self.profile:
            data["profile"] = self.profile
        if self.start_key:
            data["start_key"] = self.start_key
        if self.end_key:
            data["end_key"] = self.end_key
        if self.entry_count:
            data["entry_count"] = self.entry_count
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestFileRef":
        return cls(
            uri=data.get("uri", ""),
            sha256=data.get("sha256", ""),
            role=data.get("role", ""),
            profile=data.get("profile", ""),
            start_key=data.get("start_key", ""),
            end_key=data.get("end_key", ""),
            entry_count=int(data.get("entry_count", 0)),
        )


@dataclass
class ManifestProfile:
    """Standardized manifest profile for role=Manifest shards."""

    files: List[ManifestFileRef] = field(default_factory=list)
    partitions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.files:
            data["files"] = [entry.to_dict() for entry in self.files]
        if self.partitions:
            data["partitions"] = self.partitions
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestProfile":
        return cls(
            files=[ManifestFileRef.from_dict(entry) for entry in data.get("files", [])],
            partitions=dict(data.get("partitions", {})),
        )


@dataclass
class ShardMetadata:
    """JSON metadata stored at schema_offset in the file."""

    schema_version: str = "shard-v2.1"
    schema_uri: str = ""
    created_at: str = ""
    source_uri: str = ""
    producer: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    profile: str = ""
    sample_shard: Optional[SampleProfile] = None
    manifest: Optional[ManifestProfile] = None
    entry_metadata: Dict[str, EntryMeta] = field(default_factory=dict)

    def to_json(self) -> bytes:
        data: Dict[str, Any] = {"schema_version": self.schema_version}
        if self.schema_uri:
            data["schema_uri"] = self.schema_uri
        if self.created_at:
            data["created_at"] = self.created_at
        if self.source_uri:
            data["source_uri"] = self.source_uri
        if self.producer:
            data["producer"] = self.producer
        if self.description:
            data["description"] = self.description
        if self.tags:
            data["tags"] = self.tags
        if self.extra:
            data["extra"] = self.extra
        if self.profile:
            data["profile"] = self.profile
        if self.sample_shard is not None:
            data["sample_shard"] = self.sample_shard.to_dict()
        if self.manifest is not None:
            data["manifest"] = self.manifest.to_dict()
        if self.entry_metadata:
            data["entry_metadata"] = {
                name: meta.to_dict() for name, meta in self.entry_metadata.items()
            }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "ShardMetadata":
        data = json.loads(raw.decode("utf-8"))
        return cls(
            schema_version=data.get("schema_version", "shard-v2.1"),
            schema_uri=data.get("schema_uri", ""),
            created_at=data.get("created_at", ""),
            source_uri=data.get("source_uri", ""),
            producer=data.get("producer", ""),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            extra=dict(data.get("extra", {})),
            profile=data.get("profile", ""),
            sample_shard=(
                SampleProfile.from_dict(data["sample_shard"])
                if "sample_shard" in data
                else None
            ),
            manifest=(
                ManifestProfile.from_dict(data["manifest"])
                if "manifest" in data
                else None
            ),
            entry_metadata={
                name: EntryMeta.from_dict(meta)
                for name, meta in data.get("entry_metadata", {}).items()
            },
        )


@dataclass
class ShardHeader:
    """
    Shard v2 header (64 bytes).

    Layout:
        Bytes 0-3:   magic = 'S','H','R','D'
        Byte 4:      version (0x02)
        Byte 5:      role
        Bytes 6-7:   flags
        Byte 8:      alignment
        Byte 9:      compression_default
        Bytes 10-11: reserved
        Bytes 12-15: entry_count (uint32)
        Bytes 16-23: string_table_offset (uint64)
        Bytes 24-31: data_section_offset (uint64)
        Bytes 32-39: schema_offset (uint64)
        Bytes 40-47: total_file_size (uint64)
        Bytes 48-63: reserved
    """

    magic: bytes = SHARD_MAGIC
    version: int = SHARD_VERSION_2
    role: ShardRole = ShardRole.SAMPLE
    flags: int = 0
    alignment: int = ALIGN_64
    compression_default: int = COMPRESS_NONE
    entry_count: int = 0
    string_table_offset: int = 0
    data_section_offset: int = 0
    schema_offset: int = 0
    total_file_size: int = 0

    def to_bytes(self) -> bytes:
        """Serialize header to 64 bytes."""
        buf = bytearray(SHARD_V2_HEADER_SIZE)

        # Magic (4 bytes)
        buf[0:4] = self.magic

        # Version (1 byte)
        buf[4] = self.version

        # Role (1 byte)
        buf[5] = self.role

        # Flags (2 bytes, little-endian)
        struct.pack_into("<H", buf, 6, self.flags)

        # Alignment (1 byte)
        buf[8] = self.alignment

        # Compression default (1 byte)
        buf[9] = self.compression_default

        # Index entry size (2 bytes, little-endian) - MUST be 48
        struct.pack_into("<H", buf, 10, SHARD_V2_INDEX_ENTRY_SIZE)

        # Entry count (4 bytes, little-endian)
        struct.pack_into("<I", buf, 12, self.entry_count)

        # String table offset (8 bytes, little-endian)
        struct.pack_into("<Q", buf, 16, self.string_table_offset)

        # Data section offset (8 bytes, little-endian)
        struct.pack_into("<Q", buf, 24, self.data_section_offset)

        # Schema offset (8 bytes, little-endian)
        struct.pack_into("<Q", buf, 32, self.schema_offset)

        # Total file size (8 bytes, little-endian)
        struct.pack_into("<Q", buf, 40, self.total_file_size)

        # Reserved (16 bytes)
        buf[48:64] = b"\x00" * 16

        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ShardHeader":
        """Parse header from 64 bytes."""
        if len(data) < SHARD_V2_HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {SHARD_V2_HEADER_SIZE}")

        magic = data[0:4]
        if magic != SHARD_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")

        version = data[4]
        if version != SHARD_VERSION_2:
            raise ValueError(f"Unsupported version: {version}")

        return cls(
            magic=magic,
            version=version,
            role=ShardRole(data[5]),
            flags=struct.unpack_from("<H", data, 6)[0],
            alignment=data[8],
            compression_default=data[9],
            entry_count=struct.unpack_from("<I", data, 12)[0],
            string_table_offset=struct.unpack_from("<Q", data, 16)[0],
            data_section_offset=struct.unpack_from("<Q", data, 24)[0],
            schema_offset=struct.unpack_from("<Q", data, 32)[0],
            total_file_size=struct.unpack_from("<Q", data, 40)[0],
        )


@dataclass
class IndexEntry:
    """
    Shard v2 index entry (48 bytes).

    Layout:
        Bytes 0-7:   name_hash (xxHash64)
        Bytes 8-11:  name_offset (uint32)
        Bytes 12-13: name_len (uint16)
        Bytes 14-15: flags (uint16)
        Bytes 16-23: data_offset (uint64)
        Bytes 24-31: disk_size (uint64)
        Bytes 32-39: orig_size (uint64)
        Bytes 40-43: checksum (uint32, CRC32C)
        Bytes 44-47: reserved
    """

    name_hash: int = 0
    name_offset: int = 0
    name_len: int = 0
    flags: int = 0
    data_offset: int = 0
    disk_size: int = 0
    orig_size: int = 0
    checksum: int = 0

    # Populated after reading string table
    name: str = ""

    def to_bytes(self) -> bytes:
        """Serialize entry to 48 bytes."""
        buf = bytearray(SHARD_V2_INDEX_ENTRY_SIZE)

        struct.pack_into("<Q", buf, 0, self.name_hash)
        struct.pack_into("<I", buf, 8, self.name_offset)
        struct.pack_into("<H", buf, 12, self.name_len)
        struct.pack_into("<H", buf, 14, self.flags)
        struct.pack_into("<Q", buf, 16, self.data_offset)
        struct.pack_into("<Q", buf, 24, self.disk_size)
        struct.pack_into("<Q", buf, 32, self.orig_size)
        struct.pack_into("<I", buf, 40, self.checksum)
        # Reserved (4 bytes)
        buf[44:48] = b"\x00\x00\x00\x00"

        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> "IndexEntry":
        """Parse entry from 48 bytes."""
        if len(data) < SHARD_V2_INDEX_ENTRY_SIZE:
            raise ValueError(
                f"Entry too short: {len(data)} < {SHARD_V2_INDEX_ENTRY_SIZE}"
            )

        return cls(
            name_hash=struct.unpack_from("<Q", data, 0)[0],
            name_offset=struct.unpack_from("<I", data, 8)[0],
            name_len=struct.unpack_from("<H", data, 12)[0],
            flags=struct.unpack_from("<H", data, 14)[0],
            data_offset=struct.unpack_from("<Q", data, 16)[0],
            disk_size=struct.unpack_from("<Q", data, 24)[0],
            orig_size=struct.unpack_from("<Q", data, 32)[0],
            checksum=struct.unpack_from("<I", data, 40)[0],
        )

    def is_compressed(self) -> bool:
        """Check if entry is compressed."""
        return (self.flags & 0x0001) != 0

    def compression_type(self) -> int:
        """Get compression type (0=none, 1=zstd, 2=lz4). Matches Go bit flags."""
        if self.flags & 0x0004:
            return 2  # LZ4
        if self.flags & 0x0002:
            return 1  # zstd
        return 0


# xxHash64: Try to use fast xxhash library, fall back to pure Python
try:
    import xxhash as _xxhash

    def xxhash64(data: bytes) -> int:
        """Compute xxHash64 of data (using fast xxhash library)."""
        return _xxhash.xxh64(data).intdigest()
except ImportError:

    # Pure Python xxHash64 implementation (matches Go's cespare/xxhash/v2)
    _PRIME1 = 0x9E3779B185EBCA87
    _PRIME2 = 0xC2B2AE3D27D4EB4F
    _PRIME3 = 0x165667B19E3779F9
    _PRIME4 = 0x85EBCA77C2B2AE63
    _PRIME5 = 0x27D4EB2F165667C5
    _MASK64 = 0xFFFFFFFFFFFFFFFF

    def _rol64(x: int, k: int) -> int:
        return ((x << k) | (x >> (64 - k))) & _MASK64

    def _xxh_round(acc: int, inp: int) -> int:
        acc = (acc + inp * _PRIME2) & _MASK64
        acc = _rol64(acc, 31)
        return (acc * _PRIME1) & _MASK64

    def _xxh_merge_round(acc: int, val: int) -> int:
        val = _xxh_round(0, val)
        acc ^= val
        return (acc * _PRIME1 + _PRIME4) & _MASK64

    def _read_u64le(data: bytes, offset: int) -> int:
        return int.from_bytes(data[offset:offset + 8], "little")

    def _read_u32le(data: bytes, offset: int) -> int:
        return int.from_bytes(data[offset:offset + 4], "little")

    def xxhash64(data: bytes) -> int:
        """Compute xxHash64 of data (pure Python fallback)."""
        n = len(data)
        p = 0

        if n >= 32:
            v1 = (_PRIME1 + _PRIME2) & _MASK64
            v2 = _PRIME2
            v3 = 0
            v4 = (0 - _PRIME1) & _MASK64

            while p + 32 <= n:
                v1 = _xxh_round(v1, _read_u64le(data, p))
                v2 = _xxh_round(v2, _read_u64le(data, p + 8))
                v3 = _xxh_round(v3, _read_u64le(data, p + 16))
                v4 = _xxh_round(v4, _read_u64le(data, p + 24))
                p += 32

            h = (_rol64(v1, 1) + _rol64(v2, 7) + _rol64(v3, 12) + _rol64(v4, 18)) & _MASK64
            h = _xxh_merge_round(h, v1)
            h = _xxh_merge_round(h, v2)
            h = _xxh_merge_round(h, v3)
            h = _xxh_merge_round(h, v4)
        else:
            h = _PRIME5

        h = (h + n) & _MASK64

        while p + 8 <= n:
            k1 = _xxh_round(0, _read_u64le(data, p))
            h ^= k1
            h = (_rol64(h, 27) * _PRIME1 + _PRIME4) & _MASK64
            p += 8

        if p + 4 <= n:
            h ^= (_read_u32le(data, p) * _PRIME1) & _MASK64
            h = (_rol64(h, 23) * _PRIME2 + _PRIME3) & _MASK64
            p += 4

        while p < n:
            h ^= (data[p] * _PRIME5) & _MASK64
            h = (_rol64(h, 11) * _PRIME1) & _MASK64
            p += 1

        # Avalanche
        h ^= h >> 33
        h = (h * _PRIME2) & _MASK64
        h ^= h >> 29
        h = (h * _PRIME3) & _MASK64
        h ^= h >> 32

        return h


# CRC32C: Requires crc32c library (zlib.crc32 uses the wrong polynomial)
try:
    import crc32c as _crc32c_lib

    def crc32c(data: bytes) -> int:
        """Compute CRC32C checksum (using fast crc32c library)."""
        return _crc32c_lib.crc32c(data)
except ImportError:
    raise ImportError(
        "crc32c library is required for SampleShard CRC32C checksums. "
        "Install it with: pip install crc32c"
    )


# ============================================================
# Content type helpers
# ============================================================

_CONTENT_TYPE_NAMES = {
    CONTENT_TYPE_TENSOR: "tensor",
    CONTENT_TYPE_JSON: "json",
    CONTENT_TYPE_COWRIE: "cowrie",
    CONTENT_TYPE_GLYPH: "glyph",
    CONTENT_TYPE_TEXT: "text",
    CONTENT_TYPE_IMAGE: "image",
    CONTENT_TYPE_AUDIO: "audio",
    CONTENT_TYPE_VIDEO: "video",
    CONTENT_TYPE_PROTO: "proto",
    CONTENT_TYPE_BLOB: "blob",
}


def content_type_name(ct: int) -> str:
    """Get human-readable content type name."""
    name = _CONTENT_TYPE_NAMES.get(ct)
    if name:
        return name
    if ct >= CONTENT_TYPE_USER_BASE:
        return f"user:{ct - CONTENT_TYPE_USER_BASE}"
    return "unknown"


# ============================================================
# Path helpers (Shard v2.1)
# ============================================================

PATH_SEPARATOR = "/"


def split_path(name: str) -> List[str]:
    """Split a hierarchical name into path components."""
    return [p for p in name.split(PATH_SEPARATOR) if p]


def join_path(*parts: str) -> str:
    """Join path components into a hierarchical name."""
    return PATH_SEPARATOR.join(parts)


def path_prefix(name: str, prefix: str) -> bool:
    """Check if name starts with the given prefix."""
    if not prefix:
        return True
    if not prefix.endswith(PATH_SEPARATOR):
        prefix += PATH_SEPARATOR
    return name.startswith(prefix) or name == prefix[:-1]


def path_parent(name: str) -> str:
    """Get parent path (everything before last '/')."""
    idx = name.rfind(PATH_SEPARATOR)
    if idx < 0:
        return ""
    return name[:idx]


def path_base(name: str) -> str:
    """Get base name (everything after last '/')."""
    idx = name.rfind(PATH_SEPARATOR)
    if idx < 0:
        return name
    return name[idx + 1:]
