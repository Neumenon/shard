"""
Shard container format — reader and writer.

Binary-compatible with the Go reference implementation at go/shard/shard_format.go.
"""
import fnmatch
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import crc32c as _crc32c
import xxhash
import zstandard as zstd
import lz4.block

# ============================================================
# Constants
# ============================================================

SHARD_MAGIC = b"SHRD"
SHARD_VERSION = 0x02
HEADER_SIZE = 64
INDEX_ENTRY_SIZE = 48

# Roles
ROLE_UNKNOWN = 0
ROLE_MOSH = 1
ROLE_SAMPLE = 2
ROLE_GEMMPANEL = 3
ROLE_MANIFEST = 4
ROLE_WSHARD = 5
ROLE_UMSH = 6

# Alignment
ALIGN_NONE = 0
ALIGN_16 = 16
ALIGN_32 = 32
ALIGN_64 = 64

# Compression
COMPRESS_NONE = 0
COMPRESS_ZSTD = 1
COMPRESS_LZ4 = 2

# Header flags
FLAG_LITTLE_ENDIAN = 0x0002
FLAG_HAS_SCHEMA = 0x0010
FLAG_HAS_CHECKSUMS = 0x0020
FLAG_STREAMING = 0x0040
FLAG_HAS_CONTENT_TYPES = 0x0080

# Entry flags
ENTRY_FLAG_COMPRESSED = 0x0001
ENTRY_FLAG_ZSTD = 0x0002
ENTRY_FLAG_LZ4 = 0x0004
ENTRY_FLAG_CHUNKED = 0x0008

# Content types
CONTENT_TYPE_UNKNOWN = 0
CONTENT_TYPE_TENSOR = 1
CONTENT_TYPE_JSON = 2
CONTENT_TYPE_COWRIE = 3
CONTENT_TYPE_GLYPH = 4
CONTENT_TYPE_TEXT = 5
CONTENT_TYPE_IMAGE = 6
CONTENT_TYPE_AUDIO = 7
CONTENT_TYPE_VIDEO = 8
CONTENT_TYPE_PROTO = 9
CONTENT_TYPE_BLOB = 10
CONTENT_TYPE_QMLN = 11
CONTENT_TYPE_TENSOR_V3 = 12
CONTENT_TYPE_ANCHOR_SHARED = 13
CONTENT_TYPE_DELTA_EXPERT = 14
CONTENT_TYPE_CODEBOOK_SHARED = 15
CONTENT_TYPE_EXPERT_INDICES = 16

DEFAULT_FLAGS = FLAG_LITTLE_ENDIAN | FLAG_HAS_CHECKSUMS | FLAG_HAS_CONTENT_TYPES

# Compression thresholds (match Go reference)
MIN_COMPRESS_SIZE = 256
COMPRESSION_SAVINGS_NUM = 9
COMPRESSION_SAVINGS_DEN = 10
MAX_DECOMPRESS_SIZE = 1 << 30  # 1 GB DoS guard

# ============================================================
# Security Limits
# ============================================================

MAX_INDEX_SIZE = 1 << 30              # 1 GB
MAX_STRING_TABLE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_ENTRY_COUNT = 10_000_000


# ============================================================
# Hashing / Checksums
# ============================================================

def compute_crc32c(data: bytes) -> int:
    """CRC32C (Castagnoli) checksum, matching Go's crc32.Castagnoli."""
    return _crc32c.crc32c(data)


def compute_xxhash64(data: Union[str, bytes], seed: int = 0) -> int:
    """xxHash64, matching Go's github.com/cespare/xxhash/v2."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return xxhash.xxh64(data, seed=seed).intdigest()


# ============================================================
# Metadata Structures
# ============================================================

@dataclass
class EntryMeta:
    """Per-entry metadata stored in the schema section."""
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
        d: Dict[str, Any] = {}
        if self.content_type:
            d["content_type"] = self.content_type
        if self.tags:
            d["tags"] = self.tags
        if self.description:
            d["description"] = self.description
        if self.extra:
            d["extra"] = self.extra
        if self.codec:
            d["codec"] = self.codec
        if self.codec_version:
            d["codec_version"] = self.codec_version
        if self.schema_fingerprint:
            d["schema_fingerprint"] = self.schema_fingerprint
        if self.semantic_type:
            d["semantic_type"] = self.semantic_type
        if self.canonical_hash:
            d["canonical_hash"] = self.canonical_hash
        if self.base_hash:
            d["base_hash"] = self.base_hash
        if self.row_count:
            d["row_count"] = self.row_count
        if self.shape:
            d["shape"] = self.shape
        if self.stats:
            d["stats"] = self.stats
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntryMeta":
        return cls(
            content_type=d.get("content_type", ""),
            tags=d.get("tags", []),
            description=d.get("description", ""),
            extra=d.get("extra", {}),
            codec=d.get("codec", ""),
            codec_version=d.get("codec_version", ""),
            schema_fingerprint=d.get("schema_fingerprint", ""),
            semantic_type=d.get("semantic_type", ""),
            canonical_hash=d.get("canonical_hash", ""),
            base_hash=d.get("base_hash", ""),
            row_count=d.get("row_count", 0),
            shape=d.get("shape", []),
            stats=d.get("stats", {}),
        )


@dataclass
class SampleProfile:
    """Shard-level dataset metadata for SampleShard files."""
    dataset_name: str = ""
    sample_id_type: str = ""
    key_encoding: str = ""
    sample_count: int = 0
    dataset_schema: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, Any] = field(default_factory=dict)
    label_map: Dict[str, Any] = field(default_factory=dict)
    feature_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.dataset_name:
            d["dataset_name"] = self.dataset_name
        if self.sample_id_type:
            d["sample_id_type"] = self.sample_id_type
        if self.key_encoding:
            d["key_encoding"] = self.key_encoding
        if self.sample_count:
            d["sample_count"] = self.sample_count
        if self.dataset_schema:
            d["dataset_schema"] = self.dataset_schema
        if self.splits:
            d["splits"] = self.splits
        if self.label_map:
            d["label_map"] = self.label_map
        if self.feature_stats:
            d["feature_stats"] = self.feature_stats
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleProfile":
        return cls(
            dataset_name=d.get("dataset_name", ""),
            sample_id_type=d.get("sample_id_type", ""),
            key_encoding=d.get("key_encoding", ""),
            sample_count=d.get("sample_count", 0),
            dataset_schema=d.get("dataset_schema", {}),
            splits=d.get("splits", {}),
            label_map=d.get("label_map", {}),
            feature_stats=d.get("feature_stats", {}),
        )


@dataclass
class ManifestFileRef:
    """Single manifest reference to another shard file."""
    uri: str = ""
    sha256: str = ""
    role: str = ""
    profile: str = ""
    start_key: str = ""
    end_key: str = ""
    entry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.uri:
            d["uri"] = self.uri
        if self.sha256:
            d["sha256"] = self.sha256
        if self.role:
            d["role"] = self.role
        if self.profile:
            d["profile"] = self.profile
        if self.start_key:
            d["start_key"] = self.start_key
        if self.end_key:
            d["end_key"] = self.end_key
        if self.entry_count:
            d["entry_count"] = self.entry_count
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestFileRef":
        return cls(
            uri=d.get("uri", ""),
            sha256=d.get("sha256", ""),
            role=d.get("role", ""),
            profile=d.get("profile", ""),
            start_key=d.get("start_key", ""),
            end_key=d.get("end_key", ""),
            entry_count=d.get("entry_count", 0),
        )


@dataclass
class ManifestProfile:
    """Shard-level metadata for manifest shards."""
    files: List[ManifestFileRef] = field(default_factory=list)
    partitions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.files:
            d["files"] = [f.to_dict() for f in self.files]
        if self.partitions:
            d["partitions"] = self.partitions
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestProfile":
        return cls(
            files=[ManifestFileRef.from_dict(v) for v in d.get("files", [])],
            partitions=d.get("partitions", {}),
        )


@dataclass
class ShardMetadata:
    """JSON metadata stored at schema_offset in the file."""
    schema_version: str = "shard-v2.1"
    schema_uri: str = ""
    schema_id: str = ""
    created_at: str = ""  # ISO 8601
    source_uri: str = ""
    producer: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    tag_dictionary: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    profile: str = ""
    sample_shard: Optional[SampleProfile] = None
    manifest: Optional[ManifestProfile] = None
    entry_metadata: Dict[str, EntryMeta] = field(default_factory=dict)

    def to_json(self) -> bytes:
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
        }
        if self.schema_uri:
            d["schema_uri"] = self.schema_uri
        if self.schema_id:
            d["schema_id"] = self.schema_id
        if self.created_at:
            d["created_at"] = self.created_at
        if self.source_uri:
            d["source_uri"] = self.source_uri
        if self.producer:
            d["producer"] = self.producer
        if self.description:
            d["description"] = self.description
        if self.tags:
            d["tags"] = self.tags
        if self.tag_dictionary:
            d["tag_dictionary"] = self.tag_dictionary
        if self.extra:
            d["extra"] = self.extra
        if self.profile:
            d["profile"] = self.profile
        if self.sample_shard is not None:
            d["sample_shard"] = self.sample_shard.to_dict()
        if self.manifest is not None:
            d["manifest"] = self.manifest.to_dict()
        if self.entry_metadata:
            d["entry_metadata"] = {
                k: v.to_dict() for k, v in self.entry_metadata.items()
            }
        return json.dumps(d, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "ShardMetadata":
        d = json.loads(data.decode("utf-8"))
        em: Dict[str, EntryMeta] = {}
        for k, v in d.get("entry_metadata", {}).items():
            em[k] = EntryMeta.from_dict(v)
        return cls(
            schema_version=d.get("schema_version", "shard-v2.1"),
            schema_uri=d.get("schema_uri", ""),
            schema_id=d.get("schema_id", ""),
            created_at=d.get("created_at", ""),
            source_uri=d.get("source_uri", ""),
            producer=d.get("producer", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            tag_dictionary=d.get("tag_dictionary", []),
            extra=d.get("extra", {}),
            profile=d.get("profile", ""),
            sample_shard=(
                SampleProfile.from_dict(d["sample_shard"])
                if "sample_shard" in d
                else None
            ),
            manifest=(
                ManifestProfile.from_dict(d["manifest"])
                if "manifest" in d
                else None
            ),
            entry_metadata=em,
        )

    def compute_schema_id(self) -> str:
        """Compute a schema identifier by hashing tag_dictionary and entry metadata shape."""
        import hashlib
        h = hashlib.sha256()
        for t in self.tag_dictionary:
            h.update(t.encode("utf-8"))
            h.update(b"\x00")
        names = sorted(self.entry_metadata.keys())
        for name in names:
            em = self.entry_metadata[name]
            h.update(name.encode("utf-8"))
            h.update(b"\x00")
            h.update(em.content_type.encode("utf-8"))
            h.update(b"\x00")
            h.update(em.semantic_type.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]


# ============================================================
# Path helpers
# ============================================================

def split_path(name: str) -> List[str]:
    """Split a shard entry name on '/' into path components."""
    if not name:
        return []
    return name.split("/")


def join_path(*parts: str) -> str:
    """Join path components with '/'."""
    return "/".join(parts)


def path_parent(name: str) -> str:
    """Return everything before the last '/', or '' if no '/'."""
    idx = name.rfind("/")
    if idx < 0:
        return ""
    return name[:idx]


def path_base(name: str) -> str:
    """Return everything after the last '/', or the whole name if no '/'."""
    idx = name.rfind("/")
    if idx < 0:
        return name
    return name[idx + 1:]


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ShardHeader:
    magic: bytes = SHARD_MAGIC
    version: int = SHARD_VERSION
    role: int = ROLE_MOSH
    flags: int = DEFAULT_FLAGS
    alignment: int = ALIGN_64
    compression_default: int = COMPRESS_NONE
    index_entry_size: int = INDEX_ENTRY_SIZE
    entry_count: int = 0
    string_table_offset: int = 0
    data_section_offset: int = 0
    schema_offset: int = 0
    total_file_size: int = 0

    def to_bytes(self) -> bytes:
        buf = bytearray(HEADER_SIZE)
        buf[0:4] = self.magic
        buf[4] = self.version
        buf[5] = self.role
        struct.pack_into("<H", buf, 6, self.flags)
        buf[8] = self.alignment
        buf[9] = self.compression_default
        struct.pack_into("<H", buf, 10, self.index_entry_size)
        struct.pack_into("<I", buf, 12, self.entry_count)
        struct.pack_into("<Q", buf, 16, self.string_table_offset)
        struct.pack_into("<Q", buf, 24, self.data_section_offset)
        struct.pack_into("<Q", buf, 32, self.schema_offset)
        struct.pack_into("<Q", buf, 40, self.total_file_size)
        # bytes 48-63 reserved (zeroed)
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ShardHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"header too short: {len(data)} < {HEADER_SIZE}")
        magic = data[0:4]
        if magic != SHARD_MAGIC:
            raise ValueError(f"invalid magic: {magic!r}")
        version = data[4]
        if version != SHARD_VERSION:
            raise ValueError(f"unsupported version: {version}")
        return cls(
            magic=magic,
            version=version,
            role=data[5],
            flags=struct.unpack_from("<H", data, 6)[0],
            alignment=data[8],
            compression_default=data[9],
            index_entry_size=struct.unpack_from("<H", data, 10)[0],
            entry_count=struct.unpack_from("<I", data, 12)[0],
            string_table_offset=struct.unpack_from("<Q", data, 16)[0],
            data_section_offset=struct.unpack_from("<Q", data, 24)[0],
            schema_offset=struct.unpack_from("<Q", data, 32)[0],
            total_file_size=struct.unpack_from("<Q", data, 40)[0],
        )


@dataclass
class IndexEntry:
    name_hash: int = 0
    name_offset: int = 0
    name_len: int = 0
    flags: int = 0
    data_offset: int = 0
    disk_size: int = 0
    orig_size: int = 0
    checksum: int = 0
    reserved: int = 0  # lower 16 bits = content_type
    # Resolved fields (not serialized)
    name: str = ""

    @property
    def content_type(self) -> int:
        return self.reserved & 0xFFFF

    @property
    def tag_bits(self) -> int:
        """16-bit tag bitmask from upper 16 bits of reserved."""
        return (self.reserved >> 16) & 0xFFFF

    def has_tag_bit(self, bit: int) -> bool:
        """Check if a specific tag bit (0-15) is set."""
        if bit < 0 or bit > 15:
            return False
        return bool(self.tag_bits & (1 << bit))

    @property
    def compressed(self) -> bool:
        return bool(self.flags & ENTRY_FLAG_COMPRESSED)

    def to_bytes(self) -> bytes:
        buf = bytearray(INDEX_ENTRY_SIZE)
        struct.pack_into("<Q", buf, 0, self.name_hash)
        struct.pack_into("<I", buf, 8, self.name_offset)
        struct.pack_into("<H", buf, 12, self.name_len)
        struct.pack_into("<H", buf, 14, self.flags)
        struct.pack_into("<Q", buf, 16, self.data_offset)
        struct.pack_into("<Q", buf, 24, self.disk_size)
        struct.pack_into("<Q", buf, 32, self.orig_size)
        struct.pack_into("<I", buf, 40, self.checksum)
        struct.pack_into("<I", buf, 44, self.reserved)
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> "IndexEntry":
        if len(data) < INDEX_ENTRY_SIZE:
            raise ValueError(f"entry too short: {len(data)}")
        return cls(
            name_hash=struct.unpack_from("<Q", data, 0)[0],
            name_offset=struct.unpack_from("<I", data, 8)[0],
            name_len=struct.unpack_from("<H", data, 12)[0],
            flags=struct.unpack_from("<H", data, 14)[0],
            data_offset=struct.unpack_from("<Q", data, 16)[0],
            disk_size=struct.unpack_from("<Q", data, 24)[0],
            orig_size=struct.unpack_from("<Q", data, 32)[0],
            checksum=struct.unpack_from("<I", data, 40)[0],
            reserved=struct.unpack_from("<I", data, 44)[0],
        )


# ============================================================
# Reader
# ============================================================

class ShardReader:
    """Read Shard files."""

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)
        self._data = self._path.read_bytes()
        self._header = ShardHeader.from_bytes(self._data[:HEADER_SIZE])

        if self._header.index_entry_size != INDEX_ENTRY_SIZE:
            raise ValueError(
                f"invalid index entry size: {self._header.index_entry_size}"
            )

        # Security limits
        entry_count = self._header.entry_count
        if entry_count > MAX_ENTRY_COUNT:
            raise ValueError(
                f"entry_count {entry_count} exceeds MAX_ENTRY_COUNT {MAX_ENTRY_COUNT}"
            )

        index_size = entry_count * INDEX_ENTRY_SIZE
        if index_size > MAX_INDEX_SIZE:
            raise ValueError(
                f"index size {index_size} exceeds MAX_INDEX_SIZE {MAX_INDEX_SIZE}"
            )

        st_start = self._header.string_table_offset
        ds_start = self._header.data_section_offset
        file_size = len(self._data)

        # String table must be within the file
        if st_start > file_size:
            raise ValueError(
                f"string_table_offset {st_start} beyond file size {file_size}"
            )

        if ds_start >= st_start:
            string_table_size = ds_start - st_start
            if string_table_size > MAX_STRING_TABLE_SIZE:
                raise ValueError(
                    f"string table size {string_table_size} exceeds MAX_STRING_TABLE_SIZE"
                )

        if self._header.total_file_size != len(self._data):
            raise ValueError(
                f"total_file_size {self._header.total_file_size} != actual file size {len(self._data)}"
            )

        self._entries: List[IndexEntry] = []
        self._name_to_index: Dict[str, int] = {}
        self._parse_index()

        # Validate data offsets after parsing index
        self._validate_entry_offsets()

    def _parse_index(self):
        st_start = self._header.string_table_offset
        ds_start = self._header.data_section_offset
        string_table = self._data[st_start:ds_start]

        for i in range(self._header.entry_count):
            off = HEADER_SIZE + i * INDEX_ENTRY_SIZE
            entry = IndexEntry.from_bytes(
                self._data[off : off + INDEX_ENTRY_SIZE]
            )
            # Resolve name from string table (with bounds check matching Go)
            if entry.name_offset >= len(string_table) or entry.name_len > len(string_table) - entry.name_offset:
                entry.name = ""
            else:
                name_bytes = string_table[
                    entry.name_offset : entry.name_offset + entry.name_len
                ]
                entry.name = name_bytes.decode("utf-8")
            self._entries.append(entry)
            self._name_to_index[entry.name] = i

    def _validate_entry_offsets(self):
        """Validate all entry data_offsets are in bounds and monotonically increasing."""
        ds_start = self._header.data_section_offset
        file_size = len(self._data)
        prev_end = 0

        for i, entry in enumerate(self._entries):
            offset = entry.data_offset
            size = entry.disk_size

            # Must be within data section
            if offset < ds_start:
                raise ValueError(
                    f"entry {i} ({entry.name!r}) data_offset {offset} < data_section_offset {ds_start}"
                )
            if offset + size > file_size:
                raise ValueError(
                    f"entry {i} ({entry.name!r}) data extends past end of file"
                )

            # Monotonically increasing (no overlaps)
            if i > 0 and offset < prev_end:
                raise ValueError(
                    f"entry {i} ({entry.name!r}) data_offset {offset} overlaps with previous entry ending at {prev_end}"
                )
            prev_end = offset + size

    def header(self) -> ShardHeader:
        return self._header

    def entry_count(self) -> int:
        return len(self._entries)

    def entry_name(self, i: int) -> str:
        return self._entries[i].name

    def entry_names(self) -> List[str]:
        return [e.name for e in self._entries]

    def get_entry_info(self, i: int) -> IndexEntry:
        return self._entries[i]

    def lookup(self, name: str) -> int:
        """Return index for name, or -1 if not found."""
        return self._name_to_index.get(name, -1)

    def has_entry(self, name: str) -> bool:
        return name in self._name_to_index

    def read_entry(self, i: int, verify: bool = True) -> bytes:
        """Read entry data by index. Verifies CRC32C by default."""
        entry = self._entries[i]
        offset = entry.data_offset
        size = entry.disk_size
        data = self._data[offset : offset + size]

        if entry.compressed:
            if entry.orig_size > MAX_DECOMPRESS_SIZE:
                raise ValueError(
                    f"decompressed size {entry.orig_size} exceeds limit"
                )
            if entry.flags & ENTRY_FLAG_ZSTD:
                dctx = zstd.ZstdDecompressor()
                data = dctx.decompress(data, max_output_size=entry.orig_size)
            elif entry.flags & ENTRY_FLAG_LZ4:
                data = lz4.block.decompress(data, uncompressed_size=entry.orig_size)
            else:
                raise ValueError(f"unknown compression type for entry {i}")

        # CRC32C checksum is computed on the original uncompressed data.
        if verify and (self._header.flags & FLAG_HAS_CHECKSUMS or entry.checksum != 0):
            computed = compute_crc32c(data)
            if computed != entry.checksum:
                raise ValueError(
                    f"checksum mismatch for entry {i} ({entry.name!r}): "
                    f"expected {entry.checksum:#010x}, got {computed:#010x}"
                )

        return data

    def read_entry_by_name(self, name: str, verify: bool = True) -> bytes:
        idx = self.lookup(name)
        if idx < 0:
            raise KeyError(f"entry not found: {name!r}")
        return self.read_entry(idx, verify=verify)

    def list_prefix(self, prefix: str) -> List[str]:
        if not prefix:
            return self.entry_names()
        return [e.name for e in self._entries if e.name.startswith(prefix)]

    def list_children(self, prefix: str) -> List[str]:
        """Return immediate children (one path component deep) under prefix.

        Returns bare components matching the Go reference implementation:
          entries = ["layer.0/weight", "layer.0/bias", "layer.1/weight", "embed"]
          list_children("layer.0/") -> ["weight", "bias"]
          list_children("")         -> ["layer.0", "layer.1", "embed"]
          list_children("layer.")   -> ["0", "1"]
        """
        result: List[str] = []
        seen: set = set()

        for entry in self._entries:
            name = entry.name
            if not name.startswith(prefix):
                continue
            remainder = name[len(prefix):]
            slash_pos = remainder.find("/")
            if slash_pos >= 0:
                # Has sub-components — return bare component (no trailing slash)
                child = remainder[:slash_pos]
            else:
                # Leaf entry — bare remainder
                child = remainder

            if child and child not in seen:
                seen.add(child)
                result.append(child)

        return result

    def list_with_tag(self, tag: str) -> List[str]:
        """Return entry names that have the given tag in their per-entry metadata."""
        meta = self.read_metadata()
        if meta is None:
            return []
        result = []
        for name, em in meta.entry_metadata.items():
            if tag in em.tags:
                result.append(name)
        return result

    def list_with_tag_bit(self, bit: int) -> List[str]:
        """Return entry names where the given tag bit (0-15) is set. Pure index scan, no metadata needed."""
        if bit < 0 or bit > 15:
            return []
        return [e.name for e in self._entries if e.has_tag_bit(bit)]

    def list_with_tag_fast(self, tag: str) -> List[str]:
        """Fast tag filtering using tag_dictionary + bitmask. Falls back to empty if no dictionary."""
        meta = self.read_metadata()
        if meta is None or not meta.tag_dictionary:
            return []
        try:
            bit = meta.tag_dictionary.index(tag)
        except ValueError:
            return []
        return self.list_with_tag_bit(bit)

    def read_entry_prefix(self, i: int, max_bytes: int) -> bytes:
        """Read only the first max_bytes of an entry (for header scanning)."""
        entry = self._entries[i]
        if entry.compressed:
            # Must decompress fully, then truncate
            data = self.read_entry(i, verify=False)
            return data[:max_bytes]
        # Uncompressed: direct slice
        offset = entry.data_offset
        size = min(entry.disk_size, max_bytes)
        return self._data[offset : offset + size]

    def read_metadata(self) -> Optional[ShardMetadata]:
        """Read JSON metadata from schema section. Returns None if no schema."""
        if self._header.schema_offset == 0:
            return None
        file_len = len(self._data)
        schema_offset = self._header.schema_offset
        total_file_size = self._header.total_file_size
        if schema_offset > file_len or schema_offset > total_file_size:
            raise ValueError(
                f"schema_offset {schema_offset} out of bounds "
                f"(file_len={file_len}, total_file_size={total_file_size})"
            )
        meta_data = self._data[schema_offset : total_file_size]
        return ShardMetadata.from_json(meta_data)


# ============================================================
# Writer
# ============================================================

@dataclass
class _PendingEntry:
    name: str
    data: bytes
    content_type: int
    compress: bool = False
    comp_type: int = COMPRESS_NONE


class ShardWriter:
    """Write Shard files."""

    def __init__(self, path: Union[str, Path], role: int = ROLE_MOSH):
        self._path = Path(path)
        self._role = role
        self._alignment = ALIGN_64
        self._entries: List[_PendingEntry] = []
        self._compression = COMPRESS_NONE
        self._metadata: Optional[ShardMetadata] = None

    def set_metadata(self, meta: ShardMetadata):
        """Attach metadata to be serialized into the schema section on close()."""
        self._metadata = meta

    def set_alignment(self, align: int):
        if align not in (ALIGN_NONE, ALIGN_16, ALIGN_32, ALIGN_64):
            raise ValueError(f"invalid alignment: {align}")
        self._alignment = align

    def set_compression(self, comp: int):
        """Set default compression (COMPRESS_NONE, COMPRESS_ZSTD, COMPRESS_LZ4)."""
        self._compression = comp

    def write_entry(self, name: str, data: bytes):
        self._entries.append(_PendingEntry(name, data, CONTENT_TYPE_UNKNOWN))

    def write_entry_typed(self, name: str, data: bytes, content_type: int):
        self._entries.append(_PendingEntry(name, data, content_type))

    def write_entry_compressed(self, name: str, data: bytes):
        """Write entry using default compression."""
        self._entries.append(
            _PendingEntry(name, data, CONTENT_TYPE_UNKNOWN, compress=True, comp_type=self._compression)
        )

    def write_entry_with_options(self, name: str, data: bytes, compress: bool, comp_type: int):
        """Write entry with explicit compression control."""
        self._entries.append(
            _PendingEntry(name, data, CONTENT_TYPE_UNKNOWN, compress=compress, comp_type=comp_type)
        )

    def close(self):
        """Finalize and write the shard file."""
        n = len(self._entries)
        align = self._alignment

        # Build string table (null-terminated names)
        string_table = bytearray()
        name_offsets: List[int] = []
        for entry in self._entries:
            name_offsets.append(len(string_table))
            string_table.extend(entry.name.encode("utf-8"))
            string_table.append(0)  # null terminator

        # Compute section offsets
        index_size = n * INDEX_ENTRY_SIZE
        string_table_offset = HEADER_SIZE + index_size
        st_len = len(string_table)

        # Data section starts after string table, aligned
        data_section_offset = string_table_offset + st_len
        if align > 0:
            data_section_offset = _align_up(data_section_offset, align)

        # Layout data entries
        index_entries: List[IndexEntry] = []
        current_offset = data_section_offset

        # Precompute disk_data for each entry (applying compression where requested)
        disk_datas: List[bytes] = []
        entry_flags_list: List[int] = []
        for entry in self._entries:
            raw_data = entry.data
            disk_data = raw_data
            entry_flags = 0

            if entry.compress and len(raw_data) >= MIN_COMPRESS_SIZE:
                if entry.comp_type == COMPRESS_ZSTD:
                    cctx = zstd.ZstdCompressor()
                    compressed = cctx.compress(raw_data)
                    if len(compressed) < len(raw_data) * COMPRESSION_SAVINGS_NUM // COMPRESSION_SAVINGS_DEN:
                        disk_data = compressed
                        entry_flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD
                elif entry.comp_type == COMPRESS_LZ4:
                    compressed = lz4.block.compress(raw_data, store_size=False)
                    if len(compressed) < len(raw_data) * COMPRESSION_SAVINGS_NUM // COMPRESSION_SAVINGS_DEN:
                        disk_data = compressed
                        entry_flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4

            disk_datas.append(disk_data)
            entry_flags_list.append(entry_flags)

        for i, entry in enumerate(self._entries):
            if align > 0:
                current_offset = _align_up(current_offset, align)

            raw_data = entry.data
            disk_data = disk_datas[i]
            entry_flags = entry_flags_list[i]
            # Checksum is always computed on the original uncompressed data.
            checksum = compute_crc32c(raw_data)
            name_hash = compute_xxhash64(entry.name)
            name_bytes = entry.name.encode("utf-8")

            ie = IndexEntry(
                name_hash=name_hash,
                name_offset=name_offsets[i],
                name_len=len(name_bytes),
                flags=entry_flags,
                data_offset=current_offset,
                disk_size=len(disk_data),
                orig_size=len(raw_data),
                checksum=checksum,
                reserved=entry.content_type & 0xFFFF,
                name=entry.name,
            )
            index_entries.append(ie)
            current_offset += len(disk_data)

        # If metadata is set, serialize it and append after data section
        meta_bytes: Optional[bytes] = None
        schema_offset = 0
        hdr_flags = DEFAULT_FLAGS

        if self._metadata is not None:
            meta_bytes = self._metadata.to_json()
            schema_offset = current_offset
            hdr_flags = DEFAULT_FLAGS | FLAG_HAS_SCHEMA
            total_file_size = current_offset + len(meta_bytes)
        else:
            total_file_size = current_offset

        # Build header
        header = ShardHeader(
            role=self._role,
            flags=hdr_flags,
            alignment=self._alignment,
            entry_count=n,
            string_table_offset=string_table_offset,
            data_section_offset=data_section_offset,
            schema_offset=schema_offset,
            total_file_size=total_file_size,
        )

        # Write file
        with open(self._path, "wb") as f:
            f.write(header.to_bytes())

            for ie in index_entries:
                f.write(ie.to_bytes())

            f.write(bytes(string_table))

            # Pad to data section
            pos = string_table_offset + st_len
            if pos < data_section_offset:
                f.write(b"\x00" * (data_section_offset - pos))

            # Write data entries with alignment padding (use disk_data for compressed entries)
            for i in range(len(self._entries)):
                pos = f.tell()
                target = index_entries[i].data_offset
                if pos < target:
                    f.write(b"\x00" * (target - pos))
                f.write(disk_datas[i])

            # Write metadata (schema section) if present
            if meta_bytes is not None:
                f.write(meta_bytes)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return (value + alignment - 1) & ~(alignment - 1)


# ============================================================
# Schema Validation
# ============================================================

@dataclass
class EntrySpec:
    """Specification for an expected shard entry."""
    pattern: str
    content_type: int = CONTENT_TYPE_UNKNOWN  # 0 = any
    optional: bool = False
    description: str = ""


@dataclass
class ShardSchema:
    """Schema for validating shard contents."""
    name: str = ""
    version: int = 1
    specs: List[EntrySpec] = field(default_factory=list)

    def add_spec(self, pattern: str, content_type: int = 0, optional: bool = False, description: str = "") -> "ShardSchema":
        """Add a required (or optional) entry spec to the schema."""
        self.specs.append(EntrySpec(pattern, content_type, optional, description))
        return self

    def add_optional(self, pattern: str, content_type: int = 0, description: str = "") -> "ShardSchema":
        """Add an optional entry spec to the schema."""
        return self.add_spec(pattern, content_type, optional=True, description=description)


@dataclass
class ValidationError:
    """A single schema validation failure."""
    spec_pattern: str
    message: str


def _match_pattern(pattern: str, name: str) -> bool:
    """Match a glob pattern against an entry name.

    Supports:
    - Exact: ``"layer.0.weight"``
    - Single component wildcard: ``"layer.*.weight"`` — ``*`` does not cross
      ``.`` or ``/`` separators (matches one path component)
    - Prefix wildcard: ``"prefix.**"`` — matches everything starting with
      the text before ``**``
    """
    if "**" in pattern:
        prefix = pattern.split("**")[0]
        return name.startswith(prefix)
    # fnmatch uses * to match any sequence of characters (including separators).
    # For single-component matching we need * to *not* cross '.' or '/'.
    # Convert the glob pattern to a regex where * becomes [^./]*.
    import re
    regex_parts: List[str] = []
    for ch in pattern:
        if ch == "*":
            regex_parts.append(r"[^./]*")
        elif ch in r"\.+^${}[]|()?":
            regex_parts.append(re.escape(ch))
        else:
            regex_parts.append(ch)
    regex = "^" + "".join(regex_parts) + "$"
    return bool(re.match(regex, name))


def validate_schema(reader: "ShardReader", schema: ShardSchema) -> List[ValidationError]:
    """Validate a shard against a schema.

    Returns a list of :class:`ValidationError` objects (empty list means valid).

    Checks:
    1. Every non-optional spec has at least one matching entry.
    2. Matching entries have the correct ``content_type`` when the spec
       specifies one (non-zero).
    """
    errors: List[ValidationError] = []
    names = reader.entry_names()

    for spec in schema.specs:
        matches = [n for n in names if _match_pattern(spec.pattern, n)]

        if not matches and not spec.optional:
            errors.append(ValidationError(
                spec.pattern,
                f"required entry matching '{spec.pattern}' not found",
            ))
            continue

        if spec.content_type != CONTENT_TYPE_UNKNOWN:
            for name in matches:
                idx = reader.lookup(name)
                info = reader.get_entry_info(idx)
                if info.content_type != spec.content_type:
                    errors.append(ValidationError(
                        spec.pattern,
                        f"entry '{name}' has content_type {info.content_type}, "
                        f"expected {spec.content_type}",
                    ))

    return errors


# ============================================================
# Streaming Writer
# ============================================================

class ShardStreamWriter:
    """Zero-copy streaming writer. Data goes directly to disk.

    The header+index region is written at the END via a seek-back to position 0,
    so no temporary file or in-RAM data buffer is needed. Requires knowing
    ``max_entries`` upfront to reserve the header+index space.

    Usage::

        sw = ShardStreamWriter(path, max_entries=1000)
        sw.set_alignment(ALIGN_64)   # optional, must be before begin_data()
        sw.begin_data()
        sw.write_entry("weights", tensor_bytes)
        sw.write_entry("config",  json_bytes)
        sw.finalize()
    """

    def __init__(
        self,
        path: Union[str, Path],
        role: int = ROLE_MOSH,
        max_entries: int = 10000,
    ):
        self._path = Path(path)
        self._role = role
        self._alignment = ALIGN_64
        self._compression = COMPRESS_NONE
        self._max_entries = max_entries
        self._file = open(self._path, "wb")
        # Each tuple: (name, data_offset, disk_size, orig_size, checksum, flags)
        self._entries: List[Tuple[str, int, int, int, int, int]] = []
        self._started = False
        self._finalized = False
        self._reserved_space = 0
        self._current_offset = 0

    # ------------------------------------------------------------------
    # Configuration (must be called before begin_data)
    # ------------------------------------------------------------------

    def set_alignment(self, align: int):
        """Set alignment. Must be called before begin_data()."""
        if self._started:
            raise RuntimeError("cannot set alignment after begin_data()")
        if align not in (ALIGN_NONE, ALIGN_16, ALIGN_32, ALIGN_64):
            raise ValueError(f"invalid alignment: {align}")
        self._alignment = align

    def set_compression(self, comp: int):
        """Set default compression used by write_entry_compressed()."""
        self._compression = comp

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_data(self):
        """Calculate the reserved header+index space and seek past it.

        After this call, the file position is at the first byte of the data
        region and write_entry() calls may proceed.
        """
        if self._started:
            raise RuntimeError("begin_data() already called")
        n = self._max_entries
        # Estimate: header + index array + avg 64 bytes per entry name
        reserved = HEADER_SIZE + (n * INDEX_ENTRY_SIZE) + (n * 64)
        if self._alignment > 0:
            reserved = _align_up(reserved, self._alignment)
        self._reserved_space = reserved
        self._current_offset = reserved
        self._file.seek(reserved)
        self._started = True

    def write_entry(self, name: str, data: bytes):
        """Write ``data`` at the current file position and record metadata."""
        if not self._started:
            raise RuntimeError("call begin_data() first")
        if self._finalized:
            raise RuntimeError("already finalized")

        # Align the write position if requested
        if self._alignment > 0:
            aligned = _align_up(self._current_offset, self._alignment)
            if aligned > self._current_offset:
                self._file.write(b"\x00" * (aligned - self._current_offset))
                self._current_offset = aligned

        checksum = compute_crc32c(data)
        orig_size = len(data)
        self._file.write(data)
        self._entries.append(
            (name, self._current_offset, len(data), orig_size, checksum, 0)
        )
        self._current_offset += len(data)

    def write_entry_compressed(self, name: str, data: bytes):
        """Write ``data`` using the compression set by set_compression().

        Falls back to uncompressed storage when:
        - data is smaller than MIN_COMPRESS_SIZE, or
        - compressed form is not materially smaller, or
        - the compression library is unavailable.
        """
        if not self._started:
            raise RuntimeError("call begin_data() first")
        if self._finalized:
            raise RuntimeError("already finalized")

        # Align
        if self._alignment > 0:
            aligned = _align_up(self._current_offset, self._alignment)
            if aligned > self._current_offset:
                self._file.write(b"\x00" * (aligned - self._current_offset))
                self._current_offset = aligned

        checksum = compute_crc32c(data)
        orig_size = len(data)
        disk_data = data
        entry_flags = 0

        if self._compression != COMPRESS_NONE and len(data) >= MIN_COMPRESS_SIZE:
            try:
                if self._compression == COMPRESS_ZSTD:
                    cctx = zstd.ZstdCompressor()
                    compressed = cctx.compress(data)
                    if len(compressed) < len(data) * COMPRESSION_SAVINGS_NUM // COMPRESSION_SAVINGS_DEN:
                        disk_data = compressed
                        entry_flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD
                elif self._compression == COMPRESS_LZ4:
                    compressed = lz4.block.compress(data, store_size=False)
                    if len(compressed) < len(data) * COMPRESSION_SAVINGS_NUM // COMPRESSION_SAVINGS_DEN:
                        disk_data = compressed
                        entry_flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4
            except Exception:
                pass  # compression unavailable or failed — store raw

        self._file.write(disk_data)
        self._entries.append(
            (name, self._current_offset, len(disk_data), orig_size, checksum, entry_flags)
        )
        self._current_offset += len(disk_data)

    def finalize(self):
        """Seek to position 0 and write the header, index, and string table.

        Raises ``RuntimeError`` if the actual string table size exceeds the
        space reserved by begin_data().
        """
        if self._finalized:
            raise RuntimeError("already finalized")
        self._finalized = True

        n = len(self._entries)

        # Build string table (null-terminated names)
        string_table = bytearray()
        name_offsets: List[int] = []
        for name, *_ in self._entries:
            name_offsets.append(len(string_table))
            string_table.extend(name.encode("utf-8"))
            string_table.append(0)

        # Verify the front matter fits in the reserved region
        actual_front = HEADER_SIZE + (n * INDEX_ENTRY_SIZE) + len(string_table)
        if self._alignment > 0:
            actual_front_aligned = _align_up(actual_front, self._alignment)
        else:
            actual_front_aligned = actual_front

        if actual_front_aligned > self._reserved_space:
            self._file.close()
            raise RuntimeError(
                f"string table overflow: need {actual_front_aligned} bytes "
                f"but reserved only {self._reserved_space} "
                f"(increase max_entries or use shorter entry names)"
            )

        # Layout constants
        string_table_offset = HEADER_SIZE + n * INDEX_ENTRY_SIZE
        data_section_offset = self._reserved_space
        total_file_size = self._current_offset

        # Build header
        header = ShardHeader(
            role=self._role,
            flags=DEFAULT_FLAGS | FLAG_STREAMING,
            alignment=self._alignment,
            compression_default=self._compression,
            entry_count=n,
            string_table_offset=string_table_offset,
            data_section_offset=data_section_offset,
            total_file_size=total_file_size,
        )

        # Build index entries
        index_entries: List[IndexEntry] = []
        for i, (name, data_offset, disk_size, orig_size, checksum, flags) in enumerate(
            self._entries
        ):
            name_bytes = name.encode("utf-8")
            ie = IndexEntry(
                name_hash=compute_xxhash64(name),
                name_offset=name_offsets[i],
                name_len=len(name_bytes),
                flags=flags,
                data_offset=data_offset,
                disk_size=disk_size,
                orig_size=orig_size,
                checksum=checksum,
                name=name,
            )
            index_entries.append(ie)

        # Seek to the beginning and write the front matter
        self._file.seek(0)
        self._file.write(header.to_bytes())
        for ie in index_entries:
            self._file.write(ie.to_bytes())
        self._file.write(bytes(string_table))

        # Pad to the data section boundary
        pos = string_table_offset + len(string_table)
        if pos < data_section_offset:
            self._file.write(b"\x00" * (data_section_offset - pos))

        self._file.close()

    def close(self):
        """Abort: close the file without finalizing (leaves an incomplete file)."""
        if not self._finalized:
            self._file.close()
