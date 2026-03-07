"""
SampleShard type definitions.

This module defines the core data structures for the SampleShard format:
- ShardHeader: 64-byte v2 header
- IndexEntry: 48-byte index entry
- ShardRole: Role enumeration
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import hashlib

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


class ShardRole(IntEnum):
    """Shard role/profile type."""

    UNKNOWN = 0x00
    UMSH = 0x01  # Model weights
    SAMPLE = 0x02  # Training samples (SampleShard)
    GEMM_PANEL = 0x03  # GEMM panels


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
        """Get compression type (0=none, 1=zstd, 2=lz4)."""
        return (self.flags >> 4) & 0x0F


# xxHash64: Try to use fast xxhash library, fall back to SHA256
try:
    import xxhash as _xxhash

    def xxhash64(data: bytes) -> int:
        """Compute xxHash64 of data (using fast xxhash library)."""
        return _xxhash.xxh64(data).intdigest()
except ImportError:

    def xxhash64(data: bytes) -> int:
        """Compute xxHash64 of data (fallback to SHA256-based hash)."""
        h = hashlib.sha256(data).digest()
        return struct.unpack("<Q", h[:8])[0]


# CRC32C: Try to use fast crc32c library, fall back to zlib
try:
    import crc32c as _crc32c_lib

    def crc32c(data: bytes) -> int:
        """Compute CRC32C checksum (using fast crc32c library)."""
        return _crc32c_lib.crc32c(data)
except ImportError:
    import zlib as _zlib

    def crc32c(data: bytes) -> int:
        """Compute CRC32C checksum (fallback to zlib.crc32)."""
        return _zlib.crc32(data) & 0xFFFFFFFF
