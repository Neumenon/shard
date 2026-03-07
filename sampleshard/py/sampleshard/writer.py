"""
SampleShard writer implementation.

Writes training samples to .smpl files with the Shard v2 format.

Shard v2 layout (matching Go implementation):
    Header (64 bytes)
    Index entries (entry_count × 48 bytes)
    String table (variable)
    [Padding to alignment]
    Data section (aligned entries)
"""

import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .cowrie import encode as cowrie_encode


from .types import (
    ShardHeader,
    IndexEntry,
    ShardRole,
    SHARD_V2_HEADER_SIZE,
    SHARD_V2_INDEX_ENTRY_SIZE,
    ALIGN_64,
    COMPRESS_NONE,
    xxhash64,
    crc32c,
)


class PendingEntry:
    """Metadata for an entry being written."""

    __slots__ = ("name", "temp_offset", "disk_size", "orig_size", "checksum", "flags")

    def __init__(
        self,
        name: str,
        temp_offset: int,
        disk_size: int,
        orig_size: int,
        checksum: int,
        flags: int = 0,
    ):
        self.name = name
        self.temp_offset = temp_offset
        self.disk_size = disk_size
        self.orig_size = orig_size
        self.checksum = checksum
        self.flags = flags


class SampleShardWriter:
    """
    Writer for SampleShard (.smpl) files.

    SampleShard uses the Shard v2 format with Role=Sample (0x02).
    Each sample is stored as a Cowrie-encoded blob with its ID as the entry name.

    Example:
        >>> with SampleShardWriter("train.smpl") as w:
        ...     w.add_sample(1, {"input": [1, 2, 3], "label": 0})
        ...     w.add_sample(2, {"input": [4, 5, 6], "label": 1})
    """

    def __init__(
        self,
        path: Union[str, Path],
        alignment: int = ALIGN_64,
        compression: int = COMPRESS_NONE,
    ):
        """
        Create a new SampleShard writer.

        Args:
            path: Output file path (.smpl extension recommended)
            alignment: Data alignment (0, 16, 32, or 64 bytes)
            compression: Default compression (0=none, 1=zstd, 2=lz4)
        """
        self.path = Path(path)
        self.alignment = alignment
        self.compression = compression

        self._file = None
        self._temp_file = None
        self._temp_offset = 0
        self._entries: List[PendingEntry] = []
        self._closed = False

    def __enter__(self) -> "SampleShardWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        """Open the file for writing."""
        if self._file is not None:
            return

        self._file = open(self.path, "wb")

        # Create temp file for buffering data (same approach as Go)
        self._temp_file = tempfile.NamedTemporaryFile(
            delete=False, prefix="shard_data_"
        )
        self._temp_offset = 0

    def add_sample(self, sample_id: int, sample: Any) -> None:
        """
        Add a sample to the shard.

        Args:
            sample_id: Unique sample identifier (uint64)
            sample: Sample data (must be JSON-serializable)
        """
        if self._file is None:
            raise RuntimeError("Writer not open")
        if self._closed:
            raise RuntimeError("Writer is closed")

        # Use sample ID as entry name
        name = str(sample_id)

        # Encode sample as Cowrie binary format (byte-identical with Go)
        data = cowrie_encode(sample)

        self._write_entry(name, data)

    def add_sample_raw(self, sample_id: int, data: bytes) -> None:
        """
        Add pre-encoded sample bytes.

        Args:
            sample_id: Unique sample identifier
            data: Pre-encoded sample data (e.g., already Cowrie-encoded)
        """
        if self._file is None:
            raise RuntimeError("Writer not open")
        if self._closed:
            raise RuntimeError("Writer is closed")

        name = str(sample_id)
        self._write_entry(name, data)

    def _write_entry(self, name: str, data: bytes) -> None:
        """Write an entry to the temp file (data buffered for later)."""
        # Record temp file offset before writing
        temp_offset = self._temp_offset

        # Write to temp file
        self._temp_file.write(data)
        self._temp_offset += len(data)

        # Store entry metadata (data stays in temp file)
        entry = PendingEntry(
            name=name,
            temp_offset=temp_offset,
            disk_size=len(data),
            orig_size=len(data),
            checksum=crc32c(data),
            flags=0,  # No compression for now
        )
        self._entries.append(entry)

    def close(self) -> None:
        """Finalize and close the shard file.

        Layout: Header + Index + StringTable + [Padding] + Data
        """
        if self._file is None or self._closed:
            return

        self._closed = True

        try:
            self._finalize()
        finally:
            # Cleanup temp file
            if self._temp_file is not None:
                temp_name = self._temp_file.name
                self._temp_file.close()
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                self._temp_file = None

    def _finalize(self) -> None:
        """Write the final shard file in v2 format."""
        entry_count = len(self._entries)
        index_size = entry_count * SHARD_V2_INDEX_ENTRY_SIZE

        # Build string table
        string_table = bytearray()
        name_offsets: Dict[str, int] = {}
        for entry in self._entries:
            name_offsets[entry.name] = len(string_table)
            string_table.extend(entry.name.encode("utf-8"))
            string_table.append(0)  # null terminator

        # Calculate offsets
        string_table_offset = SHARD_V2_HEADER_SIZE + index_size
        data_section_offset = string_table_offset + len(string_table)

        # Align data section
        if self.alignment > 0:
            align = self.alignment
            data_section_offset = (data_section_offset + align - 1) & ~(align - 1)

        # Calculate entry data offsets in final file
        current_data_offset = data_section_offset
        data_offsets: List[int] = []
        for entry in self._entries:
            # Align each entry
            if self.alignment > 0:
                align = self.alignment
                current_data_offset = (current_data_offset + align - 1) & ~(align - 1)
            data_offsets.append(current_data_offset)
            current_data_offset += entry.disk_size

        total_size = current_data_offset

        # Write header
        header = ShardHeader(
            role=ShardRole.SAMPLE,
            alignment=self.alignment,
            compression_default=self.compression,
            entry_count=entry_count,
            string_table_offset=string_table_offset,
            data_section_offset=data_section_offset,
            schema_offset=0,
            total_file_size=total_size,
        )
        self._file.write(header.to_bytes())

        # Write index entries
        for i, entry in enumerate(self._entries):
            name_bytes = entry.name.encode("utf-8")
            index_entry = IndexEntry(
                name_hash=xxhash64(name_bytes),
                name_offset=name_offsets[entry.name],
                name_len=len(name_bytes),
                flags=entry.flags,
                data_offset=data_offsets[i],
                disk_size=entry.disk_size,
                orig_size=entry.orig_size,
                checksum=entry.checksum,
                name=entry.name,
            )
            self._file.write(index_entry.to_bytes())

        # Write string table
        self._file.write(bytes(string_table))

        # Write padding to data section
        current_pos = string_table_offset + len(string_table)
        if current_pos < data_section_offset:
            padding = data_section_offset - current_pos
            self._file.write(b"\x00" * padding)

        # Write data entries from temp file with alignment padding
        current_pos = data_section_offset
        self._temp_file.seek(0)

        for i, entry in enumerate(self._entries):
            # Alignment padding
            expected_offset = data_offsets[i]
            if current_pos < expected_offset:
                padding = expected_offset - current_pos
                self._file.write(b"\x00" * padding)
                current_pos = expected_offset

            # Read from temp and write to final file
            self._temp_file.seek(entry.temp_offset)
            entry_data = self._temp_file.read(entry.disk_size)
            self._file.write(entry_data)
            current_pos += entry.disk_size

        self._file.close()
        self._file = None
