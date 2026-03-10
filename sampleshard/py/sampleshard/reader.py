"""
SampleShard reader implementation.

Reads training samples from .smpl files with the Shard v2 format.
"""

import mmap
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .cowrie import decode as cowrie_decode, MAGIC as COWRIE_MAGIC

# Try to use orjson (10-20x faster), fall back to stdlib json for legacy files
try:
    import orjson

    def json_loads(data: bytes) -> Any:
        return orjson.loads(data)
except ImportError:
    try:
        import ujson

        def json_loads(data: bytes) -> Any:
            return ujson.loads(data)
    except ImportError:
        import json

        def json_loads(data: bytes) -> Any:
            return json.loads(data)


def decode_sample(data: bytes) -> Any:
    """
    Decode sample data, auto-detecting Cowrie vs JSON format.

    Cowrie files start with 'SJ' magic bytes.
    """
    if len(data) >= 2 and data[:2] == COWRIE_MAGIC:
        return cowrie_decode(data)
    return json_loads(data)


from .types import (
    ShardHeader,
    IndexEntry,
    ShardRole,
    ShardMetadata,
    SampleProfile,
    ManifestProfile,
    SHARD_V2_HEADER_SIZE,
    SHARD_V2_INDEX_ENTRY_SIZE,
    crc32c,
)


class SampleShardReader:
    """
    Reader for SampleShard (.smpl) files.

    SampleShard uses the Shard v2 format with Role=Sample (0x02).
    Supports random access by sample ID and sequential iteration.

    Example:
        >>> with SampleShardReader("train.smpl") as r:
        ...     print(r.sample_count())  # Number of samples
        ...     sample = r.get_sample(1)  # Get by ID
        ...     for sample_id, sample in r:
        ...         print(sample_id, sample)
    """

    def __init__(self, path: Union[str, Path]):
        """
        Open a SampleShard file for reading.

        Args:
            path: Path to .smpl file
        """
        self.path = Path(path)
        self._file = None
        self._mmap: Optional[mmap.mmap] = None
        self._header: Optional[ShardHeader] = None
        self._entries: List[IndexEntry] = []
        self._entry_by_name: Dict[str, IndexEntry] = {}  # name -> entry (O(1) lookup)
        self._id_map: Dict[int, int] = {}  # sample_id -> index in _sample_ids
        self._sample_ids: List[int] = []  # ordered sample IDs
        self._metadata: Optional[ShardMetadata] = None
        self._closed = False

    def __enter__(self) -> "SampleShardReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __iter__(self) -> Iterator[Tuple[int, Any]]:
        """Iterate over all samples in ascending ID order."""
        for i in range(len(self._sample_ids)):
            sample_id = self._sample_ids[i]
            sample = self.get_sample_by_index(i)
            yield sample_id, sample

    def open(self) -> None:
        """Open the file for reading.

        Shard v2 layout (matching Go implementation):
            Header (64 bytes)
            Index entries (entry_count × 48 bytes)
            String table (variable)
            [Padding to alignment]
            Data section (aligned entries)
        """
        if self._file is not None:
            return

        self._file = open(self.path, "rb")

        # Read header
        header_data = self._file.read(SHARD_V2_HEADER_SIZE)
        self._header = ShardHeader.from_bytes(header_data)

        # Verify role
        if self._header.role != ShardRole.SAMPLE:
            self.close()
            raise ValueError(
                f"Expected Sample role (0x{ShardRole.SAMPLE:02x}), "
                f"got 0x{self._header.role:02x}"
            )

        # v2 format: Index is at offset 64 (right after header)
        # Read index entries first
        index_size = self._header.entry_count * SHARD_V2_INDEX_ENTRY_SIZE
        index_data = self._file.read(index_size)

        for i in range(self._header.entry_count):
            offset = i * SHARD_V2_INDEX_ENTRY_SIZE
            entry_data = index_data[offset : offset + SHARD_V2_INDEX_ENTRY_SIZE]
            entry = IndexEntry.from_bytes(entry_data)
            self._entries.append(entry)

        # Read string table (between index and data section)
        # string_table is at position: header + index
        # string_table ends at: data_section_offset
        string_table_offset = self._header.string_table_offset
        string_table_size = self._header.data_section_offset - string_table_offset

        self._file.seek(string_table_offset)
        string_table = self._file.read(string_table_size)

        # Extract names from string table
        for entry in self._entries:
            name_end = entry.name_offset + entry.name_len
            if name_end <= len(string_table):
                entry.name = string_table[entry.name_offset : name_end].decode("utf-8")
            self._entry_by_name[entry.name] = entry  # O(1) lookup

        # Build sample index (exclude metadata entries starting with __)
        for entry in self._entries:
            if entry.name.startswith("__"):
                continue  # Skip metadata entries

            try:
                sample_id = int(entry.name)
                self._sample_ids.append(sample_id)
            except ValueError:
                # Not a numeric ID, skip
                continue

        # Sort by ascending sample ID for deterministic iteration order
        self._sample_ids.sort()

        # Build ID -> sorted index map
        self._id_map = {sid: i for i, sid in enumerate(self._sample_ids)}

    def enable_mmap(self) -> None:
        """Enable memory-mapped access for zero-copy reads."""
        if self._mmap is not None:
            return
        if self._file is None:
            raise RuntimeError("Reader not open")

        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )

    def sample_count(self) -> int:
        """Return the number of samples (excludes metadata entries)."""
        return len(self._sample_ids)

    def sample_ids(self) -> List[int]:
        """Return all sample IDs in ascending order."""
        return list(self._sample_ids)

    def sample_id_by_index(self, index: int) -> int:
        """Return the sample ID at the given index."""
        if index < 0 or index >= len(self._sample_ids):
            raise IndexError(f"Sample index out of range: {index}")
        return self._sample_ids[index]

    def read_metadata(self) -> Optional[ShardMetadata]:
        """Read shard metadata from schema_offset."""
        if self._metadata is not None:
            return self._metadata
        if self._header is None:
            raise RuntimeError("Reader not open")
        if self._header.schema_offset == 0:
            return None

        total_size = self._header.total_file_size
        schema_offset = self._header.schema_offset
        if schema_offset > total_size:
            raise ValueError(
                f"schema_offset {schema_offset} exceeds total_file_size {total_size}"
            )

        if self._mmap is not None:
            raw = self._mmap[schema_offset:total_size]
        else:
            self._file.seek(schema_offset)
            raw = self._file.read(total_size - schema_offset)

        self._metadata = ShardMetadata.from_json(bytes(raw))
        return self._metadata

    def sample_profile(self) -> Optional[SampleProfile]:
        """Return the standardized SampleShard profile metadata."""
        meta = self.read_metadata()
        if meta is None:
            return None
        return meta.sample_shard

    def manifest_profile(self) -> Optional[ManifestProfile]:
        """Return the manifest profile metadata if present."""
        meta = self.read_metadata()
        if meta is None:
            return None
        return meta.manifest

    def has_sample(self, sample_id: int) -> bool:
        """Check if a sample exists by ID."""
        return sample_id in self._id_map

    def get_sample(self, sample_id: int) -> Any:
        """
        Get a sample by ID. O(1) lookup.

        Args:
            sample_id: Sample identifier

        Returns:
            Decoded sample data

        Raises:
            KeyError: If sample not found
        """
        name = str(sample_id)
        entry = self._entry_by_name.get(name)
        if entry is None:
            raise KeyError(f"Sample not found: {sample_id}")
        return self._read_entry(entry)

    def get_sample_by_index(self, index: int) -> Any:
        """
        Get a sample by its position in the sorted sample index.

        Args:
            index: Position in the sorted sample list (0-based)

        Returns:
            Decoded sample data
        """
        if index < 0 or index >= len(self._sample_ids):
            raise IndexError(f"Sample index out of range: {index}")

        sample_id = self._sample_ids[index]
        return self.get_sample(sample_id)

    def _read_entry(self, entry: IndexEntry) -> Any:
        """Read and decode an entry."""
        if self._mmap is not None:
            data = self._mmap[entry.data_offset : entry.data_offset + entry.disk_size]
        else:
            self._file.seek(entry.data_offset)
            data = self._file.read(entry.disk_size)

        # Decompress if needed (before CRC check — Go checks CRC on decompressed data)
        if entry.is_compressed():
            comp_type = entry.compression_type()
            if comp_type == 1:  # zstd
                try:
                    import zstandard

                    data = zstandard.decompress(data)
                except ImportError:
                    raise RuntimeError(
                        "zstandard library required: pip install zstandard"
                    )
            elif comp_type == 2:  # lz4
                try:
                    import lz4.block

                    data = lz4.block.decompress(
                        data, uncompressed_size=entry.orig_size
                    )
                except ImportError:
                    raise RuntimeError("lz4 library required: pip install lz4")

        # Verify checksum on decompressed data (matches Go reference)
        actual_crc = crc32c(data)
        if actual_crc != entry.checksum:
            raise ValueError(
                f"Checksum mismatch for entry '{entry.name}': "
                f"expected 0x{entry.checksum:08x}, got 0x{actual_crc:08x}"
            )

        # Decode sample (auto-detects Cowrie vs JSON format)
        return decode_sample(data)

    def get_batch(self, sample_ids: List[int]) -> List[Any]:
        """
        Get multiple samples by their IDs.

        Args:
            sample_ids: List of sample identifiers

        Returns:
            List of decoded samples in the same order
        """
        return [self.get_sample(sid) for sid in sample_ids]

    def get_batch_by_range(self, start: int, end: int) -> List[Any]:
        """
        Get a range of samples by index [start, end).

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            List of decoded samples
        """
        start = max(0, start)
        end = min(len(self._sample_ids), end)
        return [self.get_sample_by_index(i) for i in range(start, end)]

    def close(self) -> None:
        """Close the reader."""
        if self._closed:
            return

        self._closed = True

        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

        if self._file is not None:
            self._file.close()
            self._file = None
