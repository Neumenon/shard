"""
Tests for metadata/schema, security limits, list_children, read_entry_prefix,
and path helpers added to shard_v2.
"""
import pytest
import struct
import tempfile
import os
from pathlib import Path

from shard_v2 import (
    ShardV2Reader,
    ShardV2Writer,
    ShardV2Header,
    ShardMetadata,
    EntryMeta,
    SampleProfile,
    ManifestProfile,
    ManifestFileRef,
    FLAG_HAS_SCHEMA,
    MAX_ENTRY_COUNT,
    MAX_INDEX_SIZE,
    MAX_STRING_TABLE_SIZE,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    split_path,
    join_path,
    path_parent,
    path_base,
)


# ============================================================
# Helpers
# ============================================================

def make_shard(entries, role=1, metadata=None):
    """Write a shard to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".shard", delete=False)
    tmp.close()
    w = ShardV2Writer(tmp.name, role=role)
    for name, data in entries:
        w.write_entry(name, data)
    if metadata is not None:
        w.set_metadata(metadata)
    w.close()
    return tmp.name


# ============================================================
# Path helpers
# ============================================================

class TestPathHelpers:
    def test_split_path_simple(self):
        assert split_path("layer.0/weight") == ["layer.0", "weight"]

    def test_split_path_deep(self):
        assert split_path("a/b/c/d") == ["a", "b", "c", "d"]

    def test_split_path_no_slash(self):
        assert split_path("embed") == ["embed"]

    def test_split_path_empty(self):
        assert split_path("") == []

    def test_join_path(self):
        assert join_path("layer.0", "weight") == "layer.0/weight"

    def test_join_path_single(self):
        assert join_path("embed") == "embed"

    def test_join_path_multi(self):
        assert join_path("a", "b", "c") == "a/b/c"

    def test_path_parent_with_slash(self):
        assert path_parent("layer.0/weight") == "layer.0"

    def test_path_parent_deep(self):
        assert path_parent("a/b/c") == "a/b"

    def test_path_parent_no_slash(self):
        assert path_parent("embed") == ""

    def test_path_base_with_slash(self):
        assert path_base("layer.0/weight") == "weight"

    def test_path_base_deep(self):
        assert path_base("a/b/c") == "c"

    def test_path_base_no_slash(self):
        assert path_base("embed") == "embed"


# ============================================================
# ListChildren
# ============================================================

class TestListChildren:
    def setup_method(self):
        entries = [
            ("layer.0/weight", b"w0"),
            ("layer.0/bias", b"b0"),
            ("layer.1/weight", b"w1"),
            ("embed", b"tok"),
        ]
        self._path = make_shard(entries)

    def teardown_method(self):
        os.unlink(self._path)

    def test_list_children_exact_prefix(self):
        r = ShardV2Reader(self._path)
        result = r.list_children("layer.0/")
        assert sorted(result) == ["bias", "weight"]

    def test_list_children_empty_prefix(self):
        r = ShardV2Reader(self._path)
        result = r.list_children("")
        # Should return deduplicated bare first components
        assert "layer.0" in result
        assert "layer.1" in result
        assert "embed" in result
        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_list_children_partial_prefix(self):
        r = ShardV2Reader(self._path)
        result = r.list_children("layer.")
        assert sorted(result) == ["0", "1"]

    def test_list_children_nonexistent_prefix(self):
        r = ShardV2Reader(self._path)
        result = r.list_children("nonexistent/")
        assert result == []

    def test_list_children_leaf_entry(self):
        r = ShardV2Reader(self._path)
        result = r.list_children("embed")
        # "embed" is an exact leaf match; remainder is "" which is filtered out
        assert result == []

    def test_list_children_dedup(self):
        """Multiple entries under same child dir should not produce duplicates."""
        r = ShardV2Reader(self._path)
        result = r.list_children("layer.")
        # bare component "0" appears only once even though there are 2 entries under layer.0/
        assert result.count("0") == 1

    def test_list_children_hierarchical(self):
        path = make_shard([
            ("a/b/c", b"1"),
            ("a/b/d", b"2"),
            ("a/e", b"3"),
            ("f", b"4"),
        ])
        try:
            r = ShardV2Reader(path)
            # Top-level children: bare components, no trailing slash
            top = r.list_children("")
            assert "a" in top
            assert "f" in top
            # Children under a/: bare components after the "a/" prefix
            under_a = r.list_children("a/")
            assert "b" in under_a
            assert "e" in under_a
            # Children under a/b/: bare remainders after "a/b/" prefix
            under_ab = r.list_children("a/b/")
            assert sorted(under_ab) == ["c", "d"]
        finally:
            os.unlink(path)


# ============================================================
# ReadEntryPrefix
# ============================================================

class TestReadEntryPrefix:
    def setup_method(self):
        self._data = b"Hello World, this is a test payload!"
        self._path = make_shard([("entry", self._data)])

    def teardown_method(self):
        os.unlink(self._path)

    def test_read_prefix_exact(self):
        r = ShardV2Reader(self._path)
        result = r.read_entry_prefix(0, len(self._data))
        assert result == self._data

    def test_read_prefix_partial(self):
        r = ShardV2Reader(self._path)
        result = r.read_entry_prefix(0, 5)
        assert result == b"Hello"

    def test_read_prefix_zero(self):
        r = ShardV2Reader(self._path)
        result = r.read_entry_prefix(0, 0)
        assert result == b""

    def test_read_prefix_larger_than_entry(self):
        """Requesting more bytes than entry size returns full entry."""
        r = ShardV2Reader(self._path)
        result = r.read_entry_prefix(0, 10_000)
        assert result == self._data

    def test_read_prefix_one_byte(self):
        r = ShardV2Reader(self._path)
        result = r.read_entry_prefix(0, 1)
        assert result == b"H"


# ============================================================
# ShardMetadata roundtrip
# ============================================================

class TestShardMetadata:
    def test_basic_metadata_roundtrip(self):
        meta = ShardMetadata(
            schema_version="shard-v2.1",
            producer="test-producer",
            description="A test shard",
            created_at="2026-02-21T00:00:00Z",
            tags=["ml", "test"],
        )
        path = make_shard([("data", b"hello")], metadata=meta)
        try:
            r = ShardV2Reader(path)
            h = r.header()
            # FLAG_HAS_SCHEMA should be set
            assert h.flags & FLAG_HAS_SCHEMA
            # schema_offset should point after data section
            assert h.schema_offset > 0

            got = r.read_metadata()
            assert got is not None
            assert got.producer == "test-producer"
            assert got.description == "A test shard"
            assert got.created_at == "2026-02-21T00:00:00Z"
            assert got.tags == ["ml", "test"]
            assert got.schema_version == "shard-v2.1"
        finally:
            os.unlink(path)

    def test_metadata_with_entry_metadata(self):
        em = EntryMeta(
            content_type="tensor/float32",
            tags=["weight"],
            description="Model weights",
            extra={"shape": [256, 256]},
            codec="cowrie-gen2",
            codec_version="2",
            schema_fingerprint="sha256:weights",
            semantic_type="tensor",
            canonical_hash="sha256:canonical",
            base_hash="sha256:base",
            row_count=256,
            shape=[256, 256],
            stats={"min": -1.0, "max": 1.0},
        )
        meta = ShardMetadata(
            producer="pytorch-exporter",
            entry_metadata={"model/weight": em},
        )
        path = make_shard([("model/weight", b"\x00" * 16)], metadata=meta)
        try:
            r = ShardV2Reader(path)
            got = r.read_metadata()
            assert got is not None
            assert "model/weight" in got.entry_metadata
            em2 = got.entry_metadata["model/weight"]
            assert em2.content_type == "tensor/float32"
            assert em2.tags == ["weight"]
            assert em2.description == "Model weights"
            assert em2.extra == {"shape": [256, 256]}
            assert em2.codec == "cowrie-gen2"
            assert em2.codec_version == "2"
            assert em2.schema_fingerprint == "sha256:weights"
            assert em2.semantic_type == "tensor"
            assert em2.canonical_hash == "sha256:canonical"
            assert em2.base_hash == "sha256:base"
            assert em2.row_count == 256
            assert em2.shape == [256, 256]
            assert em2.stats == {"min": -1.0, "max": 1.0}
        finally:
            os.unlink(path)

    def test_metadata_with_profiles(self):
        meta = ShardMetadata(
            profile="sampleshard.v1",
            sample_shard=SampleProfile(
                dataset_name="mnist-train",
                sample_id_type="uint64",
                key_encoding="decimal-string",
                sample_count=60000,
                dataset_schema={"input": "tensor[u8,28,28]", "target": "uint8"},
                splits={"train": {"start": 0, "end": 59999}},
                label_map={"0": "zero", "1": "one"},
                feature_stats={"input": {"mean": 0.1307, "std": 0.3081}},
            ),
            manifest=ManifestProfile(
                files=[
                    ManifestFileRef(
                        uri="s3://bucket/train-000.smpl",
                        sha256="abc123",
                        role="sample",
                        profile="sampleshard.v1",
                        start_key="0",
                        end_key="59999",
                        entry_count=60000,
                    )
                ],
                partitions={"train": ["train-000.smpl"]},
            ),
        )
        path = make_shard([("sample/0", b"payload")], metadata=meta)
        try:
            r = ShardV2Reader(path)
            got = r.read_metadata()
            assert got is not None
            assert got.profile == "sampleshard.v1"
            assert got.sample_shard is not None
            assert got.sample_shard.dataset_name == "mnist-train"
            assert got.sample_shard.sample_id_type == "uint64"
            assert got.sample_shard.key_encoding == "decimal-string"
            assert got.sample_shard.sample_count == 60000
            assert got.sample_shard.label_map["0"] == "zero"
            assert got.manifest is not None
            assert len(got.manifest.files) == 1
            assert got.manifest.files[0].uri == "s3://bucket/train-000.smpl"
            assert got.manifest.partitions["train"] == ["train-000.smpl"]
        finally:
            os.unlink(path)

    def test_no_metadata_returns_none(self):
        path = make_shard([("data", b"hello")])
        try:
            r = ShardV2Reader(path)
            h = r.header()
            assert h.schema_offset == 0
            assert r.read_metadata() is None
        finally:
            os.unlink(path)

    def test_metadata_extra_fields(self):
        meta = ShardMetadata(
            extra={"model_type": "transformer", "num_layers": 12},
        )
        path = make_shard([("cfg", b"x")], metadata=meta)
        try:
            r = ShardV2Reader(path)
            got = r.read_metadata()
            assert got is not None
            assert got.extra["model_type"] == "transformer"
            assert got.extra["num_layers"] == 12
        finally:
            os.unlink(path)

    def test_metadata_source_uri(self):
        meta = ShardMetadata(
            source_uri="s3://bucket/model.safetensors",
            schema_uri="https://example.com/schema.json",
        )
        path = make_shard([("x", b"y")], metadata=meta)
        try:
            r = ShardV2Reader(path)
            got = r.read_metadata()
            assert got is not None
            assert got.source_uri == "s3://bucket/model.safetensors"
            assert got.schema_uri == "https://example.com/schema.json"
        finally:
            os.unlink(path)

    def test_empty_metadata(self):
        meta = ShardMetadata()
        path = make_shard([("x", b"y")], metadata=meta)
        try:
            r = ShardV2Reader(path)
            got = r.read_metadata()
            assert got is not None
            assert got.schema_version == "shard-v2.1"
            assert got.producer == ""
            assert got.tags == []
        finally:
            os.unlink(path)


# ============================================================
# Security limit violations
# ============================================================

class TestSecurityLimits:
    def _make_corrupt_header(self, entry_count=0, total_file_size=None,
                             string_table_size=0, extra_bytes=b""):
        """Build a minimal valid shard binary but with manipulated fields."""
        n = entry_count
        index_size = n * INDEX_ENTRY_SIZE
        st_offset = HEADER_SIZE + index_size
        ds_offset = st_offset + string_table_size
        tfs = ds_offset if total_file_size is None else total_file_size

        buf = bytearray(HEADER_SIZE)
        buf[0:4] = b"SHRD"
        buf[4] = 0x02       # version
        buf[5] = 1          # role
        struct.pack_into("<H", buf, 6, 0x00A2)   # flags
        buf[8] = 64         # alignment
        buf[9] = 0          # compression
        struct.pack_into("<H", buf, 10, INDEX_ENTRY_SIZE)
        struct.pack_into("<I", buf, 12, n)
        struct.pack_into("<Q", buf, 16, st_offset)
        struct.pack_into("<Q", buf, 24, ds_offset)
        struct.pack_into("<Q", buf, 32, 0)  # schema_offset
        struct.pack_into("<Q", buf, 40, tfs)
        return bytes(buf) + extra_bytes

    def test_entry_count_too_large(self):
        """entry_count exceeding MAX_ENTRY_COUNT should raise ValueError."""
        # Build a header claiming 10_000_001 entries (but file is tiny)
        too_many = MAX_ENTRY_COUNT + 1
        # total_file_size must equal buffer length for validation to even reach entry_count check
        # We'll use total_file_size == HEADER_SIZE so validation hits entry_count first
        data = self._make_corrupt_header(
            entry_count=too_many,
            total_file_size=HEADER_SIZE,
        )
        with tempfile.NamedTemporaryFile(suffix=".shard", delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="MAX_ENTRY_COUNT"):
                ShardV2Reader(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_total_file_size_mismatch(self):
        """total_file_size != actual file size should raise ValueError."""
        # Write a valid small shard first
        path = make_shard([("x", b"hello")])
        try:
            # Corrupt the total_file_size field
            data = bytearray(Path(path).read_bytes())
            struct.pack_into("<Q", data, 40, 99999)  # wrong size
            Path(path).write_bytes(bytes(data))
            with pytest.raises(ValueError, match="total_file_size"):
                ShardV2Reader(path)
        finally:
            os.unlink(path)

    def test_entry_offset_out_of_bounds(self):
        """data_offset + disk_size past end of file should raise ValueError."""
        path = make_shard([("x", b"hello")])
        try:
            data = bytearray(Path(path).read_bytes())
            # Find the data_offset field of entry 0 and set it to a huge value
            # Entry 0 is at offset HEADER_SIZE; data_offset is at +16 within entry
            entry_off = HEADER_SIZE + 16
            struct.pack_into("<Q", data, entry_off, 0xFFFFFFFF)
            # Keep total_file_size consistent with actual data length
            struct.pack_into("<Q", data, 40, len(data))
            Path(path).write_bytes(bytes(data))
            with pytest.raises(ValueError):
                ShardV2Reader(path)
        finally:
            os.unlink(path)

    def test_entry_offset_before_data_section(self):
        """data_offset < data_section_offset should raise ValueError."""
        path = make_shard([("x", b"hello")])
        try:
            data = bytearray(Path(path).read_bytes())
            # Set data_offset of entry 0 to 0 (before header)
            entry_off = HEADER_SIZE + 16
            struct.pack_into("<Q", data, entry_off, 0)
            struct.pack_into("<Q", data, 40, len(data))
            Path(path).write_bytes(bytes(data))
            with pytest.raises(ValueError):
                ShardV2Reader(path)
        finally:
            os.unlink(path)
