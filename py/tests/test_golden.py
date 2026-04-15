"""Golden file parity tests — verify Python reads Go-generated shards exactly."""
import hashlib
import json
from pathlib import Path

import pytest

from shard_format import (
    ShardReader,
    compute_crc32c,
    compute_xxhash64,
)

TESTDATA = Path(__file__).resolve().parent.parent.parent / "testdata"
MANIFEST = TESTDATA / "golden_manifest.json"


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST) as f:
        return json.load(f)


class TestGoldenFiles:
    """Verify every golden shard file against the manifest."""

    def test_manifest_exists(self):
        assert MANIFEST.exists(), f"manifest not found at {MANIFEST}"

    def test_all_golden_files_exist(self, manifest):
        for gf in manifest["files"]:
            path = TESTDATA / gf["filename"]
            assert path.exists(), f"golden file missing: {path}"

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_file_sha256(self, manifest, file_index):
        gf = manifest["files"][file_index]
        path = TESTDATA / gf["filename"]
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        assert actual == gf["sha256"], (
            f"{gf['filename']}: SHA256 mismatch\n"
            f"  expected: {gf['sha256']}\n"
            f"  actual:   {actual}"
        )

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_header_fields(self, manifest, file_index):
        gf = manifest["files"][file_index]
        path = TESTDATA / gf["filename"]
        reader = ShardReader(path)
        h = reader.header()
        gh = gf["header"]

        assert h.version == gh["version"]
        assert h.role == gh["role"]
        assert h.flags == gh["flags"]
        assert h.alignment == gh["alignment"]
        assert h.compression_default == gh["compression_default"]
        assert h.entry_count == gh["entry_count"]
        assert h.index_entry_size == gh["index_entry_size"]
        assert h.string_table_offset == gh["string_table_offset"]
        assert h.data_section_offset == gh["data_section_offset"]
        assert h.schema_offset == gh["schema_offset"]
        assert h.total_file_size == gh["total_file_size"]

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_entry_count(self, manifest, file_index):
        gf = manifest["files"][file_index]
        reader = ShardReader(TESTDATA / gf["filename"])
        assert reader.entry_count() == len(gf["entries"])

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_entry_names(self, manifest, file_index):
        gf = manifest["files"][file_index]
        reader = ShardReader(TESTDATA / gf["filename"])
        expected = [e["name"] for e in gf["entries"]]
        assert reader.entry_names() == expected

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_entry_details(self, manifest, file_index):
        gf = manifest["files"][file_index]
        reader = ShardReader(TESTDATA / gf["filename"])

        for i, ge in enumerate(gf["entries"]):
            info = reader.get_entry_info(i)
            assert info.name == ge["name"], f"entry {i} name"
            assert info.name_hash == ge["name_hash"], f"entry {i} name_hash"
            assert info.content_type == ge["content_type"], f"entry {i} content_type"
            assert info.orig_size == ge["orig_size"], f"entry {i} orig_size"
            assert info.disk_size == ge["disk_size"], f"entry {i} disk_size"
            assert info.checksum == ge["checksum"], f"entry {i} checksum"
            assert info.compressed == ge["compressed"], f"entry {i} compressed"

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_entry_data_sha256(self, manifest, file_index):
        gf = manifest["files"][file_index]
        reader = ShardReader(TESTDATA / gf["filename"])

        for i, ge in enumerate(gf["entries"]):
            data = reader.read_entry(i)
            actual = hashlib.sha256(data).hexdigest()
            assert actual == ge["data_sha256"], (
                f"entry {i} ({ge['name']}): data SHA256 mismatch"
            )

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_lookup_by_name(self, manifest, file_index):
        gf = manifest["files"][file_index]
        reader = ShardReader(TESTDATA / gf["filename"])

        for i, ge in enumerate(gf["entries"]):
            idx = reader.lookup(ge["name"])
            assert idx == i, f"lookup({ge['name']!r}) = {idx}, expected {i}"
            assert reader.has_entry(ge["name"])

        assert reader.lookup("nonexistent") == -1
        assert not reader.has_entry("nonexistent")

    @pytest.mark.parametrize(
        "file_index", range(6), ids=[
            "basic", "types", "align16", "noalign", "wshard", "hierarchical"
        ]
    )
    def test_read_by_name(self, manifest, file_index):
        gf = manifest["files"][file_index]
        reader = ShardReader(TESTDATA / gf["filename"])

        for ge in gf["entries"]:
            data = reader.read_entry_by_name(ge["name"])
            actual = hashlib.sha256(data).hexdigest()
            assert actual == ge["data_sha256"]


class TestCRC32C:
    """Verify CRC32C matches Go's crc32.Castagnoli."""

    def test_hello_world(self):
        # "hello world" CRC32C from golden_basic.shard entry 0
        result = compute_crc32c(b"hello world")
        assert result == 3381945770

    def test_empty(self):
        assert compute_crc32c(b"") == 0

    def test_pattern_1k(self):
        data = bytes(i % 256 for i in range(1024))
        result = compute_crc32c(data)
        assert result == 752840335  # from golden manifest


class TestXXHash64:
    """Verify xxHash64 matches Go's cespare/xxhash/v2."""

    def test_greeting(self):
        result = compute_xxhash64("greeting")
        assert result == 16700977181434310015

    def test_pattern_1k(self):
        result = compute_xxhash64("pattern/1k")
        assert result == 11449517142474649281

    def test_metadata_model(self):
        result = compute_xxhash64("metadata/model")
        assert result == 7724208215623586332


class TestListPrefix:
    def test_hierarchical(self, manifest):
        gf = manifest["files"][5]  # golden_hierarchical.shard
        reader = ShardReader(TESTDATA / gf["filename"])

        attn = reader.list_prefix("layer.0/attention/")
        assert len(attn) == 4
        assert all("attention" in n for n in attn)

        ffn = reader.list_prefix("layer.0/ffn/")
        assert len(ffn) == 3
        assert all("ffn" in n for n in ffn)

    def test_wshard_lanes(self, manifest):
        gf = manifest["files"][4]  # golden_wshard.shard
        reader = ShardReader(TESTDATA / gf["filename"])

        signals = reader.list_prefix("signal/")
        assert len(signals) == 1
        omens = reader.list_prefix("omen/")
        assert len(omens) == 1
        residuals = reader.list_prefix("residual/")
        assert len(residuals) == 1
