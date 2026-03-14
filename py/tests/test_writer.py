"""Roundtrip tests — verify Python writer produces files the reader can parse."""
import hashlib
import tempfile
from pathlib import Path

import pytest

from shard_format import (
    ALIGN_16,
    ALIGN_64,
    ALIGN_NONE,
    CONTENT_TYPE_BLOB,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_TENSOR,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_UNKNOWN,
    ROLE_MOSH,
    ROLE_SAMPLE,
    ROLE_WSHARD,
    ShardReader,
    ShardWriter,
    compute_crc32c,
    compute_xxhash64,
)


class TestWriterRoundtrip:
    """Write a shard, read it back, verify everything matches."""

    def test_single_entry(self, tmp_path):
        path = tmp_path / "single.shard"
        w = ShardWriter(path)
        w.write_entry("hello", b"world")
        w.close()

        r = ShardReader(path)
        assert r.entry_count() == 1
        assert r.entry_names() == ["hello"]
        assert r.read_entry(0) == b"world"

    def test_multiple_entries(self, tmp_path):
        path = tmp_path / "multi.shard"
        entries = [
            ("alpha", b"data_alpha"),
            ("beta", b"data_beta_longer"),
            ("gamma", b"g"),
        ]

        w = ShardWriter(path)
        for name, data in entries:
            w.write_entry(name, data)
        w.close()

        r = ShardReader(path)
        assert r.entry_count() == 3
        for i, (name, data) in enumerate(entries):
            assert r.entry_name(i) == name
            assert r.read_entry(i) == data

    def test_typed_entries(self, tmp_path):
        path = tmp_path / "typed.shard"
        w = ShardWriter(path)
        w.write_entry_typed("model.bin", b"\x00" * 256, CONTENT_TYPE_TENSOR)
        w.write_entry_typed("config.json", b'{"key": "value"}', CONTENT_TYPE_JSON)
        w.write_entry_typed("readme.txt", b"hello world", CONTENT_TYPE_TEXT)
        w.close()

        r = ShardReader(path)
        assert r.entry_count() == 3
        assert r.get_entry_info(0).content_type == CONTENT_TYPE_TENSOR
        assert r.get_entry_info(1).content_type == CONTENT_TYPE_JSON
        assert r.get_entry_info(2).content_type == CONTENT_TYPE_TEXT

    def test_empty_shard(self, tmp_path):
        path = tmp_path / "empty.shard"
        w = ShardWriter(path)
        w.close()

        r = ShardReader(path)
        assert r.entry_count() == 0
        assert r.entry_names() == []

    def test_large_entry(self, tmp_path):
        path = tmp_path / "large.shard"
        big_data = bytes(i % 256 for i in range(100_000))

        w = ShardWriter(path)
        w.write_entry("big", big_data)
        w.close()

        r = ShardReader(path)
        assert r.read_entry(0) == big_data

    def test_alignment_none(self, tmp_path):
        path = tmp_path / "noalign.shard"
        w = ShardWriter(path)
        w.set_alignment(ALIGN_NONE)
        w.write_entry("a", b"short")
        w.write_entry("b", b"another")
        w.close()

        r = ShardReader(path)
        h = r.header()
        assert h.alignment == ALIGN_NONE
        assert r.read_entry(0) == b"short"
        assert r.read_entry(1) == b"another"

    def test_alignment_16(self, tmp_path):
        path = tmp_path / "align16.shard"
        w = ShardWriter(path)
        w.set_alignment(ALIGN_16)
        w.write_entry("x", b"hello")
        w.write_entry("y", b"world")
        w.close()

        r = ShardReader(path)
        h = r.header()
        assert h.alignment == ALIGN_16
        # Verify data offsets are 16-byte aligned
        for i in range(r.entry_count()):
            info = r.get_entry_info(i)
            assert info.data_offset % 16 == 0, (
                f"entry {i} data_offset {info.data_offset} not 16-aligned"
            )
        assert r.read_entry(0) == b"hello"
        assert r.read_entry(1) == b"world"

    def test_alignment_64_default(self, tmp_path):
        path = tmp_path / "align64.shard"
        w = ShardWriter(path)
        w.write_entry("p", b"payload")
        w.close()

        r = ShardReader(path)
        assert r.header().alignment == ALIGN_64
        info = r.get_entry_info(0)
        assert info.data_offset % 64 == 0

    def test_role(self, tmp_path):
        path = tmp_path / "sample.shard"
        w = ShardWriter(path, role=ROLE_SAMPLE)
        w.write_entry("data", b"sample_data")
        w.close()

        r = ShardReader(path)
        assert r.header().role == ROLE_SAMPLE

    def test_checksum_verification(self, tmp_path):
        path = tmp_path / "cksum.shard"
        w = ShardWriter(path)
        w.write_entry("test", b"checksum_data")
        w.close()

        r = ShardReader(path)
        info = r.get_entry_info(0)
        expected_crc = compute_crc32c(b"checksum_data")
        assert info.checksum == expected_crc

    def test_name_hash(self, tmp_path):
        path = tmp_path / "hash.shard"
        w = ShardWriter(path)
        w.write_entry("greeting", b"hi")
        w.close()

        r = ShardReader(path)
        info = r.get_entry_info(0)
        assert info.name_hash == compute_xxhash64("greeting")

    def test_lookup_by_name(self, tmp_path):
        path = tmp_path / "lookup.shard"
        w = ShardWriter(path)
        w.write_entry("first", b"1")
        w.write_entry("second", b"2")
        w.write_entry("third", b"3")
        w.close()

        r = ShardReader(path)
        assert r.lookup("first") == 0
        assert r.lookup("second") == 1
        assert r.lookup("third") == 2
        assert r.lookup("missing") == -1
        assert r.has_entry("second")
        assert not r.has_entry("missing")

    def test_read_by_name(self, tmp_path):
        path = tmp_path / "byname.shard"
        w = ShardWriter(path)
        w.write_entry("key1", b"value1")
        w.write_entry("key2", b"value2")
        w.close()

        r = ShardReader(path)
        assert r.read_entry_by_name("key1") == b"value1"
        assert r.read_entry_by_name("key2") == b"value2"
        with pytest.raises(KeyError):
            r.read_entry_by_name("nonexistent")

    def test_list_prefix(self, tmp_path):
        path = tmp_path / "prefix.shard"
        w = ShardWriter(path)
        w.write_entry("layer.0/weight", b"w0")
        w.write_entry("layer.0/bias", b"b0")
        w.write_entry("layer.1/weight", b"w1")
        w.write_entry("embed/tokens", b"tok")
        w.close()

        r = ShardReader(path)
        l0 = r.list_prefix("layer.0/")
        assert len(l0) == 2
        assert "layer.0/weight" in l0
        assert "layer.0/bias" in l0

        layers = r.list_prefix("layer.")
        assert len(layers) == 3

        embed = r.list_prefix("embed/")
        assert len(embed) == 1

        none = r.list_prefix("nonexistent/")
        assert len(none) == 0

    def test_total_file_size(self, tmp_path):
        path = tmp_path / "size.shard"
        w = ShardWriter(path)
        w.write_entry("a", b"hello")
        w.close()

        r = ShardReader(path)
        actual_size = path.stat().st_size
        assert r.header().total_file_size == actual_size

    def test_unicode_names(self, tmp_path):
        path = tmp_path / "unicode.shard"
        w = ShardWriter(path)
        w.write_entry("tensor/weights", b"\x00\x01\x02")
        w.write_entry("config/params.json", b'{"lr": 0.001}')
        w.close()

        r = ShardReader(path)
        assert r.entry_names() == ["tensor/weights", "config/params.json"]
        assert r.read_entry_by_name("tensor/weights") == b"\x00\x01\x02"

    def test_binary_data_roundtrip(self, tmp_path):
        """All 256 byte values survive roundtrip."""
        path = tmp_path / "binary.shard"
        all_bytes = bytes(range(256))
        w = ShardWriter(path)
        w.write_entry("allbytes", all_bytes)
        w.close()

        r = ShardReader(path)
        assert r.read_entry(0) == all_bytes

    def test_sha256_roundtrip(self, tmp_path):
        """SHA256 of written data matches SHA256 of read data."""
        path = tmp_path / "sha.shard"
        data = b"The quick brown fox jumps over the lazy dog"
        expected_sha = hashlib.sha256(data).hexdigest()

        w = ShardWriter(path)
        w.write_entry("fox", data)
        w.close()

        r = ShardReader(path)
        actual_sha = hashlib.sha256(r.read_entry(0)).hexdigest()
        assert actual_sha == expected_sha


class TestWriterEdgeCases:
    def test_empty_data(self, tmp_path):
        path = tmp_path / "empty_data.shard"
        w = ShardWriter(path)
        w.write_entry("empty", b"")
        w.close()

        r = ShardReader(path)
        assert r.read_entry(0) == b""
        assert r.get_entry_info(0).orig_size == 0
        assert r.get_entry_info(0).disk_size == 0

    def test_single_byte(self, tmp_path):
        path = tmp_path / "onebyte.shard"
        w = ShardWriter(path)
        w.write_entry("one", b"\xff")
        w.close()

        r = ShardReader(path)
        assert r.read_entry(0) == b"\xff"

    def test_many_entries(self, tmp_path):
        path = tmp_path / "many.shard"
        n = 100
        w = ShardWriter(path)
        for i in range(n):
            w.write_entry(f"entry_{i:04d}", f"data_{i}".encode())
        w.close()

        r = ShardReader(path)
        assert r.entry_count() == n
        for i in range(n):
            assert r.entry_name(i) == f"entry_{i:04d}"
            assert r.read_entry(i) == f"data_{i}".encode()
