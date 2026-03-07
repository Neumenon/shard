"""Compression roundtrip tests."""
import pytest
from shard_v2 import (
    COMPRESS_LZ4,
    COMPRESS_NONE,
    COMPRESS_ZSTD,
    ShardV2Reader,
    ShardV2Writer,
    compute_crc32c,
)


class TestZstdCompression:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "zstd.shard"
        data = bytes(i % 256 for i in range(10000))  # compressible
        w = ShardV2Writer(path)
        w.set_compression(COMPRESS_ZSTD)
        w.write_entry_compressed("big", data)
        w.close()

        r = ShardV2Reader(path)
        assert r.read_entry(0) == data
        info = r.get_entry_info(0)
        assert info.compressed  # should be compressed
        assert info.disk_size < info.orig_size  # actually smaller
        assert info.orig_size == len(data)

    def test_small_data_not_compressed(self, tmp_path):
        """Data < 256 bytes should not be compressed even if requested."""
        path = tmp_path / "small.shard"
        w = ShardV2Writer(path)
        w.set_compression(COMPRESS_ZSTD)
        w.write_entry_compressed("tiny", b"hello")
        w.close()

        r = ShardV2Reader(path)
        assert r.read_entry(0) == b"hello"
        assert not r.get_entry_info(0).compressed

    def test_incompressible_not_compressed(self, tmp_path):
        """Random data that doesn't compress well should be stored raw."""
        import os
        path = tmp_path / "random.shard"
        data = os.urandom(1000)
        w = ShardV2Writer(path)
        w.set_compression(COMPRESS_ZSTD)
        w.write_entry_compressed("random", data)
        w.close()

        r = ShardV2Reader(path)
        assert r.read_entry(0) == data

    def test_checksum_on_uncompressed(self, tmp_path):
        """CRC32C should match uncompressed data."""
        path = tmp_path / "cksum.shard"
        data = bytes(range(256)) * 10
        w = ShardV2Writer(path)
        w.set_compression(COMPRESS_ZSTD)
        w.write_entry_compressed("data", data)
        w.close()

        r = ShardV2Reader(path)
        info = r.get_entry_info(0)
        assert info.checksum == compute_crc32c(data)


class TestLZ4Compression:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "lz4.shard"
        data = bytes(i % 256 for i in range(10000))
        w = ShardV2Writer(path)
        w.set_compression(COMPRESS_LZ4)
        w.write_entry_compressed("big", data)
        w.close()

        r = ShardV2Reader(path)
        assert r.read_entry(0) == data
        info = r.get_entry_info(0)
        assert info.compressed
        assert info.disk_size < info.orig_size

    def test_mixed_entries(self, tmp_path):
        """Mix of compressed and uncompressed entries."""
        path = tmp_path / "mixed.shard"
        data1 = bytes(range(256)) * 10
        data2 = b"small"
        data3 = bytes(range(256)) * 20

        w = ShardV2Writer(path)
        w.set_compression(COMPRESS_LZ4)
        w.write_entry_compressed("compressed1", data1)
        w.write_entry("uncompressed", data2)
        w.write_entry_compressed("compressed2", data3)
        w.close()

        r = ShardV2Reader(path)
        assert r.read_entry(0) == data1
        assert r.read_entry(1) == data2
        assert r.read_entry(2) == data3


class TestWriteEntryWithOptions:
    def test_explicit_zstd(self, tmp_path):
        path = tmp_path / "explicit.shard"
        data = bytes(range(256)) * 10
        w = ShardV2Writer(path)
        w.write_entry_with_options("zstd", data, compress=True, comp_type=COMPRESS_ZSTD)
        w.write_entry_with_options("none", data, compress=False, comp_type=COMPRESS_NONE)
        w.close()

        r = ShardV2Reader(path)
        assert r.read_entry(0) == data
        assert r.read_entry(1) == data
        assert r.get_entry_info(0).compressed
        assert not r.get_entry_info(1).compressed
