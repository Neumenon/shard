"""Streaming writer roundtrip tests — ShardStreamWriter."""
import pytest

from shard_format import (
    ALIGN_64,
    ALIGN_NONE,
    ALIGN_16,
    ALIGN_32,
    COMPRESS_ZSTD,
    COMPRESS_LZ4,
    FLAG_STREAMING,
    ROLE_MOSH,
    ROLE_SAMPLE,
    ShardReader,
    ShardStreamWriter,
    compute_crc32c,
)


class TestStreamWriterBasic:
    def test_basic_roundtrip(self, tmp_path):
        path = tmp_path / "stream.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.begin_data()
        sw.write_entry("hello", b"world")
        sw.write_entry("foo", b"bar")
        sw.finalize()

        r = ShardReader(path)
        assert r.entry_count() == 2
        assert r.read_entry(0) == b"world"
        assert r.read_entry(1) == b"bar"
        assert r.entry_name(0) == "hello"
        assert r.entry_name(1) == "foo"
        # FLAG_STREAMING must be set
        assert r.header().flags & FLAG_STREAMING

    def test_flag_streaming_set(self, tmp_path):
        path = tmp_path / "flags.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.write_entry("x", b"data")
        sw.finalize()

        r = ShardReader(path)
        assert r.header().flags & FLAG_STREAMING, "FLAG_STREAMING must be set by StreamWriter"

    def test_empty_entries(self, tmp_path):
        """StreamWriter with zero entries should produce a valid empty shard."""
        path = tmp_path / "stream_empty.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.begin_data()
        sw.finalize()

        r = ShardReader(path)
        assert r.entry_count() == 0

    def test_single_entry(self, tmp_path):
        path = tmp_path / "stream_single.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.write_entry("only", b"payload")
        sw.finalize()

        r = ShardReader(path)
        assert r.entry_count() == 1
        assert r.read_entry(0) == b"payload"

    def test_role_propagated(self, tmp_path):
        path = tmp_path / "stream_role.shard"
        sw = ShardStreamWriter(path, role=ROLE_SAMPLE, max_entries=5)
        sw.begin_data()
        sw.write_entry("a", b"1")
        sw.finalize()

        r = ShardReader(path)
        assert r.header().role == ROLE_SAMPLE

    def test_total_file_size_correct(self, tmp_path):
        path = tmp_path / "stream_size.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.write_entry("item", b"hello")
        sw.finalize()

        r = ShardReader(path)
        actual = path.stat().st_size
        assert r.header().total_file_size == actual

    def test_lookup_by_name(self, tmp_path):
        path = tmp_path / "stream_lookup.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.begin_data()
        sw.write_entry("alpha", b"a")
        sw.write_entry("beta", b"b")
        sw.write_entry("gamma", b"c")
        sw.finalize()

        r = ShardReader(path)
        assert r.lookup("alpha") == 0
        assert r.lookup("beta") == 1
        assert r.lookup("gamma") == 2
        assert r.lookup("missing") == -1

    def test_read_by_name(self, tmp_path):
        path = tmp_path / "stream_byname.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.begin_data()
        sw.write_entry("key1", b"value1")
        sw.write_entry("key2", b"value2")
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry_by_name("key1") == b"value1"
        assert r.read_entry_by_name("key2") == b"value2"
        with pytest.raises(KeyError):
            r.read_entry_by_name("nonexistent")

    def test_checksum_stored_correctly(self, tmp_path):
        path = tmp_path / "stream_cksum.shard"
        data = b"checksum_test_data"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.write_entry("data", data)
        sw.finalize()

        r = ShardReader(path)
        info = r.get_entry_info(0)
        assert info.checksum == compute_crc32c(data)


class TestStreamWriterAlignment:
    def test_alignment_64_default(self, tmp_path):
        path = tmp_path / "stream_align64.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.begin_data()
        sw.write_entry("a", b"data_a")
        sw.write_entry("b", b"data_b")
        sw.finalize()

        r = ShardReader(path)
        assert r.header().alignment == ALIGN_64
        for i in range(r.entry_count()):
            offset = r.get_entry_info(i).data_offset
            assert offset % 64 == 0, f"entry {i} data_offset {offset} not 64-aligned"

    def test_set_alignment_64_explicit(self, tmp_path):
        path = tmp_path / "stream_aligned.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.set_alignment(ALIGN_64)
        sw.begin_data()
        sw.write_entry("a", b"data_a")
        sw.write_entry("b", b"data_b")
        sw.finalize()

        r = ShardReader(path)
        for i in range(r.entry_count()):
            assert r.get_entry_info(i).data_offset % 64 == 0

    def test_alignment_16(self, tmp_path):
        path = tmp_path / "stream_align16.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.set_alignment(ALIGN_16)
        sw.begin_data()
        sw.write_entry("x", b"hello")
        sw.write_entry("y", b"world_longer")
        sw.finalize()

        r = ShardReader(path)
        assert r.header().alignment == ALIGN_16
        for i in range(r.entry_count()):
            offset = r.get_entry_info(i).data_offset
            assert offset % 16 == 0, f"entry {i} data_offset {offset} not 16-aligned"

    def test_alignment_32(self, tmp_path):
        path = tmp_path / "stream_align32.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.set_alignment(ALIGN_32)
        sw.begin_data()
        sw.write_entry("p", b"payload_data")
        sw.write_entry("q", b"more_payload")
        sw.finalize()

        r = ShardReader(path)
        assert r.header().alignment == ALIGN_32
        for i in range(r.entry_count()):
            offset = r.get_entry_info(i).data_offset
            assert offset % 32 == 0

    def test_alignment_none(self, tmp_path):
        path = tmp_path / "stream_noalign.shard"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.set_alignment(ALIGN_NONE)
        sw.begin_data()
        sw.write_entry("a", b"x")
        sw.write_entry("b", b"yy")
        sw.finalize()

        r = ShardReader(path)
        assert r.header().alignment == ALIGN_NONE
        assert r.read_entry(0) == b"x"
        assert r.read_entry(1) == b"yy"

    def test_set_alignment_after_begin_raises(self, tmp_path):
        path = tmp_path / "stream_err.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        with pytest.raises(RuntimeError, match="cannot set alignment after begin_data"):
            sw.set_alignment(ALIGN_16)
        sw.finalize()

    def test_invalid_alignment_raises(self, tmp_path):
        path = tmp_path / "stream_invalid.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        with pytest.raises(ValueError, match="invalid alignment"):
            sw.set_alignment(7)
        sw.close()


class TestStreamWriterEntryCount:
    def test_many_entries(self, tmp_path):
        path = tmp_path / "stream_many.shard"
        n = 100
        sw = ShardStreamWriter(path, max_entries=n)
        sw.begin_data()
        for i in range(n):
            sw.write_entry(f"entry_{i:04d}", f"data_{i}".encode())
        sw.finalize()

        r = ShardReader(path)
        assert r.entry_count() == n
        for i in range(n):
            assert r.entry_name(i) == f"entry_{i:04d}"
            assert r.read_entry(i) == f"data_{i}".encode()

    def test_fewer_entries_than_max(self, tmp_path):
        """Writing fewer entries than max_entries should work fine."""
        path = tmp_path / "stream_under.shard"
        sw = ShardStreamWriter(path, max_entries=1000)
        sw.begin_data()
        sw.write_entry("only_one", b"data")
        sw.finalize()

        r = ShardReader(path)
        assert r.entry_count() == 1
        assert r.read_entry(0) == b"data"


class TestStreamWriterLargeData:
    def test_large_data(self, tmp_path):
        path = tmp_path / "stream_large.shard"
        big = bytes(i % 256 for i in range(100_000))
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.write_entry("big", big)
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry(0) == big

    def test_multiple_large_entries(self, tmp_path):
        path = tmp_path / "stream_multi_large.shard"
        chunks = [bytes((i * j) % 256 for j in range(50_000)) for i in range(1, 4)]
        sw = ShardStreamWriter(path, max_entries=10)
        sw.begin_data()
        for idx, chunk in enumerate(chunks):
            sw.write_entry(f"chunk_{idx}", chunk)
        sw.finalize()

        r = ShardReader(path)
        assert r.entry_count() == 3
        for idx, chunk in enumerate(chunks):
            assert r.read_entry(idx) == chunk

    def test_binary_all_bytes(self, tmp_path):
        """All 256 byte values survive a streaming roundtrip."""
        path = tmp_path / "stream_allbytes.shard"
        all_bytes = bytes(range(256))
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.write_entry("allbytes", all_bytes)
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry(0) == all_bytes


class TestStreamWriterCompression:
    def test_zstd_compressed_entry(self, tmp_path):
        path = tmp_path / "stream_zstd.shard"
        data = bytes(i % 256 for i in range(10_000))  # compressible
        sw = ShardStreamWriter(path, max_entries=5)
        sw.set_compression(COMPRESS_ZSTD)
        sw.begin_data()
        sw.write_entry_compressed("data", data)
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry(0) == data
        info = r.get_entry_info(0)
        assert info.compressed
        assert info.disk_size < info.orig_size

    def test_lz4_compressed_entry(self, tmp_path):
        path = tmp_path / "stream_lz4.shard"
        data = bytes(i % 256 for i in range(10_000))
        sw = ShardStreamWriter(path, max_entries=5)
        sw.set_compression(COMPRESS_LZ4)
        sw.begin_data()
        sw.write_entry_compressed("data", data)
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry(0) == data
        info = r.get_entry_info(0)
        assert info.compressed
        assert info.disk_size < info.orig_size

    def test_small_data_not_compressed(self, tmp_path):
        """Entries below MIN_COMPRESS_SIZE are stored raw even if compression is set."""
        path = tmp_path / "stream_small.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.set_compression(COMPRESS_ZSTD)
        sw.begin_data()
        sw.write_entry_compressed("tiny", b"hello")
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry(0) == b"hello"
        assert not r.get_entry_info(0).compressed

    def test_mixed_compressed_uncompressed(self, tmp_path):
        """Mix write_entry and write_entry_compressed in same shard."""
        path = tmp_path / "stream_mixed.shard"
        big = bytes(i % 256 for i in range(10_000))
        small = b"tiny"
        sw = ShardStreamWriter(path, max_entries=10)
        sw.set_compression(COMPRESS_ZSTD)
        sw.begin_data()
        sw.write_entry_compressed("big", big)
        sw.write_entry("small", small)
        sw.finalize()

        r = ShardReader(path)
        assert r.read_entry(0) == big
        assert r.read_entry(1) == small
        assert r.get_entry_info(0).compressed
        assert not r.get_entry_info(1).compressed


class TestStreamWriterStateErrors:
    def test_write_before_begin_raises(self, tmp_path):
        path = tmp_path / "stream_err1.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        with pytest.raises(RuntimeError, match="call begin_data"):
            sw.write_entry("x", b"data")
        sw.close()

    def test_begin_twice_raises(self, tmp_path):
        path = tmp_path / "stream_err2.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        with pytest.raises(RuntimeError, match="already called"):
            sw.begin_data()
        sw.finalize()

    def test_finalize_twice_raises(self, tmp_path):
        path = tmp_path / "stream_err3.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.finalize()
        with pytest.raises(RuntimeError, match="already finalized"):
            sw.finalize()

    def test_write_after_finalize_raises(self, tmp_path):
        path = tmp_path / "stream_err4.shard"
        sw = ShardStreamWriter(path, max_entries=5)
        sw.begin_data()
        sw.finalize()
        with pytest.raises(RuntimeError, match="already finalized"):
            sw.write_entry("x", b"data")
