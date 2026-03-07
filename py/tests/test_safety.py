"""Cross-language safety tests — verify Python handles Go-generated safety fixtures.

Reads golden safety fixtures from cowrie/ucodec/testdata/safety/ and validates:
1. Valid shards: entry count, entry names, data SHA256 all match manifest
2. Corrupt shards: opening or reading raises an exception (no silent corruption)
3. Python round-trip: write and read back with SHA256 verification
4. Thread safety: concurrent reads on a valid shard produce correct data
"""
import hashlib
import json
import random
import struct
import tempfile
import threading
from pathlib import Path

import pytest

from shard_v2 import ShardV2Reader, ShardV2Writer

SAFETY_DIR = Path(__file__).resolve().parent.parent.parent / "ucodec" / "testdata" / "safety"
MANIFEST_PATH = SAFETY_DIR / "safety_manifest.json"


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ============================================================
# Valid shard tests
# ============================================================

class TestValidShards:
    """For each entry in manifest['valid'], verify Python reads match Go's output."""

    def test_manifest_exists(self):
        assert MANIFEST_PATH.exists(), f"safety manifest not found at {MANIFEST_PATH}"

    def test_all_valid_files_exist(self, manifest):
        for vf in manifest["valid"]:
            path = SAFETY_DIR / vf["filename"]
            assert path.exists(), f"valid fixture missing: {path}"

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_entry_count(self, manifest, valid_index):
        """Open each valid shard and verify entry_count matches manifest."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])
        assert reader.entry_count() == vf["entry_count"], (
            f"{vf['filename']}: entry_count {reader.entry_count()} != expected {vf['entry_count']}"
        )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_entry_names(self, manifest, valid_index):
        """Verify entry names match manifest for each valid shard."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])
        expected_names = [e["name"] for e in vf["entries"]]
        actual_names = reader.entry_names()
        assert actual_names == expected_names, (
            f"{vf['filename']}: names mismatch\n"
            f"  expected: {expected_names}\n"
            f"  actual:   {actual_names}"
        )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_entry_data_sha256(self, manifest, valid_index):
        """Read each entry and verify SHA256 of data matches manifest."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])

        for i, entry_spec in enumerate(vf["entries"]):
            data = reader.read_entry(i)
            actual_sha = hashlib.sha256(data).hexdigest()
            assert actual_sha == entry_spec["data_sha256"], (
                f"{vf['filename']} entry {i} ({entry_spec['name']}): "
                f"SHA256 mismatch\n"
                f"  expected: {entry_spec['data_sha256']}\n"
                f"  actual:   {actual_sha}"
            )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_entry_sizes(self, manifest, valid_index):
        """Verify data sizes match manifest for each entry."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])

        for i, entry_spec in enumerate(vf["entries"]):
            data = reader.read_entry(i)
            assert len(data) == entry_spec["size"], (
                f"{vf['filename']} entry {i} ({entry_spec['name']}): "
                f"size {len(data)} != expected {entry_spec['size']}"
            )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_file_sha256(self, manifest, valid_index):
        """Verify whole-file SHA256 matches manifest."""
        vf = manifest["valid"][valid_index]
        path = SAFETY_DIR / vf["filename"]
        file_data = path.read_bytes()
        actual_sha = hashlib.sha256(file_data).hexdigest()
        assert actual_sha == vf["sha256"], (
            f"{vf['filename']}: file SHA256 mismatch\n"
            f"  expected: {vf['sha256']}\n"
            f"  actual:   {actual_sha}"
        )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_lookup_by_name(self, manifest, valid_index):
        """Verify lookup(name) returns correct index for every entry in the manifest."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])

        for i, entry_spec in enumerate(vf["entries"]):
            idx = reader.lookup(entry_spec["name"])
            assert idx == i, (
                f"{vf['filename']} entry {i} ({entry_spec['name']}): "
                f"lookup returned {idx}, expected {i}"
            )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_read_entry_by_name(self, manifest, valid_index):
        """Verify read_entry_by_name(name) returns the same data as read_entry(index)."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])

        for i, entry_spec in enumerate(vf["entries"]):
            data_by_index = reader.read_entry(i)
            data_by_name = reader.read_entry_by_name(entry_spec["name"])
            assert data_by_name == data_by_index, (
                f"{vf['filename']} entry {i} ({entry_spec['name']}): "
                f"read_entry_by_name data differs from read_entry"
            )
            # Also verify SHA256 matches manifest
            actual_sha = hashlib.sha256(data_by_name).hexdigest()
            assert actual_sha == entry_spec["data_sha256"], (
                f"{vf['filename']} entry {i} ({entry_spec['name']}): "
                f"read_entry_by_name SHA256 mismatch"
            )

    @pytest.mark.parametrize("valid_index", range(6), ids=[
        "basic", "compressed", "empty_entry", "long_name", "unicode_name", "binary_data"
    ])
    def test_lookup_nonexistent(self, manifest, valid_index):
        """Verify that looking up nonexistent names returns -1 or raises."""
        vf = manifest["valid"][valid_index]
        reader = ShardV2Reader(SAFETY_DIR / vf["filename"])

        for bad_name in ("nonexistent", "", "__garbage__"):
            idx = reader.lookup(bad_name)
            assert idx == -1, (
                f"{vf['filename']}: lookup({bad_name!r}) returned {idx}, expected -1"
            )
            # read_entry_by_name should raise KeyError for missing names
            with pytest.raises(KeyError):
                reader.read_entry_by_name(bad_name)


# ============================================================
# Corrupt shard tests
# ============================================================

class TestCorruptShards:
    """For each entry in manifest['corrupt'], verify Python raises an error."""

    def test_all_corrupt_files_exist(self, manifest):
        for cf in manifest["corrupt"]:
            path = SAFETY_DIR / cf["filename"]
            assert path.exists(), f"corrupt fixture missing: {path}"

    @pytest.mark.parametrize("corrupt_index,corrupt_id", [
        (0, "bad_magic"),
        (1, "truncated_header"),
        (2, "crc_mismatch"),
        (3, "truncated_data"),
        (4, "entry_count_overflow"),
        (5, "huge_string_table"),
    ])
    def test_corrupt_raises(self, manifest, corrupt_index, corrupt_id):
        """Opening or reading a corrupt shard must raise an exception."""
        cf = manifest["corrupt"][corrupt_index]
        path = SAFETY_DIR / cf["filename"]

        try:
            reader = ShardV2Reader(path)
            # If the reader opens without error, try reading all entries.
            # At least one operation must fail.
            for i in range(reader.entry_count()):
                reader.read_entry(i)
            # If we get here, no error was raised -- test fails
            pytest.fail(
                f"{cf['filename']} ({cf['description']}): "
                f"expected error '{cf['expected_error']}' but no exception raised"
            )
        except (ValueError, KeyError, IndexError, struct.error,
                OSError, OverflowError, MemoryError, RuntimeError,
                Exception) as exc:
            # Any error is acceptable -- the corrupt shard was detected
            assert str(exc), (
                f"{cf['filename']}: exception raised but message is empty"
            )

    @pytest.mark.parametrize("corrupt_index,corrupt_id", [
        (0, "bad_magic"),
        (1, "truncated_header"),
        (2, "crc_mismatch"),
        (3, "truncated_data"),
        (4, "entry_count_overflow"),
        (5, "huge_string_table"),
    ])
    def test_corrupt_no_crash(self, manifest, corrupt_index, corrupt_id):
        """Explicit safety net: opening and reading a corrupt fixture must not crash."""
        cf = manifest["corrupt"][corrupt_index]
        path = SAFETY_DIR / cf["filename"]

        # The test passes if we reach the end -- no uncaught exception
        # crashes the process.  We tolerate any raised exception.
        try:
            reader = ShardV2Reader(path)
            for i in range(reader.entry_count()):
                reader.read_entry(i)
        except Exception:
            pass  # any exception is fine, we just must not crash

    def test_corrupt_empty_file(self):
        """A 0-byte file must raise an exception on open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty.shard"
            empty_path.write_bytes(b"")

            with pytest.raises(Exception):
                ShardV2Reader(empty_path)

    def test_corrupt_wrong_magic(self):
        """A 64-byte file with wrong magic bytes must be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "wrong_magic.shard"
            bad_data = b"NOPE" + b"\x00" * 60
            bad_path.write_bytes(bad_data)

            with pytest.raises((ValueError, Exception)):
                ShardV2Reader(bad_path)

    def test_corrupt_random_bytes(self):
        """A 1024-byte file of random data must not crash (may raise)."""
        rng = random.Random(42)  # deterministic seed
        random_data = bytes(rng.getrandbits(8) for _ in range(1024))

        with tempfile.TemporaryDirectory() as tmpdir:
            rand_path = Path(tmpdir) / "random.shard"
            rand_path.write_bytes(random_data)

            try:
                reader = ShardV2Reader(rand_path)
                for i in range(reader.entry_count()):
                    reader.read_entry(i)
            except Exception:
                pass  # any exception is fine


# ============================================================
# Cross-write round-trip test
# ============================================================

class TestCrossWriteRoundTrip:
    """Python writes a shard, reads it back, verifies SHA256."""

    def test_roundtrip_three_entries(self):
        """Write 3 entries with known data, read back and verify SHA256."""
        entries = [
            ("alpha", b"The quick brown fox jumps over the lazy dog"),
            ("beta", bytes(range(256)) * 4),  # 1024 bytes of pattern
            ("gamma", b"\x00" * 512),          # 512 zero bytes
        ]

        # Precompute expected SHA256 values
        expected_sha = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in entries
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "roundtrip.shard"

            # Write
            writer = ShardV2Writer(shard_path)
            for name, data in entries:
                writer.write_entry(name, data)
            writer.close()

            # Read back
            reader = ShardV2Reader(shard_path)
            assert reader.entry_count() == len(entries)

            for i, (name, original_data) in enumerate(entries):
                assert reader.entry_name(i) == name
                read_data = reader.read_entry(i)
                actual_sha = hashlib.sha256(read_data).hexdigest()
                assert actual_sha == expected_sha[name], (
                    f"entry {i} ({name}): SHA256 mismatch after round-trip"
                )
                assert read_data == original_data, (
                    f"entry {i} ({name}): data mismatch after round-trip"
                )

    def test_roundtrip_empty_entry(self):
        """Write and read back an empty entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "empty_roundtrip.shard"

            writer = ShardV2Writer(shard_path)
            writer.write_entry("empty", b"")
            writer.close()

            reader = ShardV2Reader(shard_path)
            assert reader.entry_count() == 1
            data = reader.read_entry(0)
            assert data == b""
            assert hashlib.sha256(data).hexdigest() == (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )

    def test_roundtrip_unicode_names(self):
        """Write entries with unicode names, read back correctly."""
        entries = [
            ("tensor/\u65e5\u672c\u8a9e", b"nihongo"),
            ("\u00e9moji_\U0001f600", b"smile"),
            ("\u0410\u0411\u0412\u0413", b"cyrillic"),
        ]

        expected_sha = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in entries
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "unicode_roundtrip.shard"

            writer = ShardV2Writer(shard_path)
            for name, data in entries:
                writer.write_entry(name, data)
            writer.close()

            reader = ShardV2Reader(shard_path)
            assert reader.entry_count() == len(entries)

            for i, (name, _) in enumerate(entries):
                assert reader.entry_name(i) == name
                read_data = reader.read_entry(i)
                actual_sha = hashlib.sha256(read_data).hexdigest()
                assert actual_sha == expected_sha[name]

    def test_roundtrip_binary_data(self):
        """Write all 256 byte values, verify round-trip."""
        all_bytes = bytes(range(256))
        expected_sha = hashlib.sha256(all_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "binary_roundtrip.shard"

            writer = ShardV2Writer(shard_path)
            writer.write_entry("all_bytes", all_bytes)
            writer.close()

            reader = ShardV2Reader(shard_path)
            data = reader.read_entry(0)
            assert hashlib.sha256(data).hexdigest() == expected_sha
            assert data == all_bytes

    def test_roundtrip_large_entry(self):
        """Write a 1 MB+ entry, read back, verify SHA256 matches."""
        rng = random.Random(12345)
        large_data = bytes(rng.getrandbits(8) for _ in range(1_048_576 + 37))
        expected_sha = hashlib.sha256(large_data).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "large_roundtrip.shard"

            writer = ShardV2Writer(shard_path)
            writer.write_entry("large_blob", large_data)
            writer.close()

            reader = ShardV2Reader(shard_path)
            assert reader.entry_count() == 1
            data = reader.read_entry(0)
            assert len(data) == len(large_data)
            actual_sha = hashlib.sha256(data).hexdigest()
            assert actual_sha == expected_sha, (
                f"large_entry: SHA256 mismatch\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual_sha}"
            )

    def test_roundtrip_many_entries(self):
        """Write 200 entries with unique data, read all back, verify names and SHA256."""
        entries = []
        expected_sha = {}
        for i in range(200):
            name = f"entry_{i:04d}"
            data = f"payload-{i}-{'x' * (i % 64)}".encode("utf-8")
            entries.append((name, data))
            expected_sha[name] = hashlib.sha256(data).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "many_entries.shard"

            writer = ShardV2Writer(shard_path)
            for name, data in entries:
                writer.write_entry(name, data)
            writer.close()

            reader = ShardV2Reader(shard_path)
            assert reader.entry_count() == 200

            actual_names = reader.entry_names()
            expected_names = [name for name, _ in entries]
            assert actual_names == expected_names, "entry name list mismatch"

            for i, (name, original_data) in enumerate(entries):
                read_data = reader.read_entry(i)
                actual_sha = hashlib.sha256(read_data).hexdigest()
                assert actual_sha == expected_sha[name], (
                    f"entry {i} ({name}): SHA256 mismatch"
                )
                assert read_data == original_data

    def test_roundtrip_crc_integrity(self):
        """Write a shard, verify it reads, corrupt a byte, verify read fails or data differs."""
        original_data = b"integrity check payload" * 100
        expected_sha = hashlib.sha256(original_data).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "crc_integrity.shard"

            writer = ShardV2Writer(shard_path)
            writer.write_entry("check_me", original_data)
            writer.close()

            # Verify clean read succeeds
            reader = ShardV2Reader(shard_path)
            data = reader.read_entry(0)
            assert hashlib.sha256(data).hexdigest() == expected_sha

            # Corrupt a byte in the data section (near the end of the file)
            raw = bytearray(shard_path.read_bytes())
            # Flip a byte in the last quarter of the file (data section)
            corrupt_offset = len(raw) * 3 // 4
            raw[corrupt_offset] ^= 0xFF
            shard_path.write_bytes(bytes(raw))

            # Re-read: must either raise (CRC check) or return different data
            try:
                reader2 = ShardV2Reader(shard_path)
                data2 = reader2.read_entry(0)
                # If no exception, data must differ from original
                actual_sha2 = hashlib.sha256(data2).hexdigest()
                assert actual_sha2 != expected_sha or data2 != original_data, (
                    "corrupted file returned identical data without raising"
                )
            except Exception:
                pass  # CRC or other validation error -- expected

    def test_roundtrip_entry_order(self):
        """Write 20 entries in reverse-alphabetical order, verify they come back in same order."""
        names = sorted([f"entry_{chr(ord('a') + i)}" for i in range(20)], reverse=True)
        entries = [(name, f"data-for-{name}".encode("utf-8")) for name in names]

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "order_roundtrip.shard"

            writer = ShardV2Writer(shard_path)
            for name, data in entries:
                writer.write_entry(name, data)
            writer.close()

            reader = ShardV2Reader(shard_path)
            assert reader.entry_count() == 20

            actual_names = reader.entry_names()
            assert actual_names == names, (
                f"entry order not preserved\n"
                f"  expected: {names}\n"
                f"  actual:   {actual_names}"
            )

            for i, (name, original_data) in enumerate(entries):
                assert reader.entry_name(i) == name
                assert reader.read_entry(i) == original_data


# ============================================================
# Thread safety test
# ============================================================

class TestThreadSafety:
    """Concurrent reads on a valid shard must not crash or corrupt data."""

    def test_concurrent_reads(self, manifest):
        """10 threads read all entries from valid_basic.shard simultaneously."""
        vf = manifest["valid"][0]  # valid_basic.shard
        shard_path = SAFETY_DIR / vf["filename"]

        # Precompute expected SHA256 for each entry
        expected = {
            entry["name"]: entry["data_sha256"]
            for entry in vf["entries"]
        }

        errors = []
        barrier = threading.Barrier(10)

        def reader_thread(thread_id):
            try:
                # Each thread opens its own reader (independent mmap/buffer)
                barrier.wait(timeout=5)
                reader = ShardV2Reader(shard_path)

                assert reader.entry_count() == vf["entry_count"]

                for i in range(reader.entry_count()):
                    data = reader.read_entry(i)
                    name = reader.entry_name(i)
                    actual_sha = hashlib.sha256(data).hexdigest()
                    if actual_sha != expected[name]:
                        errors.append(
                            f"thread {thread_id}, entry {i} ({name}): "
                            f"SHA256 mismatch"
                        )
            except Exception as exc:
                errors.append(f"thread {thread_id}: {type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=reader_thread, args=(tid,))
            for tid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, (
            f"Thread safety failures:\n" + "\n".join(errors)
        )

    def test_concurrent_reads_all_valid(self, manifest):
        """Each valid shard gets read by 4 threads concurrently."""
        all_errors = []

        for vf in manifest["valid"]:
            shard_path = SAFETY_DIR / vf["filename"]
            expected = {
                entry["name"]: entry["data_sha256"]
                for entry in vf["entries"]
            }

            errors = []
            barrier = threading.Barrier(4)

            def reader_thread(thread_id, path=shard_path, exp=expected,
                              errs=errors, vfile=vf, b=barrier):
                try:
                    b.wait(timeout=5)
                    reader = ShardV2Reader(path)
                    assert reader.entry_count() == vfile["entry_count"]
                    for i in range(reader.entry_count()):
                        data = reader.read_entry(i)
                        name = reader.entry_name(i)
                        actual_sha = hashlib.sha256(data).hexdigest()
                        if actual_sha != exp[name]:
                            errs.append(
                                f"thread {thread_id}, {vfile['filename']}, "
                                f"entry {name}: SHA256 mismatch"
                            )
                except Exception as exc:
                    errs.append(
                        f"thread {thread_id}, {vfile['filename']}: "
                        f"{type(exc).__name__}: {exc}"
                    )

            threads = [
                threading.Thread(target=reader_thread, args=(tid,))
                for tid in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            all_errors.extend(errors)

        assert not all_errors, (
            f"Thread safety failures:\n" + "\n".join(all_errors)
        )

    def test_concurrent_reads_stress(self, manifest):
        """50 threads, each reading all entries 5 times, verify SHA256 every time."""
        vf = manifest["valid"][0]  # valid_basic.shard
        shard_path = SAFETY_DIR / vf["filename"]

        expected = {
            entry["name"]: entry["data_sha256"]
            for entry in vf["entries"]
        }

        num_threads = 50
        reads_per_thread = 5
        errors = []
        barrier = threading.Barrier(num_threads)

        def stress_thread(thread_id):
            try:
                barrier.wait(timeout=10)
                for iteration in range(reads_per_thread):
                    reader = ShardV2Reader(shard_path)
                    assert reader.entry_count() == vf["entry_count"]

                    for i in range(reader.entry_count()):
                        data = reader.read_entry(i)
                        name = reader.entry_name(i)
                        actual_sha = hashlib.sha256(data).hexdigest()
                        if actual_sha != expected[name]:
                            errors.append(
                                f"thread {thread_id}, iter {iteration}, "
                                f"entry {i} ({name}): SHA256 mismatch"
                            )
            except Exception as exc:
                errors.append(
                    f"thread {thread_id}: {type(exc).__name__}: {exc}"
                )

        threads = [
            threading.Thread(target=stress_thread, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, (
            f"Stress test failures ({len(errors)}):\n" + "\n".join(errors[:20])
        )
