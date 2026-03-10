"""Tests for SampleShard Python implementation."""

import tempfile
import os
from pathlib import Path

import pytest

from sampleshard import (
    EntryMeta,
    ManifestFileRef,
    SampleShardReader,
    SampleShardWriter,
    ShardMetadata,
)


class TestSampleShardRoundTrip:
    """Test write/read round-trip."""

    def test_basic_roundtrip(self):
        """Test writing and reading samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.smpl"

            # Write samples
            samples = {
                100: {"input": [1, 2, 3], "label": 0},
                200: {"input": [4, 5, 6], "label": 1},
                300: {"input": [7, 8, 9], "label": 2},
            }

            with SampleShardWriter(path) as w:
                for sample_id, sample in samples.items():
                    w.add_sample(sample_id, sample)

            # Read and verify
            with SampleShardReader(path) as r:
                assert r.sample_count() == 3

                # Verify by ID
                for sample_id, expected in samples.items():
                    actual = r.get_sample(sample_id)
                    assert actual == expected

                # Verify IDs are in order
                ids = r.sample_ids()
                assert len(ids) == 3
                assert 100 in ids
                assert 200 in ids
                assert 300 in ids

    def test_iteration(self):
        """Test iterating over samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "iter.smpl"

            # Write samples
            with SampleShardWriter(path) as w:
                for i in range(5):
                    w.add_sample(i, {"id": i, "value": i * 10})

            # Iterate
            with SampleShardReader(path) as r:
                count = 0
                for sample_id, sample in r:
                    assert sample["id"] == sample_id
                    assert sample["value"] == sample_id * 10
                    count += 1

                assert count == 5

    def test_has_sample(self):
        """Test has_sample method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "has.smpl"

            with SampleShardWriter(path) as w:
                w.add_sample(1, {"x": 1})
                w.add_sample(2, {"x": 2})

            with SampleShardReader(path) as r:
                assert r.has_sample(1)
                assert r.has_sample(2)
                assert not r.has_sample(3)
                assert not r.has_sample(999)

    def test_batch_read(self):
        """Test batch reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch.smpl"

            with SampleShardWriter(path) as w:
                for i in range(10):
                    w.add_sample(i, {"idx": i})

            with SampleShardReader(path) as r:
                # Get batch by IDs
                batch = r.get_batch([2, 5, 8])
                assert len(batch) == 3
                assert batch[0]["idx"] == 2
                assert batch[1]["idx"] == 5
                assert batch[2]["idx"] == 8

                # Get batch by range
                range_batch = r.get_batch_by_range(3, 6)
                assert len(range_batch) == 3
                assert range_batch[0]["idx"] == 3

    def test_sample_not_found(self):
        """Test error when sample not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notfound.smpl"

            with SampleShardWriter(path) as w:
                w.add_sample(1, {"x": 1})

            with SampleShardReader(path) as r:
                with pytest.raises(KeyError):
                    r.get_sample(999)

    def test_complex_data(self):
        """Test with complex nested data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "complex.smpl"

            sample = {
                "id": 12345,
                "input": {
                    "features": [1.0, 2.0, 3.0, 4.0],
                    "shape": [2, 2],
                },
                "targets": [0, 1, 0],
                "metadata": {
                    "source": "dataset-v1",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "nested": {
                        "level": 3,
                        "data": [1, 2, 3],
                    },
                },
            }

            with SampleShardWriter(path) as w:
                w.add_sample(1, sample)

            with SampleShardReader(path) as r:
                result = r.get_sample(1)
                assert result == sample

    def test_metadata_roundtrip(self):
        """Test shard-level metadata survives SampleShard roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.smpl"

            with SampleShardWriter(path) as w:
                w.set_metadata(
                    ShardMetadata(
                        producer="sampleshard-tests",
                        profile="sampleshard.v1",
                        entry_metadata={
                            "1": EntryMeta(
                                codec="cowrie-gen2",
                                codec_version="2",
                                semantic_type="sample",
                                row_count=1,
                                stats={"tokens": 42},
                            )
                        },
                    )
                )
                w.add_sample(1, {"x": 1})

            with SampleShardReader(path) as r:
                meta = r.read_metadata()
                assert meta is not None
                assert meta.producer == "sampleshard-tests"
                assert meta.entry_metadata["1"].codec == "cowrie-gen2"
                assert meta.entry_metadata["1"].stats == {"tokens": 42}

    def test_sample_profile_roundtrip(self):
        """Test standardized SampleShard profile helpers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.smpl"

            with SampleShardWriter(path) as w:
                w.set_sample_profile(
                    dataset_name="mnist-train",
                    dataset_schema={"input": "tensor[u8,28,28]", "target": "uint8"},
                    splits={"train": {"start": 0, "end": 1}},
                    label_map={"0": "zero", "1": "one"},
                    feature_stats={"input": {"mean": 0.1307, "std": 0.3081}},
                )
                w.add_sample(0, {"input": [0], "target": 0})
                w.add_sample(1, {"input": [1], "target": 1})

            with SampleShardReader(path) as r:
                meta = r.read_metadata()
                profile = r.sample_profile()
                assert meta is not None
                assert meta.profile == "sampleshard.v1"
                assert profile is not None
                assert profile.dataset_name == "mnist-train"
                assert profile.sample_id_type == "uint64"
                assert profile.key_encoding == "decimal-string"
                assert profile.sample_count == 2
                assert profile.label_map["0"] == "zero"
                assert profile.feature_stats["input"]["std"] == 0.3081

    def test_manifest_profile_roundtrip(self):
        """Test generic manifest profile metadata helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest-meta.smpl"

            with SampleShardWriter(path) as w:
                w.set_manifest_profile(
                    files=[
                        ManifestFileRef(
                            uri="s3://bucket/train-000.smpl",
                            sha256="abc123",
                            profile="sampleshard.v1",
                        )
                    ],
                    partitions={"train": ["train-000.smpl"]},
                )
                w.add_sample(0, {"x": 0})

            with SampleShardReader(path) as r:
                meta = r.read_metadata()
                manifest = r.manifest_profile()
                assert meta is not None
                assert meta.profile == "manifest.v1"
                assert manifest is not None
                assert manifest.files[0].uri == "s3://bucket/train-000.smpl"
                assert manifest.partitions["train"] == ["train-000.smpl"]


class TestSampleShardTypes:
    """Test type handling."""

    def test_various_types(self):
        """Test various JSON types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "types.smpl"

            samples = {
                1: None,
                2: True,
                3: False,
                4: 42,
                5: 3.14159,
                6: "hello",
                7: [1, 2, 3],
                8: {"key": "value"},
                9: [{"nested": [1, 2]}, "mixed", 3.14],
            }

            with SampleShardWriter(path) as w:
                for sample_id, sample in samples.items():
                    w.add_sample(sample_id, sample)

            with SampleShardReader(path) as r:
                for sample_id, expected in samples.items():
                    actual = r.get_sample(sample_id)
                    assert actual == expected, f"Mismatch for sample {sample_id}"


class TestSampleShardErrors:
    """Test error handling."""

    def test_file_not_found(self):
        """Test opening non-existent file."""
        with pytest.raises(FileNotFoundError):
            with SampleShardReader("/nonexistent/path.smpl"):
                pass

    def test_writer_closed(self):
        """Test writing to closed writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "closed.smpl"

            w = SampleShardWriter(path)
            w.open()
            w.add_sample(1, {"x": 1})
            w.close()

            with pytest.raises(RuntimeError):
                w.add_sample(2, {"x": 2})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
