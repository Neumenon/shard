"""Tests for SampleShard bridge functions.

Tests all bridge functions in sampleshard.bridges with synthetic data.
External dependencies (pandas, pyarrow, PIL) are conditionally imported
via pytest.importorskip so tests gracefully skip when unavailable.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from sampleshard import SampleShardReader
from sampleshard.bridges import from_csv, from_jsonl, from_iterable


# ---------------------------------------------------------------------------
# from_csv tests
# ---------------------------------------------------------------------------

class TestFromCSV:
    """Tests for the from_csv bridge."""

    def test_from_csv_basic(self):
        """Create a 10-row CSV, convert, verify sample count and content."""
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "data.smpl"

            # Write CSV
            rows = [{"a": i, "b": i * 10, "c": f"row{i}"} for i in range(10)]
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

            # Convert
            count = from_csv(csv_path, out_path, progress=False)
            assert count == 10

            # Verify via reader
            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 10
                profile = r.sample_profile()
                assert profile is not None
                assert profile.dataset_name == "data"
                assert profile.sample_count == 10
                assert profile.dataset_schema["fields"]["input"]["type"] == "object"
                assert (
                    profile.dataset_schema["fields"]["input"]["fields"]["a"]["type"]
                    == "int"
                )
                s0 = r.get_sample(0)
                # All columns become input dict when no target specified
                assert s0["input"]["a"] == 0
                assert s0["input"]["b"] == 0
                assert s0["input"]["c"] == "row0"

                s9 = r.get_sample(9)
                assert s9["input"]["a"] == 9
                assert s9["input"]["b"] == 90
                assert s9["input"]["c"] == "row9"

    def test_from_csv_with_target_column(self):
        """Verify target column is extracted into sample['target']."""
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [10.0, 20.0, 30.0],
                "label": [0, 1, 0],
            })
            df.to_csv(csv_path, index=False)

            count = from_csv(
                csv_path, out_path,
                target_column="label",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                s = r.get_sample(0)
                assert s["target"] == 0
                # target column should NOT be in input
                assert "label" not in s["input"]
                assert s["input"]["feat1"] == 1.0
                assert s["input"]["feat2"] == 10.0

                s2 = r.get_sample(2)
                assert s2["target"] == 0
                assert s2["input"]["feat1"] == 3.0

    def test_from_csv_max_samples(self):
        """Test max_samples truncation."""
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({
                "name": [f"item_{i}" for i in range(20)],
                "val": [i * 10 for i in range(20)],
            })
            df.to_csv(csv_path, index=False)

            count = from_csv(csv_path, out_path, max_samples=5, progress=False)
            assert count == 5

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 5
                # Last sample should be row index 4
                s = r.get_sample(4)
                assert s["input"]["name"] == "item_4"
                # Verify we don't have row 5
                assert not r.has_sample(5)

    def test_from_csv_single_input_column(self):
        """Single input column should produce a scalar input, not a dict."""
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "data.smpl"

            # Use string column to avoid numpy scalar serialization quirk
            df = pd.DataFrame({
                "text": ["hello", "world", "test"],
                "label": ["pos", "neg", "pos"],
            })
            df.to_csv(csv_path, index=False)

            count = from_csv(
                csv_path, out_path,
                input_columns=["text"],
                target_column="label",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                s = r.get_sample(0)
                # Single input column -> scalar, not dict
                assert s["input"] == "hello"
                assert s["target"] == "pos"

    def test_from_csv_with_id_column(self):
        """Test using a CSV column as the sample ID."""
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({
                "uid": [500, 600, 700],
                "data": ["a", "b", "c"],
            })
            df.to_csv(csv_path, index=False)

            count = from_csv(
                csv_path, out_path,
                id_column="uid",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                assert r.has_sample(500)
                assert r.has_sample(600)
                assert r.has_sample(700)
                assert not r.has_sample(0)

                s = r.get_sample(500)
                assert s["input"] == "a"


# ---------------------------------------------------------------------------
# from_jsonl tests
# ---------------------------------------------------------------------------

class TestFromJSONL:
    """Tests for the from_jsonl bridge."""

    def test_from_jsonl_basic(self):
        """Create a temp JSONL file, convert, verify round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "data.jsonl"
            out_path = Path(tmpdir) / "data.smpl"

            records = [
                {"text": "hello world", "label": 0},
                {"text": "foo bar", "label": 1},
                {"text": "baz qux", "label": 2},
            ]
            with open(jsonl_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            count = from_jsonl(jsonl_path, out_path, progress=False)
            assert count == 3

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 3
                profile = r.sample_profile()
                assert profile is not None
                assert profile.dataset_name == "data"
                assert profile.dataset_schema["fields"]["text"]["type"] == "string"
                assert profile.dataset_schema["fields"]["label"]["type"] == "int"
                for i, rec in enumerate(records):
                    s = r.get_sample(i)
                    assert s == rec

    def test_from_jsonl_with_id_key(self):
        """Test custom ID key extraction from JSON objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "data.jsonl"
            out_path = Path(tmpdir) / "data.smpl"

            records = [
                {"doc_id": 42, "content": "first"},
                {"doc_id": 99, "content": "second"},
                {"doc_id": 7, "content": "third"},
            ]
            with open(jsonl_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            count = from_jsonl(
                jsonl_path, out_path,
                id_key="doc_id",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                assert r.has_sample(42)
                assert r.has_sample(99)
                assert r.has_sample(7)

                s = r.get_sample(42)
                assert s["content"] == "first"
                assert s["doc_id"] == 42

    def test_from_jsonl_max_samples(self):
        """Test max_samples truncation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "data.jsonl"
            out_path = Path(tmpdir) / "data.smpl"

            with open(jsonl_path, "w") as f:
                for i in range(20):
                    f.write(json.dumps({"idx": i}) + "\n")

            count = from_jsonl(jsonl_path, out_path, max_samples=7, progress=False)
            assert count == 7

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 7
                # Last written sample should be line 6 (0-indexed)
                s = r.get_sample(6)
                assert s["idx"] == 6

    def test_from_jsonl_nested_objects(self):
        """Test with nested JSON structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "data.jsonl"
            out_path = Path(tmpdir) / "data.smpl"

            records = [
                {
                    "user": {"name": "Alice", "age": 30},
                    "scores": [95, 87, 91],
                    "meta": {"source": "test", "nested": {"deep": True}},
                },
                {
                    "user": {"name": "Bob", "age": 25},
                    "scores": [78, 82],
                    "meta": {"source": "test", "nested": {"deep": False}},
                },
            ]
            with open(jsonl_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            count = from_jsonl(jsonl_path, out_path, progress=False)
            assert count == 2

            with SampleShardReader(out_path) as r:
                s0 = r.get_sample(0)
                assert s0["user"]["name"] == "Alice"
                assert s0["scores"] == [95, 87, 91]
                assert s0["meta"]["nested"]["deep"] is True

                s1 = r.get_sample(1)
                assert s1["user"]["age"] == 25
                assert s1["meta"]["nested"]["deep"] is False

    def test_from_jsonl_empty_file(self):
        """Test with an empty JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "empty.jsonl"
            out_path = Path(tmpdir) / "empty.smpl"

            # Write empty file
            with open(jsonl_path, "w") as f:
                pass

            count = from_jsonl(jsonl_path, out_path, progress=False)
            assert count == 0

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 0
                profile = r.sample_profile()
                assert profile is not None
                assert profile.dataset_name == "empty"
                assert profile.sample_count == 0


# ---------------------------------------------------------------------------
# from_parquet tests
# ---------------------------------------------------------------------------

class TestFromParquet:
    """Tests for the from_parquet bridge (skipped if pyarrow not installed)."""

    def test_from_parquet_basic(self):
        """Create temp parquet via pandas, convert, verify."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "data.parquet"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({
                "feature_a": [1.1, 2.2, 3.3, 4.4, 5.5],
                "feature_b": [10, 20, 30, 40, 50],
                "name": ["a", "b", "c", "d", "e"],
            })
            df.to_parquet(pq_path, index=False)

            from sampleshard.bridges import from_parquet
            count = from_parquet(pq_path, out_path, progress=False)
            assert count == 5

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 5
                s0 = r.get_sample(0)
                assert s0["input"]["name"] == "a"
                assert abs(s0["input"]["feature_a"] - 1.1) < 1e-6

    def test_from_parquet_with_target(self):
        """Verify target extraction from parquet."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "data.parquet"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({
                "x1": [1.0, 2.0, 3.0],
                "x2": [4.0, 5.0, 6.0],
                "y": [0, 1, 0],
            })
            df.to_parquet(pq_path, index=False)

            from sampleshard.bridges import from_parquet
            count = from_parquet(
                pq_path, out_path,
                target_column="y",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                s = r.get_sample(0)
                assert s["target"] == 0
                assert "y" not in s["input"]
                assert abs(s["input"]["x1"] - 1.0) < 1e-6
                assert abs(s["input"]["x2"] - 4.0) < 1e-6

                s2 = r.get_sample(1)
                assert s2["target"] == 1

    def test_from_parquet_max_samples(self):
        """Test max_samples truncation on parquet."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "data.parquet"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({"v": range(50)})
            df.to_parquet(pq_path, index=False)

            from sampleshard.bridges import from_parquet
            count = from_parquet(
                pq_path, out_path,
                max_samples=10,
                progress=False,
            )
            assert count == 10

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 10
                # Verify last sample is row 9
                s = r.get_sample(9)
                assert s["input"] == 9

    def test_from_parquet_with_id_column(self):
        """Test using a parquet column as sample ID."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "data.parquet"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({
                "record_id": [1000, 2000, 3000],
                "value": ["x", "y", "z"],
            })
            df.to_parquet(pq_path, index=False)

            from sampleshard.bridges import from_parquet
            count = from_parquet(
                pq_path, out_path,
                id_column="record_id",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                assert r.has_sample(1000)
                assert r.has_sample(2000)
                assert r.has_sample(3000)
                assert not r.has_sample(0)

                s = r.get_sample(2000)
                assert s["input"] == "y"


# ---------------------------------------------------------------------------
# from_iterable tests
# ---------------------------------------------------------------------------

class TestFromIterable:
    """Tests for the from_iterable bridge."""

    def test_from_iterable_basic(self):
        """Test with a generator of (id, sample) pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "data.smpl"

            def gen():
                for i in range(8):
                    yield i, {"value": i * 3, "tag": f"item_{i}"}

            count = from_iterable(gen(), out_path, total=8, progress=False)
            assert count == 8

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 8
                for i in range(8):
                    s = r.get_sample(i)
                    assert s["value"] == i * 3
                    assert s["tag"] == f"item_{i}"

    def test_from_iterable_complex_data(self):
        """Test with nested dicts/lists in the samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "data.smpl"

            samples = [
                (0, {
                    "embeddings": [0.1, 0.2, 0.3, 0.4],
                    "metadata": {
                        "source": "wiki",
                        "sections": ["intro", "body"],
                        "stats": {"tokens": 150, "chars": 800},
                    },
                    "labels": [1, 0, 1],
                }),
                (1, {
                    "embeddings": [0.5, 0.6, 0.7, 0.8],
                    "metadata": {
                        "source": "arxiv",
                        "sections": ["abstract"],
                        "stats": {"tokens": 50, "chars": 300},
                    },
                    "labels": [0, 0, 0],
                }),
            ]

            count = from_iterable(iter(samples), out_path, progress=False)
            assert count == 2

            with SampleShardReader(out_path) as r:
                profile = r.sample_profile()
                assert profile is not None
                assert profile.dataset_name == "data"
                assert profile.dataset_schema["fields"]["embeddings"]["type"] == "array"
                s0 = r.get_sample(0)
                assert len(s0["embeddings"]) == 4
                assert abs(s0["embeddings"][0] - 0.1) < 1e-6
                assert s0["metadata"]["source"] == "wiki"
                assert s0["metadata"]["stats"]["tokens"] == 150
                assert s0["labels"] == [1, 0, 1]

                s1 = r.get_sample(1)
                assert s1["metadata"]["sections"] == ["abstract"]

    def test_from_iterable_non_sequential_ids(self):
        """Test with non-sequential sample IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "data.smpl"

            pairs = [(100, "alpha"), (5, "beta"), (999, "gamma")]
            count = from_iterable(iter(pairs), out_path, progress=False)
            assert count == 3

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 3
                assert r.get_sample(100) == "alpha"
                assert r.get_sample(5) == "beta"
                assert r.get_sample(999) == "gamma"

                # Iteration should be ascending ID order
                ids = r.sample_ids()
                assert ids == [5, 100, 999]

    def test_from_iterable_empty(self):
        """Test with an empty iterable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "data.smpl"

            count = from_iterable(iter([]), out_path, progress=False)
            assert count == 0

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 0

    def test_from_iterable_scalar_samples(self):
        """Test with scalar (non-dict) sample values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "data.smpl"

            pairs = [
                (0, 42),
                (1, 3.14),
                (2, "hello"),
                (3, True),
                (4, None),
                (5, [1, 2, 3]),
            ]
            count = from_iterable(iter(pairs), out_path, progress=False)
            assert count == 6

            with SampleShardReader(out_path) as r:
                assert r.get_sample(0) == 42
                assert abs(r.get_sample(1) - 3.14) < 1e-9
                assert r.get_sample(2) == "hello"
                assert r.get_sample(3) is True
                assert r.get_sample(4) is None
                assert r.get_sample(5) == [1, 2, 3]


# ---------------------------------------------------------------------------
# from_image_folder tests
# ---------------------------------------------------------------------------

class TestFromImageFolder:
    """Tests for from_image_folder (skipped if PIL not installed)."""

    def _make_image_folder(self, root: Path, classes: dict):
        """Helper: create a folder structure with tiny PNG images.

        Args:
            root: Root directory.
            classes: Dict mapping class_name -> number of images.
        """
        from PIL import Image

        for class_name, n_images in classes.items():
            class_dir = root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for j in range(n_images):
                img = Image.new("RGB", (4, 4), color=(j * 30, j * 10, 100))
                img.save(class_dir / f"img_{j}.png")

    def test_from_image_folder_basic(self):
        """Create temp image folder with class subdirs, convert, verify."""
        Image = pytest.importorskip("PIL.Image")
        np = pytest.importorskip("numpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_root = Path(tmpdir) / "images"
            out_path = Path(tmpdir) / "images.smpl"

            self._make_image_folder(img_root, {"cat": 3, "dog": 2})

            from sampleshard.bridges import from_image_folder
            count = from_image_folder(img_root, out_path, progress=False)
            assert count == 5

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 5

                # Check that each sample has input, target, and meta
                for sid, sample in r:
                    assert "input" in sample
                    assert "target" in sample
                    assert "meta" in sample
                    assert "class_name" in sample["meta"]
                    assert "shape" in sample["meta"]
                    # Shape should be [4, 4, 3] for 4x4 RGB
                    assert sample["meta"]["shape"] == [4, 4, 3]

    def test_from_image_folder_class_mapping(self):
        """Verify class names map to correct integer targets."""
        pytest.importorskip("PIL.Image")
        pytest.importorskip("numpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_root = Path(tmpdir) / "images"
            out_path = Path(tmpdir) / "images.smpl"

            # Alphabetical: "alpha"=0, "beta"=1, "gamma"=2
            self._make_image_folder(img_root, {"beta": 1, "alpha": 1, "gamma": 1})

            from sampleshard.bridges import from_image_folder
            count = from_image_folder(img_root, out_path, progress=False)
            assert count == 3

            with SampleShardReader(out_path) as r:
                profile = r.sample_profile()
                assert profile is not None
                assert profile.dataset_name == "images"
                assert profile.label_map == {"0": "alpha", "1": "beta", "2": "gamma"}
                class_targets = {}
                for sid, sample in r:
                    cn = sample["meta"]["class_name"]
                    class_targets[cn] = sample["target"]

                assert class_targets["alpha"] == 0
                assert class_targets["beta"] == 1
                assert class_targets["gamma"] == 2

    def test_from_image_folder_max_samples(self):
        """Test max_samples truncation on image folders."""
        pytest.importorskip("PIL.Image")
        pytest.importorskip("numpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_root = Path(tmpdir) / "images"
            out_path = Path(tmpdir) / "images.smpl"

            self._make_image_folder(img_root, {"a": 5, "b": 5})

            from sampleshard.bridges import from_image_folder
            count = from_image_folder(
                img_root, out_path,
                max_samples=4,
                progress=False,
            )
            assert count == 4

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 4

    def test_from_image_folder_with_resize(self):
        """Test resize parameter produces correct output shape."""
        pytest.importorskip("PIL.Image")
        pytest.importorskip("numpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_root = Path(tmpdir) / "images"
            out_path = Path(tmpdir) / "images.smpl"

            self._make_image_folder(img_root, {"cls": 2})

            from sampleshard.bridges import from_image_folder
            count = from_image_folder(
                img_root, out_path,
                resize=(8, 6),  # width=8, height=6
                progress=False,
            )
            assert count == 2

            with SampleShardReader(out_path) as r:
                s = r.get_sample(0)
                # PIL resize (width, height) -> numpy shape (height, width, channels)
                assert s["meta"]["shape"] == [6, 8, 3]

    def test_from_image_folder_include_path(self):
        """Test include_path stores relative file path in metadata."""
        pytest.importorskip("PIL.Image")
        pytest.importorskip("numpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_root = Path(tmpdir) / "images"
            out_path = Path(tmpdir) / "images.smpl"

            self._make_image_folder(img_root, {"myclass": 1})

            from sampleshard.bridges import from_image_folder
            count = from_image_folder(
                img_root, out_path,
                include_path=True,
                progress=False,
            )
            assert count == 1

            with SampleShardReader(out_path) as r:
                s = r.get_sample(0)
                assert "path" in s["meta"]
                # Path should be relative: myclass/img_0.png
                assert s["meta"]["path"] == os.path.join("myclass", "img_0.png")


# ---------------------------------------------------------------------------
# Cross-cutting / integration tests
# ---------------------------------------------------------------------------

class TestBridgeIntegration:
    """Integration tests combining multiple bridge aspects."""

    def test_csv_to_shard_iteration_order(self):
        """Verify iteration order is ascending sample ID regardless of write order."""
        pd = pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "data.smpl"

            df = pd.DataFrame({"v": range(5)})
            df.to_csv(csv_path, index=False)

            from_csv(csv_path, out_path, progress=False)

            with SampleShardReader(out_path) as r:
                prev_id = -1
                for sid, _ in r:
                    assert sid > prev_id
                    prev_id = sid

    def test_jsonl_large_batch(self):
        """Test JSONL bridge with a larger number of records and batch read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "large.jsonl"
            out_path = Path(tmpdir) / "large.smpl"

            n = 100
            with open(jsonl_path, "w") as f:
                for i in range(n):
                    f.write(json.dumps({"i": i, "sq": i * i}) + "\n")

            count = from_jsonl(jsonl_path, out_path, progress=False)
            assert count == n

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == n

                # Batch read a subset
                batch = r.get_batch([0, 50, 99])
                assert len(batch) == 3
                assert batch[0]["i"] == 0
                assert batch[1]["sq"] == 2500
                assert batch[2]["i"] == 99

    def test_iterable_then_mmap_read(self):
        """Write via from_iterable, then read with mmap enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "data.smpl"

            pairs = [(i, {"val": i}) for i in range(10)]
            from_iterable(iter(pairs), out_path, progress=False)

            with SampleShardReader(out_path) as r:
                r.enable_mmap()
                assert r.sample_count() == 10
                for i in range(10):
                    assert r.get_sample(i)["val"] == i


# ---------------------------------------------------------------------------
# from_arrow tests
# ---------------------------------------------------------------------------

class TestFromArrow:
    """Tests for the from_arrow bridge (skipped if pyarrow not installed)."""

    def test_from_arrow_basic(self):
        """Create a temp Feather file, convert, verify 3 samples with correct content."""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.feather

        from sampleshard.bridges import from_arrow

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / "data.feather"
            out_path = Path(tmpdir) / "data.smpl"

            table = pa.table({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            pa.feather.write_feather(table, str(arrow_path))

            count = from_arrow(arrow_path, out_path, progress=False)
            assert count == 3

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 3

                s0 = r.get_sample(0)
                assert s0["input"]["col1"] == 1
                assert s0["input"]["col2"] == 4

                s1 = r.get_sample(1)
                assert s1["input"]["col1"] == 2
                assert s1["input"]["col2"] == 5

                s2 = r.get_sample(2)
                assert s2["input"]["col1"] == 3
                assert s2["input"]["col2"] == 6

    def test_from_arrow_with_target(self):
        """Verify target column extraction."""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.feather

        from sampleshard.bridges import from_arrow

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / "data.feather"
            out_path = Path(tmpdir) / "data.smpl"

            table = pa.table({
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [10.0, 20.0, 30.0],
                "label": [0, 1, 0],
            })
            pa.feather.write_feather(table, str(arrow_path))

            count = from_arrow(
                arrow_path, out_path,
                target_column="label",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                s = r.get_sample(0)
                assert s["target"] == 0
                assert "label" not in s["input"]
                assert abs(s["input"]["feat1"] - 1.0) < 1e-6
                assert abs(s["input"]["feat2"] - 10.0) < 1e-6

                s2 = r.get_sample(2)
                assert s2["target"] == 0
                assert abs(s2["input"]["feat1"] - 3.0) < 1e-6

    def test_from_arrow_max_samples(self):
        """Write 10 rows, convert max_samples=5, verify 5."""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.feather

        from sampleshard.bridges import from_arrow

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / "data.feather"
            out_path = Path(tmpdir) / "data.smpl"

            table = pa.table({
                "x": list(range(10)),
                "y": list(range(10, 20)),
            })
            pa.feather.write_feather(table, str(arrow_path))

            count = from_arrow(
                arrow_path, out_path,
                max_samples=5,
                progress=False,
            )
            assert count == 5

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 5
                s4 = r.get_sample(4)
                assert s4["input"]["x"] == 4
                assert s4["input"]["y"] == 14
                assert not r.has_sample(5)


# ---------------------------------------------------------------------------
# from_hdf5 tests
# ---------------------------------------------------------------------------

class TestFromHDF5:
    """Tests for the from_hdf5 bridge (skipped if h5py not installed)."""

    def test_from_hdf5_basic(self):
        """Create a temp HDF5 file with a dataset of shape (10, 4), convert, verify."""
        h5py = pytest.importorskip("h5py")
        import numpy as np

        from sampleshard.bridges import from_hdf5

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "data.h5"
            out_path = Path(tmpdir) / "data.smpl"

            data = np.arange(40, dtype=np.float32).reshape(10, 4)
            with h5py.File(str(h5_path), "w") as f:
                f.create_dataset("features", data=data)

            count = from_hdf5(
                h5_path, out_path,
                dataset_name="features",
                progress=False,
            )
            assert count == 10

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 10

                s0 = r.get_sample(0)
                assert s0["input"] == [0.0, 1.0, 2.0, 3.0]
                assert s0["meta"]["shape"] == [4]

                s9 = r.get_sample(9)
                assert s9["input"] == [36.0, 37.0, 38.0, 39.0]

    def test_from_hdf5_with_targets(self):
        """Create features and labels datasets, convert with target_dataset."""
        h5py = pytest.importorskip("h5py")
        import numpy as np

        from sampleshard.bridges import from_hdf5

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "data.h5"
            out_path = Path(tmpdir) / "data.smpl"

            features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
            labels = np.array([0, 1, 0], dtype=np.int64)

            with h5py.File(str(h5_path), "w") as f:
                f.create_dataset("features", data=features)
                f.create_dataset("labels", data=labels)

            count = from_hdf5(
                h5_path, out_path,
                dataset_name="features",
                target_dataset="labels",
                progress=False,
            )
            assert count == 3

            with SampleShardReader(out_path) as r:
                s0 = r.get_sample(0)
                assert s0["input"] == [1.0, 2.0]
                assert s0["target"] == 0

                s1 = r.get_sample(1)
                assert s1["input"] == [3.0, 4.0]
                assert s1["target"] == 1

                s2 = r.get_sample(2)
                assert s2["target"] == 0

    def test_from_hdf5_max_samples(self):
        """Write 20 rows, max_samples=5."""
        h5py = pytest.importorskip("h5py")
        import numpy as np

        from sampleshard.bridges import from_hdf5

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "data.h5"
            out_path = Path(tmpdir) / "data.smpl"

            data = np.arange(80, dtype=np.float32).reshape(20, 4)
            with h5py.File(str(h5_path), "w") as f:
                f.create_dataset("data", data=data)

            count = from_hdf5(
                h5_path, out_path,
                dataset_name="data",
                max_samples=5,
                progress=False,
            )
            assert count == 5

            with SampleShardReader(out_path) as r:
                assert r.sample_count() == 5
                s4 = r.get_sample(4)
                assert s4["input"] == [16.0, 17.0, 18.0, 19.0]
                assert not r.has_sample(5)

    def test_from_hdf5_missing_dataset(self):
        """Verify KeyError with helpful message listing available datasets."""
        h5py = pytest.importorskip("h5py")
        import numpy as np

        from sampleshard.bridges import from_hdf5

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "data.h5"
            out_path = Path(tmpdir) / "data.smpl"

            with h5py.File(str(h5_path), "w") as f:
                f.create_dataset("real_data", data=np.zeros(10))
                f.create_dataset("labels", data=np.zeros(10))

            with pytest.raises(KeyError, match="nonexistent"):
                from_hdf5(
                    h5_path, out_path,
                    dataset_name="nonexistent",
                    progress=False,
                )


# ---------------------------------------------------------------------------
# from_protobuf tests (varint decoder unit tests)
# ---------------------------------------------------------------------------

class TestFromProtobuf:
    """Tests for the from_protobuf bridge varint decoder."""

    def test_decode_varint_simple(self):
        """Test _decode_varint with a simple single-byte varint."""
        from sampleshard.bridges import _decode_varint

        value, new_pos = _decode_varint(b'\x08', 0)
        assert value == 8
        assert new_pos == 1

    def test_decode_varint_multi_byte(self):
        """Test _decode_varint with a multi-byte varint encoding 300."""
        from sampleshard.bridges import _decode_varint

        # 300 = 0b100101100 -> varint: 0xAC 0x02
        value, new_pos = _decode_varint(b'\xac\x02', 0)
        assert value == 300
        assert new_pos == 2

    def test_decode_varint_truncated(self):
        """Test _decode_varint raises ValueError on truncated input."""
        from sampleshard.bridges import _decode_varint

        # 0x80 has continuation bit set but no following byte
        with pytest.raises(ValueError, match="Truncated varint"):
            _decode_varint(b'\x80', 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
