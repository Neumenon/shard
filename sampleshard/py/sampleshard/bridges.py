"""
Bridge functions for converting popular dataset formats to SampleShard.

Supported sources:
- HuggingFace datasets (datasets library)
- Parquet files (pandas/pyarrow)
- Image folders (PIL)
- CSV files (pandas)
- JSON/JSONL files
- WebDataset shards (optional)
- TFRecord files (optional, requires tensorflow)
- Apache Arrow IPC/Feather files (optional, requires pyarrow)
- HDF5 files (optional, requires h5py)
- Protobuf message files (optional, requires protobuf)

Example:
    >>> from sampleshard.bridges import from_hf_dataset, from_image_folder
    >>>
    >>> # HuggingFace dataset
    >>> from_hf_dataset("mnist", "train.smpl", split="train")
    >>>
    >>> # Image folder (ImageNet-style)
    >>> from_image_folder("data/train/", "imagenet_train.smpl")
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import json

from .writer import SampleShardWriter


def from_hf_dataset(
    dataset_name: str,
    output_path: Union[str, Path],
    split: str = "train",
    config: Optional[str] = None,
    input_key: str = "image",
    target_key: str = "label",
    max_samples: Optional[int] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
    progress: bool = True,
) -> int:
    """
    Convert a HuggingFace dataset to SampleShard format.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "mnist", "cifar10")
        output_path: Output .smpl file path
        split: Dataset split ("train", "test", "validation")
        config: Dataset configuration name (if applicable)
        input_key: Key for input data in the dataset
        target_key: Key for target/label in the dataset
        max_samples: Maximum number of samples to convert (None = all)
        transform: Optional transform function applied to each sample
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> from_hf_dataset("mnist", "mnist_train.smpl", split="train")
        60000
        >>> from_hf_dataset(
        ...     "imagenet-1k",
        ...     "imagenet.smpl",
        ...     split="train",
        ...     max_samples=10000,
        ... )
        10000
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("HuggingFace datasets library required: pip install datasets")

    # Load dataset
    ds = load_dataset(dataset_name, config, split=split)

    count = 0
    total = len(ds) if max_samples is None else min(len(ds), max_samples)

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                enumerate(ds), total=total, desc=f"Converting {dataset_name}"
            )
        except ImportError:
            iterator = enumerate(ds)
    else:
        iterator = enumerate(ds)

    with SampleShardWriter(output_path) as writer:
        for i, item in iterator:
            if max_samples is not None and i >= max_samples:
                break

            # Build sample
            sample = {}

            # Handle input (could be image, array, etc.)
            if input_key in item:
                inp = item[input_key]
                # Convert PIL Image to list if needed
                if hasattr(inp, "tobytes"):
                    import numpy as np

                    arr = np.array(inp)
                    sample["input"] = arr.tolist()
                    sample["meta"] = {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                else:
                    sample["input"] = inp

            # Handle target
            if target_key in item:
                sample["target"] = item[target_key]

            # Apply transform if provided
            if transform is not None:
                sample = transform(sample)

            writer.add_sample(i, sample)
            count += 1

    return count


def from_parquet(
    parquet_path: Union[str, Path],
    output_path: Union[str, Path],
    input_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert a Parquet file to SampleShard format.

    Args:
        parquet_path: Input Parquet file path
        output_path: Output .smpl file path
        input_columns: Columns to include as input (None = all except target)
        target_column: Column to use as target/label
        id_column: Column to use as sample ID (None = row index)
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> from_parquet(
        ...     "train.parquet",
        ...     "train.smpl",
        ...     input_columns=["feature1", "feature2"],
        ...     target_column="label",
        ... )
        50000
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas pyarrow")

    df = pd.read_parquet(parquet_path)

    if max_samples is not None:
        df = df.head(max_samples)

    # Determine input columns
    if input_columns is None:
        input_columns = [c for c in df.columns if c != target_column and c != id_column]

    count = 0
    total = len(df)

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(df.iterrows(), total=total, desc="Converting Parquet")
        except ImportError:
            iterator = df.iterrows()
    else:
        iterator = df.iterrows()

    with SampleShardWriter(output_path) as writer:
        for idx, row in iterator:
            # Determine sample ID
            if id_column is not None:
                sample_id = int(row[id_column])
            else:
                sample_id = int(idx)

            # Build sample
            sample = {}

            # Input: dict of column values or single value
            if len(input_columns) == 1:
                val = row[input_columns[0]]
                sample["input"] = val.tolist() if hasattr(val, "tolist") else val
            else:
                sample["input"] = {
                    col: row[col].tolist() if hasattr(row[col], "tolist") else row[col]
                    for col in input_columns
                }

            # Target
            if target_column is not None and target_column in row:
                val = row[target_column]
                sample["target"] = val.tolist() if hasattr(val, "tolist") else val

            writer.add_sample(sample_id, sample)
            count += 1

    return count


def from_image_folder(
    folder_path: Union[str, Path],
    output_path: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    include_path: bool = False,
    resize: Optional[Tuple[int, int]] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert an image folder (ImageNet-style) to SampleShard format.

    Expects folder structure:
        folder/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                ...

    Args:
        folder_path: Root folder containing class subfolders
        output_path: Output .smpl file path
        extensions: Valid image extensions
        include_path: Include original file path in metadata
        resize: Optional (width, height) to resize images
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> from_image_folder("imagenet/train/", "imagenet_train.smpl")
        1281167
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("PIL and numpy required: pip install pillow numpy")

    folder = Path(folder_path)

    # Collect all images with their class labels
    images: List[Tuple[Path, str]] = []
    class_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                images.append((img_path, class_name))

    # Build class to index mapping
    class_names = sorted(set(c for _, c in images))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    if max_samples is not None:
        images = images[:max_samples]

    count = 0
    total = len(images)

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(enumerate(images), total=total, desc="Converting images")
        except ImportError:
            iterator = enumerate(images)
    else:
        iterator = enumerate(images)

    with SampleShardWriter(output_path) as writer:
        for i, (img_path, class_name) in iterator:
            try:
                img = Image.open(img_path).convert("RGB")

                if resize is not None:
                    img = img.resize(resize, Image.Resampling.BILINEAR)

                arr = np.array(img)

                sample = {
                    "input": arr.tolist(),
                    "target": class_to_idx[class_name],
                    "meta": {
                        "class_name": class_name,
                        "shape": list(arr.shape),
                    },
                }

                if include_path:
                    sample["meta"]["path"] = str(img_path.relative_to(folder))

                writer.add_sample(i, sample)
                count += 1

            except Exception as e:
                print(f"Warning: failed to process {img_path}: {e}")

    # Write class mapping as metadata
    # TODO: Add support for __schema__ entry with class names

    return count


def from_csv(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    input_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
    **csv_kwargs,
) -> int:
    """
    Convert a CSV file to SampleShard format.

    Args:
        csv_path: Input CSV file path
        output_path: Output .smpl file path
        input_columns: Columns to include as input (None = all except target)
        target_column: Column to use as target/label
        id_column: Column to use as sample ID (None = row index)
        max_samples: Maximum samples to convert
        progress: Show progress bar
        **csv_kwargs: Additional arguments passed to pandas.read_csv

    Returns:
        Number of samples written
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path, **csv_kwargs)

    if max_samples is not None:
        df = df.head(max_samples)

    if input_columns is None:
        input_columns = [c for c in df.columns if c != target_column and c != id_column]

    count = 0

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(df.iterrows(), total=len(df), desc="Converting CSV")
        except ImportError:
            iterator = df.iterrows()
    else:
        iterator = df.iterrows()

    with SampleShardWriter(output_path) as writer:
        for idx, row in iterator:
            sample_id = int(row[id_column]) if id_column else int(idx)

            sample = {}
            if len(input_columns) == 1:
                sample["input"] = row[input_columns[0]]
            else:
                sample["input"] = {col: row[col] for col in input_columns}

            if target_column and target_column in row:
                sample["target"] = row[target_column]

            writer.add_sample(sample_id, sample)
            count += 1

    return count


def from_jsonl(
    jsonl_path: Union[str, Path],
    output_path: Union[str, Path],
    id_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert a JSONL file to SampleShard format.

    Each line in the JSONL file becomes one sample.

    Args:
        jsonl_path: Input JSONL file path
        output_path: Output .smpl file path
        id_key: Key in each JSON object to use as sample ID (None = line number)
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written
    """
    jsonl_path = Path(jsonl_path)

    # Count lines for progress bar
    if progress:
        with open(jsonl_path) as f:
            total = sum(1 for _ in f)
        if max_samples:
            total = min(total, max_samples)
    else:
        total = None

    count = 0

    with open(jsonl_path) as f:
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(enumerate(f), total=total, desc="Converting JSONL")
            except ImportError:
                iterator = enumerate(f)
        else:
            iterator = enumerate(f)

        with SampleShardWriter(output_path) as writer:
            for i, line in iterator:
                if max_samples is not None and i >= max_samples:
                    break

                obj = json.loads(line.strip())

                if id_key is not None and id_key in obj:
                    sample_id = int(obj[id_key])
                else:
                    sample_id = i

                writer.add_sample(sample_id, obj)
                count += 1

    return count


def from_iterable(
    iterable: Iterator[Tuple[int, Any]],
    output_path: Union[str, Path],
    total: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert any iterable of (id, sample) pairs to SampleShard.

    This is the most flexible bridge - use it when you have a custom data source.

    Args:
        iterable: Iterator yielding (sample_id, sample_data) tuples
        output_path: Output .smpl file path
        total: Total count for progress bar (optional)
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> def my_data():
        ...     for i in range(1000):
        ...         yield i, {"input": [i], "target": i % 10}
        >>> from_iterable(my_data(), "custom.smpl", total=1000)
        1000
    """
    count = 0

    if progress and total is not None:
        try:
            from tqdm import tqdm

            iterable = tqdm(iterable, total=total, desc="Converting")
        except ImportError:
            pass

    with SampleShardWriter(output_path) as writer:
        for sample_id, sample in iterable:
            writer.add_sample(sample_id, sample)
            count += 1

    return count


# Optional bridges (require additional dependencies)


def from_tfrecord(
    tfrecord_path: Union[str, Path],
    output_path: Union[str, Path],
    feature_spec: Dict[str, Any],
    input_key: str = "image",
    target_key: str = "label",
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert a TFRecord file to SampleShard format.

    Requires: pip install tensorflow

    Args:
        tfrecord_path: Input TFRecord file path
        output_path: Output .smpl file path
        feature_spec: TensorFlow feature description dict
        input_key: Key for input feature
        target_key: Key for target feature
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> feature_spec = {
        ...     "image": tf.io.FixedLenFeature([], tf.string),
        ...     "label": tf.io.FixedLenFeature([], tf.int64),
        ... }
        >>> from_tfrecord("train.tfrecord", "train.smpl", feature_spec)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required: pip install tensorflow")

    def parse_fn(example):
        return tf.io.parse_single_example(example, feature_spec)

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(parse_fn)

    if max_samples:
        dataset = dataset.take(max_samples)

    count = 0

    with SampleShardWriter(output_path) as writer:
        for i, item in enumerate(dataset):
            sample = {}

            if input_key in item:
                inp = item[input_key].numpy()
                if isinstance(inp, bytes):
                    # Decode image bytes
                    import numpy as np

                    img = tf.io.decode_image(inp).numpy()
                    sample["input"] = img.tolist()
                else:
                    sample["input"] = inp.tolist() if hasattr(inp, "tolist") else inp

            if target_key in item:
                target = item[target_key].numpy()
                sample["target"] = int(target) if target.ndim == 0 else target.tolist()

            writer.add_sample(i, sample)
            count += 1

            if progress and count % 1000 == 0:
                print(f"Converted {count} samples...")

    return count


def from_webdataset(
    wds_path: Union[str, Path],
    output_path: Union[str, Path],
    input_key: str = "jpg",
    target_key: str = "cls",
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert a WebDataset shard to SampleShard format.

    Requires: pip install webdataset

    Args:
        wds_path: Input WebDataset tar file path
        output_path: Output .smpl file path
        input_key: Key for input (e.g., "jpg", "png")
        target_key: Key for target (e.g., "cls", "json")
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written
    """
    try:
        import webdataset as wds
    except ImportError:
        raise ImportError("WebDataset required: pip install webdataset")

    try:
        from PIL import Image
        import numpy as np
        import io
    except ImportError:
        raise ImportError("PIL and numpy required: pip install pillow numpy")

    dataset = wds.WebDataset(str(wds_path))

    count = 0

    with SampleShardWriter(output_path) as writer:
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            sample = {}

            # Handle image input
            if input_key in item:
                img_bytes = item[input_key]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                arr = np.array(img)
                sample["input"] = arr.tolist()
                sample["meta"] = {"shape": list(arr.shape)}

            # Handle target
            if target_key in item:
                target = item[target_key]
                if isinstance(target, bytes):
                    target = target.decode("utf-8")
                try:
                    sample["target"] = int(target)
                except ValueError:
                    sample["target"] = target

            # Include __key__ as meta
            if "__key__" in item:
                if "meta" not in sample:
                    sample["meta"] = {}
                sample["meta"]["key"] = item["__key__"]

            writer.add_sample(i, sample)
            count += 1

            if progress and count % 1000 == 0:
                print(f"Converted {count} samples...")

    return count


def from_arrow(
    arrow_path: Union[str, Path],
    output_path: Union[str, Path],
    input_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert an Apache Arrow IPC/Feather file to SampleShard format.

    Reads Arrow IPC (streaming or file format) and Feather v2 files via
    pyarrow and writes each row as a SampleShard sample.

    Requires: pip install pyarrow

    Args:
        arrow_path: Input Arrow IPC or Feather file path
        output_path: Output .smpl file path
        input_columns: Columns to include as input (None = all except target)
        target_column: Column to use as target/label
        id_column: Column to use as sample ID (None = row index)
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> from_arrow(
        ...     "embeddings.arrow",
        ...     "embeddings.smpl",
        ...     input_columns=["embedding"],
        ...     target_column="label",
        ... )
        25000
        >>> from_arrow("features.feather", "features.smpl")
        100000
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc
        import pyarrow.feather
    except ImportError:
        raise ImportError("pyarrow required: pip install pyarrow")

    # Try Feather first, fall back to IPC stream/file
    arrow_path = Path(arrow_path)
    try:
        table = pa.feather.read_table(str(arrow_path))
    except (pa.ArrowInvalid, Exception):
        with open(arrow_path, "rb") as f:
            reader = pa.ipc.open_file(f)
            table = reader.read_all()

    if max_samples is not None:
        table = table.slice(0, max_samples)

    # Determine input columns
    all_columns = table.column_names
    if input_columns is None:
        input_columns = [c for c in all_columns if c != target_column and c != id_column]

    count = 0
    total = table.num_rows

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(range(total), total=total, desc="Converting Arrow")
        except ImportError:
            iterator = range(total)
    else:
        iterator = range(total)

    with SampleShardWriter(output_path) as writer:
        for i in iterator:
            # Determine sample ID
            if id_column is not None:
                sample_id = int(table.column(id_column)[i].as_py())
            else:
                sample_id = i

            # Build sample
            sample = {}

            # Input: dict of column values or single value
            if len(input_columns) == 1:
                val = table.column(input_columns[0])[i].as_py()
                sample["input"] = val
            else:
                sample["input"] = {
                    col: table.column(col)[i].as_py() for col in input_columns
                }

            # Target
            if target_column is not None and target_column in all_columns:
                sample["target"] = table.column(target_column)[i].as_py()

            writer.add_sample(sample_id, sample)
            count += 1

    return count


def from_hdf5(
    hdf5_path: Union[str, Path],
    output_path: Union[str, Path],
    dataset_name: str,
    target_dataset: Optional[str] = None,
    id_dataset: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert an HDF5 dataset to SampleShard format.

    Reads a named dataset from an HDF5 file and writes each row (first axis)
    as a SampleShard sample. Multi-dimensional arrays are preserved as nested
    lists.

    Requires: pip install h5py

    Args:
        hdf5_path: Input HDF5 file path
        output_path: Output .smpl file path
        dataset_name: Name of the HDF5 dataset to read (e.g., "features",
            "images", "group1/data")
        target_dataset: Optional HDF5 dataset name for targets/labels
        id_dataset: Optional HDF5 dataset name for sample IDs
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> from_hdf5(
        ...     "features.h5",
        ...     "features.smpl",
        ...     dataset_name="embeddings",
        ...     target_dataset="labels",
        ... )
        50000
        >>> from_hdf5("images.h5", "images.smpl", dataset_name="train/images")
        60000
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        raise ImportError("h5py and numpy required: pip install h5py numpy")

    with h5py.File(str(hdf5_path), "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {hdf5_path}. "
                f"Available datasets: {available}"
            )

        ds = f[dataset_name]
        total = ds.shape[0]

        if max_samples is not None:
            total = min(total, max_samples)

        # Load target dataset handle if specified
        target_ds = None
        if target_dataset is not None:
            if target_dataset not in f:
                raise KeyError(
                    f"Target dataset '{target_dataset}' not found in {hdf5_path}"
                )
            target_ds = f[target_dataset]

        # Load ID dataset handle if specified
        id_ds = None
        if id_dataset is not None:
            if id_dataset not in f:
                raise KeyError(
                    f"ID dataset '{id_dataset}' not found in {hdf5_path}"
                )
            id_ds = f[id_dataset]

        count = 0

        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(total), total=total, desc="Converting HDF5")
            except ImportError:
                iterator = range(total)
        else:
            iterator = range(total)

        with SampleShardWriter(output_path) as writer:
            for i in iterator:
                # Determine sample ID
                if id_ds is not None:
                    sample_id = int(id_ds[i])
                else:
                    sample_id = i

                # Read input data
                row = ds[i]
                if isinstance(row, np.ndarray):
                    input_val = row.tolist()
                elif isinstance(row, (np.integer, np.floating)):
                    input_val = row.item()
                else:
                    input_val = row

                sample = {
                    "input": input_val,
                    "meta": {
                        "shape": list(ds.shape[1:]),
                        "dtype": str(ds.dtype),
                    },
                }

                # Target
                if target_ds is not None:
                    target = target_ds[i]
                    if isinstance(target, np.ndarray):
                        sample["target"] = target.tolist()
                    elif isinstance(target, (np.integer, np.floating)):
                        sample["target"] = target.item()
                    else:
                        sample["target"] = target

                writer.add_sample(sample_id, sample)
                count += 1

    return count


def from_protobuf(
    proto_path: Union[str, Path],
    output_path: Union[str, Path],
    message_class: Any,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> int:
    """
    Convert a file of length-delimited serialized protobuf messages to
    SampleShard format.

    Reads varint-delimited protobuf messages from a binary file, converts
    each to a dict via MessageToDict(), and writes it as a SampleShard sample.

    Requires: pip install protobuf

    Args:
        proto_path: Input file containing serialized protobuf messages
        output_path: Output .smpl file path
        message_class: The protobuf message class (e.g., MyProto) to
            deserialize each record into
        max_samples: Maximum samples to convert
        progress: Show progress bar

    Returns:
        Number of samples written

    Example:
        >>> from my_proto_pb2 import TrainingSample
        >>> from_protobuf(
        ...     "training_data.pb",
        ...     "training.smpl",
        ...     message_class=TrainingSample,
        ... )
        10000
    """
    try:
        from google.protobuf.json_format import MessageToDict
    except ImportError:
        raise ImportError("protobuf required: pip install protobuf")

    proto_path = Path(proto_path)
    data = proto_path.read_bytes()
    pos = 0
    messages = []

    # Parse varint-delimited messages
    while pos < len(data):
        if max_samples is not None and len(messages) >= max_samples:
            break

        # Read varint length prefix
        msg_len, new_pos = _decode_varint(data, pos)
        pos = new_pos

        if pos + msg_len > len(data):
            break

        msg = message_class()
        msg.ParseFromString(data[pos : pos + msg_len])
        messages.append(msg)
        pos += msg_len

    count = 0
    total = len(messages)

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                enumerate(messages), total=total, desc="Converting Protobuf"
            )
        except ImportError:
            iterator = enumerate(messages)
    else:
        iterator = enumerate(messages)

    with SampleShardWriter(output_path) as writer:
        for i, msg in iterator:
            sample = MessageToDict(msg, preserving_proto_field_name=True)
            writer.add_sample(i, sample)
            count += 1

    return count


def _decode_varint(data: bytes, pos: int) -> tuple:
    """Decode a protobuf varint from data at the given position.

    Returns (value, new_position).
    """
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated varint")
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, pos
