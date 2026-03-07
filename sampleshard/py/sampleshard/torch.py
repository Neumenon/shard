"""
PyTorch integration for SampleShard.

Provides a torch.utils.data.Dataset implementation that wraps SampleShardReader,
enabling seamless integration with PyTorch DataLoader.

Example:
    >>> from sampleshard.torch import SampleShardDataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> dataset = SampleShardDataset("train.smpl")
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    >>>
    >>> for batch in loader:
    ...     inputs, targets = batch["input"], batch["target"]
    ...     # training step...
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .reader import SampleShardReader


class SampleShardDataset:
    """
    PyTorch Dataset wrapper for SampleShard files.

    Implements the PyTorch Dataset interface (__len__, __getitem__) for
    seamless integration with DataLoader.

    Features:
    - Memory-mapped I/O for efficient random access
    - Optional transforms for data augmentation
    - Automatic tensor conversion (if return_tensors=True)
    - Multi-worker compatible

    Example:
        >>> dataset = SampleShardDataset(
        ...     "train.smpl",
        ...     transform=lambda x: {"input": torch.tensor(x["input"]), ...}
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[Callable[[Dict], Dict]] = None,
        return_tensors: bool = False,
        mmap: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            path: Path to .smpl file
            transform: Optional transform applied to each sample
            return_tensors: If True, convert arrays to torch tensors
            mmap: If True, use memory-mapped I/O (faster for random access)
        """
        self.path = Path(path)
        self.transform = transform
        self.return_tensors = return_tensors
        self.use_mmap = mmap

        # Open reader and build index
        self._reader = SampleShardReader(self.path)
        self._reader.open()

        if mmap:
            self._reader.enable_mmap()

        self._sample_ids = self._reader.sample_ids()

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Args:
            idx: Sample index (0 to len-1)

        Returns:
            Sample dict, optionally transformed
        """
        sample = self._reader.get_sample_by_index(idx)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_tensors:
            sample = self._to_tensors(sample)

        return sample

    def _to_tensors(self, sample: Dict) -> Dict:
        """Convert arrays in sample to torch tensors."""
        try:
            import torch
            import numpy as np
        except ImportError:
            raise ImportError("PyTorch required: pip install torch")

        result = {}
        for key, value in sample.items():
            if isinstance(value, list):
                try:
                    arr = np.array(value)
                    result[key] = torch.from_numpy(arr)
                except (ValueError, TypeError):
                    result[key] = value
            elif isinstance(value, (int, float)):
                result[key] = torch.tensor(value)
            else:
                result[key] = value

        return result

    def get_sample_id(self, idx: int) -> int:
        """Get the sample ID for a given index."""
        return self._sample_ids[idx]

    def close(self):
        """Close the underlying reader."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self):
        self.close()

    # Support for pickling (needed for multi-worker DataLoader)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_sample_ids"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-open reader in the worker process
        self._reader = SampleShardReader(self.path)
        self._reader.open()
        if self.use_mmap:
            self._reader.enable_mmap()
        self._sample_ids = self._reader.sample_ids()


class IterableSampleShardDataset:
    """
    PyTorch IterableDataset for streaming large SampleShard files.

    Use this when:
    - Dataset is too large to shuffle in memory
    - You want to stream through the data sequentially
    - Combined with DistributedSampler for multi-GPU training

    Example:
        >>> dataset = IterableSampleShardDataset("huge_train.smpl")
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[Callable[[Dict], Dict]] = None,
        return_tensors: bool = False,
    ):
        """
        Initialize the iterable dataset.

        Args:
            path: Path to .smpl file
            transform: Optional transform applied to each sample
            return_tensors: If True, convert arrays to torch tensors
        """
        self.path = Path(path)
        self.transform = transform
        self.return_tensors = return_tensors

    def __iter__(self):
        """Iterate over all samples."""
        try:
            import torch
            import numpy as np
        except ImportError:
            torch = None
            np = None

        # Get worker info for multi-worker loading
        worker_info = None
        try:
            import torch.utils.data as data

            worker_info = data.get_worker_info()
        except Exception:
            pass

        reader = SampleShardReader(self.path)
        reader.open()

        try:
            total = reader.sample_count()

            # Determine which samples this worker should handle
            if worker_info is not None:
                per_worker = total // worker_info.num_workers
                worker_id = worker_info.id
                start = worker_id * per_worker
                end = (
                    start + per_worker
                    if worker_id < worker_info.num_workers - 1
                    else total
                )
            else:
                start, end = 0, total

            for i in range(start, end):
                sample = reader.get_sample_by_index(i)

                if self.transform is not None:
                    sample = self.transform(sample)

                if self.return_tensors and torch is not None:
                    sample = self._to_tensors(sample, torch, np)

                yield sample

        finally:
            reader.close()

    def _to_tensors(self, sample: Dict, torch, np) -> Dict:
        """Convert arrays in sample to torch tensors."""
        result = {}
        for key, value in sample.items():
            if isinstance(value, list):
                try:
                    arr = np.array(value)
                    result[key] = torch.from_numpy(arr)
                except (ValueError, TypeError):
                    result[key] = value
            elif isinstance(value, (int, float)):
                result[key] = torch.tensor(value)
            else:
                result[key] = value
        return result


def collate_samples(batch: List[Dict]) -> Dict[str, Any]:
    """
    Default collate function for SampleShard batches.

    Stacks tensors along batch dimension and groups other values into lists.

    Args:
        batch: List of sample dicts

    Returns:
        Batched dict with stacked tensors
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required: pip install torch")

    if not batch:
        return {}

    result = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        # Try to stack as tensors
        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                # Different shapes, keep as list
                result[key] = values
        elif all(isinstance(v, (int, float)) for v in values):
            result[key] = torch.tensor(values)
        else:
            result[key] = values

    return result
