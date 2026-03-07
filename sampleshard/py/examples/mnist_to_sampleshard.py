#!/usr/bin/env python3
"""
Example: Convert MNIST to SampleShard and train a simple classifier.

This example demonstrates:
1. Converting a HuggingFace dataset (MNIST) to SampleShard format
2. Loading SampleShard with PyTorch DataLoader
3. Training a simple CNN classifier

Usage:
    python mnist_to_sampleshard.py

Requirements:
    pip install datasets torch torchvision tqdm
"""

import time
from pathlib import Path

# Check dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch required: pip install torch torchvision")
    exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("HuggingFace datasets required: pip install datasets")
    exit(1)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sampleshard import SampleShardWriter, SampleShardReader
from sampleshard.bridges import from_hf_dataset
from sampleshard.torch import SampleShardDataset


# ============================================================
# Step 1: Convert MNIST to SampleShard
# ============================================================


def convert_mnist():
    """Convert MNIST train/test splits to SampleShard."""
    print("=" * 60)
    print("Step 1: Converting MNIST to SampleShard")
    print("=" * 60)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    train_path = output_dir / "mnist_train.smpl"
    test_path = output_dir / "mnist_test.smpl"

    if train_path.exists() and test_path.exists():
        print(f"SampleShard files already exist, skipping conversion")
        return train_path, test_path

    # Load MNIST from HuggingFace
    print("Loading MNIST from HuggingFace...")
    ds_train = load_dataset("mnist", split="train")
    ds_test = load_dataset("mnist", split="test")

    # Convert to SampleShard
    print(f"Converting {len(ds_train)} training samples...")
    start = time.time()

    with SampleShardWriter(train_path) as writer:
        for i, item in enumerate(ds_train):
            # Convert PIL Image to list
            import numpy as np

            img = np.array(item["image"])  # 28x28 grayscale

            sample = {
                "input": img.tolist(),
                "target": item["label"],
                "meta": {"shape": list(img.shape)},
            }
            writer.add_sample(i, sample)

            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(ds_train)} samples...")

    train_time = time.time() - start
    print(f"  Train set: {len(ds_train)} samples in {train_time:.2f}s")

    print(f"Converting {len(ds_test)} test samples...")
    start = time.time()

    with SampleShardWriter(test_path) as writer:
        for i, item in enumerate(ds_test):
            import numpy as np

            img = np.array(item["image"])

            sample = {
                "input": img.tolist(),
                "target": item["label"],
                "meta": {"shape": list(img.shape)},
            }
            writer.add_sample(i, sample)

    test_time = time.time() - start
    print(f"  Test set: {len(ds_test)} samples in {test_time:.2f}s")

    # Report file sizes
    train_size = train_path.stat().st_size / (1024 * 1024)
    test_size = test_path.stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  Train: {train_size:.2f} MB")
    print(f"  Test: {test_size:.2f} MB")

    return train_path, test_path


# ============================================================
# Step 2: Define a simple CNN model
# ============================================================


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x: (B, 28, 28) -> (B, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 64, 7, 7)
        x = x.view(x.size(0), -1)  # -> (B, 64*7*7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================================
# Step 3: Training loop
# ============================================================


def train_model(train_path, test_path, epochs=3, batch_size=64):
    """Train the CNN on SampleShard data."""
    print("\n" + "=" * 60)
    print("Step 2: Training with SampleShard + PyTorch DataLoader")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    print("Loading SampleShard datasets...")

    def transform(sample):
        """Convert sample to tensors."""
        import numpy as np

        img = np.array(sample["input"], dtype=np.float32) / 255.0
        return {
            "input": torch.from_numpy(img),
            "target": torch.tensor(sample["target"], dtype=torch.long),
        }

    train_dataset = SampleShardDataset(train_path, transform=transform)
    test_dataset = SampleShardDataset(test_path, transform=transform)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simplicity; use 4+ for real training
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start = time.time()

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        epoch_time = time.time() - start
        train_acc = 100.0 * train_correct / train_total

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss / train_total:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    print(f"\nFinal test accuracy: {test_acc:.2f}%")

    # Cleanup
    train_dataset.close()
    test_dataset.close()

    return model


# ============================================================
# Step 4: Inspect the SampleShard file
# ============================================================


def inspect_sampleshard(path):
    """Show SampleShard file info."""
    print("\n" + "=" * 60)
    print("Step 3: Inspecting SampleShard file")
    print("=" * 60)

    with SampleShardReader(path) as reader:
        print(f"File: {path}")
        print(f"Samples: {reader.sample_count()}")

        ids = reader.sample_ids()
        print(f"ID range: {ids[0]} - {ids[-1]}")

        # Show first sample
        sample = reader.get_sample(0)
        print(f"\nFirst sample structure:")
        for key, value in sample.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: list[{len(value)}] (first 5: {value[:5]}...)")
            else:
                print(f"  {key}: {value}")


# ============================================================
# Main
# ============================================================


def main():
    print("SampleShard MNIST Example")
    print("=" * 60)
    print()

    # Step 1: Convert MNIST
    train_path, test_path = convert_mnist()

    # Step 2: Train model
    model = train_model(train_path, test_path, epochs=3)

    # Step 3: Inspect
    inspect_sampleshard(train_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
