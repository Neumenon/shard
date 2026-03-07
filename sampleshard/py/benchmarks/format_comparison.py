#!/usr/bin/env python3
"""
Benchmark: SampleShard vs Parquet vs SQLite vs JSONL

Compares:
1. Random access latency (get sample by ID)
2. Sequential iteration throughput
3. Memory usage during indexing
4. File size
5. Write throughput

Usage:
    python format_comparison.py [--samples N] [--output results.json]

Requirements:
    pip install pandas pyarrow numpy tqdm psutil
"""

import argparse
import gc
import json
import os
import random
import shutil
import sqlite3
import statistics
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed, memory benchmarks will be skipped")

try:
    import pandas as pd
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Warning: pandas/pyarrow not installed, Parquet benchmarks will be skipped")

# Add parent to path for sampleshard import
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sampleshard import SampleShardWriter, SampleShardReader


@dataclass
class BenchmarkResult:
    """Results for a single benchmark."""

    format_name: str
    operation: str
    samples: int
    total_time_s: float
    ops_per_sec: float
    latency_mean_us: float
    latency_p50_us: float
    latency_p99_us: float
    memory_mb: Optional[float] = None
    file_size_mb: Optional[float] = None


def generate_sample(sample_id: int, feature_dim: int = 128) -> Dict[str, Any]:
    """Generate a synthetic training sample."""
    return {
        "input": np.random.randn(feature_dim).tolist(),
        "target": sample_id % 10,
        "meta": {
            "id": sample_id,
            "source": "synthetic",
            "timestamp": f"2024-01-{(sample_id % 28) + 1:02d}",
        },
    }


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


class SampleShardBenchmark:
    """Benchmark harness for SampleShard."""

    def __init__(self, path: Path, num_samples: int, feature_dim: int = 128):
        self.path = path
        self.num_samples = num_samples
        self.feature_dim = feature_dim

    def write(self) -> BenchmarkResult:
        """Benchmark write throughput."""
        gc.collect()
        start_mem = get_memory_mb()

        start = time.perf_counter()
        with SampleShardWriter(self.path) as writer:
            for i in range(self.num_samples):
                sample = generate_sample(i, self.feature_dim)
                writer.add_sample(i, sample)
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()
        file_size = self.path.stat().st_size / (1024 * 1024)

        return BenchmarkResult(
            format_name="SampleShard",
            operation="write",
            samples=self.num_samples,
            total_time_s=elapsed,
            ops_per_sec=self.num_samples / elapsed,
            latency_mean_us=(elapsed / self.num_samples) * 1e6,
            latency_p50_us=0,  # Not tracked per-op
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
            file_size_mb=file_size,
        )

    def random_access(self, num_reads: int = 1000) -> BenchmarkResult:
        """Benchmark random access latency."""
        ids = random.sample(range(self.num_samples), min(num_reads, self.num_samples))
        latencies = []

        gc.collect()
        start_mem = get_memory_mb()

        with SampleShardReader(self.path) as reader:
            open_mem = get_memory_mb()

            for sample_id in ids:
                start = time.perf_counter()
                _ = reader.get_sample(sample_id)
                latencies.append((time.perf_counter() - start) * 1e6)

        return BenchmarkResult(
            format_name="SampleShard",
            operation="random_access",
            samples=len(ids),
            total_time_s=sum(latencies) / 1e6,
            ops_per_sec=len(ids) / (sum(latencies) / 1e6),
            latency_mean_us=statistics.mean(latencies),
            latency_p50_us=statistics.median(latencies),
            latency_p99_us=sorted(latencies)[int(len(latencies) * 0.99)],
            memory_mb=open_mem - start_mem,
        )

    def sequential_read(self) -> BenchmarkResult:
        """Benchmark sequential iteration."""
        gc.collect()
        start_mem = get_memory_mb()

        count = 0
        start = time.perf_counter()
        with SampleShardReader(self.path) as reader:
            for sample_id, sample in reader:
                count += 1
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()

        return BenchmarkResult(
            format_name="SampleShard",
            operation="sequential_read",
            samples=count,
            total_time_s=elapsed,
            ops_per_sec=count / elapsed,
            latency_mean_us=(elapsed / count) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
        )


class ParquetBenchmark:
    """Benchmark harness for Parquet."""

    def __init__(self, path: Path, num_samples: int, feature_dim: int = 128):
        self.path = path
        self.num_samples = num_samples
        self.feature_dim = feature_dim

    def write(self) -> BenchmarkResult:
        """Benchmark write throughput."""
        if not HAS_PARQUET:
            return None

        gc.collect()
        start_mem = get_memory_mb()

        # Generate all data (Parquet writes columnar, so we need all data upfront)
        data = {
            "id": list(range(self.num_samples)),
            "target": [i % 10 for i in range(self.num_samples)],
        }
        # Store features as separate columns (Parquet-style)
        for j in range(self.feature_dim):
            data[f"f{j}"] = np.random.randn(self.num_samples).tolist()

        start = time.perf_counter()
        df = pd.DataFrame(data)
        df.to_parquet(self.path, index=False)
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()
        file_size = self.path.stat().st_size / (1024 * 1024)

        return BenchmarkResult(
            format_name="Parquet",
            operation="write",
            samples=self.num_samples,
            total_time_s=elapsed,
            ops_per_sec=self.num_samples / elapsed,
            latency_mean_us=(elapsed / self.num_samples) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
            file_size_mb=file_size,
        )

    def random_access(self, num_reads: int = 1000) -> BenchmarkResult:
        """Benchmark random access latency."""
        if not HAS_PARQUET:
            return None

        ids = random.sample(range(self.num_samples), min(num_reads, self.num_samples))
        latencies = []

        gc.collect()
        start_mem = get_memory_mb()

        # Load entire dataframe (Parquet's approach)
        df = pd.read_parquet(self.path)
        df = df.set_index("id")
        open_mem = get_memory_mb()

        for sample_id in ids:
            start = time.perf_counter()
            _ = df.loc[sample_id].to_dict()
            latencies.append((time.perf_counter() - start) * 1e6)

        return BenchmarkResult(
            format_name="Parquet",
            operation="random_access",
            samples=len(ids),
            total_time_s=sum(latencies) / 1e6,
            ops_per_sec=len(ids) / (sum(latencies) / 1e6),
            latency_mean_us=statistics.mean(latencies),
            latency_p50_us=statistics.median(latencies),
            latency_p99_us=sorted(latencies)[int(len(latencies) * 0.99)],
            memory_mb=open_mem - start_mem,
        )

    def sequential_read(self) -> BenchmarkResult:
        """Benchmark sequential iteration."""
        if not HAS_PARQUET:
            return None

        gc.collect()
        start_mem = get_memory_mb()

        count = 0
        start = time.perf_counter()
        df = pd.read_parquet(self.path)
        for _, row in df.iterrows():
            _ = row.to_dict()
            count += 1
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()

        return BenchmarkResult(
            format_name="Parquet",
            operation="sequential_read",
            samples=count,
            total_time_s=elapsed,
            ops_per_sec=count / elapsed,
            latency_mean_us=(elapsed / count) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
        )


class SQLiteBenchmark:
    """Benchmark harness for SQLite."""

    def __init__(self, path: Path, num_samples: int, feature_dim: int = 128):
        self.path = path
        self.num_samples = num_samples
        self.feature_dim = feature_dim

    def write(self) -> BenchmarkResult:
        """Benchmark write throughput."""
        gc.collect()
        start_mem = get_memory_mb()

        start = time.perf_counter()
        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)

        for i in range(self.num_samples):
            sample = generate_sample(i, self.feature_dim)
            cursor.execute(
                "INSERT INTO samples (id, data) VALUES (?, ?)", (i, json.dumps(sample))
            )

        conn.commit()
        conn.close()
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()
        file_size = self.path.stat().st_size / (1024 * 1024)

        return BenchmarkResult(
            format_name="SQLite",
            operation="write",
            samples=self.num_samples,
            total_time_s=elapsed,
            ops_per_sec=self.num_samples / elapsed,
            latency_mean_us=(elapsed / self.num_samples) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
            file_size_mb=file_size,
        )

    def random_access(self, num_reads: int = 1000) -> BenchmarkResult:
        """Benchmark random access latency."""
        ids = random.sample(range(self.num_samples), min(num_reads, self.num_samples))
        latencies = []

        gc.collect()
        start_mem = get_memory_mb()

        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        open_mem = get_memory_mb()

        for sample_id in ids:
            start = time.perf_counter()
            cursor.execute("SELECT data FROM samples WHERE id = ?", (sample_id,))
            row = cursor.fetchone()
            _ = json.loads(row[0])
            latencies.append((time.perf_counter() - start) * 1e6)

        conn.close()

        return BenchmarkResult(
            format_name="SQLite",
            operation="random_access",
            samples=len(ids),
            total_time_s=sum(latencies) / 1e6,
            ops_per_sec=len(ids) / (sum(latencies) / 1e6),
            latency_mean_us=statistics.mean(latencies),
            latency_p50_us=statistics.median(latencies),
            latency_p99_us=sorted(latencies)[int(len(latencies) * 0.99)],
            memory_mb=open_mem - start_mem,
        )

    def sequential_read(self) -> BenchmarkResult:
        """Benchmark sequential iteration."""
        gc.collect()
        start_mem = get_memory_mb()

        count = 0
        start = time.perf_counter()
        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, data FROM samples ORDER BY id")
        for row in cursor:
            _ = json.loads(row[1])
            count += 1
        conn.close()
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()

        return BenchmarkResult(
            format_name="SQLite",
            operation="sequential_read",
            samples=count,
            total_time_s=elapsed,
            ops_per_sec=count / elapsed,
            latency_mean_us=(elapsed / count) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
        )


class JSONLBenchmark:
    """Benchmark harness for JSON Lines."""

    def __init__(self, path: Path, num_samples: int, feature_dim: int = 128):
        self.path = path
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self._index: Dict[int, int] = {}  # id -> file offset

    def write(self) -> BenchmarkResult:
        """Benchmark write throughput."""
        gc.collect()
        start_mem = get_memory_mb()

        start = time.perf_counter()
        with open(self.path, "w") as f:
            for i in range(self.num_samples):
                sample = generate_sample(i, self.feature_dim)
                sample["_id"] = i
                f.write(json.dumps(sample) + "\n")
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()
        file_size = self.path.stat().st_size / (1024 * 1024)

        return BenchmarkResult(
            format_name="JSONL",
            operation="write",
            samples=self.num_samples,
            total_time_s=elapsed,
            ops_per_sec=self.num_samples / elapsed,
            latency_mean_us=(elapsed / self.num_samples) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
            file_size_mb=file_size,
        )

    def _build_index(self):
        """Build offset index for random access."""
        self._index = {}
        with open(self.path, "r") as f:
            offset = 0
            for line in f:
                obj = json.loads(line)
                self._index[obj["_id"]] = offset
                offset += len(line.encode("utf-8"))

    def random_access(self, num_reads: int = 1000) -> BenchmarkResult:
        """Benchmark random access latency (with index)."""
        gc.collect()
        start_mem = get_memory_mb()

        # Build index first
        self._build_index()
        open_mem = get_memory_mb()

        ids = random.sample(range(self.num_samples), min(num_reads, self.num_samples))
        latencies = []

        with open(self.path, "r") as f:
            for sample_id in ids:
                start = time.perf_counter()
                f.seek(self._index[sample_id])
                line = f.readline()
                _ = json.loads(line)
                latencies.append((time.perf_counter() - start) * 1e6)

        return BenchmarkResult(
            format_name="JSONL",
            operation="random_access",
            samples=len(ids),
            total_time_s=sum(latencies) / 1e6,
            ops_per_sec=len(ids) / (sum(latencies) / 1e6),
            latency_mean_us=statistics.mean(latencies),
            latency_p50_us=statistics.median(latencies),
            latency_p99_us=sorted(latencies)[int(len(latencies) * 0.99)],
            memory_mb=open_mem - start_mem,
        )

    def sequential_read(self) -> BenchmarkResult:
        """Benchmark sequential iteration."""
        gc.collect()
        start_mem = get_memory_mb()

        count = 0
        start = time.perf_counter()
        with open(self.path, "r") as f:
            for line in f:
                _ = json.loads(line)
                count += 1
        elapsed = time.perf_counter() - start

        end_mem = get_memory_mb()

        return BenchmarkResult(
            format_name="JSONL",
            operation="sequential_read",
            samples=count,
            total_time_s=elapsed,
            ops_per_sec=count / elapsed,
            latency_mean_us=(elapsed / count) * 1e6,
            latency_p50_us=0,
            latency_p99_us=0,
            memory_mb=end_mem - start_mem,
        )


def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""

    # Group by operation
    operations = {}
    for r in results:
        if r is None:
            continue
        if r.operation not in operations:
            operations[r.operation] = []
        operations[r.operation].append(r)

    for op, op_results in operations.items():
        print(f"\n{'=' * 70}")
        print(f" {op.upper().replace('_', ' ')}")
        print(f"{'=' * 70}")

        if op == "write":
            print(
                f"{'Format':<15} {'Samples':<10} {'Time (s)':<10} {'Ops/sec':<12} {'Size (MB)':<10}"
            )
            print("-" * 70)
            for r in sorted(op_results, key=lambda x: x.total_time_s):
                print(
                    f"{r.format_name:<15} {r.samples:<10} {r.total_time_s:<10.2f} {r.ops_per_sec:<12,.0f} {r.file_size_mb:<10.2f}"
                )

        elif op == "random_access":
            print(
                f"{'Format':<15} {'Reads':<8} {'Mean (us)':<12} {'P50 (us)':<12} {'P99 (us)':<12} {'Ops/sec':<12}"
            )
            print("-" * 70)
            for r in sorted(op_results, key=lambda x: x.latency_mean_us):
                print(
                    f"{r.format_name:<15} {r.samples:<8} {r.latency_mean_us:<12.1f} {r.latency_p50_us:<12.1f} {r.latency_p99_us:<12.1f} {r.ops_per_sec:<12,.0f}"
                )

        elif op == "sequential_read":
            print(
                f"{'Format':<15} {'Samples':<10} {'Time (s)':<10} {'Ops/sec':<12} {'Mem (MB)':<10}"
            )
            print("-" * 70)
            for r in sorted(op_results, key=lambda x: x.total_time_s):
                mem = f"{r.memory_mb:.1f}" if r.memory_mb else "N/A"
                print(
                    f"{r.format_name:<15} {r.samples:<10} {r.total_time_s:<10.2f} {r.ops_per_sec:<12,.0f} {mem:<10}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SampleShard vs other formats"
    )
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument(
        "--feature-dim", type=int, default=128, help="Feature dimension"
    )
    parser.add_argument(
        "--reads", type=int, default=1000, help="Number of random reads"
    )
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    print("=" * 70)
    print(" FORMAT COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Samples: {args.samples:,}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Random reads: {args.reads:,}")
    print()

    # Create temp directory
    tmpdir = Path(tempfile.mkdtemp(prefix="sampleshard_bench_"))
    print(f"Temp directory: {tmpdir}")

    results = []

    try:
        # SampleShard
        print("\n[1/4] Benchmarking SampleShard...")
        ss_bench = SampleShardBenchmark(
            tmpdir / "data.smpl", args.samples, args.feature_dim
        )
        results.append(ss_bench.write())
        results.append(ss_bench.random_access(args.reads))
        results.append(ss_bench.sequential_read())

        # Parquet
        if HAS_PARQUET:
            print("[2/4] Benchmarking Parquet...")
            pq_bench = ParquetBenchmark(
                tmpdir / "data.parquet", args.samples, args.feature_dim
            )
            results.append(pq_bench.write())
            results.append(pq_bench.random_access(args.reads))
            results.append(pq_bench.sequential_read())
        else:
            print("[2/4] Skipping Parquet (not installed)")

        # SQLite
        print("[3/4] Benchmarking SQLite...")
        sql_bench = SQLiteBenchmark(tmpdir / "data.db", args.samples, args.feature_dim)
        results.append(sql_bench.write())
        results.append(sql_bench.random_access(args.reads))
        results.append(sql_bench.sequential_read())

        # JSONL
        print("[4/4] Benchmarking JSONL...")
        jsonl_bench = JSONLBenchmark(
            tmpdir / "data.jsonl", args.samples, args.feature_dim
        )
        results.append(jsonl_bench.write())
        results.append(jsonl_bench.random_access(args.reads))
        results.append(jsonl_bench.sequential_read())

        # Print results
        print_results_table(results)

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump([asdict(r) for r in results if r], f, indent=2)
            print(f"\nResults saved to {args.output}")

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)
        print(f"\nCleaned up {tmpdir}")


if __name__ == "__main__":
    main()
