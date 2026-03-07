#!/usr/bin/env python3
"""Generate a golden SampleShard file for cross-language testing.

Writes 5 diverse samples to cowrie/ucodec/testdata/golden_sampleshard.smpl
and prints each sample for reference. The Go test
cowrie/ucodec/sampleshard_golden_test.go reads this file and verifies
byte-level interop.

Sample IDs and content:
  10: int value (42)
  20: float value (3.14)
  30: string value ("hello cowrie")
  40: list of mixed types [1, "two", 3.0, None, True]
  50: nested dict {"name": "test", "scores": [95, 87], "meta": {"deep": True}}
"""

import os
import sys
from pathlib import Path

# Ensure the sampleshard package is importable from the repo checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sampleshard import SampleShardWriter


def main():
    # Output path: cowrie/ucodec/testdata/golden_sampleshard.smpl
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    out_path = repo_root / "cowrie" / "ucodec" / "testdata" / "golden_sampleshard.smpl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = {
        10: 42,
        20: 3.14,
        30: "hello cowrie",
        40: [1, "two", 3.0, None, True],
        50: {"name": "test", "scores": [95, 87], "meta": {"deep": True}},
    }

    with SampleShardWriter(str(out_path)) as w:
        for sample_id, sample in sorted(samples.items()):
            w.add_sample(sample_id, sample)
            print(f"  sample {sample_id}: {sample!r}")

    print(f"\nWrote {len(samples)} samples to {out_path}")
    print(f"File size: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
