# SampleShard TypeScript/JavaScript — Parked

> **This profile is parked.** It is not built or tested by default.
> See [`../README.md`](../README.md) for the parked-profile notice.

TypeScript/JavaScript implementation of the SampleShard format for storing training samples.

## Installation

```bash
npm install @sampleshard/core
```

## Quick Start

```typescript
import { SampleShardWriter, SampleShardReader } from '@sampleshard/core';

// Writing samples
const writer = new SampleShardWriter('train.smpl');
await writer.open();
await writer.addSample(1, { input: [1, 2, 3], label: 0 });
await writer.addSample(2, { input: [4, 5, 6], label: 1 });
await writer.addSample(3, { input: [7, 8, 9], label: 2 });
await writer.close();

// Reading samples
const reader = new SampleShardReader('train.smpl');
await reader.open();

// Get sample count
console.log(`Total samples: ${reader.sampleCount()}`);

// Random access by ID
const sample = await reader.getSample(1);
console.log(sample); // { input: [1, 2, 3], label: 0 }

// Check if sample exists
if (reader.hasSample(2)) {
  console.log('Sample 2 exists!');
}

// Iterate all samples
for await (const [sampleId, sample] of reader) {
  console.log(`Sample ${sampleId}:`, sample);
}

// Batch access
const batch = await reader.getBatch([1, 2, 3]);
const rangeBatch = await reader.getBatchByRange(0, 10);

await reader.close();
```

## Features

- **Fast random access** by sample ID (O(1) lookup)
- **Deterministic iteration** order (ascending by sample ID)
- **Duplicate rejection**: Writers reject duplicate sample IDs
- **Metadata-safe**: Reserved entries (starting with `__`) excluded from sample counts
- **Async/await API** with async iterators
- **CRC32C checksums** on decompressed data for integrity
- **BigInt support** for sample IDs
- **Structured metadata** with dataset profiles and per-entry metadata

## Dataset Metadata

```typescript
const writer = new SampleShardWriter('train.smpl');
await writer.open();

// Attach dataset profile
writer.setSampleProfile({
  datasetName: 'imagenet-train',
  sampleIdType: 'uint64',
});

await writer.addSample(1, { input: [1, 2, 3], label: 0 });
await writer.close();
// sample_count auto-populated on finalize

// Read metadata without decoding samples
const reader = new SampleShardReader('train.smpl');
await reader.open();
const profile = reader.sampleProfile();
console.log(profile?.datasetName);   // "imagenet-train"
console.log(profile?.sampleCount);   // 1
console.log(reader.readMetadata()?.schemaVersion); // "shard-v2.1"
await reader.close();
```

## File Format

SampleShard uses the `.smpl` extension and the Shard binary format:

- 64-byte header with magic bytes `SHRD`
- Role byte = 0x02 (Sample)
- 48-byte index entries with xxHash64 name hashes
- Cowrie-encoded sample data (auto-detected on read)
- CRC32C checksums on decompressed data per entry
- Optional JSON metadata at `schema_offset`

## Interoperability

Byte-identical across all implementations:
- **Go**: `go/shard` — open via `shard.OpenShard()` with role 0x02
- **Python**: `sampleshard.SampleShardReader()`
- **TypeScript**: `@sampleshard/core`

## API Reference

### SampleShardWriter

```typescript
class SampleShardWriter {
  constructor(path: string, options?: { alignment?: number; compression?: number });
  async open(): Promise<void>;
  setMetadata(metadata: ShardMetadata): void;
  setSampleProfile(profile: SampleShardProfile): void;
  setManifestProfile(profile: ManifestProfile): void;
  async addSample(sampleId: number | bigint, sample: unknown): Promise<void>;
  async addSampleRaw(sampleId: number | bigint, data: Buffer): Promise<void>;
  async close(): Promise<void>;
}
```

### SampleShardReader

```typescript
class SampleShardReader {
  constructor(path: string);
  async open(): Promise<void>;
  sampleCount(): number;
  getSampleIds(): bigint[];
  sampleIdByIndex(index: number): bigint;
  hasSample(sampleId: number | bigint): boolean;
  readMetadata(): ShardMetadata | null;
  sampleProfile(): SampleShardProfile | null;
  manifestProfile(): ManifestProfile | null;
  async getSample(sampleId: number | bigint): Promise<unknown>;
  async getSampleByIndex(index: number): Promise<unknown>;
  async getBatch(sampleIds: (number | bigint)[]): Promise<unknown[]>;
  async getBatchByRange(start: number, end: number): Promise<unknown[]>;
  async close(): Promise<void>;
  [Symbol.asyncIterator](): AsyncIterableIterator<[bigint, unknown]>;
}
```

## License

MIT
