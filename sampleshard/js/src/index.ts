/**
 * SampleShard - Training sample storage format.
 *
 * SampleShard stores training samples keyed by uint64 sample ID. Each sample is
 * a JSON-serializable value that can contain tensors, metadata, and labels.
 *
 * @example
 * ```typescript
 * import { SampleShardWriter, SampleShardReader } from '@sampleshard/core';
 *
 * // Writing
 * const writer = new SampleShardWriter("train.smpl");
 * await writer.open();
 * await writer.addSample(1, { input: [1, 2, 3], label: 0 });
 * await writer.addSample(2, { input: [4, 5, 6], label: 1 });
 * await writer.close();
 *
 * // Reading
 * const reader = new SampleShardReader("train.smpl");
 * await reader.open();
 * console.log(reader.sampleCount()); // 2
 * const sample = await reader.getSample(1);
 * for await (const [id, sample] of reader) {
 *   console.log(id, sample);
 * }
 * await reader.close();
 * ```
 *
 * @packageDocumentation
 */

export { SampleShardReader } from './reader.js';
export { SampleShardWriter } from './writer.js';
export {
  ShardHeader,
  IndexEntry,
  EntryMeta,
  ShardMetadata,
  SampleShardProfile,
  ManifestFileRef,
  ManifestProfile,
  ShardRole,
  SHARD_MAGIC,
  SHARD_VERSION_2,
  SHARD_V2_HEADER_SIZE,
  SHARD_V2_INDEX_ENTRY_SIZE,
  ALIGN_NONE,
  ALIGN_16,
  ALIGN_32,
  ALIGN_64,
  COMPRESS_NONE,
  COMPRESS_ZSTD,
  COMPRESS_LZ4,
  SHARD_FLAG_HAS_SCHEMA,
  serializeMetadata,
  deserializeMetadata,
} from './types.js';
export { encode as cowrieEncode, decode as cowrieDecode, MAGIC as COWRIE_MAGIC } from './cowrie.js';
