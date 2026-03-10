/**
 * SampleShard reader implementation.
 *
 * Reads training samples from .smpl files with the Shard v2 format.
 *
 * Shard v2 layout (matching Go implementation):
 *     Header (64 bytes)
 *     Index entries (entry_count × 48 bytes)
 *     String table (variable)
 *     [Padding to alignment]
 *     Data section (aligned entries)
 */

import * as fs from 'fs';
import {
  ShardHeader,
  IndexEntry,
  ShardMetadata,
  SampleShardProfile,
  ManifestProfile,
  ShardRole,
  SHARD_V2_HEADER_SIZE,
  SHARD_V2_INDEX_ENTRY_SIZE,
  parseHeader,
  parseIndexEntry,
  isCompressed,
  getCompressionType,
  crc32,
  initXxhash,
  deserializeMetadata,
} from './types.js';
import { decode as cowrieDecode, MAGIC as COWRIE_MAGIC } from './cowrie.js';

/**
 * Reader for SampleShard (.smpl) files.
 *
 * SampleShard uses the Shard v2 format with Role=Sample (0x02).
 * Supports random access by sample ID and sequential iteration.
 *
 * @example
 * ```typescript
 * const reader = new SampleShardReader("train.smpl");
 * await reader.open();
 *
 * console.log(reader.sampleCount()); // Number of samples
 * const sample = await reader.getSample(1); // Get by ID
 *
 * for await (const [id, sample] of reader) {
 *   console.log(id, sample);
 * }
 *
 * await reader.close();
 * ```
 */
export class SampleShardReader {
  private path: string;
  private fd: number | null = null;
  private header: ShardHeader | null = null;
  private entries: IndexEntry[] = [];
  private entryByName: Map<string, IndexEntry> = new Map(); // name -> entry (O(1) lookup)
  private idMap: Map<bigint, number> = new Map(); // sample_id -> index in sampleIds
  private sampleIds: bigint[] = []; // ordered sample IDs
  private metadata: ShardMetadata | null | undefined = undefined;
  private closed = false;

  constructor(path: string) {
    this.path = path;
  }

  /**
   * Open the file for reading.
   *
   * Shard v2 layout (matching Go implementation):
   *     Header (64 bytes)
   *     Index entries (entry_count × 48 bytes)
   *     String table (variable)
   *     [Padding to alignment]
   *     Data section (aligned entries)
   */
  async open(): Promise<void> {
    if (this.fd !== null) {
      return;
    }

    this.fd = fs.openSync(this.path, 'r');

    // Read header
    const headerBuf = Buffer.alloc(SHARD_V2_HEADER_SIZE);
    fs.readSync(this.fd, headerBuf, 0, SHARD_V2_HEADER_SIZE, 0);
    this.header = parseHeader(headerBuf);

    // Verify role
    if (this.header.role !== ShardRole.SAMPLE) {
      this.close();
      throw new Error(
        `Expected Sample role (0x${ShardRole.SAMPLE.toString(16).padStart(2, '0')}), ` +
          `got 0x${this.header.role.toString(16).padStart(2, '0')}`
      );
    }

    // v2 format: Index is at offset 64 (right after header)
    const indexSize = this.header.entryCount * SHARD_V2_INDEX_ENTRY_SIZE;
    const indexData = Buffer.alloc(indexSize);
    fs.readSync(this.fd, indexData, 0, indexSize, SHARD_V2_HEADER_SIZE);

    for (let i = 0; i < this.header.entryCount; i++) {
      const offset = i * SHARD_V2_INDEX_ENTRY_SIZE;
      const entryBuf = indexData.subarray(offset, offset + SHARD_V2_INDEX_ENTRY_SIZE);
      const entry = parseIndexEntry(entryBuf);
      this.entries.push(entry);
    }

    // Read string table (between index and data section)
    const stringTableOffset = Number(this.header.stringTableOffset);
    const stringTableSize = Number(this.header.dataSectionOffset) - stringTableOffset;
    const stringTable = Buffer.alloc(stringTableSize);
    fs.readSync(this.fd, stringTable, 0, stringTableSize, stringTableOffset);

    // Extract names from string table
    for (const entry of this.entries) {
      if (entry.nameOffset + entry.nameLen <= stringTable.length) {
        entry.name = stringTable
          .subarray(entry.nameOffset, entry.nameOffset + entry.nameLen)
          .toString('utf-8');
      }
      this.entryByName.set(entry.name, entry); // O(1) lookup
    }

    // Build sample index (exclude metadata entries starting with __)
    for (const entry of this.entries) {
      if (entry.name.startsWith('__')) {
        continue; // Skip metadata entries
      }

      try {
        const sampleId = BigInt(entry.name);
        this.sampleIds.push(sampleId);
      } catch {
        // Not a numeric ID, skip
        continue;
      }
    }

    // Sort by ascending sample ID for deterministic iteration order
    // This ensures cross-language consistency (Go, Python, TypeScript)
    this.sampleIds.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

    // Build ID -> index map after sorting
    this.idMap.clear();
    for (let i = 0; i < this.sampleIds.length; i++) {
      this.idMap.set(this.sampleIds[i], i);
    }
  }

  /**
   * Return the number of samples (excludes metadata entries).
   */
  sampleCount(): number {
    return this.sampleIds.length;
  }

  /**
   * Return all sample IDs in deterministic order.
   */
  getSampleIds(): bigint[] {
    return [...this.sampleIds];
  }

  /**
   * Return the sample ID at the given index.
   */
  sampleIdByIndex(index: number): bigint {
    if (index < 0 || index >= this.sampleIds.length) {
      throw new RangeError(`Sample index out of range: ${index}`);
    }
    return this.sampleIds[index];
  }

  readMetadata(): ShardMetadata | null {
    if (this.metadata !== undefined) {
      return this.metadata;
    }
    if (!this.header || this.fd === null) {
      throw new Error('Reader not open');
    }

    const schemaOffset = Number(this.header.schemaOffset);
    if (schemaOffset === 0) {
      this.metadata = null;
      return null;
    }
    const totalFileSize = Number(this.header.totalFileSize);
    if (schemaOffset > totalFileSize) {
      throw new Error(`schema_offset ${schemaOffset} exceeds total_file_size ${totalFileSize}`);
    }

    const metaData = Buffer.alloc(totalFileSize - schemaOffset);
    fs.readSync(this.fd, metaData, 0, metaData.length, schemaOffset);
    this.metadata = deserializeMetadata(metaData);
    return this.metadata;
  }

  sampleProfile(): SampleShardProfile | null {
    return this.readMetadata()?.sampleShard ?? null;
  }

  manifestProfile(): ManifestProfile | null {
    return this.readMetadata()?.manifest ?? null;
  }

  /**
   * Check if a sample exists by ID.
   */
  hasSample(sampleId: number | bigint): boolean {
    return this.idMap.has(BigInt(sampleId));
  }

  /**
   * Get a sample by ID.
   *
   * @param sampleId - Sample identifier
   * @returns Decoded sample data
   * @throws Error if sample not found
   */
  async getSample(sampleId: number | bigint): Promise<unknown> {
    const id = BigInt(sampleId);
    const index = this.idMap.get(id);

    if (index === undefined) {
      throw new Error(`Sample not found: ${sampleId}`);
    }

    return this.getSampleByIndex(index);
  }

  /**
   * Get a sample by its position in the sample index.
   *
   * @param index - Position in the filtered sample list (0-based)
   * @returns Decoded sample data
   */
  async getSampleByIndex(index: number): Promise<unknown> {
    if (index < 0 || index >= this.sampleIds.length) {
      throw new RangeError(`Sample index out of range: ${index}`);
    }

    const sampleId = this.sampleIds[index];
    const name = sampleId.toString();

    // O(1) lookup
    const entry = this.entryByName.get(name);
    if (!entry) {
      throw new Error(`Entry not found: ${name}`);
    }

    return this.readEntry(entry);
  }

  private async readEntry(entry: IndexEntry): Promise<unknown> {
    const fd = this.fd!;

    const data = Buffer.alloc(Number(entry.diskSize));
    fs.readSync(fd, data, 0, Number(entry.diskSize), Number(entry.dataOffset));

    // Decompress if needed (before CRC check — Go checks CRC on decompressed data)
    let decompressed = data;
    if (isCompressed(entry)) {
      const compType = getCompressionType(entry);
      if (compType === 1) {
        // zstd - would need zstd library
        throw new Error('zstd decompression not implemented');
      } else if (compType === 2) {
        // lz4 - would need lz4 library
        throw new Error('lz4 decompression not implemented');
      }
    }

    // Verify checksum on decompressed data (matches Go reference)
    const actualCrc = crc32(decompressed);
    if (actualCrc !== entry.checksum) {
      throw new Error(
        `Checksum mismatch for entry '${entry.name}': ` +
          `expected 0x${entry.checksum.toString(16).padStart(8, '0')}, ` +
          `got 0x${actualCrc.toString(16).padStart(8, '0')}`
      );
    }

    // Decode sample (auto-detect Cowrie vs JSON)
    if (decompressed.length >= 2 && decompressed.subarray(0, 2).equals(COWRIE_MAGIC)) {
      return cowrieDecode(decompressed);
    }
    // Fallback to JSON for legacy files
    return JSON.parse(decompressed.toString('utf-8'));
  }

  /**
   * Get multiple samples by their IDs.
   *
   * @param sampleIds - List of sample identifiers
   * @returns List of decoded samples in the same order
   */
  async getBatch(sampleIds: (number | bigint)[]): Promise<unknown[]> {
    return Promise.all(sampleIds.map((id) => this.getSample(id)));
  }

  /**
   * Get a range of samples by index [start, end).
   *
   * @param start - Start index (inclusive)
   * @param end - End index (exclusive)
   * @returns List of decoded samples
   */
  async getBatchByRange(start: number, end: number): Promise<unknown[]> {
    start = Math.max(0, start);
    end = Math.min(this.sampleIds.length, end);

    const samples: unknown[] = [];
    for (let i = start; i < end; i++) {
      samples.push(await this.getSampleByIndex(i));
    }
    return samples;
  }

  /**
   * Async iterator for all samples.
   */
  async *[Symbol.asyncIterator](): AsyncIterableIterator<[bigint, unknown]> {
    for (let i = 0; i < this.sampleIds.length; i++) {
      const sampleId = this.sampleIds[i];
      const sample = await this.getSampleByIndex(i);
      yield [sampleId, sample];
    }
  }

  /**
   * Close the reader.
   */
  async close(): Promise<void> {
    if (this.closed || this.fd === null) {
      return;
    }

    this.closed = true;
    fs.closeSync(this.fd);
    this.fd = null;
  }
}
