/**
 * SampleShard writer implementation.
 *
 * Writes training samples to .smpl files with the Shard v2 format.
 *
 * Shard v2 layout (matching Go implementation):
 *     Header (64 bytes)
 *     Index entries (entry_count × 48 bytes)
 *     String table (variable)
 *     [Padding to alignment]
 *     Data section (aligned entries)
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  ShardHeader,
  IndexEntry,
  ShardMetadata,
  SampleShardProfile,
  ManifestProfile,
  ShardRole,
  SHARD_HEADER_SIZE,
  SHARD_INDEX_ENTRY_SIZE,
  ALIGN_64,
  COMPRESS_NONE,
  SHARD_FLAG_HAS_SCHEMA,
  createShardHeader,
  serializeHeader,
  serializeIndexEntry,
  serializeMetadata,
  simpleHash64,
  crc32,
  initXxhash,
} from './types.js';
import { encode as cowrieEncode } from './cowrie.js';

interface PendingEntry {
  name: string;
  tempOffset: number;
  diskSize: number;
  origSize: number;
  checksum: number;
  flags: number;
}

function inferSchema(value: unknown): Record<string, unknown> {
  if (value === null) {
    return { type: 'null' };
  }
  if (typeof value === 'boolean') {
    return { type: 'bool' };
  }
  if (typeof value === 'number') {
    return { type: Number.isInteger(value) ? 'int' : 'float' };
  }
  if (typeof value === 'bigint') {
    return { type: 'int' };
  }
  if (typeof value === 'string') {
    return { type: 'string' };
  }
  if (Buffer.isBuffer(value) || value instanceof Uint8Array) {
    return { type: 'bytes' };
  }
  if (Array.isArray(value)) {
    const schema: Record<string, unknown> = { type: 'array' };
    const shape: number[] = [];
    let cursor: unknown = value;
    while (Array.isArray(cursor)) {
      shape.push(cursor.length);
      if (cursor.length === 0) {
        cursor = null;
        break;
      }
      cursor = cursor[0];
    }
    if (shape.length > 0) {
      schema['shape'] = shape;
    }
    schema['items'] = cursor === null ? { type: 'unknown' } : inferSchema(cursor);
    return schema;
  }
  if (typeof value === 'object') {
    const ctor = (value as { constructor?: { name?: string } }).constructor?.name;
    if (ctor && ctor !== 'Object') {
      return { type: ctor };
    }
    return {
      type: 'object',
      fields: Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([key, entryValue]) => [
          String(key),
          inferSchema(entryValue),
        ])
      ),
    };
  }
  return { type: typeof value };
}

/**
 * Writer for SampleShard (.smpl) files.
 *
 * SampleShard uses the Shard v2 format with Role=Sample (0x02).
 * Each sample is stored as a Cowrie-encoded blob with its ID as the entry name.
 *
 * @example
 * ```typescript
 * const writer = new SampleShardWriter("train.smpl");
 * await writer.open();
 * await writer.addSample(1, { input: [1, 2, 3], label: 0 });
 * await writer.addSample(2, { input: [4, 5, 6], label: 1 });
 * await writer.close();
 * ```
 */
export class SampleShardWriter {
  private filePath: string;
  private alignment: number;
  private compression: number;

  private fd: number | null = null;
  private tempFd: number | null = null;
  private tempPath: string | null = null;
  private tempOffset = 0;
  private entries: PendingEntry[] = [];
  private seenNames = new Set<string>();
  private closed = false;
  private metadata: ShardMetadata | null = null;
  private inferredSampleSchema: Record<string, unknown> | null = null;

  constructor(
    filePath: string,
    options: {
      alignment?: number;
      compression?: number;
    } = {}
  ) {
    this.filePath = filePath;
    this.alignment = options.alignment ?? ALIGN_64;
    this.compression = options.compression ?? COMPRESS_NONE;
  }

  /**
   * Open the file for writing.
   */
  async open(): Promise<void> {
    if (this.fd !== null) {
      return;
    }

    // Initialize xxhash for proper name hashing
    await initXxhash();

    this.fd = fs.openSync(this.filePath, 'w');

    // Create temp file for buffering data (same approach as Go)
    this.tempPath = path.join(os.tmpdir(), `shard_data_${Date.now()}_${Math.random().toString(36).slice(2)}`);
    this.tempFd = fs.openSync(this.tempPath, 'w+');
    this.tempOffset = 0;
  }

  setMetadata(metadata: ShardMetadata): void {
    this.metadata = {
      ...metadata,
      schemaVersion: metadata.schemaVersion ?? 'shard-v2.1',
    };
  }

  setSampleProfile(profile: SampleShardProfile): void {
    this.metadata = {
      ...(this.metadata ?? { schemaVersion: 'shard-v2.1' }),
      schemaVersion: this.metadata?.schemaVersion ?? 'shard-v2.1',
      profile: 'sampleshard.v1',
      sampleShard: {
        sampleIdType: 'uint64',
        keyEncoding: 'decimal-string',
        ...(this.metadata?.sampleShard ?? {}),
        ...profile,
      },
    };
  }

  setManifestProfile(profile: ManifestProfile): void {
    this.metadata = {
      ...(this.metadata ?? { schemaVersion: 'shard-v2.1' }),
      schemaVersion: this.metadata?.schemaVersion ?? 'shard-v2.1',
      profile: 'manifest.v1',
      manifest: {
        ...(this.metadata?.manifest ?? {}),
        ...profile,
      },
    };
  }

  /**
   * Add a sample to the shard.
   *
   * @param sampleId - Unique sample identifier (uint64 compatible)
   * @param sample - Sample data (must be JSON-serializable)
   */
  async addSample(sampleId: number | bigint, sample: unknown): Promise<void> {
    if (this.fd === null) {
      throw new Error('Writer not open');
    }
    if (this.closed) {
      throw new Error('Writer is closed');
    }

    const name = sampleId.toString();
    if (this.inferredSampleSchema === null) {
      this.inferredSampleSchema = inferSchema(sample);
    }
    // Encode sample as Cowrie binary format (byte-identical with Go)
    const data = cowrieEncode(sample);

    await this.writeEntry(name, data);
  }

  /**
   * Add pre-encoded sample bytes.
   *
   * @param sampleId - Unique sample identifier
   * @param data - Pre-encoded sample data (e.g., already Cowrie-encoded)
   */
  async addSampleRaw(sampleId: number | bigint, data: Buffer): Promise<void> {
    if (this.fd === null) {
      throw new Error('Writer not open');
    }
    if (this.closed) {
      throw new Error('Writer is closed');
    }

    const name = sampleId.toString();
    await this.writeEntry(name, data);
  }

  private async writeEntry(name: string, data: Buffer): Promise<void> {
    if (this.seenNames.has(name)) {
      throw new Error(`Duplicate sample ID: ${name}`);
    }
    this.seenNames.add(name);

    // Record temp file offset before writing
    const tempOffset = this.tempOffset;

    // Write to temp file
    fs.writeSync(this.tempFd!, data);
    this.tempOffset += data.length;

    // Store entry metadata (data stays in temp file)
    const entry: PendingEntry = {
      name,
      tempOffset,
      diskSize: data.length,
      origSize: data.length,
      checksum: crc32(data),
      flags: 0,
    };
    this.entries.push(entry);
  }

  /**
   * Finalize and close the shard file.
   *
   * Layout: Header + Index + StringTable + [Padding] + Data
   */
  async close(): Promise<void> {
    if (this.fd === null || this.closed) {
      return;
    }

    this.closed = true;

    try {
      await this.finalize();
    } finally {
      // Cleanup temp file
      if (this.tempFd !== null) {
        fs.closeSync(this.tempFd);
        this.tempFd = null;
      }
      if (this.tempPath !== null) {
        try {
          fs.unlinkSync(this.tempPath);
        } catch {
          // Ignore cleanup errors
        }
        this.tempPath = null;
      }
    }
  }

  private buildMetadata(entryCount: number): ShardMetadata | null {
    if (this.metadata === null) {
      return {
        schemaVersion: 'shard-v2.1',
        profile: 'sampleshard.v1',
        sampleShard: {
          datasetName: path.parse(this.filePath).name,
          sampleIdType: 'uint64',
          keyEncoding: 'decimal-string',
          sampleCount: entryCount,
          ...(this.inferredSampleSchema ? { datasetSchema: this.inferredSampleSchema } : {}),
        },
      };
    }

    const metadata: ShardMetadata = {
      ...this.metadata,
      schemaVersion: this.metadata.schemaVersion ?? 'shard-v2.1',
    };
    if (metadata.profile && metadata.profile !== 'sampleshard.v1') {
      return metadata;
    }

    metadata.profile = 'sampleshard.v1';
    metadata.sampleShard = {
      ...(metadata.sampleShard ?? {}),
      datasetName: metadata.sampleShard?.datasetName ?? path.parse(this.filePath).name,
      sampleIdType: metadata.sampleShard?.sampleIdType ?? 'uint64',
      keyEncoding: metadata.sampleShard?.keyEncoding ?? 'decimal-string',
      sampleCount: entryCount,
    };
    if (!metadata.sampleShard.datasetSchema && this.inferredSampleSchema) {
      metadata.sampleShard.datasetSchema = this.inferredSampleSchema;
    }
    return metadata;
  }

  private async finalize(): Promise<void> {
    const fd = this.fd!;
    const tempFd = this.tempFd!;
    const entryCount = this.entries.length;
    const indexSize = entryCount * SHARD_INDEX_ENTRY_SIZE;

    // Build string table
    const stringTableParts: Buffer[] = [];
    const nameOffsets = new Map<string, number>();
    let stringTableSize = 0;

    for (const entry of this.entries) {
      nameOffsets.set(entry.name, stringTableSize);
      const nameBytes = Buffer.from(entry.name, 'utf-8');
      stringTableParts.push(nameBytes);
      stringTableParts.push(Buffer.from([0])); // null terminator
      stringTableSize += nameBytes.length + 1;
    }
    const stringTable = Buffer.concat(stringTableParts);

    // Calculate offsets
    const stringTableOffset = SHARD_HEADER_SIZE + indexSize;
    let dataSectionOffset = stringTableOffset + stringTable.length;

    // Align data section
    if (this.alignment > 0) {
      dataSectionOffset = Math.ceil(dataSectionOffset / this.alignment) * this.alignment;
    }

    // Calculate entry data offsets in final file
    let currentDataOffset = dataSectionOffset;
    const dataOffsets: number[] = [];
    for (const entry of this.entries) {
      // Align each entry
      if (this.alignment > 0) {
        currentDataOffset = Math.ceil(currentDataOffset / this.alignment) * this.alignment;
      }
      dataOffsets.push(currentDataOffset);
      currentDataOffset += entry.diskSize;
    }

    let metadataBuf: Buffer | null = null;
    let schemaOffset = 0n;
    let totalSize = currentDataOffset;
    const metadata = this.buildMetadata(entryCount);
    if (metadata) {
      metadataBuf = serializeMetadata(metadata);
      schemaOffset = BigInt(currentDataOffset);
      totalSize += metadataBuf.length;
    }

    // Write header
    const header = createShardHeader(ShardRole.SAMPLE);
    if (metadataBuf) {
      header.flags |= SHARD_FLAG_HAS_SCHEMA;
      header.schemaOffset = schemaOffset;
    }
    header.alignment = this.alignment;
    header.compressionDefault = this.compression;
    header.entryCount = entryCount;
    header.stringTableOffset = BigInt(stringTableOffset);
    header.dataSectionOffset = BigInt(dataSectionOffset);
    header.totalFileSize = BigInt(totalSize);
    fs.writeSync(fd, serializeHeader(header));

    // Write index entries
    for (let i = 0; i < this.entries.length; i++) {
      const entry = this.entries[i];
      const nameBytes = Buffer.from(entry.name, 'utf-8');
      const indexEntry: IndexEntry = {
        nameHash: simpleHash64(nameBytes),
        nameOffset: nameOffsets.get(entry.name)!,
        nameLen: nameBytes.length,
        flags: entry.flags,
        dataOffset: BigInt(dataOffsets[i]),
        diskSize: BigInt(entry.diskSize),
        origSize: BigInt(entry.origSize),
        checksum: entry.checksum,
        contentType: 0,
        name: entry.name,
      };
      fs.writeSync(fd, serializeIndexEntry(indexEntry));
    }

    // Write string table
    fs.writeSync(fd, stringTable);

    // Write padding to data section
    const currentPos = stringTableOffset + stringTable.length;
    if (currentPos < dataSectionOffset) {
      const padding = Buffer.alloc(dataSectionOffset - currentPos);
      fs.writeSync(fd, padding);
    }

    // Write data entries from temp file with alignment padding
    let writePos = dataSectionOffset;
    for (let i = 0; i < this.entries.length; i++) {
      const entry = this.entries[i];
      const expectedOffset = dataOffsets[i];

      // Alignment padding
      if (writePos < expectedOffset) {
        const padding = Buffer.alloc(expectedOffset - writePos);
        fs.writeSync(fd, padding);
        writePos = expectedOffset;
      }

      // Read from temp and write to final file
      const entryData = Buffer.alloc(entry.diskSize);
      fs.readSync(tempFd, entryData, 0, entry.diskSize, entry.tempOffset);
      fs.writeSync(fd, entryData);
      writePos += entry.diskSize;
    }

    if (metadataBuf) {
      fs.writeSync(fd, metadataBuf);
    }

    fs.closeSync(fd);
    this.fd = null;
  }
}
