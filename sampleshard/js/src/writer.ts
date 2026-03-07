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
  ShardRole,
  SHARD_V2_HEADER_SIZE,
  SHARD_V2_INDEX_ENTRY_SIZE,
  ALIGN_64,
  COMPRESS_NONE,
  createShardHeader,
  serializeHeader,
  serializeIndexEntry,
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
  private closed = false;

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

  private async finalize(): Promise<void> {
    const fd = this.fd!;
    const tempFd = this.tempFd!;
    const entryCount = this.entries.length;
    const indexSize = entryCount * SHARD_V2_INDEX_ENTRY_SIZE;

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
    const stringTableOffset = SHARD_V2_HEADER_SIZE + indexSize;
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

    const totalSize = currentDataOffset;

    // Write header
    const header = createShardHeader(ShardRole.SAMPLE);
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

    fs.closeSync(fd);
    this.fd = null;
  }
}
