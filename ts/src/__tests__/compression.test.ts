/**
 * Compression roundtrip tests — zstd and lz4 block format.
 *
 * Verifies that:
 *   - Compressed entries are smaller on disk than the original data.
 *   - Decompressed data matches the original exactly.
 *   - CRC32C checksums are computed on the UNCOMPRESSED data.
 *   - Small data below MIN_COMPRESS_SIZE stays uncompressed.
 *   - Mixed compressed/uncompressed entries coexist correctly.
 */

import { describe, it, expect, beforeAll } from 'vitest';

import {
  ShardV2Reader,
  ShardV2Writer,
  computeCrc32c,
  initXxhash,
  initCompression,
  COMPRESS_ZSTD,
  COMPRESS_LZ4,
  COMPRESS_NONE,
} from '../index.js';

beforeAll(async () => {
  await initXxhash();
  await initCompression();
});

// Helper: build a highly compressible buffer (repeating pattern)
function makeRepetitiveData(size: number): Buffer {
  const data = Buffer.alloc(size);
  for (let i = 0; i < size; i++) data[i] = i % 256;
  return data;
}

// ============================================================
// Zstd
// ============================================================

describe('Zstd Compression', () => {
  it('roundtrip — compressible data', () => {
    const data = makeRepetitiveData(10_000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('big', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), data)).toBe(0);
  });

  it('disk_size < orig_size for compressible data', () => {
    const data = makeRepetitiveData(10_000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('big', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    expect(info.compressed).toBe(true);
    expect(Number(info.diskSize)).toBeLessThan(Number(info.origSize));
    expect(Number(info.origSize)).toBe(data.length);
  });

  it('small data (< 256 bytes) stays uncompressed', () => {
    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('tiny', Buffer.from('hello'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntry(0).toString()).toBe('hello');
    expect(r.getEntryInfo(0).compressed).toBe(false);
  });

  it('checksum is computed on uncompressed data', () => {
    const data = makeRepetitiveData(2560);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('data', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    // The stored checksum must match the CRC32C of the original uncompressed data.
    expect(r.getEntryInfo(0).checksum).toBe(computeCrc32c(data));
  });

  it('entry flags: compressed=true, zstd bit set', () => {
    const data = makeRepetitiveData(5000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('x', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    // ENTRY_FLAG_COMPRESSED = 0x0001, ENTRY_FLAG_ZSTD = 0x0002
    expect(info.flags & 0x0001).toBe(0x0001); // compressed bit
    expect(info.flags & 0x0002).toBe(0x0002); // zstd bit
    expect(info.flags & 0x0004).toBe(0x0000); // lz4 bit must be clear
  });

  it('writeEntryWithOptions explicit zstd', () => {
    const data = makeRepetitiveData(8000);

    const w = new ShardV2Writer();
    w.writeEntryWithOptions('x', data, true, COMPRESS_ZSTD);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), data)).toBe(0);
    expect(r.getEntryInfo(0).compressed).toBe(true);
  });

  it('exactly 256 bytes at boundary: may or may not compress', () => {
    // Data exactly at MIN_COMPRESS_SIZE=256 is eligible for compression.
    // Compressibility depends on content; at least verify roundtrip is correct.
    const data = makeRepetitiveData(256);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('boundary', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), data)).toBe(0);
  });

  it('rejects zstd size mismatches against the shard index', () => {
    const data = makeRepetitiveData(4096);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('big', data);

    const buf = Buffer.from(w.toBuffer());
    const indexOffset = 64;
    buf.writeUInt32LE(1, indexOffset + 32);
    buf.writeUInt32LE(0, indexOffset + 36);

    const r = new ShardV2Reader(buf);
    expect(r.getEntryInfo(0).compressed).toBe(true);
    expect(() => r.readEntry(0)).toThrow(/does not match expected size 1/);
  });
});

// ============================================================
// LZ4
// ============================================================

describe('LZ4 Compression', () => {
  it('roundtrip — compressible data', () => {
    const data = makeRepetitiveData(10_000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('big', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), data)).toBe(0);
  });

  it('disk_size < orig_size for compressible data', () => {
    const data = makeRepetitiveData(10_000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('big', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    expect(info.compressed).toBe(true);
    expect(Number(info.diskSize)).toBeLessThan(Number(info.origSize));
  });

  it('small data (< 256 bytes) stays uncompressed', () => {
    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('tiny', Buffer.from('hello'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntry(0).toString()).toBe('hello');
    expect(r.getEntryInfo(0).compressed).toBe(false);
  });

  it('checksum is computed on uncompressed data', () => {
    const data = makeRepetitiveData(2560);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('data', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.getEntryInfo(0).checksum).toBe(computeCrc32c(data));
  });

  it('entry flags: compressed=true, lz4 bit set', () => {
    const data = makeRepetitiveData(5000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('x', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    expect(info.flags & 0x0001).toBe(0x0001); // compressed bit
    expect(info.flags & 0x0004).toBe(0x0004); // lz4 bit
    expect(info.flags & 0x0002).toBe(0x0000); // zstd bit must be clear
  });

  it('writeEntryWithOptions explicit lz4', () => {
    const data = makeRepetitiveData(8000);

    const w = new ShardV2Writer();
    w.writeEntryWithOptions('x', data, true, COMPRESS_LZ4);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), data)).toBe(0);
    expect(r.getEntryInfo(0).compressed).toBe(true);
  });

  it('large data roundtrip (100KB)', () => {
    const data = makeRepetitiveData(100_000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('large', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), data)).toBe(0);
  });
});

// ============================================================
// Mixed entries
// ============================================================

describe('Mixed entries', () => {
  it('compressed (zstd) and uncompressed entries together', () => {
    const big = makeRepetitiveData(5000);
    const small = Buffer.from('tiny');

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('compressed', big);
    w.writeEntry('plain', small);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(2);
    expect(Buffer.compare(r.readEntry(0), big)).toBe(0);
    expect(r.readEntry(1).toString()).toBe('tiny');
    expect(r.getEntryInfo(0).compressed).toBe(true);
    expect(r.getEntryInfo(1).compressed).toBe(false);
  });

  it('compressed (lz4) and uncompressed entries together', () => {
    const big = makeRepetitiveData(5000);
    const small = Buffer.from('tiny');

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_LZ4);
    w.writeEntryCompressed('compressed', big);
    w.writeEntry('plain', small);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntry(0), big)).toBe(0);
    expect(r.readEntry(1).toString()).toBe('tiny');
    expect(r.getEntryInfo(0).compressed).toBe(true);
    expect(r.getEntryInfo(1).compressed).toBe(false);
  });

  it('zstd and lz4 entries in the same shard', () => {
    const data1 = makeRepetitiveData(4000);
    const data2 = makeRepetitiveData(3000);

    const w = new ShardV2Writer();
    w.writeEntryWithOptions('zstd-entry', data1, true, COMPRESS_ZSTD);
    w.writeEntryWithOptions('lz4-entry', data2, true, COMPRESS_LZ4);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(2);
    expect(Buffer.compare(r.readEntry(0), data1)).toBe(0);
    expect(Buffer.compare(r.readEntry(1), data2)).toBe(0);

    const info0 = r.getEntryInfo(0);
    const info1 = r.getEntryInfo(1);
    expect(info0.flags & 0x0002).toBe(0x0002); // zstd
    expect(info1.flags & 0x0004).toBe(0x0004); // lz4
  });

  it('name lookup works on compressed entries', () => {
    const data = makeRepetitiveData(5000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('layer/weight', data);
    w.writeEntry('layer/bias', Buffer.from('small'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Buffer.compare(r.readEntryByName('layer/weight'), data)).toBe(0);
    expect(r.readEntryByName('layer/bias').toString()).toBe('small');
  });

  it('orig_size and disk_size are correctly recorded', () => {
    const data = makeRepetitiveData(10_000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('data', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    expect(Number(info.origSize)).toBe(data.length);
    expect(Number(info.diskSize)).toBeLessThan(data.length);
  });

  it('total_file_size matches buffer length with compressed entries', () => {
    const data = makeRepetitiveData(8000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('a', data);
    w.writeEntry('b', Buffer.from('plain'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Number(r.header().totalFileSize)).toBe(buf.length);
  });

  it('alignment is preserved with compressed entries', () => {
    const data = makeRepetitiveData(5000);

    const w = new ShardV2Writer();
    w.setCompression(COMPRESS_ZSTD);
    w.writeEntryCompressed('x', data);
    w.writeEntryCompressed('y', makeRepetitiveData(3000));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    // Default alignment is 64
    expect(r.header().alignment).toBe(64);
    for (let i = 0; i < r.entryCount(); i++) {
      expect(Number(r.getEntryInfo(i).dataOffset) % 64).toBe(0);
    }
  });
});
