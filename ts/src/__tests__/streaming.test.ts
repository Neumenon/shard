/**
 * Streaming writer roundtrip tests — ShardStreamWriter.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import {
  ShardStreamWriter,
  ShardReader,
  initXxhash,
  initCompression,
  computeCrc32c,
  FLAG_STREAMING,
  ROLE_MOSH,
  ROLE_SAMPLE,
  ALIGN_NONE,
  ALIGN_16,
  ALIGN_32,
  ALIGN_64,
  COMPRESS_ZSTD,
  COMPRESS_LZ4,
  DEFAULT_FLAGS,
} from '../index.js';

beforeAll(async () => {
  await initXxhash();
  await initCompression();
});

function tmpFile(): string {
  return path.join(
    os.tmpdir(),
    `shard_stream_test_${Date.now()}_${Math.random().toString(36).slice(2)}.shard`,
  );
}

// ============================================================
// Basic roundtrip
// ============================================================

describe('ShardStreamWriter basic roundtrip', () => {
  it('two entries — names and data preserved', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.beginData();
    sw.writeEntry('hello', Buffer.from('world'));
    sw.writeEntry('foo', Buffer.from('bar'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.entryCount()).toBe(2);
    expect(r.readEntry(0).toString()).toBe('world');
    expect(r.readEntry(1).toString()).toBe('bar');
    expect(r.entryName(0)).toBe('hello');
    expect(r.entryName(1)).toBe('foo');
    fs.unlinkSync(p);
  });

  it('FLAG_STREAMING is set in header flags', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('x', Buffer.from('data'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().flags & FLAG_STREAMING).toBeTruthy();
    fs.unlinkSync(p);
  });

  it('DEFAULT_FLAGS also present alongside FLAG_STREAMING', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('x', Buffer.from('data'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    // DEFAULT_FLAGS bits must all be present
    expect(r.header().flags & DEFAULT_FLAGS).toBe(DEFAULT_FLAGS);
    fs.unlinkSync(p);
  });

  it('single entry roundtrip', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('only', Buffer.from('payload'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.entryCount()).toBe(1);
    expect(r.readEntry(0).toString()).toBe('payload');
    fs.unlinkSync(p);
  });

  it('role is propagated to header', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_SAMPLE, 5);
    sw.beginData();
    sw.writeEntry('a', Buffer.from('1'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().role).toBe(ROLE_SAMPLE);
    fs.unlinkSync(p);
  });

  it('total_file_size matches file length on disk', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('item', Buffer.from('hello'));
    sw.finalize();

    const buf = fs.readFileSync(p);
    const r = new ShardReader(buf);
    expect(Number(r.header().totalFileSize)).toBe(buf.length);
    fs.unlinkSync(p);
  });

  it('checksum stored correctly', () => {
    const p = tmpFile();
    const data = Buffer.from('checksum_test_data');
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('data', data);
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    const info = r.getEntryInfo(0);
    expect(info.checksum).toBe(computeCrc32c(data));
    fs.unlinkSync(p);
  });

  it('binary data — all 256 byte values', () => {
    const p = tmpFile();
    const allBytes = Buffer.from(Array.from({ length: 256 }, (_, i) => i));
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('allbytes', allBytes);
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntry(0)).toEqual(allBytes);
    fs.unlinkSync(p);
  });
});

// ============================================================
// Alignment
// ============================================================

describe('ShardStreamWriter alignment', () => {
  it('default alignment is ALIGN_64', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.beginData();
    sw.writeEntry('a', Buffer.from('data_a'));
    sw.writeEntry('b', Buffer.from('data_b'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().alignment).toBe(ALIGN_64);
    for (let i = 0; i < r.entryCount(); i++) {
      const offset = Number(r.getEntryInfo(i).dataOffset);
      expect(offset % 64, `entry ${i} data_offset not 64-aligned`).toBe(0);
    }
    fs.unlinkSync(p);
  });

  it('alignment 64 explicit', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.setAlignment(ALIGN_64);
    sw.beginData();
    sw.writeEntry('x', Buffer.from('hello'));
    sw.writeEntry('y', Buffer.from('world'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().alignment).toBe(ALIGN_64);
    for (let i = 0; i < r.entryCount(); i++) {
      expect(Number(r.getEntryInfo(i).dataOffset) % 64).toBe(0);
    }
    fs.unlinkSync(p);
  });

  it('alignment 16', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.setAlignment(ALIGN_16);
    sw.beginData();
    sw.writeEntry('x', Buffer.from('hello'));
    sw.writeEntry('y', Buffer.from('world_longer'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().alignment).toBe(ALIGN_16);
    for (let i = 0; i < r.entryCount(); i++) {
      const offset = Number(r.getEntryInfo(i).dataOffset);
      expect(offset % 16, `entry ${i} data_offset not 16-aligned`).toBe(0);
    }
    fs.unlinkSync(p);
  });

  it('alignment 32', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.setAlignment(ALIGN_32);
    sw.beginData();
    sw.writeEntry('p', Buffer.from('payload_data'));
    sw.writeEntry('q', Buffer.from('more_payload'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().alignment).toBe(ALIGN_32);
    for (let i = 0; i < r.entryCount(); i++) {
      expect(Number(r.getEntryInfo(i).dataOffset) % 32).toBe(0);
    }
    fs.unlinkSync(p);
  });

  it('alignment none — entries packed tightly', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.setAlignment(ALIGN_NONE);
    sw.beginData();
    sw.writeEntry('a', Buffer.from('x'));
    sw.writeEntry('b', Buffer.from('yy'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.header().alignment).toBe(ALIGN_NONE);
    expect(r.readEntry(0).toString()).toBe('x');
    expect(r.readEntry(1).toString()).toBe('yy');
    const info0 = r.getEntryInfo(0);
    const info1 = r.getEntryInfo(1);
    expect(Number(info1.dataOffset)).toBe(Number(info0.dataOffset) + Number(info0.diskSize));
    fs.unlinkSync(p);
  });

  it('setAlignment after beginData throws', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    expect(() => sw.setAlignment(ALIGN_16)).toThrow('cannot set alignment after begin_data()');
    sw.finalize();
    fs.unlinkSync(p);
  });

  it('invalid alignment value throws', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    expect(() => sw.setAlignment(7)).toThrow(/invalid alignment/);
    sw.close();
    try { fs.unlinkSync(p); } catch (_) { /* may not exist */ }
  });
});

// ============================================================
// Many entries / large data
// ============================================================

describe('ShardStreamWriter many entries and large data', () => {
  it('100 entries roundtrip', () => {
    const p = tmpFile();
    const n = 100;
    const sw = new ShardStreamWriter(p, ROLE_MOSH, n);
    sw.beginData();
    for (let i = 0; i < n; i++) {
      sw.writeEntry(`entry_${String(i).padStart(4, '0')}`, Buffer.from(`data_${i}`));
    }
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.entryCount()).toBe(n);
    for (let i = 0; i < n; i++) {
      expect(r.entryName(i)).toBe(`entry_${String(i).padStart(4, '0')}`);
      expect(r.readEntry(i).toString()).toBe(`data_${i}`);
    }
    fs.unlinkSync(p);
  });

  it('fewer entries than maxEntries is fine', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 1000);
    sw.beginData();
    sw.writeEntry('only_one', Buffer.from('data'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.entryCount()).toBe(1);
    expect(r.readEntry(0).toString()).toBe('data');
    fs.unlinkSync(p);
  });

  it('large data (100KB)', () => {
    const p = tmpFile();
    const big = Buffer.from(Array.from({ length: 100_000 }, (_, i) => i % 256));
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.writeEntry('big', big);
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntry(0)).toEqual(big);
    fs.unlinkSync(p);
  });
});

// ============================================================
// Lookup
// ============================================================

describe('ShardStreamWriter lookup', () => {
  it('lookup by name returns correct index', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.beginData();
    sw.writeEntry('alpha', Buffer.from('a'));
    sw.writeEntry('beta', Buffer.from('b'));
    sw.writeEntry('gamma', Buffer.from('c'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.lookup('alpha')).toBe(0);
    expect(r.lookup('beta')).toBe(1);
    expect(r.lookup('gamma')).toBe(2);
    expect(r.lookup('missing')).toBe(-1);
    fs.unlinkSync(p);
  });

  it('readEntryByName works', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.beginData();
    sw.writeEntry('key1', Buffer.from('value1'));
    sw.writeEntry('key2', Buffer.from('value2'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntryByName('key1').toString()).toBe('value1');
    expect(r.readEntryByName('key2').toString()).toBe('value2');
    expect(() => r.readEntryByName('nonexistent')).toThrow();
    fs.unlinkSync(p);
  });
});

// ============================================================
// Compression
// ============================================================

describe('ShardStreamWriter compression', () => {
  it('zstd compressed entry roundtrip', () => {
    const p = tmpFile();
    const data = Buffer.from(Array.from({ length: 10_000 }, (_, i) => i % 256));
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.setCompression(COMPRESS_ZSTD);
    sw.beginData();
    sw.writeEntryCompressed('data', data);
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntry(0)).toEqual(data);
    const info = r.getEntryInfo(0);
    expect(info.compressed).toBe(true);
    expect(Number(info.diskSize)).toBeLessThan(Number(info.origSize));
    fs.unlinkSync(p);
  });

  it('lz4 compressed entry roundtrip', () => {
    const p = tmpFile();
    const data = Buffer.from(Array.from({ length: 10_000 }, (_, i) => i % 256));
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.setCompression(COMPRESS_LZ4);
    sw.beginData();
    sw.writeEntryCompressed('data', data);
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntry(0)).toEqual(data);
    const info = r.getEntryInfo(0);
    expect(info.compressed).toBe(true);
    expect(Number(info.diskSize)).toBeLessThan(Number(info.origSize));
    fs.unlinkSync(p);
  });

  it('small data below MIN_COMPRESS_SIZE stored uncompressed', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.setCompression(COMPRESS_ZSTD);
    sw.beginData();
    sw.writeEntryCompressed('tiny', Buffer.from('hello'));
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntry(0).toString()).toBe('hello');
    expect(r.getEntryInfo(0).compressed).toBe(false);
    fs.unlinkSync(p);
  });

  it('mixed compressed and uncompressed in same shard', () => {
    const p = tmpFile();
    const big = Buffer.from(Array.from({ length: 10_000 }, (_, i) => i % 256));
    const small = Buffer.from('tiny');
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 10);
    sw.setCompression(COMPRESS_ZSTD);
    sw.beginData();
    sw.writeEntryCompressed('big', big);
    sw.writeEntry('small', small);
    sw.finalize();

    const r = new ShardReader(fs.readFileSync(p));
    expect(r.readEntry(0)).toEqual(big);
    expect(r.readEntry(1)).toEqual(small);
    expect(r.getEntryInfo(0).compressed).toBe(true);
    expect(r.getEntryInfo(1).compressed).toBe(false);
    fs.unlinkSync(p);
  });
});

// ============================================================
// State errors
// ============================================================

describe('ShardStreamWriter state errors', () => {
  it('writeEntry before beginData throws', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    expect(() => sw.writeEntry('x', Buffer.from('data'))).toThrow('call begin_data() first');
    sw.close();
    try { fs.unlinkSync(p); } catch (_) { /* may not exist */ }
  });

  it('beginData called twice throws', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    expect(() => sw.beginData()).toThrow('already called');
    sw.finalize();
    fs.unlinkSync(p);
  });

  it('finalize called twice throws', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.finalize();
    expect(() => sw.finalize()).toThrow('already finalized');
    fs.unlinkSync(p);
  });

  it('writeEntry after finalize throws', () => {
    const p = tmpFile();
    const sw = new ShardStreamWriter(p, ROLE_MOSH, 5);
    sw.beginData();
    sw.finalize();
    expect(() => sw.writeEntry('x', Buffer.from('data'))).toThrow('already finalized');
    fs.unlinkSync(p);
  });
});
