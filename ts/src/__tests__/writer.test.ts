/**
 * Roundtrip tests — verify TypeScript writer produces shards the reader can parse.
 */

import { createHash } from 'node:crypto';
import { describe, it, expect, beforeAll } from 'vitest';

import {
  ShardV2Reader,
  ShardV2Writer,
  computeCrc32c,
  computeXxhash64,
  initXxhash,
  ALIGN_NONE,
  ALIGN_16,
  ALIGN_32,
  ALIGN_64,
  CONTENT_TYPE_UNKNOWN,
  CONTENT_TYPE_TENSOR,
  CONTENT_TYPE_JSON,
  CONTENT_TYPE_TEXT,
  CONTENT_TYPE_BLOB,
  ROLE_MOSH,
  ROLE_SAMPLE,
  ROLE_WSHARD,
  DEFAULT_FLAGS,
} from '../index.js';

beforeAll(async () => {
  await initXxhash();
});

function sha256hex(data: Buffer): string {
  return createHash('sha256').update(data).digest('hex');
}

// ============================================================
// TestWriterRoundtrip
// ============================================================

describe('writer roundtrip', () => {
  it('single entry', () => {
    const w = new ShardV2Writer();
    w.writeEntry('hello', Buffer.from('world'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(1);
    expect(r.entryNames()).toEqual(['hello']);
    expect(r.readEntry(0)).toEqual(Buffer.from('world'));
  });

  it('multiple entries', () => {
    const entries: [string, Buffer][] = [
      ['alpha', Buffer.from('data_alpha')],
      ['beta', Buffer.from('data_beta_longer')],
      ['gamma', Buffer.from('g')],
    ];

    const w = new ShardV2Writer();
    for (const [name, data] of entries) {
      w.writeEntry(name, data);
    }
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(3);
    for (let i = 0; i < entries.length; i++) {
      const [name, data] = entries[i];
      expect(r.entryName(i)).toBe(name);
      expect(r.readEntry(i)).toEqual(data);
    }
  });

  it('typed entries', () => {
    const w = new ShardV2Writer();
    w.writeEntryTyped('model.bin', Buffer.alloc(256, 0), CONTENT_TYPE_TENSOR);
    w.writeEntryTyped('config.json', Buffer.from('{"key": "value"}'), CONTENT_TYPE_JSON);
    w.writeEntryTyped('readme.txt', Buffer.from('hello world'), CONTENT_TYPE_TEXT);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(3);
    expect(r.getEntryInfo(0).contentType).toBe(CONTENT_TYPE_TENSOR);
    expect(r.getEntryInfo(1).contentType).toBe(CONTENT_TYPE_JSON);
    expect(r.getEntryInfo(2).contentType).toBe(CONTENT_TYPE_TEXT);
  });

  it('empty shard', () => {
    const w = new ShardV2Writer();
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(0);
    expect(r.entryNames()).toEqual([]);
  });

  it('large entry (100KB)', () => {
    const bigData = Buffer.from(Array.from({ length: 100_000 }, (_, i) => i % 256));

    const w = new ShardV2Writer();
    w.writeEntry('big', bigData);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntry(0)).toEqual(bigData);
  });

  it('alignment none', () => {
    const w = new ShardV2Writer();
    w.setAlignment(ALIGN_NONE);
    w.writeEntry('a', Buffer.from('short'));
    w.writeEntry('b', Buffer.from('another'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.header().alignment).toBe(ALIGN_NONE);
    expect(r.readEntry(0)).toEqual(Buffer.from('short'));
    expect(r.readEntry(1)).toEqual(Buffer.from('another'));
  });

  it('alignment 16', () => {
    const w = new ShardV2Writer();
    w.setAlignment(ALIGN_16);
    w.writeEntry('x', Buffer.from('hello'));
    w.writeEntry('y', Buffer.from('world'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.header().alignment).toBe(ALIGN_16);
    for (let i = 0; i < r.entryCount(); i++) {
      const info = r.getEntryInfo(i);
      expect(Number(info.dataOffset) % 16, `entry ${i} data_offset not 16-aligned`).toBe(0);
    }
    expect(r.readEntry(0)).toEqual(Buffer.from('hello'));
    expect(r.readEntry(1)).toEqual(Buffer.from('world'));
  });

  it('alignment 64 (default)', () => {
    const w = new ShardV2Writer();
    w.writeEntry('p', Buffer.from('payload'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.header().alignment).toBe(ALIGN_64);
    const info = r.getEntryInfo(0);
    expect(Number(info.dataOffset) % 64).toBe(0);
  });

  it('role setting', () => {
    const w = new ShardV2Writer(ROLE_SAMPLE);
    w.writeEntry('data', Buffer.from('sample_data'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.header().role).toBe(ROLE_SAMPLE);
  });

  it('checksum verification', () => {
    const w = new ShardV2Writer();
    w.writeEntry('test', Buffer.from('checksum_data'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    const expected = computeCrc32c(Buffer.from('checksum_data'));
    expect(info.checksum).toBe(expected);
  });

  it('name hash verification', () => {
    const w = new ShardV2Writer();
    w.writeEntry('greeting', Buffer.from('hi'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info = r.getEntryInfo(0);
    expect(info.nameHash).toBe(computeXxhash64('greeting'));
  });

  it('lookup by name', () => {
    const w = new ShardV2Writer();
    w.writeEntry('first', Buffer.from('1'));
    w.writeEntry('second', Buffer.from('2'));
    w.writeEntry('third', Buffer.from('3'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.lookup('first')).toBe(0);
    expect(r.lookup('second')).toBe(1);
    expect(r.lookup('third')).toBe(2);
    expect(r.lookup('missing')).toBe(-1);
    expect(r.hasEntry('second')).toBe(true);
    expect(r.hasEntry('missing')).toBe(false);
  });

  it('read by name', () => {
    const w = new ShardV2Writer();
    w.writeEntry('key1', Buffer.from('value1'));
    w.writeEntry('key2', Buffer.from('value2'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntryByName('key1')).toEqual(Buffer.from('value1'));
    expect(r.readEntryByName('key2')).toEqual(Buffer.from('value2'));
    expect(() => r.readEntryByName('nonexistent')).toThrow();
  });

  it('list prefix', () => {
    const w = new ShardV2Writer();
    w.writeEntry('layer.0/weight', Buffer.from('w0'));
    w.writeEntry('layer.0/bias', Buffer.from('b0'));
    w.writeEntry('layer.1/weight', Buffer.from('w1'));
    w.writeEntry('embed/tokens', Buffer.from('tok'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);

    const l0 = r.listPrefix('layer.0/');
    expect(l0.length).toBe(2);
    expect(l0).toContain('layer.0/weight');
    expect(l0).toContain('layer.0/bias');

    const layers = r.listPrefix('layer.');
    expect(layers.length).toBe(3);

    const embed = r.listPrefix('embed/');
    expect(embed.length).toBe(1);

    const none = r.listPrefix('nonexistent/');
    expect(none.length).toBe(0);
  });

  it('binary data roundtrip — all 256 byte values', () => {
    const allBytes = Buffer.from(Array.from({ length: 256 }, (_, i) => i));

    const w = new ShardV2Writer();
    w.writeEntry('allbytes', allBytes);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntry(0)).toEqual(allBytes);
  });

  it('total_file_size matches buffer length', () => {
    const w = new ShardV2Writer();
    w.writeEntry('a', Buffer.from('hello'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(Number(r.header().totalFileSize)).toBe(buf.length);
  });

  it('SHA256 roundtrip', () => {
    const data = Buffer.from('The quick brown fox jumps over the lazy dog');
    const expectedSha = sha256hex(data);

    const w = new ShardV2Writer();
    w.writeEntry('fox', data);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(sha256hex(r.readEntry(0))).toBe(expectedSha);
  });
});

// ============================================================
// Edge cases
// ============================================================

describe('writer edge cases', () => {
  it('empty data entry', () => {
    const w = new ShardV2Writer();
    w.writeEntry('empty', Buffer.alloc(0));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntry(0)).toEqual(Buffer.alloc(0));
    expect(Number(r.getEntryInfo(0).origSize)).toBe(0);
    expect(Number(r.getEntryInfo(0).diskSize)).toBe(0);
  });

  it('single byte', () => {
    const w = new ShardV2Writer();
    w.writeEntry('one', Buffer.from([0xFF]));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.readEntry(0)).toEqual(Buffer.from([0xFF]));
  });

  it('many entries (100)', () => {
    const n = 100;
    const w = new ShardV2Writer();
    for (let i = 0; i < n; i++) {
      w.writeEntry(`entry_${String(i).padStart(4, '0')}`, Buffer.from(`data_${i}`));
    }
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.entryCount()).toBe(n);
    for (let i = 0; i < n; i++) {
      expect(r.entryName(i)).toBe(`entry_${String(i).padStart(4, '0')}`);
      expect(r.readEntry(i)).toEqual(Buffer.from(`data_${i}`));
    }
  });

  it('ROLE_WSHARD is preserved', () => {
    const w = new ShardV2Writer(ROLE_WSHARD);
    w.writeEntry('signal/imu', Buffer.from([1, 2, 3]));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.header().role).toBe(ROLE_WSHARD);
  });

  it('DEFAULT_FLAGS = 162 = 0x00A2', () => {
    expect(DEFAULT_FLAGS).toBe(162);
    expect(DEFAULT_FLAGS).toBe(0x00A2);
  });

  it('header flags are DEFAULT_FLAGS', () => {
    const w = new ShardV2Writer();
    w.writeEntry('x', Buffer.from('y'));
    const buf = w.toBuffer();
    const r = new ShardV2Reader(buf);
    expect(r.header().flags).toBe(DEFAULT_FLAGS);
  });

  it('index_entry_size is always 48', () => {
    const w = new ShardV2Writer();
    w.writeEntry('x', Buffer.from('y'));
    const buf = w.toBuffer();
    const r = new ShardV2Reader(buf);
    expect(r.header().indexEntrySize).toBe(48);
  });

  it('string table offset = 64 + N*48', () => {
    const n = 5;
    const w = new ShardV2Writer();
    for (let i = 0; i < n; i++) {
      w.writeEntry(`e${i}`, Buffer.from(`data${i}`));
    }
    const buf = w.toBuffer();
    const r = new ShardV2Reader(buf);
    expect(Number(r.header().stringTableOffset)).toBe(64 + n * 48);
  });

  it('entries are 64-byte aligned by default', () => {
    const w = new ShardV2Writer();
    // Write entries of varying sizes to force alignment padding
    w.writeEntry('a', Buffer.from('x'));      // 1 byte
    w.writeEntry('b', Buffer.from('yyyyyy')); // 6 bytes
    w.writeEntry('c', Buffer.alloc(15, 0));   // 15 bytes
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    for (let i = 0; i < 3; i++) {
      expect(Number(r.getEntryInfo(i).dataOffset) % 64, `entry ${i} not 64-aligned`).toBe(0);
    }
  });

  it('no-alignment: entries are packed tightly', () => {
    const w = new ShardV2Writer();
    w.setAlignment(ALIGN_NONE);
    w.writeEntry('a', Buffer.from('abc'));
    w.writeEntry('b', Buffer.from('de'));
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    const info0 = r.getEntryInfo(0);
    const info1 = r.getEntryInfo(1);
    // Second entry starts immediately after first
    expect(Number(info1.dataOffset)).toBe(Number(info0.dataOffset) + Number(info0.diskSize));
  });

  it('content_type blob survives roundtrip', () => {
    const w = new ShardV2Writer();
    w.writeEntryTyped('bin', Buffer.from([0xDE, 0xAD, 0xBE, 0xEF]), CONTENT_TYPE_BLOB);
    const buf = w.toBuffer();

    const r = new ShardV2Reader(buf);
    expect(r.getEntryInfo(0).contentType).toBe(CONTENT_TYPE_BLOB);
    expect(r.readEntry(0)).toEqual(Buffer.from([0xDE, 0xAD, 0xBE, 0xEF]));
  });
});
