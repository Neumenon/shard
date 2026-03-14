/**
 * Tests for metadata/schema, security limits, listChildren, readEntryPrefix,
 * and path helpers added to shard TypeScript.
 */

import { describe, it, expect, beforeAll } from 'vitest';

import {
  ShardReader,
  ShardWriter,
  ShardMetadata,
  EntryMeta,
  initXxhash,
  FLAG_HAS_SCHEMA,
  MAX_ENTRY_COUNT,
  MAX_INDEX_SIZE,
  MAX_STRING_TABLE_SIZE,
  splitPath,
  joinPath,
  pathParent,
  pathBase,
  serializeMetadata,
  deserializeMetadata,
} from '../index.js';

beforeAll(async () => {
  await initXxhash();
});

// ============================================================
// Helper
// ============================================================

function makeShardBuffer(
  entries: Array<[string, Buffer]>,
  metadata?: ShardMetadata,
): Buffer {
  const w = new ShardWriter();
  for (const [name, data] of entries) {
    w.writeEntry(name, data);
  }
  if (metadata !== undefined) {
    w.setMetadata(metadata);
  }
  return w.toBuffer();
}

// ============================================================
// Path helpers
// ============================================================

describe('path helpers', () => {
  it('splitPath simple', () => {
    expect(splitPath('layer.0/weight')).toEqual(['layer.0', 'weight']);
  });

  it('splitPath deep', () => {
    expect(splitPath('a/b/c/d')).toEqual(['a', 'b', 'c', 'd']);
  });

  it('splitPath no slash', () => {
    expect(splitPath('embed')).toEqual(['embed']);
  });

  it('splitPath empty', () => {
    expect(splitPath('')).toEqual([]);
  });

  it('joinPath two parts', () => {
    expect(joinPath('layer.0', 'weight')).toBe('layer.0/weight');
  });

  it('joinPath single', () => {
    expect(joinPath('embed')).toBe('embed');
  });

  it('joinPath multi', () => {
    expect(joinPath('a', 'b', 'c')).toBe('a/b/c');
  });

  it('pathParent with slash', () => {
    expect(pathParent('layer.0/weight')).toBe('layer.0');
  });

  it('pathParent deep', () => {
    expect(pathParent('a/b/c')).toBe('a/b');
  });

  it('pathParent no slash', () => {
    expect(pathParent('embed')).toBe('');
  });

  it('pathBase with slash', () => {
    expect(pathBase('layer.0/weight')).toBe('weight');
  });

  it('pathBase deep', () => {
    expect(pathBase('a/b/c')).toBe('c');
  });

  it('pathBase no slash', () => {
    expect(pathBase('embed')).toBe('embed');
  });
});

// ============================================================
// listChildren
// ============================================================

describe('listChildren', () => {
  it('exact prefix with trailing slash', () => {
    const buf = makeShardBuffer([
      ['layer.0/weight', Buffer.from('w0')],
      ['layer.0/bias', Buffer.from('b0')],
      ['layer.1/weight', Buffer.from('w1')],
      ['embed', Buffer.from('tok')],
    ]);
    const r = new ShardReader(buf);
    const result = r.listChildren('layer.0/');
    expect(result.sort()).toEqual(['bias', 'weight']);
  });

  it('empty prefix returns top-level children', () => {
    const buf = makeShardBuffer([
      ['layer.0/weight', Buffer.from('w0')],
      ['layer.0/bias', Buffer.from('b0')],
      ['layer.1/weight', Buffer.from('w1')],
      ['embed', Buffer.from('tok')],
    ]);
    const r = new ShardReader(buf);
    const result = r.listChildren('');
    expect(result).toContain('layer.0');
    expect(result).toContain('layer.1');
    expect(result).toContain('embed');
    // No duplicates
    expect(result.length).toBe(new Set(result).size);
  });

  it('partial prefix returns directory entries', () => {
    const buf = makeShardBuffer([
      ['layer.0/weight', Buffer.from('w0')],
      ['layer.0/bias', Buffer.from('b0')],
      ['layer.1/weight', Buffer.from('w1')],
      ['embed', Buffer.from('tok')],
    ]);
    const r = new ShardReader(buf);
    const result = r.listChildren('layer.');
    expect(result.sort()).toEqual(['0', '1']);
  });

  it('nonexistent prefix returns empty', () => {
    const buf = makeShardBuffer([
      ['a/b', Buffer.from('x')],
    ]);
    const r = new ShardReader(buf);
    expect(r.listChildren('nonexistent/')).toEqual([]);
  });

  it('deduplicated directories', () => {
    const buf = makeShardBuffer([
      ['a/b', Buffer.from('1')],
      ['a/c', Buffer.from('2')],
      ['a/d', Buffer.from('3')],
    ]);
    const r = new ShardReader(buf);
    // Under "" we should see "a" only once
    const result = r.listChildren('');
    expect(result.filter(x => x === 'a').length).toBe(1);
  });

  it('hierarchical three levels', () => {
    const buf = makeShardBuffer([
      ['a/b/c', Buffer.from('1')],
      ['a/b/d', Buffer.from('2')],
      ['a/e', Buffer.from('3')],
      ['f', Buffer.from('4')],
    ]);
    const r = new ShardReader(buf);

    const top = r.listChildren('');
    expect(top).toContain('a');
    expect(top).toContain('f');

    const underA = r.listChildren('a/');
    expect(underA).toContain('b');
    expect(underA).toContain('e');

    const underAB = r.listChildren('a/b/');
    expect(underAB.sort()).toEqual(['c', 'd']);
  });
});

// ============================================================
// readEntryPrefix
// ============================================================

describe('readEntryPrefix', () => {
  const data = Buffer.from('Hello World, this is a test payload!');

  it('returns first N bytes', () => {
    const buf = makeShardBuffer([['entry', data]]);
    const r = new ShardReader(buf);
    expect(r.readEntryPrefix(0, 5)).toEqual(Buffer.from('Hello'));
  });

  it('returns full entry when maxBytes equals entry length', () => {
    const buf = makeShardBuffer([['entry', data]]);
    const r = new ShardReader(buf);
    expect(r.readEntryPrefix(0, data.length)).toEqual(data);
  });

  it('returns full entry when maxBytes exceeds entry length', () => {
    const buf = makeShardBuffer([['entry', data]]);
    const r = new ShardReader(buf);
    expect(r.readEntryPrefix(0, 100_000)).toEqual(data);
  });

  it('returns empty buffer for maxBytes = 0', () => {
    const buf = makeShardBuffer([['entry', data]]);
    const r = new ShardReader(buf);
    expect(r.readEntryPrefix(0, 0)).toEqual(Buffer.alloc(0));
  });

  it('returns one byte', () => {
    const buf = makeShardBuffer([['entry', data]]);
    const r = new ShardReader(buf);
    expect(r.readEntryPrefix(0, 1)).toEqual(Buffer.from('H'));
  });

  it('throws on out-of-bounds index', () => {
    const buf = makeShardBuffer([['entry', data]]);
    const r = new ShardReader(buf);
    expect(() => r.readEntryPrefix(99, 10)).toThrow();
  });
});

// ============================================================
// ShardMetadata roundtrip
// ============================================================

describe('ShardMetadata roundtrip', () => {
  it('basic metadata survives write/read', () => {
    const meta: ShardMetadata = {
      schemaVersion: 'shard-v2.1',
      producer: 'test-producer',
      description: 'A test shard',
      createdAt: '2026-02-21T00:00:00Z',
      tags: ['ml', 'test'],
    };
    const buf = makeShardBuffer([['data', Buffer.from('hello')]], meta);
    const r = new ShardReader(buf);
    expect(r.header().flags & FLAG_HAS_SCHEMA).toBeTruthy();
    expect(Number(r.header().schemaOffset)).toBeGreaterThan(0);

    const got = r.readMetadata();
    expect(got).not.toBeNull();
    expect(got!.producer).toBe('test-producer');
    expect(got!.description).toBe('A test shard');
    expect(got!.createdAt).toBe('2026-02-21T00:00:00Z');
    expect(got!.tags).toEqual(['ml', 'test']);
    expect(got!.schemaVersion).toBe('shard-v2.1');
  });

  it('entry metadata roundtrip', () => {
    const em: EntryMeta = {
      contentType: 'tensor/float32',
      tags: ['weight'],
      description: 'Model weights',
      extra: { shape: [256, 256] },
      codec: 'cowrie-gen2',
      codecVersion: '2',
      schemaFingerprint: 'sha256:weights',
      semanticType: 'tensor',
      canonicalHash: 'sha256:canonical',
      baseHash: 'sha256:base',
      rowCount: 256,
      shape: [256, 256],
      stats: { min: -1, max: 1 },
    };
    const meta: ShardMetadata = {
      schemaVersion: 'shard-v2.1',
      producer: 'pytorch-exporter',
      entryMetadata: { 'model/weight': em },
    };
    const buf = makeShardBuffer([['model/weight', Buffer.alloc(16)]], meta);
    const r = new ShardReader(buf);
    const got = r.readMetadata();
    expect(got).not.toBeNull();
    expect(got!.entryMetadata).toBeDefined();
    const gotEm = got!.entryMetadata!['model/weight'];
    expect(gotEm).toBeDefined();
    expect(gotEm.contentType).toBe('tensor/float32');
    expect(gotEm.tags).toEqual(['weight']);
    expect(gotEm.description).toBe('Model weights');
    expect(gotEm.extra).toEqual({ shape: [256, 256] });
    expect(gotEm.codec).toBe('cowrie-gen2');
    expect(gotEm.codecVersion).toBe('2');
    expect(gotEm.schemaFingerprint).toBe('sha256:weights');
    expect(gotEm.semanticType).toBe('tensor');
    expect(gotEm.canonicalHash).toBe('sha256:canonical');
    expect(gotEm.baseHash).toBe('sha256:base');
    expect(gotEm.rowCount).toBe(256);
    expect(gotEm.shape).toEqual([256, 256]);
    expect(gotEm.stats).toEqual({ min: -1, max: 1 });
  });

  it('no metadata returns null', () => {
    const buf = makeShardBuffer([['data', Buffer.from('hello')]]);
    const r = new ShardReader(buf);
    expect(Number(r.header().schemaOffset)).toBe(0);
    expect(r.readMetadata()).toBeNull();
  });

  it('metadata with extra fields', () => {
    const meta: ShardMetadata = {
      schemaVersion: 'shard-v2.1',
      extra: { model_type: 'transformer', num_layers: 12 },
    };
    const buf = makeShardBuffer([['cfg', Buffer.from('x')]], meta);
    const r = new ShardReader(buf);
    const got = r.readMetadata();
    expect(got).not.toBeNull();
    expect(got!.extra!['model_type']).toBe('transformer');
    expect(got!.extra!['num_layers']).toBe(12);
  });

  it('metadata source_uri and schema_uri roundtrip', () => {
    const meta: ShardMetadata = {
      schemaVersion: 'shard-v2.1',
      sourceUri: 's3://bucket/model.safetensors',
      schemaUri: 'https://example.com/schema.json',
    };
    const buf = makeShardBuffer([['x', Buffer.from('y')]], meta);
    const r = new ShardReader(buf);
    const got = r.readMetadata();
    expect(got!.sourceUri).toBe('s3://bucket/model.safetensors');
    expect(got!.schemaUri).toBe('https://example.com/schema.json');
  });

  it('serializeMetadata / deserializeMetadata are inverses', () => {
    const meta: ShardMetadata = {
      schemaVersion: 'shard-v2.1',
      producer: 'unit-test',
      tags: ['a', 'b'],
      extra: { n: 42 },
      entryMetadata: {
        'foo/bar': { contentType: 'text/plain', description: 'test' },
      },
    };
    const serialized = serializeMetadata(meta);
    const got = deserializeMetadata(serialized);
    expect(got.producer).toBe('unit-test');
    expect(got.tags).toEqual(['a', 'b']);
    expect(got.extra!['n']).toBe(42);
    expect(got.entryMetadata!['foo/bar'].contentType).toBe('text/plain');
    expect(got.entryMetadata!['foo/bar'].description).toBe('test');
  });

  it('profile metadata roundtrip', () => {
    const meta: ShardMetadata = {
      schemaVersion: 'shard-v2.1',
      profile: 'sampleshard.v1',
      sampleShard: {
        datasetName: 'mnist-train',
        sampleIdType: 'uint64',
        keyEncoding: 'decimal-string',
        sampleCount: 60000,
        datasetSchema: { input: 'tensor[u8,28,28]', target: 'uint8' },
        splits: { train: { start: 0, end: 59999 } },
        labelMap: { '0': 'zero', '1': 'one' },
        featureStats: { input: { mean: 0.1307, std: 0.3081 } },
      },
      manifest: {
        files: [
          {
            uri: 's3://bucket/train-000.smpl',
            sha256: 'abc123',
            role: 'sample',
            profile: 'sampleshard.v1',
            startKey: '0',
            endKey: '59999',
            entryCount: 60000,
          },
        ],
        partitions: { train: ['train-000.smpl'] },
      },
    };

    const serialized = serializeMetadata(meta);
    const got = deserializeMetadata(serialized);
    expect(got.profile).toBe('sampleshard.v1');
    expect(got.sampleShard?.datasetName).toBe('mnist-train');
    expect(got.sampleShard?.keyEncoding).toBe('decimal-string');
    expect(got.sampleShard?.sampleCount).toBe(60000);
    expect(got.manifest?.files?.[0].uri).toBe('s3://bucket/train-000.smpl');
    expect(got.manifest?.partitions?.train).toEqual(['train-000.smpl']);
  });
});

// ============================================================
// Security limits
// ============================================================

describe('security limits', () => {
  it('entry_count too large', () => {
    // Build a header with entry_count > MAX_ENTRY_COUNT
    // and total_file_size = HEADER_SIZE so the size check passes
    const buf = Buffer.alloc(64, 0);
    buf.write('SHRD', 0, 'ascii');
    buf[4] = 0x02;           // version
    buf[5] = 1;              // role
    buf.writeUInt16LE(0x00A2, 6); // flags
    buf[8] = 64;             // alignment
    buf[9] = 0;              // compression
    buf.writeUInt16LE(48, 10); // index_entry_size
    buf.writeUInt32LE(MAX_ENTRY_COUNT + 1, 12); // entry_count too large
    buf.writeBigUInt64LE(64n, 16); // string_table_offset
    buf.writeBigUInt64LE(64n, 24); // data_section_offset
    buf.writeBigUInt64LE(0n, 32);  // schema_offset
    buf.writeBigUInt64LE(BigInt(buf.length), 40); // total_file_size
    expect(() => new ShardReader(buf)).toThrow(/MAX_ENTRY_COUNT/);
  });

  it('total_file_size mismatch throws', () => {
    // Make a valid shard then corrupt total_file_size
    const buf = makeShardBuffer([['x', Buffer.from('hello')]]);
    const corrupt = Buffer.from(buf);
    corrupt.writeBigUInt64LE(99999n, 40); // wrong total_file_size
    expect(() => new ShardReader(corrupt)).toThrow(/total_file_size/);
  });

  it('data_offset before data section throws', () => {
    const buf = makeShardBuffer([['x', Buffer.from('hello')]]);
    const corrupt = Buffer.from(buf);
    // Entry 0 data_offset is at HEADER_SIZE + 16 = 80
    corrupt.writeBigUInt64LE(0n, 64 + 16); // set to 0 (before header)
    // Keep total_file_size consistent
    corrupt.writeBigUInt64LE(BigInt(corrupt.length), 40);
    expect(() => new ShardReader(corrupt)).toThrow();
  });

  it('data extends past end of file throws', () => {
    const buf = makeShardBuffer([['x', Buffer.from('hello')]]);
    const corrupt = Buffer.from(buf);
    // Set data_offset to huge value
    corrupt.writeBigUInt64LE(0xFFFFFFFFn, 64 + 16);
    corrupt.writeBigUInt64LE(BigInt(corrupt.length), 40);
    expect(() => new ShardReader(corrupt)).toThrow();
  });
});
