/**
 * Shard binary container format — TypeScript implementation.
 *
 * Binary-compatible with the Go reference implementation at cowrie/ucodec/shard_format.go.
 *
 * Layout:
 *   [Header 64B] [Index N×48B] [String Table] [padding to alignment] [Data entries]
 */

import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import xxhashInit from 'xxhash-wasm';
import { init as initZstdWasm, compress as zstdWasmCompress, decompress as zstdWasmDecompress } from '@bokuweb/zstd-wasm';

// lz4js is CJS-only; load via createRequire so we work in ESM context.
const _require = createRequire(import.meta.url);
const _lz4js = _require('lz4js') as {
  compressBound(n: number): number;
  compressBlock(input: Buffer, output: Buffer, startIdx: number, endIdx: number, hashTable: Int32Array): number;
  decompressBlock(input: Buffer, output: Buffer, startIdx: number, endIdx: number, outputOffset: number): number;
};

// ============================================================
// xxHash64 init
// ============================================================

let _xxh64fn: ((input: string | Uint8Array) => bigint) | null = null;

/** Initialize the xxhash-wasm module. Must be called before using computeXxhash64. */
export async function initXxhash(): Promise<void> {
  if (_xxh64fn !== null) return;
  const h = await xxhashInit();
  _xxh64fn = (input: string | Uint8Array): bigint => {
    // xxhash-wasm h64 returns a BigInt
    return h.h64Raw(typeof input === 'string' ? Buffer.from(input, 'utf8') : input);
  };
}

/** Get the raw xxhash-wasm module with h64Raw available. Exposed for internal use. */
let _xxhashModule: Awaited<ReturnType<typeof xxhashInit>> | null = null;

async function getXxhashModule() {
  if (_xxhashModule === null) {
    _xxhashModule = await xxhashInit();
    _xxh64fn = (input: string | Uint8Array): bigint =>
      _xxhashModule!.h64Raw(typeof input === 'string' ? Buffer.from(input, 'utf8') : input);
  }
  return _xxhashModule;
}

// ============================================================
// Zstd / LZ4 init
// ============================================================

let _zstdReady = false;
const ZSTD_FRAME_MAGIC = 0xFD2FB528;

/**
 * Initialize compression libraries (zstd WASM).
 * Must be called before writing or reading compressed entries using COMPRESS_ZSTD.
 * LZ4 does not require any async initialization.
 */
export async function initCompression(): Promise<void> {
  if (_zstdReady) return;
  await initZstdWasm();
  _zstdReady = true;
}

/** Compress data with zstd (level 3, matching Go's default). */
function zstdCompress(data: Uint8Array): Uint8Array {
  if (!_zstdReady) {
    throw new Error('zstd not initialized — call await initCompression() first');
  }
  return zstdWasmCompress(data, 3);
}

/** Decompress zstd data. */
function zstdDecompress(
  data: Uint8Array,
  expectedSize?: number,
  maxOutputSize: number = MAX_DECOMPRESS_SIZE,
): Uint8Array {
  if (!_zstdReady) {
    throw new Error('zstd not initialized — call await initCompression() first');
  }

  if (expectedSize !== undefined && expectedSize > maxOutputSize) {
    throw new Error(`decompressed size ${expectedSize} exceeds limit ${maxOutputSize}`);
  }

  const declaredSize = zstdFrameContentSize(data);
  if (declaredSize !== null) {
    if (declaredSize > maxOutputSize) {
      throw new Error(`zstd frame content size ${declaredSize} exceeds limit ${maxOutputSize}`);
    }
    if (expectedSize !== undefined && declaredSize !== expectedSize) {
      throw new Error(
        `zstd frame content size ${declaredSize} does not match expected size ${expectedSize}`,
      );
    }
  }

  const output = zstdWasmDecompress(data);
  if (output.length > maxOutputSize) {
    throw new Error(`zstd output size ${output.length} exceeds limit ${maxOutputSize}`);
  }
  if (expectedSize !== undefined && output.length !== expectedSize) {
    throw new Error(`zstd output size ${output.length} does not match expected size ${expectedSize}`);
  }
  return output;
}

function zstdFrameContentSize(data: Uint8Array): number | null {
  const buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  if (buf.length < 5) {
    throw new Error('truncated zstd frame');
  }
  if (buf.readUInt32LE(0) !== ZSTD_FRAME_MAGIC) {
    throw new Error('invalid zstd frame magic');
  }

  const descriptor = buf[4];
  const frameContentSizeFlag = descriptor >> 6;
  const singleSegment = (descriptor & 0x20) !== 0;
  const dictIdFlag = descriptor & 0x03;

  let offset = 5;
  if (!singleSegment) {
    if (buf.length < offset + 1) {
      throw new Error('truncated zstd window descriptor');
    }
    offset += 1;
  }

  switch (dictIdFlag) {
    case 0:
      break;
    case 1:
      offset += 1;
      break;
    case 2:
      offset += 2;
      break;
    case 3:
      offset += 4;
      break;
  }

  let fcsSize = 0;
  switch (frameContentSizeFlag) {
    case 0:
      fcsSize = singleSegment ? 1 : 0;
      break;
    case 1:
      fcsSize = 2;
      break;
    case 2:
      fcsSize = 4;
      break;
    case 3:
      fcsSize = 8;
      break;
  }

  if (fcsSize === 0) {
    return null;
  }
  if (buf.length < offset + fcsSize) {
    throw new Error('truncated zstd frame content size');
  }

  let size: bigint;
  switch (fcsSize) {
    case 1:
      size = BigInt(buf[offset]);
      break;
    case 2:
      size = BigInt(buf.readUInt16LE(offset) + 256);
      break;
    case 4:
      size = BigInt(buf.readUInt32LE(offset));
      break;
    case 8:
      size = readUint64LE(buf, offset);
      break;
    default:
      return null;
  }

  if (size > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`zstd frame content size ${size.toString()} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return Number(size);
}

/**
 * Compress data with LZ4 block format (no frame header).
 * Matches Go's lz4.CompressBlock / Python's lz4.block.compress(store_size=False).
 * Returns empty Buffer if data is incompressible.
 */
function lz4BlockCompress(data: Buffer): Buffer {
  const bound = _lz4js.compressBound(data.length);
  const output = Buffer.alloc(bound);
  const hashTable = new Int32Array(1 << 16);
  const n = _lz4js.compressBlock(data, output, 0, data.length, hashTable);
  if (n === 0) {
    // Data is incompressible
    return Buffer.alloc(0);
  }
  return Buffer.from(output.subarray(0, n));
}

/**
 * Decompress LZ4 block data (no frame header).
 * Matches Go's lz4.UncompressBlock / Python's lz4.block.decompress(uncompressed_size=...).
 * origSize must be the exact expected decompressed length.
 */
function lz4BlockDecompress(data: Buffer, origSize: number): Buffer {
  const output = Buffer.alloc(origSize);
  const n = _lz4js.decompressBlock(data, output, 0, data.length, 0);
  if (n !== origSize) {
    throw new Error(`lz4 size mismatch: got ${n}, expected ${origSize}`);
  }
  return output;
}

// ============================================================
// Constants — Roles
// ============================================================

export const ROLE_UNKNOWN = 0;
export const ROLE_MOSH = 1;
export const ROLE_SAMPLE = 2;
export const ROLE_GEMMPANEL = 3;
export const ROLE_MANIFEST = 4;
export const ROLE_WSHARD = 5;
export const ROLE_UMSH = 6;

// ============================================================
// Constants — Alignment
// ============================================================

export const ALIGN_NONE = 0;
export const ALIGN_16 = 16;
export const ALIGN_32 = 32;
export const ALIGN_64 = 64;

// ============================================================
// Constants — Compression
// ============================================================

export const COMPRESS_NONE = 0;
export const COMPRESS_ZSTD = 1;
export const COMPRESS_LZ4 = 2;

// ============================================================
// Constants — Header flags
// ============================================================

export const FLAG_LITTLE_ENDIAN = 0x0002;
export const FLAG_HAS_SCHEMA = 0x0010;
export const FLAG_HAS_CHECKSUMS = 0x0020;
export const FLAG_STREAMING = 0x0040;
export const FLAG_HAS_CONTENT_TYPES = 0x0080;

export const DEFAULT_FLAGS = FLAG_LITTLE_ENDIAN | FLAG_HAS_CHECKSUMS | FLAG_HAS_CONTENT_TYPES; // 0x00A2 = 162

// ============================================================
// Security Limits
// ============================================================

export const MAX_INDEX_SIZE = 1 << 30;                   // 1 GB
export const MAX_STRING_TABLE_SIZE = 100 * 1024 * 1024;  // 100 MB
export const MAX_ENTRY_COUNT = 10_000_000;

// ============================================================
// Constants — Entry flags
// ============================================================

export const ENTRY_FLAG_COMPRESSED = 0x0001;
export const ENTRY_FLAG_ZSTD = 0x0002;
export const ENTRY_FLAG_LZ4 = 0x0004;
export const ENTRY_FLAG_CHUNKED = 0x0008;

// ============================================================
// Constants — Content types
// ============================================================

export const CONTENT_TYPE_UNKNOWN = 0;
export const CONTENT_TYPE_TENSOR = 1;
export const CONTENT_TYPE_JSON = 2;
export const CONTENT_TYPE_COWRIE = 3;
export const CONTENT_TYPE_GLYPH = 4;
export const CONTENT_TYPE_TEXT = 5;
export const CONTENT_TYPE_IMAGE = 6;
export const CONTENT_TYPE_AUDIO = 7;
export const CONTENT_TYPE_VIDEO = 8;
export const CONTENT_TYPE_PROTO = 9;
export const CONTENT_TYPE_BLOB = 10;
export const CONTENT_TYPE_QMLN = 11;
export const CONTENT_TYPE_TENSOR_V3 = 12;
export const CONTENT_TYPE_ANCHOR_SHARED = 13;
export const CONTENT_TYPE_DELTA_EXPERT = 14;
export const CONTENT_TYPE_CODEBOOK_SHARED = 15;
export const CONTENT_TYPE_EXPERT_INDICES = 16;

// ============================================================
// Compression constants (match Go/Python reference)
// ============================================================

const MIN_COMPRESS_SIZE = 256;
const COMPRESSION_SAVINGS_NUM = 9;
const COMPRESSION_SAVINGS_DEN = 10;
const MAX_DECOMPRESS_SIZE = 1 << 30; // 1 GB DoS guard

// ============================================================
// ShardMetadata
// ============================================================

export interface EntryMeta {
  contentType?: string;
  tags?: string[];
  description?: string;
  extra?: Record<string, unknown>;
  codec?: string;
  codecVersion?: string;
  schemaFingerprint?: string;
  semanticType?: string;
  canonicalHash?: string;
  baseHash?: string;
  rowCount?: number;
  shape?: number[];
  stats?: Record<string, unknown>;
}

export interface SampleShardProfile {
  datasetName?: string;
  sampleIdType?: string;
  keyEncoding?: string;
  sampleCount?: number;
  datasetSchema?: Record<string, unknown>;
  splits?: Record<string, unknown>;
  labelMap?: Record<string, unknown>;
  featureStats?: Record<string, unknown>;
}

export interface ManifestFileRef {
  uri?: string;
  sha256?: string;
  role?: string;
  profile?: string;
  startKey?: string;
  endKey?: string;
  entryCount?: number;
}

export interface ManifestProfile {
  files?: ManifestFileRef[];
  partitions?: Record<string, unknown>;
}

export interface ShardMetadata {
  schemaVersion: string;
  schemaUri?: string;
  createdAt?: string;
  sourceUri?: string;
  producer?: string;
  description?: string;
  tags?: string[];
  extra?: Record<string, unknown>;
  profile?: string;
  sampleShard?: SampleShardProfile;
  manifest?: ManifestProfile;
  entryMetadata?: Record<string, EntryMeta>;
  tagDictionary?: string[];
  schemaId?: string;
}

/** Serialize ShardMetadata to a compact JSON Buffer. */
export function serializeMetadata(meta: ShardMetadata): Buffer {
  const obj: Record<string, unknown> = {
    schema_version: meta.schemaVersion,
  };
  if (meta.schemaUri) obj['schema_uri'] = meta.schemaUri;
  if (meta.schemaId) obj['schema_id'] = meta.schemaId;
  if (meta.createdAt) obj['created_at'] = meta.createdAt;
  if (meta.sourceUri) obj['source_uri'] = meta.sourceUri;
  if (meta.producer) obj['producer'] = meta.producer;
  if (meta.description) obj['description'] = meta.description;
  if (meta.tags && meta.tags.length > 0) obj['tags'] = meta.tags;
  if (meta.tagDictionary && meta.tagDictionary.length > 0) obj['tag_dictionary'] = meta.tagDictionary;
  if (meta.extra && Object.keys(meta.extra).length > 0) obj['extra'] = meta.extra;
  if (meta.profile) obj['profile'] = meta.profile;
  if (meta.sampleShard && Object.keys(meta.sampleShard).length > 0) {
    const sampleShard: Record<string, unknown> = {};
    if (meta.sampleShard.datasetName) sampleShard['dataset_name'] = meta.sampleShard.datasetName;
    if (meta.sampleShard.sampleIdType) sampleShard['sample_id_type'] = meta.sampleShard.sampleIdType;
    if (meta.sampleShard.keyEncoding) sampleShard['key_encoding'] = meta.sampleShard.keyEncoding;
    if (meta.sampleShard.sampleCount !== undefined) sampleShard['sample_count'] = meta.sampleShard.sampleCount;
    if (meta.sampleShard.datasetSchema && Object.keys(meta.sampleShard.datasetSchema).length > 0) {
      sampleShard['dataset_schema'] = meta.sampleShard.datasetSchema;
    }
    if (meta.sampleShard.splits && Object.keys(meta.sampleShard.splits).length > 0) {
      sampleShard['splits'] = meta.sampleShard.splits;
    }
    if (meta.sampleShard.labelMap && Object.keys(meta.sampleShard.labelMap).length > 0) {
      sampleShard['label_map'] = meta.sampleShard.labelMap;
    }
    if (meta.sampleShard.featureStats && Object.keys(meta.sampleShard.featureStats).length > 0) {
      sampleShard['feature_stats'] = meta.sampleShard.featureStats;
    }
    obj['sample_shard'] = sampleShard;
  }
  if (meta.manifest && (meta.manifest.files?.length || (meta.manifest.partitions && Object.keys(meta.manifest.partitions).length > 0))) {
    const manifest: Record<string, unknown> = {};
    if (meta.manifest.files && meta.manifest.files.length > 0) {
      manifest['files'] = meta.manifest.files.map((file) => {
        const out: Record<string, unknown> = {};
        if (file.uri) out['uri'] = file.uri;
        if (file.sha256) out['sha256'] = file.sha256;
        if (file.role) out['role'] = file.role;
        if (file.profile) out['profile'] = file.profile;
        if (file.startKey) out['start_key'] = file.startKey;
        if (file.endKey) out['end_key'] = file.endKey;
        if (file.entryCount !== undefined) out['entry_count'] = file.entryCount;
        return out;
      });
    }
    if (meta.manifest.partitions && Object.keys(meta.manifest.partitions).length > 0) {
      manifest['partitions'] = meta.manifest.partitions;
    }
    obj['manifest'] = manifest;
  }
  if (meta.entryMetadata && Object.keys(meta.entryMetadata).length > 0) {
    const em: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(meta.entryMetadata)) {
      const ev: Record<string, unknown> = {};
      if (v.contentType) ev['content_type'] = v.contentType;
      if (v.tags && v.tags.length > 0) ev['tags'] = v.tags;
      if (v.description) ev['description'] = v.description;
      if (v.extra && Object.keys(v.extra).length > 0) ev['extra'] = v.extra;
      if (v.codec) ev['codec'] = v.codec;
      if (v.codecVersion) ev['codec_version'] = v.codecVersion;
      if (v.schemaFingerprint) ev['schema_fingerprint'] = v.schemaFingerprint;
      if (v.semanticType) ev['semantic_type'] = v.semanticType;
      if (v.canonicalHash) ev['canonical_hash'] = v.canonicalHash;
      if (v.baseHash) ev['base_hash'] = v.baseHash;
      if (v.rowCount !== undefined) ev['row_count'] = v.rowCount;
      if (v.shape && v.shape.length > 0) ev['shape'] = v.shape;
      if (v.stats && Object.keys(v.stats).length > 0) ev['stats'] = v.stats;
      em[k] = ev;
    }
    obj['entry_metadata'] = em;
  }
  return Buffer.from(JSON.stringify(obj), 'utf8');
}

/** Deserialize ShardMetadata from a JSON Buffer. */
export function deserializeMetadata(data: Buffer): ShardMetadata {
  const obj = JSON.parse(data.toString('utf8')) as Record<string, unknown>;
  const em: Record<string, EntryMeta> = {};
  const rawEm = obj['entry_metadata'] as Record<string, Record<string, unknown>> | undefined;
  if (rawEm) {
    for (const [k, v] of Object.entries(rawEm)) {
      em[k] = {
        contentType: v['content_type'] as string | undefined,
        tags: v['tags'] as string[] | undefined,
        description: v['description'] as string | undefined,
        extra: v['extra'] as Record<string, unknown> | undefined,
        codec: v['codec'] as string | undefined,
        codecVersion: v['codec_version'] as string | undefined,
        schemaFingerprint: v['schema_fingerprint'] as string | undefined,
        semanticType: v['semantic_type'] as string | undefined,
        canonicalHash: v['canonical_hash'] as string | undefined,
        baseHash: v['base_hash'] as string | undefined,
        rowCount: v['row_count'] as number | undefined,
        shape: v['shape'] as number[] | undefined,
        stats: v['stats'] as Record<string, unknown> | undefined,
      };
    }
  }
  const rawSampleShard = obj['sample_shard'] as Record<string, unknown> | undefined;
  const rawManifest = obj['manifest'] as Record<string, unknown> | undefined;
  return {
    schemaVersion: (obj['schema_version'] as string) ?? 'shard-v2.1',
    schemaUri: obj['schema_uri'] as string | undefined,
    schemaId: obj['schema_id'] as string | undefined,
    createdAt: obj['created_at'] as string | undefined,
    sourceUri: obj['source_uri'] as string | undefined,
    producer: obj['producer'] as string | undefined,
    description: obj['description'] as string | undefined,
    tags: obj['tags'] as string[] | undefined,
    extra: obj['extra'] as Record<string, unknown> | undefined,
    profile: obj['profile'] as string | undefined,
    sampleShard: rawSampleShard
      ? {
          datasetName: rawSampleShard['dataset_name'] as string | undefined,
          sampleIdType: rawSampleShard['sample_id_type'] as string | undefined,
          keyEncoding: rawSampleShard['key_encoding'] as string | undefined,
          sampleCount: rawSampleShard['sample_count'] as number | undefined,
          datasetSchema: rawSampleShard['dataset_schema'] as Record<string, unknown> | undefined,
          splits: rawSampleShard['splits'] as Record<string, unknown> | undefined,
          labelMap: rawSampleShard['label_map'] as Record<string, unknown> | undefined,
          featureStats: rawSampleShard['feature_stats'] as Record<string, unknown> | undefined,
        }
      : undefined,
    manifest: rawManifest
      ? {
          files: Array.isArray(rawManifest['files'])
            ? (rawManifest['files'] as Record<string, unknown>[]).map((file) => ({
                uri: file['uri'] as string | undefined,
                sha256: file['sha256'] as string | undefined,
                role: file['role'] as string | undefined,
                profile: file['profile'] as string | undefined,
                startKey: file['start_key'] as string | undefined,
                endKey: file['end_key'] as string | undefined,
                entryCount: file['entry_count'] as number | undefined,
              }))
            : undefined,
          partitions: rawManifest['partitions'] as Record<string, unknown> | undefined,
        }
      : undefined,
    tagDictionary: obj['tag_dictionary'] as string[] | undefined,
    entryMetadata: Object.keys(em).length > 0 ? em : undefined,
  };
}

// ============================================================
// Path helpers
// ============================================================

/** Split a shard entry name on '/' into path components. */
export function splitPath(name: string): string[] {
  if (!name) return [];
  return name.split('/');
}

/** Join path components with '/'. */
export function joinPath(...parts: string[]): string {
  return parts.join('/');
}

/** Return everything before the last '/', or '' if no '/'. */
export function pathParent(name: string): string {
  const idx = name.lastIndexOf('/');
  return idx < 0 ? '' : name.slice(0, idx);
}

/** Return everything after the last '/', or the whole name if no '/'. */
export function pathBase(name: string): string {
  const idx = name.lastIndexOf('/');
  return idx < 0 ? name : name.slice(idx + 1);
}

// ============================================================
// Binary format sizes
// ============================================================

const HEADER_SIZE = 64;
const INDEX_ENTRY_SIZE = 48;
const SHARD_MAGIC = Buffer.from([0x53, 0x48, 0x52, 0x44]); // 'S','H','R','D'
const SHARD_VERSION = 0x02;

// ============================================================
// CRC32C — Castagnoli polynomial 0x82F63B78
// ============================================================

// Pre-computed CRC32C lookup table using Castagnoli polynomial.
// This is NOT the same as IEEE CRC32 (0xEDB88320).
const CRC32C_TABLE = (() => {
  const table = new Uint32Array(256);
  const POLY = 0x82F63B78; // Castagnoli
  for (let i = 0; i < 256; i++) {
    let crc = i;
    for (let j = 0; j < 8; j++) {
      if (crc & 1) {
        crc = (crc >>> 1) ^ POLY;
      } else {
        crc = crc >>> 1;
      }
    }
    table[i] = crc;
  }
  return table;
})();

/**
 * Compute CRC32C checksum using Castagnoli polynomial (0x82F63B78).
 * This matches Go's crc32.Castagnoli and is NOT the same as IEEE CRC32.
 */
export function computeCrc32c(data: Buffer): number {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    crc = (crc >>> 8) ^ CRC32C_TABLE[(crc ^ data[i]) & 0xFF];
  }
  return (crc ^ 0xFFFFFFFF) >>> 0; // unsigned 32-bit
}

// ============================================================
// xxHash64
// ============================================================

/**
 * Compute xxHash64 of a UTF-8 string, using seed=0.
 * Matches Go's github.com/cespare/xxhash/v2 Sum64String.
 * Requires initXxhash() to have been called first.
 */
export function computeXxhash64(name: string): bigint {
  if (_xxh64fn === null) {
    throw new Error('xxhash not initialized — call await initXxhash() first');
  }
  return _xxh64fn(name);
}

// ============================================================
// Data structures
// ============================================================

export interface ShardHeader {
  magic: Buffer;
  version: number;
  role: number;
  flags: number;
  alignment: number;
  compressionDefault: number;
  indexEntrySize: number;
  entryCount: number;
  stringTableOffset: bigint;
  dataSectionOffset: bigint;
  schemaOffset: bigint;
  totalFileSize: bigint;
}

export interface IndexEntry {
  nameHash: bigint;
  nameOffset: number;
  nameLen: number;
  flags: number;
  dataOffset: bigint;
  diskSize: bigint;
  origSize: bigint;
  checksum: number;
  reserved: number;
  // Resolved fields (not serialized)
  name: string;
  contentType: number;
  compressed: boolean;
}

/** Get 16-bit tag bitmask from upper 16 bits of entry reserved field. */
export function getTagBits(entry: IndexEntry): number {
  return (entry.reserved >>> 16) & 0xFFFF;
}

/** Check if a specific tag bit (0-15) is set. */
export function hasTagBit(entry: IndexEntry, bit: number): boolean {
  if (bit < 0 || bit > 15) return false;
  return (getTagBits(entry) & (1 << bit)) !== 0;
}

/** Set a specific tag bit (0-15) on an entry's reserved field. */
export function setTagBit(entry: IndexEntry, bit: number): void {
  if (bit < 0 || bit > 15) return;
  entry.reserved = (entry.reserved & 0x0000FFFF) | (((getTagBits(entry) | (1 << bit)) & 0xFFFF) << 16);
}

// ============================================================
// Binary parsing helpers
// ============================================================

function readUint64LE(buf: Buffer, offset: number): bigint {
  const lo = BigInt(buf.readUInt32LE(offset));
  const hi = BigInt(buf.readUInt32LE(offset + 4));
  return lo | (hi << 32n);
}

function writeUint64LE(buf: Buffer, offset: number, value: bigint): void {
  const lo = Number(value & 0xFFFFFFFFn);
  const hi = Number((value >> 32n) & 0xFFFFFFFFn);
  buf.writeUInt32LE(lo, offset);
  buf.writeUInt32LE(hi, offset + 4);
}

// ============================================================
// Header parsing
// ============================================================

function parseHeader(buf: Buffer): ShardHeader {
  if (buf.length < HEADER_SIZE) {
    throw new Error(`header too short: ${buf.length} < ${HEADER_SIZE}`);
  }

  const magic = buf.slice(0, 4);
  if (!magic.equals(SHARD_MAGIC)) {
    throw new Error(`invalid magic: ${magic.toString('hex')}`);
  }

  const version = buf[4];
  if (version !== SHARD_VERSION) {
    throw new Error(`unsupported version: ${version}`);
  }

  return {
    magic: Buffer.from(magic),
    version,
    role: buf[5],
    flags: buf.readUInt16LE(6),
    alignment: buf[8],
    compressionDefault: buf[9],
    indexEntrySize: buf.readUInt16LE(10),
    entryCount: buf.readUInt32LE(12),
    stringTableOffset: readUint64LE(buf, 16),
    dataSectionOffset: readUint64LE(buf, 24),
    schemaOffset: readUint64LE(buf, 32),
    totalFileSize: readUint64LE(buf, 40),
  };
}

function serializeHeader(h: ShardHeader): Buffer {
  const buf = Buffer.alloc(HEADER_SIZE, 0);
  SHARD_MAGIC.copy(buf, 0);
  buf[4] = h.version;
  buf[5] = h.role;
  buf.writeUInt16LE(h.flags, 6);
  buf[8] = h.alignment;
  buf[9] = h.compressionDefault;
  buf.writeUInt16LE(h.indexEntrySize, 10);
  buf.writeUInt32LE(h.entryCount, 12);
  writeUint64LE(buf, 16, h.stringTableOffset);
  writeUint64LE(buf, 24, h.dataSectionOffset);
  writeUint64LE(buf, 32, h.schemaOffset);
  writeUint64LE(buf, 40, h.totalFileSize);
  // bytes 48-63 reserved (zeroed)
  return buf;
}

// ============================================================
// Index entry parsing
// ============================================================

function parseIndexEntry(buf: Buffer, offset: number, stringTable: Buffer, stOffset: number): IndexEntry {
  const nameHash = readUint64LE(buf, offset + 0);
  const nameOffset = buf.readUInt32LE(offset + 8);
  const nameLen = buf.readUInt16LE(offset + 12);
  const flags = buf.readUInt16LE(offset + 14);
  const dataOffset = readUint64LE(buf, offset + 16);
  const diskSize = readUint64LE(buf, offset + 24);
  const origSize = readUint64LE(buf, offset + 32);
  const checksum = buf.readUInt32LE(offset + 40);
  const reserved = buf.readUInt32LE(offset + 44);

  // Resolve name from string table (with bounds check matching Go)
  let name: string;
  if (nameOffset >= stringTable.length || nameLen > stringTable.length - nameOffset) {
    name = '';
  } else {
    name = stringTable.slice(nameOffset, nameOffset + nameLen).toString('utf8');
  }

  return {
    nameHash,
    nameOffset,
    nameLen,
    flags,
    dataOffset,
    diskSize,
    origSize,
    checksum,
    reserved,
    name,
    contentType: reserved & 0xFFFF,
    compressed: (flags & ENTRY_FLAG_COMPRESSED) !== 0,
  };
}

function serializeIndexEntry(
  nameHash: bigint,
  nameOffset: number,
  nameLen: number,
  flags: number,
  dataOffset: bigint,
  diskSize: bigint,
  origSize: bigint,
  checksum: number,
  reserved: number,
): Buffer {
  const buf = Buffer.alloc(INDEX_ENTRY_SIZE, 0);
  writeUint64LE(buf, 0, nameHash);
  buf.writeUInt32LE(nameOffset, 8);
  buf.writeUInt16LE(nameLen, 12);
  buf.writeUInt16LE(flags, 14);
  writeUint64LE(buf, 16, dataOffset);
  writeUint64LE(buf, 24, diskSize);
  writeUint64LE(buf, 32, origSize);
  buf.writeUInt32LE(checksum, 40);
  buf.writeUInt32LE(reserved, 44);
  return buf;
}

// ============================================================
// Alignment helper
// ============================================================

function alignUp(value: bigint, alignment: number): bigint {
  if (alignment <= 0) return value;
  const a = BigInt(alignment);
  return (value + a - 1n) & ~(a - 1n);
}

function alignUpNum(value: number, alignment: number): number {
  if (alignment <= 0) return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

// ============================================================
// ShardReader
// ============================================================

export class ShardReader {
  private readonly _buf: Buffer;
  private readonly _header: ShardHeader;
  private readonly _entries: IndexEntry[];
  private readonly _nameToIndex: Map<string, number>;

  constructor(buffer: Buffer) {
    this._buf = buffer;
    this._header = parseHeader(buffer);

    if (this._header.indexEntrySize !== INDEX_ENTRY_SIZE) {
      throw new Error(`invalid index entry size: ${this._header.indexEntrySize}`);
    }

    // Security limits
    const entryCount = this._header.entryCount;
    if (entryCount > MAX_ENTRY_COUNT) {
      throw new Error(`entry_count ${entryCount} exceeds MAX_ENTRY_COUNT ${MAX_ENTRY_COUNT}`);
    }
    const indexSize = entryCount * INDEX_ENTRY_SIZE;
    if (indexSize > MAX_INDEX_SIZE) {
      throw new Error(`index size ${indexSize} exceeds MAX_INDEX_SIZE ${MAX_INDEX_SIZE}`);
    }

    const stStart = Number(this._header.stringTableOffset);
    const stEnd = Number(this._header.dataSectionOffset);
    if (stStart > buffer.length || stEnd > buffer.length) {
      throw new Error(`string table beyond file: offset ${stStart}, data_section ${stEnd}, file_size ${buffer.length}`);
    }
    if (stEnd < stStart) {
      throw new Error(`data_section_offset ${stEnd} < string_table_offset ${stStart}`);
    }
    const stringTableSize = stEnd - stStart;
    if (stringTableSize > MAX_STRING_TABLE_SIZE) {
      throw new Error(`string table size ${stringTableSize} exceeds MAX_STRING_TABLE_SIZE ${MAX_STRING_TABLE_SIZE}`);
    }

    const totalFileSize = Number(this._header.totalFileSize);
    if (totalFileSize !== buffer.length) {
      throw new Error(`total_file_size ${totalFileSize} !== actual buffer length ${buffer.length}`);
    }

    // String table: from stringTableOffset to dataSectionOffset
    const stringTable = buffer.slice(stStart, stEnd);

    this._entries = [];
    this._nameToIndex = new Map();

    for (let i = 0; i < entryCount; i++) {
      const off = HEADER_SIZE + i * INDEX_ENTRY_SIZE;
      const entry = parseIndexEntry(buffer, off, stringTable, stStart);
      this._entries.push(entry);
      this._nameToIndex.set(entry.name, i);
    }

    // Validate entry offsets
    this._validateEntryOffsets();
  }

  private _validateEntryOffsets(): void {
    const dsStart = Number(this._header.dataSectionOffset);
    const fileSize = this._buf.length;
    let prevEnd = 0;

    for (let i = 0; i < this._entries.length; i++) {
      const entry = this._entries[i];
      const offset = Number(entry.dataOffset);
      const size = Number(entry.diskSize);

      if (offset < dsStart) {
        throw new Error(
          `entry ${i} (${entry.name}) data_offset ${offset} < data_section_offset ${dsStart}`
        );
      }
      if (offset + size > fileSize) {
        throw new Error(`entry ${i} (${entry.name}) data extends past end of file`);
      }
      if (i > 0 && offset < prevEnd) {
        throw new Error(
          `entry ${i} (${entry.name}) data_offset ${offset} overlaps with previous entry ending at ${prevEnd}`
        );
      }
      prevEnd = offset + size;
    }
  }

  header(): ShardHeader {
    return this._header;
  }

  entryCount(): number {
    return this._entries.length;
  }

  entryName(i: number): string {
    if (i < 0 || i >= this._entries.length) return '';
    return this._entries[i].name;
  }

  entryNames(): string[] {
    return this._entries.map(e => e.name);
  }

  getEntryInfo(i: number): IndexEntry {
    if (i < 0 || i >= this._entries.length) {
      throw new Error(`entry index out of bounds: ${i}`);
    }
    return this._entries[i];
  }

  lookup(name: string): number {
    const idx = this._nameToIndex.get(name);
    return idx !== undefined ? idx : -1;
  }

  hasEntry(name: string): boolean {
    return this._nameToIndex.has(name);
  }

  readEntry(i: number, verify = true): Buffer {
    if (i < 0 || i >= this._entries.length) {
      throw new Error(`entry index out of bounds: ${i}`);
    }
    const entry = this._entries[i];
    const offset = Number(entry.dataOffset);
    const size = Number(entry.diskSize);
    let data: Buffer = Buffer.from(this._buf.subarray(offset, offset + size));

    if (entry.compressed) {
      const origSize = Number(entry.origSize);
      if (origSize > MAX_DECOMPRESS_SIZE) {
        throw new Error(`decompressed size ${origSize} exceeds limit ${MAX_DECOMPRESS_SIZE}`);
      }
      if (entry.flags & ENTRY_FLAG_ZSTD) {
        data = Buffer.from(zstdDecompress(data, origSize, MAX_DECOMPRESS_SIZE));
      } else if (entry.flags & ENTRY_FLAG_LZ4) {
        data = lz4BlockDecompress(data, origSize);
      } else {
        throw new Error(`unknown compression type for entry ${i} (${entry.name})`);
      }
    }

    // CRC32C checksum is computed on the original uncompressed data.
    if (verify && ((this._header.flags & FLAG_HAS_CHECKSUMS) !== 0 || entry.checksum !== 0)) {
      const computed = computeCrc32c(data);
      if (computed !== entry.checksum) {
        throw new Error(
          `checksum mismatch for entry ${i} (${entry.name}): ` +
          `expected 0x${entry.checksum.toString(16).padStart(8, '0')}, ` +
          `got 0x${computed.toString(16).padStart(8, '0')}`
        );
      }
    }

    return data;
  }

  readEntryByName(name: string, verify = true): Buffer {
    const idx = this.lookup(name);
    if (idx < 0) {
      throw new Error(`entry not found: ${JSON.stringify(name)}`);
    }
    return this.readEntry(idx, verify);
  }

  /**
   * Returns all entry names whose name starts with the given prefix string.
   * Uses simple string prefix matching (not path-aware like the Go PathPrefix).
   * For path-aware matching, pass the prefix with trailing slash.
   */
  listPrefix(prefix: string): string[] {
    if (!prefix) return this.entryNames();
    return this._entries
      .filter(e => e.name.startsWith(prefix))
      .map(e => e.name);
  }

  /**
   * Returns immediate children (one path component deep) under prefix.
   * Returns bare components matching the Go reference implementation.
   *
   * @example
   * // entries: ["layer.0/weight", "layer.0/bias", "layer.1/weight", "embed"]
   * listChildren("layer.0/") // => ["weight", "bias"]
   * listChildren("")          // => ["layer.0", "layer.1", "embed"]
   * listChildren("layer.")    // => ["0", "1"]
   */
  listChildren(prefix: string): string[] {
    const result: string[] = [];
    const seen = new Set<string>();

    for (const entry of this._entries) {
      const name = entry.name;
      if (!name.startsWith(prefix)) continue;

      const remainder = name.slice(prefix.length);
      const slashPos = remainder.indexOf('/');
      let child: string;
      if (slashPos >= 0) {
        // Has sub-components — bare component (no trailing slash)
        child = remainder.slice(0, slashPos);
      } else {
        // Leaf entry — bare remainder
        child = remainder;
      }

      if (child && !seen.has(child)) {
        seen.add(child);
        result.push(child);
      }
    }

    return result;
  }

  /**
   * Returns entry names that have the given tag in their per-entry metadata.
   */
  listWithTag(tag: string): string[] {
    const meta = this.readMetadata();
    if (!meta?.entryMetadata) return [];
    const result: string[] = [];
    for (const [name, em] of Object.entries(meta.entryMetadata)) {
      if (em.tags?.includes(tag)) {
        result.push(name);
      }
    }
    return result;
  }

  /**
   * Returns entry names where the given tag bit (0-15) is set. Pure index scan, no metadata needed.
   */
  listWithTagBit(bit: number): string[] {
    if (bit < 0 || bit > 15) return [];
    return this._entries.filter(e => hasTagBit(e, bit)).map(e => e.name);
  }

  /**
   * Fast tag filtering using tag_dictionary + bitmask. Returns empty if no dictionary.
   */
  listWithTagFast(tag: string): string[] {
    const meta = this.readMetadata();
    if (!meta?.tagDictionary?.length) return [];
    const bit = meta.tagDictionary.indexOf(tag);
    if (bit < 0) return [];
    return this.listWithTagBit(bit);
  }

  /**
   * Read only the first maxBytes of an entry (for header scanning).
   * For compressed entries, decompresses fully then truncates.
   */
  readEntryPrefix(i: number, maxBytes: number): Buffer {
    if (i < 0 || i >= this._entries.length) {
      throw new Error(`entry index out of bounds: ${i}`);
    }
    const entry = this._entries[i];
    if (entry.compressed) {
      const data = this.readEntry(i, false);
      return Buffer.from(data.subarray(0, maxBytes));
    }
    const offset = Number(entry.dataOffset);
    const size = Math.min(Number(entry.diskSize), maxBytes);
    return Buffer.from(this._buf.subarray(offset, offset + size));
  }

  /**
   * Read JSON metadata from schema section. Returns null if no schema.
   */
  readMetadata(): ShardMetadata | null {
    const schemaOffset = Number(this._header.schemaOffset);
    if (schemaOffset === 0) return null;
    const totalFileSize = Number(this._header.totalFileSize);
    const fileLen = this._buf.length;
    if (schemaOffset > fileLen || schemaOffset > totalFileSize) {
      throw new Error(
        `schema_offset ${schemaOffset} out of bounds (file_len=${fileLen}, total_file_size=${totalFileSize})`
      );
    }
    const metaData = this._buf.subarray(schemaOffset, totalFileSize);
    return deserializeMetadata(Buffer.from(metaData));
  }
}

// ============================================================
// ShardWriter
// ============================================================

interface PendingEntry {
  name: string;
  data: Buffer;
  contentType: number;
  compress: boolean;
  compType: number;
}

// ============================================================
// Schema Validation
// ============================================================

export interface EntrySpec {
  /** Glob pattern or exact name to match against entry names. */
  pattern: string;
  /** Expected content type. 0 or undefined = accept any content type. */
  contentType?: number;
  /** When true, a missing entry is not an error. */
  optional?: boolean;
  /** Human-readable description (not used during validation). */
  description?: string;
}

export interface ShardSchema {
  name: string;
  version: number;
  specs: EntrySpec[];
}

export interface ValidationError {
  specPattern: string;
  message: string;
}

/** Create a new empty ShardSchema. */
export function createSchema(name = '', version = 1): ShardSchema {
  return { name, version, specs: [] };
}

/** Append an entry spec to a schema and return the schema (for chaining). */
export function addSpec(
  schema: ShardSchema,
  pattern: string,
  contentType?: number,
  optional?: boolean,
  description?: string,
): ShardSchema {
  schema.specs.push({
    pattern,
    contentType: contentType ?? 0,
    optional: optional ?? false,
    description,
  });
  return schema;
}

/**
 * Match a glob pattern against an entry name.
 *
 * Rules:
 * - ``**`` prefix wildcard: everything before ``**`` must be a prefix of the name.
 * - ``*`` single component wildcard: matches any characters except ``.`` and ``/``.
 * - All other characters match literally.
 */
function matchPattern(pattern: string, name: string): boolean {
  if (pattern.includes('**')) {
    const prefix = pattern.split('**')[0];
    return name.startsWith(prefix);
  }
  // Convert glob to regex: escape regex metacharacters, then replace * with
  // a character class that excludes . and /.
  let regexStr = '^';
  for (const ch of pattern) {
    if (ch === '*') {
      regexStr += '[^./]*';
    } else if (/[.+^${}[\]|()?\\]/.test(ch)) {
      regexStr += '\\' + ch;
    } else {
      regexStr += ch;
    }
  }
  regexStr += '$';
  return new RegExp(regexStr).test(name);
}

/**
 * Validate a shard against a schema.
 *
 * Returns an array of {@link ValidationError} objects.  An empty array means
 * the shard satisfies the schema.
 *
 * Checks performed:
 * 1. Every non-optional spec matches at least one entry.
 * 2. Matched entries have the expected ``contentType`` when the spec specifies
 *    one (non-zero).
 */
export function validateSchema(reader: ShardReader, schema: ShardSchema): ValidationError[] {
  const errors: ValidationError[] = [];
  const names = reader.entryNames();

  for (const spec of schema.specs) {
    const matches = names.filter(n => matchPattern(spec.pattern, n));

    if (matches.length === 0 && !spec.optional) {
      errors.push({
        specPattern: spec.pattern,
        message: `required entry matching '${spec.pattern}' not found`,
      });
      continue;
    }

    const expectedType = spec.contentType ?? 0;
    if (expectedType !== 0) {
      for (const name of matches) {
        const idx = reader.lookup(name);
        const info = reader.getEntryInfo(idx);
        if (info.contentType !== expectedType) {
          errors.push({
            specPattern: spec.pattern,
            message: `entry '${name}' has content_type ${info.contentType}, expected ${expectedType}`,
          });
        }
      }
    }
  }

  return errors;
}

// ============================================================
// ShardWriter
// ============================================================

// ============================================================
// ShardStreamWriter
// ============================================================

/**
 * Zero-copy streaming writer. Data goes directly to disk entry-by-entry.
 *
 * The header+index region is written at the END via a seek-back to file
 * position 0, so no temporary in-RAM buffer is required. Callers must supply
 * ``maxEntries`` upfront so the front-matter region can be pre-reserved.
 *
 * Usage:
 *   const sw = new ShardStreamWriter('/tmp/out.shard', ROLE_MOSH, 1000);
 *   sw.setAlignment(ALIGN_64);  // optional, before beginData()
 *   sw.beginData();
 *   sw.writeEntry('weights', tensorBytes);
 *   sw.writeEntry('config',  jsonBytes);
 *   sw.finalize();
 */
export class ShardStreamWriter {
  private fd: number;
  private readonly role: number;
  private alignment: number;
  private compression: number;
  private readonly maxEntries: number;
  private entries: Array<{
    name: string;
    dataOffset: number;
    diskSize: number;
    origSize: number;
    checksum: number;
    flags: number;
    contentType: number;
  }> = [];
  private started = false;
  private finalized = false;
  private reservedSpace = 0;
  private currentOffset = 0;

  constructor(path: string, role: number = ROLE_MOSH, maxEntries: number = 10000) {
    this.fd = fs.openSync(path, 'w');
    this.role = role;
    this.alignment = ALIGN_64;
    this.compression = COMPRESS_NONE;
    this.maxEntries = maxEntries;
  }

  /** Set data alignment. Must be called before beginData(). */
  setAlignment(align: number): void {
    if (this.started) throw new Error('cannot set alignment after begin_data()');
    if (align !== ALIGN_NONE && align !== ALIGN_16 && align !== ALIGN_32 && align !== ALIGN_64) {
      throw new Error(`invalid alignment: ${align}`);
    }
    this.alignment = align;
  }

  /** Set default compression used by writeEntryCompressed(). */
  setCompression(comp: number): void {
    this.compression = comp;
  }

  /**
   * Calculate the reserved header+index space and seek past it.
   * After this call write operations may proceed.
   */
  beginData(): void {
    if (this.started) throw new Error('begin_data() already called');
    const n = this.maxEntries;
    // Reserve: header + index array + ~64 bytes per name for string table
    let reserved = HEADER_SIZE + (n * INDEX_ENTRY_SIZE) + (n * 64);
    if (this.alignment > 0) {
      reserved = alignUpNum(reserved, this.alignment);
    }
    this.reservedSpace = reserved;
    this.currentOffset = reserved;
    // Write zeros to pre-allocate the reserved region on disk
    const zeros = Buffer.alloc(reserved);
    fs.writeSync(this.fd, zeros, 0, reserved, 0);
    this.started = true;
  }

  /** Write data at current position, recording metadata for finalize(). */
  writeEntry(name: string, data: Buffer): void {
    this.writeEntryTyped(name, data, CONTENT_TYPE_UNKNOWN);
  }

  /** Write data with an explicit content type. */
  writeEntryTyped(name: string, data: Buffer, contentType: number): void {
    if (!this.started) throw new Error('call begin_data() first');
    if (this.finalized) throw new Error('already finalized');

    // Align the write position
    if (this.alignment > 0) {
      const aligned = alignUpNum(this.currentOffset, this.alignment);
      if (aligned > this.currentOffset) {
        const pad = Buffer.alloc(aligned - this.currentOffset);
        fs.writeSync(this.fd, pad, 0, pad.length, this.currentOffset);
        this.currentOffset = aligned;
      }
    }

    const checksum = computeCrc32c(data);
    fs.writeSync(this.fd, data, 0, data.length, this.currentOffset);

    this.entries.push({
      name,
      dataOffset: this.currentOffset,
      diskSize: data.length,
      origSize: data.length,
      checksum,
      flags: 0,
      contentType,
    });
    this.currentOffset += data.length;
  }

  /**
   * Write data with compression applied (if set via setCompression()).
   * Falls back to uncompressed when data is too small or incompressible.
   */
  writeEntryCompressed(name: string, data: Buffer): void {
    if (!this.started) throw new Error('call begin_data() first');
    if (this.finalized) throw new Error('already finalized');

    // Align the write position
    if (this.alignment > 0) {
      const aligned = alignUpNum(this.currentOffset, this.alignment);
      if (aligned > this.currentOffset) {
        const pad = Buffer.alloc(aligned - this.currentOffset);
        fs.writeSync(this.fd, pad, 0, pad.length, this.currentOffset);
        this.currentOffset = aligned;
      }
    }

    const checksum = computeCrc32c(data);
    let diskData = data;
    let entryFlags = 0;

    if (this.compression !== COMPRESS_NONE && data.length >= MIN_COMPRESS_SIZE) {
      const threshold = Math.floor(data.length * COMPRESSION_SAVINGS_NUM / COMPRESSION_SAVINGS_DEN);

      if (this.compression === COMPRESS_ZSTD) {
        try {
          const compressed = Buffer.from(zstdCompress(data));
          if (compressed.length < threshold) {
            diskData = compressed;
            entryFlags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
          }
        } catch (_) {
          // compression unavailable — store raw
        }
      } else if (this.compression === COMPRESS_LZ4) {
        const compressed = lz4BlockCompress(data);
        if (compressed.length > 0 && compressed.length < threshold) {
          diskData = compressed;
          entryFlags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4;
        }
      }
    }

    fs.writeSync(this.fd, diskData, 0, diskData.length, this.currentOffset);

    this.entries.push({
      name,
      dataOffset: this.currentOffset,
      diskSize: diskData.length,
      origSize: data.length,
      checksum,
      flags: entryFlags,
      contentType: CONTENT_TYPE_UNKNOWN,
    });
    this.currentOffset += diskData.length;
  }

  /**
   * Seek to file position 0 and write the complete header, index entries,
   * string table, and padding. Closes the file descriptor.
   */
  finalize(): void {
    if (this.finalized) throw new Error('already finalized');
    this.finalized = true;

    if (_xxh64fn === null) {
      fs.closeSync(this.fd);
      throw new Error('xxhash not initialized — call await initXxhash() first');
    }

    const n = this.entries.length;

    // Build string table
    const parts: Buffer[] = [];
    const nameOffsets: number[] = [];
    let stLen = 0;
    for (const entry of this.entries) {
      nameOffsets.push(stLen);
      const nameBytes = Buffer.from(entry.name, 'utf-8');
      parts.push(nameBytes);
      parts.push(Buffer.from([0]));
      stLen += nameBytes.length + 1;
    }
    const stringTable = parts.length > 0 ? Buffer.concat(parts) : Buffer.alloc(0);

    // Verify the front-matter fits in the reserved region
    const actualFront = HEADER_SIZE + (n * INDEX_ENTRY_SIZE) + stLen;
    const actualFrontAligned = this.alignment > 0
      ? alignUpNum(actualFront, this.alignment)
      : actualFront;
    if (actualFrontAligned > this.reservedSpace) {
      fs.closeSync(this.fd);
      throw new Error(
        `string table overflow: need ${actualFrontAligned} but reserved ${this.reservedSpace} ` +
        `(increase maxEntries or use shorter entry names)`,
      );
    }

    const stringTableOffset = HEADER_SIZE + n * INDEX_ENTRY_SIZE;
    const dataSectionOffset = this.reservedSpace;
    const totalFileSize = this.currentOffset;

    // Build and write header (seek to 0)
    const headerBuf = Buffer.alloc(HEADER_SIZE, 0);
    headerBuf[0] = 0x53; headerBuf[1] = 0x48; headerBuf[2] = 0x52; headerBuf[3] = 0x44;
    headerBuf[4] = SHARD_VERSION;
    headerBuf[5] = this.role;
    headerBuf.writeUInt16LE(DEFAULT_FLAGS | FLAG_STREAMING, 6);
    headerBuf[8] = this.alignment;
    headerBuf[9] = this.compression;
    headerBuf.writeUInt16LE(INDEX_ENTRY_SIZE, 10);
    headerBuf.writeUInt32LE(n, 12);
    writeUint64LE(headerBuf, 16, BigInt(stringTableOffset));
    writeUint64LE(headerBuf, 24, BigInt(dataSectionOffset));
    writeUint64LE(headerBuf, 32, 0n); // schema_offset
    writeUint64LE(headerBuf, 40, BigInt(totalFileSize));
    fs.writeSync(this.fd, headerBuf, 0, HEADER_SIZE, 0);

    // Write index entries
    let indexOffset = HEADER_SIZE;
    for (let i = 0; i < n; i++) {
      const e = this.entries[i];
      const nameBytes = Buffer.from(e.name, 'utf-8');
      const nameHash = _xxh64fn!(e.name);
      const entryBuf = Buffer.alloc(INDEX_ENTRY_SIZE, 0);
      writeUint64LE(entryBuf, 0, nameHash);
      entryBuf.writeUInt32LE(nameOffsets[i], 8);
      entryBuf.writeUInt16LE(nameBytes.length, 12);
      entryBuf.writeUInt16LE(e.flags, 14);
      writeUint64LE(entryBuf, 16, BigInt(e.dataOffset));
      writeUint64LE(entryBuf, 24, BigInt(e.diskSize));
      writeUint64LE(entryBuf, 32, BigInt(e.origSize));
      entryBuf.writeUInt32LE(e.checksum, 40);
      entryBuf.writeUInt32LE(e.contentType & 0xFFFF, 44); // reserved / content_type
      fs.writeSync(this.fd, entryBuf, 0, INDEX_ENTRY_SIZE, indexOffset);
      indexOffset += INDEX_ENTRY_SIZE;
    }

    // Write string table
    if (stringTable.length > 0) {
      fs.writeSync(this.fd, stringTable, 0, stringTable.length, stringTableOffset);
    }

    // Pad from end of string table to data section boundary
    const padStart = stringTableOffset + stLen;
    const padSize = dataSectionOffset - padStart;
    if (padSize > 0) {
      const pad = Buffer.alloc(padSize);
      fs.writeSync(this.fd, pad, 0, padSize, padStart);
    }

    fs.closeSync(this.fd);
  }

  /** Abort: close the file descriptor without finalizing (leaves an incomplete file). */
  close(): void {
    if (!this.finalized) {
      try { fs.closeSync(this.fd); } catch (_) { /* ignore double-close */ }
    }
  }
}

export class ShardWriter {
  private readonly _role: number;
  private _alignment: number;
  private _compression: number;
  private readonly _entries: PendingEntry[];
  private _metadata: ShardMetadata | null = null;

  constructor(role = ROLE_MOSH) {
    this._role = role;
    this._alignment = ALIGN_64;
    this._compression = COMPRESS_NONE;
    this._entries = [];
  }

  /** Attach metadata to be serialized into the schema section in toBuffer(). */
  setMetadata(meta: ShardMetadata): void {
    this._metadata = meta;
  }

  setAlignment(align: number): void {
    if (align !== ALIGN_NONE && align !== ALIGN_16 && align !== ALIGN_32 && align !== ALIGN_64) {
      throw new Error(`invalid alignment: ${align}`);
    }
    this._alignment = align;
  }

  /** Set the default compression algorithm for writeEntryCompressed(). */
  setCompression(comp: number): void {
    if (comp !== COMPRESS_NONE && comp !== COMPRESS_ZSTD && comp !== COMPRESS_LZ4) {
      throw new Error(`invalid compression: ${comp}`);
    }
    this._compression = comp;
  }

  writeEntry(name: string, data: Buffer): void {
    this._entries.push({ name, data, contentType: CONTENT_TYPE_UNKNOWN, compress: false, compType: COMPRESS_NONE });
  }

  writeEntryTyped(name: string, data: Buffer, contentType: number): void {
    this._entries.push({ name, data, contentType, compress: false, compType: COMPRESS_NONE });
  }

  /** Write an entry using the default compression set by setCompression(). */
  writeEntryCompressed(name: string, data: Buffer): void {
    this._entries.push({
      name,
      data,
      contentType: CONTENT_TYPE_UNKNOWN,
      compress: true,
      compType: this._compression,
    });
  }

  /** Write an entry with explicit compression control. */
  writeEntryWithOptions(name: string, data: Buffer, compress: boolean, compType: number): void {
    this._entries.push({ name, data, contentType: CONTENT_TYPE_UNKNOWN, compress, compType });
  }

  /**
   * Finalize and return the complete shard as a Buffer.
   * Requires initXxhash() to have been called.
   * If any entries use COMPRESS_ZSTD, initCompression() must also have been called.
   * LZ4 compression (COMPRESS_LZ4) requires no async initialization.
   */
  toBuffer(): Buffer {
    if (_xxh64fn === null) {
      throw new Error('xxhash not initialized — call await initXxhash() first');
    }

    const n = this._entries.length;
    const align = this._alignment;

    // Build string table (null-terminated names)
    const stringTableParts: Buffer[] = [];
    const nameOffsets: number[] = [];
    let stLen = 0;
    for (const entry of this._entries) {
      nameOffsets.push(stLen);
      const nameBytes = Buffer.from(entry.name, 'utf8');
      stringTableParts.push(nameBytes);
      stringTableParts.push(Buffer.from([0])); // null terminator
      stLen += nameBytes.length + 1;
    }
    const stringTable = Buffer.concat(stringTableParts);

    // Compute section offsets
    const indexSize = n * INDEX_ENTRY_SIZE;
    const stringTableOffset = HEADER_SIZE + indexSize;

    let dataSectionOffset = stringTableOffset + stLen;
    if (align > 0) {
      dataSectionOffset = alignUpNum(dataSectionOffset, align);
    }

    // Precompute disk data for each entry (applying compression where requested).
    // Checksum is always computed on the ORIGINAL uncompressed data.
    const diskDatas: Buffer[] = [];
    const entryFlagsList: number[] = [];

    for (const entry of this._entries) {
      const rawData = entry.data;
      let diskData = rawData;
      let entryFlags = 0;

      if (entry.compress && rawData.length >= MIN_COMPRESS_SIZE) {
        const threshold = Math.floor(rawData.length * COMPRESSION_SAVINGS_NUM / COMPRESSION_SAVINGS_DEN);

        if (entry.compType === COMPRESS_ZSTD) {
          const compressed = Buffer.from(zstdCompress(rawData));
          if (compressed.length < threshold) {
            diskData = compressed;
            entryFlags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
          }
        } else if (entry.compType === COMPRESS_LZ4) {
          const compressed = lz4BlockCompress(rawData);
          // lz4BlockCompress returns empty Buffer if data is incompressible
          if (compressed.length > 0 && compressed.length < threshold) {
            diskData = compressed;
            entryFlags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4;
          }
        }
      }

      diskDatas.push(diskData);
      entryFlagsList.push(entryFlags);
    }

    // Layout data entries — compute offsets and checksums
    interface BuiltEntry {
      nameHash: bigint;
      nameOffset: number;
      nameLen: number;
      flags: number;
      dataOffset: bigint;
      diskSize: bigint;
      origSize: bigint;
      checksum: number;
      reserved: number;
    }

    const builtEntries: BuiltEntry[] = [];
    let currentOffset = dataSectionOffset;

    for (let i = 0; i < n; i++) {
      const entry = this._entries[i];
      if (align > 0) {
        currentOffset = alignUpNum(currentOffset, align);
      }

      const rawData = entry.data;
      const diskData = diskDatas[i];
      const checksum = computeCrc32c(rawData); // always checksum the uncompressed data
      const nameHash = _xxh64fn!(entry.name);
      const nameBytes = Buffer.from(entry.name, 'utf8');

      builtEntries.push({
        nameHash,
        nameOffset: nameOffsets[i],
        nameLen: nameBytes.length,
        flags: entryFlagsList[i],
        dataOffset: BigInt(currentOffset),
        diskSize: BigInt(diskData.length),
        origSize: BigInt(rawData.length),
        checksum,
        reserved: entry.contentType & 0xFFFF,
      });

      currentOffset += diskData.length;
    }

    // If metadata is set, serialize it
    let metaBytes: Buffer | null = null;
    let schemaOffset = 0n;
    let hdrFlags = DEFAULT_FLAGS;

    if (this._metadata !== null) {
      metaBytes = serializeMetadata(this._metadata);
      schemaOffset = BigInt(currentOffset);
      hdrFlags = DEFAULT_FLAGS | FLAG_HAS_SCHEMA;
    }

    const totalFileSize = metaBytes !== null
      ? currentOffset + metaBytes.length
      : currentOffset;

    // Build header
    const header: ShardHeader = {
      magic: Buffer.from(SHARD_MAGIC),
      version: SHARD_VERSION,
      role: this._role,
      flags: hdrFlags,
      alignment: align,
      compressionDefault: COMPRESS_NONE,
      indexEntrySize: INDEX_ENTRY_SIZE,
      entryCount: n,
      stringTableOffset: BigInt(stringTableOffset),
      dataSectionOffset: BigInt(dataSectionOffset),
      schemaOffset,
      totalFileSize: BigInt(totalFileSize),
    };

    // Assemble the output buffer
    const parts: Buffer[] = [];

    // 1. Header
    parts.push(serializeHeader(header));

    // 2. Index entries
    for (let i = 0; i < n; i++) {
      const be = builtEntries[i];
      parts.push(serializeIndexEntry(
        be.nameHash,
        be.nameOffset,
        be.nameLen,
        be.flags,
        be.dataOffset,
        be.diskSize,
        be.origSize,
        be.checksum,
        be.reserved,
      ));
    }

    // 3. String table
    parts.push(stringTable);

    // 4. Padding to data section
    const afterST = stringTableOffset + stLen;
    if (afterST < dataSectionOffset) {
      parts.push(Buffer.alloc(dataSectionOffset - afterST, 0));
    }

    // 5. Data entries with alignment padding (uses compressed disk data where applicable)
    let pos = dataSectionOffset;
    for (let i = 0; i < n; i++) {
      const target = Number(builtEntries[i].dataOffset);
      if (pos < target) {
        parts.push(Buffer.alloc(target - pos, 0));
        pos = target;
      }
      const diskData = diskDatas[i];
      parts.push(diskData);
      pos += diskData.length;
    }

    // 6. Metadata (schema section) if present
    if (metaBytes !== null) {
      parts.push(metaBytes);
    }

    return Buffer.concat(parts);
  }
}
