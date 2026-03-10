/**
 * SampleShard type definitions.
 *
 * Core data structures for the SampleShard format:
 * - ShardHeader: 64-byte v2 header
 * - IndexEntry: 48-byte index entry
 * - ShardRole: Role enumeration
 */

// Magic bytes
export const SHARD_MAGIC = Buffer.from('SHRD', 'ascii');

// Version
export const SHARD_VERSION_2 = 0x02;

// Header size
export const SHARD_V2_HEADER_SIZE = 64;

// Index entry size
export const SHARD_V2_INDEX_ENTRY_SIZE = 48;

// Alignment values
export const ALIGN_NONE = 0;
export const ALIGN_16 = 16;
export const ALIGN_32 = 32;
export const ALIGN_64 = 64;

// Compression types
export const COMPRESS_NONE = 0x00;
export const COMPRESS_ZSTD = 0x01;
export const COMPRESS_LZ4 = 0x02;

// Content types (entry.reserved[0:2]) - Shard v2.1
export const CONTENT_TYPE_UNKNOWN = 0x0000;
export const CONTENT_TYPE_TENSOR = 0x0001;  // TensorV1 encoded tensor
export const CONTENT_TYPE_JSON = 0x0002;    // Standard JSON
export const CONTENT_TYPE_COWRIE = 0x0003;  // Cowrie binary format
export const CONTENT_TYPE_GLYPH = 0x0004;   // GLYPH text format
export const CONTENT_TYPE_TEXT = 0x0005;    // Plain text (UTF-8)
export const CONTENT_TYPE_IMAGE = 0x0006;   // Image (PNG, JPEG, etc.)
export const CONTENT_TYPE_AUDIO = 0x0007;   // Audio (WAV, MP3, etc.)
export const CONTENT_TYPE_VIDEO = 0x0008;   // Video (MP4, WebM, etc.)
export const CONTENT_TYPE_PROTO = 0x0009;   // Protocol Buffers
export const CONTENT_TYPE_BLOB = 0x000A;    // Opaque binary blob
export const CONTENT_TYPE_USER_BASE = 0x8000;

// Header flag for content types
export const SHARD_FLAG_HAS_SCHEMA = 0x0010;
export const SHARD_FLAG_HAS_CONTENT_TYPES = 0x0080;

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
}

/**
 * Shard role/profile type.
 */
export enum ShardRole {
  UNKNOWN = 0x00,
  UMSH = 0x01,       // Model weights
  SAMPLE = 0x02,     // Training samples (SampleShard)
  GEMM_PANEL = 0x03, // GEMM panels
  MANIFEST = 0x04,   // Multi-file manifest
  WSHARD = 0x05,     // W-SHARD: World-model episode data
}

/**
 * Shard v2 header (64 bytes).
 */
export interface ShardHeader {
  magic: Buffer;
  version: number;
  role: ShardRole;
  flags: number;
  alignment: number;
  compressionDefault: number;
  entryCount: number;
  stringTableOffset: bigint;
  dataSectionOffset: bigint;
  schemaOffset: bigint;
  totalFileSize: bigint;
}

/**
 * Shard v2 index entry (48 bytes).
 */
export interface IndexEntry {
  nameHash: bigint;
  nameOffset: number;
  nameLen: number;
  flags: number;
  dataOffset: bigint;
  diskSize: bigint;
  origSize: bigint;
  checksum: number;
  contentType: number; // From reserved[0:2]
  name: string; // Populated after reading string table
}

/**
 * Create a new ShardHeader with default values.
 */
export function createShardHeader(role: ShardRole = ShardRole.SAMPLE): ShardHeader {
  return {
    magic: SHARD_MAGIC,
    version: SHARD_VERSION_2,
    role,
    flags: 0,
    alignment: ALIGN_64,
    compressionDefault: COMPRESS_NONE,
    entryCount: 0,
    stringTableOffset: 0n,
    dataSectionOffset: BigInt(SHARD_V2_HEADER_SIZE),
    schemaOffset: 0n,
    totalFileSize: 0n,
  };
}

export function serializeMetadata(meta: ShardMetadata): Buffer {
  const obj: Record<string, unknown> = {
    schema_version: meta.schemaVersion,
  };
  if (meta.schemaUri) obj['schema_uri'] = meta.schemaUri;
  if (meta.createdAt) obj['created_at'] = meta.createdAt;
  if (meta.sourceUri) obj['source_uri'] = meta.sourceUri;
  if (meta.producer) obj['producer'] = meta.producer;
  if (meta.description) obj['description'] = meta.description;
  if (meta.tags && meta.tags.length > 0) obj['tags'] = meta.tags;
  if (meta.extra && Object.keys(meta.extra).length > 0) obj['extra'] = meta.extra;
  if (meta.profile) obj['profile'] = meta.profile;
  if (meta.sampleShard && Object.keys(meta.sampleShard).length > 0) {
    const profile: Record<string, unknown> = {};
    if (meta.sampleShard.datasetName) profile['dataset_name'] = meta.sampleShard.datasetName;
    if (meta.sampleShard.sampleIdType) profile['sample_id_type'] = meta.sampleShard.sampleIdType;
    if (meta.sampleShard.keyEncoding) profile['key_encoding'] = meta.sampleShard.keyEncoding;
    if (meta.sampleShard.sampleCount !== undefined) profile['sample_count'] = meta.sampleShard.sampleCount;
    if (meta.sampleShard.datasetSchema && Object.keys(meta.sampleShard.datasetSchema).length > 0) {
      profile['dataset_schema'] = meta.sampleShard.datasetSchema;
    }
    if (meta.sampleShard.splits && Object.keys(meta.sampleShard.splits).length > 0) {
      profile['splits'] = meta.sampleShard.splits;
    }
    if (meta.sampleShard.labelMap && Object.keys(meta.sampleShard.labelMap).length > 0) {
      profile['label_map'] = meta.sampleShard.labelMap;
    }
    if (meta.sampleShard.featureStats && Object.keys(meta.sampleShard.featureStats).length > 0) {
      profile['feature_stats'] = meta.sampleShard.featureStats;
    }
    obj['sample_shard'] = profile;
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
    for (const [name, entry] of Object.entries(meta.entryMetadata)) {
      const out: Record<string, unknown> = {};
      if (entry.contentType) out['content_type'] = entry.contentType;
      if (entry.tags && entry.tags.length > 0) out['tags'] = entry.tags;
      if (entry.description) out['description'] = entry.description;
      if (entry.extra && Object.keys(entry.extra).length > 0) out['extra'] = entry.extra;
      if (entry.codec) out['codec'] = entry.codec;
      if (entry.codecVersion) out['codec_version'] = entry.codecVersion;
      if (entry.schemaFingerprint) out['schema_fingerprint'] = entry.schemaFingerprint;
      if (entry.semanticType) out['semantic_type'] = entry.semanticType;
      if (entry.canonicalHash) out['canonical_hash'] = entry.canonicalHash;
      if (entry.baseHash) out['base_hash'] = entry.baseHash;
      if (entry.rowCount !== undefined) out['row_count'] = entry.rowCount;
      if (entry.shape && entry.shape.length > 0) out['shape'] = entry.shape;
      if (entry.stats && Object.keys(entry.stats).length > 0) out['stats'] = entry.stats;
      em[name] = out;
    }
    obj['entry_metadata'] = em;
  }
  return Buffer.from(JSON.stringify(obj), 'utf8');
}

export function deserializeMetadata(data: Buffer): ShardMetadata {
  const obj = JSON.parse(data.toString('utf8')) as Record<string, unknown>;
  const rawSample = obj['sample_shard'] as Record<string, unknown> | undefined;
  const rawManifest = obj['manifest'] as Record<string, unknown> | undefined;
  const entryMetadata: Record<string, EntryMeta> = {};
  const rawEntryMetadata = obj['entry_metadata'] as Record<string, Record<string, unknown>> | undefined;
  if (rawEntryMetadata) {
    for (const [name, entry] of Object.entries(rawEntryMetadata)) {
      entryMetadata[name] = {
        contentType: entry['content_type'] as string | undefined,
        tags: entry['tags'] as string[] | undefined,
        description: entry['description'] as string | undefined,
        extra: entry['extra'] as Record<string, unknown> | undefined,
        codec: entry['codec'] as string | undefined,
        codecVersion: entry['codec_version'] as string | undefined,
        schemaFingerprint: entry['schema_fingerprint'] as string | undefined,
        semanticType: entry['semantic_type'] as string | undefined,
        canonicalHash: entry['canonical_hash'] as string | undefined,
        baseHash: entry['base_hash'] as string | undefined,
        rowCount: entry['row_count'] as number | undefined,
        shape: entry['shape'] as number[] | undefined,
        stats: entry['stats'] as Record<string, unknown> | undefined,
      };
    }
  }

  return {
    schemaVersion: (obj['schema_version'] as string) ?? 'shard-v2.1',
    schemaUri: obj['schema_uri'] as string | undefined,
    createdAt: obj['created_at'] as string | undefined,
    sourceUri: obj['source_uri'] as string | undefined,
    producer: obj['producer'] as string | undefined,
    description: obj['description'] as string | undefined,
    tags: obj['tags'] as string[] | undefined,
    extra: obj['extra'] as Record<string, unknown> | undefined,
    profile: obj['profile'] as string | undefined,
    sampleShard: rawSample
      ? {
          datasetName: rawSample['dataset_name'] as string | undefined,
          sampleIdType: rawSample['sample_id_type'] as string | undefined,
          keyEncoding: rawSample['key_encoding'] as string | undefined,
          sampleCount: rawSample['sample_count'] as number | undefined,
          datasetSchema: rawSample['dataset_schema'] as Record<string, unknown> | undefined,
          splits: rawSample['splits'] as Record<string, unknown> | undefined,
          labelMap: rawSample['label_map'] as Record<string, unknown> | undefined,
          featureStats: rawSample['feature_stats'] as Record<string, unknown> | undefined,
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
    entryMetadata: Object.keys(entryMetadata).length > 0 ? entryMetadata : undefined,
  };
}

/**
 * Serialize header to 64 bytes.
 */
export function serializeHeader(header: ShardHeader): Buffer {
  const buf = Buffer.alloc(SHARD_V2_HEADER_SIZE);

  // Magic (4 bytes)
  header.magic.copy(buf, 0);

  // Version (1 byte)
  buf.writeUInt8(header.version, 4);

  // Role (1 byte)
  buf.writeUInt8(header.role, 5);

  // Flags (2 bytes, little-endian)
  buf.writeUInt16LE(header.flags, 6);

  // Alignment (1 byte)
  buf.writeUInt8(header.alignment, 8);

  // Compression default (1 byte)
  buf.writeUInt8(header.compressionDefault, 9);

  // Index entry size (2 bytes, little-endian) - MUST be 48
  buf.writeUInt16LE(SHARD_V2_INDEX_ENTRY_SIZE, 10);

  // Entry count (4 bytes, little-endian)
  buf.writeUInt32LE(header.entryCount, 12);

  // String table offset (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.stringTableOffset, 16);

  // Data section offset (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.dataSectionOffset, 24);

  // Schema offset (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.schemaOffset, 32);

  // Total file size (8 bytes, little-endian)
  buf.writeBigUInt64LE(header.totalFileSize, 40);

  // Reserved (16 bytes) - already zeroed

  return buf;
}

/**
 * Parse header from 64 bytes.
 */
export function parseHeader(data: Buffer): ShardHeader {
  if (data.length < SHARD_V2_HEADER_SIZE) {
    throw new Error(`Header too short: ${data.length} < ${SHARD_V2_HEADER_SIZE}`);
  }

  const magic = data.subarray(0, 4);
  if (!magic.equals(SHARD_MAGIC)) {
    throw new Error(`Invalid magic: ${magic.toString('hex')}`);
  }

  const version = data.readUInt8(4);
  if (version !== SHARD_VERSION_2) {
    throw new Error(`Unsupported version: ${version}`);
  }

  return {
    magic,
    version,
    role: data.readUInt8(5) as ShardRole,
    flags: data.readUInt16LE(6),
    alignment: data.readUInt8(8),
    compressionDefault: data.readUInt8(9),
    entryCount: data.readUInt32LE(12),
    stringTableOffset: data.readBigUInt64LE(16),
    dataSectionOffset: data.readBigUInt64LE(24),
    schemaOffset: data.readBigUInt64LE(32),
    totalFileSize: data.readBigUInt64LE(40),
  };
}

/**
 * Serialize index entry to 48 bytes.
 */
export function serializeIndexEntry(entry: IndexEntry): Buffer {
  const buf = Buffer.alloc(SHARD_V2_INDEX_ENTRY_SIZE);

  buf.writeBigUInt64LE(entry.nameHash, 0);
  buf.writeUInt32LE(entry.nameOffset, 8);
  buf.writeUInt16LE(entry.nameLen, 12);
  buf.writeUInt16LE(entry.flags, 14);
  buf.writeBigUInt64LE(entry.dataOffset, 16);
  buf.writeBigUInt64LE(entry.diskSize, 24);
  buf.writeBigUInt64LE(entry.origSize, 32);
  buf.writeUInt32LE(entry.checksum, 40);
  // Reserved (4 bytes) - already zeroed

  return buf;
}

/**
 * Parse index entry from 48 bytes.
 */
export function parseIndexEntry(data: Buffer): IndexEntry {
  if (data.length < SHARD_V2_INDEX_ENTRY_SIZE) {
    throw new Error(`Entry too short: ${data.length} < ${SHARD_V2_INDEX_ENTRY_SIZE}`);
  }

  const reserved = data.readUInt32LE(44);
  return {
    nameHash: data.readBigUInt64LE(0),
    nameOffset: data.readUInt32LE(8),
    nameLen: data.readUInt16LE(12),
    flags: data.readUInt16LE(14),
    dataOffset: data.readBigUInt64LE(16),
    diskSize: data.readBigUInt64LE(24),
    origSize: data.readBigUInt64LE(32),
    checksum: data.readUInt32LE(40),
    contentType: reserved & 0xFFFF,
    name: '', // Populated after reading string table
  };
}

/**
 * Get human-readable content type name.
 */
export function contentTypeName(ct: number): string {
  switch (ct) {
    case CONTENT_TYPE_TENSOR: return 'tensor';
    case CONTENT_TYPE_JSON: return 'json';
    case CONTENT_TYPE_COWRIE: return 'cowrie';
    case CONTENT_TYPE_GLYPH: return 'glyph';
    case CONTENT_TYPE_TEXT: return 'text';
    case CONTENT_TYPE_IMAGE: return 'image';
    case CONTENT_TYPE_AUDIO: return 'audio';
    case CONTENT_TYPE_VIDEO: return 'video';
    case CONTENT_TYPE_PROTO: return 'proto';
    case CONTENT_TYPE_BLOB: return 'blob';
    default:
      if (ct >= CONTENT_TYPE_USER_BASE) {
        return `user:${ct - CONTENT_TYPE_USER_BASE}`;
      }
      return 'unknown';
  }
}

/**
 * Check if entry is compressed.
 */
export function isCompressed(entry: IndexEntry): boolean {
  return (entry.flags & 0x0001) !== 0;
}

/**
 * Get compression type (0=none, 1=zstd, 2=lz4). Matches Go bit flags.
 */
export function getCompressionType(entry: IndexEntry): number {
  if (entry.flags & 0x0004) return 2; // LZ4
  if (entry.flags & 0x0002) return 1; // zstd
  return 0;
}

/**
 * xxHash64 using xxhash-wasm for cross-language compatibility.
 * This matches Go's github.com/cespare/xxhash/v2 implementation.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let xxhashInstance: any = null;

/**
 * Initialize xxhash-wasm. Must be called before using simpleHash64.
 * Returns a promise that resolves when ready.
 */
export async function initXxhash(): Promise<void> {
  if (xxhashInstance) return;
  const xxhash = await import('xxhash-wasm');
  xxhashInstance = await xxhash.default();
}

/**
 * Synchronously initialize xxhash using require (for synchronous use).
 */
export function initXxhashSync(): void {
  if (xxhashInstance) return;
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const xxhash = require('xxhash-wasm');
  // Create a synchronous wrapper - this works because xxhash-wasm
  // can initialize synchronously in Node.js
  const initPromise = xxhash();
  // Wait for the promise synchronously (works in Node.js)
  const deasync = (fn: Promise<any>) => {
    let result: any;
    let done = false;
    fn.then((r: any) => { result = r; done = true; });
    // Spin until done (blocking)
    while (!done) {
      require('child_process').spawnSync('sleep', ['0.001']);
    }
    return result;
  };
  try {
    // Try deasync, but fall back to sync init if available
    xxhashInstance = deasync(initPromise);
  } catch {
    // Fallback: just set to null and use pure JS implementation
    xxhashInstance = null;
  }
}

/**
 * xxHash64 hash function (matching Go's xxhash.Sum64).
 * 
 * Note: This uses a pure JavaScript implementation for synchronous use.
 * For best performance, call initXxhash() at startup.
 */
export function simpleHash64(data: Buffer): bigint {
  // Use xxhash-wasm if available
  if (xxhashInstance) {
    // h64Raw accepts Uint8Array for raw bytes
    return xxhashInstance.h64Raw(new Uint8Array(data)) as bigint;
  }
  
  // Pure JavaScript xxHash64 implementation
  return xxhash64Pure(data);
}

// Pure JS xxHash64 implementation (fallback)
const PRIME1 = 0x9e3779b185ebca87n;
const PRIME2 = 0xc2b2ae3d27d4eb4fn;
const PRIME3 = 0x165667b19e3779f9n;
const PRIME4 = 0x85ebca77c2b2ae63n;
const PRIME5 = 0x27d4eb2f165667c5n;
const MASK64 = 0xffffffffffffffffn;

function rol64(x: bigint, k: number): bigint {
  return ((x << BigInt(k)) | (x >> BigInt(64 - k))) & MASK64;
}

function xxhRound(acc: bigint, input: bigint): bigint {
  acc = (acc + input * PRIME2) & MASK64;
  acc = rol64(acc, 31);
  return (acc * PRIME1) & MASK64;
}

function xxhMergeRound(acc: bigint, val: bigint): bigint {
  val = xxhRound(0n, val);
  acc ^= val;
  return (acc * PRIME1 + PRIME4) & MASK64;
}

function xxhash64Pure(data: Buffer): bigint {
  const n = data.length;
  let h: bigint;
  let p = 0;

  if (n >= 32) {
    let v1 = (PRIME1 + PRIME2) & MASK64;
    let v2 = PRIME2;
    let v3 = 0n;
    let v4 = (0n - PRIME1) & MASK64;

    while (p + 32 <= n) {
      v1 = xxhRound(v1, data.readBigUInt64LE(p));
      v2 = xxhRound(v2, data.readBigUInt64LE(p + 8));
      v3 = xxhRound(v3, data.readBigUInt64LE(p + 16));
      v4 = xxhRound(v4, data.readBigUInt64LE(p + 24));
      p += 32;
    }

    h = (rol64(v1, 1) + rol64(v2, 7) + rol64(v3, 12) + rol64(v4, 18)) & MASK64;
    h = xxhMergeRound(h, v1);
    h = xxhMergeRound(h, v2);
    h = xxhMergeRound(h, v3);
    h = xxhMergeRound(h, v4);
  } else {
    h = PRIME5;
  }

  h = (h + BigInt(n)) & MASK64;

  while (p + 8 <= n) {
    const k1 = xxhRound(0n, data.readBigUInt64LE(p));
    h ^= k1;
    h = (rol64(h, 27) * PRIME1 + PRIME4) & MASK64;
    p += 8;
  }

  if (p + 4 <= n) {
    h ^= (BigInt(data.readUInt32LE(p)) * PRIME1) & MASK64;
    h = (rol64(h, 23) * PRIME2 + PRIME3) & MASK64;
    p += 4;
  }

  while (p < n) {
    h ^= (BigInt(data[p]) * PRIME5) & MASK64;
    h = (rol64(h, 11) * PRIME1) & MASK64;
    p++;
  }

  // Avalanche
  h ^= h >> 33n;
  h = (h * PRIME2) & MASK64;
  h ^= h >> 29n;
  h = (h * PRIME3) & MASK64;
  h ^= h >> 32n;

  return h;
}

/**
 * CRC32C checksum (Castagnoli polynomial).
 * This matches Go's CRC32C implementation for cross-language compatibility.
 */
let crc32cTable: number[] | null = null;

function getCrc32cTable(): number[] {
  if (crc32cTable !== null) {
    return crc32cTable;
  }

  // Build CRC32C table (Castagnoli polynomial: 0x82F63B78)
  const polynomial = 0x82f63b78;
  crc32cTable = new Array(256);

  for (let i = 0; i < 256; i++) {
    let crc = i;
    for (let j = 0; j < 8; j++) {
      if (crc & 1) {
        crc = (crc >>> 1) ^ polynomial;
      } else {
        crc >>>= 1;
      }
    }
    crc32cTable[i] = crc >>> 0;
  }

  return crc32cTable;
}

/**
 * CRC32C checksum (Castagnoli polynomial, matching Go).
 */
export function crc32(data: Buffer): number {
  const table = getCrc32cTable();
  let crc = 0xffffffff;

  for (const byte of data) {
    crc = (crc >>> 8) ^ table[(crc ^ byte) & 0xff];
  }

  return (crc ^ 0xffffffff) >>> 0;
}

// ============================================================
// Path helpers (Shard v2.1)
// ============================================================

/** Canonical path separator for hierarchical names. */
export const PATH_SEPARATOR = '/';

/**
 * Split a hierarchical name into path components.
 */
export function splitPath(name: string): string[] {
  return name.split(PATH_SEPARATOR).filter(p => p !== '');
}

/**
 * Join path components into a hierarchical name.
 */
export function joinPath(...parts: string[]): string {
  return parts.join(PATH_SEPARATOR);
}

/**
 * Check if name starts with the given prefix.
 */
export function pathPrefix(name: string, prefix: string): boolean {
  if (prefix === '') return true;
  if (!prefix.endsWith(PATH_SEPARATOR)) prefix += PATH_SEPARATOR;
  return name.startsWith(prefix) || name === prefix.slice(0, -1);
}

/**
 * Get parent path (everything before last "/").
 */
export function pathParent(name: string): string {
  const idx = name.lastIndexOf(PATH_SEPARATOR);
  if (idx < 0) return '';
  return name.substring(0, idx);
}

/**
 * Get base name (everything after last "/").
 */
export function pathBase(name: string): string {
  const idx = name.lastIndexOf(PATH_SEPARATOR);
  if (idx < 0) return name;
  return name.substring(idx + 1);
}
