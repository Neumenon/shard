/**
 * Cowrie v3 encoder/decoder for TypeScript.
 *
 * Implements the Cowrie v3 binary format to ensure byte-identical output
 * with Go and Python implementations.
 *
 * Wire format:
 * - Header: 'S' 'J' 0x02 0x00 (4 bytes)
 * - Dictionary: count:uvarint + entries (len:uvarint + utf8 bytes)
 * - Root value: type-tagged recursive structure
 *
 * Type tags:
 * - 0x00: NULL
 * - 0x01: FALSE
 * - 0x02: TRUE
 * - 0x03: INT64 (zigzag varint)
 * - 0x04: FLOAT64 (8 bytes LE)
 * - 0x05: STRING (len:uvarint + utf8)
 * - 0x06: ARRAY (count:uvarint + values)
 * - 0x07: OBJECT (count:uvarint + (fieldID:uvarint + value) pairs)
 * - 0x08: BYTES (len:uvarint + raw bytes)
 * - 0x09: UINT64 (uvarint)
 * - 0x40-0xBF: FIXINT (value = tag - 0x40)
 * - 0xC0-0xCF: FIXARRAY (count = tag - 0xC0)
 * - 0xD0-0xDF: FIXMAP (count = tag - 0xD0)
 * - 0xE0-0xEF: FIXNEG (value = -(tag - 0xE0) - 1)
 */

// Magic bytes and version
export const MAGIC = Buffer.from([0x53, 0x4a]); // 'SJ'
export const VERSION = 0x02;

// Type tags
const TAG_NULL = 0x00;
const TAG_FALSE = 0x01;
const TAG_TRUE = 0x02;
const TAG_INT64 = 0x03;
const TAG_FLOAT64 = 0x04;
const TAG_STRING = 0x05;
const TAG_ARRAY = 0x06;
const TAG_OBJECT = 0x07;
const TAG_BYTES = 0x08;
const TAG_UINT64 = 0x09;

// v3 inline types
const FIXINT_BASE = 0x40;
const FIXINT_MAX = 0xBF;
const FIXARRAY_BASE = 0xC0;
const FIXARRAY_MAX = 0xCF;
const FIXMAP_BASE = 0xD0;
const FIXMAP_MAX = 0xDF;
const FIXNEG_BASE = 0xE0;
const FIXNEG_MAX = 0xEF;

/**
 * Encode a JavaScript value to Cowrie v2 binary format.
 *
 * Supports: null, boolean, number, bigint, string, Buffer, Array, Object
 */
export function encode(value: unknown): Buffer {
  // Build dictionary of all object keys
  const keys: string[] = [];
  const keyIndex = new Map<string, number>();
  collectKeys(value, keys, keyIndex);

  const chunks: Buffer[] = [];
  // Write header
  chunks.push(MAGIC);
  chunks.push(Buffer.from([VERSION, 0x00])); // version + flags

  // Write dictionary
  chunks.push(writeUvarint(keys.length));
  for (const key of keys) {
    const keyBytes = Buffer.from(key, 'utf-8');
    chunks.push(writeUvarint(keyBytes.length));
    chunks.push(keyBytes);
  }

  // Write root value
  encodeValue(value, keyIndex, chunks);
  return Buffer.concat(chunks);
}

/**
 * Decode Cowrie v2 binary format to a JavaScript value.
 */
export function decode(data: Buffer): unknown {
  if (data.length < 4) {
    throw new Error('Cowrie data too short');
  }
  if (!data.subarray(0, 2).equals(MAGIC)) {
    throw new Error(`Invalid Cowrie magic: ${data.subarray(0, 2).toString('hex')}`);
  }
  if (data[2] !== VERSION) {
    throw new Error(`Unsupported Cowrie version: ${data[2]}`);
  }

  // Skip header
  let pos = 4;

  // Read dictionary
  const [dictLen, newPos1] = readUvarint(data, pos);
  pos = newPos1;
  const keys: string[] = [];
  for (let i = 0; i < dictLen; i++) {
    const [keyLen, newPos2] = readUvarint(data, pos);
    pos = newPos2;
    const key = data.subarray(pos, pos + keyLen).toString('utf-8');
    keys.push(key);
    pos += keyLen;
  }

  // Read root value
  const [value] = decodeValue(data, pos, keys);
  return value;
}

// ============================================================
// Encoding helpers
// ============================================================

function collectKeys(value: unknown, keys: string[], keyIndex: Map<string, number>): void {
  if (value === null || value === undefined) {
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectKeys(item, keys, keyIndex);
    }
  } else if (typeof value === 'object' && !Buffer.isBuffer(value)) {
    for (const [k, v] of Object.entries(value)) {
      if (!keyIndex.has(k)) {
        keyIndex.set(k, keys.length);
        keys.push(k);
      }
      collectKeys(v, keys, keyIndex);
    }
  }
}

function writeUvarint(n: number | bigint): Buffer {
  const num = typeof n === 'bigint' ? n : BigInt(n);
  const bytes: number[] = [];
  let val = num;
  while (val >= 0x80n) {
    bytes.push(Number(val & 0x7fn) | 0x80);
    val >>= 7n;
  }
  bytes.push(Number(val));
  return Buffer.from(bytes);
}

function zigzagEncode(n: bigint): bigint {
  const n64 = n & 0xffffffffffffffffn;
  const signed = n64 >= 0x8000000000000000n ? n64 - 0x10000000000000000n : n64;
  return ((signed << 1n) ^ (signed >> 63n)) & 0xffffffffffffffffn;
}

function encodeValue(value: unknown, keyIndex: Map<string, number>, chunks: Buffer[]): void {
  if (value === null || value === undefined) {
    chunks.push(Buffer.from([TAG_NULL]));
    return;
  }
  if (typeof value === 'boolean') {
    chunks.push(Buffer.from([value ? TAG_TRUE : TAG_FALSE]));
    return;
  }
  if (typeof value === 'number') {
    if (Number.isInteger(value)) {
      if (value >= 0 && value <= 127) {
        chunks.push(Buffer.from([FIXINT_BASE + value]));
      } else if (value >= -16 && value <= -1) {
        chunks.push(Buffer.from([FIXNEG_BASE + (-1 - value)]));
      } else {
        chunks.push(Buffer.from([TAG_INT64]));
        chunks.push(writeUvarint(zigzagEncode(BigInt(value))));
      }
    } else {
      chunks.push(Buffer.from([TAG_FLOAT64]));
      const buf = Buffer.alloc(8);
      buf.writeDoubleLE(value, 0);
      chunks.push(buf);
    }
    return;
  }
  if (typeof value === 'bigint') {
    if (value >= 0n && value <= 127n) {
      chunks.push(Buffer.from([FIXINT_BASE + Number(value)]));
    } else if (value >= -16n && value <= -1n) {
      chunks.push(Buffer.from([FIXNEG_BASE + Number(-1n - value)]));
    } else {
      chunks.push(Buffer.from([TAG_INT64]));
      chunks.push(writeUvarint(zigzagEncode(value)));
    }
    return;
  }
  if (typeof value === 'string') {
    chunks.push(Buffer.from([TAG_STRING]));
    const strBytes = Buffer.from(value, 'utf-8');
    chunks.push(writeUvarint(strBytes.length));
    chunks.push(strBytes);
    return;
  }
  if (Buffer.isBuffer(value)) {
    chunks.push(Buffer.from([TAG_BYTES]));
    chunks.push(writeUvarint(value.length));
    chunks.push(value);
    return;
  }
  if (Array.isArray(value)) {
    if (value.length <= 15) {
      chunks.push(Buffer.from([FIXARRAY_BASE + value.length]));
    } else {
      chunks.push(Buffer.from([TAG_ARRAY]));
      chunks.push(writeUvarint(value.length));
    }
    for (const item of value) {
      encodeValue(item, keyIndex, chunks);
    }
    return;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value);
    if (entries.length <= 15) {
      chunks.push(Buffer.from([FIXMAP_BASE + entries.length]));
    } else {
      chunks.push(Buffer.from([TAG_OBJECT]));
      chunks.push(writeUvarint(entries.length));
    }
    for (const [k, v] of entries) {
      const idx = keyIndex.get(k);
      if (idx === undefined) {
        throw new Error(`Key not in index: ${k}`);
      }
      chunks.push(writeUvarint(idx));
      encodeValue(v, keyIndex, chunks);
    }
    return;
  }
  throw new Error(`Cannot encode type: ${typeof value}`);
}

// ============================================================
// Decoding helpers
// ============================================================

function readUvarint(data: Buffer, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  while (true) {
    if (pos >= data.length) {
      throw new Error('Unexpected end of data reading uvarint');
    }
    const b = data[pos];
    pos++;
    result |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) {
      return [result, pos];
    }
    shift += 7;
  }
}

function zigzagDecodeBigInt(n: bigint): bigint {
  return (n >> 1n) ^ -(n & 1n);
}

function decodeValue(data: Buffer, pos: number, keys: string[]): [unknown, number] {
  if (pos >= data.length) {
    throw new Error('Unexpected end of data');
  }
  const tag = data[pos];
  pos++;

  switch (tag) {
    case TAG_NULL:
      return [null, pos];
    case TAG_FALSE:
      return [false, pos];
    case TAG_TRUE:
      return [true, pos];
    case TAG_INT64: {
      let result = 0n;
      let shift = 0n;
      while (true) {
        if (pos >= data.length) {
          throw new Error('Unexpected end of data reading INT64');
        }
        const b = BigInt(data[pos]);
        pos++;
        result |= (b & 0x7fn) << shift;
        if ((b & 0x80n) === 0n) {
          break;
        }
        shift += 7n;
      }
      const decoded = zigzagDecodeBigInt(result);
      if (decoded >= BigInt(Number.MIN_SAFE_INTEGER) && decoded <= BigInt(Number.MAX_SAFE_INTEGER)) {
        return [Number(decoded), pos];
      }
      return [decoded, pos];
    }
    case TAG_UINT64: {
      const [value, newPos] = readUvarint(data, pos);
      return [value, newPos];
    }
    case TAG_FLOAT64: {
      const value = data.readDoubleLE(pos);
      return [value, pos + 8];
    }
    case TAG_STRING: {
      const [length, newPos] = readUvarint(data, pos);
      pos = newPos;
      const value = data.subarray(pos, pos + length).toString('utf-8');
      return [value, pos + length];
    }
    case TAG_BYTES: {
      const [length, newPos] = readUvarint(data, pos);
      pos = newPos;
      const value = Buffer.from(data.subarray(pos, pos + length));
      return [value, pos + length];
    }
    case TAG_ARRAY: {
      const [count, newPos] = readUvarint(data, pos);
      pos = newPos;
      const result: unknown[] = [];
      for (let i = 0; i < count; i++) {
        const [item, itemPos] = decodeValue(data, pos, keys);
        result.push(item);
        pos = itemPos;
      }
      return [result, pos];
    }
    case TAG_OBJECT: {
      const [count, newPos] = readUvarint(data, pos);
      pos = newPos;
      const result: Record<string, unknown> = {};
      for (let i = 0; i < count; i++) {
        const [keyIdx, keyPos] = readUvarint(data, pos);
        pos = keyPos;
        if (keyIdx >= keys.length) {
          throw new Error(`Invalid field ID: ${keyIdx}`);
        }
        const key = keys[keyIdx];
        const [value, valuePos] = decodeValue(data, pos, keys);
        result[key] = value;
        pos = valuePos;
      }
      return [result, pos];
    }
    default:
      // v3 inline types
      if (tag >= FIXINT_BASE && tag <= FIXINT_MAX) {
        return [tag - FIXINT_BASE, pos];
      }
      if (tag >= FIXARRAY_BASE && tag <= FIXARRAY_MAX) {
        const count = tag - FIXARRAY_BASE;
        const result: unknown[] = [];
        for (let i = 0; i < count; i++) {
          const [item, itemPos] = decodeValue(data, pos, keys);
          result.push(item);
          pos = itemPos;
        }
        return [result, pos];
      }
      if (tag >= FIXMAP_BASE && tag <= FIXMAP_MAX) {
        const count = tag - FIXMAP_BASE;
        const result: Record<string, unknown> = {};
        for (let i = 0; i < count; i++) {
          const [keyIdx, keyPos] = readUvarint(data, pos);
          pos = keyPos;
          if (keyIdx >= keys.length) {
            throw new Error(`Invalid field ID: ${keyIdx}`);
          }
          const key = keys[keyIdx];
          const [value, valuePos] = decodeValue(data, pos, keys);
          result[key] = value;
          pos = valuePos;
        }
        return [result, pos];
      }
      if (tag >= FIXNEG_BASE && tag <= FIXNEG_MAX) {
        return [-(tag - FIXNEG_BASE) - 1, pos];
      }
      throw new Error(`Unknown type tag: 0x${tag.toString(16).padStart(2, '0')}`);
  }
}
