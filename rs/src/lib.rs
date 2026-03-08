//! Shard v2 binary container format — Rust implementation.
//!
//! # Format Overview
//!
//! ```text
//! Header (64 bytes):
//!   Bytes 0-3:   magic 'S','H','R','D'
//!   Byte 4:      version 0x02
//!   Byte 5:      role (0=unknown, 1=mosh, 2=sample, 3=gemmpanel, 4=manifest, 5=wshard, 6=umsh)
//!   Bytes 6-7:   flags (uint16 LE)
//!   Byte 8:      alignment (0=none, 16, 32, 64)
//!   Byte 9:      compression_default (0=none, 1=zstd, 2=lz4)
//!   Bytes 10-11: index_entry_size (always 48, uint16 LE)
//!   Bytes 12-15: entry_count (uint32 LE)
//!   Bytes 16-23: string_table_offset (uint64 LE)
//!   Bytes 24-31: data_section_offset (uint64 LE)
//!   Bytes 32-39: schema_offset (uint64 LE)
//!   Bytes 40-47: total_file_size (uint64 LE)
//!   Bytes 48-63: reserved (zeroes)
//!
//! Index Entry (48 bytes each, immediately after header):
//!   Bytes 0-7:   name_hash (xxHash64 of entry name, uint64 LE)
//!   Bytes 8-11:  name_offset (into string table, uint32 LE)
//!   Bytes 12-13: name_len (uint16 LE)
//!   Bytes 14-15: flags (uint16 LE)
//!   Bytes 16-23: data_offset (absolute file offset, uint64 LE)
//!   Bytes 24-31: disk_size (uint64 LE)
//!   Bytes 32-39: orig_size (uint64 LE)
//!   Bytes 40-43: checksum (CRC32C, uint32 LE)
//!   Bytes 44-47: reserved (lower 16 bits = content_type, uint32 LE)
//! ```

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::Mmap;

use serde::{Deserialize, Serialize};

// ============================================================
// Constants
// ============================================================

pub const SHARD_MAGIC: &[u8; 4] = b"SHRD";
pub const SHARD_VERSION2: u8 = 0x02;
pub const HEADER_SIZE: usize = 64;
pub const INDEX_ENTRY_SIZE: usize = 48;

// Roles
pub const ROLE_UNKNOWN: u8 = 0;
pub const ROLE_MOSH: u8 = 1;
pub const ROLE_SAMPLE: u8 = 2;
pub const ROLE_GEMMPANEL: u8 = 3;
pub const ROLE_MANIFEST: u8 = 4;
pub const ROLE_WSHARD: u8 = 5;
pub const ROLE_UMSH: u8 = 6;

// Alignment
pub const ALIGN_NONE: u8 = 0;
pub const ALIGN_16: u8 = 16;
pub const ALIGN_32: u8 = 32;
pub const ALIGN_64: u8 = 64;

// Compression
pub const COMPRESS_NONE: u8 = 0;
pub const COMPRESS_ZSTD: u8 = 1;
pub const COMPRESS_LZ4: u8 = 2;

// Header flags
pub const FLAG_LITTLE_ENDIAN: u16 = 0x0002;
pub const FLAG_HAS_SCHEMA: u16 = 0x0010;
pub const FLAG_HAS_CHECKSUMS: u16 = 0x0020;
pub const FLAG_STREAMING: u16 = 0x0040;
pub const FLAG_HAS_CONTENT_TYPES: u16 = 0x0080;

// Entry flags
pub const ENTRY_FLAG_COMPRESSED: u16 = 0x0001;
pub const ENTRY_FLAG_ZSTD: u16 = 0x0002;
pub const ENTRY_FLAG_LZ4: u16 = 0x0004;
pub const ENTRY_FLAG_CHUNKED: u16 = 0x0008;

// Content types
pub const CONTENT_TYPE_UNKNOWN: u16 = 0;
pub const CONTENT_TYPE_TENSOR: u16 = 1;
pub const CONTENT_TYPE_JSON: u16 = 2;
pub const CONTENT_TYPE_COWRIE: u16 = 3;
pub const CONTENT_TYPE_GLYPH: u16 = 4;
pub const CONTENT_TYPE_TEXT: u16 = 5;
pub const CONTENT_TYPE_IMAGE: u16 = 6;
pub const CONTENT_TYPE_AUDIO: u16 = 7;
pub const CONTENT_TYPE_VIDEO: u16 = 8;
pub const CONTENT_TYPE_PROTO: u16 = 9;
pub const CONTENT_TYPE_BLOB: u16 = 10;
pub const CONTENT_TYPE_EMBEDDING: u16 = 11;
pub const CONTENT_TYPE_TOKEN_IDS: u16 = 12;
pub const CONTENT_TYPE_LOGITS: u16 = 13;
pub const CONTENT_TYPE_ATTENTION_MASK: u16 = 14;
pub const CONTENT_TYPE_POSITION_IDS: u16 = 15;
pub const CONTENT_TYPE_EXPERT_INDICES: u16 = 16;

/// Default flags: little-endian + checksums + content types (0x00A2).
pub const DEFAULT_FLAGS: u16 = FLAG_LITTLE_ENDIAN | FLAG_HAS_CHECKSUMS | FLAG_HAS_CONTENT_TYPES;

// Compression thresholds and limits
const MIN_COMPRESS_SIZE: usize = 256;
const COMPRESSION_SAVINGS_NUM: usize = 9;
const COMPRESSION_SAVINGS_DEN: usize = 10;
const MAX_DECOMPRESS_SIZE: u64 = 1 << 30; // 1 GB

// ============================================================
// Security Limits
// ============================================================

pub const MAX_INDEX_SIZE: usize = 1 << 30;              // 1 GB
pub const MAX_STRING_TABLE_SIZE: usize = 100 * 1024 * 1024; // 100 MB
pub const MAX_ENTRY_COUNT: usize = 10_000_000;

// ============================================================
// Error type
// ============================================================

/// Errors returned by the Shard v2 reader and writer.
#[derive(Debug)]
pub enum ShardError {
    /// The magic bytes do not match 'SHRD'.
    InvalidMagic,
    /// The version byte is not 0x02.
    UnsupportedVersion(u8),
    /// The index_entry_size field is not 48.
    InvalidIndexEntrySize(u16),
    /// CRC32C checksum mismatch on data read.
    ChecksumMismatch {
        entry: usize,
        expected: u32,
        got: u32,
    },
    /// Named entry not found in the index.
    EntryNotFound(String),
    /// Compression or decompression is not supported.
    CompressionNotSupported,
    /// Decompressed size exceeds the safety limit.
    DecompressTooLarge(u64),
    /// Decompression failed.
    DecompressFailed(String),
    /// Security limit exceeded (entry count, index size, string table size, etc).
    SecurityLimitExceeded(String),
    /// total_file_size in header does not match actual file size.
    FileSizeMismatch { expected: u64, actual: u64 },
    /// An entry's data_offset or disk_size is out of valid range.
    EntryOffsetOutOfRange { index: usize },
    /// Entry index is out of bounds.
    IndexOutOfBounds { index: usize, count: usize },
    /// JSON serialization/deserialization error (metadata).
    JsonError(String),
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::InvalidMagic => write!(f, "invalid shard magic bytes"),
            ShardError::UnsupportedVersion(v) => write!(f, "unsupported shard version: {}", v),
            ShardError::InvalidIndexEntrySize(s) => {
                write!(f, "invalid index entry size: {} (expected 48)", s)
            }
            ShardError::ChecksumMismatch {
                entry,
                expected,
                got,
            } => write!(
                f,
                "checksum mismatch for entry {}: expected 0x{:08x}, got 0x{:08x}",
                entry, expected, got
            ),
            ShardError::EntryNotFound(name) => write!(f, "entry not found: {}", name),
            ShardError::CompressionNotSupported => write!(f, "compression not supported"),
            ShardError::DecompressTooLarge(size) => {
                write!(f, "decompressed size {} exceeds allowed limit", size)
            }
            ShardError::DecompressFailed(msg) => write!(f, "decompression failed: {}", msg),
            ShardError::SecurityLimitExceeded(msg) => write!(f, "security limit exceeded: {}", msg),
            ShardError::FileSizeMismatch { expected, actual } => write!(
                f,
                "total_file_size {} != actual file size {}",
                expected, actual
            ),
            ShardError::EntryOffsetOutOfRange { index } => {
                write!(f, "entry {} data_offset/disk_size out of range", index)
            }
            ShardError::IndexOutOfBounds { index, count } => {
                write!(f, "entry index {} out of bounds (count: {})", index, count)
            }
            ShardError::JsonError(msg) => write!(f, "JSON error: {}", msg),
            ShardError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<std::io::Error> for ShardError {
    fn from(e: std::io::Error) -> Self {
        ShardError::Io(e)
    }
}

// ============================================================
// Helper functions
// ============================================================

/// Compute CRC32C (Castagnoli) checksum of `data`.
pub fn compute_crc32c(data: &[u8]) -> u32 {
    crc32c::crc32c(data)
}

/// Compute xxHash64 of `name` using seed=0, matching Go's `github.com/cespare/xxhash/v2`.
pub fn compute_xxhash64(name: &str) -> u64 {
    xxhash_rust::xxh64::xxh64(name.as_bytes(), 0)
}

fn entry_data_range(data_len: usize, offset: u64, size: u64) -> Result<std::ops::Range<usize>, ShardError> {
    let offset = usize::try_from(offset).map_err(|_| {
        ShardError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "entry data offset does not fit in memory address space",
        ))
    })?;
    let size = usize::try_from(size).map_err(|_| {
        ShardError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "entry data size does not fit in memory address space",
        ))
    })?;
    let end = offset.checked_add(size).ok_or_else(|| {
        ShardError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "entry data range overflows address space",
        ))
    })?;
    if end > data_len {
        return Err(ShardError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "entry data extends past end of file",
        )));
    }
    Ok(offset..end)
}

fn decompress_zstd_bounded(raw: &[u8], expected_size: u64) -> Result<Vec<u8>, ShardError> {
    if expected_size > MAX_DECOMPRESS_SIZE {
        return Err(ShardError::DecompressTooLarge(expected_size));
    }

    let mut decoder = zstd::stream::read::Decoder::new(raw)
        .map_err(|e| ShardError::DecompressFailed(e.to_string()))?;
    let mut limited = decoder.by_ref().take(expected_size.saturating_add(1));
    let capacity = usize::try_from(expected_size).unwrap_or(usize::MAX).min(64 * 1024);
    let mut out = Vec::with_capacity(capacity);
    limited
        .read_to_end(&mut out)
        .map_err(|e| ShardError::DecompressFailed(e.to_string()))?;

    if out.len() as u64 > expected_size {
        return Err(ShardError::DecompressTooLarge(out.len() as u64));
    }
    if out.len() as u64 != expected_size {
        return Err(ShardError::DecompressFailed(format!(
            "zstd size mismatch: got {}, expected {}",
            out.len(),
            expected_size
        )));
    }

    Ok(out)
}

fn decompress_entry_data(raw: &[u8], entry: &IndexEntryV2) -> Result<Vec<u8>, ShardError> {
    if entry.orig_size > MAX_DECOMPRESS_SIZE {
        return Err(ShardError::DecompressTooLarge(entry.orig_size));
    }

    if entry.flags & ENTRY_FLAG_ZSTD != 0 {
        decompress_zstd_bounded(raw, entry.orig_size)
    } else if entry.flags & ENTRY_FLAG_LZ4 != 0 {
        lz4_flex::decompress(raw, entry.orig_size as usize)
            .map_err(|e| ShardError::DecompressFailed(e.to_string()))
    } else {
        Err(ShardError::CompressionNotSupported)
    }
}

/// Align `offset` upward to the given `alignment`. Returns `offset` when `alignment` is 0.
fn align_up(offset: usize, alignment: u8) -> usize {
    if alignment == 0 {
        return offset;
    }
    let a = alignment as usize;
    (offset + a - 1) & !(a - 1)
}

fn max_entry_data_end(entries: &[IndexEntryV2], data_section_offset: usize) -> Result<usize, ShardError> {
    let mut max_end = data_section_offset;
    for (index, entry) in entries.iter().enumerate() {
        let offset = usize::try_from(entry.data_offset)
            .map_err(|_| ShardError::EntryOffsetOutOfRange { index })?;
        let size = usize::try_from(entry.disk_size)
            .map_err(|_| ShardError::EntryOffsetOutOfRange { index })?;
        let end = offset
            .checked_add(size)
            .ok_or(ShardError::EntryOffsetOutOfRange { index })?;
        max_end = max_end.max(end);
    }
    Ok(max_end)
}

fn validate_schema_offset(
    header: &ShardV2Header,
    file_size: usize,
    min_schema_offset: usize,
) -> Result<Option<usize>, ShardError> {
    let schema_offset = usize::try_from(header.schema_offset).map_err(|_| {
        ShardError::SecurityLimitExceeded(format!(
            "schema_offset {} does not fit in usize",
            header.schema_offset
        ))
    })?;

    if schema_offset == 0 {
        return Ok(None);
    }
    if schema_offset > file_size {
        return Err(ShardError::SecurityLimitExceeded(format!(
            "schema_offset {} beyond file size {}",
            schema_offset, file_size
        )));
    }
    if schema_offset < min_schema_offset {
        return Err(ShardError::SecurityLimitExceeded(format!(
            "schema_offset {} overlaps data ending at {}",
            schema_offset, min_schema_offset
        )));
    }

    Ok(Some(schema_offset))
}

fn metadata_slice<'a>(
    buf: &'a [u8],
    header: &ShardV2Header,
    entries: &[IndexEntryV2],
) -> Result<Option<&'a [u8]>, ShardError> {
    let min_schema_offset = max_entry_data_end(entries, header.data_section_offset as usize)?;
    let Some(schema_offset) = validate_schema_offset(header, buf.len(), min_schema_offset)? else {
        return Ok(None);
    };

    let total_file_size = header.total_file_size as usize;
    Ok(Some(&buf[schema_offset..total_file_size]))
}

// ============================================================
// Metadata Structures
// ============================================================

/// Per-entry metadata stored in the schema section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntryMeta {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// JSON metadata stored at schema_offset in the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub schema_uri: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub created_at: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_uri: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub producer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub entry_metadata: HashMap<String, EntryMeta>,
}

fn default_schema_version() -> String {
    "shard-v2.1".to_string()
}

impl Default for ShardMetadata {
    fn default() -> Self {
        ShardMetadata {
            schema_version: "shard-v2.1".to_string(),
            schema_uri: String::new(),
            created_at: String::new(),
            source_uri: String::new(),
            producer: String::new(),
            description: String::new(),
            tags: Vec::new(),
            extra: serde_json::Map::new(),
            entry_metadata: HashMap::new(),
        }
    }
}

// ============================================================
// Path helpers
// ============================================================

/// Split a shard entry name on '/' into path components.
pub fn split_path(name: &str) -> Vec<&str> {
    if name.is_empty() {
        return Vec::new();
    }
    name.split('/').collect()
}

/// Join path components with '/'.
pub fn join_path(parts: &[&str]) -> String {
    parts.join("/")
}

/// Return everything before the last '/', or "" if no '/'.
pub fn path_parent(name: &str) -> &str {
    match name.rfind('/') {
        Some(idx) => &name[..idx],
        None => "",
    }
}

/// Return everything after the last '/', or the whole name if no '/'.
pub fn path_base(name: &str) -> &str {
    match name.rfind('/') {
        Some(idx) => &name[idx + 1..],
        None => name,
    }
}

// ============================================================
// Structs
// ============================================================

/// The 64-byte Shard v2 file header.
#[derive(Debug, Clone)]
pub struct ShardV2Header {
    pub magic: [u8; 4],
    pub version: u8,
    pub role: u8,
    pub flags: u16,
    pub alignment: u8,
    pub compression_default: u8,
    pub index_entry_size: u16,
    pub entry_count: u32,
    pub string_table_offset: u64,
    pub data_section_offset: u64,
    pub schema_offset: u64,
    pub total_file_size: u64,
}

impl ShardV2Header {
    /// Parse a header from a 64-byte buffer.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, ShardError> {
        if buf.len() < HEADER_SIZE {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "header too short",
            )));
        }

        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if &magic != SHARD_MAGIC {
            return Err(ShardError::InvalidMagic);
        }

        let version = buf[4];
        if version != SHARD_VERSION2 {
            return Err(ShardError::UnsupportedVersion(version));
        }

        let index_entry_size = u16::from_le_bytes([buf[10], buf[11]]);
        if index_entry_size != INDEX_ENTRY_SIZE as u16 {
            return Err(ShardError::InvalidIndexEntrySize(index_entry_size));
        }

        Ok(ShardV2Header {
            magic,
            version,
            role: buf[5],
            flags: u16::from_le_bytes([buf[6], buf[7]]),
            alignment: buf[8],
            compression_default: buf[9],
            index_entry_size,
            entry_count: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            string_table_offset: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            data_section_offset: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            schema_offset: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            total_file_size: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
        })
    }

    /// Serialize header to a 64-byte buffer.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4] = self.version;
        buf[5] = self.role;
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8] = self.alignment;
        buf[9] = self.compression_default;
        buf[10..12].copy_from_slice(&self.index_entry_size.to_le_bytes());
        buf[12..16].copy_from_slice(&self.entry_count.to_le_bytes());
        buf[16..24].copy_from_slice(&self.string_table_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.data_section_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.schema_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.total_file_size.to_le_bytes());
        // bytes 48-63 remain zero (reserved)
        buf
    }
}

/// A single 48-byte entry in the index section.
#[derive(Debug, Clone)]
pub struct IndexEntryV2 {
    pub name_hash: u64,
    pub name_offset: u32,
    pub name_len: u16,
    pub flags: u16,
    pub data_offset: u64,
    pub disk_size: u64,
    pub orig_size: u64,
    pub checksum: u32,
    /// Lower 16 bits: content_type.
    pub reserved: u32,
    /// Entry name (populated by the reader after string table lookup).
    pub name: String,
}

impl IndexEntryV2 {
    /// Returns the content type stored in the lower 16 bits of `reserved`.
    pub fn content_type(&self) -> u16 {
        (self.reserved & 0xFFFF) as u16
    }

    /// Returns true if the entry data is compressed.
    pub fn compressed(&self) -> bool {
        self.flags & ENTRY_FLAG_COMPRESSED != 0
    }

    /// Parse an index entry from a 48-byte buffer (name will be empty; fill separately).
    fn from_bytes(buf: &[u8]) -> Self {
        IndexEntryV2 {
            name_hash: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            name_offset: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            name_len: u16::from_le_bytes([buf[12], buf[13]]),
            flags: u16::from_le_bytes([buf[14], buf[15]]),
            data_offset: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            disk_size: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            orig_size: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            checksum: u32::from_le_bytes(buf[40..44].try_into().unwrap()),
            reserved: u32::from_le_bytes(buf[44..48].try_into().unwrap()),
            name: String::new(),
        }
    }

    /// Serialize the index entry to a 48-byte buffer (name field is not written).
    fn to_bytes(&self) -> [u8; 48] {
        let mut buf = [0u8; 48];
        buf[0..8].copy_from_slice(&self.name_hash.to_le_bytes());
        buf[8..12].copy_from_slice(&self.name_offset.to_le_bytes());
        buf[12..14].copy_from_slice(&self.name_len.to_le_bytes());
        buf[14..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..24].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.disk_size.to_le_bytes());
        buf[32..40].copy_from_slice(&self.orig_size.to_le_bytes());
        buf[40..44].copy_from_slice(&self.checksum.to_le_bytes());
        buf[44..48].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }
}

// ============================================================
// Reader
// ============================================================

/// Reads an existing Shard v2 file.
#[derive(Debug)]
pub struct ShardV2Reader {
    /// Raw file bytes (entire file loaded into memory).
    data: Vec<u8>,
    header: ShardV2Header,
    entries: Vec<IndexEntryV2>,
    name_to_index: HashMap<String, usize>,
}

impl ShardV2Reader {
    /// Open a Shard v2 file by path.
    pub fn from_file(path: &Path) -> Result<Self, ShardError> {
        let data = fs::read(path)?;
        Self::from_bytes(data)
    }

    /// Parse a Shard v2 from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, ShardError> {
        if data.len() < HEADER_SIZE {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too small for header",
            )));
        }

        let header = ShardV2Header::from_bytes(&data[..HEADER_SIZE])?;

        // Security: validate total_file_size
        if header.total_file_size != data.len() as u64 {
            return Err(ShardError::FileSizeMismatch {
                expected: header.total_file_size,
                actual: data.len() as u64,
            });
        }
        let (entries, name_to_index) = parse_index_from_bytes(&data, &header)?;

        Ok(ShardV2Reader {
            data,
            header,
            entries,
            name_to_index,
        })
    }

    /// Return a reference to the parsed header.
    pub fn header(&self) -> &ShardV2Header {
        &self.header
    }

    /// Number of entries in the shard.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Name of entry `i`.
    pub fn entry_name(&self, i: usize) -> Result<&str, ShardError> {
        self.entries
            .get(i)
            .map(|e| e.name.as_str())
            .ok_or(ShardError::IndexOutOfBounds { index: i, count: self.entries.len() })
    }

    /// All entry names in index order.
    pub fn entry_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Return the index entry metadata for entry `i`.
    pub fn get_entry_info(&self, i: usize) -> Result<&IndexEntryV2, ShardError> {
        self.entries
            .get(i)
            .ok_or(ShardError::IndexOutOfBounds { index: i, count: self.entries.len() })
    }

    /// Look up the index of `name`. Returns `None` if not found.
    pub fn lookup(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Return true if `name` exists in the shard.
    pub fn has_entry(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Read entry `i` by index, decompressing if necessary and verifying CRC32C.
    ///
    /// For uncompressed entries the data is returned as an owned Vec<u8>.
    /// For compressed entries the data is decompressed and the checksum is verified
    /// against the decompressed content.
    pub fn read_entry(&self, i: usize) -> Result<Vec<u8>, ShardError> {
        let entry = self.entries.get(i).ok_or(ShardError::IndexOutOfBounds {
            index: i,
            count: self.entries.len(),
        })?;
        let range = entry_data_range(self.data.len(), entry.data_offset, entry.disk_size)?;
        let raw = &self.data[range];

        let data = if entry.compressed() {
            decompress_entry_data(raw, entry)?
        } else {
            raw.to_vec()
        };

        // CRC32C is computed on the DECOMPRESSED data.
        if self.header.flags & FLAG_HAS_CHECKSUMS != 0 {
            let computed = compute_crc32c(&data);
            if computed != entry.checksum {
                return Err(ShardError::ChecksumMismatch {
                    entry: i,
                    expected: entry.checksum,
                    got: computed,
                });
            }
        }

        Ok(data)
    }

    /// Read entry by name, decompressing if necessary and verifying CRC32C.
    pub fn read_entry_by_name(&self, name: &str) -> Result<Vec<u8>, ShardError> {
        let i = self
            .lookup(name)
            .ok_or_else(|| ShardError::EntryNotFound(name.to_string()))?;
        self.read_entry(i)
    }

    /// Return all entry names that start with `prefix/` (or equal `prefix`).
    pub fn list_prefix(&self, prefix: &str) -> Vec<&str> {
        let full_prefix = if prefix.ends_with('/') {
            prefix.to_string()
        } else {
            format!("{}/", prefix)
        };

        self.entries
            .iter()
            .filter(|e| e.name.starts_with(&full_prefix) || e.name == prefix)
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Return immediate children (one path component deep) under `prefix`.
    ///
    /// # Examples
    ///
    /// For entries `["layer.0/weight", "layer.0/bias", "layer.1/weight", "embed"]`:
    /// - `list_children("layer.0/")` → `["layer.0/weight", "layer.0/bias"]`
    /// - `list_children("")`         → `["layer.0/", "layer.1/", "embed"]`
    /// - `list_children("layer.")`   → `["layer.0/", "layer.1/"]`
    pub fn list_children(&self, prefix: &str) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for entry in &self.entries {
            let name = &entry.name;
            if !name.starts_with(prefix) {
                continue;
            }
            let remainder = &name[prefix.len()..];
            let child = match remainder.find('/') {
                Some(slash_pos) => format!("{}{}/", prefix, &remainder[..slash_pos]),
                None => name.clone(),
            };

            if seen.insert(child.clone()) {
                result.push(child);
            }
        }

        result
    }

    /// Read only the first `max_bytes` of entry `i` (for header scanning).
    ///
    /// For compressed entries, decompresses fully then truncates.
    /// For uncompressed entries, returns a direct slice without full read.
    pub fn read_entry_prefix(&self, i: usize, max_bytes: usize) -> Result<Vec<u8>, ShardError> {
        let entry = self.entries.get(i).ok_or(ShardError::IndexOutOfBounds {
            index: i,
            count: self.entries.len(),
        })?;
        if entry.compressed() {
            let data = self.read_entry(i)?;
            let end = max_bytes.min(data.len());
            return Ok(data[..end].to_vec());
        }
        // Uncompressed: direct slice
        let offset = entry.data_offset as usize;
        let size = (entry.disk_size as usize).min(max_bytes);
        if offset + size > self.data.len() {
            return Err(ShardError::EntryOffsetOutOfRange { index: i });
        }
        Ok(self.data[offset..offset + size].to_vec())
    }

    /// Read JSON metadata from schema section.
    ///
    /// Returns `None` if no schema (`schema_offset == 0`).
    pub fn read_metadata(&self) -> Result<Option<ShardMetadata>, ShardError> {
        let Some(meta_bytes) = metadata_slice(&self.data, &self.header, &self.entries)? else {
            return Ok(None);
        };
        let meta: ShardMetadata = serde_json::from_slice(meta_bytes)
            .map_err(|e| ShardError::JsonError(e.to_string()))?;
        Ok(Some(meta))
    }
}

// ============================================================
// Shared index-parsing helper
// ============================================================

/// Parse the index entries and build a name→index map from a byte slice.
///
/// `buf` must be the complete file contents (either a `Vec<u8>` or an `Mmap`
/// deref'd to `[u8]`).  The returned `(entries, name_to_index)` are identical
/// to what `ShardV2Reader::from_bytes` builds internally.
fn parse_index_from_bytes(
    buf: &[u8],
    header: &ShardV2Header,
) -> Result<(Vec<IndexEntryV2>, HashMap<String, usize>), ShardError> {
    let entry_count = header.entry_count as usize;
    let string_table_offset = header.string_table_offset as usize;
    let data_section_offset = header.data_section_offset as usize;
    let file_size = buf.len();

    // Security: validate entry_count
    if entry_count > MAX_ENTRY_COUNT {
        return Err(ShardError::SecurityLimitExceeded(format!(
            "entry_count {} exceeds MAX_ENTRY_COUNT {}",
            entry_count, MAX_ENTRY_COUNT
        )));
    }

    // Security: validate index size
    let index_size = entry_count * INDEX_ENTRY_SIZE;
    if index_size > MAX_INDEX_SIZE {
        return Err(ShardError::SecurityLimitExceeded(format!(
            "index size {} exceeds MAX_INDEX_SIZE {}",
            index_size, MAX_INDEX_SIZE
        )));
    }

    // Security: validate string table size
    if data_section_offset >= string_table_offset {
        let st_size = data_section_offset - string_table_offset;
        if st_size > MAX_STRING_TABLE_SIZE {
            return Err(ShardError::SecurityLimitExceeded(format!(
                "string table size {} exceeds MAX_STRING_TABLE_SIZE {}",
                st_size, MAX_STRING_TABLE_SIZE
            )));
        }
    }

    // Bounds checks
    let index_start = HEADER_SIZE;
    let index_end = index_start + entry_count * INDEX_ENTRY_SIZE;
    if buf.len() < index_end {
        return Err(ShardError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "file too small for index",
        )));
    }
    if string_table_offset > buf.len() || data_section_offset > buf.len() {
        return Err(ShardError::SecurityLimitExceeded(format!(
            "string table or data section beyond file: string_table_offset={}, data_section_offset={}, file_size={}",
            string_table_offset, data_section_offset, buf.len()
        )));
    }

    if data_section_offset < string_table_offset {
        return Err(ShardError::SecurityLimitExceeded(format!(
            "data_section_offset {} < string_table_offset {}",
            data_section_offset, string_table_offset
        )));
    }

    let string_table = &buf[string_table_offset..data_section_offset];

    // Parse each index entry and resolve name from string table.
    let mut entries = Vec::with_capacity(entry_count);
    for i in 0..entry_count {
        let off = index_start + i * INDEX_ENTRY_SIZE;
        let mut entry = IndexEntryV2::from_bytes(&buf[off..off + INDEX_ENTRY_SIZE]);

        let name_off = entry.name_offset as usize;
        let name_len = entry.name_len as usize;
        if name_off + name_len <= string_table.len() {
            entry.name =
                String::from_utf8_lossy(&string_table[name_off..name_off + name_len])
                    .into_owned();
        }

        entries.push(entry);
    }

    // Security: validate entry data offsets
    let mut prev_end: usize = 0;
    for (i, entry) in entries.iter().enumerate() {
        let offset = entry.data_offset as usize;
        let size = entry.disk_size as usize;
        let end = offset
            .checked_add(size)
            .ok_or(ShardError::EntryOffsetOutOfRange { index: i })?;

        if offset < data_section_offset {
            return Err(ShardError::EntryOffsetOutOfRange { index: i });
        }
        if end > file_size {
            return Err(ShardError::EntryOffsetOutOfRange { index: i });
        }
        if i > 0 && offset < prev_end {
            return Err(ShardError::EntryOffsetOutOfRange { index: i });
        }
        prev_end = end;
    }

    let min_schema_offset = prev_end.max(data_section_offset);
    let _ = validate_schema_offset(header, file_size, min_schema_offset)?;

    // Build name → index map.
    let mut name_to_index = HashMap::with_capacity(entry_count);
    for (i, e) in entries.iter().enumerate() {
        name_to_index.insert(e.name.clone(), i);
    }

    Ok((entries, name_to_index))
}

// ============================================================
// MmapShardV2Reader
// ============================================================

/// A Shard v2 reader backed by a memory-mapped file.
///
/// `read_entry` returns a zero-copy `&[u8]` slice directly into the mapped
/// region for uncompressed entries.  For compressed entries, use
/// `read_entry_decompressed` which returns an owned `Vec<u8>`.
pub struct MmapShardV2Reader {
    /// Keep the file open for the lifetime of the mapping.
    _file: File,
    mmap: Mmap,
    header: ShardV2Header,
    entries: Vec<IndexEntryV2>,
    name_to_index: HashMap<String, usize>,
}

impl MmapShardV2Reader {
    /// Open a Shard v2 file using `mmap`.
    ///
    /// The entire file is mapped read-only.  `read_entry` on uncompressed
    /// entries returns zero-copy slices into the mapping.
    pub fn open(path: &Path) -> Result<Self, ShardError> {
        let file = File::open(path).map_err(ShardError::Io)?;
        // SAFETY: we hold the file open for the lifetime of `mmap` via `_file`.
        // The mapping is read-only (PROT_READ / MAP_PRIVATE).
        let mmap = unsafe { Mmap::map(&file).map_err(ShardError::Io)? };

        if mmap.len() < HEADER_SIZE {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too small for header",
            )));
        }

        let header = ShardV2Header::from_bytes(&mmap[..HEADER_SIZE])?;

        // Validate total_file_size matches the mapping length.
        if header.total_file_size != mmap.len() as u64 {
            return Err(ShardError::FileSizeMismatch {
                expected: header.total_file_size,
                actual: mmap.len() as u64,
            });
        }

        let (entries, name_to_index) = parse_index_from_bytes(&mmap, &header)?;

        Ok(MmapShardV2Reader {
            _file: file,
            mmap,
            header,
            entries,
            name_to_index,
        })
    }

    /// Return a reference to the parsed header.
    pub fn header(&self) -> &ShardV2Header {
        &self.header
    }

    /// Number of entries in the shard.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Name of entry `i`.
    pub fn entry_name(&self, i: usize) -> Result<&str, ShardError> {
        self.entries
            .get(i)
            .map(|e| e.name.as_str())
            .ok_or(ShardError::IndexOutOfBounds { index: i, count: self.entries.len() })
    }

    /// All entry names in index order.
    pub fn entry_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Return the index entry metadata for entry `i`.
    pub fn get_entry_info(&self, i: usize) -> Result<&IndexEntryV2, ShardError> {
        self.entries
            .get(i)
            .ok_or(ShardError::IndexOutOfBounds { index: i, count: self.entries.len() })
    }

    /// Look up the index of `name`.  Returns `None` if not found.
    pub fn lookup(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Return true if `name` exists in the shard.
    pub fn has_entry(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Zero-copy read of an uncompressed entry.
    ///
    /// Returns a `&[u8]` slice directly into the memory-mapped region.
    /// The slice is valid as long as this `MmapShardV2Reader` is alive.
    ///
    /// Returns `Err(ShardError::CompressionNotSupported)` for compressed
    /// entries — use `read_entry_decompressed` instead.
    pub fn read_entry(&self, i: usize) -> Result<&[u8], ShardError> {
        let entry = self.entries.get(i).ok_or(ShardError::IndexOutOfBounds {
            index: i,
            count: self.entries.len(),
        })?;

        if entry.compressed() {
            return Err(ShardError::CompressionNotSupported);
        }

        let offset = entry.data_offset as usize;
        let size = entry.disk_size as usize;

        if offset + size > self.mmap.len() {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "entry data extends past end of file",
            )));
        }

        let data = &self.mmap[offset..offset + size];

        if self.header.flags & FLAG_HAS_CHECKSUMS != 0 {
            let computed = compute_crc32c(data);
            if computed != entry.checksum {
                return Err(ShardError::ChecksumMismatch {
                    entry: i,
                    expected: entry.checksum,
                    got: computed,
                });
            }
        }

        Ok(data)
    }

    /// Read entry by name (zero-copy, uncompressed only).
    ///
    /// Returns `Err(ShardError::EntryNotFound)` if the name is not found, or
    /// `Err(ShardError::CompressionNotSupported)` if the entry is compressed.
    pub fn read_entry_by_name(&self, name: &str) -> Result<&[u8], ShardError> {
        let i = self
            .lookup(name)
            .ok_or_else(|| ShardError::EntryNotFound(name.to_string()))?;
        self.read_entry(i)
    }

    /// Read and decompress any entry, returning an owned `Vec<u8>`.
    ///
    /// For uncompressed entries the raw bytes are copied into a new `Vec`.
    /// CRC32C is verified against the decompressed content.
    pub fn read_entry_decompressed(&self, i: usize) -> Result<Vec<u8>, ShardError> {
        let entry = self.entries.get(i).ok_or(ShardError::IndexOutOfBounds {
            index: i,
            count: self.entries.len(),
        })?;
        let range = entry_data_range(self.mmap.len(), entry.data_offset, entry.disk_size)?;
        let raw = &self.mmap[range];

        let data = if entry.compressed() {
            decompress_entry_data(raw, entry)?
        } else {
            raw.to_vec()
        };

        // CRC32C is computed on the DECOMPRESSED data.
        if self.header.flags & FLAG_HAS_CHECKSUMS != 0 {
            let computed = compute_crc32c(&data);
            if computed != entry.checksum {
                return Err(ShardError::ChecksumMismatch {
                    entry: i,
                    expected: entry.checksum,
                    got: computed,
                });
            }
        }

        Ok(data)
    }

    /// Return all entry names that start with `prefix/` (or equal `prefix`).
    pub fn list_prefix(&self, prefix: &str) -> Vec<&str> {
        let full_prefix = if prefix.ends_with('/') {
            prefix.to_string()
        } else {
            format!("{}/", prefix)
        };

        self.entries
            .iter()
            .filter(|e| e.name.starts_with(&full_prefix) || e.name == prefix)
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Return immediate children (one path component deep) under `prefix`.
    pub fn list_children(&self, prefix: &str) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for entry in &self.entries {
            let name = &entry.name;
            if !name.starts_with(prefix) {
                continue;
            }
            let remainder = &name[prefix.len()..];
            let child = match remainder.find('/') {
                Some(slash_pos) => format!("{}{}/", prefix, &remainder[..slash_pos]),
                None => name.clone(),
            };

            if seen.insert(child.clone()) {
                result.push(child);
            }
        }

        result
    }

    /// Read JSON metadata from schema section, if present.
    pub fn read_metadata(&self) -> Result<Option<ShardMetadata>, ShardError> {
        let Some(meta_bytes) = metadata_slice(&self.mmap, &self.header, &self.entries)? else {
            return Ok(None);
        };
        let meta: ShardMetadata = serde_json::from_slice(meta_bytes)
            .map_err(|e| ShardError::JsonError(e.to_string()))?;
        Ok(Some(meta))
    }
}

// ============================================================
// Schema Validation
// ============================================================

/// Specification for an expected entry in a shard.
#[derive(Debug, Clone)]
pub struct EntrySpec {
    /// Glob pattern or exact entry name to match.
    pub pattern: String,
    /// Expected content type.  0 = accept any content type.
    pub content_type: u16,
    /// When `true`, a missing entry is not a validation error.
    pub optional: bool,
    /// Human-readable description (not used during validation).
    pub description: String,
}

/// Schema that describes expected contents of a shard.
#[derive(Debug, Clone)]
pub struct ShardSchema {
    pub name: String,
    pub version: u16,
    pub specs: Vec<EntrySpec>,
}

/// A single schema validation failure.
#[derive(Debug)]
pub struct ValidationError {
    pub spec_pattern: String,
    pub message: String,
}

impl ShardSchema {
    /// Create a new empty schema with the given name.
    pub fn new(name: &str) -> Self {
        ShardSchema {
            name: name.to_string(),
            version: 1,
            specs: Vec::new(),
        }
    }

    /// Add a spec and return `&mut Self` for method chaining.
    pub fn add_spec(&mut self, pattern: &str, content_type: u16, optional: bool) -> &mut Self {
        self.specs.push(EntrySpec {
            pattern: pattern.to_string(),
            content_type,
            optional,
            description: String::new(),
        });
        self
    }

    /// Add a required spec with `content_type = 0` (any).
    pub fn add_required(&mut self, pattern: &str) -> &mut Self {
        self.add_spec(pattern, 0, false)
    }

    /// Add an optional spec with `content_type = 0` (any).
    pub fn add_optional(&mut self, pattern: &str) -> &mut Self {
        self.add_spec(pattern, 0, true)
    }
}

/// Match a glob `pattern` against an entry `name`.
///
/// Rules:
/// - `**` prefix wildcard: everything before `**` must be a prefix of `name`.
/// - `*` single-component wildcard: matches any characters **except** `.` and `/`.
/// - All other characters match literally.
pub fn match_pattern(pattern: &str, name: &str) -> bool {
    if pattern.contains("**") {
        let prefix = pattern.split("**").next().unwrap_or("");
        return name.starts_with(prefix);
    }
    glob_match(pattern, name)
}

/// Iterative glob match where `*` does **not** cross `.` or `/` separators.
///
/// The algorithm splits the pattern on `*` into literal segments.  The first
/// segment must match at the start of `text`, the last must match at the end,
/// and intermediate segments must appear in order without any consumed
/// character being a separator (`'.'` or `'/'`).
fn glob_match(pattern: &str, text: &str) -> bool {
    // Split the pattern on '*' to get literal segments.
    let segments: Vec<&str> = pattern.split('*').collect();

    if segments.len() == 1 {
        // No wildcard — must match exactly.
        return pattern == text;
    }

    let first = segments[0];
    let last = segments[segments.len() - 1];

    // The text must start with the first literal segment.
    if !text.starts_with(first) {
        return false;
    }

    // The text must end with the last literal segment.
    if !text.ends_with(last) {
        return false;
    }

    // For patterns with more than two segments (multiple `*`), also check that
    // each middle segment appears in order and the skipped regions contain no
    // separator characters.  For the common single-`*` case (two segments)
    // we only need to verify the gap contains no separator.
    let mut pos = first.len(); // current position in `text`
    let text_bytes = text.as_bytes();

    for seg in &segments[1..segments.len() - 1] {
        // Find `seg` in text starting at `pos`.
        let remaining = &text[pos..];
        match remaining.find(seg) {
            None => return false,
            Some(offset) => {
                // The characters from `pos` to `pos + offset` are consumed by
                // the wildcard and must not include any separator.
                let gap = &text[pos..pos + offset];
                if gap.contains('.') || gap.contains('/') {
                    return false;
                }
                pos += offset + seg.len();
            }
        }
    }

    // Verify the gap before the final literal segment contains no separator.
    // `pos` is now just after the last middle segment (or just after `first`
    // if there are only two segments).
    let end_pos = text_bytes.len() - last.len();
    let gap = &text[pos..end_pos];
    if gap.contains('.') || gap.contains('/') {
        return false;
    }

    true
}

/// Validate a shard against a schema.
///
/// Returns a `Vec<ValidationError>`.  An empty vector means the shard satisfies
/// the schema.
///
/// Checks:
/// 1. Every non-optional spec matches at least one entry name.
/// 2. Matched entries have the expected `content_type` when the spec specifies
///    one (non-zero).
pub fn validate_schema(reader: &ShardV2Reader, schema: &ShardSchema) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let names = reader.entry_names();

    for spec in &schema.specs {
        let matches: Vec<&&str> = names
            .iter()
            .filter(|n| match_pattern(&spec.pattern, n))
            .collect();

        if matches.is_empty() && !spec.optional {
            errors.push(ValidationError {
                spec_pattern: spec.pattern.clone(),
                message: format!(
                    "required entry matching '{}' not found",
                    spec.pattern
                ),
            });
            continue;
        }

        if spec.content_type != 0 {
            for name in &matches {
                if let Some(idx) = reader.lookup(name) {
                    let Ok(info) = reader.get_entry_info(idx) else { continue };
                    if info.content_type() != spec.content_type {
                        errors.push(ValidationError {
                            spec_pattern: spec.pattern.clone(),
                            message: format!(
                                "entry '{}' has content_type {}, expected {}",
                                name,
                                info.content_type(),
                                spec.content_type,
                            ),
                        });
                    }
                }
            }
        }
    }

    errors
}

// ============================================================
// Writer
// ============================================================

/// A pending entry buffered by the writer before finalization.
struct PendingEntry {
    name: String,
    data: Vec<u8>,
    content_type: u16,
    compress: bool,
    comp_type: u8,
}

/// Builds and serializes a Shard v2 binary.
pub struct ShardV2Writer {
    role: u8,
    alignment: u8,
    compression: u8,
    entries: Vec<PendingEntry>,
    metadata: Option<ShardMetadata>,
}

impl ShardV2Writer {
    /// Create a new writer with the given role.
    pub fn new(role: u8) -> Self {
        ShardV2Writer {
            role,
            alignment: ALIGN_64,
            compression: COMPRESS_NONE,
            entries: Vec::new(),
            metadata: None,
        }
    }

    /// Attach shard-level metadata. Written as JSON at the end of the file.
    pub fn set_metadata(&mut self, meta: ShardMetadata) {
        self.metadata = Some(meta);
    }

    /// Set the data alignment (0, 16, 32, or 64).
    pub fn set_alignment(&mut self, align: u8) {
        self.alignment = align;
    }

    /// Set the default compression codec (COMPRESS_NONE, COMPRESS_ZSTD, COMPRESS_LZ4).
    pub fn set_compression(&mut self, comp: u8) {
        self.compression = comp;
    }

    /// Append an uncompressed entry with unknown content type.
    pub fn write_entry(&mut self, name: &str, data: &[u8]) {
        self.entries.push(PendingEntry {
            name: name.to_string(),
            data: data.to_vec(),
            content_type: CONTENT_TYPE_UNKNOWN,
            compress: false,
            comp_type: COMPRESS_NONE,
        });
    }

    /// Append an uncompressed entry with a specific content type.
    pub fn write_entry_typed(&mut self, name: &str, data: &[u8], content_type: u16) {
        self.entries.push(PendingEntry {
            name: name.to_string(),
            data: data.to_vec(),
            content_type,
            compress: false,
            comp_type: COMPRESS_NONE,
        });
    }

    /// Append a compressed entry using the writer's default compression codec.
    pub fn write_entry_compressed(&mut self, name: &str, data: &[u8]) {
        let comp_type = self.compression;
        self.entries.push(PendingEntry {
            name: name.to_string(),
            data: data.to_vec(),
            content_type: CONTENT_TYPE_UNKNOWN,
            compress: true,
            comp_type,
        });
    }

    /// Append an entry with explicit compress flag and compression type.
    pub fn write_entry_with_options(
        &mut self,
        name: &str,
        data: &[u8],
        compress: bool,
        comp_type: u8,
    ) {
        self.entries.push(PendingEntry {
            name: name.to_string(),
            data: data.to_vec(),
            content_type: CONTENT_TYPE_UNKNOWN,
            compress,
            comp_type,
        });
    }

    /// Finalize and return the complete shard as a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let entry_count = self.entries.len();

        // Build string table (null-terminated names).
        let mut string_table: Vec<u8> = Vec::new();
        let mut name_offsets: Vec<u32> = Vec::with_capacity(entry_count);
        for e in &self.entries {
            name_offsets.push(string_table.len() as u32);
            string_table.extend_from_slice(e.name.as_bytes());
            string_table.push(0); // null terminator
        }

        // Compute layout offsets (pre-compression, because compressed sizes are
        // determined per-entry; we must materialize the disk data first).
        let string_table_offset = HEADER_SIZE + entry_count * INDEX_ENTRY_SIZE;
        let data_section_offset_raw = string_table_offset + string_table.len();
        let data_section_offset = align_up(data_section_offset_raw, self.alignment);

        // Materialize compressed (or plain) data for each entry.
        struct EntryDisk {
            disk_data: Vec<u8>,
            orig_size: u64,
            entry_flags: u16,
            checksum: u32, // CRC32C of UNCOMPRESSED data
        }

        let mut disk_entries: Vec<EntryDisk> = Vec::with_capacity(entry_count);
        for e in &self.entries {
            // Checksum is always over the original (uncompressed) data.
            let checksum = compute_crc32c(&e.data);

            let (disk_data, entry_flags) =
                if e.compress && e.data.len() >= MIN_COMPRESS_SIZE {
                    match e.comp_type {
                        COMPRESS_ZSTD => {
                            let compressed =
                                zstd::encode_all(&e.data[..], 0 /* default level */)
                                    .unwrap_or_else(|_| e.data.clone());
                            if compressed.len()
                                < e.data.len() * COMPRESSION_SAVINGS_NUM / COMPRESSION_SAVINGS_DEN
                            {
                                (compressed, ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD)
                            } else {
                                (e.data.clone(), 0u16)
                            }
                        }
                        COMPRESS_LZ4 => {
                            let compressed = lz4_flex::compress(&e.data);
                            if compressed.len()
                                < e.data.len() * COMPRESSION_SAVINGS_NUM / COMPRESSION_SAVINGS_DEN
                            {
                                (compressed, ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4)
                            } else {
                                (e.data.clone(), 0u16)
                            }
                        }
                        _ => (e.data.clone(), 0u16),
                    }
                } else {
                    (e.data.clone(), 0u16)
                };

            disk_entries.push(EntryDisk {
                orig_size: e.data.len() as u64,
                checksum,
                disk_data,
                entry_flags,
            });
        }

        // Compute per-entry data offsets based on actual disk sizes.
        let mut data_offsets: Vec<usize> = Vec::with_capacity(entry_count);
        let mut current = data_section_offset;
        for de in &disk_entries {
            let aligned = align_up(current, self.alignment);
            data_offsets.push(aligned);
            current = aligned + de.disk_data.len();
        }
        let data_end = current;

        // Serialize metadata (if any) to JSON.
        let meta_bytes: Option<Vec<u8>> = self.metadata.as_ref().map(|m| {
            serde_json::to_vec(m).expect("metadata serialization failed")
        });

        // Compute schema_offset and total_size.
        let (schema_offset, total_size) = match &meta_bytes {
            Some(mb) => (data_end as u64, data_end + mb.len()),
            None => (0u64, data_end),
        };

        // Compute flags for header.
        let mut hdr_flags: u16 = FLAG_LITTLE_ENDIAN | FLAG_HAS_CHECKSUMS;
        for e in &self.entries {
            if e.content_type != CONTENT_TYPE_UNKNOWN {
                hdr_flags |= FLAG_HAS_CONTENT_TYPES;
                break;
            }
        }
        if meta_bytes.is_some() {
            hdr_flags |= FLAG_HAS_SCHEMA;
        }

        // Determine compression_default for header byte.
        let compression_default = if self.entries.iter().any(|e| e.compress) {
            self.compression
        } else {
            COMPRESS_NONE
        };

        // Allocate output buffer.
        let mut out = vec![0u8; total_size];

        // Header
        {
            let hdr = ShardV2Header {
                magic: *SHARD_MAGIC,
                version: SHARD_VERSION2,
                role: self.role,
                flags: hdr_flags,
                alignment: self.alignment,
                compression_default,
                index_entry_size: INDEX_ENTRY_SIZE as u16,
                entry_count: entry_count as u32,
                string_table_offset: string_table_offset as u64,
                data_section_offset: data_section_offset as u64,
                schema_offset,
                total_file_size: total_size as u64,
            };
            let hdr_bytes = hdr.to_bytes();
            out[..HEADER_SIZE].copy_from_slice(&hdr_bytes);
        }

        // Index entries
        for (i, e) in self.entries.iter().enumerate() {
            let de = &disk_entries[i];
            let entry = IndexEntryV2 {
                name_hash: compute_xxhash64(&e.name),
                name_offset: name_offsets[i],
                name_len: e.name.len() as u16,
                flags: de.entry_flags,
                data_offset: data_offsets[i] as u64,
                disk_size: de.disk_data.len() as u64,
                orig_size: de.orig_size,
                checksum: de.checksum,
                reserved: e.content_type as u32,
                name: e.name.clone(),
            };
            let entry_bytes = entry.to_bytes();
            let off = HEADER_SIZE + i * INDEX_ENTRY_SIZE;
            out[off..off + INDEX_ENTRY_SIZE].copy_from_slice(&entry_bytes);
        }

        // String table (starts immediately after index)
        {
            let off = string_table_offset;
            out[off..off + string_table.len()].copy_from_slice(&string_table);
            // bytes between string table end and data_section_offset remain zero (padding)
        }

        // Data entries (compressed or plain)
        for (i, de) in disk_entries.iter().enumerate() {
            let off = data_offsets[i];
            out[off..off + de.disk_data.len()].copy_from_slice(&de.disk_data);
        }

        // Metadata JSON (appended after data section)
        if let Some(mb) = &meta_bytes {
            let off = schema_offset as usize;
            out[off..off + mb.len()].copy_from_slice(mb);
        }

        out
    }

    /// Finalize and write the shard to a file.
    pub fn write_to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        let bytes = self.to_bytes();
        fs::write(path, bytes)
    }
}

// ============================================================
// Streaming Writer
// ============================================================

/// Metadata recorded for each entry written by `ShardV2StreamWriter`.
struct StreamEntry {
    name: String,
    data_offset: u64,
    disk_size: u64,
    orig_size: u64,
    checksum: u32,
    flags: u16,
}

/// Zero-copy streaming writer.  Data is written directly to the output file
/// at a pre-calculated offset.  The header + index region is written at the
/// **end** via a seek-back to position 0, so no in-memory data buffer is
/// required.  `max_entries` must be known upfront to reserve the correct
/// amount of space at the beginning of the file.
///
/// # Usage
///
/// ```no_run
/// use std::path::Path;
/// use shard_format::{ShardV2StreamWriter, ROLE_MOSH, ALIGN_64};
///
/// let mut sw = ShardV2StreamWriter::new(Path::new("out.shard"), ROLE_MOSH, 1000).unwrap();
/// sw.set_alignment(ALIGN_64).unwrap();
/// sw.begin_data().unwrap();
/// sw.write_entry("weights", &[0u8; 512]).unwrap();
/// sw.write_entry("config",  b"{}").unwrap();
/// sw.finalize().unwrap();
/// ```
pub struct ShardV2StreamWriter {
    file: std::fs::File,
    role: u8,
    alignment: u8,
    compression: u8,
    max_entries: usize,
    entries: Vec<StreamEntry>,
    reserved_space: u64,
    current_offset: u64,
    started: bool,
    finalized: bool,
}

impl ShardV2StreamWriter {
    /// Open `path` for writing and create a stream writer with `max_entries`
    /// reserved index slots.
    pub fn new(path: &Path, role: u8, max_entries: usize) -> Result<Self, ShardError> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(ShardV2StreamWriter {
            file,
            role,
            alignment: ALIGN_64,
            compression: COMPRESS_NONE,
            max_entries,
            entries: Vec::new(),
            reserved_space: 0,
            current_offset: 0,
            started: false,
            finalized: false,
        })
    }

    /// Set the data alignment.  Must be called **before** `begin_data()`.
    pub fn set_alignment(&mut self, align: u8) -> Result<(), ShardError> {
        if self.started {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot set alignment after begin_data()",
            )));
        }
        self.alignment = align;
        Ok(())
    }

    /// Set the default compression codec used by `write_entry_compressed()`.
    pub fn set_compression(&mut self, comp: u8) {
        self.compression = comp;
    }

    /// Calculate the reserved header + index space and seek the file position
    /// past it.  After this returns, `write_entry()` may be called.
    pub fn begin_data(&mut self) -> Result<(), ShardError> {
        use std::io::Seek;
        if self.started {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "begin_data() already called",
            )));
        }
        // Estimate: header + full index array + 64 bytes per entry name
        let n = self.max_entries;
        let mut reserved = HEADER_SIZE + n * INDEX_ENTRY_SIZE + n * 64;
        if self.alignment > 0 {
            reserved = align_up(reserved, self.alignment);
        }
        self.reserved_space = reserved as u64;
        self.current_offset = reserved as u64;
        self.file.seek(std::io::SeekFrom::Start(self.reserved_space))?;
        self.started = true;
        Ok(())
    }

    /// Write `data` at the current file position (with alignment padding) and
    /// record the entry metadata.
    pub fn write_entry(&mut self, name: &str, data: &[u8]) -> Result<(), ShardError> {
        use std::io::Write;
        if !self.started {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "call begin_data() first",
            )));
        }
        if self.finalized {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "already finalized",
            )));
        }

        // Alignment padding before entry data
        if self.alignment > 0 {
            let aligned = align_up(self.current_offset as usize, self.alignment) as u64;
            if aligned > self.current_offset {
                let pad = vec![0u8; (aligned - self.current_offset) as usize];
                self.file.write_all(&pad)?;
                self.current_offset = aligned;
            }
        }

        let checksum = compute_crc32c(data);
        let disk_size = data.len() as u64;
        self.file.write_all(data)?;
        self.entries.push(StreamEntry {
            name: name.to_string(),
            data_offset: self.current_offset,
            disk_size,
            orig_size: disk_size,
            checksum,
            flags: 0,
        });
        self.current_offset += disk_size;
        Ok(())
    }

    /// Write `data` using the codec set by `set_compression()`.  Falls back to
    /// uncompressed storage when data is small, incompressible, or the codec is
    /// `COMPRESS_NONE`.
    pub fn write_entry_compressed(&mut self, name: &str, data: &[u8]) -> Result<(), ShardError> {
        use std::io::Write;
        if !self.started {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "call begin_data() first",
            )));
        }
        if self.finalized {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "already finalized",
            )));
        }

        // Alignment padding before entry data
        if self.alignment > 0 {
            let aligned = align_up(self.current_offset as usize, self.alignment) as u64;
            if aligned > self.current_offset {
                let pad = vec![0u8; (aligned - self.current_offset) as usize];
                self.file.write_all(&pad)?;
                self.current_offset = aligned;
            }
        }

        let checksum = compute_crc32c(data);
        let orig_size = data.len() as u64;

        // Attempt compression only for large enough payloads
        let (disk_data, entry_flags): (Vec<u8>, u16) =
            if self.compression != COMPRESS_NONE && data.len() >= MIN_COMPRESS_SIZE {
                match self.compression {
                    COMPRESS_ZSTD => {
                        match zstd::encode_all(data, 3) {
                            Ok(c)
                                if c.len() * COMPRESSION_SAVINGS_DEN
                                    < data.len() * COMPRESSION_SAVINGS_NUM =>
                            {
                                (c, ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD)
                            }
                            _ => (data.to_vec(), 0),
                        }
                    }
                    COMPRESS_LZ4 => {
                        let c = lz4_flex::compress(data);
                        if c.len() * COMPRESSION_SAVINGS_DEN < data.len() * COMPRESSION_SAVINGS_NUM
                        {
                            (c, ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4)
                        } else {
                            (data.to_vec(), 0)
                        }
                    }
                    _ => (data.to_vec(), 0),
                }
            } else {
                (data.to_vec(), 0)
            };

        let disk_size = disk_data.len() as u64;
        self.file.write_all(&disk_data)?;
        self.entries.push(StreamEntry {
            name: name.to_string(),
            data_offset: self.current_offset,
            disk_size,
            orig_size,
            checksum,
            flags: entry_flags,
        });
        self.current_offset += disk_size;
        Ok(())
    }

    /// Seek to position 0 and write the header, index, and string table.
    ///
    /// Returns an error if the actual string table exceeds the reserved space
    /// (i.e. `max_entries * 64` bytes was not enough for all entry names).
    pub fn finalize(&mut self) -> Result<(), ShardError> {
        use std::io::{Seek, Write};
        if self.finalized {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "already finalized",
            )));
        }
        self.finalized = true;

        let n = self.entries.len();

        // Build string table (null-terminated names)
        let mut string_table: Vec<u8> = Vec::new();
        let mut name_offsets: Vec<u32> = Vec::with_capacity(n);
        for e in &self.entries {
            name_offsets.push(string_table.len() as u32);
            string_table.extend_from_slice(e.name.as_bytes());
            string_table.push(0); // null terminator
        }

        // Verify front-matter fits in reserved region
        let actual_front = HEADER_SIZE + n * INDEX_ENTRY_SIZE + string_table.len();
        let actual_front_aligned = if self.alignment > 0 {
            align_up(actual_front, self.alignment)
        } else {
            actual_front
        };
        if actual_front_aligned as u64 > self.reserved_space {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "string table overflow: need {} bytes but reserved {}",
                    actual_front_aligned, self.reserved_space
                ),
            )));
        }

        let string_table_offset = (HEADER_SIZE + n * INDEX_ENTRY_SIZE) as u64;
        let data_section_offset = self.reserved_space;
        let total_file_size = self.current_offset;

        // Build header
        let header = ShardV2Header {
            magic: *SHARD_MAGIC,
            version: SHARD_VERSION2,
            role: self.role,
            flags: DEFAULT_FLAGS | FLAG_STREAMING,
            alignment: self.alignment,
            compression_default: self.compression,
            index_entry_size: INDEX_ENTRY_SIZE as u16,
            entry_count: n as u32,
            string_table_offset,
            data_section_offset,
            schema_offset: 0,
            total_file_size,
        };

        // Seek to the beginning and write front matter
        self.file.seek(std::io::SeekFrom::Start(0))?;
        self.file.write_all(&header.to_bytes())?;

        // Write index entries
        for (i, e) in self.entries.iter().enumerate() {
            let entry = IndexEntryV2 {
                name_hash: compute_xxhash64(&e.name),
                name_offset: name_offsets[i],
                name_len: e.name.len() as u16,
                flags: e.flags,
                data_offset: e.data_offset,
                disk_size: e.disk_size,
                orig_size: e.orig_size,
                checksum: e.checksum,
                reserved: 0,
                name: e.name.clone(),
            };
            self.file.write_all(&entry.to_bytes())?;
        }

        // Write string table
        self.file.write_all(&string_table)?;

        // Pad to the data section boundary
        let pos_after_st = string_table_offset as usize + string_table.len();
        if pos_after_st < data_section_offset as usize {
            let pad = vec![0u8; data_section_offset as usize - pos_after_st];
            self.file.write_all(&pad)?;
        }

        self.file.flush()?;
        Ok(())
    }
}
