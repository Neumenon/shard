//! Fuzz target: zstd and lz4 decompression with garbage input.
//!
//! Matches Go's FuzzDecompressData coverage:
//! - arbitrary bytes fed to zstd decoder
//! - arbitrary bytes fed to lz4 decoder
//! - huge orig_size values (capped to avoid OOM)
//! - invalid compression type codes
//! - empty input
//!
//! We exercise the decompressors both directly (to catch codec panics) and
//! indirectly via ShardReader.read_entry (to cover the full entry path).
#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use shard_format::{
    COMPRESS_LZ4, COMPRESS_ZSTD, ENTRY_FLAG_COMPRESSED, ENTRY_FLAG_LZ4, ENTRY_FLAG_ZSTD,
    FLAG_HAS_CHECKSUMS, HEADER_SIZE, INDEX_ENTRY_SIZE, SHARD_MAGIC, SHARD_VERSION,
};

/// Structured fuzz input so libfuzzer can explore comp_type and orig_size
/// independently of the raw compressed payload.
#[derive(Arbitrary, Debug)]
struct DecompressInput {
    /// Raw bytes to feed to the decompressor.
    data: Vec<u8>,
    /// 0 = none, 1 = zstd, 2 = lz4, anything else = unknown.
    comp_type: u8,
    /// Claimed original (uncompressed) size; capped to 1 MB to avoid OOM.
    orig_size: u32,
}

fuzz_target!(|input: DecompressInput| {
    // Safety cap: never allocate more than 1 MB during fuzzing.
    let orig_size = (input.orig_size as usize).min(1 << 20);

    // --- Direct codec tests ---
    // The codecs themselves must never panic on garbage input.
    match input.comp_type {
        COMPRESS_ZSTD => {
            let _ = zstd::decode_all(&input.data[..]);
        }
        COMPRESS_LZ4 => {
            let _ = lz4_flex::decompress(&input.data, orig_size);
        }
        _ => {}
    }

    // --- End-to-end path through ShardReader ---
    // Build a minimal single-entry shard whose entry data is `input.data` and
    // whose index entry flags request the chosen compression codec.
    //
    // Layout (no alignment padding to keep it simple):
    //   [0..64)      header
    //   [64..112)    index entry
    //   [112..116)   string table: "e\0" + 2 padding bytes
    //   [116..116+N) entry data (input.data)
    let name = b"e";
    let st_offset: u64 = (HEADER_SIZE + INDEX_ENTRY_SIZE) as u64; // 112
    let data_offset: u64 = st_offset + 4; // 116  (string table = 4 bytes: 'e','\0',pad,pad)
    let disk_size: u64 = input.data.len() as u64;
    let total: usize = data_offset as usize + input.data.len();

    let mut buf = vec![0u8; total];

    // Header
    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = 1; // ROLE_MOSH
    // Flags: little-endian | checksums (so read_entry verifies CRC)
    buf[6..8].copy_from_slice(&FLAG_HAS_CHECKSUMS.to_le_bytes());
    buf[8] = 0;  // ALIGN_NONE
    buf[9] = 0;  // compression_default=none (header field, independent of entry flags)
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&1u32.to_le_bytes()); // entry_count=1
    buf[16..24].copy_from_slice(&st_offset.to_le_bytes());
    buf[24..32].copy_from_slice(&data_offset.to_le_bytes());
    buf[32..40].copy_from_slice(&0u64.to_le_bytes()); // schema_offset=0
    buf[40..48].copy_from_slice(&(total as u64).to_le_bytes());

    // Index entry
    // name_hash (8 bytes) — leave as 0 (hash mismatch won't cause an error in reader)
    // name_offset (4 bytes) = 0
    buf[HEADER_SIZE + 8..HEADER_SIZE + 12].copy_from_slice(&0u32.to_le_bytes());
    // name_len (2 bytes) = 1
    buf[HEADER_SIZE + 12..HEADER_SIZE + 14].copy_from_slice(&(name.len() as u16).to_le_bytes());
    // flags (2 bytes): set compression flags based on comp_type
    let entry_flags: u16 = match input.comp_type {
        COMPRESS_ZSTD => ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD,
        COMPRESS_LZ4 => ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4,
        _ => 0,
    };
    buf[HEADER_SIZE + 14..HEADER_SIZE + 16].copy_from_slice(&entry_flags.to_le_bytes());
    // data_offset (8 bytes)
    buf[HEADER_SIZE + 16..HEADER_SIZE + 24].copy_from_slice(&data_offset.to_le_bytes());
    // disk_size (8 bytes)
    buf[HEADER_SIZE + 24..HEADER_SIZE + 32].copy_from_slice(&disk_size.to_le_bytes());
    // orig_size (8 bytes) — use the (possibly huge) fuzzed value to trigger DecompressTooLarge
    buf[HEADER_SIZE + 32..HEADER_SIZE + 40].copy_from_slice(&(input.orig_size as u64).to_le_bytes());
    // checksum (4 bytes) — deliberately wrong; we want to hit decompression paths, not CRC success
    buf[HEADER_SIZE + 40..HEADER_SIZE + 44].copy_from_slice(&0u32.to_le_bytes());
    // reserved (4 bytes) = 0

    // String table: 'e', '\0', 0, 0
    buf[st_offset as usize] = b'e';
    buf[st_offset as usize + 1] = 0;

    // Entry data
    if !input.data.is_empty() {
        let start = data_offset as usize;
        buf[start..start + input.data.len()].copy_from_slice(&input.data);
    }

    // Open the constructed shard — must never panic.
    if let Ok(reader) = shard_format::ShardReader::from_bytes(buf) {
        let _ = reader.read_entry(0);
        let _ = reader.read_entry_by_name("e");
    }
});
