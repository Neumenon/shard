//! Deterministic seed tests that exercise the same code paths as the five
//! fuzz targets.  These run under `cargo test` without cargo-fuzz and serve
//! two purposes:
//!   1. Verify the fuzz targets' logic compiles and does not panic on
//!      canonical edge-case inputs.
//!   2. Provide regression coverage for any crash found by the fuzzer
//!      (paste the minimised input as a new test case here).
//!
//! Matches Go fuzz seeds for:
//!   FuzzReadShardHeader, FuzzParseIndexEntry, FuzzOpenShard,
//!   FuzzDecompressData, FuzzReadEntryByName.

use shard_format::{
    ShardHeader, ShardReader, ShardWriter,
    ENTRY_FLAG_COMPRESSED, ENTRY_FLAG_LZ4, ENTRY_FLAG_ZSTD,
    FLAG_HAS_CHECKSUMS,
    HEADER_SIZE, INDEX_ENTRY_SIZE, ROLE_MOSH,
    SHARD_MAGIC, SHARD_VERSION,
};

// ============================================================
// Helpers
// ============================================================

/// Build a minimal syntactically-valid header buffer.
/// All fields except those in `override_fn` hold the writer defaults for an
/// empty shard (entry_count=0).
fn make_valid_header_buf() -> [u8; 64] {
    let mut buf = [0u8; 64];
    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = ROLE_MOSH;
    buf[6..8].copy_from_slice(&0x00A2u16.to_le_bytes()); // DEFAULT_FLAGS
    buf[8] = 0;  // ALIGN_NONE
    buf[9] = 0;  // COMPRESS_NONE
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes()); // entry_count=0
    buf[16..24].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // string_table_offset=64
    buf[24..32].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // data_section_offset=64
    buf[32..40].copy_from_slice(&0u64.to_le_bytes()); // schema_offset=0
    buf[40..48].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // total_file_size=64
    buf
}

// ============================================================
// FuzzReadShardHeader seeds
// ============================================================

#[test]
fn test_fuzz_header_empty() {
    assert!(ShardHeader::from_bytes(&[]).is_err());
}

#[test]
fn test_fuzz_header_too_short() {
    for len in [1usize, 4, 10, 32, 63] {
        let buf = vec![0u8; len];
        assert!(
            ShardHeader::from_bytes(&buf).is_err(),
            "expected error for len={len}"
        );
    }
}

#[test]
fn test_fuzz_header_bad_magic_all_zeros() {
    let buf = [0u8; 64];
    assert!(ShardHeader::from_bytes(&buf).is_err());
}

#[test]
fn test_fuzz_header_all_0xff() {
    let buf = [0xFFu8; 64];
    assert!(ShardHeader::from_bytes(&buf).is_err());
}

#[test]
fn test_fuzz_header_valid_magic_wrong_version() {
    let mut buf = [0u8; 64];
    buf[0..4].copy_from_slice(b"SHRD");
    buf[4] = 0xFF; // unsupported version
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    assert!(ShardHeader::from_bytes(&buf).is_err());
}

#[test]
fn test_fuzz_header_wrong_index_entry_size() {
    let mut buf = make_valid_header_buf();
    buf[10..12].copy_from_slice(&0xFFFFu16.to_le_bytes()); // wrong index_entry_size
    assert!(ShardHeader::from_bytes(&buf).is_err());
}

#[test]
fn test_fuzz_header_zero_index_entry_size() {
    let mut buf = make_valid_header_buf();
    buf[10..12].copy_from_slice(&0u16.to_le_bytes());
    assert!(ShardHeader::from_bytes(&buf).is_err());
}

#[test]
fn test_fuzz_header_max_entry_count() {
    // entry_count = u32::MAX — no panic; just parsed (no validation at header level)
    let mut buf = make_valid_header_buf();
    buf[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    // from_bytes on just 64 bytes succeeds — entry_count is not validated here
    // (that happens in ShardReader::from_bytes)
    let _ = ShardHeader::from_bytes(&buf);
}

#[test]
fn test_fuzz_header_valid_parses_ok() {
    let buf = make_valid_header_buf();
    let hdr = ShardHeader::from_bytes(&buf).expect("valid header should parse");
    assert_eq!(&hdr.magic, SHARD_MAGIC);
    assert_eq!(hdr.version, SHARD_VERSION);
    assert_eq!(hdr.index_entry_size as usize, INDEX_ENTRY_SIZE);
}

// ============================================================
// FuzzParseIndexEntry seeds
// ============================================================
//
// The internal IndexEntry::from_bytes is private, so we exercise it through
// the ShardReader path (same strategy as fuzz_index_entry.rs).

/// Build a minimal shard whose single index entry is the given 48-byte slice.
fn shard_with_raw_entry(entry_bytes: &[u8; 48]) -> Vec<u8> {
    const ST_OFFSET: usize = HEADER_SIZE + INDEX_ENTRY_SIZE; // 112
    let mut buf = vec![0u8; ST_OFFSET];

    // Header
    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = ROLE_MOSH;
    buf[6..8].copy_from_slice(&FLAG_HAS_CHECKSUMS.to_le_bytes());
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&1u32.to_le_bytes()); // entry_count=1
    buf[16..24].copy_from_slice(&(ST_OFFSET as u64).to_le_bytes()); // string_table_offset
    buf[24..32].copy_from_slice(&(ST_OFFSET as u64).to_le_bytes()); // data_section_offset
    buf[40..48].copy_from_slice(&(ST_OFFSET as u64).to_le_bytes()); // total_file_size

    // Index entry
    buf[HEADER_SIZE..HEADER_SIZE + INDEX_ENTRY_SIZE].copy_from_slice(entry_bytes);
    buf
}

#[test]
fn test_fuzz_index_entry_all_zeros() {
    let entry = [0u8; 48];
    let buf = shard_with_raw_entry(&entry);
    // The entry has data_offset=0 which is < data_section_offset (112), so
    // the reader should reject it with EntryOffsetOutOfRange.
    let _ = ShardReader::from_bytes(buf);
}

#[test]
fn test_fuzz_index_entry_all_0xff() {
    let entry = [0xFFu8; 48];
    let buf = shard_with_raw_entry(&entry);
    // Extremely large data_offset / disk_size — reader must not panic.
    let _ = ShardReader::from_bytes(buf);
}

#[test]
fn test_fuzz_index_entry_compressed_flags() {
    let mut entry = [0u8; 48];
    // Set flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD
    let flags: u16 = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
    entry[14..16].copy_from_slice(&flags.to_le_bytes());
    // data_offset = ST_OFFSET (= 112), disk_size = 0 → valid, but decompression will fail
    let st: u64 = (HEADER_SIZE + INDEX_ENTRY_SIZE) as u64;
    entry[16..24].copy_from_slice(&st.to_le_bytes()); // data_offset
    // disk_size=0, orig_size=0
    let buf = shard_with_raw_entry(&entry);
    if let Ok(reader) = ShardReader::from_bytes(buf) {
        let info = reader.get_entry_info(0).unwrap();
        assert!(info.compressed());
        // Attempt read — decompression of empty bytes will fail gracefully.
        let _ = reader.read_entry(0);
    }
}

#[test]
fn test_fuzz_index_entry_chunked_flag() {
    let mut entry = [0u8; 48];
    let flags: u16 = shard_format::ENTRY_FLAG_CHUNKED;
    entry[14..16].copy_from_slice(&flags.to_le_bytes());
    // data_offset = 112, disk_size = 0
    let st: u64 = (HEADER_SIZE + INDEX_ENTRY_SIZE) as u64;
    entry[16..24].copy_from_slice(&st.to_le_bytes());
    let buf = shard_with_raw_entry(&entry);
    // Reader should accept this (chunked flag doesn't block reads in current impl).
    let _ = ShardReader::from_bytes(buf);
}

#[test]
fn test_fuzz_index_entry_max_name_offset() {
    let mut entry = [0u8; 48];
    // name_offset = u32::MAX, name_len = 1 — string table is empty, so name
    // resolution should silently produce an empty string.
    entry[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    entry[12..14].copy_from_slice(&1u16.to_le_bytes());
    let st: u64 = (HEADER_SIZE + INDEX_ENTRY_SIZE) as u64;
    entry[16..24].copy_from_slice(&st.to_le_bytes()); // data_offset=112
    let buf = shard_with_raw_entry(&entry);
    let _ = ShardReader::from_bytes(buf);
}

// ============================================================
// FuzzOpenShard seeds
// ============================================================

#[test]
fn test_fuzz_open_empty_bytes() {
    assert!(ShardReader::from_bytes(vec![]).is_err());
}

#[test]
fn test_fuzz_open_only_header_zero_entries() {
    let buf = make_valid_header_buf().to_vec();
    // total_file_size=64, entry_count=0 → should open successfully.
    let r = ShardReader::from_bytes(buf).expect("empty shard with 0 entries should open");
    assert_eq!(r.entry_count(), 0);
    assert!(r.entry_names().is_empty());
    assert_eq!(r.lookup("x"), None);
    assert!(!r.has_entry("x"));
    assert!(r.list_prefix("").is_empty());
    assert!(r.list_children("").is_empty());
}

#[test]
fn test_fuzz_open_inverted_offsets() {
    // data_section_offset < string_table_offset — invalid layout.
    let mut buf = make_valid_header_buf().to_vec();
    // string_table_offset = 200, data_section_offset = 100
    buf[16..24].copy_from_slice(&200u64.to_le_bytes());
    buf[24..32].copy_from_slice(&100u64.to_le_bytes());
    buf[40..48].copy_from_slice(&200u64.to_le_bytes()); // total_file_size=200 (still mismatch with vec len)
    let _ = ShardReader::from_bytes(buf);
}

#[test]
fn test_fuzz_open_file_size_mismatch() {
    let mut buf = make_valid_header_buf().to_vec();
    // total_file_size claims 9999 but actual bytes = 64
    buf[40..48].copy_from_slice(&9999u64.to_le_bytes());
    assert!(ShardReader::from_bytes(buf).is_err());
}

#[test]
fn test_fuzz_open_max_entry_count_exceeds_security_limit() {
    let mut buf = make_valid_header_buf().to_vec();
    // entry_count = 10_000_001 (> MAX_ENTRY_COUNT=10_000_000)
    buf[12..16].copy_from_slice(&10_000_001u32.to_le_bytes());
    buf[40..48].copy_from_slice(&64u64.to_le_bytes()); // total_file_size matches vec
    assert!(ShardReader::from_bytes(buf).is_err());
}

#[test]
fn test_fuzz_open_all_zeros() {
    // 64 bytes of zeros — bad magic.
    assert!(ShardReader::from_bytes(vec![0u8; 64]).is_err());
}

#[test]
fn test_fuzz_open_truncated_mid_index() {
    // Valid header claiming 5 entries, but file is only 64 bytes long.
    let mut buf = make_valid_header_buf().to_vec();
    buf[12..16].copy_from_slice(&5u32.to_le_bytes()); // entry_count=5
    // total_file_size must match actual len=64
    buf[40..48].copy_from_slice(&64u64.to_le_bytes());
    // Reader should reject: file too small for index.
    assert!(ShardReader::from_bytes(buf).is_err());
}

// ============================================================
// FuzzDecompressData seeds
// ============================================================

#[test]
fn test_fuzz_decompress_zstd_empty() {
    let result = zstd::decode_all(&[][..]);
    // Empty input is not valid zstd — expect an error.
    assert!(result.is_err());
}

#[test]
fn test_fuzz_decompress_zstd_garbage() {
    let garbage = b"\xFF\xFE\xFD\xFC\x00\x01\x02\x03";
    let _ = zstd::decode_all(&garbage[..]);
}

#[test]
fn test_fuzz_decompress_lz4_empty() {
    let _ = lz4_flex::decompress(&[], 0);
}

#[test]
fn test_fuzz_decompress_lz4_garbage() {
    let garbage = b"\xFF\xFE\xFD\xFC\x00\x01\x02\x03";
    // Various orig_size values including huge ones.
    for &orig in &[0usize, 1, 1024, 1 << 20] {
        let _ = lz4_flex::decompress(garbage, orig);
    }
}

#[test]
fn test_fuzz_decompress_via_reader_zstd_garbage() {
    // Build a shard with ZSTD-flagged entry containing garbage data.
    // ShardReader::read_entry should return Err, not panic.
    let garbage = b"\x00\x01\x02\x03\x04\x05\x06\x07";
    let st_offset: usize = HEADER_SIZE + INDEX_ENTRY_SIZE; // 112
    let data_offset: usize = st_offset + 4; // 116
    let total: usize = data_offset + garbage.len();
    let mut buf = vec![0u8; total];

    // Header
    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = ROLE_MOSH;
    buf[6..8].copy_from_slice(&FLAG_HAS_CHECKSUMS.to_le_bytes());
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&1u32.to_le_bytes());
    buf[16..24].copy_from_slice(&(st_offset as u64).to_le_bytes());
    buf[24..32].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[40..48].copy_from_slice(&(total as u64).to_le_bytes());

    // Index entry: ZSTD-compressed, orig_size=1024
    let flags: u16 = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
    buf[HEADER_SIZE + 14..HEADER_SIZE + 16].copy_from_slice(&flags.to_le_bytes());
    buf[HEADER_SIZE + 16..HEADER_SIZE + 24].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[HEADER_SIZE + 24..HEADER_SIZE + 32]
        .copy_from_slice(&(garbage.len() as u64).to_le_bytes());
    buf[HEADER_SIZE + 32..HEADER_SIZE + 40].copy_from_slice(&1024u64.to_le_bytes());

    // name = 'e'
    buf[st_offset] = b'e';
    buf[st_offset + 1] = 0;
    buf[HEADER_SIZE + 12..HEADER_SIZE + 14].copy_from_slice(&1u16.to_le_bytes()); // name_len=1

    // Data
    buf[data_offset..data_offset + garbage.len()].copy_from_slice(garbage);

    if let Ok(reader) = ShardReader::from_bytes(buf) {
        assert!(reader.read_entry(0).is_err(), "should fail decompression");
    }
}

#[test]
fn test_fuzz_decompress_via_reader_lz4_garbage() {
    let garbage = b"\x00\x01\x02\x03\x04\x05\x06\x07";
    let st_offset: usize = HEADER_SIZE + INDEX_ENTRY_SIZE;
    let data_offset: usize = st_offset + 4;
    let total: usize = data_offset + garbage.len();
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = ROLE_MOSH;
    buf[6..8].copy_from_slice(&FLAG_HAS_CHECKSUMS.to_le_bytes());
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&1u32.to_le_bytes());
    buf[16..24].copy_from_slice(&(st_offset as u64).to_le_bytes());
    buf[24..32].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[40..48].copy_from_slice(&(total as u64).to_le_bytes());

    let flags: u16 = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4;
    buf[HEADER_SIZE + 14..HEADER_SIZE + 16].copy_from_slice(&flags.to_le_bytes());
    buf[HEADER_SIZE + 16..HEADER_SIZE + 24].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[HEADER_SIZE + 24..HEADER_SIZE + 32]
        .copy_from_slice(&(garbage.len() as u64).to_le_bytes());
    buf[HEADER_SIZE + 32..HEADER_SIZE + 40].copy_from_slice(&512u64.to_le_bytes()); // orig_size

    buf[st_offset] = b'e';
    buf[st_offset + 1] = 0;
    buf[HEADER_SIZE + 12..HEADER_SIZE + 14].copy_from_slice(&1u16.to_le_bytes());

    buf[data_offset..data_offset + garbage.len()].copy_from_slice(garbage);

    if let Ok(reader) = ShardReader::from_bytes(buf) {
        assert!(reader.read_entry(0).is_err(), "should fail lz4 decompression");
    }
}

#[test]
fn test_fuzz_decompress_huge_orig_size_triggers_limit() {
    // orig_size > MAX_DECOMPRESS_SIZE (1 GB) should return DecompressTooLarge.
    let garbage = b"\x28\xB5\x2F\xFD"; // valid zstd magic, but truncated
    let st_offset: usize = HEADER_SIZE + INDEX_ENTRY_SIZE;
    let data_offset: usize = st_offset + 4;
    let total: usize = data_offset + garbage.len();
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&1u32.to_le_bytes());
    buf[16..24].copy_from_slice(&(st_offset as u64).to_le_bytes());
    buf[24..32].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[40..48].copy_from_slice(&(total as u64).to_le_bytes());

    let flags: u16 = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
    buf[HEADER_SIZE + 14..HEADER_SIZE + 16].copy_from_slice(&flags.to_le_bytes());
    buf[HEADER_SIZE + 16..HEADER_SIZE + 24].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[HEADER_SIZE + 24..HEADER_SIZE + 32]
        .copy_from_slice(&(garbage.len() as u64).to_le_bytes());
    // orig_size = 2 GB > MAX_DECOMPRESS_SIZE (1 GB)
    let huge_orig: u64 = 2 * (1 << 30);
    buf[HEADER_SIZE + 32..HEADER_SIZE + 40].copy_from_slice(&huge_orig.to_le_bytes());

    buf[st_offset] = b'e';
    buf[st_offset + 1] = 0;
    buf[HEADER_SIZE + 12..HEADER_SIZE + 14].copy_from_slice(&1u16.to_le_bytes());
    buf[data_offset..data_offset + garbage.len()].copy_from_slice(garbage);

    if let Ok(reader) = ShardReader::from_bytes(buf) {
        let err = reader.read_entry(0).unwrap_err();
        assert!(
            matches!(err, shard_format::ShardError::DecompressTooLarge(_)),
            "expected DecompressTooLarge, got: {:?}", err
        );
    }
}

#[test]
fn test_fuzz_decompress_zstd_declared_orig_size_mismatch_is_bounded() {
    let payload = vec![b'A'; 4096];
    let compressed = zstd::stream::encode_all(&payload[..], 1).expect("zstd encode");

    let st_offset: usize = HEADER_SIZE + INDEX_ENTRY_SIZE;
    let data_offset: usize = st_offset + 4;
    let total: usize = data_offset + compressed.len();
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = ROLE_MOSH;
    buf[6..8].copy_from_slice(&FLAG_HAS_CHECKSUMS.to_le_bytes());
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&1u32.to_le_bytes());
    buf[16..24].copy_from_slice(&(st_offset as u64).to_le_bytes());
    buf[24..32].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[40..48].copy_from_slice(&(total as u64).to_le_bytes());

    let flags: u16 = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
    buf[HEADER_SIZE + 14..HEADER_SIZE + 16].copy_from_slice(&flags.to_le_bytes());
    buf[HEADER_SIZE + 16..HEADER_SIZE + 24].copy_from_slice(&(data_offset as u64).to_le_bytes());
    buf[HEADER_SIZE + 24..HEADER_SIZE + 32]
        .copy_from_slice(&(compressed.len() as u64).to_le_bytes());
    buf[HEADER_SIZE + 32..HEADER_SIZE + 40].copy_from_slice(&32u64.to_le_bytes());

    buf[st_offset] = b'e';
    buf[st_offset + 1] = 0;
    buf[HEADER_SIZE + 12..HEADER_SIZE + 14].copy_from_slice(&1u16.to_le_bytes());
    buf[data_offset..data_offset + compressed.len()].copy_from_slice(&compressed);

    let reader = ShardReader::from_bytes(buf).expect("reader");
    let err = reader.read_entry(0).unwrap_err();
    match err {
        shard_format::ShardError::DecompressTooLarge(size) => assert_eq!(size, 33),
        other => panic!("expected DecompressTooLarge, got {:?}", other),
    }
}

// ============================================================
// FuzzReadEntryByName seeds
// ============================================================

#[test]
fn test_fuzz_lookup_empty_name() {
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("normal", b"data");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();
    assert_eq!(r.lookup(""), None);
    assert!(!r.has_entry(""));
    assert!(r.read_entry_by_name("").is_err());
}

#[test]
fn test_fuzz_lookup_null_byte_name() {
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("normal", b"data");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();
    assert_eq!(r.lookup("\x00"), None);
    assert!(r.read_entry_by_name("\x00").is_err());
}

#[test]
fn test_fuzz_lookup_very_long_name() {
    let long_name = "x".repeat(4096);
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("normal", b"data");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();
    assert_eq!(r.lookup(&long_name), None);
}

#[test]
fn test_fuzz_lookup_unicode_names() {
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("unicode-café", b"data1");
    w.write_entry("日本語/テスト", b"data2");
    w.write_entry("emoji-\u{1F600}", b"data3");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();

    assert!(r.lookup("unicode-café").is_some());
    assert!(r.lookup("日本語/テスト").is_some());
    assert!(r.lookup("emoji-\u{1F600}").is_some());
    assert_eq!(r.lookup("unicode-cafe"), None); // ASCII lookalike
    assert_eq!(r.lookup("nonexistent"), None);
}

#[test]
fn test_fuzz_lookup_deep_path() {
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("a/b/c/d/e/f/g", b"deep");
    w.write_entry("a/b/c/d/e/f/h", b"deep2");
    w.write_entry("a/b/x", b"shallow");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();

    assert!(r.lookup("a/b/c/d/e/f/g").is_some());
    assert!(r.lookup("a/b/c/d/e/f/h").is_some());
    assert!(r.lookup("a/b/x").is_some());
    assert_eq!(r.lookup("nonexistent/deep/path/name"), None);

    // Prefix listing
    let prefix_ab = r.list_prefix("a/b");
    assert!(prefix_ab.iter().any(|&n| n == "a/b/x"));

    let children_ab = r.list_children("a/b/");
    assert!(children_ab.iter().any(|n| n == "a/b/c/"));
    assert!(children_ab.iter().any(|n| n == "a/b/x"));
}

#[test]
fn test_fuzz_lookup_prefix_is_also_entry() {
    // "layer.0" is both an exact entry AND a prefix of "layer.0/weight".
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("layer.0", b"direct");
    w.write_entry("layer.0/weight", b"weight");
    w.write_entry("layer.0/bias", b"bias");
    w.write_entry("layer.1/weight", b"w2");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();

    assert!(r.lookup("layer.0").is_some());
    assert!(r.lookup("layer.0/weight").is_some());

    let prefixed = r.list_prefix("layer.0");
    // "layer.0" itself and "layer.0/weight", "layer.0/bias"
    assert!(prefixed.contains(&"layer.0"));
    assert!(prefixed.contains(&"layer.0/weight"));
    assert!(!prefixed.contains(&"layer.1/weight"));

    let children = r.list_children("layer.");
    // Should see "layer.0/" and "layer.0" (exact match) and "layer.1/"
    assert!(children.iter().any(|n| n.starts_with("layer.0")));
    assert!(children.iter().any(|n| n.starts_with("layer.1")));
}

#[test]
fn test_fuzz_lookup_adversarial_names_no_panic() {
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("normal", b"data");
    w.write_entry("with/slashes/deep", b"data2");
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();

    // None of these should panic.
    let _ = r.lookup("");
    let _ = r.lookup("\x00");
    let _ = r.lookup("\u{FFFE}");
    let _ = r.lookup(&"/".repeat(1000));
    let _ = r.lookup(&"a".repeat(65536));
    let _ = r.list_prefix("");
    let _ = r.list_prefix("/");
    let _ = r.list_prefix(&"/".repeat(100));
    let _ = r.list_children("");
    let _ = r.list_children("/");
}

#[test]
fn test_fuzz_lookup_many_entries_writer_reader_roundtrip() {
    // Build a shard with 50 entries using various path patterns.
    let mut w = ShardWriter::new(ROLE_MOSH);
    for i in 0..50 {
        let name = format!("layer.{}/weight", i);
        w.write_entry(&name, format!("data_{}", i).as_bytes());
    }
    let bytes = w.to_bytes();
    let r = ShardReader::from_bytes(bytes).unwrap();

    assert_eq!(r.entry_count(), 50);

    for i in 0..50 {
        let name = format!("layer.{}/weight", i);
        assert!(r.lookup(&name).is_some(), "missing entry: {name}");
        assert!(r.has_entry(&name));
        let data = r.read_entry_by_name(&name).unwrap();
        assert_eq!(data, format!("data_{}", i).as_bytes());
    }

    // list_prefix("layer.0") matches entries whose name starts with "layer.0/"
    // or equals "layer.0" exactly.  Entry "layer.0/weight" starts with "layer.0/".
    let layer0 = r.list_prefix("layer.0");
    assert_eq!(layer0, vec!["layer.0/weight"]);

    // list_prefix("layer") matches entries starting with "layer/" or equal "layer".
    // Our entries are "layer.N/weight" — none start with "layer/" (they have a dot),
    // so we exercise the prefix API with a prefix that produces some matches.
    // Use entry_names() to verify all 50 entries round-tripped correctly.
    let all_names = r.entry_names();
    assert_eq!(all_names.len(), 50);

    // Children of root: all entries are "layer.N/weight", so each "layer.N/"
    // should appear as a child.
    let children = r.list_children("");
    assert!(!children.is_empty());
}
