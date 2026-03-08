//! Roundtrip and writer correctness tests for Shard v2.

use shard_format::{
    compute_crc32c, compute_xxhash64, ShardV2Reader, ShardV2Writer, ShardError,
    ALIGN_16, ALIGN_64, ALIGN_NONE, CONTENT_TYPE_BLOB, CONTENT_TYPE_JSON,
    CONTENT_TYPE_TENSOR, CONTENT_TYPE_TEXT,
    ENTRY_FLAG_COMPRESSED, FLAG_HAS_CHECKSUMS, FLAG_HAS_CONTENT_TYPES, FLAG_LITTLE_ENDIAN,
    HEADER_SIZE, INDEX_ENTRY_SIZE, ROLE_MOSH, ROLE_SAMPLE, SHARD_MAGIC, SHARD_VERSION2,
};

// ============================================================
// Helpers
// ============================================================

/// Build a shard in memory and immediately read it back.
fn roundtrip(writer: ShardV2Writer) -> ShardV2Reader {
    let bytes = writer.to_bytes();
    ShardV2Reader::from_bytes(bytes).expect("roundtrip: from_bytes failed")
}

// ============================================================
// Single entry roundtrip
// ============================================================

#[test]
fn test_single_entry_roundtrip() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("hello", b"world");
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 1);
    assert_eq!(r.entry_name(0).unwrap(), "hello");
    let data = r.read_entry(0).expect("read_entry");
    assert_eq!(data, b"world");
}

// ============================================================
// Multiple entries
// ============================================================

#[test]
fn test_multiple_entries_roundtrip() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("alpha", b"data_alpha");
    w.write_entry("beta", b"data_beta");
    w.write_entry("gamma", b"data_gamma_longer");
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 3);
    assert_eq!(r.read_entry(0).unwrap(), b"data_alpha");
    assert_eq!(r.read_entry(1).unwrap(), b"data_beta");
    assert_eq!(r.read_entry(2).unwrap(), b"data_gamma_longer");
}

// ============================================================
// Typed entries
// ============================================================

#[test]
fn test_typed_entries_roundtrip() {
    let mut w = ShardV2Writer::new(ROLE_SAMPLE);
    w.write_entry_typed("weights", &[0u8; 256], CONTENT_TYPE_TENSOR);
    w.write_entry_typed("config", b"{\"layers\":8}", CONTENT_TYPE_JSON);
    w.write_entry_typed("readme", b"hello", CONTENT_TYPE_TEXT);
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 3);
    assert_eq!(r.get_entry_info(0).unwrap().content_type(), CONTENT_TYPE_TENSOR);
    assert_eq!(r.get_entry_info(1).unwrap().content_type(), CONTENT_TYPE_JSON);
    assert_eq!(r.get_entry_info(2).unwrap().content_type(), CONTENT_TYPE_TEXT);

    assert_eq!(r.read_entry(0).unwrap(), &[0u8; 256]);
    assert_eq!(r.read_entry(1).unwrap(), b"{\"layers\":8}");
    assert_eq!(r.read_entry(2).unwrap(), b"hello");
}

// ============================================================
// Empty shard
// ============================================================

#[test]
fn test_empty_shard() {
    let w = ShardV2Writer::new(ROLE_MOSH);
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 0);
    assert_eq!(r.entry_names(), Vec::<&str>::new());
    assert_eq!(r.lookup("anything"), None);
    assert!(!r.has_entry("anything"));
}

// ============================================================
// Large entry
// ============================================================

#[test]
fn test_large_entry_roundtrip() {
    let big: Vec<u8> = (0..65536).map(|i: usize| (i % 256) as u8).collect();

    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("bigdata", &big);
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 1);
    assert_eq!(r.read_entry(0).unwrap(), big.as_slice());
}

// ============================================================
// Alignment: none
// ============================================================

#[test]
fn test_alignment_none() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_alignment(ALIGN_NONE);
    w.write_entry("a", b"12345");
    w.write_entry("b", b"678");
    let bytes = w.to_bytes();
    let r = ShardV2Reader::from_bytes(bytes.clone()).expect("from_bytes");

    assert_eq!(r.header().alignment, ALIGN_NONE);

    // Verify data section offset is NOT padded (no alignment gap expected).
    let hdr = r.header();
    // string table offset = HEADER_SIZE + 2*INDEX_ENTRY_SIZE
    let expected_str_off = HEADER_SIZE + 2 * INDEX_ENTRY_SIZE;
    assert_eq!(hdr.string_table_offset, expected_str_off as u64);
    // string table = "a\0b\0" = 4 bytes
    // data section offset = string_table_offset + 4 (no alignment padding)
    assert_eq!(
        hdr.data_section_offset,
        hdr.string_table_offset + 4,
        "no-align: data section should follow string table immediately"
    );

    assert_eq!(r.read_entry(0).unwrap(), b"12345");
    assert_eq!(r.read_entry(1).unwrap(), b"678");
}

// ============================================================
// Alignment: 16
// ============================================================

#[test]
fn test_alignment_16() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_alignment(ALIGN_16);
    w.write_entry("x", b"data");
    w.write_entry("y", b"more_data_here");
    let bytes = w.to_bytes();
    let r = ShardV2Reader::from_bytes(bytes).expect("from_bytes");

    assert_eq!(r.header().alignment, ALIGN_16);

    // data_section_offset must be 16-byte aligned
    let dso = r.header().data_section_offset as usize;
    assert_eq!(dso % 16, 0, "data_section_offset not 16-byte aligned: {}", dso);

    // Each entry's data_offset must be 16-byte aligned
    for i in 0..r.entry_count() {
        let info = r.get_entry_info(i).unwrap();
        assert_eq!(
            info.data_offset as usize % 16,
            0,
            "entry {} data_offset not 16-byte aligned: {}",
            i,
            info.data_offset
        );
    }

    assert_eq!(r.read_entry(0).unwrap(), b"data");
    assert_eq!(r.read_entry(1).unwrap(), b"more_data_here");
}

// ============================================================
// Alignment: 64
// ============================================================

#[test]
fn test_alignment_64() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_alignment(ALIGN_64);
    w.write_entry("tensor", &[0xAAu8; 512]);
    w.write_entry("bias", &[0xBBu8; 32]);
    let bytes = w.to_bytes();
    let r = ShardV2Reader::from_bytes(bytes).expect("from_bytes");

    assert_eq!(r.header().alignment, ALIGN_64);

    let dso = r.header().data_section_offset as usize;
    assert_eq!(dso % 64, 0, "data_section_offset not 64-byte aligned");

    for i in 0..r.entry_count() {
        let info = r.get_entry_info(i).unwrap();
        assert_eq!(
            info.data_offset as usize % 64,
            0,
            "entry {} data_offset not 64-byte aligned: {}",
            i,
            info.data_offset
        );
    }

    assert_eq!(r.read_entry(0).unwrap(), &[0xAAu8; 512]);
    assert_eq!(r.read_entry(1).unwrap(), &[0xBBu8; 32]);
}

// ============================================================
// Checksum verification
// ============================================================

#[test]
fn test_checksum_verification() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("data", b"hello world");
    let r = roundtrip(w);

    let info = r.get_entry_info(0).unwrap();
    let expected_crc = compute_crc32c(b"hello world");
    assert_eq!(info.checksum, expected_crc, "checksum field");
    assert_eq!(info.checksum, 3381945770, "checksum known value");

    // read_entry verifies checksum — should succeed
    r.read_entry(0).expect("checksum verification should pass");
}

#[test]
fn test_checksum_mismatch_detected() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("data", b"hello world");
    let mut bytes = w.to_bytes();

    // Corrupt the data byte — entry data starts at data_section_offset.
    // Parse the header to find the offset.
    let r_original = ShardV2Reader::from_bytes(bytes.clone()).unwrap();
    let data_off = r_original.get_entry_info(0).unwrap().data_offset as usize;
    bytes[data_off] ^= 0xFF; // flip bits to corrupt

    let r_corrupt = ShardV2Reader::from_bytes(bytes).expect("parse should succeed with corrupt data");
    let result = r_corrupt.read_entry(0);
    assert!(
        matches!(result, Err(ShardError::ChecksumMismatch { .. })),
        "expected ChecksumMismatch, got: {:?}",
        result
    );
}

// ============================================================
// Hash verification
// ============================================================

#[test]
fn test_name_hash_is_xxhash64() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("greeting", b"hello");
    w.write_entry("pattern/1k", &[0u8; 1024]);
    let r = roundtrip(w);

    assert_eq!(
        r.get_entry_info(0).unwrap().name_hash,
        compute_xxhash64("greeting"),
        "name_hash for 'greeting'"
    );
    assert_eq!(
        r.get_entry_info(1).unwrap().name_hash,
        compute_xxhash64("pattern/1k"),
        "name_hash for 'pattern/1k'"
    );
}

// ============================================================
// Lookup
// ============================================================

#[test]
fn test_lookup() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("foo", b"foo_data");
    w.write_entry("bar", b"bar_data");
    w.write_entry("baz/qux", b"nested");
    let r = roundtrip(w);

    assert_eq!(r.lookup("foo"), Some(0));
    assert_eq!(r.lookup("bar"), Some(1));
    assert_eq!(r.lookup("baz/qux"), Some(2));
    assert_eq!(r.lookup("missing"), None);
    assert!(r.has_entry("foo"));
    assert!(!r.has_entry("missing"));
}

// ============================================================
// Binary data: all 256 byte values
// ============================================================

#[test]
fn test_binary_all_byte_values() {
    let all_bytes: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();

    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("all_bytes", &all_bytes);
    let r = roundtrip(w);

    assert_eq!(r.read_entry(0).unwrap(), all_bytes.as_slice());
}

// ============================================================
// Header field correctness
// ============================================================

#[test]
fn test_header_magic_and_version() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("x", b"y");
    let bytes = w.to_bytes();
    let r = ShardV2Reader::from_bytes(bytes).unwrap();

    assert_eq!(&r.header().magic, SHARD_MAGIC);
    assert_eq!(r.header().version, SHARD_VERSION2);
    assert_eq!(r.header().index_entry_size, INDEX_ENTRY_SIZE as u16);
}

#[test]
fn test_header_role() {
    let w = ShardV2Writer::new(ROLE_SAMPLE);
    let r = roundtrip(w);
    assert_eq!(r.header().role, ROLE_SAMPLE);
}

#[test]
fn test_header_total_file_size() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("item", b"content");
    let bytes = w.to_bytes();
    let len = bytes.len() as u64;
    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.header().total_file_size, len);
}

#[test]
fn test_header_flags_with_content_types() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_typed("t", b"data", CONTENT_TYPE_TENSOR);
    let r = roundtrip(w);

    let flags = r.header().flags;
    assert!(flags & FLAG_LITTLE_ENDIAN != 0, "FLAG_LITTLE_ENDIAN");
    assert!(flags & FLAG_HAS_CHECKSUMS != 0, "FLAG_HAS_CHECKSUMS");
    assert!(flags & FLAG_HAS_CONTENT_TYPES != 0, "FLAG_HAS_CONTENT_TYPES");
}

#[test]
fn test_header_flags_without_content_types() {
    // When all entries use CONTENT_TYPE_UNKNOWN, FLAG_HAS_CONTENT_TYPES should not be set.
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("plain", b"data");
    let r = roundtrip(w);

    let flags = r.header().flags;
    assert!(flags & FLAG_LITTLE_ENDIAN != 0);
    assert!(flags & FLAG_HAS_CHECKSUMS != 0);
    assert_eq!(
        flags & FLAG_HAS_CONTENT_TYPES,
        0,
        "FLAG_HAS_CONTENT_TYPES should not be set when all entries are UNKNOWN"
    );
}

// ============================================================
// Entry details correctness
// ============================================================

#[test]
fn test_entry_sizes() {
    let data = b"twelve bytes";
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("item", data);
    let r = roundtrip(w);

    let info = r.get_entry_info(0).unwrap();
    assert_eq!(info.disk_size, data.len() as u64);
    assert_eq!(info.orig_size, data.len() as u64);
}

#[test]
fn test_entry_not_compressed() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("plain", b"uncompressed data");
    let r = roundtrip(w);

    let info = r.get_entry_info(0).unwrap();
    assert!(!info.compressed(), "plain entry should not be compressed");
    assert_eq!(info.flags & ENTRY_FLAG_COMPRESSED, 0);
}

// ============================================================
// list_prefix
// ============================================================

#[test]
fn test_list_prefix_basic() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("layer/a", b"");
    w.write_entry("layer/b", b"");
    w.write_entry("other/c", b"");
    let r = roundtrip(w);

    let layer = r.list_prefix("layer");
    assert_eq!(layer.len(), 2);
    assert!(layer.contains(&"layer/a"));
    assert!(layer.contains(&"layer/b"));

    let other = r.list_prefix("other");
    assert_eq!(other.len(), 1);
    assert!(other.contains(&"other/c"));

    let none = r.list_prefix("missing");
    assert!(none.is_empty());
}

#[test]
fn test_list_prefix_with_trailing_slash() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("a/x", b"");
    w.write_entry("a/y", b"");
    w.write_entry("ab/z", b""); // should NOT match prefix "a/"
    let r = roundtrip(w);

    let a = r.list_prefix("a");
    assert_eq!(a.len(), 2, "prefix 'a' should match a/x and a/y only");
    assert!(a.contains(&"a/x"));
    assert!(a.contains(&"a/y"));
    // "ab/z" starts with "ab/" not "a/", should not be included
    assert!(!a.contains(&"ab/z"));
}

// ============================================================
// read_entry_by_name
// ============================================================

#[test]
fn test_read_entry_by_name() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("key1", b"value1");
    w.write_entry("key2", b"value2");
    let r = roundtrip(w);

    assert_eq!(r.read_entry_by_name("key1").unwrap(), b"value1");
    assert_eq!(r.read_entry_by_name("key2").unwrap(), b"value2");

    let result = r.read_entry_by_name("missing");
    assert!(
        matches!(result, Err(ShardError::EntryNotFound(_))),
        "expected EntryNotFound"
    );
}

// ============================================================
// Invalid magic / version errors
// ============================================================

#[test]
fn test_invalid_magic_error() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("x", b"y");
    let mut bytes = w.to_bytes();
    bytes[0] = 0xFF; // corrupt magic

    let result = ShardV2Reader::from_bytes(bytes);
    assert!(
        matches!(result, Err(ShardError::InvalidMagic)),
        "expected InvalidMagic"
    );
}

#[test]
fn test_unsupported_version_error() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("x", b"y");
    let mut bytes = w.to_bytes();
    bytes[4] = 0x01; // corrupt version byte

    let result = ShardV2Reader::from_bytes(bytes);
    assert!(
        matches!(result, Err(ShardError::UnsupportedVersion(0x01))),
        "expected UnsupportedVersion(1)"
    );
}

// ============================================================
// Hierarchical paths
// ============================================================

#[test]
fn test_hierarchical_paths() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("layer.0/attention/q_proj/weight", &[1u8; 64]);
    w.write_entry("layer.0/attention/k_proj/weight", &[2u8; 64]);
    w.write_entry("layer.0/ffn/gate/weight", &[3u8; 128]);
    w.write_entry("layer.0/norm", &[4u8; 16]);
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 4);

    let attn = r.list_prefix("layer.0/attention");
    assert_eq!(attn.len(), 2);

    let ffn = r.list_prefix("layer.0/ffn");
    assert_eq!(ffn.len(), 1);

    let all = r.list_prefix("layer.0");
    assert_eq!(all.len(), 4);

    assert_eq!(r.read_entry_by_name("layer.0/norm").unwrap(), &[4u8; 16]);
}

// ============================================================
// Empty entry (zero bytes)
// ============================================================

#[test]
fn test_empty_entry_data() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("empty", b"");
    let r = roundtrip(w);

    assert_eq!(r.entry_count(), 1);
    let info = r.get_entry_info(0).unwrap();
    assert_eq!(info.disk_size, 0);
    assert_eq!(info.orig_size, 0);
    assert_eq!(info.checksum, 0, "crc32c of empty is 0");
    let data = r.read_entry(0).unwrap();
    assert!(data.is_empty());
}

// ============================================================
// Blob content type
// ============================================================

#[test]
fn test_blob_content_type() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_typed("raw", &[0u8; 100], CONTENT_TYPE_BLOB);
    let r = roundtrip(w);

    assert_eq!(r.get_entry_info(0).unwrap().content_type(), CONTENT_TYPE_BLOB);
}

// ============================================================
// Index offsets sanity
// ============================================================

#[test]
fn test_string_table_offset_is_after_index() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("a", b"1");
    w.write_entry("bb", b"22");
    w.write_entry("ccc", b"333");
    let r = roundtrip(w);

    let h = r.header();
    let expected_str_off = (HEADER_SIZE + 3 * INDEX_ENTRY_SIZE) as u64;
    assert_eq!(h.string_table_offset, expected_str_off);
}

#[test]
fn test_data_offsets_within_file() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("x", &[0u8; 100]);
    w.write_entry("y", &[0u8; 200]);
    let bytes = w.to_bytes();
    let file_size = bytes.len() as u64;
    let r = ShardV2Reader::from_bytes(bytes).unwrap();

    for i in 0..r.entry_count() {
        let info = r.get_entry_info(i).unwrap();
        assert!(
            info.data_offset + info.disk_size <= file_size,
            "entry {} data extends past file end",
            i
        );
    }
}
