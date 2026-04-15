//! Memory-mapped reader tests for Shard.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use shard_format::{
    MmapShardReader, ShardError, ShardWriter,
    ROLE_MOSH, FLAG_HAS_CHECKSUMS, HEADER_SIZE,
};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

// ============================================================
// Helpers
// ============================================================

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

fn testdata_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    Path::new(&manifest).join("../testdata")
}

// ============================================================
// Manifest deserialization (mirrors golden_test.rs)
// ============================================================

#[derive(Deserialize)]
struct Manifest {
    files: Vec<GoldenFile>,
}

#[derive(Deserialize)]
struct GoldenFile {
    filename: String,
    entries: Vec<GoldenEntry>,
}

#[derive(Deserialize)]
struct GoldenEntry {
    name: String,
    orig_size: u64,
    disk_size: u64,
    checksum: u32,
    compressed: bool,
    data_sha256: String,
}

fn load_manifest() -> Manifest {
    let path = testdata_dir().join("golden_manifest.json");
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse manifest: {}", e))
}

// ============================================================
// Basic open / read tests
// ============================================================

#[test]
fn test_mmap_basic_read() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("hello", b"world");
    w.write_entry("foo", b"bar");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    assert_eq!(r.entry_count(), 2);
    assert_eq!(r.entry_name(0).unwrap(), "hello");
    assert_eq!(r.entry_name(1).unwrap(), "foo");
    assert_eq!(r.read_entry(0).unwrap(), b"world");
    assert_eq!(r.read_entry(1).unwrap(), b"bar");
}

#[test]
fn test_mmap_entry_count() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    for i in 0..10u32 {
        w.write_entry(&format!("entry/{}", i), &i.to_le_bytes());
    }
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    assert_eq!(r.entry_count(), 10);
}

#[test]
fn test_mmap_lookup() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("alpha", b"aaa");
    w.write_entry("beta", b"bbb");
    w.write_entry("gamma", b"ccc");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();

    assert_eq!(r.lookup("alpha"), Some(0));
    assert_eq!(r.lookup("beta"), Some(1));
    assert_eq!(r.lookup("gamma"), Some(2));
    assert_eq!(r.lookup("__missing__"), None);

    assert!(r.has_entry("alpha"));
    assert!(!r.has_entry("delta"));
}

#[test]
fn test_mmap_read_entry_by_name() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("key", b"value");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let data = r.read_entry_by_name("key").unwrap();
    assert_eq!(data, b"value");

    let err = r.read_entry_by_name("missing");
    assert!(matches!(err, Err(shard_format::ShardError::EntryNotFound(_))));
}

#[test]
fn test_mmap_entry_info() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let payload: Vec<u8> = (0u8..=255).collect();

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("data", &payload);
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let info = r.get_entry_info(0).unwrap();

    assert_eq!(info.disk_size, 256);
    assert_eq!(info.orig_size, 256);
    assert!(!info.compressed());

    let data = r.read_entry(0).unwrap();
    let computed = shard_format::compute_crc32c(data);
    assert_eq!(computed, info.checksum);
}

#[test]
fn test_mmap_large_entry() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    // 1 MB of pseudo-random-ish data
    let large: Vec<u8> = (0u32..262_144)
        .flat_map(|i| i.to_le_bytes())
        .collect();

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("big", &large);
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let data = r.read_entry(0).unwrap();
    assert_eq!(data.len(), large.len());
    assert_eq!(data, large.as_slice());
}

#[test]
fn test_mmap_empty_entry() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("empty", b"");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    assert_eq!(r.entry_count(), 1);
    let data = r.read_entry(0).unwrap();
    assert_eq!(data, b"");
}

#[test]
fn test_mmap_header_matches_regular_reader() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("a", b"hello");
    w.write_entry("b", b"world");
    w.write_to_file(&path).unwrap();

    let regular = shard_format::ShardReader::from_file(&path).unwrap();
    let mmap = MmapShardReader::open(&path).unwrap();

    let rh = regular.header();
    let mh = mmap.header();
    assert_eq!(rh.version, mh.version);
    assert_eq!(rh.role, mh.role);
    assert_eq!(rh.flags, mh.flags);
    assert_eq!(rh.alignment, mh.alignment);
    assert_eq!(rh.entry_count, mh.entry_count);
    assert_eq!(rh.string_table_offset, mh.string_table_offset);
    assert_eq!(rh.data_section_offset, mh.data_section_offset);
    assert_eq!(rh.total_file_size, mh.total_file_size);
}

#[test]
fn test_mmap_entry_names() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("layer.0/weight", b"w");
    w.write_entry("layer.0/bias", b"b");
    w.write_entry("config", b"{}");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let names = r.entry_names();
    assert_eq!(names, vec!["layer.0/weight", "layer.0/bias", "config"]);
}

#[test]
fn test_mmap_list_prefix() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("layer.0/weight", b"w0");
    w.write_entry("layer.0/bias", b"b0");
    w.write_entry("layer.1/weight", b"w1");
    w.write_entry("config", b"{}");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();

    let mut l0: Vec<&str> = r.list_prefix("layer.0");
    l0.sort();
    assert_eq!(l0, vec!["layer.0/bias", "layer.0/weight"]);

    // list_prefix("layer.0") matches all entries under "layer.0/"
    let mut l0_again: Vec<&str> = r.list_prefix("layer.0");
    l0_again.sort();
    assert_eq!(l0_again, vec!["layer.0/bias", "layer.0/weight"]);

    let empty: Vec<&str> = r.list_prefix("nonexistent");
    assert!(empty.is_empty());
}

#[test]
fn test_mmap_list_children() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("layer.0/weight", b"w0");
    w.write_entry("layer.0/bias", b"b0");
    w.write_entry("layer.1/weight", b"w1");
    w.write_entry("config", b"{}");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();

    let mut top = r.list_children("");
    top.sort();
    assert_eq!(top, vec!["config", "layer.0/", "layer.1/"]);

    let mut l0 = r.list_children("layer.0/");
    l0.sort();
    assert_eq!(l0, vec!["layer.0/bias", "layer.0/weight"]);
}

#[test]
fn test_mmap_decompressed_uncompressed_entry() {
    // read_entry_decompressed on an uncompressed entry should return a copy of the data.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("plain", b"plaintext");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let data = r.read_entry_decompressed(0).unwrap();
    assert_eq!(data, b"plaintext");
}

#[test]
fn test_mmap_compressed_entry_zstd() {
    // read_entry on compressed returns error; read_entry_decompressed succeeds.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_zstd.shard");

    // Build a shard with zstd compression
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.set_compression(shard_format::COMPRESS_ZSTD);
    // Use a large repetitive payload so compression actually triggers
    let payload: Vec<u8> = vec![0xABu8; 1024];
    w.write_entry_compressed("compressed_entry", &payload);
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let info = r.get_entry_info(0).unwrap();

    if info.compressed() {
        // Zero-copy read should return error for compressed entries
        let zero_copy = r.read_entry(0);
        assert!(matches!(
            zero_copy,
            Err(shard_format::ShardError::CompressionNotSupported)
        ));

        // Decompressed read should succeed and return original payload
        let decompressed = r.read_entry_decompressed(0).unwrap();
        assert_eq!(decompressed, payload);
    } else {
        // Data wasn't compressible enough — both methods should work
        let via_zero_copy = r.read_entry(0).unwrap();
        let via_decompress = r.read_entry_decompressed(0).unwrap();
        assert_eq!(via_zero_copy, via_decompress.as_slice());
    }
}

#[test]
fn test_mmap_compressed_entry_lz4() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_lz4.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.set_compression(shard_format::COMPRESS_LZ4);
    let payload: Vec<u8> = vec![0xCDu8; 1024];
    w.write_entry_compressed("lz4_entry", &payload);
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let info = r.get_entry_info(0).unwrap();

    if info.compressed() {
        let err = r.read_entry(0);
        assert!(matches!(
            err,
            Err(shard_format::ShardError::CompressionNotSupported)
        ));

        let decompressed = r.read_entry_decompressed(0).unwrap();
        assert_eq!(decompressed, payload);
    } else {
        let via_zero_copy = r.read_entry(0).unwrap();
        let via_decompress = r.read_entry_decompressed(0).unwrap();
        assert_eq!(via_zero_copy, via_decompress.as_slice());
    }
}

#[test]
fn test_mmap_zstd_declared_orig_size_is_enforced() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bounded_zstd.shard");

    let payload = vec![0xAAu8; 4096];
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.set_compression(shard_format::COMPRESS_ZSTD);
    w.write_entry_compressed("bomb", &payload);
    w.write_to_file(&path).unwrap();

    let info = MmapShardReader::open(&path).unwrap().get_entry_info(0).unwrap().clone();
    assert!(info.compressed(), "payload should be stored compressed for this test");

    let mut bytes = std::fs::read(&path).unwrap();
    let declared_size = 64u64;
    let orig_size_off = HEADER_SIZE + 32;
    bytes[orig_size_off..orig_size_off + 8].copy_from_slice(&declared_size.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    let err = r.read_entry_decompressed(0).unwrap_err();
    assert!(
        matches!(err, ShardError::DecompressTooLarge(size) if size > declared_size),
        "expected bounded zstd decode failure, got: {:?}",
        err
    );
}

// ============================================================
// Golden file parity tests
// ============================================================

#[test]
fn test_mmap_golden_files() {
    let manifest = load_manifest();
    let dir = testdata_dir();

    for gf in &manifest.files {
        let path = dir.join(&gf.filename);
        let r = MmapShardReader::open(&path)
            .unwrap_or_else(|e| panic!("mmap open {}: {}", gf.filename, e));

        assert_eq!(
            r.entry_count(),
            gf.entries.len(),
            "{}: entry_count",
            gf.filename
        );

        for (i, ge) in gf.entries.iter().enumerate() {
            // Name
            assert_eq!(
                r.entry_name(i).unwrap(), ge.name,
                "{} entry {}: name",
                gf.filename, i
            );

            // Index entry metadata
            let info = r.get_entry_info(i).unwrap();
            assert_eq!(info.orig_size, ge.orig_size, "{} entry {}: orig_size", gf.filename, i);
            assert_eq!(info.disk_size, ge.disk_size, "{} entry {}: disk_size", gf.filename, i);
            assert_eq!(info.checksum, ge.checksum, "{} entry {}: checksum", gf.filename, i);
            assert_eq!(info.compressed(), ge.compressed, "{} entry {}: compressed", gf.filename, i);

            // Data content via read_entry_decompressed (works for both compressed and not)
            let data = r
                .read_entry_decompressed(i)
                .unwrap_or_else(|e| panic!("{} entry {}: read_entry_decompressed: {}", gf.filename, i, e));

            let got_sha = sha256_hex(&data);
            assert_eq!(
                got_sha, ge.data_sha256,
                "{} entry {}: data SHA256 mismatch",
                gf.filename, i
            );

            // For uncompressed entries, also verify zero-copy read returns same bytes
            if !ge.compressed {
                let zero_copy = r.read_entry(i)
                    .unwrap_or_else(|e| panic!("{} entry {}: read_entry: {}", gf.filename, i, e));
                assert_eq!(
                    zero_copy, data.as_slice(),
                    "{} entry {}: zero-copy != decompressed",
                    gf.filename, i
                );
            }
        }

        // Lookup all entries by name
        for (i, ge) in gf.entries.iter().enumerate() {
            let idx = r.lookup(&ge.name);
            assert_eq!(
                idx,
                Some(i),
                "{}: lookup({}) returned {:?}, expected Some({})",
                gf.filename,
                ge.name,
                idx,
                i
            );
        }

        // Non-existent lookup
        assert_eq!(
            r.lookup("__nonexistent__"),
            None,
            "{}: lookup of missing name should be None",
            gf.filename
        );
    }
}

#[test]
fn test_mmap_golden_data_matches_regular_reader() {
    // Verify mmap reader produces identical data to the regular heap-backed reader.
    let manifest = load_manifest();
    let dir = testdata_dir();

    for gf in &manifest.files {
        let path = dir.join(&gf.filename);

        let regular = shard_format::ShardReader::from_file(&path)
            .unwrap_or_else(|e| panic!("regular open {}: {}", gf.filename, e));
        let mmap = MmapShardReader::open(&path)
            .unwrap_or_else(|e| panic!("mmap open {}: {}", gf.filename, e));

        assert_eq!(
            regular.entry_count(),
            mmap.entry_count(),
            "{}: entry_count mismatch",
            gf.filename
        );

        for i in 0..regular.entry_count() {
            let regular_data = regular
                .read_entry(i)
                .unwrap_or_else(|e| panic!("{} entry {}: regular read_entry: {}", gf.filename, i, e));
            let mmap_data = mmap
                .read_entry_decompressed(i)
                .unwrap_or_else(|e| panic!("{} entry {}: mmap read_entry_decompressed: {}", gf.filename, i, e));

            assert_eq!(
                regular_data, mmap_data,
                "{} entry {}: regular != mmap data",
                gf.filename, i
            );

            // For uncompressed, also check zero-copy slice
            let info = mmap.get_entry_info(i).unwrap();
            if !info.compressed() {
                let zero_copy = mmap
                    .read_entry(i)
                    .unwrap_or_else(|e| panic!("{} entry {}: mmap read_entry: {}", gf.filename, i, e));
                assert_eq!(
                    regular_data,
                    zero_copy,
                    "{} entry {}: regular != mmap zero-copy",
                    gf.filename, i
                );
            }
        }
    }
}

#[test]
fn test_mmap_checksum_flag() {
    // Verify checksums are actually verified when FLAG_HAS_CHECKSUMS is set.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.shard");

    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("check", b"data");
    w.write_to_file(&path).unwrap();

    let r = MmapShardReader::open(&path).unwrap();
    assert!(r.header().flags & FLAG_HAS_CHECKSUMS != 0, "FLAG_HAS_CHECKSUMS should be set");

    // Normal read should succeed (checksum matches)
    let data = r.read_entry(0).unwrap();
    assert_eq!(data, b"data");
}
