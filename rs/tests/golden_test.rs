//! Golden file parity tests.
//!
//! Reads the manifest from `../ucodec/testdata/golden_manifest.json` and verifies
//! that the Rust Shard v2 reader produces identical results to the Go implementation.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use shard_format::{
    compute_crc32c, compute_xxhash64, ShardV2Reader, SHARD_VERSION2, INDEX_ENTRY_SIZE,
};
use std::path::{Path, PathBuf};

// ============================================================
// Manifest deserialization structs
// ============================================================

#[derive(Deserialize)]
struct Manifest {
    files: Vec<GoldenFile>,
}

#[derive(Deserialize)]
struct GoldenFile {
    filename: String,
    sha256: String,
    header: GoldenHeader,
    entries: Vec<GoldenEntry>,
}

#[derive(Deserialize)]
struct GoldenHeader {
    version: u8,
    role: u8,
    flags: u16,
    alignment: u8,
    compression_default: u8,
    entry_count: u32,
    index_entry_size: u16,
    string_table_offset: u64,
    data_section_offset: u64,
    schema_offset: u64,
    total_file_size: u64,
}

#[derive(Deserialize)]
struct GoldenEntry {
    name: String,
    name_hash: u64,
    content_type: u16,
    orig_size: u64,
    disk_size: u64,
    checksum: u32,
    compressed: bool,
    data_sha256: String,
}

// ============================================================
// Helpers
// ============================================================

fn testdata_dir() -> PathBuf {
    // tests/ is at <crate_root>/tests/, testdata is at ../ucodec/testdata/
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    Path::new(&manifest).join("../ucodec/testdata")
}

fn load_manifest() -> Manifest {
    let path = testdata_dir().join("golden_manifest.json");
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&contents)
        .unwrap_or_else(|e| panic!("Failed to parse manifest: {}", e))
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

// ============================================================
// Tests
// ============================================================

#[test]
fn test_manifest_exists() {
    let path = testdata_dir().join("golden_manifest.json");
    assert!(
        path.exists(),
        "golden_manifest.json not found at {}",
        path.display()
    );
}

#[test]
fn test_all_golden_files_exist() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let p = dir.join(&gf.filename);
        assert!(
            p.exists(),
            "golden file {} not found at {}",
            gf.filename,
            p.display()
        );
    }
}

#[test]
fn test_file_sha256() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let data = std::fs::read(dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("read {}: {}", gf.filename, e));
        let got = sha256_hex(&data);
        assert_eq!(
            got, gf.sha256,
            "SHA256 mismatch for {}",
            gf.filename
        );
    }
}

#[test]
fn test_header_fields() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        let h = reader.header();
        let g = &gf.header;

        assert_eq!(h.version, g.version, "{}: version", gf.filename);
        assert_eq!(h.role, g.role, "{}: role", gf.filename);
        assert_eq!(h.flags, g.flags, "{}: flags", gf.filename);
        assert_eq!(h.alignment, g.alignment, "{}: alignment", gf.filename);
        assert_eq!(
            h.compression_default, g.compression_default,
            "{}: compression_default",
            gf.filename
        );
        assert_eq!(
            h.entry_count, g.entry_count,
            "{}: entry_count",
            gf.filename
        );
        assert_eq!(
            h.index_entry_size, g.index_entry_size,
            "{}: index_entry_size",
            gf.filename
        );
        assert_eq!(
            h.string_table_offset, g.string_table_offset,
            "{}: string_table_offset",
            gf.filename
        );
        assert_eq!(
            h.data_section_offset, g.data_section_offset,
            "{}: data_section_offset",
            gf.filename
        );
        assert_eq!(
            h.schema_offset, g.schema_offset,
            "{}: schema_offset",
            gf.filename
        );
        assert_eq!(
            h.total_file_size, g.total_file_size,
            "{}: total_file_size",
            gf.filename
        );
    }
}

#[test]
fn test_entry_count() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        assert_eq!(
            reader.entry_count(),
            gf.entries.len(),
            "{}: entry_count",
            gf.filename
        );
    }
}

#[test]
fn test_entry_names() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        let names = reader.entry_names();
        let expected: Vec<&str> = gf.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, expected, "{}: entry names", gf.filename);
    }
}

#[test]
fn test_entry_details() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        for (i, ge) in gf.entries.iter().enumerate() {
            let info = reader.get_entry_info(i);
            let name = reader.entry_name(i);

            assert_eq!(name, ge.name, "{} entry {}: name", gf.filename, i);
            assert_eq!(
                info.name_hash, ge.name_hash,
                "{} entry {}: name_hash",
                gf.filename, i
            );
            assert_eq!(
                info.content_type(),
                ge.content_type,
                "{} entry {}: content_type",
                gf.filename, i
            );
            assert_eq!(
                info.orig_size, ge.orig_size,
                "{} entry {}: orig_size",
                gf.filename, i
            );
            assert_eq!(
                info.disk_size, ge.disk_size,
                "{} entry {}: disk_size",
                gf.filename, i
            );
            assert_eq!(
                info.checksum, ge.checksum,
                "{} entry {}: checksum",
                gf.filename, i
            );
            assert_eq!(
                info.compressed(),
                ge.compressed,
                "{} entry {}: compressed",
                gf.filename, i
            );
        }
    }
}

#[test]
fn test_entry_data_sha256() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        for (i, ge) in gf.entries.iter().enumerate() {
            let data = reader
                .read_entry(i)
                .unwrap_or_else(|e| panic!("{} entry {}: read_entry error: {}", gf.filename, i, e));
            let got = sha256_hex(&data);
            assert_eq!(
                got, ge.data_sha256,
                "{} entry {}: data SHA256 mismatch",
                gf.filename, i
            );
        }
    }
}

#[test]
fn test_lookup_by_name() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        for (i, ge) in gf.entries.iter().enumerate() {
            let idx = reader.lookup(&ge.name);
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
        // Non-existent entry
        assert_eq!(
            reader.lookup("__nonexistent__"),
            None,
            "{}: lookup of missing name should be None",
            gf.filename
        );
    }
}

#[test]
fn test_read_by_name() {
    let manifest = load_manifest();
    let dir = testdata_dir();
    for gf in &manifest.files {
        let reader = ShardV2Reader::from_file(&dir.join(&gf.filename))
            .unwrap_or_else(|e| panic!("open {}: {}", gf.filename, e));
        for (i, ge) in gf.entries.iter().enumerate() {
            let by_name = reader
                .read_entry_by_name(&ge.name)
                .unwrap_or_else(|e| panic!("{} entry {}: read_entry_by_name: {}", gf.filename, i, e));
            let by_index = reader
                .read_entry(i)
                .unwrap_or_else(|e| panic!("{} entry {}: read_entry: {}", gf.filename, i, e));
            assert_eq!(
                by_name, by_index,
                "{} entry {}: read_by_name != read_by_index",
                gf.filename, i
            );
        }
    }
}

#[test]
fn test_crc32c_known_values() {
    assert_eq!(
        compute_crc32c(b"hello world"),
        3381945770,
        "crc32c(\"hello world\")"
    );
    assert_eq!(compute_crc32c(b""), 0, "crc32c(\"\")");
}

#[test]
fn test_xxhash64_known_values() {
    assert_eq!(
        compute_xxhash64("greeting"),
        16700977181434310015,
        "xxhash64(\"greeting\")"
    );
    assert_eq!(
        compute_xxhash64("pattern/1k"),
        11449517142474649281,
        "xxhash64(\"pattern/1k\")"
    );
}

#[test]
fn test_list_prefix_hierarchical() {
    let dir = testdata_dir();
    let reader = ShardV2Reader::from_file(&dir.join("golden_hierarchical.shard"))
        .expect("open golden_hierarchical.shard");

    // All 4 attention entries
    let mut attention = reader.list_prefix("layer.0/attention");
    attention.sort();
    assert_eq!(attention.len(), 4, "layer.0/attention prefix count");
    assert!(attention.contains(&"layer.0/attention/q_proj/weight"));
    assert!(attention.contains(&"layer.0/attention/k_proj/weight"));
    assert!(attention.contains(&"layer.0/attention/v_proj/weight"));
    assert!(attention.contains(&"layer.0/attention/o_proj/weight"));

    // All 3 ffn entries
    let mut ffn = reader.list_prefix("layer.0/ffn");
    ffn.sort();
    assert_eq!(ffn.len(), 3, "layer.0/ffn prefix count");
    assert!(ffn.contains(&"layer.0/ffn/gate/weight"));
    assert!(ffn.contains(&"layer.0/ffn/up/weight"));
    assert!(ffn.contains(&"layer.0/ffn/down/weight"));

    // All layer.0 entries (8 total)
    let layer0 = reader.list_prefix("layer.0");
    assert_eq!(layer0.len(), 8, "layer.0 prefix count");

    // Non-existent prefix
    let empty = reader.list_prefix("layer.99");
    assert!(empty.is_empty(), "layer.99 prefix should be empty");
}

#[test]
fn test_list_prefix_wshard() {
    let dir = testdata_dir();
    let reader = ShardV2Reader::from_file(&dir.join("golden_wshard.shard"))
        .expect("open golden_wshard.shard");

    // signal/ prefix
    let signal = reader.list_prefix("signal");
    assert_eq!(signal.len(), 1, "signal/ prefix");
    assert!(signal.contains(&"signal/imu"));

    // omen/ prefix
    let omen = reader.list_prefix("omen");
    assert_eq!(omen.len(), 1, "omen/ prefix");
    assert!(omen.contains(&"omen/imu/mlp_v1"));

    // residual/ prefix
    let residual = reader.list_prefix("residual");
    assert_eq!(residual.len(), 1, "residual/ prefix");
    assert!(residual.contains(&"residual/imu/sign2nddiff"));

    // Non-existent prefix
    let none = reader.list_prefix("nonexistent");
    assert!(none.is_empty(), "nonexistent prefix should be empty");
}

#[test]
fn test_version_constant() {
    assert_eq!(SHARD_VERSION2, 0x02);
}

#[test]
fn test_index_entry_size_constant() {
    assert_eq!(INDEX_ENTRY_SIZE, 48);
}
