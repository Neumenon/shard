//! Streaming writer roundtrip tests for `ShardStreamWriter`.

use std::path::PathBuf;

use shard_format::{
    ShardReader, ShardStreamWriter,
    ALIGN_16, ALIGN_32, ALIGN_64, ALIGN_NONE,
    COMPRESS_LZ4, COMPRESS_ZSTD,
    FLAG_STREAMING,
    ROLE_MOSH, ROLE_SAMPLE,
    compute_crc32c,
};

// ============================================================
// Helpers
// ============================================================

fn tmp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir();
    dir.join(format!("shard_stream_test_{}_{}", name, std::process::id()))
}

// ============================================================
// Basic roundtrip
// ============================================================

#[test]
fn test_basic_roundtrip() {
    let path = tmp_path("basic");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("hello", b"world").unwrap();
    sw.write_entry("foo", b"bar").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.entry_count(), 2);
    assert_eq!(r.read_entry(0).unwrap(), b"world");
    assert_eq!(r.read_entry(1).unwrap(), b"bar");
    assert_eq!(r.entry_name(0).unwrap(), "hello");
    assert_eq!(r.entry_name(1).unwrap(), "foo");
    // FLAG_STREAMING must be set
    assert!(r.header().flags & FLAG_STREAMING != 0, "FLAG_STREAMING must be set");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_flag_streaming_is_set() {
    let path = tmp_path("flags");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("x", b"data").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert!(
        r.header().flags & FLAG_STREAMING != 0,
        "FLAG_STREAMING should be set by StreamWriter"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_empty_shard() {
    let path = tmp_path("empty");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.begin_data().unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.entry_count(), 0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_single_entry() {
    let path = tmp_path("single");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("only", b"payload").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.entry_count(), 1);
    assert_eq!(r.read_entry(0).unwrap(), b"payload");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_role_propagated() {
    let path = tmp_path("role");
    let mut sw = ShardStreamWriter::new(&path, ROLE_SAMPLE, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("a", b"1").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.header().role, ROLE_SAMPLE);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_total_file_size_correct() {
    let path = tmp_path("filesize");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("item", b"hello").unwrap();
    sw.finalize().unwrap();

    let actual = std::fs::metadata(&path).unwrap().len();
    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.header().total_file_size, actual);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_lookup_by_name() {
    let path = tmp_path("lookup");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("alpha", b"a").unwrap();
    sw.write_entry("beta", b"b").unwrap();
    sw.write_entry("gamma", b"c").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.lookup("alpha"), Some(0));
    assert_eq!(r.lookup("beta"), Some(1));
    assert_eq!(r.lookup("gamma"), Some(2));
    assert_eq!(r.lookup("missing"), None);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_read_by_name() {
    let path = tmp_path("byname");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("key1", b"value1").unwrap();
    sw.write_entry("key2", b"value2").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.read_entry_by_name("key1").unwrap(), b"value1");
    assert_eq!(r.read_entry_by_name("key2").unwrap(), b"value2");
    assert!(r.read_entry_by_name("missing").is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_checksum_stored_correctly() {
    let path = tmp_path("cksum");
    let data = b"checksum_test_data";
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("data", data).unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    let info = r.get_entry_info(0).unwrap();
    assert_eq!(info.checksum, compute_crc32c(data));
    let _ = std::fs::remove_file(&path);
}

// ============================================================
// Alignment
// ============================================================

#[test]
fn test_alignment_64_default() {
    let path = tmp_path("align64_default");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("a", b"data_a").unwrap();
    sw.write_entry("b", b"data_b").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.header().alignment, ALIGN_64);
    for i in 0..r.entry_count() {
        let offset = r.get_entry_info(i).unwrap().data_offset;
        assert_eq!(offset % 64, 0, "entry {} offset {} not 64-aligned", i, offset);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_alignment_64_explicit() {
    let path = tmp_path("align64_explicit");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.set_alignment(ALIGN_64).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("a", b"data_a").unwrap();
    sw.write_entry("b", b"data_b").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    for i in 0..r.entry_count() {
        assert_eq!(r.get_entry_info(i).unwrap().data_offset % 64, 0);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_alignment_16() {
    let path = tmp_path("align16");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.set_alignment(ALIGN_16).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("x", b"hello").unwrap();
    sw.write_entry("y", b"world_longer_string").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.header().alignment, ALIGN_16);
    for i in 0..r.entry_count() {
        let offset = r.get_entry_info(i).unwrap().data_offset;
        assert_eq!(offset % 16, 0, "entry {} offset {} not 16-aligned", i, offset);
    }
    assert_eq!(r.read_entry(0).unwrap(), b"hello");
    assert_eq!(r.read_entry(1).unwrap(), b"world_longer_string");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_alignment_32() {
    let path = tmp_path("align32");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.set_alignment(ALIGN_32).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("p", b"payload").unwrap();
    sw.write_entry("q", b"more_payload").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.header().alignment, ALIGN_32);
    for i in 0..r.entry_count() {
        assert_eq!(r.get_entry_info(i).unwrap().data_offset % 32, 0);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_alignment_none() {
    let path = tmp_path("align_none");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.set_alignment(ALIGN_NONE).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("a", b"x").unwrap();
    sw.write_entry("b", b"yy").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.header().alignment, ALIGN_NONE);
    assert_eq!(r.read_entry(0).unwrap(), b"x");
    assert_eq!(r.read_entry(1).unwrap(), b"yy");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_set_alignment_after_begin_errors() {
    let path = tmp_path("align_err");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    let result = sw.set_alignment(ALIGN_16);
    assert!(result.is_err(), "set_alignment after begin_data should fail");
    sw.finalize().unwrap();
    let _ = std::fs::remove_file(&path);
}

// ============================================================
// Many entries
// ============================================================

#[test]
fn test_many_entries() {
    let path = tmp_path("many");
    let n: usize = 100;
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, n).unwrap();
    sw.begin_data().unwrap();
    for i in 0..n {
        let name = format!("entry_{:04}", i);
        let data = format!("data_{}", i);
        sw.write_entry(&name, data.as_bytes()).unwrap();
    }
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.entry_count(), n);
    for i in 0..n {
        assert_eq!(r.entry_name(i).unwrap(), format!("entry_{:04}", i));
        assert_eq!(r.read_entry(i).unwrap(), format!("data_{}", i).as_bytes());
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_fewer_entries_than_max() {
    let path = tmp_path("under_max");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 1000).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("only_one", b"data").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.entry_count(), 1);
    assert_eq!(r.read_entry(0).unwrap(), b"data");
    let _ = std::fs::remove_file(&path);
}

// ============================================================
// Large data
// ============================================================

#[test]
fn test_large_data() {
    let path = tmp_path("large");
    let big: Vec<u8> = (0..100_000usize).map(|i| (i % 256) as u8).collect();
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("big", &big).unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), big.as_slice());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_multiple_large_entries() {
    let path = tmp_path("multi_large");
    let chunks: Vec<Vec<u8>> = (1..4usize)
        .map(|i| (0..50_000usize).map(|j| ((i * j) % 256) as u8).collect())
        .collect();

    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.begin_data().unwrap();
    for (idx, chunk) in chunks.iter().enumerate() {
        sw.write_entry(&format!("chunk_{}", idx), chunk).unwrap();
    }
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.entry_count(), 3);
    for (idx, chunk) in chunks.iter().enumerate() {
        assert_eq!(r.read_entry(idx).unwrap(), chunk.as_slice());
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_binary_all_bytes() {
    let path = tmp_path("allbytes");
    let all_bytes: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.write_entry("allbytes", &all_bytes).unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), all_bytes.as_slice());
    let _ = std::fs::remove_file(&path);
}

// ============================================================
// Compression
// ============================================================

#[test]
fn test_zstd_compressed_entry() {
    let path = tmp_path("zstd");
    let data: Vec<u8> = (0..10_000usize).map(|i| (i % 256) as u8).collect();
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.set_compression(COMPRESS_ZSTD);
    sw.begin_data().unwrap();
    sw.write_entry_compressed("data", &data).unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    // Note: Rust reader doesn't decompress; we verify the flags and sizes
    let info = r.get_entry_info(0).unwrap();
    assert!(info.compressed(), "entry should be marked compressed");
    assert!(info.disk_size < info.orig_size, "compressed size should be smaller");
    assert_eq!(info.orig_size, data.len() as u64);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_lz4_compressed_entry() {
    let path = tmp_path("lz4");
    let data: Vec<u8> = (0..10_000usize).map(|i| (i % 256) as u8).collect();
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.set_compression(COMPRESS_LZ4);
    sw.begin_data().unwrap();
    sw.write_entry_compressed("data", &data).unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    let info = r.get_entry_info(0).unwrap();
    assert!(info.compressed(), "entry should be marked compressed");
    assert!(info.disk_size < info.orig_size, "compressed size should be smaller");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_small_data_not_compressed() {
    let path = tmp_path("small_nocompress");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.set_compression(COMPRESS_ZSTD);
    sw.begin_data().unwrap();
    sw.write_entry_compressed("tiny", b"hello").unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    let info = r.get_entry_info(0).unwrap();
    assert!(!info.compressed(), "tiny entry should not be compressed");
    assert_eq!(r.read_entry(0).unwrap(), b"hello");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_mixed_compressed_uncompressed() {
    let path = tmp_path("mixed_comp");
    let big: Vec<u8> = (0..10_000usize).map(|i| (i % 256) as u8).collect();
    let small = b"tiny";
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 10).unwrap();
    sw.set_compression(COMPRESS_ZSTD);
    sw.begin_data().unwrap();
    sw.write_entry_compressed("big", &big).unwrap();
    sw.write_entry("small", small).unwrap();
    sw.finalize().unwrap();

    let r = ShardReader::from_file(&path).unwrap();
    assert!(r.get_entry_info(0).unwrap().compressed(), "big should be compressed");
    assert!(!r.get_entry_info(1).unwrap().compressed(), "small should not be compressed");
    // Verify small entry reads back correctly (uncompressed)
    assert_eq!(r.read_entry(1).unwrap(), b"tiny");
    let _ = std::fs::remove_file(&path);
}

// ============================================================
// Error states
// ============================================================

#[test]
fn test_write_before_begin_errors() {
    let path = tmp_path("err_before_begin");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    let result = sw.write_entry("x", b"data");
    assert!(result.is_err(), "write_entry before begin_data should fail");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_begin_twice_errors() {
    let path = tmp_path("err_begin_twice");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    let result = sw.begin_data();
    assert!(result.is_err(), "second begin_data() should fail");
    sw.finalize().unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_finalize_twice_errors() {
    let path = tmp_path("err_fin_twice");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.finalize().unwrap();
    let result = sw.finalize();
    assert!(result.is_err(), "second finalize() should fail");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_write_after_finalize_errors() {
    let path = tmp_path("err_write_after_fin");
    let mut sw = ShardStreamWriter::new(&path, ROLE_MOSH, 5).unwrap();
    sw.begin_data().unwrap();
    sw.finalize().unwrap();
    let result = sw.write_entry("x", b"data");
    assert!(result.is_err(), "write_entry after finalize() should fail");
    let _ = std::fs::remove_file(&path);
}
