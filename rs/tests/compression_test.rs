//! Compression roundtrip and edge-case tests for Shard v2.

use shard_format::{
    compute_crc32c, ShardV2Reader, ShardV2Writer,
    COMPRESS_LZ4, COMPRESS_NONE, COMPRESS_ZSTD, ROLE_MOSH,
};

#[test]
fn test_zstd_roundtrip() {
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_ZSTD);
    w.write_entry_compressed("big", &data);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), data);
    let info = r.get_entry_info(0);
    assert!(info.compressed(), "entry should be marked compressed");
    assert!(
        info.disk_size < info.orig_size,
        "disk_size {} should be smaller than orig_size {} after compression",
        info.disk_size,
        info.orig_size
    );
    assert_eq!(info.orig_size, data.len() as u64);
}

#[test]
fn test_lz4_roundtrip() {
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_LZ4);
    w.write_entry_compressed("big", &data);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), data);
    let info = r.get_entry_info(0);
    assert!(info.compressed(), "entry should be marked compressed");
    assert_eq!(info.orig_size, data.len() as u64);
}

#[test]
fn test_small_data_not_compressed() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_ZSTD);
    w.write_entry_compressed("tiny", b"hello");
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), b"hello");
    assert!(
        !r.get_entry_info(0).compressed(),
        "small data should not be compressed"
    );
}

#[test]
fn test_checksum_on_uncompressed_data() {
    let data: Vec<u8> = (0..256).cycle().take(2560).map(|x: usize| x as u8).collect();
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_ZSTD);
    w.write_entry_compressed("data", &data);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    // Checksum in the index must be over the ORIGINAL (uncompressed) data.
    assert_eq!(
        r.get_entry_info(0).checksum,
        compute_crc32c(&data),
        "checksum must be CRC32C of uncompressed data"
    );
}

#[test]
fn test_mixed_compressed_and_plain() {
    let big: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
    let small = b"tiny";

    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_ZSTD);
    w.write_entry_compressed("compressed", &big);
    w.write_entry("plain", small);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), big);
    assert_eq!(r.read_entry(1).unwrap(), small.as_ref());
    assert!(r.get_entry_info(0).compressed(), "first entry should be compressed");
    assert!(!r.get_entry_info(1).compressed(), "second entry should not be compressed");
}

#[test]
fn test_write_entry_with_options() {
    let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_with_options("zstd", &data, true, COMPRESS_ZSTD);
    w.write_entry_with_options("none", &data, false, COMPRESS_NONE);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), data);
    assert_eq!(r.read_entry(1).unwrap(), data);
    assert!(r.get_entry_info(0).compressed(), "zstd entry should be compressed");
    assert!(!r.get_entry_info(1).compressed(), "none entry should not be compressed");
}

#[test]
fn test_lz4_write_entry_with_options() {
    let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_with_options("lz4", &data, true, COMPRESS_LZ4);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    assert_eq!(r.read_entry(0).unwrap(), data);
    assert!(r.get_entry_info(0).compressed());
}

#[test]
fn test_compression_read_by_name() {
    let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_ZSTD);
    w.write_entry_compressed("mykey", &data);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    let by_name = r.read_entry_by_name("mykey").unwrap();
    let by_index = r.read_entry(0).unwrap();
    assert_eq!(by_name, by_index);
    assert_eq!(by_name, data);
}

#[test]
fn test_incompressible_data_stored_uncompressed() {
    // Random-looking data that won't compress well (already "random").
    // We fill it with a pseudo-random pattern that zstd cannot compress >=10%.
    // Use a simple XOR-shift pattern to simulate incompressible data.
    let mut data = vec![0u8; 1024];
    let mut state: u64 = 0xdeadbeef_cafebabe;
    for b in data.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = state as u8;
    }

    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.set_compression(COMPRESS_ZSTD);
    w.write_entry_compressed("random", &data);
    let bytes = w.to_bytes();

    let r = ShardV2Reader::from_bytes(bytes).unwrap();
    // Whether compressed or not, the decompressed data must round-trip correctly.
    assert_eq!(r.read_entry(0).unwrap(), data);
}
