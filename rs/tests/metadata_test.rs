/// Tests for metadata/schema, security limits, list_children, read_entry_prefix,
/// and path helpers added to shard_v2.
use shard_format::{
    EntryMeta, FLAG_HAS_SCHEMA, HEADER_SIZE, INDEX_ENTRY_SIZE, MAX_ENTRY_COUNT, ROLE_MOSH,
    ShardMetadata, ShardV2Reader, ShardV2Writer, join_path, path_base, path_parent, split_path,
};

// ============================================================
// Helper
// ============================================================

fn make_shard(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    for (name, data) in entries {
        w.write_entry(name, data);
    }
    w.to_bytes()
}

fn make_shard_with_meta(entries: &[(&str, &[u8])], meta: ShardMetadata) -> Vec<u8> {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    for (name, data) in entries {
        w.write_entry(name, data);
    }
    w.set_metadata(meta);
    w.to_bytes()
}

// ============================================================
// Path helpers
// ============================================================

#[test]
fn split_path_simple() {
    assert_eq!(split_path("layer.0/weight"), vec!["layer.0", "weight"]);
}

#[test]
fn split_path_deep() {
    assert_eq!(split_path("a/b/c/d"), vec!["a", "b", "c", "d"]);
}

#[test]
fn split_path_no_slash() {
    assert_eq!(split_path("embed"), vec!["embed"]);
}

#[test]
fn split_path_empty() {
    let result: Vec<&str> = split_path("");
    assert!(result.is_empty());
}

#[test]
fn join_path_two_parts() {
    assert_eq!(join_path(&["layer.0", "weight"]), "layer.0/weight");
}

#[test]
fn join_path_single() {
    assert_eq!(join_path(&["embed"]), "embed");
}

#[test]
fn join_path_multi() {
    assert_eq!(join_path(&["a", "b", "c"]), "a/b/c");
}

#[test]
fn path_parent_with_slash() {
    assert_eq!(path_parent("layer.0/weight"), "layer.0");
}

#[test]
fn path_parent_deep() {
    assert_eq!(path_parent("a/b/c"), "a/b");
}

#[test]
fn path_parent_no_slash() {
    assert_eq!(path_parent("embed"), "");
}

#[test]
fn path_base_with_slash() {
    assert_eq!(path_base("layer.0/weight"), "weight");
}

#[test]
fn path_base_deep() {
    assert_eq!(path_base("a/b/c"), "c");
}

#[test]
fn path_base_no_slash() {
    assert_eq!(path_base("embed"), "embed");
}

// ============================================================
// list_children
// ============================================================

fn make_layered_shard() -> Vec<u8> {
    make_shard(&[
        ("layer.0/weight", b"w0"),
        ("layer.0/bias", b"b0"),
        ("layer.1/weight", b"w1"),
        ("embed", b"tok"),
    ])
}

#[test]
fn list_children_exact_prefix_with_slash() {
    let buf = make_layered_shard();
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let mut result = r.list_children("layer.0/");
    result.sort();
    assert_eq!(result, vec!["layer.0/bias", "layer.0/weight"]);
}

#[test]
fn list_children_empty_prefix_returns_top_level() {
    let buf = make_layered_shard();
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let result = r.list_children("");
    assert!(result.contains(&"layer.0/".to_string()));
    assert!(result.contains(&"layer.1/".to_string()));
    assert!(result.contains(&"embed".to_string()));
    // No duplicates
    let mut sorted = result.clone();
    sorted.dedup();
    assert_eq!(result.len(), sorted.len());
}

#[test]
fn list_children_partial_prefix() {
    let buf = make_layered_shard();
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let mut result = r.list_children("layer.");
    result.sort();
    assert_eq!(result, vec!["layer.0/", "layer.1/"]);
}

#[test]
fn list_children_nonexistent_prefix() {
    let buf = make_shard(&[("a/b", b"x")]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let result = r.list_children("nonexistent/");
    assert!(result.is_empty());
}

#[test]
fn list_children_dedup_directories() {
    let buf = make_shard(&[
        ("a/b", b"1"),
        ("a/c", b"2"),
        ("a/d", b"3"),
    ]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let result = r.list_children("");
    // "a/" should appear exactly once
    assert_eq!(result.iter().filter(|x| x.as_str() == "a/").count(), 1);
}

#[test]
fn list_children_hierarchical_three_levels() {
    let buf = make_shard(&[
        ("a/b/c", b"1"),
        ("a/b/d", b"2"),
        ("a/e", b"3"),
        ("f", b"4"),
    ]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();

    let top = r.list_children("");
    assert!(top.contains(&"a/".to_string()));
    assert!(top.contains(&"f".to_string()));

    let under_a = r.list_children("a/");
    assert!(under_a.contains(&"a/b/".to_string()));
    assert!(under_a.contains(&"a/e".to_string()));

    let mut under_ab = r.list_children("a/b/");
    under_ab.sort();
    assert_eq!(under_ab, vec!["a/b/c", "a/b/d"]);
}

// ============================================================
// read_entry_prefix
// ============================================================

const PAYLOAD: &[u8] = b"Hello World, this is a test payload!";

#[test]
fn read_entry_prefix_first_n_bytes() {
    let buf = make_shard(&[("entry", PAYLOAD)]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_entry_prefix(0, 5).unwrap();
    assert_eq!(got, b"Hello");
}

#[test]
fn read_entry_prefix_full_entry() {
    let buf = make_shard(&[("entry", PAYLOAD)]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_entry_prefix(0, PAYLOAD.len()).unwrap();
    assert_eq!(got, PAYLOAD);
}

#[test]
fn read_entry_prefix_exceeds_entry_length() {
    let buf = make_shard(&[("entry", PAYLOAD)]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_entry_prefix(0, 100_000).unwrap();
    assert_eq!(got, PAYLOAD);
}

#[test]
fn read_entry_prefix_zero_bytes() {
    let buf = make_shard(&[("entry", PAYLOAD)]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_entry_prefix(0, 0).unwrap();
    assert!(got.is_empty());
}

#[test]
fn read_entry_prefix_one_byte() {
    let buf = make_shard(&[("entry", PAYLOAD)]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_entry_prefix(0, 1).unwrap();
    assert_eq!(got, b"H");
}

// ============================================================
// ShardMetadata roundtrip
// ============================================================

#[test]
fn metadata_basic_roundtrip() {
    let mut meta = ShardMetadata::default();
    meta.schema_version = "shard-v2.1".to_string();
    meta.producer = "test-producer".to_string();
    meta.description = "A test shard".to_string();
    meta.created_at = "2026-02-21T00:00:00Z".to_string();
    meta.tags = vec!["ml".to_string(), "test".to_string()];

    let buf = make_shard_with_meta(&[("data", b"hello")], meta);
    let r = ShardV2Reader::from_bytes(buf).unwrap();

    // FLAG_HAS_SCHEMA must be set
    assert!(r.header().flags & FLAG_HAS_SCHEMA != 0);
    assert!(r.header().schema_offset > 0);

    let got = r.read_metadata().unwrap().expect("metadata should be Some");
    assert_eq!(got.producer, "test-producer");
    assert_eq!(got.description, "A test shard");
    assert_eq!(got.created_at, "2026-02-21T00:00:00Z");
    assert_eq!(got.tags, vec!["ml", "test"]);
    assert_eq!(got.schema_version, "shard-v2.1");
}

#[test]
fn metadata_entry_metadata_roundtrip() {
    let mut em = EntryMeta::default();
    em.content_type = "tensor/float32".to_string();
    em.tags = vec!["weight".to_string()];
    em.description = "Model weights".to_string();
    em.extra.insert(
        "shape".to_string(),
        serde_json::json!([256, 256]),
    );

    let mut meta = ShardMetadata::default();
    meta.producer = "pytorch-exporter".to_string();
    meta.entry_metadata.insert("model/weight".to_string(), em);

    let buf = make_shard_with_meta(&[("model/weight", &[0u8; 16])], meta);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_metadata().unwrap().expect("metadata should be Some");

    assert!(got.entry_metadata.contains_key("model/weight"));
    let got_em = &got.entry_metadata["model/weight"];
    assert_eq!(got_em.content_type, "tensor/float32");
    assert_eq!(got_em.tags, vec!["weight"]);
    assert_eq!(got_em.description, "Model weights");
    assert_eq!(got_em.extra["shape"], serde_json::json!([256, 256]));
}

#[test]
fn metadata_no_metadata_returns_none() {
    let buf = make_shard(&[("data", b"hello")]);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    assert_eq!(r.header().schema_offset, 0);
    assert!(r.read_metadata().unwrap().is_none());
}

#[test]
fn metadata_extra_fields() {
    let mut meta = ShardMetadata::default();
    meta.extra.insert("model_type".to_string(), serde_json::json!("transformer"));
    meta.extra.insert("num_layers".to_string(), serde_json::json!(12));

    let buf = make_shard_with_meta(&[("cfg", b"x")], meta);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_metadata().unwrap().expect("metadata should be Some");
    assert_eq!(got.extra["model_type"], serde_json::json!("transformer"));
    assert_eq!(got.extra["num_layers"], serde_json::json!(12));
}

#[test]
fn metadata_source_uri_schema_uri() {
    let mut meta = ShardMetadata::default();
    meta.source_uri = "s3://bucket/model.safetensors".to_string();
    meta.schema_uri = "https://example.com/schema.json".to_string();

    let buf = make_shard_with_meta(&[("x", b"y")], meta);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_metadata().unwrap().expect("metadata should be Some");
    assert_eq!(got.source_uri, "s3://bucket/model.safetensors");
    assert_eq!(got.schema_uri, "https://example.com/schema.json");
}

#[test]
fn metadata_default_schema_version() {
    let meta = ShardMetadata::default();
    let buf = make_shard_with_meta(&[("x", b"y")], meta);
    let r = ShardV2Reader::from_bytes(buf).unwrap();
    let got = r.read_metadata().unwrap().expect("metadata should be Some");
    assert_eq!(got.schema_version, "shard-v2.1");
}

#[test]
fn security_schema_offset_beyond_file_is_rejected() {
    let mut buf = make_shard_with_meta(&[("data", b"hello")], ShardMetadata::default());
    let invalid = (buf.len() as u64) + 1;
    buf[32..40].copy_from_slice(&invalid.to_le_bytes());

    let err = ShardV2Reader::from_bytes(buf).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("schema_offset"), "expected schema_offset error, got: {}", msg);
}

#[test]
fn security_schema_offset_overlapping_data_is_rejected() {
    let mut buf = make_shard_with_meta(&[("data", b"hello world")], ShardMetadata::default());
    let entry_data_offset_off = HEADER_SIZE + 16;
    let data_offset = u64::from_le_bytes(
        buf[entry_data_offset_off..entry_data_offset_off + 8]
            .try_into()
            .unwrap(),
    );
    buf[32..40].copy_from_slice(&data_offset.to_le_bytes());

    let err = ShardV2Reader::from_bytes(buf).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("schema_offset"), "expected schema_offset error, got: {}", msg);
}

// ============================================================
// Security limits
// ============================================================

fn make_corrupt_header(entry_count: u32, total_file_size_override: Option<u64>) -> Vec<u8> {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"SHRD");
    buf[4] = 0x02; // version
    buf[5] = 1;    // role
    buf[6..8].copy_from_slice(&0x00A2u16.to_le_bytes()); // flags
    buf[8] = 64;   // alignment
    buf[9] = 0;    // compression
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes());
    buf[12..16].copy_from_slice(&entry_count.to_le_bytes());

    let st_off = HEADER_SIZE as u64;
    let ds_off = st_off;
    buf[16..24].copy_from_slice(&st_off.to_le_bytes()); // string_table_offset
    buf[24..32].copy_from_slice(&ds_off.to_le_bytes()); // data_section_offset
    buf[32..40].copy_from_slice(&0u64.to_le_bytes());   // schema_offset = 0

    let tfs = total_file_size_override.unwrap_or(HEADER_SIZE as u64);
    buf[40..48].copy_from_slice(&tfs.to_le_bytes());    // total_file_size

    buf
}

#[test]
fn security_entry_count_too_large() {
    let too_many = (MAX_ENTRY_COUNT + 1) as u32;
    let buf = make_corrupt_header(too_many, Some(HEADER_SIZE as u64));
    let err = ShardV2Reader::from_bytes(buf).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("MAX_ENTRY_COUNT"),
        "expected MAX_ENTRY_COUNT in error, got: {}",
        msg
    );
}

#[test]
fn security_total_file_size_mismatch() {
    // Build a valid shard then corrupt total_file_size
    let mut buf = make_shard(&[("x", b"hello")]);
    // Overwrite total_file_size at offset 40
    let corrupt_size = 99999u64;
    buf[40..48].copy_from_slice(&corrupt_size.to_le_bytes());
    let err = ShardV2Reader::from_bytes(buf).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("total_file_size"),
        "expected total_file_size in error, got: {}",
        msg
    );
}

#[test]
fn security_data_offset_before_data_section() {
    // Build a valid shard, then set entry 0 data_offset to 0 (before header)
    let mut buf = make_shard(&[("x", b"hello")]);
    let entry_off = HEADER_SIZE + 16; // data_offset field is at index+16
    buf[entry_off..entry_off + 8].copy_from_slice(&0u64.to_le_bytes());
    // Keep total_file_size consistent
    let tfs = buf.len() as u64;
    buf[40..48].copy_from_slice(&tfs.to_le_bytes());
    assert!(ShardV2Reader::from_bytes(buf).is_err());
}

#[test]
fn security_data_extends_past_end_of_file() {
    // Build a valid shard, then set entry 0 data_offset to huge value
    let mut buf = make_shard(&[("x", b"hello")]);
    let entry_off = HEADER_SIZE + 16;
    buf[entry_off..entry_off + 8].copy_from_slice(&0xFFFF_FFFFu64.to_le_bytes());
    let tfs = buf.len() as u64;
    buf[40..48].copy_from_slice(&tfs.to_le_bytes());
    assert!(ShardV2Reader::from_bytes(buf).is_err());
}
