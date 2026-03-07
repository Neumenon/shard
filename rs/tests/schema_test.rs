//! Schema validation tests for shard-rs.

use shard_format::{
    match_pattern, validate_schema, ShardSchema, ShardV2Reader, ShardV2Writer,
    CONTENT_TYPE_JSON, CONTENT_TYPE_TENSOR, CONTENT_TYPE_TEXT, CONTENT_TYPE_UNKNOWN,
    ROLE_MOSH,
};

// ============================================================
// Helpers
// ============================================================

fn roundtrip(writer: ShardV2Writer) -> ShardV2Reader {
    let bytes = writer.to_bytes();
    ShardV2Reader::from_bytes(bytes).expect("roundtrip: from_bytes failed")
}

// ============================================================
// match_pattern unit tests
// ============================================================

#[test]
fn test_match_exact() {
    assert!(match_pattern("embed.weight", "embed.weight"));
    assert!(!match_pattern("embed.weight", "embed.bias"));
}

#[test]
fn test_match_star_dot_component() {
    // * matches one component between separators
    assert!(match_pattern("layer.*.weight", "layer.0.weight"));
    assert!(match_pattern("layer.*.weight", "layer.10.weight"));
    assert!(!match_pattern("layer.*.weight", "layer.0.bias"));
}

#[test]
fn test_match_star_slash_component() {
    assert!(match_pattern("layer/*/weight", "layer/0/weight"));
    assert!(match_pattern("layer/*/weight", "layer/42/weight"));
    assert!(!match_pattern("layer/*/weight", "layer/0/bias"));
}

#[test]
fn test_match_star_does_not_cross_dot() {
    // "layer.*.weight" should NOT match "layer.0.extra.weight" (extra separator)
    assert!(!match_pattern("layer.*.weight", "layer.0.extra.weight"));
}

#[test]
fn test_match_star_does_not_cross_slash() {
    // "layer/*/weight" should NOT match "layer/0/sub/weight"
    assert!(!match_pattern("layer/*/weight", "layer/0/sub/weight"));
}

#[test]
fn test_match_double_star_prefix() {
    assert!(match_pattern("layers/**", "layers/0/attn/q"));
    assert!(match_pattern("layers/**", "layers/0/attn/k"));
    assert!(match_pattern("layers/**", "layers/"));
    assert!(!match_pattern("layers/**", "other/0/attn"));
}

#[test]
fn test_match_no_wildcard_no_separator() {
    assert!(match_pattern("config", "config"));
    assert!(!match_pattern("config", "configs"));
}

// ============================================================
// validate_schema tests
// ============================================================

#[test]
fn test_valid_schema_exact_names_and_types() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_typed("model/weight", &[1, 2, 3], CONTENT_TYPE_TENSOR);
    w.write_entry_typed("config.json", b"{}", CONTENT_TYPE_JSON);
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("test");
    schema.add_spec("model/weight", CONTENT_TYPE_TENSOR, false);
    schema.add_spec("config.json", CONTENT_TYPE_JSON, false);

    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
}

#[test]
fn test_missing_required_entry() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("only_one", b"data");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("only_one");
    schema.add_required("missing_entry");

    let errors = validate_schema(&r, &schema);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("missing_entry"), "message: {}", errors[0].message);
}

#[test]
fn test_optional_missing_is_not_error() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("required", b"data");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("required");
    schema.add_optional("optional_entry");

    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty());
}

#[test]
fn test_dot_wildcard_pattern() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("layer.0.weight", b"w0");
    w.write_entry("layer.1.weight", b"w1");
    w.write_entry("layer.2.weight", b"w2");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("layer.*.weight");

    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty());
}

#[test]
fn test_slash_wildcard_pattern() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("layer/0/weight", b"w0");
    w.write_entry("layer/1/weight", b"w1");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("layer/*/weight");

    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty());
}

#[test]
fn test_prefix_wildcard() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("layers/0/attn/q", b"q");
    w.write_entry("layers/0/attn/k", b"k");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("layers/**");

    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty());
}

#[test]
fn test_wrong_content_type() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_typed("data", b"not_a_tensor", CONTENT_TYPE_TEXT);
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_spec("data", CONTENT_TYPE_TENSOR, false);

    let errors = validate_schema(&r, &schema);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("data"), "message: {}", errors[0].message);
    assert!(
        errors[0].message.contains(&CONTENT_TYPE_TEXT.to_string()),
        "message: {}",
        errors[0].message
    );
}

#[test]
fn test_empty_schema_always_valid() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("anything", b"data");
    let r = roundtrip(w);

    let schema = ShardSchema::new("");
    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty());
}

#[test]
fn test_multiple_errors() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("only_this", b"data");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("only_this");
    schema.add_required("missing_a");
    schema.add_required("missing_b");

    let errors = validate_schema(&r, &schema);
    assert_eq!(errors.len(), 2);
    let patterns: std::collections::HashSet<&str> =
        errors.iter().map(|e| e.spec_pattern.as_str()).collect();
    assert!(patterns.contains("missing_a"));
    assert!(patterns.contains("missing_b"));
}

#[test]
fn test_content_type_zero_means_any() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_typed("data", b"x", CONTENT_TYPE_JSON);
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    // content_type = 0 (UNKNOWN) → accept anything
    schema.add_spec("data", CONTENT_TYPE_UNKNOWN, false);

    let errors = validate_schema(&r, &schema);
    assert!(errors.is_empty());
}

#[test]
fn test_add_spec_chaining() {
    let mut schema = ShardSchema::new("");
    let result = schema.add_required("a");
    // add_required returns &mut Self — the chain compiles and the spec was added.
    let _ = result;
    assert_eq!(schema.specs.len(), 1);
    schema.add_required("b").add_optional("c");
    assert_eq!(schema.specs.len(), 3);
}

#[test]
fn test_validation_error_fields() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("present", b"data");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("absent.entry");

    let errors = validate_schema(&r, &schema);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].spec_pattern, "absent.entry");
    assert!(errors[0].message.contains("absent.entry"));
}

#[test]
fn test_optional_entry_that_exists_is_type_checked() {
    // An optional entry that IS present should still have its type verified.
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry_typed("meta", b"text", CONTENT_TYPE_TEXT);
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_spec("meta", CONTENT_TYPE_JSON, true /* optional */);

    let errors = validate_schema(&r, &schema);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("meta"));
}

#[test]
fn test_required_wildcard_no_match_is_error() {
    let mut w = ShardV2Writer::new(ROLE_MOSH);
    w.write_entry("other.entry", b"x");
    let r = roundtrip(w);

    let mut schema = ShardSchema::new("");
    schema.add_required("layer.*.weight");

    let errors = validate_schema(&r, &schema);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("layer.*.weight"));
}
