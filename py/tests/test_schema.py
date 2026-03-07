"""Schema validation tests for shard_v2."""
import pytest

from shard_v2 import (
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_TENSOR,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_UNKNOWN,
    ShardSchema,
    ShardV2Reader,
    ShardV2Writer,
    ValidationError,
    validate_schema,
)


class TestSchemaValidation:
    def test_valid_schema(self, tmp_path):
        path = tmp_path / "valid.shard"
        w = ShardV2Writer(path)
        w.write_entry_typed("model/weight", b"data", CONTENT_TYPE_TENSOR)
        w.write_entry_typed("config.json", b"{}", CONTENT_TYPE_JSON)
        w.close()

        schema = ShardSchema(name="test")
        schema.add_spec("model/weight", CONTENT_TYPE_TENSOR)
        schema.add_spec("config.json", CONTENT_TYPE_JSON)

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_missing_required(self, tmp_path):
        path = tmp_path / "missing.shard"
        w = ShardV2Writer(path)
        w.write_entry("only_one", b"data")
        w.close()

        schema = ShardSchema()
        schema.add_spec("only_one")
        schema.add_spec("missing_entry")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 1
        assert "missing_entry" in errors[0].message

    def test_optional_missing_ok(self, tmp_path):
        path = tmp_path / "optional.shard"
        w = ShardV2Writer(path)
        w.write_entry("required", b"data")
        w.close()

        schema = ShardSchema()
        schema.add_spec("required")
        schema.add_optional("optional_entry")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_wildcard_dot_pattern(self, tmp_path):
        path = tmp_path / "wild.shard"
        w = ShardV2Writer(path)
        w.write_entry("layer.0.weight", b"w0")
        w.write_entry("layer.1.weight", b"w1")
        w.write_entry("layer.2.weight", b"w2")
        w.close()

        schema = ShardSchema()
        schema.add_spec("layer.*.weight")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_wildcard_slash_pattern(self, tmp_path):
        """* should not cross / separator when used in slash-delimited paths."""
        path = tmp_path / "wildslash.shard"
        w = ShardV2Writer(path)
        w.write_entry("layer/0/weight", b"w0")
        w.write_entry("layer/1/weight", b"w1")
        w.close()

        schema = ShardSchema()
        schema.add_spec("layer/*/weight")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_prefix_wildcard(self, tmp_path):
        path = tmp_path / "prefix.shard"
        w = ShardV2Writer(path)
        w.write_entry("layers/0/attn/q", b"q")
        w.write_entry("layers/0/attn/k", b"k")
        w.close()

        schema = ShardSchema()
        schema.add_spec("layers/**")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_wrong_content_type(self, tmp_path):
        path = tmp_path / "wrongtype.shard"
        w = ShardV2Writer(path)
        w.write_entry_typed("data", b"not_a_tensor", CONTENT_TYPE_TEXT)
        w.close()

        schema = ShardSchema()
        schema.add_spec("data", CONTENT_TYPE_TENSOR)

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 1
        assert "data" in errors[0].message
        assert str(CONTENT_TYPE_TEXT) in errors[0].message

    def test_empty_schema_always_valid(self, tmp_path):
        path = tmp_path / "any.shard"
        w = ShardV2Writer(path)
        w.write_entry("anything", b"data")
        w.close()

        r = ShardV2Reader(path)
        errors = validate_schema(r, ShardSchema())
        assert len(errors) == 0

    def test_multiple_errors(self, tmp_path):
        path = tmp_path / "multi_err.shard"
        w = ShardV2Writer(path)
        w.write_entry("only_this", b"data")
        w.close()

        schema = ShardSchema()
        schema.add_spec("only_this")
        schema.add_spec("missing_a")
        schema.add_spec("missing_b")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 2
        patterns = {e.spec_pattern for e in errors}
        assert "missing_a" in patterns
        assert "missing_b" in patterns

    def test_exact_match_required(self, tmp_path):
        path = tmp_path / "exact.shard"
        w = ShardV2Writer(path)
        w.write_entry("embed.weight", b"e")
        w.close()

        schema = ShardSchema()
        schema.add_spec("embed.weight")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_wildcard_no_match_is_error(self, tmp_path):
        """A required wildcard spec with no matching entries is an error."""
        path = tmp_path / "no_wild_match.shard"
        w = ShardV2Writer(path)
        w.write_entry("other.entry", b"x")
        w.close()

        schema = ShardSchema()
        schema.add_spec("layer.*.weight")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 1
        assert "layer.*.weight" in errors[0].message

    def test_content_type_unknown_means_any(self, tmp_path):
        """A spec with content_type=UNKNOWN (0) accepts any content type."""
        path = tmp_path / "anytype.shard"
        w = ShardV2Writer(path)
        w.write_entry_typed("data", b"x", CONTENT_TYPE_JSON)
        w.close()

        schema = ShardSchema()
        schema.add_spec("data", CONTENT_TYPE_UNKNOWN)

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_add_spec_chaining(self, tmp_path):
        path = tmp_path / "chain.shard"
        w = ShardV2Writer(path)
        w.write_entry("a", b"1")
        w.write_entry("b", b"2")
        w.close()

        schema = ShardSchema()
        result = schema.add_spec("a").add_spec("b")
        assert result is schema

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 0

    def test_validation_error_fields(self, tmp_path):
        path = tmp_path / "fields.shard"
        w = ShardV2Writer(path)
        w.write_entry("present", b"data")
        w.close()

        schema = ShardSchema()
        schema.add_spec("absent.entry")

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 1
        err = errors[0]
        assert err.spec_pattern == "absent.entry"
        assert "absent.entry" in err.message

    def test_optional_wrong_type_still_errors(self, tmp_path):
        """Optional entries that DO exist still get type-checked."""
        path = tmp_path / "opt_wrong_type.shard"
        w = ShardV2Writer(path)
        w.write_entry_typed("meta", b"text", CONTENT_TYPE_TEXT)
        w.close()

        schema = ShardSchema()
        schema.add_optional("meta", CONTENT_TYPE_JSON)

        r = ShardV2Reader(path)
        errors = validate_schema(r, schema)
        assert len(errors) == 1
        assert "meta" in errors[0].message
