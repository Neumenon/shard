/**
 * Schema validation tests for shard-ts.
 */

import { describe, it, expect, beforeAll } from 'vitest';

import {
  ShardV2Reader,
  ShardV2Writer,
  initXxhash,
  createSchema,
  addSpec,
  validateSchema,
  CONTENT_TYPE_UNKNOWN,
  CONTENT_TYPE_TENSOR,
  CONTENT_TYPE_JSON,
  CONTENT_TYPE_TEXT,
} from '../index.js';

beforeAll(async () => {
  await initXxhash();
});

// ============================================================
// Helpers
// ============================================================

function makeReader(write: (w: ShardV2Writer) => void): ShardV2Reader {
  const w = new ShardV2Writer();
  write(w);
  return new ShardV2Reader(w.toBuffer());
}

// ============================================================
// Tests
// ============================================================

describe('schema validation', () => {
  it('valid schema — exact names and content types match', () => {
    const r = makeReader(w => {
      w.writeEntryTyped('model/weight', Buffer.from('data'), CONTENT_TYPE_TENSOR);
      w.writeEntryTyped('config.json', Buffer.from('{}'), CONTENT_TYPE_JSON);
    });

    const schema = createSchema('test');
    addSpec(schema, 'model/weight', CONTENT_TYPE_TENSOR);
    addSpec(schema, 'config.json', CONTENT_TYPE_JSON);

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('missing required entry produces one error', () => {
    const r = makeReader(w => {
      w.writeEntry('only_one', Buffer.from('data'));
    });

    const schema = createSchema();
    addSpec(schema, 'only_one');
    addSpec(schema, 'missing_entry');

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toContain('missing_entry');
  });

  it('optional missing entry is not an error', () => {
    const r = makeReader(w => {
      w.writeEntry('required', Buffer.from('data'));
    });

    const schema = createSchema();
    addSpec(schema, 'required');
    addSpec(schema, 'optional_entry', 0, true);

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('dot wildcard pattern matches layer.*.weight', () => {
    const r = makeReader(w => {
      w.writeEntry('layer.0.weight', Buffer.from('w0'));
      w.writeEntry('layer.1.weight', Buffer.from('w1'));
      w.writeEntry('layer.2.weight', Buffer.from('w2'));
    });

    const schema = createSchema();
    addSpec(schema, 'layer.*.weight');

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('slash wildcard pattern matches layer/*/weight', () => {
    const r = makeReader(w => {
      w.writeEntry('layer/0/weight', Buffer.from('w0'));
      w.writeEntry('layer/1/weight', Buffer.from('w1'));
    });

    const schema = createSchema();
    addSpec(schema, 'layer/*/weight');

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('prefix wildcard ** matches everything after prefix', () => {
    const r = makeReader(w => {
      w.writeEntry('layers/0/attn/q', Buffer.from('q'));
      w.writeEntry('layers/0/attn/k', Buffer.from('k'));
    });

    const schema = createSchema();
    addSpec(schema, 'layers/**');

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('wrong content type produces an error', () => {
    const r = makeReader(w => {
      w.writeEntryTyped('data', Buffer.from('not_a_tensor'), CONTENT_TYPE_TEXT);
    });

    const schema = createSchema();
    addSpec(schema, 'data', CONTENT_TYPE_TENSOR);

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toContain('data');
    expect(errors[0].message).toContain(String(CONTENT_TYPE_TEXT));
  });

  it('empty schema is always valid', () => {
    const r = makeReader(w => {
      w.writeEntry('anything', Buffer.from('data'));
    });

    expect(validateSchema(r, createSchema())).toHaveLength(0);
  });

  it('multiple missing entries produce multiple errors', () => {
    const r = makeReader(w => {
      w.writeEntry('only_this', Buffer.from('data'));
    });

    const schema = createSchema();
    addSpec(schema, 'only_this');
    addSpec(schema, 'missing_a');
    addSpec(schema, 'missing_b');

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(2);
    const patterns = new Set(errors.map(e => e.specPattern));
    expect(patterns.has('missing_a')).toBe(true);
    expect(patterns.has('missing_b')).toBe(true);
  });

  it('exact match required', () => {
    const r = makeReader(w => {
      w.writeEntry('embed.weight', Buffer.from('e'));
    });

    const schema = createSchema();
    addSpec(schema, 'embed.weight');

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('required wildcard with no matching entries is an error', () => {
    const r = makeReader(w => {
      w.writeEntry('other.entry', Buffer.from('x'));
    });

    const schema = createSchema();
    addSpec(schema, 'layer.*.weight');

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toContain('layer.*.weight');
  });

  it('content_type 0 (UNKNOWN) accepts any content type', () => {
    const r = makeReader(w => {
      w.writeEntryTyped('data', Buffer.from('x'), CONTENT_TYPE_JSON);
    });

    const schema = createSchema();
    addSpec(schema, 'data', CONTENT_TYPE_UNKNOWN);

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('undefined contentType also accepts any content type', () => {
    const r = makeReader(w => {
      w.writeEntryTyped('data', Buffer.from('x'), CONTENT_TYPE_JSON);
    });

    const schema = createSchema();
    schema.specs.push({ pattern: 'data' }); // no contentType set

    expect(validateSchema(r, schema)).toHaveLength(0);
  });

  it('addSpec returns the schema for chaining', () => {
    const schema = createSchema();
    const returned = addSpec(schema, 'a');
    expect(returned).toBe(schema);
  });

  it('ValidationError has correct specPattern and message fields', () => {
    const r = makeReader(w => {
      w.writeEntry('present', Buffer.from('data'));
    });

    const schema = createSchema();
    addSpec(schema, 'absent.entry');

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
    expect(errors[0].specPattern).toBe('absent.entry');
    expect(errors[0].message).toContain('absent.entry');
  });

  it('optional entry that exists gets type-checked', () => {
    const r = makeReader(w => {
      w.writeEntryTyped('meta', Buffer.from('text'), CONTENT_TYPE_TEXT);
    });

    const schema = createSchema();
    addSpec(schema, 'meta', CONTENT_TYPE_JSON, true);

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toContain('meta');
  });

  it('* does not cross . separator (no cross-component match)', () => {
    // "layer.*.weight" should NOT match "layer.0.extra.weight"
    const r = makeReader(w => {
      w.writeEntry('layer.0.extra.weight', Buffer.from('x'));
    });

    const schema = createSchema();
    addSpec(schema, 'layer.*.weight');

    // The wildcard only matches one component, so no match → required error.
    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
  });

  it('* does not cross / separator (no cross-path match)', () => {
    // "layer/*/weight" should NOT match "layer/0/sub/weight"
    const r = makeReader(w => {
      w.writeEntry('layer/0/sub/weight', Buffer.from('x'));
    });

    const schema = createSchema();
    addSpec(schema, 'layer/*/weight');

    const errors = validateSchema(r, schema);
    expect(errors).toHaveLength(1);
  });
});
