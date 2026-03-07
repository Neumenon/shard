/**
 * Tests for SampleShard TypeScript implementation.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

import { SampleShardWriter, SampleShardReader } from '../src/index.js';

describe('SampleShard', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sampleshard-test-'));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('round-trip', () => {
    it('should write and read samples', async () => {
      const filePath = path.join(tmpDir, 'test.smpl');

      const samples = new Map<bigint, unknown>([
        [100n, { input: [1, 2, 3], label: 0 }],
        [200n, { input: [4, 5, 6], label: 1 }],
        [300n, { input: [7, 8, 9], label: 2 }],
      ]);

      // Write
      const writer = new SampleShardWriter(filePath);
      await writer.open();
      for (const [id, sample] of samples) {
        await writer.addSample(id, sample);
      }
      await writer.close();

      // Read and verify
      const reader = new SampleShardReader(filePath);
      await reader.open();

      expect(reader.sampleCount()).toBe(3);

      // Verify by ID
      for (const [id, expected] of samples) {
        const actual = await reader.getSample(id);
        expect(actual).toEqual(expected);
      }

      // Verify IDs exist
      const ids = reader.getSampleIds();
      expect(ids.length).toBe(3);
      expect(ids).toContain(100n);
      expect(ids).toContain(200n);
      expect(ids).toContain(300n);

      await reader.close();
    });

    it('should iterate over samples', async () => {
      const filePath = path.join(tmpDir, 'iter.smpl');

      // Write
      const writer = new SampleShardWriter(filePath);
      await writer.open();
      for (let i = 0; i < 5; i++) {
        await writer.addSample(i, { id: i, value: i * 10 });
      }
      await writer.close();

      // Iterate
      const reader = new SampleShardReader(filePath);
      await reader.open();

      let count = 0;
      for await (const [sampleId, sample] of reader) {
        expect((sample as { id: number }).id).toBe(Number(sampleId));
        expect((sample as { value: number }).value).toBe(Number(sampleId) * 10);
        count++;
      }

      expect(count).toBe(5);

      await reader.close();
    });

    it('should check if sample exists', async () => {
      const filePath = path.join(tmpDir, 'has.smpl');

      const writer = new SampleShardWriter(filePath);
      await writer.open();
      await writer.addSample(1, { x: 1 });
      await writer.addSample(2, { x: 2 });
      await writer.close();

      const reader = new SampleShardReader(filePath);
      await reader.open();

      expect(reader.hasSample(1)).toBe(true);
      expect(reader.hasSample(2)).toBe(true);
      expect(reader.hasSample(3)).toBe(false);
      expect(reader.hasSample(999)).toBe(false);

      await reader.close();
    });

    it('should read batch by IDs', async () => {
      const filePath = path.join(tmpDir, 'batch.smpl');

      const writer = new SampleShardWriter(filePath);
      await writer.open();
      for (let i = 0; i < 10; i++) {
        await writer.addSample(i, { idx: i });
      }
      await writer.close();

      const reader = new SampleShardReader(filePath);
      await reader.open();

      const batch = await reader.getBatch([2, 5, 8]);
      expect(batch.length).toBe(3);
      expect((batch[0] as { idx: number }).idx).toBe(2);
      expect((batch[1] as { idx: number }).idx).toBe(5);
      expect((batch[2] as { idx: number }).idx).toBe(8);

      const rangeBatch = await reader.getBatchByRange(3, 6);
      expect(rangeBatch.length).toBe(3);
      expect((rangeBatch[0] as { idx: number }).idx).toBe(3);

      await reader.close();
    });
  });

  describe('error handling', () => {
    it('should throw on sample not found', async () => {
      const filePath = path.join(tmpDir, 'notfound.smpl');

      const writer = new SampleShardWriter(filePath);
      await writer.open();
      await writer.addSample(1, { x: 1 });
      await writer.close();

      const reader = new SampleShardReader(filePath);
      await reader.open();

      await expect(reader.getSample(999)).rejects.toThrow('Sample not found');

      await reader.close();
    });

    it('should throw on file not found', async () => {
      const reader = new SampleShardReader('/nonexistent/path.smpl');
      await expect(reader.open()).rejects.toThrow();
    });

    it('should throw on write after close', async () => {
      const filePath = path.join(tmpDir, 'closed.smpl');

      const writer = new SampleShardWriter(filePath);
      await writer.open();
      await writer.addSample(1, { x: 1 });
      await writer.close();

      await expect(writer.addSample(2, { x: 2 })).rejects.toThrow('Writer not open');
    });
  });

  describe('complex data', () => {
    it('should handle various JSON types', async () => {
      const filePath = path.join(tmpDir, 'types.smpl');

      const samples = new Map<bigint, unknown>([
        [1n, null],
        [2n, true],
        [3n, false],
        [4n, 42],
        [5n, 3.14159],
        [6n, 'hello'],
        [7n, [1, 2, 3]],
        [8n, { key: 'value' }],
        [9n, [{ nested: [1, 2] }, 'mixed', 3.14]],
      ]);

      const writer = new SampleShardWriter(filePath);
      await writer.open();
      for (const [id, sample] of samples) {
        await writer.addSample(id, sample);
      }
      await writer.close();

      const reader = new SampleShardReader(filePath);
      await reader.open();

      for (const [id, expected] of samples) {
        const actual = await reader.getSample(id);
        expect(actual).toEqual(expected);
      }

      await reader.close();
    });
  });
});
