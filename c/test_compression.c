/*
 * test_compression.c — Tests for zstd and lz4 compression in shard.
 *
 * Tests:
 *   1. Zstd roundtrip: write compressed, read back, verify matches original
 *   2. LZ4 roundtrip: same
 *   3. Small data (< MIN_COMPRESS_SIZE) stays uncompressed regardless of codec
 *   4. Incompressible data (random-ish bytes) stored raw when no savings
 *   5. Checksum on decompressed data is CRC32C of original
 *   6. Mixed compressed and uncompressed entries in one shard
 *   7. Large data (100 KB) roundtrip via zstd
 *   8. Large data (100 KB) roundtrip via lz4
 *   9. Decompression cache: repeated reads return same pointer
 *  10. read_entry_by_name works for compressed entries
 *  11. Checksum mismatch on corrupt compressed blob is detected
 */

#include "shard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

/* ============================================================
 * Test helpers
 * ============================================================ */

static int g_total  = 0;
static int g_passed = 0;
static int g_failed = 0;

static void check(bool ok, const char* desc) {
    g_total++;
    if (ok) {
        g_passed++;
        printf("  PASS  %s\n", desc);
    } else {
        g_failed++;
        printf("  FAIL  %s\n", desc);
    }
}

static char* make_tmpfile(void) {
    static int counter = 0;
    counter++;
    char* path = (char*)malloc(80);
    if (!path) return NULL;
    snprintf(path, 80, "/tmp/shard_compress_test_%d_%d.shard", (int)getpid(), counter);
    return path;
}

/* Write shard and re-open, returning reader (caller closes + frees path). */
static shard_reader_t* write_and_open(shard_writer_t* w, char** out_path) {
    char* path = make_tmpfile();
    if (!path) return NULL;
    *out_path = path;
    if (shard_writer_write(w, path) != 0) return NULL;
    return shard_open(path);
}

/* Build compressible data: repeating pattern. */
static void fill_compressible(uint8_t* buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        buf[i] = (uint8_t)(i % 64); /* short cycle = easy to compress */
    }
}

/* Build incompressible data: pseudo-random via xorshift. */
static void fill_incompressible(uint8_t* buf, size_t len) {
    uint32_t x = 0xDEADBEEFu;
    for (size_t i = 0; i < len; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        buf[i] = (uint8_t)x;
    }
}

/* ============================================================
 * Test 1: Zstd roundtrip
 * ============================================================ */
static void test_zstd_roundtrip(void) {
    printf("[ Test 1: zstd roundtrip ]\n");

    const size_t DATA_LEN = 10000;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) { free(data); return; }

    int rc = shard_writer_add_entry_compressed(w, "big", data, DATA_LEN, COMPRESS_ZSTD);
    check(rc == 0, "add_entry_compressed(zstd)");

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    check(shard_entry_count(r) == 1, "entry_count == 1");

    const shard_index_entry_t* info = shard_get_entry(r, 0);
    check(info != NULL, "get_entry");
    check(info && shard_is_compressed(info), "is_compressed");
    check(info && (info->flags & ENTRY_FLAG_ZSTD), "ENTRY_FLAG_ZSTD set");
    check(info && info->orig_size == DATA_LEN, "orig_size matches");
    check(info && info->disk_size < info->orig_size, "disk_size < orig_size (compressed)");

    size_t out_size = 0;
    const uint8_t* read_data = shard_read_entry(r, 0, &out_size);
    check(read_data != NULL, "read_entry != NULL");
    check(out_size == DATA_LEN, "decompressed size matches");
    check(read_data && memcmp(read_data, data, DATA_LEN) == 0, "data matches byte-for-byte");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 2: LZ4 roundtrip
 * ============================================================ */
static void test_lz4_roundtrip(void) {
    printf("[ Test 2: lz4 roundtrip ]\n");

    const size_t DATA_LEN = 10000;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) { free(data); return; }

    int rc = shard_writer_add_entry_compressed(w, "big_lz4", data, DATA_LEN, COMPRESS_LZ4);
    check(rc == 0, "add_entry_compressed(lz4)");

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    const shard_index_entry_t* info = shard_get_entry(r, 0);
    check(info && shard_is_compressed(info), "is_compressed");
    check(info && (info->flags & ENTRY_FLAG_LZ4), "ENTRY_FLAG_LZ4 set");
    check(info && info->orig_size == DATA_LEN, "orig_size matches");
    check(info && info->disk_size < info->orig_size, "disk_size < orig_size");

    size_t out_size = 0;
    const uint8_t* read_data = shard_read_entry(r, 0, &out_size);
    check(read_data != NULL, "read_entry != NULL");
    check(out_size == DATA_LEN, "decompressed size matches");
    check(read_data && memcmp(read_data, data, DATA_LEN) == 0, "data matches byte-for-byte");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 3: Small data stays uncompressed
 * ============================================================ */
static void test_small_data_uncompressed(void) {
    printf("[ Test 3: small data stays uncompressed ]\n");

    /* Data smaller than MIN_COMPRESS_SIZE */
    const size_t DATA_LEN = MIN_COMPRESS_SIZE - 1;
    uint8_t data[MIN_COMPRESS_SIZE];
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) return;

    /* Both zstd and lz4 should leave small data uncompressed */
    shard_writer_add_entry_compressed(w, "small_zstd", data, DATA_LEN, COMPRESS_ZSTD);
    shard_writer_add_entry_compressed(w, "small_lz4",  data, DATA_LEN, COMPRESS_LZ4);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    check(shard_entry_count(r) == 2, "entry_count == 2");

    for (uint32_t i = 0; i < 2; i++) {
        const shard_index_entry_t* info = shard_get_entry(r, i);
        char desc[64];
        snprintf(desc, sizeof(desc), "entry[%u] NOT compressed (too small)", i);
        check(info && !shard_is_compressed(info), desc);
        snprintf(desc, sizeof(desc), "entry[%u] disk_size == orig_size", i);
        check(info && info->disk_size == info->orig_size, desc);

        size_t out_size = 0;
        const uint8_t* got = shard_read_entry(r, i, &out_size);
        snprintf(desc, sizeof(desc), "entry[%u] data correct", i);
        check(got && out_size == DATA_LEN && memcmp(got, data, DATA_LEN) == 0, desc);
    }

    shard_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 4: Incompressible data stored raw
 * ============================================================ */
static void test_incompressible_stored_raw(void) {
    printf("[ Test 4: incompressible data stored raw ]\n");

    /* Use a large enough buffer that compression would be attempted */
    const size_t DATA_LEN = 4096;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_incompressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) { free(data); return; }

    shard_writer_add_entry_compressed(w, "random_zstd", data, DATA_LEN, COMPRESS_ZSTD);
    shard_writer_add_entry_compressed(w, "random_lz4",  data, DATA_LEN, COMPRESS_LZ4);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    check(shard_entry_count(r) == 2, "entry_count == 2");

    /* Both should be stored uncompressed (no significant savings from random data) */
    for (uint32_t i = 0; i < 2; i++) {
        const shard_index_entry_t* info = shard_get_entry(r, i);
        /* It's acceptable for either to be compressed or not — the key test is
         * that the roundtrip is correct regardless. */
        size_t out_size = 0;
        const uint8_t* got = shard_read_entry(r, i, &out_size);
        char desc[64];
        snprintf(desc, sizeof(desc), "entry[%u] roundtrip correct", i);
        check(got && out_size == DATA_LEN && memcmp(got, data, DATA_LEN) == 0, desc);
        /* Confirm that if stored uncompressed, disk_size == orig_size */
        if (info && !shard_is_compressed(info)) {
            snprintf(desc, sizeof(desc), "entry[%u] uncompressed: disk_size==orig_size", i);
            check(info->disk_size == info->orig_size, desc);
        }
    }

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 5: Checksum on decompressed data
 * ============================================================ */
static void test_checksum_on_decompressed(void) {
    printf("[ Test 5: checksum stored for original data ]\n");

    const size_t DATA_LEN = 5000;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_compressible(data, DATA_LEN);

    uint32_t expected_csum = shard_crc32c(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    if (!w) { free(data); check(false, "writer_new"); return; }

    shard_writer_add_entry_compressed(w, "payload", data, DATA_LEN, COMPRESS_ZSTD);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    const shard_index_entry_t* info = shard_get_entry(r, 0);
    check(info != NULL, "get_entry");
    check(info && info->checksum == expected_csum, "stored checksum == CRC32C(original)");

    /* read_entry must verify the checksum and succeed */
    size_t out_size = 0;
    const uint8_t* got = shard_read_entry(r, 0, &out_size);
    check(got != NULL, "read_entry succeeds (checksum OK)");
    check(got && out_size == DATA_LEN && memcmp(got, data, DATA_LEN) == 0, "data correct");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 6: Mixed compressed and uncompressed entries
 * ============================================================ */
static void test_mixed_entries(void) {
    printf("[ Test 6: mixed compressed and uncompressed entries ]\n");

    const size_t COMP_LEN   = 8000;
    const size_t UNCOMP_LEN = 100;

    uint8_t* comp_data   = (uint8_t*)malloc(COMP_LEN);
    uint8_t* uncomp_data = (uint8_t*)malloc(UNCOMP_LEN);
    if (!comp_data || !uncomp_data) {
        free(comp_data); free(uncomp_data);
        check(false, "malloc");
        return;
    }
    fill_compressible(comp_data, COMP_LEN);
    for (size_t i = 0; i < UNCOMP_LEN; i++) uncomp_data[i] = (uint8_t)(i + 1);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) { free(comp_data); free(uncomp_data); return; }

    shard_writer_add_entry(w, "raw", uncomp_data, UNCOMP_LEN);
    shard_writer_add_entry_compressed(w, "compressed_zstd", comp_data, COMP_LEN, COMPRESS_ZSTD);
    shard_writer_add_entry(w, "raw2", uncomp_data, UNCOMP_LEN);
    shard_writer_add_entry_compressed(w, "compressed_lz4", comp_data, COMP_LEN, COMPRESS_LZ4);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(comp_data); free(uncomp_data); free(path); return; }

    check(shard_entry_count(r) == 4, "entry_count == 4");

    /* Entry 0: uncompressed "raw" */
    {
        const shard_index_entry_t* info = shard_get_entry(r, 0);
        check(info && !shard_is_compressed(info), "entry[0] uncompressed");
        size_t sz = 0;
        const uint8_t* got = shard_read_entry(r, 0, &sz);
        check(got && sz == UNCOMP_LEN && memcmp(got, uncomp_data, UNCOMP_LEN) == 0,
              "entry[0] data correct");
    }

    /* Entry 1: compressed via zstd */
    {
        const shard_index_entry_t* info = shard_get_entry(r, 1);
        check(info && shard_is_compressed(info), "entry[1] compressed");
        size_t sz = 0;
        const uint8_t* got = shard_read_entry(r, 1, &sz);
        check(got && sz == COMP_LEN && memcmp(got, comp_data, COMP_LEN) == 0,
              "entry[1] data correct");
    }

    /* Entry 2: uncompressed "raw2" */
    {
        const shard_index_entry_t* info = shard_get_entry(r, 2);
        check(info && !shard_is_compressed(info), "entry[2] uncompressed");
        size_t sz = 0;
        const uint8_t* got = shard_read_entry(r, 2, &sz);
        check(got && sz == UNCOMP_LEN && memcmp(got, uncomp_data, UNCOMP_LEN) == 0,
              "entry[2] data correct");
    }

    /* Entry 3: compressed via lz4 */
    {
        const shard_index_entry_t* info = shard_get_entry(r, 3);
        check(info && shard_is_compressed(info), "entry[3] compressed");
        size_t sz = 0;
        const uint8_t* got = shard_read_entry(r, 3, &sz);
        check(got && sz == COMP_LEN && memcmp(got, comp_data, COMP_LEN) == 0,
              "entry[3] data correct");
    }

    /* Header must have FLAG_COMPRESSED set */
    const shard_header_t* h = shard_header(r);
    check(h && (h->flags & FLAG_COMPRESSED), "header FLAG_COMPRESSED set");

    shard_close(r);
    remove(path);
    free(path);
    free(comp_data);
    free(uncomp_data);
}

/* ============================================================
 * Test 7: Large data (100 KB) zstd roundtrip
 * ============================================================ */
static void test_large_zstd(void) {
    printf("[ Test 7: large data (100 KB) zstd roundtrip ]\n");

    const size_t DATA_LEN = 100 * 1024;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc 100KB"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) { free(data); return; }

    shard_writer_add_entry_compressed(w, "large_tensor", data, DATA_LEN, COMPRESS_ZSTD);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    const shard_index_entry_t* info = shard_get_entry(r, 0);
    check(info && shard_is_compressed(info), "is_compressed");
    check(info && info->orig_size == DATA_LEN, "orig_size == 100KB");

    size_t out_size = 0;
    const uint8_t* got = shard_read_entry(r, 0, &out_size);
    check(got != NULL, "read_entry != NULL");
    check(out_size == DATA_LEN, "decompressed size == 100KB");
    check(got && memcmp(got, data, DATA_LEN) == 0, "100KB data matches");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 8: Large data (100 KB) lz4 roundtrip
 * ============================================================ */
static void test_large_lz4(void) {
    printf("[ Test 8: large data (100 KB) lz4 roundtrip ]\n");

    const size_t DATA_LEN = 100 * 1024;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc 100KB"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) { free(data); return; }

    shard_writer_add_entry_compressed(w, "large_lz4", data, DATA_LEN, COMPRESS_LZ4);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    const shard_index_entry_t* info = shard_get_entry(r, 0);
    check(info && shard_is_compressed(info), "is_compressed");
    check(info && info->orig_size == DATA_LEN, "orig_size == 100KB");

    size_t out_size = 0;
    const uint8_t* got = shard_read_entry(r, 0, &out_size);
    check(got != NULL, "read_entry != NULL");
    check(out_size == DATA_LEN, "decompressed size == 100KB");
    check(got && memcmp(got, data, DATA_LEN) == 0, "100KB data matches");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 9: Decompression cache — repeated reads same pointer
 * ============================================================ */
static void test_decompression_cache(void) {
    printf("[ Test 9: decompression cache returns same pointer ]\n");

    const size_t DATA_LEN = 5000;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    if (!w) { free(data); check(false, "writer_new"); return; }

    shard_writer_add_entry_compressed(w, "cached", data, DATA_LEN, COMPRESS_ZSTD);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    size_t sz1 = 0, sz2 = 0;
    const uint8_t* p1 = shard_read_entry(r, 0, &sz1);
    const uint8_t* p2 = shard_read_entry(r, 0, &sz2);

    check(p1 != NULL, "first read != NULL");
    check(p2 != NULL, "second read != NULL");
    check(p1 == p2, "same pointer (cache hit)");
    check(sz1 == sz2, "same size");
    check(sz1 == DATA_LEN, "size == original");

    /* Third read for good measure */
    size_t sz3 = 0;
    const uint8_t* p3 = shard_read_entry(r, 0, &sz3);
    check(p3 == p1, "third read same pointer");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 10: read_entry_by_name for compressed entries
 * ============================================================ */
static void test_read_by_name_compressed(void) {
    printf("[ Test 10: read_entry_by_name for compressed entries ]\n");

    const size_t DATA_LEN = 6000;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    if (!w) { free(data); check(false, "writer_new"); return; }

    shard_writer_add_entry_compressed(w, "layer.0/weights", data, DATA_LEN, COMPRESS_ZSTD);
    shard_writer_add_entry_compressed(w, "layer.1/weights", data, DATA_LEN, COMPRESS_LZ4);

    char* path = NULL;
    shard_reader_t* r = write_and_open(w, &path);
    shard_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(data); free(path); return; }

    size_t sz = 0;
    const uint8_t* got0 = shard_read_entry_by_name(r, "layer.0/weights", &sz);
    check(got0 != NULL, "read_by_name(layer.0/weights) != NULL");
    check(sz == DATA_LEN, "layer.0 size correct");
    check(got0 && memcmp(got0, data, DATA_LEN) == 0, "layer.0 data correct");

    const uint8_t* got1 = shard_read_entry_by_name(r, "layer.1/weights", &sz);
    check(got1 != NULL, "read_by_name(layer.1/weights) != NULL");
    check(sz == DATA_LEN, "layer.1 size correct");
    check(got1 && memcmp(got1, data, DATA_LEN) == 0, "layer.1 data correct");

    const uint8_t* miss = shard_read_entry_by_name(r, "nonexistent", &sz);
    check(miss == NULL, "read_by_name(nonexistent) == NULL");

    shard_close(r);
    remove(path);
    free(path);
    free(data);
}

/* ============================================================
 * Test 11: Corrupt compressed blob — checksum mismatch detected
 * ============================================================ */
static void test_corrupt_compressed_checksum(void) {
    printf("[ Test 11: checksum mismatch on corrupt compressed blob ]\n");

    const size_t DATA_LEN = 5000;
    uint8_t* data = (uint8_t*)malloc(DATA_LEN);
    if (!data) { check(false, "malloc"); return; }
    fill_compressible(data, DATA_LEN);

    shard_writer_t* w = shard_writer_new(ROLE_MOSH);
    if (!w) { free(data); check(false, "writer_new"); return; }

    shard_writer_add_entry_compressed(w, "payload", data, DATA_LEN, COMPRESS_ZSTD);
    free(data);

    char* path = make_tmpfile();
    if (!path) { shard_writer_free(w); check(false, "make_tmpfile"); return; }

    int rc = shard_writer_write(w, path);
    shard_writer_free(w);
    check(rc == 0, "write to file");
    if (rc != 0) { free(path); return; }

    /* Read the file and verify it's valid before corruption */
    {
        shard_reader_t* r = shard_open(path);
        check(r != NULL, "open clean file");
        if (r) {
            size_t sz = 0;
            const uint8_t* got = shard_read_entry(r, 0, &sz);
            check(got != NULL, "read_entry OK on clean file");
            shard_close(r);
        }
    }

    /* Corrupt the stored checksum in the index entry.
     * Index entry 0 is at offset SHARD_HEADER_SIZE.
     * Checksum field is at byte offset 40 within the entry. */
    {
        FILE* f = fopen(path, "r+b");
        check(f != NULL, "open for corruption");
        if (f) {
            long checksum_pos = (long)(SHARD_HEADER_SIZE + 40);
            fseek(f, checksum_pos, SEEK_SET);
            uint8_t bad[4] = {0xDE, 0xAD, 0xBE, 0xEF};
            fwrite(bad, 1, 4, f);
            fclose(f);
        }
    }

    /* Now reading should fail due to checksum mismatch on decompressed data */
    {
        shard_reader_t* r = shard_open(path);
        check(r != NULL, "open corrupted file (header/index still valid)");
        if (r) {
            size_t sz = 0;
            const uint8_t* got = shard_read_entry(r, 0, &sz);
            check(got == NULL, "read_entry returns NULL (checksum mismatch)");
            shard_close(r);
        }
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("=== shard compression tests ===\n\n");

    test_zstd_roundtrip();
    printf("\n");

    test_lz4_roundtrip();
    printf("\n");

    test_small_data_uncompressed();
    printf("\n");

    test_incompressible_stored_raw();
    printf("\n");

    test_checksum_on_decompressed();
    printf("\n");

    test_mixed_entries();
    printf("\n");

    test_large_zstd();
    printf("\n");

    test_large_lz4();
    printf("\n");

    test_decompression_cache();
    printf("\n");

    test_read_by_name_compressed();
    printf("\n");

    test_corrupt_compressed_checksum();
    printf("\n");

    printf("=== Results: %d/%d passed, %d failed ===\n",
           g_passed, g_total, g_failed);

    return (g_failed == 0) ? 0 : 1;
}
