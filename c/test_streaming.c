/*
 * test_streaming.c — Roundtrip tests for shard_v2 streaming writer.
 *
 * Tests:
 *   1.  Basic two-entry roundtrip
 *   2.  FLAG_STREAMING is set in header flags
 *   3.  Single entry roundtrip
 *   4.  Role propagated to header
 *   5.  total_file_size matches file size on disk
 *   6.  Alignment 64 default — entries are 64-byte aligned
 *   7.  Alignment 16
 *   8.  Alignment none — entries packed tightly
 *   9.  Many entries (100)
 *  10.  Large data (1 MB)
 *  11.  Lookup by name
 *  12.  Checksum stored correctly
 *  13.  Binary data — all 256 byte values
 *  14.  Fewer entries than max_entries
 *  15.  DEFAULT_FLAGS also present alongside FLAG_STREAMING
 */

#include "shard_v2.h"

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
    char* path = (char*)malloc(64);
    if (!path) return NULL;
    snprintf(path, 64, "/tmp/shard_stream_test_%d_%d.shard", (int)getpid(), counter);
    return path;
}

/* ============================================================
 * Test 1: Basic two-entry roundtrip
 * ============================================================ */
static void test_basic_roundtrip(void) {
    printf("[ Test 1: basic two-entry roundtrip ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 10);
    check(sw != NULL, "stream_writer_new");
    if (!sw) { free(path); return; }

    check(shard_v2_stream_writer_begin_data(sw) == 0, "begin_data");
    check(shard_v2_stream_writer_write_entry(sw, "hello", (const uint8_t*)"world", 5) == 0,
          "write_entry hello");
    check(shard_v2_stream_writer_write_entry(sw, "foo", (const uint8_t*)"bar", 3) == 0,
          "write_entry foo");
    check(shard_v2_stream_writer_finalize(sw) == 0, "finalize");
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open after finalize");
    if (r) {
        check(shard_v2_entry_count(r) == 2, "entry_count == 2");

        const char* n0 = shard_v2_entry_name(r, 0);
        check(n0 && strcmp(n0, "hello") == 0, "entry[0].name == \"hello\"");

        const char* n1 = shard_v2_entry_name(r, 1);
        check(n1 && strcmp(n1, "foo") == 0, "entry[1].name == \"foo\"");

        size_t sz;
        const uint8_t* d0 = shard_v2_read_entry(r, 0, &sz);
        check(d0 && sz == 5 && memcmp(d0, "world", 5) == 0, "entry[0] data == \"world\"");

        const uint8_t* d1 = shard_v2_read_entry(r, 1, &sz);
        check(d1 && sz == 3 && memcmp(d1, "bar", 3) == 0, "entry[1] data == \"bar\"");

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 2: FLAG_STREAMING is set in header flags
 * ============================================================ */
static void test_flag_streaming(void) {
    printf("[ Test 2: FLAG_STREAMING set ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 5);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "x", (const uint8_t*)"data", 4);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open after finalize");
    if (r) {
        const shard_v2_header_t* h = shard_v2_header(r);
        check((h->flags & FLAG_STREAMING) != 0, "FLAG_STREAMING is set");
        check((h->flags & DEFAULT_FLAGS) == DEFAULT_FLAGS, "DEFAULT_FLAGS bits also present");
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 3: Single entry roundtrip
 * ============================================================ */
static void test_single_entry(void) {
    printf("[ Test 3: single entry roundtrip ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 5);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "only", (const uint8_t*)"payload", 7);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        check(shard_v2_entry_count(r) == 1, "entry_count == 1");

        size_t sz;
        const uint8_t* d = shard_v2_read_entry(r, 0, &sz);
        check(d && sz == 7 && memcmp(d, "payload", 7) == 0, "data == \"payload\"");

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 4: Role propagated to header
 * ============================================================ */
static void test_role(void) {
    printf("[ Test 4: role propagated ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_SAMPLE, 5);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "a", (const uint8_t*)"1", 1);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        const shard_v2_header_t* h = shard_v2_header(r);
        check(h->role == ROLE_SAMPLE, "header.role == ROLE_SAMPLE");
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 5: total_file_size matches file size on disk
 * ============================================================ */
static void test_total_file_size(void) {
    printf("[ Test 5: total_file_size matches disk size ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 5);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "item", (const uint8_t*)"hello", 5);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    /* Determine file size by opening and seeking */
    long file_sz = 0;
    {
        FILE* f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            file_sz = ftell(f);
            fclose(f);
        }
    }

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        const shard_v2_header_t* h = shard_v2_header(r);
        check((long)h->total_file_size == file_sz, "header.total_file_size == file size on disk");
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 6: Alignment 64 (default) — entries are 64-byte aligned
 * ============================================================ */
static void test_alignment_64(void) {
    printf("[ Test 6: alignment 64 default ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 10);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    /* default alignment is 64 */
    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "a", (const uint8_t*)"data_a", 6);
    shard_v2_stream_writer_write_entry(sw, "b", (const uint8_t*)"data_b", 6);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        const shard_v2_header_t* h = shard_v2_header(r);
        check(h->alignment == ALIGN_64, "header.alignment == 64");

        for (uint32_t i = 0; i < shard_v2_entry_count(r); i++) {
            const shard_v2_index_entry_t* e = shard_v2_get_entry(r, i);
            char desc[64];
            snprintf(desc, sizeof(desc), "entry[%u].data_offset %% 64 == 0", i);
            check(e && (e->data_offset % 64 == 0), desc);
        }

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 7: Alignment 16
 * ============================================================ */
static void test_alignment_16(void) {
    printf("[ Test 7: alignment 16 ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 10);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_set_alignment(sw, ALIGN_16);
    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "x", (const uint8_t*)"hello", 5);
    shard_v2_stream_writer_write_entry(sw, "y", (const uint8_t*)"world_longer", 12);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        const shard_v2_header_t* h = shard_v2_header(r);
        check(h->alignment == ALIGN_16, "header.alignment == 16");
        check(h->data_section_offset % 16 == 0, "data_section_offset 16-aligned");

        for (uint32_t i = 0; i < shard_v2_entry_count(r); i++) {
            const shard_v2_index_entry_t* e = shard_v2_get_entry(r, i);
            char desc[64];
            snprintf(desc, sizeof(desc), "entry[%u].data_offset %% 16 == 0", i);
            check(e && (e->data_offset % 16 == 0), desc);
        }

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 8: Alignment none — entries packed tightly
 * ============================================================ */
static void test_alignment_none(void) {
    printf("[ Test 8: alignment none ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 10);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_set_alignment(sw, ALIGN_NONE);
    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "a", (const uint8_t*)"x", 1);
    shard_v2_stream_writer_write_entry(sw, "b", (const uint8_t*)"yy", 2);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        const shard_v2_header_t* h = shard_v2_header(r);
        check(h->alignment == ALIGN_NONE, "header.alignment == 0");

        const shard_v2_index_entry_t* e0 = shard_v2_get_entry(r, 0);
        const shard_v2_index_entry_t* e1 = shard_v2_get_entry(r, 1);
        check(e0 && e1, "both entries present");
        if (e0 && e1) {
            check(e1->data_offset == e0->data_offset + e0->disk_size,
                  "entries packed tightly (no alignment gap)");
        }

        size_t sz;
        const uint8_t* d0 = shard_v2_read_entry(r, 0, &sz);
        check(d0 && sz == 1 && d0[0] == 'x', "entry[0] data correct");

        const uint8_t* d1 = shard_v2_read_entry(r, 1, &sz);
        check(d1 && sz == 2 && memcmp(d1, "yy", 2) == 0, "entry[1] data correct");

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 9: Many entries (100)
 * ============================================================ */
static void test_many_entries(void) {
    printf("[ Test 9: many entries (100) ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    const int N = 100;
    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, (uint32_t)N);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    for (int i = 0; i < N; i++) {
        char name[32], data[32];
        snprintf(name, sizeof(name), "entry_%04d", i);
        snprintf(data, sizeof(data), "data_%d", i);
        int rc = shard_v2_stream_writer_write_entry(sw, name,
                     (const uint8_t*)data, strlen(data));
        if (rc != 0) { check(false, "write_entry failed mid-loop"); break; }
    }
    check(shard_v2_stream_writer_finalize(sw) == 0, "finalize");
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        check(shard_v2_entry_count(r) == (uint32_t)N, "entry_count == 100");

        bool all_ok = true;
        for (int i = 0; i < N; i++) {
            char expected_name[32], expected_data[32];
            snprintf(expected_name, sizeof(expected_name), "entry_%04d", i);
            snprintf(expected_data, sizeof(expected_data), "data_%d", i);

            const char* nm = shard_v2_entry_name(r, (uint32_t)i);
            if (!nm || strcmp(nm, expected_name) != 0) { all_ok = false; break; }

            size_t sz;
            const uint8_t* d = shard_v2_read_entry(r, (uint32_t)i, &sz);
            size_t exp_len = strlen(expected_data);
            if (!d || sz != exp_len || memcmp(d, expected_data, exp_len) != 0) {
                all_ok = false; break;
            }
        }
        check(all_ok, "all 100 entries name+data correct");

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 10: Large data (1 MB)
 * ============================================================ */
static void test_large_data(void) {
    printf("[ Test 10: large data (1 MB) ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    size_t large_sz = 1024 * 1024;
    uint8_t* large_data = (uint8_t*)malloc(large_sz);
    check(large_data != NULL, "malloc 1MB");
    if (!large_data) { free(path); return; }

    for (size_t i = 0; i < large_sz; i++) {
        large_data[i] = (uint8_t)(i % 256);
    }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 5);
    if (!sw) { check(false, "stream_writer_new"); free(large_data); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    check(shard_v2_stream_writer_write_entry(sw, "big_tensor", large_data, large_sz) == 0,
          "write_entry 1MB");
    check(shard_v2_stream_writer_finalize(sw) == 0, "finalize");
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        check(shard_v2_entry_count(r) == 1, "entry_count == 1");

        size_t sz;
        const uint8_t* got = shard_v2_read_entry(r, 0, &sz);
        check(got && sz == large_sz, "read 1MB");
        if (got) {
            check(memcmp(got, large_data, large_sz) == 0, "1MB data matches byte-for-byte");
        }

        shard_v2_close(r);
    }

    free(large_data);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 11: Lookup by name
 * ============================================================ */
static void test_lookup(void) {
    printf("[ Test 11: lookup by name ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 10);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    const uint8_t d[] = {1};
    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "alpha", d, 1);
    shard_v2_stream_writer_write_entry(sw, "beta",  d, 1);
    shard_v2_stream_writer_write_entry(sw, "gamma", d, 1);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        check(shard_v2_lookup(r, "alpha") == 0,  "lookup alpha == 0");
        check(shard_v2_lookup(r, "beta")  == 1,  "lookup beta == 1");
        check(shard_v2_lookup(r, "gamma") == 2,  "lookup gamma == 2");
        check(shard_v2_lookup(r, "missing") == -1, "lookup missing == -1");

        check(shard_v2_has_entry(r, "beta"),    "has_entry(beta) == true");
        check(!shard_v2_has_entry(r, "nope"),   "has_entry(nope) == false");

        size_t sz;
        const uint8_t* got = shard_v2_read_entry_by_name(r, "gamma", &sz);
        check(got && sz == 1 && got[0] == 1, "read_entry_by_name(gamma) correct");

        const uint8_t* miss = shard_v2_read_entry_by_name(r, "nonexistent", &sz);
        check(miss == NULL, "read_entry_by_name(nonexistent) == NULL");

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 12: Checksum stored correctly
 * ============================================================ */
static void test_checksum(void) {
    printf("[ Test 12: checksum stored correctly ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    const uint8_t data[] = "checksum_test_data";
    size_t data_len = sizeof(data) - 1;

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 5);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "data", data, data_len);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        const shard_v2_index_entry_t* e = shard_v2_get_entry(r, 0);
        uint32_t expected = shard_v2_crc32c(data, data_len);
        check(e && e->checksum == expected, "stored checksum == computed CRC32C");
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 13: Binary data — all 256 byte values
 * ============================================================ */
static void test_binary_allbytes(void) {
    printf("[ Test 13: binary data — all 256 byte values ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    uint8_t all_bytes[256];
    for (int i = 0; i < 256; i++) all_bytes[i] = (uint8_t)i;

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 5);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "allbytes", all_bytes, 256);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        size_t sz;
        const uint8_t* got = shard_v2_read_entry(r, 0, &sz);
        check(got && sz == 256, "read 256 bytes");
        if (got) {
            check(memcmp(got, all_bytes, 256) == 0, "all 256 byte values preserved");
        }
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 14: Fewer entries than max_entries
 * ============================================================ */
static void test_fewer_than_max(void) {
    printf("[ Test 14: fewer entries than max_entries ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 1000);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    shard_v2_stream_writer_write_entry(sw, "only_one", (const uint8_t*)"data", 4);
    shard_v2_stream_writer_finalize(sw);
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        check(shard_v2_entry_count(r) == 1, "entry_count == 1");
        size_t sz;
        const uint8_t* d = shard_v2_read_entry(r, 0, &sz);
        check(d && sz == 4 && memcmp(d, "data", 4) == 0, "data correct");
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 15: Empty shard (no entries written)
 * ============================================================ */
static void test_empty_shard(void) {
    printf("[ Test 15: empty shard (zero entries) ]\n");

    char* path = make_tmpfile();
    if (!path) { check(false, "make_tmpfile"); return; }

    shard_v2_stream_writer_t* sw = shard_v2_stream_writer_new(path, ROLE_MOSH, 10);
    if (!sw) { check(false, "stream_writer_new"); free(path); return; }

    shard_v2_stream_writer_begin_data(sw);
    /* Write no entries */
    check(shard_v2_stream_writer_finalize(sw) == 0, "finalize with zero entries");
    shard_v2_stream_writer_free(sw);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open");
    if (r) {
        check(shard_v2_entry_count(r) == 0, "entry_count == 0");
        check(shard_v2_lookup(r, "anything") == -1, "lookup == -1");
        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("=== shard_v2 streaming writer tests ===\n\n");

    test_basic_roundtrip();     printf("\n");
    test_flag_streaming();      printf("\n");
    test_single_entry();        printf("\n");
    test_role();                printf("\n");
    test_total_file_size();     printf("\n");
    test_alignment_64();        printf("\n");
    test_alignment_16();        printf("\n");
    test_alignment_none();      printf("\n");
    test_many_entries();        printf("\n");
    test_large_data();          printf("\n");
    test_lookup();              printf("\n");
    test_checksum();            printf("\n");
    test_binary_allbytes();     printf("\n");
    test_fewer_than_max();      printf("\n");
    test_empty_shard();         printf("\n");

    printf("=== Results: %d/%d passed, %d failed ===\n",
           g_passed, g_total, g_failed);

    return (g_failed == 0) ? 0 : 1;
}
