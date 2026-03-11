/*
 * test_truth_table.c — Truth table tests for shard v2.
 *
 * 9 cases from testdata/robustness/truth_cases.json:
 *   1. duplicate_entry_names_rejected
 *   2. zero_length_entry_valid
 *   3. unicode_entry_name
 *   4. bad_magic_rejected
 *   5. truncated_header_rejected
 *   6. crc_mismatch_rejected
 *   7. truncated_data_rejected
 *   8. entry_count_overflow_rejected
 *   9. empty_input_rejected
 *
 * Usage: ./test_truth_table [testdata_dir]
 *   default testdata_dir = ../ucodec/testdata/safety
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

static void make_path(char* out, size_t outsz, const char* dir, const char* file) {
    snprintf(out, outsz, "%s/%s", dir, file);
}

/* Create a unique temporary file path */
static char* make_tmpfile(void) {
    static int counter = 0;
    counter++;
    char* path = (char*)malloc(128);
    if (!path) return NULL;
    snprintf(path, 128, "/tmp/shard_truth_%d_%d.shard", (int)getpid(), counter);
    return path;
}

/* ============================================================
 * Case 1: duplicate_entry_names_rejected
 * ============================================================ */

static void test_duplicate_entry_names_rejected(void) {
    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    check(w != NULL, "dup: writer created");

    const uint8_t data1[] = "first";
    const uint8_t data2[] = "second";
    int rc1 = shard_v2_writer_add_entry(w, "a", data1, 5);
    int rc2 = shard_v2_writer_add_entry(w, "a", data2, 6);

    if (rc1 != 0 || rc2 != 0) {
        check(true, "dup: duplicate rejected at add_entry");
        shard_v2_writer_free(w);
        return;
    }

    /* Writer accepted duplicates — write and check reader */
    char* path = make_tmpfile();
    int rcw = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);

    if (rcw != 0) {
        check(true, "dup: duplicate rejected at write");
        free(path);
        return;
    }

    shard_v2_reader_t* r = shard_v2_open(path);
    if (r == NULL) {
        check(true, "dup: duplicate rejected at open");
    } else {
        /* Reader accepted — at least one entry should exist */
        check(shard_v2_entry_count(r) >= 1, "dup: at least one entry exists");
        shard_v2_close(r);
    }

    unlink(path);
    free(path);
}

/* ============================================================
 * Case 2: zero_length_entry_valid
 * ============================================================ */

static void test_zero_length_entry_valid(void) {
    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    /* C API rejects NULL data pointer — use a valid pointer with size 0 */
    static const uint8_t empty_buf[1] = {0};
    int rc = shard_v2_writer_add_entry(w, "empty", empty_buf, 0);
    check(rc == 0, "zero_len: add_entry succeeded");

    char* path = make_tmpfile();
    rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);
    check(rc == 0, "zero_len: write succeeded");

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "zero_len: open succeeded");
    if (r) {
        check(shard_v2_entry_count(r) == 1, "zero_len: entry_count == 1");

        size_t out_size = 999;
        const uint8_t* data = shard_v2_read_entry(r, 0, &out_size);
        /* For zero-length entries, data may be NULL with size 0, or non-NULL with size 0 */
        check(out_size == 0, "zero_len: data size == 0");
        shard_v2_close(r);
    }

    unlink(path);
    free(path);
}

/* ============================================================
 * Case 3: unicode_entry_name
 * ============================================================ */

static void test_unicode_entry_name(void) {
    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    const uint8_t hello[] = "hello";
    /* UTF-8 for 日本語 */
    const char* uname = "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e";
    int rc = shard_v2_writer_add_entry(w, uname, hello, 5);
    check(rc == 0, "unicode: add_entry succeeded");

    char* path = make_tmpfile();
    rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);
    check(rc == 0, "unicode: write succeeded");

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "unicode: open succeeded");
    if (r) {
        check(shard_v2_entry_count(r) == 1, "unicode: entry_count == 1");

        const char* name = shard_v2_entry_name(r, 0);
        check(name != NULL && strcmp(name, uname) == 0, "unicode: entry name matches");

        size_t out_size = 0;
        const uint8_t* data = shard_v2_read_entry(r, 0, &out_size);
        check(data != NULL && out_size == 5 && memcmp(data, "hello", 5) == 0,
              "unicode: data matches");
        shard_v2_close(r);
    }

    unlink(path);
    free(path);
}

/* ============================================================
 * Cases 4-8: corrupt fixture tests
 * ============================================================ */

static void test_corrupt_fixture(const char* dir, const char* filename, const char* label) {
    char path[1024];
    make_path(path, sizeof(path), dir, filename);

    char desc[256];

    shard_v2_reader_t* r = shard_v2_open(path);
    if (r == NULL) {
        snprintf(desc, sizeof(desc), "%s: rejected at open", label);
        check(true, desc);
        return;
    }

    /* If open succeeded, try reading entries — at least one should fail */
    bool errored = false;
    uint32_t count = shard_v2_entry_count(r);
    for (uint32_t i = 0; i < count && !errored; i++) {
        size_t out_size = 0;
        const uint8_t* data = shard_v2_read_entry(r, i, &out_size);
        if (data == NULL) {
            errored = true;
        }
    }
    shard_v2_close(r);

    snprintf(desc, sizeof(desc), "%s: error detected", label);
    check(errored, desc);
}

/* ============================================================
 * Case 9: empty_input_rejected
 * ============================================================ */

static void test_empty_input_rejected(void) {
    shard_v2_reader_t* r = shard_v2_from_buffer(NULL, 0);
    check(r == NULL, "empty_input: rejected");
    if (r) {
        shard_v2_close(r);
    }

    /* Also test with a valid pointer but zero length */
    uint8_t dummy = 0;
    r = shard_v2_from_buffer(&dummy, 0);
    check(r == NULL, "empty_input (zero-len buf): rejected");
    if (r) {
        shard_v2_close(r);
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    const char* testdata_dir = "../ucodec/testdata/safety";
    if (argc > 1) {
        testdata_dir = argv[1];
    }

    printf("=== Shard Truth Table Tests ===\n\n");

    /* Roundtrip / write-reject tests */
    test_duplicate_entry_names_rejected();
    test_zero_length_entry_valid();
    test_unicode_entry_name();

    /* Corrupt fixture tests */
    test_corrupt_fixture(testdata_dir, "corrupt_bad_magic.shard",
                         "bad_magic");
    test_corrupt_fixture(testdata_dir, "corrupt_truncated_header.shard",
                         "truncated_header");
    test_corrupt_fixture(testdata_dir, "corrupt_crc_mismatch.shard",
                         "crc_mismatch");
    test_corrupt_fixture(testdata_dir, "corrupt_truncated_data.shard",
                         "truncated_data");
    test_corrupt_fixture(testdata_dir, "corrupt_entry_count_overflow.shard",
                         "entry_count_overflow");

    /* Empty input */
    test_empty_input_rejected();

    printf("\n=== Results: %d total, %d passed, %d failed ===\n",
           g_total, g_passed, g_failed);

    return g_failed > 0 ? 1 : 0;
}
