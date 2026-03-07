/*
 * test_safety.c -- Cross-language safety tests for shard_v2.c
 *
 * Validates correct handling of:
 *   - Valid shards (basic, compressed, empty entry, long name, unicode, binary data)
 *   - Corrupt shards (bad magic, truncated header, CRC mismatch, truncated data,
 *     entry count overflow, huge string table)
 *
 * Expected values are hard-coded from:
 *   cowrie/ucodec/testdata/safety/safety_manifest.json
 *
 * Usage: ./test_safety [testdata_dir]
 *   default testdata_dir = ../ucodec/testdata/safety
 */

#include "shard_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <pthread.h>

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

/* Build a path string: dir + "/" + filename */
static void make_path(char* out, size_t outsz, const char* dir, const char* file) {
    snprintf(out, outsz, "%s/%s", dir, file);
}

static void write_u16_le(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void write_u32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void write_u64_le(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
    p[4] = (uint8_t)((v >> 32) & 0xFFu);
    p[5] = (uint8_t)((v >> 40) & 0xFFu);
    p[6] = (uint8_t)((v >> 48) & 0xFFu);
    p[7] = (uint8_t)((v >> 56) & 0xFFu);
}

/* ============================================================
 * Signal-based crash guard for corrupt shard tests
 *
 * We want to verify that opening/reading corrupt files does NOT
 * segfault. We use setjmp/longjmp + signal handlers as a safety net.
 * ============================================================ */

static sigjmp_buf g_jmpbuf;
static volatile sig_atomic_t g_in_guard = 0;

static void crash_handler(int sig) {
    (void)sig;
    if (g_in_guard) {
        siglongjmp(g_jmpbuf, 1);
    }
    /* If not in guard, re-raise to get normal crash behavior */
    signal(sig, SIG_DFL);
    raise(sig);
}

/* ============================================================
 * Expected entry descriptor (simplified for safety tests)
 * ============================================================ */

typedef struct {
    const char* name;
    uint64_t    size;   /* orig_size / decompressed size */
} expected_safety_entry_t;

/* ============================================================
 * Valid shard tests
 * ============================================================ */

static void test_valid_shard(const char* dir, const char* filename,
                             uint32_t exp_entry_count,
                             const expected_safety_entry_t* exp_entries) {
    char path[1024];
    make_path(path, sizeof(path), dir, filename);

    char desc[512];

    /* Test: open succeeds */
    snprintf(desc, sizeof(desc), "%s: open succeeds", filename);
    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, desc);
    if (!r) return;

    /* Test: entry_count matches */
    snprintf(desc, sizeof(desc), "%s: entry_count == %u", filename, exp_entry_count);
    uint32_t count = shard_v2_entry_count(r);
    check(count == exp_entry_count, desc);

    /* Test: each entry can be read and data size matches */
    for (uint32_t i = 0; i < exp_entry_count && i < count; i++) {
        const expected_safety_entry_t* ex = &exp_entries[i];

        /* Verify entry name */
        const char* name = shard_v2_entry_name(r, i);
        snprintf(desc, sizeof(desc), "%s[%u]: name == \"%s\"", filename, i, ex->name);
        check(name != NULL && strcmp(name, ex->name) == 0, desc);

        /* Read entry data */
        size_t out_size = 0;
        const uint8_t* data = shard_v2_read_entry(r, i, &out_size);
        snprintf(desc, sizeof(desc), "%s[%u]: read_entry succeeds", filename, i);
        check(data != NULL || ex->size == 0, desc);

        /* Verify data size matches expected orig_size */
        if (data || ex->size == 0) {
            snprintf(desc, sizeof(desc), "%s[%u]: data size == %llu",
                     filename, i, (unsigned long long)ex->size);
            check(out_size == (size_t)ex->size, desc);
        }

        /* Verify lookup by name */
        if (name) {
            int32_t idx = shard_v2_lookup(r, name);
            snprintf(desc, sizeof(desc), "%s[%u]: lookup(\"%s\") == %u",
                     filename, i, name, i);
            check(idx == (int32_t)i, desc);
        }
    }

    /* Verify lookup for non-existent name returns -1 */
    snprintf(desc, sizeof(desc), "%s: lookup(\"__nonexistent__\") == -1", filename);
    check(shard_v2_lookup(r, "__nonexistent__") == -1, desc);

    shard_v2_close(r);
}

/* ============================================================
 * Corrupt shard tests
 * ============================================================ */

static void test_corrupt_shard(const char* dir, const char* filename,
                               const char* expected_error) {
    char path[1024];
    make_path(path, sizeof(path), dir, filename);

    char desc[512];

    /* Install crash guard */
    struct sigaction sa_old_segv, sa_old_bus, sa_old_abrt;
    struct sigaction sa_new;
    memset(&sa_new, 0, sizeof(sa_new));
    sa_new.sa_handler = crash_handler;
    sigemptyset(&sa_new.sa_mask);
    sa_new.sa_flags = 0;

    sigaction(SIGSEGV, &sa_new, &sa_old_segv);
    sigaction(SIGBUS,  &sa_new, &sa_old_bus);
    sigaction(SIGABRT, &sa_new, &sa_old_abrt);

    g_in_guard = 1;

    if (sigsetjmp(g_jmpbuf, 1) != 0) {
        /* We got here from a signal handler -- the code crashed */
        g_in_guard = 0;
        sigaction(SIGSEGV, &sa_old_segv, NULL);
        sigaction(SIGBUS,  &sa_old_bus,  NULL);
        sigaction(SIGABRT, &sa_old_abrt, NULL);

        snprintf(desc, sizeof(desc), "%s (%s): must not crash -- CRASHED!",
                 filename, expected_error);
        check(false, desc);
        return;
    }

    /* Attempt to open the corrupt shard */
    shard_v2_reader_t* r = shard_v2_open(path);

    if (r == NULL) {
        /* Open failed gracefully -- this is the expected outcome for most
         * corrupt files (bad magic, truncated header, overflow, etc.). */
        snprintf(desc, sizeof(desc), "%s (%s): open returns NULL (rejected)",
                 filename, expected_error);
        check(true, desc);

        snprintf(desc, sizeof(desc), "%s (%s): did not crash",
                 filename, expected_error);
        check(true, desc);
    } else {
        /* Open succeeded -- some corrupt files (e.g., CRC mismatch, truncated
         * data) might parse the header/index OK but fail on read_entry. */
        snprintf(desc, sizeof(desc), "%s (%s): open succeeded (header parsed)",
                 filename, expected_error);
        /* Not necessarily a failure -- the error may manifest on read */
        printf("  INFO  %s\n", desc);

        uint32_t count = shard_v2_entry_count(r);
        bool any_read_failed = false;

        for (uint32_t i = 0; i < count; i++) {
            size_t out_size = 0;
            const uint8_t* data = shard_v2_read_entry(r, i, &out_size);
            if (data == NULL) {
                any_read_failed = true;
            }
        }

        /* For corrupt files, either open should fail OR at least one read
         * should fail. If both succeed, the corruption was not detected. */
        snprintf(desc, sizeof(desc), "%s (%s): corruption detected (open=NULL or read=NULL)",
                 filename, expected_error);
        check(any_read_failed, desc);

        snprintf(desc, sizeof(desc), "%s (%s): did not crash",
                 filename, expected_error);
        check(true, desc);

        shard_v2_close(r);
    }

    g_in_guard = 0;

    /* Restore original signal handlers */
    sigaction(SIGSEGV, &sa_old_segv, NULL);
    sigaction(SIGBUS,  &sa_old_bus,  NULL);
    sigaction(SIGABRT, &sa_old_abrt, NULL);
}

/* ============================================================
 * Synthetic corrupt file tests (write temp files, verify rejection)
 * ============================================================ */

static void write_temp_file(const char* path, const uint8_t* data, size_t len) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    if (len > 0) fwrite(data, 1, len, f);
    fclose(f);
}

/* Test: 0-byte file must be rejected */
static void test_corrupt_empty_file(void) {
    const char* path = "/tmp/test_safety_empty.shard";

    printf("[ corrupt_empty_file (synthetic) ]\n");

    /* Create an empty file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        check(false, "corrupt_empty_file: failed to create temp file");
        return;
    }
    fclose(f);

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r == NULL, "corrupt_empty_file: open returns NULL (0-byte file rejected)");
    if (r) shard_v2_close(r);

    unlink(path);
    printf("\n");
}

/* Test: file with wrong magic bytes must be rejected */
static void test_corrupt_wrong_magic(void) {
    const char* path = "/tmp/test_safety_wrong_magic.shard";

    printf("[ corrupt_wrong_magic (synthetic) ]\n");

    uint8_t buf[64];
    memset(buf, 0, sizeof(buf));
    memcpy(buf, "NOPE", 4);  /* wrong magic, should be "SHRD" */

    write_temp_file(path, buf, sizeof(buf));

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r == NULL, "corrupt_wrong_magic: open returns NULL (bad magic rejected)");
    if (r) shard_v2_close(r);

    unlink(path);
    printf("\n");
}

/* Test: 1024 bytes of pseudo-random data must not crash */
static void test_corrupt_random_bytes(void) {
    const char* path = "/tmp/test_safety_random.shard";

    printf("[ corrupt_random_bytes (synthetic) ]\n");

    uint8_t buf[1024];
    for (size_t i = 0; i < sizeof(buf); i++) {
        buf[i] = (uint8_t)((i * 7 + 13) % 256);
    }
    write_temp_file(path, buf, sizeof(buf));

    /* Install crash guard */
    struct sigaction sa_old_segv, sa_old_bus, sa_old_abrt;
    struct sigaction sa_new;
    memset(&sa_new, 0, sizeof(sa_new));
    sa_new.sa_handler = crash_handler;
    sigemptyset(&sa_new.sa_mask);
    sa_new.sa_flags = 0;

    sigaction(SIGSEGV, &sa_new, &sa_old_segv);
    sigaction(SIGBUS,  &sa_new, &sa_old_bus);
    sigaction(SIGABRT, &sa_new, &sa_old_abrt);

    g_in_guard = 1;

    if (sigsetjmp(g_jmpbuf, 1) != 0) {
        g_in_guard = 0;
        sigaction(SIGSEGV, &sa_old_segv, NULL);
        sigaction(SIGBUS,  &sa_old_bus,  NULL);
        sigaction(SIGABRT, &sa_old_abrt, NULL);
        check(false, "corrupt_random_bytes: CRASHED on random data!");
        unlink(path);
        printf("\n");
        return;
    }

    shard_v2_reader_t* r = shard_v2_open(path);
    if (r != NULL) {
        /* If open succeeded despite random data, try reading entries */
        uint32_t count = shard_v2_entry_count(r);
        for (uint32_t i = 0; i < count && i < 100; i++) {
            size_t out_size = 0;
            shard_v2_read_entry(r, i, &out_size);
        }
        shard_v2_close(r);
    }

    g_in_guard = 0;
    sigaction(SIGSEGV, &sa_old_segv, NULL);
    sigaction(SIGBUS,  &sa_old_bus,  NULL);
    sigaction(SIGABRT, &sa_old_abrt, NULL);

    check(true, "corrupt_random_bytes: did not crash on 1024 random bytes");

    unlink(path);
    printf("\n");
}

/* Test: 128 bytes of all-zeros must be rejected */
static void test_corrupt_all_zeros(void) {
    const char* path = "/tmp/test_safety_all_zeros.shard";

    printf("[ corrupt_all_zeros (synthetic) ]\n");

    uint8_t buf[128];
    memset(buf, 0, sizeof(buf));

    write_temp_file(path, buf, sizeof(buf));

    shard_v2_reader_t* r = shard_v2_open(path);
    check(r == NULL, "corrupt_all_zeros: open returns NULL (all-zeros rejected)");
    if (r) shard_v2_close(r);

    unlink(path);
    printf("\n");
}

static void test_corrupt_overflowing_entry_range(void) {
    printf("[ corrupt_overflowing_entry_range (synthetic) ]\n");

    uint8_t buf[SHARD_HEADER_SIZE + SHARD_INDEX_ENTRY_SIZE + 2];
    memset(buf, 0, sizeof(buf));

    memcpy(buf, SHARD_MAGIC, 4);
    buf[4] = SHARD_VERSION2;
    buf[5] = ROLE_UNKNOWN;
    write_u16_le(buf + 6, FLAG_LITTLE_ENDIAN);
    buf[8] = ALIGN_NONE;
    buf[9] = COMPRESS_NONE;
    write_u16_le(buf + 10, SHARD_INDEX_ENTRY_SIZE);
    write_u32_le(buf + 12, 1);
    write_u64_le(buf + 16, SHARD_HEADER_SIZE + SHARD_INDEX_ENTRY_SIZE);
    write_u64_le(buf + 24, SHARD_HEADER_SIZE + SHARD_INDEX_ENTRY_SIZE + 2);
    write_u64_le(buf + 32, 0);
    write_u64_le(buf + 40, sizeof(buf));

    uint8_t* entry = buf + SHARD_HEADER_SIZE;
    write_u64_le(entry + 0, 0);
    write_u32_le(entry + 8, 0);
    write_u16_le(entry + 12, 1);
    write_u16_le(entry + 14, 0);
    write_u64_le(entry + 16, UINT64_MAX - 7);
    write_u64_le(entry + 24, 16);
    write_u64_le(entry + 32, 16);
    write_u32_le(entry + 40, 0);
    write_u32_le(entry + 44, 0);

    buf[SHARD_HEADER_SIZE + SHARD_INDEX_ENTRY_SIZE] = 'x';
    buf[SHARD_HEADER_SIZE + SHARD_INDEX_ENTRY_SIZE + 1] = '\0';

    shard_v2_reader_t* r = shard_v2_from_buffer(buf, sizeof(buf));
    check(r == NULL, "corrupt_overflowing_entry_range: open returns NULL (overflowing data range rejected)");
    if (r) shard_v2_close(r);

    printf("\n");
}

/* ============================================================
 * Concurrent reads test (pthreads)
 *
 * NOTE: link with -lpthread
 * ============================================================ */

typedef struct {
    const char* path;           /* path to valid_basic.shard */
    uint32_t    expected_count; /* 3 */
    uint64_t    expected_sizes[3]; /* 1024, 2048, 512 */
    int         thread_id;
    bool        passed;         /* set by thread */
} concurrent_arg_t;

static void* concurrent_reader(void* arg) {
    concurrent_arg_t* a = (concurrent_arg_t*)arg;
    a->passed = true;

    shard_v2_reader_t* r = shard_v2_open(a->path);
    if (!r) {
        fprintf(stderr, "  thread %d: open failed\n", a->thread_id);
        a->passed = false;
        return NULL;
    }

    uint32_t count = shard_v2_entry_count(r);
    if (count != a->expected_count) {
        fprintf(stderr, "  thread %d: entry_count %u != %u\n",
                a->thread_id, count, a->expected_count);
        a->passed = false;
    }

    for (uint32_t i = 0; i < count && i < a->expected_count; i++) {
        size_t out_size = 0;
        const uint8_t* data = shard_v2_read_entry(r, i, &out_size);
        if (!data && a->expected_sizes[i] != 0) {
            fprintf(stderr, "  thread %d: read_entry(%u) returned NULL\n",
                    a->thread_id, i);
            a->passed = false;
        }
        if (out_size != (size_t)a->expected_sizes[i]) {
            fprintf(stderr, "  thread %d: entry[%u] size %zu != %llu\n",
                    a->thread_id, i, out_size,
                    (unsigned long long)a->expected_sizes[i]);
            a->passed = false;
        }
    }

    shard_v2_close(r);
    return NULL;
}

static void test_concurrent_reads(const char* testdata) {
    printf("[ concurrent_reads (4 threads) ]\n");

    char path[1024];
    make_path(path, sizeof(path), testdata, "valid_basic.shard");

    /* Verify the file exists first */
    shard_v2_reader_t* probe = shard_v2_open(path);
    if (!probe) {
        check(false, "concurrent_reads: cannot open valid_basic.shard");
        printf("\n");
        return;
    }
    shard_v2_close(probe);

    #define NUM_THREADS 4
    pthread_t threads[NUM_THREADS];
    concurrent_arg_t args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].path = path;
        args[i].expected_count = 3;
        args[i].expected_sizes[0] = 1024;
        args[i].expected_sizes[1] = 2048;
        args[i].expected_sizes[2] = 512;
        args[i].thread_id = i;
        args[i].passed = false;

        int rc = pthread_create(&threads[i], NULL, concurrent_reader, &args[i]);
        if (rc != 0) {
            char desc[128];
            snprintf(desc, sizeof(desc),
                     "concurrent_reads: pthread_create thread %d", i);
            check(false, desc);
            /* Join any threads that were successfully created */
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            printf("\n");
            return;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        char desc[128];
        snprintf(desc, sizeof(desc),
                 "concurrent_reads: thread %d read all entries correctly", i);
        check(args[i].passed, desc);
    }
    #undef NUM_THREADS

    printf("\n");
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char* argv[]) {
    const char* testdata = (argc > 1) ? argv[1]
                                      : "../ucodec/testdata/safety";

    printf("=== shard_v2 safety tests ===\n\n");

    /* ============================================================
     * Valid shard tests
     * ============================================================ */

    /* --- valid_basic.shard (3 entries) --- */
    printf("[ valid_basic.shard ]\n");
    {
        static const expected_safety_entry_t entries[] = {
            { "tensor_a", 1024 },
            { "tensor_b", 2048 },
            { "tensor_c", 512  },
        };
        test_valid_shard(testdata, "valid_basic.shard", 3, entries);
    }
    printf("\n");

    /* --- valid_compressed.shard (3 entries, compressed data) --- */
    printf("[ valid_compressed.shard ]\n");
    {
        static const expected_safety_entry_t entries[] = {
            { "comp_a", 4096 },
            { "comp_b", 8192 },
            { "comp_c", 2048 },
        };
        test_valid_shard(testdata, "valid_compressed.shard", 3, entries);
    }
    printf("\n");

    /* --- valid_empty_entry.shard (1 entry, zero-length data) --- */
    printf("[ valid_empty_entry.shard ]\n");
    {
        static const expected_safety_entry_t entries[] = {
            { "empty", 0 },
        };
        test_valid_shard(testdata, "valid_empty_entry.shard", 1, entries);
    }
    printf("\n");

    /* --- valid_long_name.shard (1 entry, 1000-char name) --- */
    printf("[ valid_long_name.shard ]\n");
    {
        /* 1000 'a' characters */
        static char long_name[1001];
        memset(long_name, 'a', 1000);
        long_name[1000] = '\0';
        const expected_safety_entry_t entries[] = {
            { long_name, 21 },
        };
        test_valid_shard(testdata, "valid_long_name.shard", 1, entries);
    }
    printf("\n");

    /* --- valid_unicode_name.shard (4 entries with unicode names) --- */
    printf("[ valid_unicode_name.shard ]\n");
    {
        static const expected_safety_entry_t entries[] = {
            { "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",                           12 },  /* 日本語 */
            { "\xc3\xa9moji_\xf0\x9f\x98\x80",                                   11 },  /* emoji_smiley */
            { "\xe4\xb8\xad\xe6\x96\x87/\xe5\xb1\x82/\xe6\x9d\x83\xe9\x87\x8d", 12 },  /* 中文/层/权重 */
            { "\xd0\x90\xd0\x91\xd0\x92\xd0\x93",                                 13 },  /* АБВГ */
        };
        test_valid_shard(testdata, "valid_unicode_name.shard", 4, entries);
    }
    printf("\n");

    /* --- valid_binary_data.shard (1 entry, all 256 byte values) --- */
    printf("[ valid_binary_data.shard ]\n");
    {
        static const expected_safety_entry_t entries[] = {
            { "all_256_bytes", 256 },
        };
        test_valid_shard(testdata, "valid_binary_data.shard", 1, entries);
    }
    printf("\n");

    /* ============================================================
     * Corrupt shard tests
     * ============================================================ */

    printf("[ corrupt shards ]\n");

    test_corrupt_shard(testdata, "corrupt_bad_magic.shard",
                       "invalid magic");

    test_corrupt_shard(testdata, "corrupt_truncated_header.shard",
                       "header too short");

    test_corrupt_shard(testdata, "corrupt_crc_mismatch.shard",
                       "checksum mismatch");

    test_corrupt_shard(testdata, "corrupt_truncated_data.shard",
                       "truncated data");

    test_corrupt_shard(testdata, "corrupt_entry_count_overflow.shard",
                       "entry count overflow");

    test_corrupt_shard(testdata, "corrupt_huge_string_table.shard",
                       "string table beyond file");

    printf("\n");

    /* ============================================================
     * Synthetic corrupt file tests (temp files)
     * ============================================================ */

    test_corrupt_empty_file();
    test_corrupt_wrong_magic();
    test_corrupt_random_bytes();
    test_corrupt_all_zeros();
    test_corrupt_overflowing_entry_range();

    /* ============================================================
     * Concurrent reads test
     * ============================================================ */

    test_concurrent_reads(testdata);

    /* ============================================================
     * Summary
     * ============================================================ */

    printf("=== Results: %d/%d passed, %d failed ===\n",
           g_passed, g_total, g_failed);

    return (g_failed == 0) ? 0 : 1;
}
