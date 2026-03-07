/*
 * test_metadata.c — Tests for security limits, list_children, read_entry_prefix,
 * and path helpers added to shard_v2.
 */

#include "shard_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

/* ============================================================
 * Test framework (minimal)
 * ============================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  " #name " ... "); \
    fflush(stdout); \
    test_##name(); \
    printf("ok\n"); \
    tests_passed++; \
} while(0)

/* ============================================================
 * Helpers
 * ============================================================ */

/* Write a little-endian uint32 into buf at offset. */
static void put_le32(uint8_t* buf, size_t off, uint32_t v) {
    buf[off+0] = (uint8_t)(v);
    buf[off+1] = (uint8_t)(v >>  8);
    buf[off+2] = (uint8_t)(v >> 16);
    buf[off+3] = (uint8_t)(v >> 24);
}

/* Write a little-endian uint64 into buf at offset. */
static void put_le64(uint8_t* buf, size_t off, uint64_t v) {
    buf[off+0] = (uint8_t)(v);
    buf[off+1] = (uint8_t)(v >>  8);
    buf[off+2] = (uint8_t)(v >> 16);
    buf[off+3] = (uint8_t)(v >> 24);
    buf[off+4] = (uint8_t)(v >> 32);
    buf[off+5] = (uint8_t)(v >> 40);
    buf[off+6] = (uint8_t)(v >> 48);
    buf[off+7] = (uint8_t)(v >> 56);
}

/* Build a minimal shard buffer with given entries.
 * entries: array of { name, data, data_len }.
 * Returns malloc'd buffer; caller frees. Sets *out_len. */
typedef struct {
    const char* name;
    const uint8_t* data;
    size_t len;
} test_entry_t;

static uint8_t* build_shard(const test_entry_t* entries, uint32_t n,
                             uint8_t alignment, size_t* out_len) {
    if (!alignment) alignment = 64;

    /* Build string table */
    size_t st_size = 0;
    for (uint32_t i = 0; i < n; i++) {
        st_size += strlen(entries[i].name) + 1;
    }

    uint64_t st_off = SHARD_HEADER_SIZE + (uint64_t)n * SHARD_INDEX_ENTRY_SIZE;
    uint64_t ds_off_raw = st_off + (uint64_t)st_size;

    /* Align data section */
    uint64_t a = (uint64_t)alignment;
    uint64_t ds_off = (ds_off_raw + a - 1) & ~(a - 1);

    /* Compute per-entry data offsets */
    uint64_t* data_offs = (uint64_t*)malloc(sizeof(uint64_t) * (n + 1));
    assert(data_offs);
    uint64_t cur = ds_off;
    for (uint32_t i = 0; i < n; i++) {
        cur = (cur + a - 1) & ~(a - 1);
        data_offs[i] = cur;
        cur += (uint64_t)entries[i].len;
    }
    uint64_t total = cur;

    uint8_t* buf = (uint8_t*)calloc(1, (size_t)total);
    assert(buf);

    /* Header */
    memcpy(buf, "SHRD", 4);
    buf[4] = 0x02;  /* version */
    buf[5] = 1;     /* role MOSH */
    /* flags: little-endian + checksums + content types = 0x00A2 */
    buf[6] = 0xA2; buf[7] = 0x00;
    buf[8] = alignment;
    buf[9] = 0; /* no compression */
    buf[10] = (uint8_t)SHARD_INDEX_ENTRY_SIZE;
    buf[11] = 0;
    put_le32(buf, 12, n);
    put_le64(buf, 16, st_off);
    put_le64(buf, 24, ds_off);
    put_le64(buf, 32, 0); /* schema_offset = 0 */
    put_le64(buf, 40, total);

    /* String table */
    size_t st_pos = 0;
    uint32_t* name_offs = (uint32_t*)malloc(sizeof(uint32_t) * (n + 1));
    assert(name_offs);
    for (uint32_t i = 0; i < n; i++) {
        name_offs[i] = (uint32_t)st_pos;
        size_t nlen = strlen(entries[i].name);
        memcpy(buf + st_off + st_pos, entries[i].name, nlen);
        st_pos += nlen;
        buf[st_off + st_pos++] = '\0';
    }

    /* Index entries */
    for (uint32_t i = 0; i < n; i++) {
        size_t ioff = SHARD_HEADER_SIZE + (size_t)i * SHARD_INDEX_ENTRY_SIZE;
        uint64_t hash = shard_v2_xxhash64(entries[i].name, strlen(entries[i].name));
        put_le64(buf, ioff +  0, hash);
        put_le32(buf, ioff +  8, name_offs[i]);
        buf[ioff + 12] = (uint8_t)strlen(entries[i].name);
        buf[ioff + 13] = 0;
        buf[ioff + 14] = 0; /* flags = 0 */
        buf[ioff + 15] = 0;
        put_le64(buf, ioff + 16, data_offs[i]);
        put_le64(buf, ioff + 24, entries[i].len);
        put_le64(buf, ioff + 32, entries[i].len);
        uint32_t crc = shard_v2_crc32c(entries[i].data, entries[i].len);
        put_le32(buf, ioff + 40, crc);
        put_le32(buf, ioff + 44, 0); /* content type = unknown */

        /* Write data */
        memcpy(buf + data_offs[i], entries[i].data, entries[i].len);
    }

    free(data_offs);
    free(name_offs);

    if (out_len) *out_len = (size_t)total;
    return buf;
}

/* ============================================================
 * Path helper tests
 * ============================================================ */

TEST(path_parent_with_slash) {
    char buf[256];
    char* r = shard_v2_path_parent("layer.0/weight", buf, sizeof(buf));
    assert(r != NULL);
    assert(strcmp(r, "layer.0") == 0);
}

TEST(path_parent_deep) {
    char buf[256];
    char* r = shard_v2_path_parent("a/b/c", buf, sizeof(buf));
    assert(r != NULL);
    assert(strcmp(r, "a/b") == 0);
}

TEST(path_parent_no_slash) {
    char buf[256];
    char* r = shard_v2_path_parent("embed", buf, sizeof(buf));
    assert(r != NULL);
    assert(strcmp(r, "") == 0);
}

TEST(path_base_with_slash) {
    const char* r = shard_v2_path_base("layer.0/weight");
    assert(r != NULL);
    assert(strcmp(r, "weight") == 0);
}

TEST(path_base_deep) {
    const char* r = shard_v2_path_base("a/b/c");
    assert(r != NULL);
    assert(strcmp(r, "c") == 0);
}

TEST(path_base_no_slash) {
    const char* r = shard_v2_path_base("embed");
    assert(r != NULL);
    assert(strcmp(r, "embed") == 0);
}

TEST(split_path_simple) {
    char** parts = NULL;
    char*  buf   = NULL;
    int n = shard_v2_split_path("layer.0/weight", &parts, &buf);
    assert(n == 2);
    assert(strcmp(parts[0], "layer.0") == 0);
    assert(strcmp(parts[1], "weight") == 0);
    assert(parts[2] == NULL);
    free(parts);
    free(buf);
}

TEST(split_path_deep) {
    char** parts = NULL;
    char*  buf   = NULL;
    int n = shard_v2_split_path("a/b/c/d", &parts, &buf);
    assert(n == 4);
    assert(strcmp(parts[0], "a") == 0);
    assert(strcmp(parts[3], "d") == 0);
    free(parts);
    free(buf);
}

TEST(split_path_no_slash) {
    char** parts = NULL;
    char*  buf   = NULL;
    int n = shard_v2_split_path("embed", &parts, &buf);
    assert(n == 1);
    assert(strcmp(parts[0], "embed") == 0);
    free(parts);
    free(buf);
}

TEST(split_path_empty) {
    char** parts = NULL;
    char*  buf   = NULL;
    int n = shard_v2_split_path("", &parts, &buf);
    assert(n == 0);
    assert(parts[0] == NULL);
    free(parts);
    /* buf may be NULL for empty path */
    if (buf) free(buf);
}

TEST(join_path_two_parts) {
    const char* parts[] = {"layer.0", "weight"};
    char* r = shard_v2_join_path(parts, 2);
    assert(r != NULL);
    assert(strcmp(r, "layer.0/weight") == 0);
    free(r);
}

TEST(join_path_single) {
    const char* parts[] = {"embed"};
    char* r = shard_v2_join_path(parts, 1);
    assert(r != NULL);
    assert(strcmp(r, "embed") == 0);
    free(r);
}

TEST(join_path_multi) {
    const char* parts[] = {"a", "b", "c"};
    char* r = shard_v2_join_path(parts, 3);
    assert(r != NULL);
    assert(strcmp(r, "a/b/c") == 0);
    free(r);
}

/* ============================================================
 * list_children tests
 * ============================================================ */

/* make_layered_reader returns reader + buffer.  Caller must free buf AFTER closing reader,
 * because shard_v2_from_buffer borrows the pointer (owns_buf=false). */
static shard_v2_reader_t* make_layered_reader(uint8_t** out_buf) {
    test_entry_t entries[] = {
        {"layer.0/weight", (const uint8_t*)"w0", 2},
        {"layer.0/bias",   (const uint8_t*)"b0", 2},
        {"layer.1/weight", (const uint8_t*)"w1", 2},
        {"embed",          (const uint8_t*)"tok", 3},
    };
    size_t len;
    *out_buf = build_shard(entries, 4, 64, &len);
    return shard_v2_from_buffer(*out_buf, len);
}

/* Helper: check that array `arr` of `n` strings contains `needle`. */
static bool arr_contains(char** arr, uint32_t n, const char* needle) {
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(arr[i], needle) == 0) return true;
    }
    return false;
}

TEST(list_children_exact_prefix_with_slash) {
    uint8_t* buf;
    shard_v2_reader_t* r = make_layered_reader(&buf);
    assert(r != NULL);
    uint32_t count = 0;
    char** children = shard_v2_list_children(r, "layer.0/", &count);
    assert(children != NULL);
    assert(count == 2);
    assert(arr_contains(children, count, "layer.0/weight"));
    assert(arr_contains(children, count, "layer.0/bias"));
    shard_v2_list_children_free(children, count);
    shard_v2_close(r);
    free(buf);
}

TEST(list_children_empty_prefix_returns_top_level) {
    uint8_t* buf;
    shard_v2_reader_t* r = make_layered_reader(&buf);
    assert(r != NULL);
    uint32_t count = 0;
    char** children = shard_v2_list_children(r, "", &count);
    assert(children != NULL);
    assert(arr_contains(children, count, "layer.0/"));
    assert(arr_contains(children, count, "layer.1/"));
    assert(arr_contains(children, count, "embed"));
    /* No duplicates: count should be 3 */
    assert(count == 3);
    shard_v2_list_children_free(children, count);
    shard_v2_close(r);
    free(buf);
}

TEST(list_children_partial_prefix) {
    uint8_t* buf;
    shard_v2_reader_t* r = make_layered_reader(&buf);
    assert(r != NULL);
    uint32_t count = 0;
    char** children = shard_v2_list_children(r, "layer.", &count);
    assert(children != NULL);
    assert(count == 2);
    assert(arr_contains(children, count, "layer.0/"));
    assert(arr_contains(children, count, "layer.1/"));
    shard_v2_list_children_free(children, count);
    shard_v2_close(r);
    free(buf);
}

TEST(list_children_nonexistent_prefix) {
    uint8_t* buf;
    shard_v2_reader_t* r = make_layered_reader(&buf);
    assert(r != NULL);
    uint32_t count = 0;
    char** children = shard_v2_list_children(r, "nonexistent/", &count);
    assert(children != NULL);
    assert(count == 0);
    shard_v2_list_children_free(children, count);
    shard_v2_close(r);
    free(buf);
}

TEST(list_children_deduplicated_directories) {
    test_entry_t entries[] = {
        {"a/b", (const uint8_t*)"1", 1},
        {"a/c", (const uint8_t*)"2", 1},
        {"a/d", (const uint8_t*)"3", 1},
    };
    size_t len;
    uint8_t* buf = build_shard(entries, 3, 64, &len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, len);
    assert(r != NULL);

    uint32_t count = 0;
    char** children = shard_v2_list_children(r, "", &count);
    assert(children != NULL);
    /* "a/" should appear exactly once */
    int a_count = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(children[i], "a/") == 0) a_count++;
    }
    assert(a_count == 1);
    shard_v2_list_children_free(children, count);
    shard_v2_close(r);
    free(buf);
}

TEST(list_children_hierarchical_three_levels) {
    test_entry_t entries[] = {
        {"a/b/c", (const uint8_t*)"1", 1},
        {"a/b/d", (const uint8_t*)"2", 1},
        {"a/e",   (const uint8_t*)"3", 1},
        {"f",     (const uint8_t*)"4", 1},
    };
    size_t len;
    uint8_t* buf = build_shard(entries, 4, 64, &len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, len);
    assert(r != NULL);

    /* Top-level */
    uint32_t count = 0;
    char** top = shard_v2_list_children(r, "", &count);
    assert(top != NULL);
    assert(arr_contains(top, count, "a/"));
    assert(arr_contains(top, count, "f"));
    shard_v2_list_children_free(top, count);

    /* Under a/ */
    char** under_a = shard_v2_list_children(r, "a/", &count);
    assert(under_a != NULL);
    assert(arr_contains(under_a, count, "a/b/"));
    assert(arr_contains(under_a, count, "a/e"));
    shard_v2_list_children_free(under_a, count);

    /* Under a/b/ */
    char** under_ab = shard_v2_list_children(r, "a/b/", &count);
    assert(under_ab != NULL);
    assert(count == 2);
    assert(arr_contains(under_ab, count, "a/b/c"));
    assert(arr_contains(under_ab, count, "a/b/d"));
    shard_v2_list_children_free(under_ab, count);

    shard_v2_close(r);
    free(buf);
}

/* ============================================================
 * read_entry_prefix tests
 * ============================================================ */

static const uint8_t PAYLOAD[] = "Hello World, this is a test payload!";
static const size_t  PAYLOAD_LEN = 36; /* strlen of above */

TEST(read_entry_prefix_first_n_bytes) {
    test_entry_t entries[] = {
        {"entry", PAYLOAD, PAYLOAD_LEN},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r != NULL);

    size_t out_size = 0;
    const uint8_t* got = shard_v2_read_entry_prefix(r, 0, 5, &out_size);
    assert(got != NULL);
    assert(out_size == 5);
    assert(memcmp(got, "Hello", 5) == 0);
    shard_v2_close(r);
    free(buf);
}

TEST(read_entry_prefix_full_entry) {
    test_entry_t entries[] = {
        {"entry", PAYLOAD, PAYLOAD_LEN},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r != NULL);

    size_t out_size = 0;
    const uint8_t* got = shard_v2_read_entry_prefix(r, 0, PAYLOAD_LEN, &out_size);
    assert(got != NULL);
    assert(out_size == PAYLOAD_LEN);
    assert(memcmp(got, PAYLOAD, PAYLOAD_LEN) == 0);
    shard_v2_close(r);
    free(buf);
}

TEST(read_entry_prefix_exceeds_entry_length) {
    test_entry_t entries[] = {
        {"entry", PAYLOAD, PAYLOAD_LEN},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r != NULL);

    size_t out_size = 0;
    const uint8_t* got = shard_v2_read_entry_prefix(r, 0, 100000, &out_size);
    assert(got != NULL);
    assert(out_size == PAYLOAD_LEN);
    assert(memcmp(got, PAYLOAD, PAYLOAD_LEN) == 0);
    shard_v2_close(r);
    free(buf);
}

TEST(read_entry_prefix_zero_bytes) {
    test_entry_t entries[] = {
        {"entry", PAYLOAD, PAYLOAD_LEN},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r != NULL);

    size_t out_size = 99;
    const uint8_t* got = shard_v2_read_entry_prefix(r, 0, 0, &out_size);
    assert(got != NULL);
    assert(out_size == 0);
    shard_v2_close(r);
    free(buf);
}

TEST(read_entry_prefix_one_byte) {
    test_entry_t entries[] = {
        {"entry", PAYLOAD, PAYLOAD_LEN},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r != NULL);

    size_t out_size = 0;
    const uint8_t* got = shard_v2_read_entry_prefix(r, 0, 1, &out_size);
    assert(got != NULL);
    assert(out_size == 1);
    assert(got[0] == 'H');
    shard_v2_close(r);
    free(buf);
}

/* ============================================================
 * Security limit tests
 * ============================================================ */

/* Build a raw SHARD_HEADER_SIZE-byte corrupt header buffer with specified fields. */
static uint8_t* make_corrupt_header(uint32_t entry_count, uint64_t total_file_size) {
    uint8_t* buf = (uint8_t*)calloc(1, SHARD_HEADER_SIZE);
    assert(buf);
    memcpy(buf, "SHRD", 4);
    buf[4] = 0x02;  /* version */
    buf[5] = 1;     /* role */
    buf[6] = 0xA2; buf[7] = 0x00; /* flags */
    buf[8] = 64;    /* alignment */
    buf[9] = 0;     /* compression */
    buf[10] = (uint8_t)SHARD_INDEX_ENTRY_SIZE;
    buf[11] = 0;
    put_le32(buf, 12, entry_count);
    put_le64(buf, 16, (uint64_t)SHARD_HEADER_SIZE); /* string_table_offset */
    put_le64(buf, 24, (uint64_t)SHARD_HEADER_SIZE); /* data_section_offset */
    put_le64(buf, 32, 0);                            /* schema_offset */
    put_le64(buf, 40, total_file_size);
    return buf;
}

TEST(security_entry_count_too_large) {
    uint32_t too_many = MAX_ENTRY_COUNT + 1;
    uint8_t* buf = make_corrupt_header(too_many, SHARD_HEADER_SIZE);
    /* from_buffer should return NULL because entry_count > MAX_ENTRY_COUNT */
    shard_v2_reader_t* r = shard_v2_from_buffer(buf, SHARD_HEADER_SIZE);
    assert(r == NULL);
    free(buf);
}

TEST(security_total_file_size_mismatch) {
    /* Build a valid small shard, then corrupt total_file_size */
    test_entry_t entries[] = {
        {"x", (const uint8_t*)"hello", 5},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);

    /* Corrupt total_file_size at offset 40 */
    put_le64(buf, 40, 99999);

    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r == NULL);
    free(buf);
}

TEST(security_data_offset_before_data_section) {
    /* Build a valid shard, then set entry 0 data_offset to 0 (before header) */
    test_entry_t entries[] = {
        {"x", (const uint8_t*)"hello", 5},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);

    /* Entry 0's data_offset field is at SHARD_HEADER_SIZE + 16 */
    put_le64(buf, SHARD_HEADER_SIZE + 16, 0);
    /* Fix total_file_size */
    put_le64(buf, 40, (uint64_t)buf_len);

    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r == NULL);
    free(buf);
}

TEST(security_data_extends_past_end) {
    /* Build a valid shard, then set entry 0 data_offset to huge value */
    test_entry_t entries[] = {
        {"x", (const uint8_t*)"hello", 5},
    };
    size_t buf_len;
    uint8_t* buf = build_shard(entries, 1, 64, &buf_len);

    put_le64(buf, SHARD_HEADER_SIZE + 16, 0xFFFFFFFFu);
    put_le64(buf, 40, (uint64_t)buf_len);

    shard_v2_reader_t* r = shard_v2_from_buffer(buf, buf_len);
    assert(r == NULL);
    free(buf);
}

/* ============================================================
 * main
 * ============================================================ */

int main(void) {
    printf("test_metadata\n");

    /* Path helpers */
    RUN(path_parent_with_slash);
    RUN(path_parent_deep);
    RUN(path_parent_no_slash);
    RUN(path_base_with_slash);
    RUN(path_base_deep);
    RUN(path_base_no_slash);
    RUN(split_path_simple);
    RUN(split_path_deep);
    RUN(split_path_no_slash);
    RUN(split_path_empty);
    RUN(join_path_two_parts);
    RUN(join_path_single);
    RUN(join_path_multi);

    /* list_children */
    RUN(list_children_exact_prefix_with_slash);
    RUN(list_children_empty_prefix_returns_top_level);
    RUN(list_children_partial_prefix);
    RUN(list_children_nonexistent_prefix);
    RUN(list_children_deduplicated_directories);
    RUN(list_children_hierarchical_three_levels);

    /* read_entry_prefix */
    RUN(read_entry_prefix_first_n_bytes);
    RUN(read_entry_prefix_full_entry);
    RUN(read_entry_prefix_exceeds_entry_length);
    RUN(read_entry_prefix_zero_bytes);
    RUN(read_entry_prefix_one_byte);

    /* Security limits */
    RUN(security_entry_count_too_large);
    RUN(security_total_file_size_mismatch);
    RUN(security_data_offset_before_data_section);
    RUN(security_data_extends_past_end);

    printf("\n%d tests passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
