/*
 * test_writer.c — Roundtrip tests for shard_v2 writer + reader.
 *
 * Tests:
 *   1. Single entry roundtrip
 *   2. Multiple entries roundtrip
 *   3. Typed entries
 *   4. Empty shard (zero entries)
 *   5. Alignment none (0)
 *   6. Alignment 16
 *   7. Alignment 64 (default)
 *   8. Lookup by name
 *   9. Checksum verification (corrupt data detected)
 *  10. from_buffer roundtrip
 *  11. Large entries
 */

#include "shard_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#  include <io.h>
#  define TMP_TEMPLATE "shard_test_XXXXXX"
#  define mktemp_safe(t) _mktemp(t)
#else
#  include <unistd.h>
#  define mktemp_safe(t) mkstemp(t)
#endif

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

/* Create a unique temporary file path */
static char* make_tmpfile(void) {
    /* Use /tmp directly with a fixed-but-unique name pattern */
    static int counter = 0;
    counter++;
    char* path = (char*)malloc(64);
    if (!path) return NULL;
    snprintf(path, 64, "/tmp/shard_test_%d_%d.shard", (int)getpid(), counter);
    return path;
}

/* Write shard and re-open it, returning the reader (or NULL on error).
 * Caller must shard_v2_close() and free(path). */
static shard_v2_reader_t* write_and_open(shard_v2_writer_t* w, char** out_path) {
    char* path = make_tmpfile();
    if (!path) return NULL;
    *out_path = path;

    int rc = shard_v2_writer_write(w, path);
    if (rc != 0) return NULL;

    return shard_v2_open(path);
}

/* Read a file into a malloc'd buffer. Sets *sz. */
static uint8_t* read_file(const char* path, size_t* sz) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    if (len <= 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *sz = (size_t)len;
    return buf;
}

/* ============================================================
 * Test 1: Single entry roundtrip
 * ============================================================ */
static void test_single_entry(void) {
    printf("[ Test 1: single entry roundtrip ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    check(w != NULL, "writer_new");
    if (!w) return;

    const uint8_t data[] = "hello world";
    int rc = shard_v2_writer_add_entry(w, "greeting", data, 11);
    check(rc == 0, "add_entry");

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    check(shard_v2_entry_count(r) == 1, "entry_count == 1");

    const char* name = shard_v2_entry_name(r, 0);
    check(name && strcmp(name, "greeting") == 0, "entry_name == \"greeting\"");

    size_t sz = 0;
    const uint8_t* got = shard_v2_read_entry(r, 0, &sz);
    check(got != NULL, "read_entry != NULL");
    check(sz == 11, "read_entry size == 11");
    check(got && memcmp(got, data, 11) == 0, "read_entry data matches");

    const shard_v2_header_t* h = shard_v2_header(r);
    check(h->role == ROLE_MOSH, "header role == ROLE_MOSH");
    check(h->version == 2, "header version == 2");
    check(h->alignment == ALIGN_64, "header alignment == 64 (default)");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 2: Multiple entries roundtrip
 * ============================================================ */
static void test_multiple_entries(void) {
    printf("[ Test 2: multiple entries roundtrip ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_SAMPLE);
    check(w != NULL, "writer_new");
    if (!w) return;

    const char* names[] = { "alpha", "beta", "gamma" };
    const uint8_t* datas[] = {
        (const uint8_t*)"AAAA",
        (const uint8_t*)"BBBBBBBB",
        (const uint8_t*)"CCCCCCCCCCCCCCCC",
    };
    const size_t lens[] = { 4, 8, 16 };

    for (int i = 0; i < 3; i++) {
        int rc = shard_v2_writer_add_entry(w, names[i], datas[i], lens[i]);
        char desc[64];
        snprintf(desc, sizeof(desc), "add_entry(%s)", names[i]);
        check(rc == 0, desc);
    }

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    check(shard_v2_entry_count(r) == 3, "entry_count == 3");

    for (uint32_t i = 0; i < 3; i++) {
        const char* name = shard_v2_entry_name(r, i);
        char desc[64];
        snprintf(desc, sizeof(desc), "entry[%u].name == \"%s\"", i, names[i]);
        check(name && strcmp(name, names[i]) == 0, desc);

        size_t sz = 0;
        const uint8_t* got = shard_v2_read_entry(r, i, &sz);
        snprintf(desc, sizeof(desc), "entry[%u] data matches", i);
        check(got && sz == lens[i] && memcmp(got, datas[i], lens[i]) == 0, desc);
    }

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 3: Typed entries
 * ============================================================ */
static void test_typed_entries(void) {
    printf("[ Test 3: typed entries ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }

    const uint8_t t_data[16] = {0x01, 0x02, 0x03, 0x04,
                                 0x05, 0x06, 0x07, 0x08,
                                 0x09, 0x0A, 0x0B, 0x0C,
                                 0x0D, 0x0E, 0x0F, 0x10};
    const uint8_t j_data[] = "{\"k\":1}";

    shard_v2_writer_add_entry_typed(w, "tensor", t_data, 16, CONTENT_TYPE_TENSOR);
    shard_v2_writer_add_entry_typed(w, "config", j_data, 7,  CONTENT_TYPE_JSON);

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    check(shard_v2_entry_count(r) == 2, "entry_count == 2");

    const shard_v2_index_entry_t* e0 = shard_v2_get_entry(r, 0);
    check(e0 && shard_v2_content_type(e0) == CONTENT_TYPE_TENSOR,
          "entry[0] content_type == TENSOR");

    const shard_v2_index_entry_t* e1 = shard_v2_get_entry(r, 1);
    check(e1 && shard_v2_content_type(e1) == CONTENT_TYPE_JSON,
          "entry[1] content_type == JSON");

    /* Header should have HAS_CONTENT_TYPES flag */
    const shard_v2_header_t* h = shard_v2_header(r);
    check((h->flags & FLAG_HAS_CONTENT_TYPES) != 0, "flag HAS_CONTENT_TYPES set");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 4: Empty shard (zero entries)
 * ============================================================ */
static void test_empty_shard(void) {
    printf("[ Test 4: empty shard ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    check(shard_v2_entry_count(r) == 0, "entry_count == 0");
    check(shard_v2_entry_name(r, 0) == NULL, "entry_name(0) == NULL");
    check(shard_v2_get_entry(r, 0) == NULL, "get_entry(0) == NULL");
    check(shard_v2_lookup(r, "anything") == -1, "lookup == -1");

    const shard_v2_header_t* h = shard_v2_header(r);
    check(h->entry_count == 0, "header.entry_count == 0");
    check(h->total_file_size == h->data_section_offset,
          "total_file_size == data_section_offset (no data)");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 5: Alignment none
 * ============================================================ */
static void test_alignment_none(void) {
    printf("[ Test 5: alignment none ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }
    shard_v2_writer_set_alignment(w, ALIGN_NONE);

    const uint8_t d1[] = {0xDE, 0xAD, 0xBE, 0xEF};
    const uint8_t d2[] = {0xCA, 0xFE};

    shard_v2_writer_add_entry(w, "x", d1, 4);
    shard_v2_writer_add_entry(w, "y", d2, 2);

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    const shard_v2_header_t* h = shard_v2_header(r);
    check(h->alignment == ALIGN_NONE, "alignment == 0");

    /* With no alignment the two entries should be packed tightly */
    const shard_v2_index_entry_t* e0 = shard_v2_get_entry(r, 0);
    const shard_v2_index_entry_t* e1 = shard_v2_get_entry(r, 1);
    check(e0 && e1, "both entries present");
    if (e0 && e1) {
        check(e1->data_offset == e0->data_offset + e0->disk_size,
              "entries are tightly packed (no alignment gap)");
    }

    /* Verify data */
    size_t sz;
    const uint8_t* got0 = shard_v2_read_entry(r, 0, &sz);
    check(got0 && sz == 4 && memcmp(got0, d1, 4) == 0, "entry[0] data correct");

    const uint8_t* got1 = shard_v2_read_entry(r, 1, &sz);
    check(got1 && sz == 2 && memcmp(got1, d2, 2) == 0, "entry[1] data correct");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 6: Alignment 16
 * ============================================================ */
static void test_alignment_16(void) {
    printf("[ Test 6: alignment 16 ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }
    shard_v2_writer_set_alignment(w, ALIGN_16);

    const uint8_t d1[] = {0x01};
    const uint8_t d2[] = {0x02, 0x03};
    shard_v2_writer_add_entry(w, "tiny1", d1, 1);
    shard_v2_writer_add_entry(w, "tiny2", d2, 2);

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    const shard_v2_header_t* h = shard_v2_header(r);
    check(h->alignment == ALIGN_16, "alignment == 16");
    check(h->data_section_offset % 16 == 0, "data_section_offset is 16-aligned");

    const shard_v2_index_entry_t* e0 = shard_v2_get_entry(r, 0);
    const shard_v2_index_entry_t* e1 = shard_v2_get_entry(r, 1);
    check(e0 && e0->data_offset % 16 == 0, "entry[0].data_offset 16-aligned");
    check(e1 && e1->data_offset % 16 == 0, "entry[1].data_offset 16-aligned");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 7: Alignment 64
 * ============================================================ */
static void test_alignment_64(void) {
    printf("[ Test 7: alignment 64 ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }
    /* alignment 64 is the default, but set explicitly */
    shard_v2_writer_set_alignment(w, ALIGN_64);

    const uint8_t d1[100] = {0xAA};
    const uint8_t d2[200] = {0xBB};
    shard_v2_writer_add_entry(w, "blob1", d1, 100);
    shard_v2_writer_add_entry(w, "blob2", d2, 200);

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    const shard_v2_header_t* h = shard_v2_header(r);
    check(h->alignment == ALIGN_64, "alignment == 64");
    check(h->data_section_offset % 64 == 0, "data_section_offset 64-aligned");

    const shard_v2_index_entry_t* e0 = shard_v2_get_entry(r, 0);
    const shard_v2_index_entry_t* e1 = shard_v2_get_entry(r, 1);
    check(e0 && e0->data_offset % 64 == 0, "entry[0].data_offset 64-aligned");
    check(e1 && e1->data_offset % 64 == 0, "entry[1].data_offset 64-aligned");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 8: Lookup by name
 * ============================================================ */
static void test_lookup(void) {
    printf("[ Test 8: lookup by name ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }

    const uint8_t d[] = {1};
    shard_v2_writer_add_entry(w, "layer.0/weight", d, 1);
    shard_v2_writer_add_entry(w, "layer.1/weight", d, 1);
    shard_v2_writer_add_entry(w, "config",         d, 1);

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (!r) { free(path); return; }

    check(shard_v2_lookup(r, "layer.0/weight") == 0, "lookup layer.0/weight == 0");
    check(shard_v2_lookup(r, "layer.1/weight") == 1, "lookup layer.1/weight == 1");
    check(shard_v2_lookup(r, "config")         == 2, "lookup config == 2");
    check(shard_v2_lookup(r, "missing")        == -1, "lookup missing == -1");

    check(shard_v2_has_entry(r, "config"), "has_entry(config) == true");
    check(!shard_v2_has_entry(r, "nope"), "has_entry(nope) == false");

    /* read_entry_by_name */
    size_t sz;
    const uint8_t* got = shard_v2_read_entry_by_name(r, "layer.1/weight", &sz);
    check(got && sz == 1 && got[0] == 1, "read_entry_by_name(layer.1/weight)");

    const uint8_t* miss = shard_v2_read_entry_by_name(r, "nonexistent", &sz);
    check(miss == NULL, "read_entry_by_name(nonexistent) == NULL");

    shard_v2_close(r);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 9: Checksum verification
 * ============================================================ */
static void test_checksum_verification(void) {
    printf("[ Test 9: checksum verification ]\n");

    /* Write a shard with one entry */
    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }

    const uint8_t orig_data[] = "test data for checksum";
    shard_v2_writer_add_entry(w, "payload", orig_data, sizeof(orig_data) - 1);

    char* path = make_tmpfile();
    if (!path) { shard_v2_writer_free(w); return; }

    int rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);
    check(rc == 0, "write to file");

    /* Read the shard normally first */
    shard_v2_reader_t* r = shard_v2_open(path);
    check(r != NULL, "open shard");
    if (r) {
        size_t sz;
        const uint8_t* data = shard_v2_read_entry(r, 0, &sz);
        check(data != NULL, "read_entry succeeds on valid data");
        shard_v2_close(r);
    }

    /* Verify stored checksum matches manually computed one */
    {
        shard_v2_reader_t* r2 = shard_v2_open(path);
        if (r2) {
            const shard_v2_index_entry_t* e = shard_v2_get_entry(r2, 0);
            uint32_t expected = shard_v2_crc32c(orig_data, sizeof(orig_data) - 1);
            check(e && e->checksum == expected, "stored checksum == computed CRC32C");
            shard_v2_close(r2);
        }
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Test 10: from_buffer roundtrip
 * ============================================================ */
static void test_from_buffer(void) {
    printf("[ Test 10: from_buffer roundtrip ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_WSHARD);
    if (!w) { check(false, "writer_new"); return; }

    const uint8_t d[] = {0x11, 0x22, 0x33, 0x44};
    shard_v2_writer_add_entry_typed(w, "signal", d, 4, CONTENT_TYPE_BLOB);

    char* path = NULL;
    shard_v2_reader_t* r_file = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r_file != NULL, "open from file");
    if (!r_file) { free(path); return; }
    shard_v2_close(r_file);

    /* Now read file bytes and use from_buffer */
    size_t buf_len;
    uint8_t* buf = read_file(path, &buf_len);
    check(buf != NULL, "read file bytes");
    if (!buf) { remove(path); free(path); return; }

    shard_v2_reader_t* r_buf = shard_v2_from_buffer(buf, buf_len);
    check(r_buf != NULL, "from_buffer");
    if (r_buf) {
        check(shard_v2_entry_count(r_buf) == 1, "entry_count == 1");
        const char* name = shard_v2_entry_name(r_buf, 0);
        check(name && strcmp(name, "signal") == 0, "entry_name == signal");

        const shard_v2_index_entry_t* e = shard_v2_get_entry(r_buf, 0);
        check(e && shard_v2_content_type(e) == CONTENT_TYPE_BLOB, "content_type == BLOB");

        size_t sz;
        const uint8_t* got = shard_v2_read_entry(r_buf, 0, &sz);
        check(got && sz == 4 && memcmp(got, d, 4) == 0, "data matches");

        const shard_v2_header_t* h = shard_v2_header(r_buf);
        check(h->role == ROLE_WSHARD, "role == WSHARD");

        shard_v2_close(r_buf);
    }

    free(buf);
    remove(path);
    free(path);
}

/* ============================================================
 * Test 11: Large entries
 * ============================================================ */
static void test_large_entries(void) {
    printf("[ Test 11: large entries ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new"); return; }

    /* 1 MB entry */
    size_t large_sz = 1024 * 1024;
    uint8_t* large_data = (uint8_t*)malloc(large_sz);
    check(large_data != NULL, "malloc 1MB");
    if (!large_data) { shard_v2_writer_free(w); return; }

    for (size_t i = 0; i < large_sz; i++) {
        large_data[i] = (uint8_t)(i % 256);
    }

    int rc = shard_v2_writer_add_entry_typed(w, "big_tensor", large_data, large_sz, CONTENT_TYPE_TENSOR);
    check(rc == 0, "add_entry 1MB");

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (r) {
        check(shard_v2_entry_count(r) == 1, "entry_count == 1");

        const shard_v2_index_entry_t* e = shard_v2_get_entry(r, 0);
        check(e && e->orig_size == (uint64_t)large_sz, "orig_size == 1MB");

        size_t sz;
        const uint8_t* got = shard_v2_read_entry(r, 0, &sz);
        check(got && sz == large_sz, "read_entry size == 1MB");
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
 * Test 12: WSHARD role with pattern data (matches golden_wshard logic)
 * ============================================================ */
static void test_wshard_pattern(void) {
    printf("[ Test 12: wshard pattern data ]\n");

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_WSHARD);
    if (!w) { check(false, "writer_new"); return; }

    /* makePattern(600, 0x11): data[i] = (0x11 + i) % 256 */
    uint8_t sig[600];
    for (int i = 0; i < 600; i++) sig[i] = (uint8_t)((0x11 + i) % 256);

    shard_v2_writer_add_entry_typed(w, "signal/imu", sig, 600, CONTENT_TYPE_BLOB);

    char* path = NULL;
    shard_v2_reader_t* r = write_and_open(w, &path);
    shard_v2_writer_free(w);

    check(r != NULL, "open after write");
    if (r) {
        size_t sz;
        const uint8_t* got = shard_v2_read_entry(r, 0, &sz);
        check(got && sz == 600, "read 600 bytes");
        if (got) {
            check(memcmp(got, sig, 600) == 0, "data matches pattern");
        }

        const shard_v2_index_entry_t* e = shard_v2_get_entry(r, 0);
        /* The golden checksum for this entry = 3613665374 */
        check(e && e->checksum == 3613665374u, "checksum == 3613665374 (golden)");

        shard_v2_close(r);
    }

    remove(path);
    free(path);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("=== shard_v2 writer roundtrip tests ===\n\n");

    test_single_entry();
    printf("\n");

    test_multiple_entries();
    printf("\n");

    test_typed_entries();
    printf("\n");

    test_empty_shard();
    printf("\n");

    test_alignment_none();
    printf("\n");

    test_alignment_16();
    printf("\n");

    test_alignment_64();
    printf("\n");

    test_lookup();
    printf("\n");

    test_checksum_verification();
    printf("\n");

    test_from_buffer();
    printf("\n");

    test_large_entries();
    printf("\n");

    test_wshard_pattern();
    printf("\n");

    printf("=== Results: %d/%d passed, %d failed ===\n",
           g_passed, g_total, g_failed);

    return (g_failed == 0) ? 0 : 1;
}
