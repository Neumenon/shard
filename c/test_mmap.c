/*
 * test_mmap.c — mmap reader tests for shard_v2.c
 *
 * Usage: ./test_mmap [testdata_dir]
 *
 * Opens each golden file via shard_v2_open_mmap() and verifies that all
 * entries read correctly.  Also exercises the roundtrip: write with the
 * writer, read back via mmap.
 */

#include "shard_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

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

/* ============================================================
 * Expected entry descriptor (same as test_golden.c)
 * ============================================================ */

typedef struct {
    const char* name;
    uint64_t    name_hash;
    uint16_t    content_type;
    uint64_t    orig_size;
    uint64_t    disk_size;
    uint32_t    checksum;
    bool        compressed;
} expected_entry_t;

/* ============================================================
 * Per-file mmap test
 * ============================================================ */

static void test_mmap_file(const char* dir, const char* filename,
                            uint8_t exp_version, uint8_t exp_role,
                            uint16_t exp_flags, uint8_t exp_alignment,
                            uint32_t exp_entry_count,
                            uint16_t exp_index_entry_size,
                            uint64_t exp_string_table_offset,
                            uint64_t exp_data_section_offset,
                            uint64_t exp_schema_offset,
                            uint64_t exp_total_file_size,
                            const expected_entry_t* exp_entries) {

    char path[1024];
    make_path(path, sizeof(path), dir, filename);

    char desc[512];

    /* Open via mmap */
    snprintf(desc, sizeof(desc), "shard_v2_open_mmap(%s) != NULL", filename);
    shard_v2_reader_t* r = shard_v2_open_mmap(path);
    check(r != NULL, desc);
    if (!r) return;

    /* Header checks */
    const shard_v2_header_t* h = shard_v2_header(r);

    snprintf(desc, sizeof(desc), "mmap %s: magic == SHRD", filename);
    check(memcmp(h->magic, "SHRD", 4) == 0, desc);

    snprintf(desc, sizeof(desc), "mmap %s: version == %u", filename, exp_version);
    check(h->version == exp_version, desc);

    snprintf(desc, sizeof(desc), "mmap %s: role == %u", filename, exp_role);
    check(h->role == exp_role, desc);

    snprintf(desc, sizeof(desc), "mmap %s: flags == 0x%04x", filename, exp_flags);
    check(h->flags == exp_flags, desc);

    snprintf(desc, sizeof(desc), "mmap %s: alignment == %u", filename, exp_alignment);
    check(h->alignment == exp_alignment, desc);

    snprintf(desc, sizeof(desc), "mmap %s: index_entry_size == %u", filename, exp_index_entry_size);
    check(h->index_entry_size == exp_index_entry_size, desc);

    snprintf(desc, sizeof(desc), "mmap %s: entry_count == %u", filename, exp_entry_count);
    check(h->entry_count == exp_entry_count, desc);

    snprintf(desc, sizeof(desc), "mmap %s: string_table_offset == %llu", filename,
             (unsigned long long)exp_string_table_offset);
    check(h->string_table_offset == exp_string_table_offset, desc);

    snprintf(desc, sizeof(desc), "mmap %s: data_section_offset == %llu", filename,
             (unsigned long long)exp_data_section_offset);
    check(h->data_section_offset == exp_data_section_offset, desc);

    snprintf(desc, sizeof(desc), "mmap %s: schema_offset == %llu", filename,
             (unsigned long long)exp_schema_offset);
    check(h->schema_offset == exp_schema_offset, desc);

    snprintf(desc, sizeof(desc), "mmap %s: total_file_size == %llu", filename,
             (unsigned long long)exp_total_file_size);
    check(h->total_file_size == exp_total_file_size, desc);

    snprintf(desc, sizeof(desc), "mmap %s: entry_count() == %u", filename, exp_entry_count);
    check(shard_v2_entry_count(r) == exp_entry_count, desc);

    /* Per-entry checks */
    for (uint32_t i = 0; i < exp_entry_count; i++) {
        const expected_entry_t* ex = &exp_entries[i];

        const char* name = shard_v2_entry_name(r, i);
        snprintf(desc, sizeof(desc), "mmap %s[%u]: name == \"%s\"", filename, i, ex->name);
        check(name && strcmp(name, ex->name) == 0, desc);

        const shard_v2_index_entry_t* e = shard_v2_get_entry(r, i);
        if (!e) {
            snprintf(desc, sizeof(desc), "mmap %s[%u]: get_entry != NULL", filename, i);
            check(false, desc);
            continue;
        }

        snprintf(desc, sizeof(desc), "mmap %s[%u]: name_hash == %llu", filename, i,
                 (unsigned long long)ex->name_hash);
        check(e->name_hash == ex->name_hash, desc);

        snprintf(desc, sizeof(desc), "mmap %s[%u]: content_type == %u", filename, i, ex->content_type);
        check(shard_v2_content_type(e) == ex->content_type, desc);

        snprintf(desc, sizeof(desc), "mmap %s[%u]: orig_size == %llu", filename, i,
                 (unsigned long long)ex->orig_size);
        check(e->orig_size == ex->orig_size, desc);

        snprintf(desc, sizeof(desc), "mmap %s[%u]: disk_size == %llu", filename, i,
                 (unsigned long long)ex->disk_size);
        check(e->disk_size == ex->disk_size, desc);

        snprintf(desc, sizeof(desc), "mmap %s[%u]: checksum == 0x%08x", filename, i, ex->checksum);
        check(e->checksum == ex->checksum, desc);

        snprintf(desc, sizeof(desc), "mmap %s[%u]: compressed == %s", filename, i,
                 ex->compressed ? "true" : "false");
        check(shard_v2_is_compressed(e) == ex->compressed, desc);

        /* Read entry — for uncompressed entries read_entry returns a pointer
         * directly into the mmap'd region (zero-copy). Verify CRC32C. */
        if (!ex->compressed) {
            size_t out_size = 0;
            const uint8_t* data = shard_v2_read_entry(r, i, &out_size);
            snprintf(desc, sizeof(desc), "mmap %s[%u]: read_entry != NULL", filename, i);
            check(data != NULL, desc);

            if (data) {
                snprintf(desc, sizeof(desc), "mmap %s[%u]: read size == %llu", filename, i,
                         (unsigned long long)ex->disk_size);
                check(out_size == (size_t)ex->disk_size, desc);

                uint32_t crc = shard_v2_crc32c(data, out_size);
                snprintf(desc, sizeof(desc), "mmap %s[%u]: CRC32C matches stored checksum", filename, i);
                check(crc == ex->checksum, desc);
            }
        }

        /* Lookup by name */
        if (name) {
            int32_t idx = shard_v2_lookup(r, name);
            snprintf(desc, sizeof(desc), "mmap %s[%u]: lookup(\"%s\") == %u", filename, i, name, i);
            check(idx == (int32_t)i, desc);
        }
    }

    /* has_entry for first entry name */
    if (exp_entry_count > 0) {
        snprintf(desc, sizeof(desc), "mmap %s: has_entry(\"%s\") == true", filename, exp_entries[0].name);
        check(shard_v2_has_entry(r, exp_entries[0].name), desc);
    }

    /* Lookup missing name */
    snprintf(desc, sizeof(desc), "mmap %s: lookup(\"__nonexistent__\") == -1", filename);
    check(shard_v2_lookup(r, "__nonexistent__") == -1, desc);

    shard_v2_close(r);
}

/* ============================================================
 * Compare mmap reader against regular reader
 * ============================================================ */

static void test_mmap_vs_regular(const char* dir, const char* filename) {
    char path[1024];
    make_path(path, sizeof(path), dir, filename);

    char desc[512];

    shard_v2_reader_t* mmap_r = shard_v2_open_mmap(path);
    shard_v2_reader_t* reg_r  = shard_v2_open(path);

    snprintf(desc, sizeof(desc), "mmap_vs_regular %s: both opened", filename);
    check(mmap_r != NULL && reg_r != NULL, desc);

    if (!mmap_r || !reg_r) {
        if (mmap_r) shard_v2_close(mmap_r);
        if (reg_r)  shard_v2_close(reg_r);
        return;
    }

    uint32_t n = shard_v2_entry_count(mmap_r);
    snprintf(desc, sizeof(desc), "mmap_vs_regular %s: entry_count matches", filename);
    check(n == shard_v2_entry_count(reg_r), desc);

    for (uint32_t i = 0; i < n; i++) {
        /* Names match */
        const char* mname = shard_v2_entry_name(mmap_r, i);
        const char* rname = shard_v2_entry_name(reg_r,  i);
        snprintf(desc, sizeof(desc), "mmap_vs_regular %s[%u]: name matches", filename, i);
        check(mname && rname && strcmp(mname, rname) == 0, desc);

        /* Data bytes match */
        size_t msz = 0, rsz = 0;
        const uint8_t* mdata = shard_v2_read_entry(mmap_r, i, &msz);
        const uint8_t* rdata = shard_v2_read_entry(reg_r,  i, &rsz);

        snprintf(desc, sizeof(desc), "mmap_vs_regular %s[%u]: read_entry not NULL", filename, i);
        check(mdata != NULL && rdata != NULL, desc);

        if (mdata && rdata) {
            snprintf(desc, sizeof(desc), "mmap_vs_regular %s[%u]: size matches", filename, i);
            check(msz == rsz, desc);

            snprintf(desc, sizeof(desc), "mmap_vs_regular %s[%u]: bytes identical", filename, i);
            check(msz == rsz && memcmp(mdata, rdata, msz) == 0, desc);
        }
    }

    shard_v2_close(mmap_r);
    shard_v2_close(reg_r);
}

/* ============================================================
 * Roundtrip: write → read back via mmap
 * ============================================================ */

static void test_mmap_roundtrip(void) {
    printf("[ mmap roundtrip (write + read back) ]\n");

    const char* path = "/tmp/test_shard_v2_mmap_roundtrip.shard";

    /* Write a small shard */
    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    if (!w) { check(false, "writer_new != NULL"); return; }

    static const uint8_t hello[] = "world";
    static const uint8_t foo[]   = "bar";
    static const uint8_t data1[] = {0x01, 0x02, 0x03, 0x04, 0x05};

    shard_v2_writer_add_entry(w, "hello",  hello,  5);
    shard_v2_writer_add_entry(w, "foo",    foo,    3);
    shard_v2_writer_add_entry(w, "binary", data1, 5);
    int rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);

    check(rc == 0, "writer_write returned 0");
    if (rc != 0) return;

    /* Read back via mmap */
    shard_v2_reader_t* r = shard_v2_open_mmap(path);
    check(r != NULL, "open_mmap roundtrip != NULL");
    if (!r) return;

    check(shard_v2_entry_count(r) == 3, "roundtrip entry_count == 3");

    size_t sz = 0;
    const uint8_t* d;

    d = shard_v2_read_entry(r, 0, &sz);
    check(d != NULL && sz == 5 && memcmp(d, "world", 5) == 0, "roundtrip entry[0] == 'world'");

    d = shard_v2_read_entry(r, 1, &sz);
    check(d != NULL && sz == 3 && memcmp(d, "bar", 3) == 0, "roundtrip entry[1] == 'bar'");

    d = shard_v2_read_entry(r, 2, &sz);
    check(d != NULL && sz == 5 && memcmp(d, data1, 5) == 0, "roundtrip entry[2] == binary data");

    /* Lookup by name */
    int32_t idx = shard_v2_lookup(r, "hello");
    check(idx == 0, "roundtrip lookup('hello') == 0");

    idx = shard_v2_lookup(r, "foo");
    check(idx == 1, "roundtrip lookup('foo') == 1");

    check(!shard_v2_has_entry(r, "missing"), "roundtrip has_entry('missing') == false");

    shard_v2_close(r);
    printf("\n");
}

/* ============================================================
 * Test large entry via mmap
 * ============================================================ */

static void test_mmap_large_entry(void) {
    printf("[ mmap large entry ]\n");

    const char* path = "/tmp/test_shard_v2_mmap_large.shard";
    const size_t large_sz = 1024 * 1024; /* 1 MB */

    uint8_t* large = (uint8_t*)malloc(large_sz);
    if (!large) { check(false, "malloc large buffer"); return; }

    /* Fill with a simple pattern */
    for (size_t i = 0; i < large_sz; i++) {
        large[i] = (uint8_t)(i & 0xFF);
    }

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    shard_v2_writer_add_entry(w, "large", large, large_sz);
    int rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);

    check(rc == 0, "large: writer_write == 0");
    if (rc != 0) { free(large); return; }

    shard_v2_reader_t* r = shard_v2_open_mmap(path);
    check(r != NULL, "large: open_mmap != NULL");

    if (r) {
        size_t out_sz = 0;
        const uint8_t* data = shard_v2_read_entry(r, 0, &out_sz);
        check(data != NULL, "large: read_entry != NULL");
        check(out_sz == large_sz, "large: size matches");
        if (data && out_sz == large_sz) {
            check(memcmp(data, large, large_sz) == 0, "large: bytes match");
        }
        shard_v2_close(r);
    }

    free(large);
    printf("\n");
}

/* ============================================================
 * Test empty entry via mmap
 * ============================================================ */

static void test_mmap_empty_entry(void) {
    printf("[ mmap empty entry ]\n");

    const char* path = "/tmp/test_shard_v2_mmap_empty.shard";

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    static const uint8_t empty_data[1] = {0};
    shard_v2_writer_add_entry(w, "empty", empty_data, 0);
    int rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);

    check(rc == 0, "empty: writer_write == 0");
    if (rc != 0) return;

    shard_v2_reader_t* r = shard_v2_open_mmap(path);
    check(r != NULL, "empty: open_mmap != NULL");

    if (r) {
        check(shard_v2_entry_count(r) == 1, "empty: entry_count == 1");

        size_t sz = 99;
        const uint8_t* d = shard_v2_read_entry(r, 0, &sz);
        check(d != NULL, "empty: read_entry != NULL");
        check(sz == 0, "empty: size == 0");

        shard_v2_close(r);
    }
    printf("\n");
}

/* ============================================================
 * Test bad path returns NULL
 * ============================================================ */

static void test_mmap_invalid(void) {
    printf("[ mmap invalid path ]\n");
    shard_v2_reader_t* r = shard_v2_open_mmap("/nonexistent/path/shard.shard");
    check(r == NULL, "open_mmap of nonexistent path returns NULL");
    printf("\n");
}

/* ============================================================
 * list_children via mmap
 * ============================================================ */

static void test_mmap_list_children(const char* testdata) {
    printf("[ mmap list_children ]\n");

    char path[1024];
    make_path(path, sizeof(path), testdata, "golden_hierarchical.shard");

    shard_v2_reader_t* r = shard_v2_open_mmap(path);
    check(r != NULL, "open_mmap golden_hierarchical != NULL");
    if (!r) { printf("\n"); return; }

    /* List children at top level */
    uint32_t count = 0;
    char** children = shard_v2_list_children(r, "", &count);
    check(children != NULL, "list_children('') != NULL");
    if (children) {
        /* Should get "layer.0" as the only top-level bare component */
        check(count == 1, "list_children('') count == 1");
        check(count > 0 && strcmp(children[0], "layer.0") == 0, "list_children('')[0] == 'layer.0'");
        shard_v2_list_children_free(children, count);
    }

    /* Children of layer.0/: attention/, ffn/, norm (leaf) = 3 entries */
    children = shard_v2_list_children(r, "layer.0/", &count);
    check(children != NULL, "list_children('layer.0/') != NULL");
    if (children) {
        check(count == 3, "list_children('layer.0/') count == 3");
        shard_v2_list_children_free(children, count);
    }

    shard_v2_close(r);
    printf("\n");
}

/* ============================================================
 * read_entry_prefix via mmap
 * ============================================================ */

static void test_mmap_read_prefix(void) {
    printf("[ mmap read_entry_prefix ]\n");

    const char* path = "/tmp/test_shard_v2_mmap_prefix.shard";

    shard_v2_writer_t* w = shard_v2_writer_new(ROLE_MOSH);
    static const uint8_t payload[] = {0,1,2,3,4,5,6,7,8,9};
    shard_v2_writer_add_entry(w, "data", payload, 10);
    int rc = shard_v2_writer_write(w, path);
    shard_v2_writer_free(w);

    check(rc == 0, "prefix: writer_write == 0");
    if (rc != 0) return;

    shard_v2_reader_t* r = shard_v2_open_mmap(path);
    check(r != NULL, "prefix: open_mmap != NULL");

    if (r) {
        size_t sz = 0;
        const uint8_t* d;

        /* Read first 4 bytes */
        d = shard_v2_read_entry_prefix(r, 0, 4, &sz);
        check(d != NULL && sz == 4, "prefix: first 4 bytes");
        check(d && sz == 4 && memcmp(d, payload, 4) == 0, "prefix: bytes 0-3 correct");

        /* Read more than entry size — should clamp */
        d = shard_v2_read_entry_prefix(r, 0, 1000, &sz);
        check(d != NULL && sz == 10, "prefix: clamped to entry size");

        shard_v2_close(r);
    }
    printf("\n");
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char* argv[]) {
    const char* testdata = (argc > 1) ? argv[1] : "../ucodec/testdata";

    printf("=== shard_v2 mmap tests ===\n\n");

    /* --- Standalone roundtrip tests --- */
    test_mmap_invalid();
    test_mmap_roundtrip();
    test_mmap_large_entry();
    test_mmap_empty_entry();
    test_mmap_read_prefix();
    test_mmap_list_children(testdata);

    /* ============================================================
     * golden_basic.shard
     * ============================================================ */
    printf("[ mmap golden_basic.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "greeting",        UINT64_C(16700977181434310015), 5,  11,  11,  3381945770u, false },
            { "pattern/1k",      UINT64_C(11449517142474649281), 10, 1024, 1024, 752840335u, false },
            { "metadata/model",  UINT64_C( 7724208215623586332), 2,  63,  63, 4262061395u, false },
        };
        test_mmap_file(testdata, "golden_basic.shard",
                       2, 1, 162, 64, 3, 48, 208, 256, 0, 1407, entries);
        test_mmap_vs_regular(testdata, "golden_basic.shard");
    }
    printf("\n");

    /* ============================================================
     * golden_types.shard
     * ============================================================ */
    printf("[ mmap golden_types.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "tensor/weights",  UINT64_C(14268158869293769122), 1, 4096, 4096, 1817819105u, false },
            { "config.json",     UINT64_C( 4010899755935844745), 2,   27,   27, 2923727015u, false },
            { "config.glyph",    UINT64_C( 8537223467359895312), 4,   22,   22, 2175825978u, false },
            { "readme.txt",      UINT64_C( 7447530330435853781), 5,   35,   35,  436240408u, false },
            { "image/thumbnail", UINT64_C(12685321276006778552), 6,  256,  256, 4184433400u, false },
        };
        test_mmap_file(testdata, "golden_types.shard",
                       2, 2, 162, 64, 5, 48, 304, 384, 0, 4928, entries);
        test_mmap_vs_regular(testdata, "golden_types.shard");
    }
    printf("\n");

    /* ============================================================
     * golden_align16.shard
     * ============================================================ */
    printf("[ mmap golden_align16.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "a", UINT64_C(15154266338359012955), 5,   5,   5, 1623414395u, false },
            { "b", UINT64_C( 8666379929374662555), 10, 100, 100, 3235686185u, false },
        };
        test_mmap_file(testdata, "golden_align16.shard",
                       2, 1, 162, 16, 2, 48, 160, 176, 0, 292, entries);
        test_mmap_vs_regular(testdata, "golden_align16.shard");
    }
    printf("\n");

    /* ============================================================
     * golden_noalign.shard
     * ============================================================ */
    printf("[ mmap golden_noalign.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "x", UINT64_C( 6665539201184043299), 10, 4, 4, 4057757582u, false },
            { "y", UINT64_C(13923454618160480178), 10, 8, 8, 3145601971u, false },
        };
        test_mmap_file(testdata, "golden_noalign.shard",
                       2, 1, 162, 0, 2, 48, 160, 164, 0, 176, entries);
        test_mmap_vs_regular(testdata, "golden_noalign.shard");
    }
    printf("\n");

    /* ============================================================
     * golden_wshard.shard
     * ============================================================ */
    printf("[ mmap golden_wshard.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "signal/imu",               UINT64_C( 4930816044968525988), 10, 600, 600, 3613665374u, false },
            { "omen/imu/mlp_v1",          UINT64_C(16966689604114254082), 10, 600, 600, 4013429539u, false },
            { "residual/imu/sign2nddiff", UINT64_C( 4414458669179889149), 10,  75,  75, 4037150595u, false },
        };
        test_mmap_file(testdata, "golden_wshard.shard",
                       2, 5, 162, 64, 3, 48, 208, 320, 0, 1675, entries);
        test_mmap_vs_regular(testdata, "golden_wshard.shard");
    }
    printf("\n");

    /* ============================================================
     * golden_hierarchical.shard
     * ============================================================ */
    printf("[ mmap golden_hierarchical.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "layer.0/attention/q_proj/weight", UINT64_C( 5849259052093732705), 1,  512,  512, 3374896620u, false },
            { "layer.0/attention/k_proj/weight", UINT64_C(14695571574228440880), 1,  512,  512, 3600458702u, false },
            { "layer.0/attention/v_proj/weight", UINT64_C( 8985939025931988576), 1,  512,  512, 2404250664u, false },
            { "layer.0/attention/o_proj/weight", UINT64_C( 4427070529554003615), 1,  512,  512, 3252991925u, false },
            { "layer.0/ffn/gate/weight",         UINT64_C( 7777247805686235758), 1, 2048, 2048, 3801602145u, false },
            { "layer.0/ffn/up/weight",           UINT64_C( 5945135977106567835), 1, 2048, 2048, 1273217590u, false },
            { "layer.0/ffn/down/weight",         UINT64_C( 9799028502101065133), 1, 2048, 2048,  324665330u, false },
            { "layer.0/norm",                    UINT64_C( 5070234111195826601), 1,  128,  128,  412440727u, false },
        };
        test_mmap_file(testdata, "golden_hierarchical.shard",
                       2, 1, 162, 64, 8, 48, 448, 704, 0, 9024, entries);
        test_mmap_vs_regular(testdata, "golden_hierarchical.shard");
    }
    printf("\n");

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("=== Results: %d/%d passed, %d failed ===\n",
           g_passed, g_total, g_failed);

    return (g_failed == 0) ? 0 : 1;
}
