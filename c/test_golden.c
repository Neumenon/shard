/*
 * test_golden.c — Golden-file parity tests for shard.c
 *
 * Usage: ./test_golden <testdata_dir>
 *
 * Expected values are hard-coded from:
 *   testdata/golden_manifest.json
 *
 * Verifies:
 *   - Header fields (version, role, flags, alignment, entry_count, offsets)
 *   - Entry fields  (name, name_hash, content_type, orig_size, disk_size, checksum)
 *   - CRC32C computed from the entry data matches the stored checksum
 *   - Lookup-by-name works
 *   - xxHash64 known values
 *   - CRC32C known values
 */

#include "shard.h"

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

/* Build a path string: dir + "/" + filename */
static void make_path(char* out, size_t outsz, const char* dir, const char* file) {
    snprintf(out, outsz, "%s/%s", dir, file);
}

/* ============================================================
 * Expected entry descriptor
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
 * Individual file tests
 * ============================================================ */

static void test_file(const char* dir, const char* filename,
                      /* header fields */
                      uint8_t exp_version, uint8_t exp_role,
                      uint16_t exp_flags, uint8_t exp_alignment,
                      uint32_t exp_entry_count,
                      uint16_t exp_index_entry_size,
                      uint64_t exp_string_table_offset,
                      uint64_t exp_data_section_offset,
                      uint64_t exp_schema_offset,
                      uint64_t exp_total_file_size,
                      /* entries */
                      const expected_entry_t* exp_entries) {

    char path[1024];
    make_path(path, sizeof(path), dir, filename);

    char desc[256];
    snprintf(desc, sizeof(desc), "open(%s)", filename);

    shard_reader_t* r = shard_open(path);
    check(r != NULL, desc);
    if (!r) return;

    /* Header checks */
    const shard_header_t* h = shard_header(r);

    snprintf(desc, sizeof(desc), "%s: magic == SHRD", filename);
    check(memcmp(h->magic, "SHRD", 4) == 0, desc);

    snprintf(desc, sizeof(desc), "%s: version == %u", filename, exp_version);
    check(h->version == exp_version, desc);

    snprintf(desc, sizeof(desc), "%s: role == %u", filename, exp_role);
    check(h->role == exp_role, desc);

    snprintf(desc, sizeof(desc), "%s: flags == 0x%04x", filename, exp_flags);
    check(h->flags == exp_flags, desc);

    snprintf(desc, sizeof(desc), "%s: alignment == %u", filename, exp_alignment);
    check(h->alignment == exp_alignment, desc);

    snprintf(desc, sizeof(desc), "%s: compression_default == 0", filename);
    check(h->compression_default == 0, desc);

    snprintf(desc, sizeof(desc), "%s: index_entry_size == %u", filename, exp_index_entry_size);
    check(h->index_entry_size == exp_index_entry_size, desc);

    snprintf(desc, sizeof(desc), "%s: entry_count == %u", filename, exp_entry_count);
    check(h->entry_count == exp_entry_count, desc);

    snprintf(desc, sizeof(desc), "%s: string_table_offset == %llu", filename,
             (unsigned long long)exp_string_table_offset);
    check(h->string_table_offset == exp_string_table_offset, desc);

    snprintf(desc, sizeof(desc), "%s: data_section_offset == %llu", filename,
             (unsigned long long)exp_data_section_offset);
    check(h->data_section_offset == exp_data_section_offset, desc);

    snprintf(desc, sizeof(desc), "%s: schema_offset == %llu", filename,
             (unsigned long long)exp_schema_offset);
    check(h->schema_offset == exp_schema_offset, desc);

    snprintf(desc, sizeof(desc), "%s: total_file_size == %llu", filename,
             (unsigned long long)exp_total_file_size);
    check(h->total_file_size == exp_total_file_size, desc);

    snprintf(desc, sizeof(desc), "%s: entry_count() == %u", filename, exp_entry_count);
    check(shard_entry_count(r) == exp_entry_count, desc);

    /* Entry checks */
    for (uint32_t i = 0; i < exp_entry_count; i++) {
        const expected_entry_t* ex = &exp_entries[i];

        /* Name */
        const char* name = shard_entry_name(r, i);
        snprintf(desc, sizeof(desc), "%s[%u]: name == \"%s\"", filename, i, ex->name);
        check(name && strcmp(name, ex->name) == 0, desc);

        const shard_index_entry_t* e = shard_get_entry(r, i);
        if (!e) {
            snprintf(desc, sizeof(desc), "%s[%u]: get_entry != NULL", filename, i);
            check(false, desc);
            continue;
        }

        /* name_hash */
        snprintf(desc, sizeof(desc), "%s[%u]: name_hash == %llu", filename, i,
                 (unsigned long long)ex->name_hash);
        check(e->name_hash == ex->name_hash, desc);

        /* content_type */
        snprintf(desc, sizeof(desc), "%s[%u]: content_type == %u", filename, i, ex->content_type);
        check(shard_content_type(e) == ex->content_type, desc);

        /* orig_size */
        snprintf(desc, sizeof(desc), "%s[%u]: orig_size == %llu", filename, i,
                 (unsigned long long)ex->orig_size);
        check(e->orig_size == ex->orig_size, desc);

        /* disk_size */
        snprintf(desc, sizeof(desc), "%s[%u]: disk_size == %llu", filename, i,
                 (unsigned long long)ex->disk_size);
        check(e->disk_size == ex->disk_size, desc);

        /* checksum */
        snprintf(desc, sizeof(desc), "%s[%u]: checksum == 0x%08x", filename, i, ex->checksum);
        check(e->checksum == ex->checksum, desc);

        /* compressed flag */
        snprintf(desc, sizeof(desc), "%s[%u]: compressed == %s", filename, i,
                 ex->compressed ? "true" : "false");
        check(shard_is_compressed(e) == ex->compressed, desc);

        /* Read entry and verify CRC32C matches stored checksum */
        size_t out_size = 0;
        const uint8_t* data = shard_read_entry(r, i, &out_size);
        snprintf(desc, sizeof(desc), "%s[%u]: read_entry != NULL (checksum ok)", filename, i);
        check(data != NULL, desc);

        if (data) {
            snprintf(desc, sizeof(desc), "%s[%u]: read data size == %llu", filename, i,
                     (unsigned long long)ex->disk_size);
            check(out_size == (size_t)ex->disk_size, desc);

            uint32_t computed_crc = shard_crc32c(data, out_size);
            snprintf(desc, sizeof(desc), "%s[%u]: computed CRC32C == stored checksum", filename, i);
            check(computed_crc == ex->checksum, desc);
        }

        /* Lookup by name */
        if (name) {
            int32_t idx = shard_lookup(r, name);
            snprintf(desc, sizeof(desc), "%s[%u]: lookup(\"%s\") == %u", filename, i, name, i);
            check(idx == (int32_t)i, desc);
        }
    }

    /* has_entry for a known name */
    if (exp_entry_count > 0) {
        snprintf(desc, sizeof(desc), "%s: has_entry(\"%s\") == true", filename, exp_entries[0].name);
        check(shard_has_entry(r, exp_entries[0].name), desc);
    }

    /* lookup for a non-existent name */
    snprintf(desc, sizeof(desc), "%s: lookup(\"__nonexistent__\") == -1", filename);
    check(shard_lookup(r, "__nonexistent__") == -1, desc);

    shard_close(r);
}

/* ============================================================
 * CRC32C known-value tests
 * ============================================================ */

static void test_crc32c(void) {
    /* CRC32C of empty string = 0x00000000 */
    {
        uint32_t got = shard_crc32c((const uint8_t*)"", 0);
        check(got == 0x00000000u, "crc32c(\"\", 0) == 0x00000000");
    }
    /* CRC32C("123456789") = 0xE3069283 */
    {
        const uint8_t* s = (const uint8_t*)"123456789";
        uint32_t got = shard_crc32c(s, 9);
        check(got == 0xE3069283u, "crc32c(\"123456789\") == 0xE3069283");
    }
    /* CRC32C of single zero byte */
    {
        const uint8_t b = 0x00;
        uint32_t got = shard_crc32c(&b, 1);
        check(got == 0x527D5351u, "crc32c({0x00}) == 0x527D5351");
    }
    /* CRC32C of "hello world" == checksum from golden_basic entry 0 */
    {
        const uint8_t* s = (const uint8_t*)"hello world";
        uint32_t got = shard_crc32c(s, 11);
        /* Expected: checksum from golden_manifest entry greeting = 3381945770 */
        check(got == 3381945770u, "crc32c(\"hello world\") == 3381945770");
    }
}

/* ============================================================
 * xxHash64 known-value tests
 * ============================================================ */

static void test_xxhash64(void) {
    struct {
        const char* s;
        uint64_t    expected;
    } cases[] = {
        { "greeting",        UINT64_C(16700977181434310015) },
        { "pattern/1k",      UINT64_C(11449517142474649281) },
        { "metadata/model",  UINT64_C( 7724208215623586332) },
        { "tensor/weights",  UINT64_C(14268158869293769122) },
        { "config.json",     UINT64_C( 4010899755935844745) },
        { "config.glyph",    UINT64_C( 8537223467359895312) },
        { "readme.txt",      UINT64_C( 7447530330435853781) },
        { "image/thumbnail", UINT64_C(12685321276006778552) },
        { "a",               UINT64_C(15154266338359012955) },
        { "b",               UINT64_C( 8666379929374662555) },
        { "x",               UINT64_C( 6665539201184043299) },
        { "y",               UINT64_C(13923454618160480178) },
        { "signal/imu",      UINT64_C( 4930816044968525988) },
        { "omen/imu/mlp_v1", UINT64_C(16966689604114254082) },
        { "residual/imu/sign2nddiff", UINT64_C(4414458669179889149) },
    };

    char desc[256];
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        uint64_t got = shard_xxhash64(cases[i].s, strlen(cases[i].s));
        snprintf(desc, sizeof(desc), "xxhash64(\"%s\") == %llu",
                 cases[i].s, (unsigned long long)cases[i].expected);
        check(got == cases[i].expected, desc);
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char* argv[]) {
    const char* testdata = (argc > 1) ? argv[1]
                                      : "../testdata";

    printf("=== shard golden tests ===\n\n");

    /* --- CRC32C --- */
    printf("[ CRC32C known values ]\n");
    test_crc32c();
    printf("\n");

    /* --- xxHash64 --- */
    printf("[ xxHash64 known values ]\n");
    test_xxhash64();
    printf("\n");

    /* ============================================================
     * golden_basic.shard
     * header: version=2, role=1, flags=162, align=64, entries=3
     *         string_table_offset=208, data_section_offset=256
     *         schema_offset=0, total_file_size=1407
     * ============================================================ */
    printf("[ golden_basic.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "greeting",        UINT64_C(16700977181434310015), 5,  11,  11,  3381945770u, false },
            { "pattern/1k",      UINT64_C(11449517142474649281), 10, 1024, 1024, 752840335u, false },
            { "metadata/model",  UINT64_C( 7724208215623586332), 2,  63,  63, 4262061395u, false },
        };
        test_file(testdata, "golden_basic.shard",
                  2, 1, 162, 64, 3, 48, 208, 256, 0, 1407, entries);
    }
    printf("\n");

    /* ============================================================
     * golden_types.shard
     * header: version=2, role=2, flags=162, align=64, entries=5
     *         string_table_offset=304, data_section_offset=384
     *         schema_offset=0, total_file_size=4928
     * ============================================================ */
    printf("[ golden_types.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "tensor/weights",  UINT64_C(14268158869293769122), 1, 4096, 4096, 1817819105u, false },
            { "config.json",     UINT64_C( 4010899755935844745), 2,   27,   27, 2923727015u, false },
            { "config.glyph",    UINT64_C( 8537223467359895312), 4,   22,   22, 2175825978u, false },
            { "readme.txt",      UINT64_C( 7447530330435853781), 5,   35,   35,  436240408u, false },
            { "image/thumbnail", UINT64_C(12685321276006778552), 6,  256,  256, 4184433400u, false },
        };
        test_file(testdata, "golden_types.shard",
                  2, 2, 162, 64, 5, 48, 304, 384, 0, 4928, entries);
    }
    printf("\n");

    /* ============================================================
     * golden_align16.shard
     * header: version=2, role=1, flags=162, align=16, entries=2
     *         string_table_offset=160, data_section_offset=176
     *         schema_offset=0, total_file_size=292
     * ============================================================ */
    printf("[ golden_align16.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "a", UINT64_C(15154266338359012955), 5,   5,   5, 1623414395u, false },
            { "b", UINT64_C( 8666379929374662555), 10, 100, 100, 3235686185u, false },
        };
        test_file(testdata, "golden_align16.shard",
                  2, 1, 162, 16, 2, 48, 160, 176, 0, 292, entries);
    }
    printf("\n");

    /* ============================================================
     * golden_noalign.shard
     * header: version=2, role=1, flags=162, align=0, entries=2
     *         string_table_offset=160, data_section_offset=164
     *         schema_offset=0, total_file_size=176
     * ============================================================ */
    printf("[ golden_noalign.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "x", UINT64_C( 6665539201184043299), 10, 4, 4, 4057757582u, false },
            { "y", UINT64_C(13923454618160480178), 10, 8, 8, 3145601971u, false },
        };
        test_file(testdata, "golden_noalign.shard",
                  2, 1, 162, 0, 2, 48, 160, 164, 0, 176, entries);
    }
    printf("\n");

    /* ============================================================
     * golden_wshard.shard
     * header: version=2, role=5, flags=162, align=64, entries=3
     *         string_table_offset=208, data_section_offset=320
     *         schema_offset=0, total_file_size=1675
     * ============================================================ */
    printf("[ golden_wshard.shard ]\n");
    {
        static const expected_entry_t entries[] = {
            { "signal/imu",               UINT64_C( 4930816044968525988), 10, 600, 600, 3613665374u, false },
            { "omen/imu/mlp_v1",          UINT64_C(16966689604114254082), 10, 600, 600, 4013429539u, false },
            { "residual/imu/sign2nddiff", UINT64_C( 4414458669179889149), 10,  75,  75, 4037150595u, false },
        };
        test_file(testdata, "golden_wshard.shard",
                  2, 5, 162, 64, 3, 48, 208, 320, 0, 1675, entries);
    }
    printf("\n");

    /* ============================================================
     * golden_hierarchical.shard
     * header: version=2, role=1, flags=162, align=64, entries=8
     *         string_table_offset=448, data_section_offset=704
     *         schema_offset=0, total_file_size=9024
     * ============================================================ */
    printf("[ golden_hierarchical.shard ]\n");
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
        test_file(testdata, "golden_hierarchical.shard",
                  2, 1, 162, 64, 8, 48, 448, 704, 0, 9024, entries);
    }
    printf("\n");

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("=== Results: %d/%d passed, %d failed ===\n",
           g_passed, g_total, g_failed);

    return (g_failed == 0) ? 0 : 1;
}
