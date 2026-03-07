/*
 * shard_v2.c — C implementation of the Shard v2 binary container format.
 *
 * Zero external dependencies. CRC32C and xxHash64 are embedded directly.
 *
 * Wire format (little-endian throughout):
 *
 *   Header (64 bytes)
 *     0-3   magic  'S','H','R','D'
 *     4     version 0x02
 *     5     role
 *     6-7   flags (uint16 LE)
 *     8     alignment (0=none, 16, 32, 64)
 *     9     compression_default
 *     10-11 index_entry_size = 48 (uint16 LE)
 *     12-15 entry_count (uint32 LE)
 *     16-23 string_table_offset (uint64 LE)
 *     24-31 data_section_offset (uint64 LE)
 *     32-39 schema_offset (uint64 LE)
 *     40-47 total_file_size (uint64 LE)
 *     48-63 reserved (zeroes)
 *
 *   Index Entry (48 bytes each)
 *     0-7   name_hash (uint64 LE)
 *     8-11  name_offset (uint32 LE)
 *     12-13 name_len (uint16 LE)
 *     14-15 flags (uint16 LE)
 *     16-23 data_offset (uint64 LE)
 *     24-31 disk_size (uint64 LE)
 *     32-39 orig_size (uint64 LE)
 *     40-43 checksum CRC32C (uint32 LE)
 *     44-47 reserved (lower 16 bits = content_type)
 */

#include "shard_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <zstd.h>
#include <lz4.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ============================================================
 * Endian helpers (always write/read LE regardless of host)
 * ============================================================ */

static inline uint16_t read_le16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t read_le32(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] <<  8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline uint64_t read_le64(const uint8_t* p) {
    return (uint64_t)p[0]
         | ((uint64_t)p[1] <<  8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

static inline void write_le16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
}

static inline void write_le32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >>  8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static inline void write_le64(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >>  8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}

/* ============================================================
 * CRC32C (Castagnoli, polynomial 0x82F63B78)
 * ============================================================ */

static uint32_t crc32c_table[256];
static int      crc32c_table_ready = 0;

static void crc32c_init_table(void) {
    if (crc32c_table_ready) return;
    for (int i = 0; i < 256; i++) {
        uint32_t c = (uint32_t)i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
        }
        crc32c_table[i] = c;
    }
    crc32c_table_ready = 1;
}

uint32_t shard_v2_crc32c(const uint8_t* data, size_t len) {
    crc32c_init_table();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = crc32c_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/* ============================================================
 * xxHash64 — seed=0, identical to cespare/xxhash/v2
 *
 * Reference: https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
 * ============================================================ */

#define PRIME64_1  UINT64_C(0x9E3779B185EBCA87)
#define PRIME64_2  UINT64_C(0xC2B2AE3D27D4EB4F)
#define PRIME64_3  UINT64_C(0x165667B19E3779F9)
#define PRIME64_4  UINT64_C(0x85EBCA77C2B2AE63)
#define PRIME64_5  UINT64_C(0x27D4EB2F165667C5)

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc  = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) {
    val  = xxh64_round(0, val);
    acc ^= val;
    acc  = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t xxh64_avalanche(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

uint64_t shard_v2_xxhash64(const char* name, size_t len) {
    const uint8_t* p    = (const uint8_t*)name;
    const uint8_t* end  = p + len;
    uint64_t h64;

    if (len >= 32) {
        uint64_t v1 = UINT64_C(0)              + PRIME64_1 + PRIME64_2;
        uint64_t v2 = UINT64_C(0)              + PRIME64_2;
        uint64_t v3 = UINT64_C(0);
        uint64_t v4 = UINT64_C(0)              - PRIME64_1;

        const uint8_t* limit = end - 32;
        do {
            v1 = xxh64_round(v1, read_le64(p));  p += 8;
            v2 = xxh64_round(v2, read_le64(p));  p += 8;
            v3 = xxh64_round(v3, read_le64(p));  p += 8;
            v4 = xxh64_round(v4, read_le64(p));  p += 8;
        } while (p <= limit);

        h64  = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64  = xxh64_merge_round(h64, v1);
        h64  = xxh64_merge_round(h64, v2);
        h64  = xxh64_merge_round(h64, v3);
        h64  = xxh64_merge_round(h64, v4);
    } else {
        h64 = UINT64_C(0) + PRIME64_5;
    }

    h64 += (uint64_t)len;

    /* 8-byte chunks */
    while (p + 8 <= end) {
        uint64_t k1 = xxh64_round(0, read_le64(p));
        h64 ^= k1;
        h64  = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p   += 8;
    }

    /* 4-byte chunk */
    if (p + 4 <= end) {
        h64 ^= (uint64_t)read_le32(p) * PRIME64_1;
        h64  = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p   += 4;
    }

    /* Remaining bytes */
    while (p < end) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64  = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    return xxh64_avalanche(h64);
}

/* ============================================================
 * Header serialize / deserialize
 * ============================================================ */

static void serialize_header(const shard_v2_header_t* h, uint8_t* buf) {
    memset(buf, 0, SHARD_HEADER_SIZE);
    memcpy(buf, h->magic, 4);
    buf[4] = h->version;
    buf[5] = h->role;
    write_le16(buf +  6, h->flags);
    buf[8] = h->alignment;
    buf[9] = h->compression_default;
    write_le16(buf + 10, h->index_entry_size);
    write_le32(buf + 12, h->entry_count);
    write_le64(buf + 16, h->string_table_offset);
    write_le64(buf + 24, h->data_section_offset);
    write_le64(buf + 32, h->schema_offset);
    write_le64(buf + 40, h->total_file_size);
    /* bytes 48-63: reserved, already zeroed */
}

static bool parse_header(const uint8_t* buf, size_t buf_len, shard_v2_header_t* h) {
    if (buf_len < SHARD_HEADER_SIZE) return false;
    memcpy(h->magic, buf, 4);
    if (memcmp(h->magic, SHARD_MAGIC, 4) != 0) return false;
    h->version = buf[4];
    if (h->version != SHARD_VERSION2) return false;
    h->role               = buf[5];
    h->flags              = read_le16(buf +  6);
    h->alignment          = buf[8];
    h->compression_default = buf[9];
    h->index_entry_size   = read_le16(buf + 10);
    if (h->index_entry_size != SHARD_INDEX_ENTRY_SIZE) return false;
    h->entry_count          = read_le32(buf + 12);
    h->string_table_offset  = read_le64(buf + 16);
    h->data_section_offset  = read_le64(buf + 24);
    h->schema_offset        = read_le64(buf + 32);
    h->total_file_size      = read_le64(buf + 40);
    memcpy(h->reserved, buf + 48, 16);
    return true;
}

/* ============================================================
 * Index entry serialize / deserialize
 * ============================================================ */

static void serialize_entry(const shard_v2_index_entry_t* e, uint8_t* buf) {
    memset(buf, 0, SHARD_INDEX_ENTRY_SIZE);
    write_le64(buf +  0, e->name_hash);
    write_le32(buf +  8, e->name_offset);
    write_le16(buf + 12, e->name_len);
    write_le16(buf + 14, e->flags);
    write_le64(buf + 16, e->data_offset);
    write_le64(buf + 24, e->disk_size);
    write_le64(buf + 32, e->orig_size);
    write_le32(buf + 40, e->checksum);
    write_le32(buf + 44, e->reserved);
}

static void parse_entry(const uint8_t* buf, shard_v2_index_entry_t* e) {
    e->name_hash   = read_le64(buf +  0);
    e->name_offset = read_le32(buf +  8);
    e->name_len    = read_le16(buf + 12);
    e->flags       = read_le16(buf + 14);
    e->data_offset = read_le64(buf + 16);
    e->disk_size   = read_le64(buf + 24);
    e->orig_size   = read_le64(buf + 32);
    e->checksum    = read_le32(buf + 40);
    e->reserved    = read_le32(buf + 44);
}

/* ============================================================
 * Entry info helpers
 * ============================================================ */

uint16_t shard_v2_content_type(const shard_v2_index_entry_t* e) {
    return (uint16_t)(e->reserved & 0xFFFF);
}

bool shard_v2_is_compressed(const shard_v2_index_entry_t* e) {
    return (e->flags & ENTRY_FLAG_COMPRESSED) != 0;
}

/* ============================================================
 * Reader
 * ============================================================ */

struct shard_v2_reader {
    uint8_t*               buf;        /* malloc'd file contents; NULL if from_buffer or mmap */
    const uint8_t*         data;       /* points to buf, mmap_ptr, or external buffer */
    size_t                 data_len;
    bool                   owns_buf;   /* free buf on close */
    bool                   is_mmap;    /* true when backed by mmap */
    void*                  mmap_ptr;   /* mmap base address (when is_mmap) */
    size_t                 mmap_len;   /* mmap length in bytes (when is_mmap) */
    int                    mmap_fd;    /* file descriptor to close (when is_mmap) */
    shard_v2_header_t      header;
    shard_v2_index_entry_t* entries;   /* array of entry_count entries */
    char**                 names;      /* array of malloc'd name strings */
    uint32_t               entry_count;
    /* Decompression cache: lazily populated for compressed entries.
     * decompressed_cache[i] is NULL until first read; then holds malloc'd buffer.
     * decompressed_sizes[i] holds the decompressed byte count. */
    uint8_t**              decompressed_cache;
    size_t*                decompressed_sizes;
};

/* Internal: parse header + index from a populated reader->data buffer. */
static shard_v2_reader_t* reader_parse(shard_v2_reader_t* r) {
    uint32_t n = 0; /* declare before any goto */

    /* Parse header */
    if (!parse_header(r->data, r->data_len, &r->header)) {
        goto fail;
    }

    /* Security: validate entry_count */
    if (r->header.entry_count > MAX_ENTRY_COUNT) {
        goto fail;
    }

    /* Security: validate index size */
    {
        size_t index_sz = (size_t)r->header.entry_count * SHARD_INDEX_ENTRY_SIZE;
        if (index_sz > MAX_INDEX_SIZE) goto fail;
    }

    /* Security: validate string table size */
    {
        uint64_t st_off = r->header.string_table_offset;
        uint64_t ds_off = r->header.data_section_offset;
        if (ds_off > st_off) {
            uint64_t st_size = ds_off - st_off;
            if (st_size > MAX_STRING_TABLE_SIZE) goto fail;
        }
    }

    /* Validate total_file_size matches buffer length */
    if (r->header.total_file_size != (uint64_t)r->data_len) {
        goto fail;
    }

    n = r->header.entry_count;
    r->entry_count = n;

    /* Parse index entries */
    size_t index_start = SHARD_HEADER_SIZE;
    size_t index_size  = (size_t)n * SHARD_INDEX_ENTRY_SIZE;
    if (index_start + index_size > r->data_len) goto fail;

    r->entries = (shard_v2_index_entry_t*)malloc(
        sizeof(shard_v2_index_entry_t) * (n + 1));
    if (!r->entries) goto fail;

    for (uint32_t i = 0; i < n; i++) {
        parse_entry(r->data + index_start + (size_t)i * SHARD_INDEX_ENTRY_SIZE,
                    &r->entries[i]);
    }

    /* Parse string table — slice of the buffer between string_table_offset
     * and data_section_offset (includes alignment padding). */
    uint64_t st_off = r->header.string_table_offset;
    uint64_t ds_off = r->header.data_section_offset;
    if (st_off > r->data_len || ds_off > r->data_len || ds_off < st_off) goto fail;

    /* Resolve names */
    r->names = (char**)calloc(n + 1, sizeof(char*));
    if (!r->names) goto fail;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t name_off = r->entries[i].name_offset;
        uint16_t name_len = r->entries[i].name_len;
        uint64_t abs_off  = st_off + name_off;
        if (abs_off + name_len > r->data_len) goto fail;
        r->names[i] = (char*)malloc(name_len + 1);
        if (!r->names[i]) goto fail;
        memcpy(r->names[i], r->data + abs_off, name_len);
        r->names[i][name_len] = '\0';
    }

    /* Security: validate entry data offsets */
    {
        uint64_t ds_off = r->header.data_section_offset;
        uint64_t prev_end = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t off  = r->entries[i].data_offset;
            uint64_t size = r->entries[i].disk_size;
            if (off < ds_off) goto fail;
            if (off + size > (uint64_t)r->data_len) goto fail;
            if (i > 0 && off < prev_end) goto fail;
            prev_end = off + size;
        }
    }

    /* Allocate decompression cache arrays (calloc zeroes them = all NULL/0). */
    if (n > 0) {
        r->decompressed_cache = (uint8_t**)calloc(n, sizeof(uint8_t*));
        if (!r->decompressed_cache) goto fail;
        r->decompressed_sizes = (size_t*)calloc(n, sizeof(size_t));
        if (!r->decompressed_sizes) goto fail;
    }

    return r;

fail:
    /* Clean up partially allocated arrays */
    if (r->decompressed_cache) {
        for (uint32_t i = 0; i < n; i++) {
            free(r->decompressed_cache[i]);
        }
        free(r->decompressed_cache);
        r->decompressed_cache = NULL;
    }
    free(r->decompressed_sizes);
    r->decompressed_sizes = NULL;
    if (r->names) {
        for (uint32_t i = 0; i < n; i++) {
            if (r->names[i]) free(r->names[i]);
        }
        free(r->names);
        r->names = NULL;
    }
    free(r->entries);
    r->entries = NULL;
    if (r->is_mmap) {
        if (r->mmap_ptr) munmap(r->mmap_ptr, r->mmap_len);
        if (r->mmap_fd >= 0) close(r->mmap_fd);
    } else if (r->owns_buf) {
        free(r->buf);
        r->buf = NULL;
    }
    free(r);
    return NULL;
}

shard_v2_reader_t* shard_v2_open(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || sz < SHARD_HEADER_SIZE) { fclose(f); return NULL; }
    rewind(f);

    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }

    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    shard_v2_reader_t* r = (shard_v2_reader_t*)calloc(1, sizeof(*r));
    if (!r) { free(buf); return NULL; }

    r->buf       = buf;
    r->data      = buf;
    r->data_len  = (size_t)sz;
    r->owns_buf  = true;

    return reader_parse(r);
}

shard_v2_reader_t* shard_v2_from_buffer(const uint8_t* data, size_t len) {
    shard_v2_reader_t* r = (shard_v2_reader_t*)calloc(1, sizeof(*r));
    if (!r) return NULL;

    r->buf       = NULL;
    r->data      = data;
    r->data_len  = len;
    r->owns_buf  = false;

    return reader_parse(r);
}

shard_v2_reader_t* shard_v2_open_mmap(const char* path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    if ((size_t)st.st_size < SHARD_HEADER_SIZE) {
        close(fd);
        return NULL;
    }

    size_t len = (size_t)st.st_size;
    void* mapped = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    /* Construct the reader using the mapped memory as a read-only buffer. */
    shard_v2_reader_t* r = (shard_v2_reader_t*)calloc(1, sizeof(*r));
    if (!r) {
        munmap(mapped, len);
        close(fd);
        return NULL;
    }

    r->buf       = NULL;
    r->data      = (const uint8_t*)mapped;
    r->data_len  = len;
    r->owns_buf  = false;
    r->is_mmap   = true;
    r->mmap_ptr  = mapped;
    r->mmap_len  = len;
    r->mmap_fd   = fd;

    /* reader_parse handles mmap cleanup on failure (is_mmap + mmap_ptr/mmap_fd
     * are set before calling it, so the fail: path will call munmap + close). */
    return reader_parse(r);
}

void shard_v2_close(shard_v2_reader_t* r) {
    if (!r) return;
    /* Free decompression cache */
    if (r->decompressed_cache) {
        for (uint32_t i = 0; i < r->entry_count; i++) {
            free(r->decompressed_cache[i]);
        }
        free(r->decompressed_cache);
    }
    free(r->decompressed_sizes);
    if (r->names) {
        for (uint32_t i = 0; i < r->entry_count; i++) {
            free(r->names[i]);
        }
        free(r->names);
    }
    free(r->entries);
    if (r->is_mmap) {
        if (r->mmap_ptr) munmap(r->mmap_ptr, r->mmap_len);
        if (r->mmap_fd >= 0) close(r->mmap_fd);
    } else if (r->owns_buf) {
        free(r->buf);
    }
    free(r);
}

const shard_v2_header_t* shard_v2_header(const shard_v2_reader_t* r) {
    return r ? &r->header : NULL;
}

uint32_t shard_v2_entry_count(const shard_v2_reader_t* r) {
    return r ? r->entry_count : 0;
}

const char* shard_v2_entry_name(const shard_v2_reader_t* r, uint32_t i) {
    if (!r || i >= r->entry_count) return NULL;
    return r->names[i];
}

const shard_v2_index_entry_t* shard_v2_get_entry(const shard_v2_reader_t* r, uint32_t i) {
    if (!r || i >= r->entry_count) return NULL;
    return &r->entries[i];
}

int32_t shard_v2_lookup(const shard_v2_reader_t* r, const char* name) {
    if (!r || !name) return -1;
    size_t   len  = strlen(name);
    uint64_t hash = shard_v2_xxhash64(name, len);
    for (uint32_t i = 0; i < r->entry_count; i++) {
        if (r->entries[i].name_hash == hash) {
            if (r->names[i] && strcmp(r->names[i], name) == 0) {
                return (int32_t)i;
            }
        }
    }
    return -1;
}

bool shard_v2_has_entry(const shard_v2_reader_t* r, const char* name) {
    return shard_v2_lookup(r, name) >= 0;
}

const uint8_t* shard_v2_read_entry(const shard_v2_reader_t* r, uint32_t i, size_t* out_size) {
    if (!r || i >= r->entry_count) return NULL;
    const shard_v2_index_entry_t* e = &r->entries[i];

    if (e->data_offset + e->disk_size > r->data_len) return NULL;

    if (shard_v2_is_compressed(e)) {
        /* Compressed path: decompress into cache on first access. */
        shard_v2_reader_t* mutable_r = (shard_v2_reader_t*)r; /* cast away const for cache */

        if (mutable_r->decompressed_cache && mutable_r->decompressed_cache[i]) {
            /* Cache hit */
            if (out_size) *out_size = mutable_r->decompressed_sizes[i];
            return mutable_r->decompressed_cache[i];
        }

        size_t comp_size = (size_t)e->disk_size;
        size_t orig_size = (size_t)e->orig_size;

        if (orig_size > MAX_DECOMPRESS_SIZE) return NULL;
        if (orig_size == 0) return NULL;

        const uint8_t* compressed = r->data + e->data_offset;

        uint8_t* decompressed = (uint8_t*)malloc(orig_size);
        if (!decompressed) return NULL;

        size_t result = 0;
        if (e->flags & ENTRY_FLAG_ZSTD) {
            size_t ret = ZSTD_decompress(decompressed, orig_size, compressed, comp_size);
            if (ZSTD_isError(ret)) { free(decompressed); return NULL; }
            result = ret;
        } else if (e->flags & ENTRY_FLAG_LZ4) {
            int ret = LZ4_decompress_safe((const char*)compressed, (char*)decompressed,
                                          (int)comp_size, (int)orig_size);
            if (ret < 0) { free(decompressed); return NULL; }
            result = (size_t)ret;
        } else {
            /* ENTRY_FLAG_COMPRESSED set but no codec flag: unknown codec */
            free(decompressed);
            return NULL;
        }

        if (result != orig_size) { free(decompressed); return NULL; }

        /* Verify CRC32C on the decompressed data */
        if (r->header.flags & FLAG_HAS_CHECKSUMS) {
            uint32_t computed = shard_v2_crc32c(decompressed, result);
            if (computed != e->checksum) { free(decompressed); return NULL; }
        }

        /* Store in cache */
        if (mutable_r->decompressed_cache) {
            mutable_r->decompressed_cache[i] = decompressed;
            mutable_r->decompressed_sizes[i] = result;
        }

        if (out_size) *out_size = result;
        return decompressed;
    }

    /* Uncompressed path */
    const uint8_t* ptr = r->data + e->data_offset;
    size_t         sz  = (size_t)e->disk_size;

    /* Checksum verification */
    if (r->header.flags & FLAG_HAS_CHECKSUMS) {
        uint32_t got = shard_v2_crc32c(ptr, sz);
        if (got != e->checksum) return NULL;
    }

    if (out_size) *out_size = sz;
    return ptr;
}

const uint8_t* shard_v2_read_entry_by_name(const shard_v2_reader_t* r, const char* name, size_t* out_size) {
    int32_t idx = shard_v2_lookup(r, name);
    if (idx < 0) return NULL;
    return shard_v2_read_entry(r, (uint32_t)idx, out_size);
}

const uint8_t* shard_v2_read_entry_prefix(const shard_v2_reader_t* r, uint32_t i,
                                           size_t max_bytes, size_t* out_size) {
    if (!r || i >= r->entry_count) return NULL;
    const shard_v2_index_entry_t* e = &r->entries[i];
    /* Not supported for compressed entries */
    if (shard_v2_is_compressed(e)) return NULL;
    size_t sz = (size_t)e->disk_size;
    if (max_bytes < sz) sz = max_bytes;
    if (e->data_offset + sz > r->data_len) return NULL;
    if (out_size) *out_size = sz;
    return r->data + e->data_offset;
}

/* ============================================================
 * list_children
 * ============================================================ */

char** shard_v2_list_children(const shard_v2_reader_t* r, const char* prefix,
                               uint32_t* out_count) {
    if (!r || !prefix || !out_count) return NULL;

    size_t prefix_len = strlen(prefix);

    /* We'll collect up to entry_count children (worst case one per entry). */
    uint32_t n = r->entry_count;
    char** result = (char**)malloc(sizeof(char*) * (n + 1));
    if (!result) return NULL;

    uint32_t count = 0;

    for (uint32_t i = 0; i < n; i++) {
        const char* name = r->names[i];
        if (!name) continue;
        size_t name_len = strlen(name);

        /* Entry must start with prefix */
        if (name_len < prefix_len) continue;
        if (memcmp(name, prefix, prefix_len) != 0) continue;

        /* remainder after prefix */
        const char* rem = name + prefix_len;

        /* Find the first '/' in remainder */
        const char* slash = strchr(rem, '/');

        char* child;
        if (slash) {
            /* Directory entry: prefix + component + "/" */
            size_t comp_len = (size_t)(slash - rem);
            size_t child_len = prefix_len + comp_len + 1; /* +1 for "/" */
            child = (char*)malloc(child_len + 1);
            if (!child) {
                shard_v2_list_children_free(result, count);
                return NULL;
            }
            memcpy(child, prefix, prefix_len);
            memcpy(child + prefix_len, rem, comp_len);
            child[prefix_len + comp_len] = '/';
            child[child_len] = '\0';
        } else {
            /* Leaf entry: full name */
            child = (char*)malloc(name_len + 1);
            if (!child) {
                shard_v2_list_children_free(result, count);
                return NULL;
            }
            memcpy(child, name, name_len);
            child[name_len] = '\0';
        }

        /* Deduplicate: check if already in result */
        bool dup = false;
        for (uint32_t j = 0; j < count; j++) {
            if (strcmp(result[j], child) == 0) {
                dup = true;
                break;
            }
        }
        if (dup) {
            free(child);
        } else {
            result[count++] = child;
        }
    }

    result[count] = NULL;
    *out_count = count;
    return result;
}

void shard_v2_list_children_free(char** children, uint32_t count) {
    if (!children) return;
    for (uint32_t i = 0; i < count; i++) {
        free(children[i]);
    }
    free(children);
}

/* ============================================================
 * Path helpers
 * ============================================================ */

int shard_v2_split_path(const char* path, char*** parts_out, char** buf_out) {
    if (!path || !parts_out || !buf_out) return -1;

    if (*path == '\0') {
        /* Empty path: no components */
        char** parts = (char**)malloc(sizeof(char*));
        if (!parts) return -1;
        parts[0] = NULL;
        *parts_out = parts;
        *buf_out = NULL;
        return 0;
    }

    /* Count components (number of '/' + 1) */
    int count = 1;
    for (const char* p = path; *p; p++) {
        if (*p == '/') count++;
    }

    /* Duplicate path into a buffer we can mutate */
    size_t len = strlen(path);
    char* buf = (char*)malloc(len + 1);
    if (!buf) return -1;
    memcpy(buf, path, len + 1);

    char** parts = (char**)malloc(sizeof(char*) * (count + 1));
    if (!parts) { free(buf); return -1; }

    int n = 0;
    char* saveptr = NULL;
    char* tok = buf;
    /* Manual split on '/' */
    char* p = buf;
    parts[n++] = p;
    while (*p) {
        if (*p == '/') {
            *p = '\0';
            parts[n++] = p + 1;
        }
        p++;
    }
    parts[n] = NULL;
    (void)saveptr;
    (void)tok;

    *parts_out = parts;
    *buf_out   = buf;
    return n;
}

char* shard_v2_join_path(const char* const* parts, int n) {
    if (!parts || n <= 0) {
        char* empty = (char*)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    /* Calculate total length */
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        total += strlen(parts[i]);
    }
    total += (size_t)(n - 1); /* '/' separators */

    char* out = (char*)malloc(total + 1);
    if (!out) return NULL;

    size_t pos = 0;
    for (int i = 0; i < n; i++) {
        size_t len = strlen(parts[i]);
        memcpy(out + pos, parts[i], len);
        pos += len;
        if (i < n - 1) out[pos++] = '/';
    }
    out[pos] = '\0';
    return out;
}

char* shard_v2_path_parent(const char* path, char* buf, size_t buf_size) {
    if (!path || !buf || buf_size == 0) return NULL;
    const char* slash = strrchr(path, '/');
    if (!slash) {
        if (buf_size < 1) return NULL;
        buf[0] = '\0';
        return buf;
    }
    size_t parent_len = (size_t)(slash - path);
    if (parent_len + 1 > buf_size) return NULL;
    memcpy(buf, path, parent_len);
    buf[parent_len] = '\0';
    return buf;
}

const char* shard_v2_path_base(const char* path) {
    if (!path) return NULL;
    const char* slash = strrchr(path, '/');
    if (!slash) return path;
    return slash + 1;
}

/* ============================================================
 * Writer
 * ============================================================ */

typedef struct {
    char*    name;
    uint8_t* data;      /* disk data (may be compressed) */
    size_t   len;       /* disk data length (disk_size) */
    size_t   orig_len;  /* original uncompressed length */
    uint32_t checksum;  /* CRC32C of original (uncompressed) data */
    uint16_t flags;     /* entry flags (ENTRY_FLAG_* bits) */
    uint16_t content_type;
} pending_entry_t;

struct shard_v2_writer {
    uint8_t        role;
    uint8_t        alignment;
    uint8_t        compression; /* default compression for add_entry_compressed */
    pending_entry_t* entries;
    size_t           entry_count;
    size_t           entry_cap;
};

shard_v2_writer_t* shard_v2_writer_new(uint8_t role) {
    shard_v2_writer_t* w = (shard_v2_writer_t*)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->role        = role;
    w->alignment   = ALIGN_64; /* default */
    w->compression = COMPRESS_NONE;
    w->entry_cap   = 16;
    w->entries     = (pending_entry_t*)malloc(sizeof(pending_entry_t) * w->entry_cap);
    if (!w->entries) { free(w); return NULL; }
    return w;
}

void shard_v2_writer_set_alignment(shard_v2_writer_t* w, uint8_t align) {
    if (!w) return;
    w->alignment = align;
}

void shard_v2_writer_set_compression(shard_v2_writer_t* w, uint8_t comp) {
    if (!w) return;
    w->compression = comp;
}

/* Internal helper: store one entry (disk_data/disk_len may differ from orig data/len
 * when compression is active). Flags are the ENTRY_FLAG_* bits.
 * checksum is CRC32C of the original (uncompressed) data. */
static int writer_add_raw(shard_v2_writer_t* w, const char* name,
                           const uint8_t* disk_data, size_t disk_len,
                           size_t orig_len, uint32_t checksum,
                           uint16_t flags, uint16_t ct) {
    if (!w || !name || !disk_data) return -1;

    /* Grow array if needed */
    if (w->entry_count == w->entry_cap) {
        size_t new_cap = w->entry_cap * 2;
        pending_entry_t* ne = (pending_entry_t*)realloc(
            w->entries, sizeof(pending_entry_t) * new_cap);
        if (!ne) return -1;
        w->entries   = ne;
        w->entry_cap = new_cap;
    }

    pending_entry_t* e = &w->entries[w->entry_count];
    e->name = (char*)malloc(strlen(name) + 1);
    if (!e->name) return -1;
    strcpy(e->name, name);

    e->data = (uint8_t*)malloc(disk_len + 1); /* +1 to allow disk_len==0 */
    if (!e->data) { free(e->name); return -1; }
    memcpy(e->data, disk_data, disk_len);

    e->len          = disk_len;
    e->orig_len     = orig_len;
    e->checksum     = checksum;
    e->flags        = flags;
    e->content_type = ct;
    w->entry_count++;
    return 0;
}

static int writer_add(shard_v2_writer_t* w, const char* name,
                      const uint8_t* data, size_t len, uint16_t ct) {
    /* Checksum computed here for uncompressed entries */
    uint32_t csum = shard_v2_crc32c(data, len);
    return writer_add_raw(w, name, data, len, len, csum, 0, ct);
}

int shard_v2_writer_add_entry(shard_v2_writer_t* w, const char* name,
                               const uint8_t* data, size_t len) {
    return writer_add(w, name, data, len, CONTENT_TYPE_UNKNOWN);
}

int shard_v2_writer_add_entry_typed(shard_v2_writer_t* w, const char* name,
                                     const uint8_t* data, size_t len,
                                     uint16_t content_type) {
    return writer_add(w, name, data, len, content_type);
}

int shard_v2_writer_add_entry_compressed(shard_v2_writer_t* w, const char* name,
                                          const uint8_t* data, size_t len,
                                          uint8_t comp_type) {
    if (!w || !name || !data) return -1;

    /* Checksum is always over the original (uncompressed) data */
    uint32_t orig_checksum = shard_v2_crc32c(data, len);

    /* Below minimum size threshold or no compression requested: store uncompressed */
    if (len < MIN_COMPRESS_SIZE || comp_type == COMPRESS_NONE) {
        return writer_add_raw(w, name, data, len, len, orig_checksum, 0,
                              CONTENT_TYPE_UNKNOWN);
    }

    uint16_t       entry_flags    = 0;
    const uint8_t* disk_data      = data;
    size_t         disk_len       = len;
    uint8_t*       compressed_buf = NULL;

    if (comp_type == COMPRESS_ZSTD) {
        size_t bound = ZSTD_compressBound(len);
        compressed_buf = (uint8_t*)malloc(bound);
        if (compressed_buf) {
            size_t result = ZSTD_compress(compressed_buf, bound, data, len,
                                          0 /* default level */);
            if (!ZSTD_isError(result) &&
                result < len * COMPRESSION_SAVINGS_NUM / COMPRESSION_SAVINGS_DEN) {
                disk_data   = compressed_buf;
                disk_len    = result;
                entry_flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_ZSTD;
            }
        }
    } else if (comp_type == COMPRESS_LZ4) {
        int bound = LZ4_compressBound((int)len);
        if (bound > 0) {
            compressed_buf = (uint8_t*)malloc((size_t)bound);
            if (compressed_buf) {
                int result = LZ4_compress_default((const char*)data,
                                                   (char*)compressed_buf,
                                                   (int)len, bound);
                if (result > 0 &&
                    (size_t)result < len * COMPRESSION_SAVINGS_NUM / COMPRESSION_SAVINGS_DEN) {
                    disk_data   = compressed_buf;
                    disk_len    = (size_t)result;
                    entry_flags = ENTRY_FLAG_COMPRESSED | ENTRY_FLAG_LZ4;
                }
            }
        }
    }

    int rc = writer_add_raw(w, name, disk_data, disk_len, len, orig_checksum,
                            entry_flags, CONTENT_TYPE_UNKNOWN);
    free(compressed_buf);
    return rc;
}

/*
 * Layout (mirrors the Go ShardV2Writer.Close() logic exactly):
 *
 *   [header 64b][index N*48b][string_table][padding][data entries]
 *
 * data_section_offset is aligned to `alignment` bytes.
 * Each data entry is also aligned to `alignment` bytes.
 */
int shard_v2_writer_write(shard_v2_writer_t* w, const char* path) {
    if (!w || !path) return -1;

    uint32_t n     = (uint32_t)w->entry_count;
    uint8_t  align = w->alignment;

    /* Build string table */
    size_t st_size = 0;
    for (uint32_t i = 0; i < n; i++) {
        st_size += strlen(w->entries[i].name) + 1; /* +1 null terminator */
    }

    uint8_t* st = NULL;
    uint32_t* name_offsets = NULL;

    if (n > 0) {
        st = (uint8_t*)malloc(st_size);
        if (!st) return -1;
        name_offsets = (uint32_t*)malloc(sizeof(uint32_t) * n);
        if (!name_offsets) { free(st); return -1; }
    }

    size_t pos = 0;
    for (uint32_t i = 0; i < n; i++) {
        name_offsets[i] = (uint32_t)pos;
        size_t nlen = strlen(w->entries[i].name);
        memcpy(st + pos, w->entries[i].name, nlen);
        pos += nlen;
        st[pos++] = '\0';
    }

    /* Calculate offsets */
    uint64_t string_table_offset  = (uint64_t)(SHARD_HEADER_SIZE + (size_t)n * SHARD_INDEX_ENTRY_SIZE);
    uint64_t data_section_offset  = string_table_offset + (uint64_t)st_size;

    /* Align data_section_offset */
    if (align > 0) {
        uint64_t a = (uint64_t)align;
        data_section_offset = (data_section_offset + a - 1) & ~(a - 1);
    }

    /* Calculate per-entry data offsets */
    uint64_t* data_offsets = NULL;
    if (n > 0) {
        data_offsets = (uint64_t*)malloc(sizeof(uint64_t) * n);
        if (!data_offsets) { free(st); free(name_offsets); return -1; }
    }

    uint64_t cur = data_section_offset;
    for (uint32_t i = 0; i < n; i++) {
        if (align > 0) {
            uint64_t a = (uint64_t)align;
            cur = (cur + a - 1) & ~(a - 1);
        }
        data_offsets[i] = cur;
        cur += (uint64_t)w->entries[i].len;
    }
    uint64_t total_size = cur;

    /* Determine flags */
    uint16_t hdr_flags = FLAG_LITTLE_ENDIAN | FLAG_HAS_CHECKSUMS;
    bool     any_compressed = false;
    for (uint32_t i = 0; i < n; i++) {
        if (w->entries[i].content_type != CONTENT_TYPE_UNKNOWN) {
            hdr_flags |= FLAG_HAS_CONTENT_TYPES;
        }
        if (w->entries[i].flags & ENTRY_FLAG_COMPRESSED) {
            any_compressed = true;
        }
    }
    if (any_compressed) {
        hdr_flags |= FLAG_COMPRESSED;
    }

    /* Open output file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        free(st); free(name_offsets); free(data_offsets);
        return -1;
    }

    /* Write header */
    uint8_t hdr_buf[SHARD_HEADER_SIZE];
    shard_v2_header_t h;
    memcpy(h.magic, SHARD_MAGIC, 4);
    h.version              = SHARD_VERSION2;
    h.role                 = w->role;
    h.flags                = hdr_flags;
    h.alignment            = align;
    h.compression_default  = w->compression;
    h.index_entry_size     = SHARD_INDEX_ENTRY_SIZE;
    h.entry_count          = n;
    h.string_table_offset  = string_table_offset;
    h.data_section_offset  = data_section_offset;
    h.schema_offset        = 0;
    h.total_file_size      = total_size;
    memset(h.reserved, 0, 16);
    serialize_header(&h, hdr_buf);

    if (fwrite(hdr_buf, 1, SHARD_HEADER_SIZE, f) != SHARD_HEADER_SIZE) goto io_fail;

    /* Write index entries */
    for (uint32_t i = 0; i < n; i++) {
        shard_v2_index_entry_t e;
        const char* nm = w->entries[i].name;

        e.name_hash   = shard_v2_xxhash64(nm, strlen(nm));
        e.name_offset = name_offsets[i];
        e.name_len    = (uint16_t)strlen(nm);
        e.flags       = w->entries[i].flags;
        e.data_offset = data_offsets[i];
        e.disk_size   = (uint64_t)w->entries[i].len;
        e.orig_size   = (uint64_t)w->entries[i].orig_len;
        e.checksum    = w->entries[i].checksum;
        e.reserved    = (uint32_t)w->entries[i].content_type;

        uint8_t ebuf[SHARD_INDEX_ENTRY_SIZE];
        serialize_entry(&e, ebuf);
        if (fwrite(ebuf, 1, SHARD_INDEX_ENTRY_SIZE, f) != SHARD_INDEX_ENTRY_SIZE) goto io_fail;
    }

    /* Write string table */
    if (st_size > 0) {
        if (fwrite(st, 1, st_size, f) != st_size) goto io_fail;
    }

    /* Padding between string table and data section */
    {
        uint64_t written_so_far = string_table_offset + (uint64_t)st_size;
        if (written_so_far < data_section_offset) {
            uint64_t pad_len = data_section_offset - written_so_far;
            uint8_t  zero[64] = {0};
            while (pad_len > 0) {
                size_t chunk = (pad_len > 64) ? 64 : (size_t)pad_len;
                if (fwrite(zero, 1, chunk, f) != chunk) goto io_fail;
                pad_len -= (uint64_t)chunk;
            }
        }
    }

    /* Write data entries with inter-entry alignment padding */
    {
        uint64_t file_pos = data_section_offset;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t target = data_offsets[i];
            if (file_pos < target) {
                uint64_t pad_len = target - file_pos;
                uint8_t  zero[64] = {0};
                while (pad_len > 0) {
                    size_t chunk = (pad_len > 64) ? 64 : (size_t)pad_len;
                    if (fwrite(zero, 1, chunk, f) != chunk) goto io_fail;
                    pad_len -= (uint64_t)chunk;
                }
                file_pos = target;
            }

            size_t dl = w->entries[i].len;
            if (dl > 0) {
                if (fwrite(w->entries[i].data, 1, dl, f) != dl) goto io_fail;
            }
            file_pos += (uint64_t)dl;
        }
    }

    fclose(f);
    free(st);
    free(name_offsets);
    free(data_offsets);
    return 0;

io_fail:
    fclose(f);
    free(st);
    free(name_offsets);
    free(data_offsets);
    return -1;
}

void shard_v2_writer_free(shard_v2_writer_t* w) {
    if (!w) return;
    for (size_t i = 0; i < w->entry_count; i++) {
        free(w->entries[i].name);
        free(w->entries[i].data);
    }
    free(w->entries);
    free(w);
}

/* ============================================================
 * Streaming Writer
 * ============================================================ */

/* Per-entry metadata recorded by write_entry, used during finalize. */
typedef struct {
    char*    name;
    uint64_t data_offset;
    uint64_t disk_size;
    uint64_t orig_size;
    uint32_t checksum;
    uint16_t flags;
} stream_entry_meta_t;

struct shard_v2_stream_writer {
    FILE*               file;
    uint8_t             role;
    uint8_t             alignment;
    uint8_t             compression;
    uint32_t            max_entries;
    uint32_t            count;
    uint64_t            reserved_space;
    uint64_t            current_offset;
    bool                started;
    bool                finalized;
    stream_entry_meta_t* entries;  /* array of max_entries */
};

/* Helper: write `len` zero bytes at the current file position. */
static int sw_write_zeros(FILE* f, size_t len) {
    uint8_t zero[256] = {0};
    while (len > 0) {
        size_t chunk = (len > sizeof(zero)) ? sizeof(zero) : len;
        if (fwrite(zero, 1, chunk, f) != chunk) return -1;
        len -= chunk;
    }
    return 0;
}

shard_v2_stream_writer_t* shard_v2_stream_writer_new(const char* path,
                                                       uint8_t role,
                                                       uint32_t max_entries) {
    if (!path || max_entries == 0) return NULL;

    shard_v2_stream_writer_t* sw =
        (shard_v2_stream_writer_t*)calloc(1, sizeof(*sw));
    if (!sw) return NULL;

    sw->file = fopen(path, "wb");
    if (!sw->file) {
        free(sw);
        return NULL;
    }

    sw->entries = (stream_entry_meta_t*)calloc(max_entries, sizeof(stream_entry_meta_t));
    if (!sw->entries) {
        fclose(sw->file);
        free(sw);
        return NULL;
    }

    sw->role        = role;
    sw->alignment   = ALIGN_64;
    sw->compression = COMPRESS_NONE;
    sw->max_entries = max_entries;
    sw->count       = 0;
    sw->started     = false;
    sw->finalized   = false;
    sw->reserved_space = 0;
    sw->current_offset = 0;

    return sw;
}

void shard_v2_stream_writer_set_alignment(shard_v2_stream_writer_t* sw, uint8_t align) {
    if (!sw) return;
    sw->alignment = align;
}

void shard_v2_stream_writer_set_compression(shard_v2_stream_writer_t* sw, uint8_t comp) {
    if (!sw) return;
    sw->compression = comp;
}

int shard_v2_stream_writer_begin_data(shard_v2_stream_writer_t* sw) {
    if (!sw) return -1;
    if (sw->started) return -1;  /* already called */

    uint32_t n = sw->max_entries;

    /* Reserve: header + index array + ~64 bytes per name for string table */
    uint64_t reserved = (uint64_t)SHARD_HEADER_SIZE
                      + (uint64_t)n * (uint64_t)SHARD_INDEX_ENTRY_SIZE
                      + (uint64_t)n * 64ULL;

    if (sw->alignment > 0) {
        uint64_t a = (uint64_t)sw->alignment;
        reserved = (reserved + a - 1) & ~(a - 1);
    }

    sw->reserved_space = reserved;
    sw->current_offset = reserved;

    /* Write zeros to pre-allocate the reserved region on disk */
    if (sw_write_zeros(sw->file, (size_t)reserved) != 0) return -1;

    sw->started = true;
    return 0;
}

int shard_v2_stream_writer_write_entry(shard_v2_stream_writer_t* sw,
                                        const char* name,
                                        const uint8_t* data,
                                        size_t len) {
    if (!sw || !name || !data) return -1;
    if (!sw->started || sw->finalized)  return -1;
    if (sw->count >= sw->max_entries)   return -1;

    /* Align the write position */
    if (sw->alignment > 0) {
        uint64_t a       = (uint64_t)sw->alignment;
        uint64_t aligned = (sw->current_offset + a - 1) & ~(a - 1);
        if (aligned > sw->current_offset) {
            size_t pad = (size_t)(aligned - sw->current_offset);
            if (sw_write_zeros(sw->file, pad) != 0) return -1;
            sw->current_offset = aligned;
        }
    }

    /* Write data */
    if (len > 0) {
        if (fwrite(data, 1, len, sw->file) != len) return -1;
    }

    uint32_t checksum = shard_v2_crc32c(data, len);

    /* Record metadata — copy name string */
    stream_entry_meta_t* e = &sw->entries[sw->count];
    e->name = (char*)malloc(strlen(name) + 1);
    if (!e->name) return -1;
    strcpy(e->name, name);

    e->data_offset = sw->current_offset;
    e->disk_size   = (uint64_t)len;
    e->orig_size   = (uint64_t)len;
    e->checksum    = checksum;
    e->flags       = 0;

    sw->current_offset += (uint64_t)len;
    sw->count++;
    return 0;
}

int shard_v2_stream_writer_finalize(shard_v2_stream_writer_t* sw) {
    if (!sw) return -1;
    if (sw->finalized) return -1;
    sw->finalized = true;

    uint32_t n = sw->count;

    /* Build string table in memory */
    size_t    st_size     = 0;
    uint32_t* name_offsets = NULL;
    uint8_t*  st           = NULL;

    for (uint32_t i = 0; i < n; i++) {
        st_size += strlen(sw->entries[i].name) + 1;
    }

    if (n > 0) {
        name_offsets = (uint32_t*)malloc(sizeof(uint32_t) * n);
        if (!name_offsets) goto fail_nomem;

        if (st_size > 0) {
            st = (uint8_t*)malloc(st_size);
            if (!st) goto fail_nomem;

            size_t pos = 0;
            for (uint32_t i = 0; i < n; i++) {
                name_offsets[i] = (uint32_t)pos;
                size_t nlen = strlen(sw->entries[i].name);
                memcpy(st + pos, sw->entries[i].name, nlen);
                pos += nlen;
                st[pos++] = '\0';
            }
        }
    }

    /* Verify string table fits in the reserved region */
    {
        uint64_t actual_front = (uint64_t)SHARD_HEADER_SIZE
                              + (uint64_t)n * (uint64_t)SHARD_INDEX_ENTRY_SIZE
                              + (uint64_t)st_size;
        if (sw->alignment > 0) {
            uint64_t a = (uint64_t)sw->alignment;
            actual_front = (actual_front + a - 1) & ~(a - 1);
        }
        if (actual_front > sw->reserved_space) goto io_fail;
    }

    {
        uint64_t string_table_offset = (uint64_t)SHARD_HEADER_SIZE
                                     + (uint64_t)n * (uint64_t)SHARD_INDEX_ENTRY_SIZE;
        uint64_t data_section_offset = sw->reserved_space;
        uint64_t total_file_size     = sw->current_offset;

        /* Seek to beginning of file */
        if (fseek(sw->file, 0, SEEK_SET) != 0) goto io_fail;

        /* Write header */
        {
            shard_v2_header_t h;
            memcpy(h.magic, SHARD_MAGIC, 4);
            h.version             = SHARD_VERSION2;
            h.role                = sw->role;
            h.flags               = DEFAULT_FLAGS | FLAG_STREAMING;
            h.alignment           = sw->alignment;
            h.compression_default = sw->compression;
            h.index_entry_size    = SHARD_INDEX_ENTRY_SIZE;
            h.entry_count         = n;
            h.string_table_offset = string_table_offset;
            h.data_section_offset = data_section_offset;
            h.schema_offset       = 0;
            h.total_file_size     = total_file_size;
            memset(h.reserved, 0, 16);

            uint8_t hdr_buf[SHARD_HEADER_SIZE];
            serialize_header(&h, hdr_buf);
            if (fwrite(hdr_buf, 1, SHARD_HEADER_SIZE, sw->file) != SHARD_HEADER_SIZE)
                goto io_fail;
        }

        /* Write index entries */
        for (uint32_t i = 0; i < n; i++) {
            stream_entry_meta_t* e = &sw->entries[i];
            shard_v2_index_entry_t ie;
            ie.name_hash   = shard_v2_xxhash64(e->name, strlen(e->name));
            ie.name_offset = name_offsets ? name_offsets[i] : 0;
            ie.name_len    = (uint16_t)strlen(e->name);
            ie.flags       = e->flags;
            ie.data_offset = e->data_offset;
            ie.disk_size   = e->disk_size;
            ie.orig_size   = e->orig_size;
            ie.checksum    = e->checksum;
            ie.reserved    = 0;

            uint8_t ebuf[SHARD_INDEX_ENTRY_SIZE];
            serialize_entry(&ie, ebuf);
            if (fwrite(ebuf, 1, SHARD_INDEX_ENTRY_SIZE, sw->file) != SHARD_INDEX_ENTRY_SIZE)
                goto io_fail;
        }

        /* Write string table */
        if (st_size > 0 && st != NULL) {
            if (fwrite(st, 1, st_size, sw->file) != st_size) goto io_fail;
        }

        /* Pad from end of string table to data section boundary */
        {
            uint64_t written_so_far = string_table_offset + (uint64_t)st_size;
            if (written_so_far < data_section_offset) {
                uint64_t pad_len = data_section_offset - written_so_far;
                if (sw_write_zeros(sw->file, (size_t)pad_len) != 0) goto io_fail;
            }
        }
    }

    fclose(sw->file);
    sw->file = NULL;
    free(st);
    free(name_offsets);
    return 0;

fail_nomem:
    free(st);
    free(name_offsets);
    fclose(sw->file);
    sw->file = NULL;
    return -1;

io_fail:
    free(st);
    free(name_offsets);
    fclose(sw->file);
    sw->file = NULL;
    return -1;
}

void shard_v2_stream_writer_free(shard_v2_stream_writer_t* sw) {
    if (!sw) return;
    if (sw->file) {
        fclose(sw->file);
        sw->file = NULL;
    }
    for (uint32_t i = 0; i < sw->count; i++) {
        free(sw->entries[i].name);
    }
    free(sw->entries);
    free(sw);
}
