#ifndef SHARD_H
#define SHARD_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================
 * Constants
 * ============================================================ */

#define SHARD_MAGIC            "SHRD"
#define SHARD_VERSION         0x02
#define SHARD_HEADER_SIZE      64
#define SHARD_INDEX_ENTRY_SIZE 48

/* Roles */
#define ROLE_UNKNOWN     0
#define ROLE_MOSH        1
#define ROLE_SAMPLE      2
#define ROLE_GEMMPANEL   3
#define ROLE_MANIFEST    4
#define ROLE_WSHARD      5
#define ROLE_UMSH        6

/* Alignment */
#define ALIGN_NONE  0
#define ALIGN_16    16
#define ALIGN_32    32
#define ALIGN_64    64

/* Compression */
#define COMPRESS_NONE  0
#define COMPRESS_ZSTD  1
#define COMPRESS_LZ4   2

/* Header flags */
#define FLAG_COMPRESSED        0x0001
#define FLAG_LITTLE_ENDIAN     0x0002
#define FLAG_BIG_ENDIAN        0x0004
#define FLAG_HAS_SCHEMA        0x0010
#define FLAG_HAS_CHECKSUMS     0x0020
#define FLAG_STREAMING         0x0040
#define FLAG_HAS_CONTENT_TYPES 0x0080
#define DEFAULT_FLAGS          (FLAG_LITTLE_ENDIAN | FLAG_HAS_CHECKSUMS | FLAG_HAS_CONTENT_TYPES)

/* Entry flags */
#define ENTRY_FLAG_COMPRESSED  0x0001
#define ENTRY_FLAG_ZSTD        0x0002
#define ENTRY_FLAG_LZ4         0x0004
#define ENTRY_FLAG_CHUNKED     0x0008

/* Content types */
#define CONTENT_TYPE_UNKNOWN        0
#define CONTENT_TYPE_TENSOR         1
#define CONTENT_TYPE_JSON           2
#define CONTENT_TYPE_COWRIE         3
#define CONTENT_TYPE_GLYPH          4
#define CONTENT_TYPE_TEXT           5
#define CONTENT_TYPE_IMAGE          6
#define CONTENT_TYPE_AUDIO          7
#define CONTENT_TYPE_VIDEO          8
#define CONTENT_TYPE_PROTO          9
#define CONTENT_TYPE_BLOB              10
#define CONTENT_TYPE_QMLN              11
#define CONTENT_TYPE_TENSOR_V3         12
#define CONTENT_TYPE_ANCHOR_SHARED     13
#define CONTENT_TYPE_DELTA_EXPERT      14
#define CONTENT_TYPE_CODEBOOK_SHARED   15
#define CONTENT_TYPE_EXPERT_INDICES    16
#define CONTENT_TYPE_USER_BASE      0x8000

/* ============================================================
 * Compression parameters (match Go reference implementation)
 * ============================================================ */

#define MIN_COMPRESS_SIZE       256
#define COMPRESSION_SAVINGS_NUM 9
#define COMPRESSION_SAVINGS_DEN 10
#define MAX_DECOMPRESS_SIZE     (1ULL << 30)   /* 1 GB safety cap */

/* ============================================================
 * Security limits
 * ============================================================ */

#define MAX_ENTRY_COUNT         10000000u           /* 10 million */
#define MAX_INDEX_SIZE          (1u << 30)           /* 1 GB */
#define MAX_STRING_TABLE_SIZE   (100u * 1024u * 1024u) /* 100 MB */

/* ============================================================
 * Structures (wire-format compatible, manually serialized)
 * ============================================================ */

/* In-memory representation of the 64-byte header.
 * NOT packed — we serialize/deserialize manually. */
typedef struct {
    uint8_t  magic[4];
    uint8_t  version;
    uint8_t  role;
    uint16_t flags;
    uint8_t  alignment;
    uint8_t  compression_default;
    uint16_t index_entry_size;
    uint32_t entry_count;
    uint64_t string_table_offset;
    uint64_t data_section_offset;
    uint64_t schema_offset;
    uint64_t total_file_size;
    uint8_t  reserved[16];
} shard_header_t;

/* In-memory representation of the 48-byte index entry. */
typedef struct {
    uint64_t name_hash;
    uint32_t name_offset;
    uint16_t name_len;
    uint16_t flags;
    uint64_t data_offset;
    uint64_t disk_size;
    uint64_t orig_size;
    uint32_t checksum;
    uint32_t reserved;   /* lower 16 bits = content_type */
} shard_index_entry_t;

/* ============================================================
 * Reader API
 * ============================================================ */

typedef struct shard_reader shard_reader_t;

/* Open from file path (reads entire file into malloc'd buffer). */
shard_reader_t* shard_open(const char* path);

/* Open from file path using mmap (zero-copy reads into the mapped region).
 * shard_read_entry() returns pointers directly into the mapping.
 * Call shard_close() to munmap and close the file descriptor. */
shard_reader_t* shard_open_mmap(const char* path);

/* Open from an existing memory buffer (does NOT take ownership). */
shard_reader_t* shard_from_buffer(const uint8_t* data, size_t len);

void shard_close(shard_reader_t* r);

const shard_header_t*      shard_header(const shard_reader_t* r);
uint32_t                       shard_entry_count(const shard_reader_t* r);
const char*                    shard_entry_name(const shard_reader_t* r, uint32_t i);
const shard_index_entry_t*  shard_get_entry(const shard_reader_t* r, uint32_t i);

/* Returns entry index, or -1 if not found. */
int32_t shard_lookup(const shard_reader_t* r, const char* name);
bool    shard_has_entry(const shard_reader_t* r, const char* name);

/* Returns pointer into the internal buffer. Sets *out_size.
 * The pointer is valid as long as the reader is open.
 * Returns NULL on error (bad index, checksum mismatch, etc.). */
const uint8_t* shard_read_entry(const shard_reader_t* r, uint32_t i, size_t* out_size);
const uint8_t* shard_read_entry_by_name(const shard_reader_t* r, const char* name, size_t* out_size);

/* Returns up to max_bytes of entry i without decompressing compressed entries
 * (returns NULL for compressed). Sets *out_size to actual bytes returned.
 * Returns NULL on error. Caller must NOT free the returned pointer. */
const uint8_t* shard_read_entry_prefix(const shard_reader_t* r, uint32_t i,
                                           size_t max_bytes, size_t* out_size);

/* Entry info helpers */
uint16_t shard_content_type(const shard_index_entry_t* e);
bool     shard_is_compressed(const shard_index_entry_t* e);

/* ============================================================
 * list_children
 *
 * Returns a NULL-terminated array of malloc'd strings, each the name of
 * an immediate child under `prefix`. The caller must free each string and
 * then the array itself.
 *
 * - If prefix is "" (empty), returns top-level components.
 * - If prefix ends with '/', returns entries that are exactly one
 *   component deeper (leaves are full names; directories are prefix+comp+"/").
 * - Directories are deduplicated.
 *
 * Returns NULL on allocation failure.  *out_count is set to the number of
 * strings (not counting the NULL terminator).
 * ============================================================ */
char** shard_list_children(const shard_reader_t* r, const char* prefix,
                               uint32_t* out_count);

/* Free the result of shard_list_children(). */
void shard_list_children_free(char** children, uint32_t count);

/* ============================================================
 * Path helpers
 * ============================================================ */

/* Split `path` on '/'. Fills `parts` with pointers into a single
 * malloc'd buffer (returned via *buf_out; caller frees *buf_out only,
 * not individual parts). Returns the number of components, or -1 on error.
 * parts and buf_out must not be NULL. */
int shard_split_path(const char* path, char*** parts_out, char** buf_out);

/* Join `n` parts with '/'. Returns a malloc'd string; caller frees. */
char* shard_join_path(const char* const* parts, int n);

/* Return a pointer to everything before the last '/' in `path`.
 * Writes into caller-provided `buf` of size `buf_size`.
 * Returns buf, or NULL if buf_size is too small.
 * If there is no '/', returns "". */
char* shard_path_parent(const char* path, char* buf, size_t buf_size);

/* Return a pointer to everything after the last '/' in `path`, or path
 * itself if there is no '/'. The returned pointer is into `path` directly
 * (no allocation). */
const char* shard_path_base(const char* path);

/* ============================================================
 * Hash / Checksum utilities
 * ============================================================ */

uint32_t shard_crc32c(const uint8_t* data, size_t len);
uint64_t shard_xxhash64(const char* name, size_t len);

/* ============================================================
 * Writer API
 * ============================================================ */

typedef struct shard_writer shard_writer_t;

shard_writer_t* shard_writer_new(uint8_t role);
void shard_writer_set_alignment(shard_writer_t* w, uint8_t align);

/* Returns 0 on success, -1 on error. */
int  shard_writer_add_entry(shard_writer_t* w, const char* name,
                                const uint8_t* data, size_t len);
int  shard_writer_add_entry_typed(shard_writer_t* w, const char* name,
                                      const uint8_t* data, size_t len,
                                      uint16_t content_type);

/* Add an entry with optional compression (COMPRESS_NONE, COMPRESS_ZSTD, COMPRESS_LZ4).
 * If len < MIN_COMPRESS_SIZE, or if compression does not achieve COMPRESSION_SAVINGS_NUM /
 * COMPRESSION_SAVINGS_DEN savings, the entry is stored uncompressed.
 * Returns 0 on success, -1 on error. */
int  shard_writer_add_entry_compressed(shard_writer_t* w, const char* name,
                                           const uint8_t* data, size_t len,
                                           uint8_t comp_type);

/* Set the default compression type for subsequent add_entry calls.
 * Currently applies only to add_entry_compressed. */
void shard_writer_set_compression(shard_writer_t* w, uint8_t comp);

int  shard_writer_write(shard_writer_t* w, const char* path);
void shard_writer_free(shard_writer_t* w);

/* ============================================================
 * Streaming Writer API
 *
 * Zero-copy streaming writer: data is written directly to disk entry-by-entry.
 * The header and index are written at the END via fseek(0) once all entries
 * are recorded. Requires max_entries upfront to pre-reserve the front-matter.
 *
 * Typical usage:
 *   shard_stream_writer_t* sw = shard_stream_writer_new(path, ROLE_MOSH, 1000);
 *   shard_stream_writer_set_alignment(sw, ALIGN_64);  // optional, before begin_data
 *   shard_stream_writer_begin_data(sw);
 *   shard_stream_writer_write_entry(sw, "weights", data, len);
 *   shard_stream_writer_finalize(sw);
 *   shard_stream_writer_free(sw);
 * ============================================================ */

typedef struct shard_stream_writer shard_stream_writer_t;

/* Create a new streaming writer. Returns NULL on allocation/IO failure. */
shard_stream_writer_t* shard_stream_writer_new(const char* path,
                                                       uint8_t role,
                                                       uint32_t max_entries);

/* Set alignment (must be called before begin_data). 0 = no alignment. */
void shard_stream_writer_set_alignment(shard_stream_writer_t* sw, uint8_t align);

/* Set default compression used by write_entry. Currently unused (plain writer
 * stores raw data). Reserved for future compressed streaming support. */
void shard_stream_writer_set_compression(shard_stream_writer_t* sw, uint8_t comp);

/* Compute reserved space and seek past it. Returns 0 on success, -1 on error. */
int shard_stream_writer_begin_data(shard_stream_writer_t* sw);

/* Write one entry. Alignment padding is inserted automatically.
 * Returns 0 on success, -1 on error. */
int shard_stream_writer_write_entry(shard_stream_writer_t* sw,
                                        const char* name,
                                        const uint8_t* data,
                                        size_t len);

/* Write one entry with an explicit content type.
 * Returns 0 on success, -1 on error. */
int shard_stream_writer_write_entry_typed(shard_stream_writer_t* sw,
                                              const char* name,
                                              const uint8_t* data,
                                              size_t len,
                                              uint16_t content_type);

/* Seek to file position 0 and write the complete header, index, and string
 * table. Closes the file. Returns 0 on success, -1 on error. */
int shard_stream_writer_finalize(shard_stream_writer_t* sw);

/* Abort: close the file without finalizing (leaves an incomplete file). */
void shard_stream_writer_free(shard_stream_writer_t* sw);

#endif /* SHARD_H */
