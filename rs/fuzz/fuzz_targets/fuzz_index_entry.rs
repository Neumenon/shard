//! Fuzz target: malformed IndexEntry parsing.
//!
//! Matches Go's FuzzParseIndexEntry coverage:
//! - valid 48-byte buffers with arbitrary field values
//! - compressed/chunked flag combinations
//! - max values for offsets, sizes, checksums, content types
//!
//! We exercise this through the public ShardReader path rather than the
//! private IndexEntry::from_bytes, by constructing a minimal valid shard
//! whose single index entry contains the fuzzed 48 bytes.
#![no_main]
use libfuzzer_sys::fuzz_target;
use shard_format::{HEADER_SIZE, INDEX_ENTRY_SIZE, SHARD_MAGIC, SHARD_VERSION};

/// Build the smallest possible shard wrapper around a raw 48-byte index entry
/// blob and attempt to open it.  The reader will:
///   1. Parse the header (which we make syntactically valid).
///   2. Parse the index entry directly from `entry_bytes`.
///   3. Validate data offsets, string table bounds, etc.
/// This exercises every branch in the entry-parsing and bounds-checking code
/// without exposing the private `from_bytes` constructor.
fuzz_target!(|entry_bytes: &[u8]| {
    if entry_bytes.len() < INDEX_ENTRY_SIZE {
        return;
    }
    let entry_slice = &entry_bytes[..INDEX_ENTRY_SIZE];

    // Layout:
    //   [0..64)   header
    //   [64..112) index entry (our fuzzed bytes)
    //   112..     string table starts here, data section starts here too
    //             (entry_count=1, so string_table_offset=112, data_section_offset=112)
    //             total_file_size = 112

    const ST_OFFSET: u64 = (HEADER_SIZE + INDEX_ENTRY_SIZE) as u64; // 112
    const TOTAL: usize = HEADER_SIZE + INDEX_ENTRY_SIZE; // 112

    let mut buf = vec![0u8; TOTAL];

    // Write a syntactically valid header.
    buf[0..4].copy_from_slice(SHARD_MAGIC);
    buf[4] = SHARD_VERSION;
    buf[5] = 1; // ROLE_MOSH
    buf[6..8].copy_from_slice(&0x00A2u16.to_le_bytes()); // DEFAULT_FLAGS
    buf[8] = 0; // ALIGN_NONE
    buf[9] = 0; // COMPRESS_NONE
    buf[10..12].copy_from_slice(&(INDEX_ENTRY_SIZE as u16).to_le_bytes()); // index_entry_size=48
    buf[12..16].copy_from_slice(&1u32.to_le_bytes()); // entry_count=1
    buf[16..24].copy_from_slice(&ST_OFFSET.to_le_bytes()); // string_table_offset
    buf[24..32].copy_from_slice(&ST_OFFSET.to_le_bytes()); // data_section_offset (= string_table_offset, 0-length string table)
    buf[32..40].copy_from_slice(&0u64.to_le_bytes()); // schema_offset=0
    buf[40..48].copy_from_slice(&(TOTAL as u64).to_le_bytes()); // total_file_size

    // Paste the fuzzed index entry bytes directly after the header.
    buf[HEADER_SIZE..HEADER_SIZE + INDEX_ENTRY_SIZE].copy_from_slice(entry_slice);

    // Attempt to open — must never panic.
    let result = shard_format::ShardReader::from_bytes(buf);
    if let Ok(reader) = result {
        // If the reader opened (entry's data_offset and disk_size happened to
        // be zero or point inside the file), exercise the entry accessors.
        for i in 0..reader.entry_count() {
            let info = reader.get_entry_info(i);
            let _ = info.content_type();
            let _ = info.compressed();
            let _ = reader.entry_name(i);
            // Attempt a read — may return Err(ChecksumMismatch) or Ok.
            let _ = reader.read_entry(i);
        }
    }
});
