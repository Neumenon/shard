//! Fuzz target: full ShardV2Reader::from_bytes with arbitrary crafted files.
//!
//! Matches Go's FuzzOpenShardV2 coverage:
//! - inverted offsets (string_table_offset > data_section_offset)
//! - max entry counts
//! - huge string table
//! - data offsets that overlap, underflow, or overflow
//! - total_file_size mismatches
//! - all byte patterns that pass magic + version checks
#![no_main]
use libfuzzer_sys::fuzz_target;
use shard_format::ShardV2Reader;

fuzz_target!(|data: &[u8]| {
    // Try to open arbitrary bytes as a shard.
    // Must never panic — the reader must return Ok or a well-typed Err.
    if let Ok(reader) = ShardV2Reader::from_bytes(data.to_vec()) {
        // If it opened successfully, exercise all reader methods.
        let count = reader.entry_count();
        let _ = reader.header();
        let _ = reader.entry_names();

        // Walk up to 10 entries to avoid excessive work during fuzzing.
        for i in 0..count.min(10) {
            let _ = reader.entry_name(i);
            let _ = reader.get_entry_info(i);
            let _ = reader.read_entry(i);
            let _ = reader.read_entry_prefix(i, 64);
        }

        let _ = reader.lookup("test");
        let _ = reader.lookup("");
        let _ = reader.lookup("\x00");
        let _ = reader.has_entry("test");
        let _ = reader.list_prefix("layer.");
        let _ = reader.list_prefix("");
        let _ = reader.list_children("");
        let _ = reader.list_children("layer.");
        let _ = reader.read_metadata();
    }
});
