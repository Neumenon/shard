//! Fuzz target: lookup and prefix search with arbitrary entry names.
//!
//! Matches Go's FuzzReadEntryByName coverage:
//! - empty names
//! - very long names
//! - Unicode names (multi-byte UTF-8)
//! - names containing null bytes
//! - deep path names (many '/' separators)
//! - names that are prefixes of other names
//!
//! Strategy: build a real shard via ShardV2Writer using the fuzzed name list,
//! then exercise all lookup and prefix APIs.
#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use shard_format::{ShardV2Reader, ShardV2Writer, ROLE_MOSH};

#[derive(Arbitrary, Debug)]
struct LookupInput {
    /// Names to insert into the shard.
    names: Vec<String>,
    /// Names to look up (may differ from inserted names).
    queries: Vec<String>,
    /// Prefixes to list.
    prefixes: Vec<String>,
}

fuzz_target!(|input: LookupInput| {
    // Sanity-bound the number of entries to keep fuzzing fast.
    if input.names.len() > 100 {
        return;
    }

    let mut w = ShardV2Writer::new(ROLE_MOSH);
    for (i, name) in input.names.iter().enumerate().take(50) {
        // Skip names that would cause obvious writer issues.
        if name.len() > 1000 {
            continue;
        }
        // Names containing null bytes are valid UTF-8 strings in Rust but the
        // string table stores raw UTF-8 with a null terminator — a null byte
        // inside a name would corrupt the string table. The writer writes the
        // raw bytes, so the reader may resolve a shorter name. This exercises
        // the tolerance of the reader's string table parsing.
        w.write_entry(name, format!("payload_{i}").as_bytes());
    }

    let bytes = w.to_bytes();
    let reader = match ShardV2Reader::from_bytes(bytes) {
        Ok(r) => r,
        Err(_) => return, // writer produced an unreadable shard — skip
    };

    // Look up inserted names (most should succeed).
    for name in &input.names {
        let _ = reader.lookup(name);
        let _ = reader.has_entry(name);
        let _ = reader.read_entry_by_name(name);
    }

    // Look up arbitrary query strings (most will return None/Err).
    for q in input.queries.iter().take(20) {
        let _ = reader.lookup(q);
        let _ = reader.has_entry(q);
        let _ = reader.read_entry_by_name(q);
    }

    // Adversarial fixed queries — these must never panic.
    let _ = reader.lookup("");
    let _ = reader.lookup("\x00");
    let _ = reader.lookup("\u{FFFE}"); // BOM-like
    let _ = reader.lookup("nonexistent/deep/path/name");
    let _ = reader.lookup(&"a".repeat(4096)); // very long

    // Prefix listing.
    for pfx in input.prefixes.iter().take(10) {
        let _ = reader.list_prefix(pfx);
        let _ = reader.list_children(pfx);
    }
    let _ = reader.list_prefix("");
    let _ = reader.list_prefix("a/");
    let _ = reader.list_children("");
    let _ = reader.list_children("layer.");

    // Read every entry by index.
    for i in 0..reader.entry_count() {
        let _ = reader.read_entry(i);
        let _ = reader.read_entry_prefix(i, 32);
    }
});
