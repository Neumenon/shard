//! Fuzz target: malformed ShardV2Header parsing.
//!
//! Matches Go's FuzzReadShardV2Header coverage:
//! - truncated input
//! - bad magic bytes
//! - wrong version
//! - max entry count / huge index_entry_size
//! - all-zeros, all-0xFF inputs
#![no_main]
use libfuzzer_sys::fuzz_target;
use shard_format::ShardV2Header;

fuzz_target!(|data: &[u8]| {
    // Must never panic — only return Ok or Err.
    let _ = ShardV2Header::from_bytes(data);
});
