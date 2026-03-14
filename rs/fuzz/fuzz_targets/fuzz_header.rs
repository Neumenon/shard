//! Fuzz target: malformed ShardHeader parsing.
//!
//! Matches Go's FuzzReadShardHeader coverage:
//! - truncated input
//! - bad magic bytes
//! - wrong version
//! - max entry count / huge index_entry_size
//! - all-zeros, all-0xFF inputs
#![no_main]
use libfuzzer_sys::fuzz_target;
use shard_format::ShardHeader;

fuzz_target!(|data: &[u8]| {
    // Must never panic — only return Ok or Err.
    let _ = ShardHeader::from_bytes(data);
});
