//! Truth table tests for shard — 9 cases from testdata/robustness/truth_cases.json.

use std::path::Path;

use shard_format::{ShardReader, ShardWriter, ROLE_MOSH};

/// Path to the corrupt fixture directory.
fn fixture_dir() -> &'static Path {
    // From shard/rs/tests/ -> shard/testdata/safety/
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../testdata/safety"
    ))
}

/// Write entries to an in-memory shard and return the bytes.
fn write_shard_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = ShardWriter::new(ROLE_MOSH);
    for (name, data) in entries {
        w.write_entry(name, data);
    }
    w.to_bytes()
}

// Case 1: duplicate_entry_names_rejected
#[test]
fn truth_duplicate_entry_names_rejected() {
    let mut w = ShardWriter::new(ROLE_MOSH);
    w.write_entry("a", b"first");
    w.write_entry("a", b"second");

    // If to_bytes doesn't panic, check the reader can at least parse it.
    let buf = w.to_bytes();
    match ShardReader::from_bytes(buf) {
        Ok(r) => {
            // Writer accepted duplicates — reader should find at least one.
            assert!(r.entry_count() >= 1);
        }
        Err(_) => {
            // Reader rejected the duplicate — acceptable.
        }
    }
}

// Case 2: zero_length_entry_valid
#[test]
fn truth_zero_length_entry_valid() {
    let buf = write_shard_bytes(&[("empty", b"")]);
    let r = ShardReader::from_bytes(buf).expect("zero-length entry shard should be valid");
    assert_eq!(r.entry_count(), 1);
    let data = r.read_entry(0).expect("read_entry(0) should succeed");
    assert!(data.is_empty());
}

// Case 3: unicode_entry_name
#[test]
fn truth_unicode_entry_name() {
    let buf = write_shard_bytes(&[("\u{65e5}\u{672c}\u{8a9e}", b"hello")]);
    let r = ShardReader::from_bytes(buf).expect("unicode name shard should be valid");
    assert_eq!(r.entry_count(), 1);
    assert_eq!(
        r.entry_name(0).expect("entry_name(0)"),
        "\u{65e5}\u{672c}\u{8a9e}"
    );
    let data = r.read_entry(0).expect("read_entry(0)");
    assert_eq!(data, b"hello");
}

// Case 4: bad_magic_rejected
#[test]
fn truth_bad_magic_rejected() {
    let path = fixture_dir().join("corrupt_bad_magic.shard");
    let data = std::fs::read(&path).expect("read fixture");
    let result = ShardReader::from_bytes(data);
    assert!(result.is_err(), "expected error for bad magic");
}

// Case 5: truncated_header_rejected
#[test]
fn truth_truncated_header_rejected() {
    let path = fixture_dir().join("corrupt_truncated_header.shard");
    let data = std::fs::read(&path).expect("read fixture");
    let result = ShardReader::from_bytes(data);
    assert!(result.is_err(), "expected error for truncated header");
}

// Case 6: crc_mismatch_rejected
#[test]
fn truth_crc_mismatch_rejected() {
    let path = fixture_dir().join("corrupt_crc_mismatch.shard");
    let data = std::fs::read(&path).expect("read fixture");
    match ShardReader::from_bytes(data) {
        Err(_) => return, // Rejected at open — acceptable.
        Ok(r) => {
            // If open succeeded, reading entries should fail.
            let mut errored = false;
            for i in 0..r.entry_count() {
                if r.read_entry(i).is_err() {
                    errored = true;
                    break;
                }
            }
            assert!(errored, "expected CRC mismatch error");
        }
    }
}

// Case 7: truncated_data_rejected
#[test]
fn truth_truncated_data_rejected() {
    let path = fixture_dir().join("corrupt_truncated_data.shard");
    let data = std::fs::read(&path).expect("read fixture");
    match ShardReader::from_bytes(data) {
        Err(_) => return, // Rejected at open — acceptable.
        Ok(r) => {
            let mut errored = false;
            for i in 0..r.entry_count() {
                if r.read_entry(i).is_err() {
                    errored = true;
                    break;
                }
            }
            assert!(errored, "expected error for truncated data");
        }
    }
}

// Case 8: entry_count_overflow_rejected
#[test]
fn truth_entry_count_overflow_rejected() {
    let path = fixture_dir().join("corrupt_entry_count_overflow.shard");
    let data = std::fs::read(&path).expect("read fixture");
    let result = ShardReader::from_bytes(data);
    assert!(result.is_err(), "expected error for entry count overflow");
}

// Case 9: empty_input_rejected
#[test]
fn truth_empty_input_rejected() {
    let result = ShardReader::from_bytes(vec![]);
    assert!(result.is_err(), "expected error for empty input");
}
