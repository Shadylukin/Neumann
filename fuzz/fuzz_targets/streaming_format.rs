#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use tensor_compress::streaming::StreamingReader;

fuzz_target!(|data: &[u8]| {
    // Test that StreamingReader never panics on arbitrary bytes
    // Minimum size needed for a valid streaming format:
    // 4 (magic) + 8 (trailer_len) = 12 bytes minimum
    if data.len() < 12 {
        return;
    }

    // Try to open the streaming reader with arbitrary data
    let cursor = Cursor::new(data.to_vec());
    if let Ok(mut reader) = StreamingReader::open(cursor) {
        // If opening succeeds, try to iterate through entries
        for entry_result in reader.by_ref() {
            match entry_result {
                Ok(_entry) => {
                    // Valid entry - continue
                }
                Err(_) => {
                    // Error reading entry - expected for malformed data
                    break;
                }
            }
        }
    }
    // Opening failed - expected for malformed data
});
