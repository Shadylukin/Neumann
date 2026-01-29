// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_compress::{compress_ids, decompress_ids};

fuzz_target!(|data: &[u8]| {
    // Test roundtrip: decompress arbitrary bytes, then recompress
    let decompressed = decompress_ids(data);
    if !decompressed.is_empty() {
        let recompressed = compress_ids(&decompressed);
        let final_decompress = decompress_ids(&recompressed);
        assert_eq!(
            decompressed, final_decompress,
            "compress_ids roundtrip failed"
        );
    }
});
