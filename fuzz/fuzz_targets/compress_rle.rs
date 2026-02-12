// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{rle_decode, rle_encode};

#[derive(Arbitrary, Debug)]
struct RleInput {
    values: Vec<i64>,
}

fuzz_target!(|input: RleInput| {
    // Test RLE roundtrip with arbitrary i64 sequences
    let encoded = rle_encode(&input.values);
    let decoded = rle_decode(&encoded);
    assert_eq!(input.values, decoded, "RLE roundtrip failed");
});
