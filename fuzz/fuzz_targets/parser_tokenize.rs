// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Lexer expects UTF-8 strings
    if let Ok(s) = std::str::from_utf8(data) {
        // Tokenize should handle all valid/invalid inputs gracefully
        let _ = neumann_parser::tokenize(s);
    }
});
