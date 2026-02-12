// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Parser expects UTF-8 strings
    if let Ok(s) = std::str::from_utf8(data) {
        // parse_all handles multiple semicolon-separated statements
        let _ = neumann_parser::parse_all(s);
    }
});
