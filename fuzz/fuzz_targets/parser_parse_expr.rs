// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Parser expects UTF-8 strings
    if let Ok(s) = std::str::from_utf8(data) {
        // Expression parser uses Pratt parsing with bounded depth
        let _ = neumann_parser::parse_expr(s);
    }
});
