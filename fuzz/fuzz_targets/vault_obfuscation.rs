// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_vault::{MasterKey, Obfuscator};

#[derive(Arbitrary, Debug)]
struct ObfuscationInput {
    key_bytes: [u8; 32],
    secret_keys: Vec<String>,
}

fuzz_target!(|input: ObfuscationInput| {
    let master = MasterKey::from_bytes(input.key_bytes);
    let obfuscator = Obfuscator::new(&master);

    for raw_key in &input.secret_keys {
        // Clamp length to avoid excessive allocation
        if raw_key.len() > 256 {
            continue;
        }

        let obfuscated = obfuscator.obfuscate_key(raw_key);

        // Determinism: same input always produces same output
        let again = obfuscator.obfuscate_key(raw_key);
        assert_eq!(obfuscated, again, "obfuscation must be deterministic");

        // Non-empty keys should produce a different obfuscated form
        if !raw_key.is_empty() {
            assert_ne!(
                obfuscated, *raw_key,
                "obfuscated key must differ from original"
            );
        }

        // Obfuscated output should be hex-encoded (only hex chars)
        assert!(
            obfuscated.chars().all(|c| c.is_ascii_hexdigit()),
            "obfuscated key must be hex-encoded"
        );
    }
});
