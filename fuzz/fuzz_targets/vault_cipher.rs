// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_vault::{Cipher, MasterKey};

#[derive(Arbitrary, Debug)]
struct CipherInput {
    plaintext: Vec<u8>,
    key_bytes: [u8; 32],
}

fuzz_target!(|input: CipherInput| {
    // Create cipher from arbitrary key bytes
    let key = MasterKey::from_bytes(input.key_bytes);
    let cipher = Cipher::new(&key);

    // Test encrypt/decrypt roundtrip
    if let Ok((ciphertext, nonce)) = cipher.encrypt(&input.plaintext) {
        let decrypted = cipher
            .decrypt(&ciphertext, &nonce)
            .expect("Decryption must succeed for freshly encrypted data");
        assert_eq!(input.plaintext, decrypted, "Cipher roundtrip failed");
    }
});
