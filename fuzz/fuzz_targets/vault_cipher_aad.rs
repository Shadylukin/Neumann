// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_vault::{Cipher, MasterKey};

#[derive(Arbitrary, Debug)]
struct AadInput {
    plaintext: Vec<u8>,
    aad: Vec<u8>,
    key_bytes: [u8; 32],
}

fuzz_target!(|input: AadInput| {
    let key = MasterKey::from_bytes(input.key_bytes);
    let cipher = Cipher::new(&key);

    if let Ok((ciphertext, nonce)) = cipher.encrypt_with_aad(&input.plaintext, &input.aad) {
        // Same AAD roundtrips
        if let Ok(decrypted) = cipher.decrypt_with_aad(&ciphertext, &nonce, &input.aad) {
            assert_eq!(input.plaintext, decrypted);
        }
        // Different AAD must fail
        let mut wrong_aad = input.aad.clone();
        wrong_aad.push(0xFF);
        assert!(
            cipher
                .decrypt_with_aad(&ciphertext, &nonce, &wrong_aad)
                .is_err()
        );
    }
});
