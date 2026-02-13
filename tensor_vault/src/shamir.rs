// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shamir secret sharing for master key splitting and reconstruction.
//!
//! Uses GF(256) arithmetic with the AES irreducible polynomial (0x11B)
//! to split and reconstruct secrets without external dependencies.

use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{
    key::{MasterKey, KEY_SIZE},
    Result, VaultError,
};

/// GF(256) finite field arithmetic using the AES irreducible polynomial.
///
/// All operations use the AES reduction polynomial x^8 + x^4 + x^3 + x + 1
/// (0x11B), making this field identical to the one used in AES.
mod gf256 {
    /// Multiply two elements in GF(256) via Russian Peasant multiplication.
    ///
    /// Reduces modulo the AES polynomial 0x11B after each doubling step.
    pub const fn mul(mut a: u8, mut b: u8) -> u8 {
        let mut result: u8 = 0;
        let mut i = 0;
        while i < 8 {
            if b & 1 != 0 {
                result ^= a;
            }
            let carry = a & 0x80;
            a <<= 1;
            if carry != 0 {
                a ^= 0x1B;
            }
            b >>= 1;
            i += 1;
        }
        result
    }

    /// Compute the multiplicative inverse via Fermat's little theorem.
    ///
    /// Returns a^254 in GF(256), which equals a^(-1) for nonzero a.
    /// Returns 0 for input 0 (no inverse exists).
    pub const fn inv(a: u8) -> u8 {
        if a == 0 {
            return 0;
        }
        let a2 = mul(a, a);
        let a4 = mul(a2, a2);
        let a8 = mul(a4, a4);
        let a16 = mul(a8, a8);
        let a32 = mul(a16, a16);
        let a64 = mul(a32, a32);
        let a128 = mul(a64, a64);
        // a^254 = a^(128+64+32+16+8+4+2)
        mul(mul(mul(mul(mul(mul(a128, a64), a32), a16), a8), a4), a2)
    }

    /// Evaluate polynomial at `x` using Horner's method.
    ///
    /// Coefficients are `[a0, a1, ..., a_{n-1}]` where `a0` is the constant
    /// term (the secret byte).
    pub fn eval_poly(coeffs: &[u8], x: u8) -> u8 {
        let mut result = 0u8;
        for &coeff in coeffs.iter().rev() {
            result = mul(result, x) ^ coeff;
        }
        result
    }
}

/// Configuration for Shamir secret sharing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShamirConfig {
    /// Total number of shares to generate.
    pub total_shares: u8,
    /// Minimum shares needed for reconstruction.
    pub threshold: u8,
}

impl ShamirConfig {
    /// Validate the configuration parameters.
    fn validate(&self) -> Result<()> {
        if self.threshold < 2 {
            return Err(VaultError::ShamirError(
                "threshold must be at least 2".to_string(),
            ));
        }
        if self.threshold > self.total_shares {
            return Err(VaultError::ShamirError(
                "threshold cannot exceed total shares".to_string(),
            ));
        }
        if self.total_shares < 2 {
            return Err(VaultError::ShamirError(
                "total shares must be at least 2".to_string(),
            ));
        }
        Ok(())
    }
}

/// A single key share from Shamir splitting.
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KeyShare {
    /// Share index (1-based x-coordinate in GF(256)).
    pub index: u8,
    /// Raw share data: `[x, y_0, y_1, ..., y_31]` (33 bytes for a 32-byte key).
    pub data: Vec<u8>,
}

/// Split a master key into shares using Shamir secret sharing over GF(256).
///
/// For each byte of the key, constructs a random polynomial of degree
/// `threshold - 1` with the key byte as the constant term, then evaluates
/// at x-coordinates 1 through `total_shares`.
///
/// # Errors
///
/// Returns `VaultError::ShamirError` if the configuration is invalid.
pub fn split_master_key(key: &MasterKey, config: &ShamirConfig) -> Result<Vec<KeyShare>> {
    config.validate()?;

    let key_bytes = key.as_bytes();
    let threshold = config.threshold as usize;

    // Pre-allocate share buffers: [x_coord, y_0, y_1, ..., y_31]
    let mut shares: Vec<Vec<u8>> = (1..=config.total_shares)
        .map(|x| {
            let mut data = vec![0u8; KEY_SIZE + 1];
            data[0] = x;
            data
        })
        .collect();

    let mut coeffs = vec![0u8; threshold];
    let mut rng_buf = vec![0u8; threshold - 1];

    for byte_idx in 0..KEY_SIZE {
        coeffs[0] = key_bytes[byte_idx];

        OsRng.fill_bytes(&mut rng_buf);
        coeffs[1..].copy_from_slice(&rng_buf);

        for share in &mut shares {
            let x = share[0];
            share[byte_idx + 1] = gf256::eval_poly(&coeffs, x);
        }
    }

    coeffs.zeroize();
    rng_buf.zeroize();

    Ok(shares
        .into_iter()
        .map(|data| KeyShare {
            index: data[0],
            data,
        })
        .collect())
}

/// Reconstruct a master key from shares using Lagrange interpolation over GF(256).
///
/// Any subset of shares whose size meets or exceeds the original threshold
/// will recover the correct key. Fewer shares produce a random result.
///
/// # Errors
///
/// Returns `VaultError::ShamirError` if shares are malformed, duplicated,
/// or fewer than 2.
pub fn reconstruct_master_key(shares: &[KeyShare]) -> Result<MasterKey> {
    if shares.len() < 2 {
        return Err(VaultError::ShamirError(
            "need at least 2 shares to reconstruct".to_string(),
        ));
    }

    let expected_len = KEY_SIZE + 1;
    for share in shares {
        if share.data.len() != expected_len {
            return Err(VaultError::ShamirError(format!(
                "invalid share length: {} (expected {expected_len})",
                share.data.len()
            )));
        }
        if share.data[0] == 0 {
            return Err(VaultError::ShamirError(
                "share x-coordinate cannot be zero".to_string(),
            ));
        }
    }

    // Reject duplicate x-coordinates
    let mut xs: Vec<u8> = shares.iter().map(|s| s.data[0]).collect();
    xs.sort_unstable();
    for w in xs.windows(2) {
        if w[0] == w[1] {
            return Err(VaultError::ShamirError(
                "duplicate share x-coordinates".to_string(),
            ));
        }
    }

    let n = shares.len();
    let x_coords: Vec<u8> = shares.iter().map(|s| s.data[0]).collect();

    // Precompute Lagrange basis coefficients evaluated at x=0.
    // L_j(0) = prod_{k!=j} x_k / (x_j XOR x_k)  [in GF(2^8), subtraction = XOR]
    let mut basis = vec![0u8; n];
    for j in 0..n {
        let mut num = 1u8;
        let mut den = 1u8;
        for k in 0..n {
            if k == j {
                continue;
            }
            num = gf256::mul(num, x_coords[k]);
            den = gf256::mul(den, x_coords[j] ^ x_coords[k]);
        }
        basis[j] = gf256::mul(num, gf256::inv(den));
    }

    // Interpolate each byte position independently
    let mut key_bytes = [0u8; KEY_SIZE];
    for (byte_idx, key_byte) in key_bytes.iter_mut().enumerate() {
        let mut value = 0u8;
        for j in 0..n {
            value ^= gf256::mul(shares[j].data[byte_idx + 1], basis[j]);
        }
        *key_byte = value;
    }

    Ok(MasterKey::from_bytes(key_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> MasterKey {
        MasterKey::from_bytes([42u8; KEY_SIZE])
    }

    // -- GF(256) arithmetic tests --

    #[test]
    fn test_gf256_mul_identity() {
        for a in 0..=255u8 {
            assert_eq!(gf256::mul(a, 1), a);
            assert_eq!(gf256::mul(1, a), a);
        }
    }

    #[test]
    fn test_gf256_mul_zero() {
        for a in 0..=255u8 {
            assert_eq!(gf256::mul(a, 0), 0);
            assert_eq!(gf256::mul(0, a), 0);
        }
    }

    #[test]
    fn test_gf256_mul_commutative() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(gf256::mul(a, b), gf256::mul(b, a));
            }
        }
    }

    #[test]
    fn test_gf256_inv_exhaustive() {
        // For every nonzero a, mul(a, inv(a)) must equal 1
        for a in 1..=255u8 {
            let ai = gf256::inv(a);
            assert_ne!(ai, 0, "inverse of {a} should be nonzero");
            assert_eq!(
                gf256::mul(a, ai),
                1,
                "a={a}, inv={ai}, a*inv={}, expected 1",
                gf256::mul(a, ai)
            );
        }
    }

    #[test]
    fn test_gf256_inv_zero() {
        assert_eq!(gf256::inv(0), 0);
    }

    #[test]
    fn test_gf256_eval_poly_constant() {
        // p(x) = 42
        assert_eq!(gf256::eval_poly(&[42], 0), 42);
        assert_eq!(gf256::eval_poly(&[42], 1), 42);
        assert_eq!(gf256::eval_poly(&[42], 255), 42);
    }

    #[test]
    fn test_gf256_eval_poly_linear() {
        // p(x) = 5 + 3*x; p(0) = 5, p(1) = 5^3 = 6
        let coeffs = [5u8, 3u8];
        assert_eq!(gf256::eval_poly(&coeffs, 0), 5);
        assert_eq!(gf256::eval_poly(&coeffs, 1), 5 ^ 3); // GF add = XOR
    }

    // -- Public API tests --

    #[test]
    fn test_split_and_reconstruct_roundtrip() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 5,
            threshold: 3,
        };

        let shares = split_master_key(&key, &config).unwrap();
        assert_eq!(shares.len(), 5);

        let reconstructed = reconstruct_master_key(&shares[..3]).unwrap();
        assert_eq!(reconstructed.as_bytes(), key.as_bytes());
    }

    #[test]
    fn test_reconstruct_with_all_shares() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 5,
            threshold: 3,
        };

        let shares = split_master_key(&key, &config).unwrap();
        let reconstructed = reconstruct_master_key(&shares).unwrap();
        assert_eq!(reconstructed.as_bytes(), key.as_bytes());
    }

    #[test]
    fn test_reconstruct_with_different_subset() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 5,
            threshold: 3,
        };

        let shares = split_master_key(&key, &config).unwrap();

        // Use shares 1, 3, 4 (0-indexed)
        let subset = vec![shares[1].clone(), shares[3].clone(), shares[4].clone()];
        let reconstructed = reconstruct_master_key(&subset).unwrap();
        assert_eq!(reconstructed.as_bytes(), key.as_bytes());
    }

    #[test]
    fn test_insufficient_shares_fails() {
        let result = reconstruct_master_key(&[KeyShare {
            index: 1,
            data: vec![1; KEY_SIZE + 1],
        }]);
        assert!(result.is_err());
        assert!(matches!(result, Err(VaultError::ShamirError(_))));
    }

    #[test]
    fn test_threshold_too_low() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 5,
            threshold: 1,
        };
        assert!(split_master_key(&key, &config).is_err());
    }

    #[test]
    fn test_threshold_exceeds_total() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 3,
            threshold: 5,
        };
        assert!(split_master_key(&key, &config).is_err());
    }

    #[test]
    fn test_total_shares_too_low() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 1,
            threshold: 1,
        };
        assert!(split_master_key(&key, &config).is_err());
    }

    #[test]
    fn test_minimum_config() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 2,
            threshold: 2,
        };

        let shares = split_master_key(&key, &config).unwrap();
        assert_eq!(shares.len(), 2);

        let reconstructed = reconstruct_master_key(&shares).unwrap();
        assert_eq!(reconstructed.as_bytes(), key.as_bytes());
    }

    #[test]
    fn test_shares_have_unique_indices() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 5,
            threshold: 3,
        };

        let shares = split_master_key(&key, &config).unwrap();
        let mut indices: Vec<u8> = shares.iter().map(|s| s.index).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_different_keys_different_shares() {
        let key1 = MasterKey::from_bytes([1u8; KEY_SIZE]);
        let key2 = MasterKey::from_bytes([2u8; KEY_SIZE]);
        let config = ShamirConfig {
            total_shares: 3,
            threshold: 2,
        };

        let shares1 = split_master_key(&key1, &config).unwrap();
        let shares2 = split_master_key(&key2, &config).unwrap();

        // Shares from different keys should not reconstruct to the same key
        let r1 = reconstruct_master_key(&shares1[..2]).unwrap();
        let r2 = reconstruct_master_key(&shares2[..2]).unwrap();
        assert_ne!(r1.as_bytes(), r2.as_bytes());
    }

    #[test]
    fn test_zeroize_on_drop() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 3,
            threshold: 2,
        };

        let shares = split_master_key(&key, &config).unwrap();
        // KeyShare derives ZeroizeOnDrop; verify it compiles and the type is valid
        assert!(!shares[0].data.is_empty());
    }

    #[test]
    fn test_large_threshold() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 10,
            threshold: 10,
        };

        let shares = split_master_key(&key, &config).unwrap();
        assert_eq!(shares.len(), 10);

        let reconstructed = reconstruct_master_key(&shares).unwrap();
        assert_eq!(reconstructed.as_bytes(), key.as_bytes());
    }

    #[test]
    fn test_empty_shares_fails() {
        let result = reconstruct_master_key(&[]);
        assert!(result.is_err());
    }

    // -- New validation tests --

    #[test]
    fn test_duplicate_shares_rejected() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 3,
            threshold: 2,
        };
        let shares = split_master_key(&key, &config).unwrap();
        let dupes = vec![shares[0].clone(), shares[0].clone()];
        match reconstruct_master_key(&dupes) {
            Err(e) => assert!(e.to_string().contains("duplicate")),
            Ok(_) => panic!("expected error for duplicate shares"),
        }
    }

    #[test]
    fn test_wrong_share_length_rejected() {
        let short = vec![
            KeyShare {
                index: 1,
                data: vec![1, 2, 3],
            },
            KeyShare {
                index: 2,
                data: vec![2, 3, 4],
            },
        ];
        match reconstruct_master_key(&short) {
            Err(e) => assert!(e.to_string().contains("invalid share length")),
            Ok(_) => panic!("expected error for wrong length"),
        }
    }

    #[test]
    fn test_zero_x_coordinate_rejected() {
        let bad = vec![
            KeyShare {
                index: 0,
                data: vec![0; KEY_SIZE + 1],
            },
            KeyShare {
                index: 1,
                data: vec![1; KEY_SIZE + 1],
            },
        ];
        match reconstruct_master_key(&bad) {
            Err(e) => assert!(e.to_string().contains("x-coordinate cannot be zero")),
            Ok(_) => panic!("expected error for zero x-coordinate"),
        }
    }

    #[test]
    fn test_share_data_format() {
        let key = test_key();
        let config = ShamirConfig {
            total_shares: 3,
            threshold: 2,
        };

        let shares = split_master_key(&key, &config).unwrap();
        for (i, share) in shares.iter().enumerate() {
            // x-coordinate is 1-based
            assert_eq!(share.index, u8::try_from(i + 1).unwrap());
            assert_eq!(share.data[0], share.index);
            // data = [x, y_0, ..., y_31] = 33 bytes
            assert_eq!(share.data.len(), KEY_SIZE + 1);
        }
    }

    #[test]
    fn test_random_key_roundtrip() {
        // Test with a random key to avoid fixed-pattern coincidences
        let mut key_bytes = [0u8; KEY_SIZE];
        OsRng.fill_bytes(&mut key_bytes);
        let key = MasterKey::from_bytes(key_bytes);

        let config = ShamirConfig {
            total_shares: 5,
            threshold: 3,
        };

        let shares = split_master_key(&key, &config).unwrap();

        // Try multiple subsets
        for combo in &[
            vec![0, 1, 2],
            vec![0, 2, 4],
            vec![1, 3, 4],
            vec![0, 1, 2, 3, 4],
        ] {
            let subset: Vec<_> = combo.iter().map(|&i| shares[i].clone()).collect();
            let r = reconstruct_master_key(&subset).unwrap();
            assert_eq!(r.as_bytes(), key.as_bytes());
        }
    }
}
