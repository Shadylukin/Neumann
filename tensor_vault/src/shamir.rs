// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shamir secret sharing for master key splitting and reconstruction.

use serde::{Deserialize, Serialize};
use sharks::{Share, Sharks};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{
    key::{MasterKey, KEY_SIZE},
    Result, VaultError,
};

/// Configuration for Shamir secret sharing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShamirConfig {
    /// Total number of shares to generate.
    pub total_shares: u8,
    /// Minimum shares needed for reconstruction.
    pub threshold: u8,
}

impl ShamirConfig {
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
    /// Share index (1-based).
    pub index: u8,
    /// Raw share data.
    pub data: Vec<u8>,
}

/// Split a master key into shares using Shamir secret sharing.
pub fn split_master_key(key: &MasterKey, config: &ShamirConfig) -> Result<Vec<KeyShare>> {
    config.validate()?;

    let engine = Sharks(config.threshold);
    let dealer = engine.dealer(key.as_bytes());

    let result = dealer
        .take(config.total_shares as usize)
        .map(|share| {
            let bytes: Vec<u8> = (&share).into();
            KeyShare {
                index: bytes[0],
                data: bytes,
            }
        })
        .collect();

    Ok(result)
}

/// Reconstruct a master key from shares.
pub fn reconstruct_master_key(shares: &[KeyShare]) -> Result<MasterKey> {
    if shares.len() < 2 {
        return Err(VaultError::ShamirError(
            "need at least 2 shares to reconstruct".to_string(),
        ));
    }

    let share_refs: Vec<Share> = shares
        .iter()
        .map(|ks| {
            Share::try_from(ks.data.as_slice())
                .map_err(|e| VaultError::ShamirError(format!("invalid share data: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let engine = Sharks(shares.len() as u8);
    let secret = engine
        .recover(&share_refs)
        .map_err(|e| VaultError::ShamirError(format!("reconstruction failed: {e}")))?;

    if secret.len() != KEY_SIZE {
        return Err(VaultError::ShamirError(format!(
            "reconstructed key has wrong size: {} (expected {KEY_SIZE})",
            secret.len()
        )));
    }

    let mut key_bytes = [0u8; KEY_SIZE];
    key_bytes.copy_from_slice(&secret);
    Ok(MasterKey::from_bytes(key_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> MasterKey {
        MasterKey::from_bytes([42u8; KEY_SIZE])
    }

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
            data: vec![1, 2, 3],
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
}
