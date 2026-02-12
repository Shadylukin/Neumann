// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Distance-based permission attenuation for graph-based access control.
//!
//! Permissions degrade with graph distance (hop count):
//! - Direct (1 hop): Admin preserved
//! - 2 hops: Admin attenuated to Write
//! - 3+ hops: Write attenuated to Read
//!
//! The policy is configurable via `AttenuationPolicy`.

use serde::{Deserialize, Serialize};

use crate::Permission;

/// Smooth exponential decay attenuation policy.
///
/// Permission strength decays as `exp(-lambda * hops)`. Thresholds map
/// the continuous strength to discrete permission levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialAttenuationPolicy {
    /// Decay rate lambda (default: 0.5). Higher = faster decay.
    pub decay_rate: f64,
    /// Minimum strength to preserve Admin permission (default: 0.7).
    pub admin_threshold: f64,
    /// Minimum strength to preserve Write permission (default: 0.3).
    pub write_threshold: f64,
    /// Minimum strength to preserve Read permission (default: 0.05).
    pub read_threshold: f64,
    /// Hard BFS depth cap (default: 20).
    pub max_depth: usize,
}

impl Default for ExponentialAttenuationPolicy {
    fn default() -> Self {
        Self {
            decay_rate: 0.5,
            admin_threshold: 0.7,
            write_threshold: 0.3,
            read_threshold: 0.05,
            max_depth: 20,
        }
    }
}

impl ExponentialAttenuationPolicy {
    /// Compute the strength at a given hop count: `exp(-decay_rate * hops)`.
    pub fn strength(&self, hops: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)] // hop count will never exceed 2^52
        let h = hops as f64;
        (-self.decay_rate * h).exp()
    }

    /// Attenuate a permission based on exponential decay.
    ///
    /// Returns `None` if the hop count exceeds `max_depth` or the strength
    /// drops below the read threshold.
    pub fn attenuate(&self, perm: Permission, hops: usize) -> Option<Permission> {
        if hops > self.max_depth {
            return None;
        }

        let s = self.strength(hops);

        match perm {
            Permission::Admin => {
                if s >= self.admin_threshold {
                    Some(Permission::Admin)
                } else if s >= self.write_threshold {
                    Some(Permission::Write)
                } else if s >= self.read_threshold {
                    Some(Permission::Read)
                } else {
                    None
                }
            },
            Permission::Write => {
                if s >= self.write_threshold {
                    Some(Permission::Write)
                } else if s >= self.read_threshold {
                    Some(Permission::Read)
                } else {
                    None
                }
            },
            Permission::Read => {
                if s >= self.read_threshold {
                    Some(Permission::Read)
                } else {
                    None
                }
            },
        }
    }
}

/// Policy controlling how permissions attenuate with graph distance.
///
/// Each limit is the maximum hop count at which the named permission
/// level is preserved. Beyond that limit the permission degrades.
#[derive(Debug, Clone)]
pub struct AttenuationPolicy {
    /// Hops at which Admin is still preserved (default: 1).
    pub admin_limit: usize,
    /// Hops at which Write is still preserved (default: 2).
    pub write_limit: usize,
    /// Maximum traversal depth; also the BFS cutoff (default: 10).
    pub horizon: usize,
}

impl Default for AttenuationPolicy {
    fn default() -> Self {
        Self {
            admin_limit: 1,
            write_limit: 2,
            horizon: 10,
        }
    }
}

impl AttenuationPolicy {
    /// No attenuation (legacy behavior). Admin at any depth.
    pub fn none() -> Self {
        Self {
            admin_limit: usize::MAX,
            write_limit: usize::MAX,
            horizon: usize::MAX,
        }
    }

    /// Attenuate a permission based on the number of hops traversed.
    ///
    /// Returns `None` if the hop count exceeds `horizon`.
    pub fn attenuate(&self, perm: Permission, hops: usize) -> Option<Permission> {
        if hops > self.horizon {
            return None;
        }

        let attenuated = match perm {
            Permission::Admin => {
                if hops <= self.admin_limit {
                    Permission::Admin
                } else if hops <= self.write_limit {
                    Permission::Write
                } else {
                    Permission::Read
                }
            },
            Permission::Write => {
                if hops <= self.write_limit {
                    Permission::Write
                } else {
                    Permission::Read
                }
            },
            Permission::Read => Permission::Read,
        };

        Some(attenuated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy_direct_admin_preserved() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Admin, 1),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_default_policy_2hop_admin_attenuated_to_write() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Admin, 2),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_default_policy_3hop_admin_attenuated_to_read() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Admin, 3),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_default_policy_beyond_max_hops_denied() {
        let policy = AttenuationPolicy::default();
        assert_eq!(policy.attenuate(Permission::Admin, 11), None);
        assert_eq!(policy.attenuate(Permission::Write, 11), None);
        assert_eq!(policy.attenuate(Permission::Read, 11), None);
    }

    #[test]
    fn test_default_policy_write_preserved_at_2_hops() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Write, 1),
            Some(Permission::Write)
        );
        assert_eq!(
            policy.attenuate(Permission::Write, 2),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_default_policy_write_attenuated_to_read_at_3_hops() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Write, 3),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_default_policy_read_always_read() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Read, 1),
            Some(Permission::Read)
        );
        assert_eq!(
            policy.attenuate(Permission::Read, 5),
            Some(Permission::Read)
        );
        assert_eq!(
            policy.attenuate(Permission::Read, 10),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_no_attenuation_policy() {
        let policy = AttenuationPolicy::none();
        assert_eq!(
            policy.attenuate(Permission::Admin, 100),
            Some(Permission::Admin)
        );
        assert_eq!(
            policy.attenuate(Permission::Write, 100),
            Some(Permission::Write)
        );
        assert_eq!(
            policy.attenuate(Permission::Read, 100),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_custom_policy() {
        let policy = AttenuationPolicy {
            admin_limit: 2,
            write_limit: 4,
            horizon: 6,
        };
        assert_eq!(
            policy.attenuate(Permission::Admin, 2),
            Some(Permission::Admin)
        );
        assert_eq!(
            policy.attenuate(Permission::Admin, 3),
            Some(Permission::Write)
        );
        assert_eq!(
            policy.attenuate(Permission::Admin, 5),
            Some(Permission::Read)
        );
        assert_eq!(policy.attenuate(Permission::Admin, 7), None);
    }

    #[test]
    fn test_zero_hops() {
        let policy = AttenuationPolicy::default();
        assert_eq!(
            policy.attenuate(Permission::Admin, 0),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_boundary_at_horizon() {
        let policy = AttenuationPolicy::default();
        // horizon = 10
        assert_eq!(
            policy.attenuate(Permission::Read, 10),
            Some(Permission::Read)
        );
        assert_eq!(policy.attenuate(Permission::Read, 11), None);
    }

    // ===== Exponential Attenuation Tests =====

    #[test]
    fn test_exp_strength_zero_hops() {
        let policy = ExponentialAttenuationPolicy::default();
        let s = policy.strength(0);
        assert!(
            (s - 1.0).abs() < f64::EPSILON,
            "strength at 0 hops should be 1.0"
        );
    }

    #[test]
    fn test_exp_strength_decays() {
        let policy = ExponentialAttenuationPolicy::default();
        let s0 = policy.strength(0);
        let s1 = policy.strength(1);
        let s2 = policy.strength(2);
        assert!(s0 > s1, "strength should decrease with hops");
        assert!(s1 > s2, "strength should decrease with hops");
        assert!(s2 > 0.0, "strength should remain positive");
    }

    #[test]
    fn test_exp_admin_preserved() {
        let policy = ExponentialAttenuationPolicy::default();
        // At 0 hops, strength=1.0 > 0.7 threshold
        assert_eq!(
            policy.attenuate(Permission::Admin, 0),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_exp_admin_to_write() {
        let policy = ExponentialAttenuationPolicy::default();
        // decay_rate=0.5, strength(1)=exp(-0.5)~0.607 < 0.7, > 0.3
        assert_eq!(
            policy.attenuate(Permission::Admin, 1),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_exp_admin_to_read() {
        let policy = ExponentialAttenuationPolicy::default();
        // strength(2)=exp(-1.0)~0.368 > 0.3 -> Write
        // strength(3)=exp(-1.5)~0.223 < 0.3, > 0.05 -> Read
        assert_eq!(
            policy.attenuate(Permission::Admin, 3),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_exp_denied_below_threshold() {
        let policy = ExponentialAttenuationPolicy::default();
        // strength(6)=exp(-3.0)~0.0498 < 0.05 threshold -> denied
        assert_eq!(policy.attenuate(Permission::Admin, 6), None);
        assert_eq!(policy.attenuate(Permission::Read, 6), None);
    }

    #[test]
    fn test_exp_max_depth() {
        let policy = ExponentialAttenuationPolicy::default();
        assert_eq!(policy.attenuate(Permission::Admin, 21), None);
        assert_eq!(policy.attenuate(Permission::Read, 21), None);
    }

    #[test]
    fn test_exp_custom_decay_rate() {
        let policy = ExponentialAttenuationPolicy {
            decay_rate: 0.1,
            admin_threshold: 0.7,
            write_threshold: 0.3,
            read_threshold: 0.05,
            max_depth: 50,
        };
        // With slow decay, admin should persist further
        // strength(3)=exp(-0.3)~0.741 > 0.7
        assert_eq!(
            policy.attenuate(Permission::Admin, 3),
            Some(Permission::Admin)
        );
        // strength(10)=exp(-1.0)~0.368 > 0.3
        assert_eq!(
            policy.attenuate(Permission::Admin, 10),
            Some(Permission::Write)
        );
    }
}
