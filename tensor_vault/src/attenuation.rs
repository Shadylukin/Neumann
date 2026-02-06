// SPDX-License-Identifier: MIT OR Apache-2.0
//! Distance-based permission attenuation for graph-based access control.
//!
//! Permissions degrade with graph distance (hop count):
//! - Direct (1 hop): Admin preserved
//! - 2 hops: Admin attenuated to Write
//! - 3+ hops: Write attenuated to Read
//!
//! The policy is configurable via `AttenuationPolicy`.

use crate::Permission;

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
}
