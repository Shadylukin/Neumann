// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared helper for dispatching typed `QueryRouter` calls.
//!
//! Used by REST handlers and gRPC services (`PointsService`,
//! `CollectionsService`) so neither has to duplicate the identity-aware
//! lock pattern that `service::QueryServiceImpl::execute_query` uses, and so
//! the modules don't depend on each other.

use std::sync::Arc;

use parking_lot::RwLock;
use query_router::{QueryRouter, RouterError};

/// Execute `f` against the router with optional identity scoped to the call.
///
/// - When `identity` is `Some`, the router is taken under a write lock so that
///   `set_identity` / `clear_identity` are applied atomically around `f`. This
///   matches `service::QueryServiceImpl::execute_query`.
/// - When `identity` is `None`, the router is taken under a read lock and `f`
///   runs directly.
///
/// `f` returns a [`Result`] whose error type is [`RouterError`]; callers map
/// the error to their surface representation (`ApiError` for REST,
/// `tonic::Status` for gRPC) at their call site.
///
/// # Errors
///
/// Returns the error produced by `f` unchanged. The lock and the optional
/// identity scope are always released before this function returns.
pub fn dispatch_with_identity<R>(
    router: &Arc<RwLock<QueryRouter>>,
    identity: Option<&str>,
    f: impl FnOnce(&QueryRouter) -> Result<R, RouterError>,
) -> Result<R, RouterError> {
    if let Some(id) = identity {
        let mut r = router.write();
        r.set_identity(id);
        let out = f(&r);
        r.clear_identity();
        out
    } else {
        f(&router.read())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_anonymous_does_not_set_identity() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let result: Result<(), RouterError> = dispatch_with_identity(&router, None, |r| {
            assert!(r.current_identity().is_none());
            Ok(())
        });
        assert!(result.is_ok());
        assert!(router.read().current_identity().is_none());
    }

    #[test]
    fn dispatch_with_identity_sets_and_clears() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let captured: Result<String, RouterError> =
            dispatch_with_identity(&router, Some("alice"), |r| {
                Ok(r.current_identity()
                    .map(ToString::to_string)
                    .unwrap_or_default())
            });
        assert_eq!(captured.unwrap(), "alice");
        assert!(
            router.read().current_identity().is_none(),
            "identity must be cleared after dispatch"
        );
    }

    #[test]
    fn dispatch_with_identity_propagates_errors() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let result: Result<(), RouterError> = dispatch_with_identity(&router, Some("bob"), |_| {
            Err(RouterError::AuthenticationRequired)
        });
        assert!(matches!(result, Err(RouterError::AuthenticationRequired)));
        assert!(router.read().current_identity().is_none());
    }
}
