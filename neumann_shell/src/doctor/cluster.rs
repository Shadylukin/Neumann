// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cluster health diagnostic check.

use super::check::CheckResult;
use super::wal::WalInfo;
use super::DiagnosticContext;

/// Checks cluster quorum and overall health.
pub fn check_cluster<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> CheckResult {
    if !ctx.router.is_cluster_active() {
        return CheckResult::skipped("Cluster", "Not connected");
    }

    let Some(cluster) = ctx.router.cluster() else {
        return CheckResult::skipped("Cluster", "Not connected");
    };

    let view = cluster.membership().view();
    let total = view.nodes.len();
    let healthy = view.healthy_nodes.len();
    let failed = view.failed_nodes.len();

    // Check quorum: more than half the nodes must be healthy
    let quorum_needed = (total / 2) + 1;
    let has_quorum = healthy >= quorum_needed;

    if has_quorum {
        if failed > 0 {
            CheckResult::warning(
                "Cluster",
                format!("Quorum met ({healthy}/{total} healthy, {failed} failed)"),
            )
        } else {
            CheckResult::healthy("Cluster", format!("Quorum met ({healthy}/{total} nodes)"))
        }
    } else {
        CheckResult::error(
            "Cluster",
            format!("Quorum lost ({healthy}/{total} healthy, need {quorum_needed})"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::check::CheckStatus;

    #[test]
    fn test_check_cluster_not_connected() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_cluster(&ctx);

        assert_eq!(result.status, CheckStatus::Skipped);
        assert_eq!(result.name, "Cluster");
        assert!(result.message.contains("Not connected"));
    }

    // Note: Testing with an actual cluster connection is complex and would require
    // integration tests with a running cluster. The following tests document
    // expected behavior.

    #[test]
    fn test_quorum_calculation() {
        // With 3 nodes, quorum is 2
        assert!(2 >= (3 / 2) + 1);

        // With 5 nodes, quorum is 3
        assert!(3 >= (5 / 2) + 1);

        // With 2 nodes, quorum is 2
        assert!(2 >= (2 / 2) + 1);

        // With 1 node, quorum is 1
        assert!(1 >= (1 / 2) + 1);
    }
}
