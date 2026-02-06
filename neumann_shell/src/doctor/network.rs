// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Network latency diagnostic check.

use super::check::CheckResult;
use super::wal::WalInfo;
use super::DiagnosticContext;

/// Checks network latency to cluster nodes.
pub fn check_network<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> CheckResult {
    if !ctx.router.is_cluster_active() {
        return CheckResult::skipped("Network", "Cluster not connected");
    }

    let Some(cluster) = ctx.router.cluster() else {
        return CheckResult::skipped("Network", "Cluster not connected");
    };

    let view = cluster.membership().view();
    let threshold = ctx.latency_threshold_ms;

    let mut slow_nodes = Vec::new();
    let mut max_rtt: Option<u64> = None;

    for node in &view.nodes {
        if let Some(rtt) = node.rtt_ms {
            if rtt > threshold {
                slow_nodes.push((node.node_id.clone(), rtt));
            }
            max_rtt = Some(max_rtt.map_or(rtt, |m| m.max(rtt)));
        }
    }

    if slow_nodes.is_empty() {
        max_rtt.map_or_else(
            || CheckResult::healthy("Network", "All nodes responsive"),
            |rtt| CheckResult::healthy("Network", format!("All nodes responsive (max {rtt}ms)")),
        )
    } else {
        let details = slow_nodes
            .iter()
            .map(|(id, rtt)| format!("{id}: {rtt}ms"))
            .collect::<Vec<_>>()
            .join(", ");

        CheckResult::warning(
            "Network",
            format!(
                "{} node(s) slow (>{threshold}ms threshold)",
                slow_nodes.len()
            ),
        )
        .with_details(details)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::check::CheckStatus;

    #[test]
    fn test_check_network_not_connected() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_network(&ctx);

        assert_eq!(result.status, CheckStatus::Skipped);
        assert_eq!(result.name, "Network");
        assert!(result.message.contains("not connected"));
    }

    #[test]
    fn test_default_latency_threshold() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        assert_eq!(ctx.latency_threshold_ms, 50);
    }

    #[test]
    fn test_custom_latency_threshold() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx =
            DiagnosticContext::<crate::wal::Wal>::new(&router, None).with_latency_threshold(100);
        assert_eq!(ctx.latency_threshold_ms, 100);
    }
}
