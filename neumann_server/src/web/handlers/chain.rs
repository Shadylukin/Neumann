// SPDX-License-Identifier: MIT OR Apache-2.0
//! Consensus / chain dashboard handlers.
//!
//! Displays Raft consensus state, distributed transaction statistics,
//! and deadlock detection metrics from a [`ChainStatus`](crate::web::ChainStatus) snapshot.

use std::sync::Arc;

use axum::extract::State;
use maud::{html, Markup};

use crate::web::templates::{layout, m_breadcrumb, m_card, m_empty, m_header, m_stat, m_tabs};
use crate::web::{AdminContext, NavItem};

/// Chain section navigation tabs.
fn chain_tabs(active: &str) -> Markup {
    m_tabs(&[
        ("CONSENSUS", "/chain", active == "consensus"),
        (
            "TRANSACTIONS",
            "/chain/transactions",
            active == "transactions",
        ),
        ("DEADLOCKS", "/chain/deadlocks", active == "deadlocks"),
    ])
}

/// Render a percentage from a 0.0..1.0 rate.
fn pct(rate: f32) -> String {
    #[allow(clippy::cast_possible_truncation)]
    let rounded = (f64::from(rate) * 1000.0).round() / 10.0;
    format!("{rounded:.1}%")
}

/// Raft consensus overview.
pub async fn consensus(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.chain.as_ref().map_or_else(
        || {
            html! {
                (m_header("CONSENSUS", Some("Raft cluster state and health")))
                (chain_tabs("consensus"))
                (m_empty("Chain Not Configured", "No consensus engine is attached to this server"))
            }
        },
        |chain| {
            html! {
                (m_header("CONSENSUS", Some("Raft cluster state and health")))
                (chain_tabs("consensus"))

                // Role + leader
                div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" {
                    (m_stat("STATE", &chain.raft_state, "role", "chain"))
                    (m_stat("TERM", &chain.current_term.to_string(), "current", "chain"))
                    (m_stat("COMMIT", &chain.commit_index.to_string(), "index", "chain"))
                    (m_stat("LOG", &chain.log_length.to_string(), "entries", "chain"))
                }

                div class="grid grid-cols-1 lg:grid-cols-2 gap-6" {
                    // Heartbeat health
                    (m_card("HEARTBEAT HEALTH", html! {
                        div class="grid grid-cols-2 gap-4" {
                            (m_stat("SUCCESS RATE", &pct(chain.heartbeat_success_rate), "heartbeat", "chain"))
                            (m_stat("SUCCESSES", &chain.heartbeat_successes.to_string(), "sent", "chain"))
                            (m_stat("FAILURES", &chain.heartbeat_failures.to_string(), "dropped", "chain"))
                            (m_stat("LEADER", chain.leader_id.as_deref().unwrap_or("unknown"), "current", "chain"))
                        }
                    }))

                    // Quorum + fast-path
                    (m_card("CLUSTER HEALTH", html! {
                        div class="grid grid-cols-2 gap-4" {
                            (m_stat("FAST PATH", &pct(chain.fast_path_rate), "acceptance", "chain"))
                            (m_stat("QUORUM CHECKS", &chain.quorum_checks.to_string(), "performed", "chain"))
                            (m_stat("QUORUM LOST", &chain.quorum_lost_events.to_string(), "events", "chain"))
                            (m_stat("STEP DOWNS", &chain.leader_step_downs.to_string(), "leadership", "chain"))
                        }
                    }))
                }
            }
        },
    );

    layout("Consensus", NavItem::Chain, content)
}

/// Distributed transaction statistics.
pub async fn transactions(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.chain.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/chain", "CHAIN"), ("", "TRANSACTIONS")]))
                (m_header("TRANSACTIONS", Some("Distributed transaction statistics")))
                (chain_tabs("transactions"))
                (m_empty("Chain Not Configured", "No consensus engine is attached to this server"))
            }
        },
        |chain| {
            html! {
                (m_breadcrumb(&[("/chain", "CHAIN"), ("", "TRANSACTIONS")]))
                (m_header("TRANSACTIONS", Some("Distributed transaction statistics")))
                (chain_tabs("transactions"))

                // Hero stats
                div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" {
                    (m_stat("STARTED", &chain.tx_started.to_string(), "total", "chain"))
                    (m_stat("COMMITTED", &chain.tx_committed.to_string(), "successful", "chain"))
                    (m_stat("ABORTED", &chain.tx_aborted.to_string(), "rolled back", "chain"))
                    (m_stat("PENDING", &chain.tx_pending.to_string(), "in-flight", "chain"))
                }

                div class="grid grid-cols-1 lg:grid-cols-2 gap-6" {
                    // Rates
                    (m_card("RATES", html! {
                        div class="grid grid-cols-2 gap-4" {
                            (m_stat("COMMIT RATE", &pct(chain.tx_commit_rate), "committed/started", "chain"))
                            (m_stat("CONFLICT RATE", &pct(chain.tx_conflict_rate), "conflicts/started", "chain"))
                        }
                    }))

                    // Failures
                    (m_card("FAILURE BREAKDOWN", html! {
                        div class="grid grid-cols-2 gap-4" {
                            (m_stat("TIMEOUTS", &chain.tx_timed_out.to_string(), "expired", "chain"))
                            (m_stat("CONFLICTS", &chain.tx_conflicts.to_string(), "detected", "chain"))
                        }
                    }))
                }
            }
        },
    );

    layout("Transactions", NavItem::Chain, content)
}

/// Deadlock detection statistics.
pub async fn deadlocks(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.chain.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/chain", "CHAIN"), ("", "DEADLOCKS")]))
                (m_header("DEADLOCKS", Some("Wait-for graph deadlock detection")))
                (chain_tabs("deadlocks"))
                (m_empty("Chain Not Configured", "No consensus engine is attached to this server"))
            }
        },
        |chain| {
            html! {
                (m_breadcrumb(&[("/chain", "CHAIN"), ("", "DEADLOCKS")]))
                (m_header("DEADLOCKS", Some("Wait-for graph deadlock detection")))
                (chain_tabs("deadlocks"))

                @if chain.deadlock_enabled {
                    div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" {
                        (m_stat("DETECTED", &chain.deadlocks_detected.to_string(), "cycles", "chain"))
                        (m_stat("VICTIMS", &chain.victims_aborted.to_string(), "aborted", "chain"))
                        (m_stat("CYCLES", &chain.detection_cycles.to_string(), "runs", "chain"))
                        (m_stat("MAX LENGTH", &chain.max_cycle_length.to_string(), "nodes", "chain"))
                    }
                } @else {
                    (m_empty("Detection Disabled", "Deadlock detection is not enabled on this node"))
                }
            }
        },
    );

    layout("Deadlocks", NavItem::Chain, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::ChainStatus;
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext::new(
            Arc::new(RelationalEngine::new()),
            Arc::new(VectorEngine::new()),
            Arc::new(GraphEngine::new()),
        ))
    }

    fn sample_chain_status() -> Arc<ChainStatus> {
        Arc::new(ChainStatus {
            raft_state: "Leader".to_string(),
            current_term: 42,
            commit_index: 1500,
            log_length: 1600,
            leader_id: Some("node-1".to_string()),
            fast_path_rate: 0.95,
            heartbeat_success_rate: 0.99,
            heartbeat_successes: 5000,
            heartbeat_failures: 50,
            quorum_checks: 1200,
            quorum_lost_events: 2,
            leader_step_downs: 3,
            tx_started: 10_000,
            tx_committed: 9500,
            tx_aborted: 300,
            tx_timed_out: 100,
            tx_conflicts: 100,
            tx_commit_rate: 0.95,
            tx_conflict_rate: 0.01,
            tx_pending: 5,
            deadlocks_detected: 7,
            victims_aborted: 7,
            detection_cycles: 500,
            max_cycle_length: 4,
            deadlock_enabled: true,
        })
    }

    // === Consensus ===

    #[tokio::test]
    async fn test_consensus_no_chain() {
        let ctx = create_test_context();
        let result = consensus(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("CONSENSUS"));
        assert!(html.contains("Chain Not Configured"));
    }

    #[tokio::test]
    async fn test_consensus_with_chain() {
        let ctx = Arc::new(
            AdminContext::new(
                Arc::new(RelationalEngine::new()),
                Arc::new(VectorEngine::new()),
                Arc::new(GraphEngine::new()),
            )
            .with_chain(Some(sample_chain_status())),
        );
        let result = consensus(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Leader"));
        assert!(html.contains("42"));
        assert!(html.contains("1500"));
        assert!(html.contains("1600"));
        assert!(html.contains("node-1"));
        assert!(html.contains("95.0%"));
        assert!(html.contains("99.0%"));
    }

    // === Transactions ===

    #[tokio::test]
    async fn test_transactions_no_chain() {
        let ctx = create_test_context();
        let result = transactions(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("TRANSACTIONS"));
        assert!(html.contains("Chain Not Configured"));
    }

    #[tokio::test]
    async fn test_transactions_with_chain() {
        let ctx = Arc::new(
            AdminContext::new(
                Arc::new(RelationalEngine::new()),
                Arc::new(VectorEngine::new()),
                Arc::new(GraphEngine::new()),
            )
            .with_chain(Some(sample_chain_status())),
        );
        let result = transactions(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("10000"));
        assert!(html.contains("9500"));
        assert!(html.contains("300"));
        assert!(html.contains("95.0%"));
        assert!(html.contains("1.0%"));
    }

    // === Deadlocks ===

    #[tokio::test]
    async fn test_deadlocks_no_chain() {
        let ctx = create_test_context();
        let result = deadlocks(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("DEADLOCKS"));
        assert!(html.contains("Chain Not Configured"));
    }

    #[tokio::test]
    async fn test_deadlocks_enabled() {
        let ctx = Arc::new(
            AdminContext::new(
                Arc::new(RelationalEngine::new()),
                Arc::new(VectorEngine::new()),
                Arc::new(GraphEngine::new()),
            )
            .with_chain(Some(sample_chain_status())),
        );
        let result = deadlocks(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("7"));
        assert!(html.contains("500"));
        assert!(html.contains("4"));
        assert!(!html.contains("Detection Disabled"));
    }

    #[tokio::test]
    async fn test_deadlocks_disabled() {
        let status = Arc::new(ChainStatus {
            raft_state: "Follower".to_string(),
            current_term: 1,
            commit_index: 0,
            log_length: 0,
            leader_id: None,
            fast_path_rate: 0.0,
            heartbeat_success_rate: 0.0,
            heartbeat_successes: 0,
            heartbeat_failures: 0,
            quorum_checks: 0,
            quorum_lost_events: 0,
            leader_step_downs: 0,
            tx_started: 0,
            tx_committed: 0,
            tx_aborted: 0,
            tx_timed_out: 0,
            tx_conflicts: 0,
            tx_commit_rate: 0.0,
            tx_conflict_rate: 0.0,
            tx_pending: 0,
            deadlocks_detected: 0,
            victims_aborted: 0,
            detection_cycles: 0,
            max_cycle_length: 0,
            deadlock_enabled: false,
        });
        let ctx = Arc::new(
            AdminContext::new(
                Arc::new(RelationalEngine::new()),
                Arc::new(VectorEngine::new()),
                Arc::new(GraphEngine::new()),
            )
            .with_chain(Some(status)),
        );
        let result = deadlocks(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Detection Disabled"));
    }

    #[test]
    fn test_pct_formatting() {
        assert_eq!(pct(0.0), "0.0%");
        assert_eq!(pct(1.0), "100.0%");
        assert_eq!(pct(0.5), "50.0%");
        assert_eq!(pct(0.999), "99.9%");
    }

    #[tokio::test]
    async fn test_consensus_no_leader() {
        let status = Arc::new(ChainStatus {
            raft_state: "Candidate".to_string(),
            current_term: 5,
            commit_index: 100,
            log_length: 110,
            leader_id: None,
            fast_path_rate: 0.0,
            heartbeat_success_rate: 0.0,
            heartbeat_successes: 0,
            heartbeat_failures: 10,
            quorum_checks: 50,
            quorum_lost_events: 5,
            leader_step_downs: 1,
            tx_started: 0,
            tx_committed: 0,
            tx_aborted: 0,
            tx_timed_out: 0,
            tx_conflicts: 0,
            tx_commit_rate: 0.0,
            tx_conflict_rate: 0.0,
            tx_pending: 0,
            deadlocks_detected: 0,
            victims_aborted: 0,
            detection_cycles: 0,
            max_cycle_length: 0,
            deadlock_enabled: false,
        });
        let ctx = Arc::new(
            AdminContext::new(
                Arc::new(RelationalEngine::new()),
                Arc::new(VectorEngine::new()),
                Arc::new(GraphEngine::new()),
            )
            .with_chain(Some(status)),
        );
        let result = consensus(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Candidate"));
        assert!(html.contains("unknown"));
    }
}
