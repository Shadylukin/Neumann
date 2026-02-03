// SPDX-License-Identifier: MIT OR Apache-2.0
//! HNSW index validation check.

use super::check::CheckResult;
use super::wal::WalInfo;
use super::DiagnosticContext;

/// Checks HNSW index health by counting indexed vectors.
pub fn check_hnsw<W: WalInfo>(ctx: &DiagnosticContext<'_, W>) -> CheckResult {
    // Use VectorEngine::count() for accurate embedding count
    let count = ctx.router.vector().count();

    if count == 0 {
        return CheckResult::skipped("HNSW", "No vectors stored");
    }

    let count_str = format_count(count);
    let has_index = ctx.router.has_hnsw_index();

    if has_index {
        CheckResult::healthy("HNSW", format!("1 index, {count_str} vectors"))
    } else {
        CheckResult::healthy("HNSW", format!("{count_str} vectors (no index built)"))
    }
}

/// Formats a count with appropriate suffix (K, M).
#[allow(clippy::cast_precision_loss)]
fn format_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::check::CheckStatus;

    #[test]
    fn test_format_count_small() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(1), "1");
        assert_eq!(format_count(999), "999");
    }

    #[test]
    fn test_format_count_thousands() {
        assert_eq!(format_count(1000), "1.0K");
        assert_eq!(format_count(1500), "1.5K");
        assert_eq!(format_count(10000), "10.0K");
        assert_eq!(format_count(999999), "1000.0K");
    }

    #[test]
    fn test_format_count_millions() {
        assert_eq!(format_count(1_000_000), "1.0M");
        assert_eq!(format_count(1_200_000), "1.2M");
        assert_eq!(format_count(10_000_000), "10.0M");
    }

    #[test]
    fn test_check_hnsw_no_vectors() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();
        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_hnsw(&ctx);

        assert_eq!(result.status, CheckStatus::Skipped);
        assert_eq!(result.name, "HNSW");
        assert!(result.message.contains("No vectors"));
    }

    #[test]
    fn test_check_hnsw_with_vectors() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();

        // Store some embeddings using VectorEngine API
        for i in 0..5 {
            let key = format!("test_{i}");
            router
                .vector()
                .store_embedding(&key, vec![1.0, 2.0, 3.0])
                .unwrap();
        }

        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_hnsw(&ctx);

        assert_eq!(result.status, CheckStatus::Healthy);
        assert_eq!(result.name, "HNSW");
        // Should say "no index built" since we haven't built an HNSW index
        assert!(result.message.contains("5 vectors"));
        assert!(result.message.contains("no index built"));
    }

    #[test]
    fn test_check_hnsw_large_count() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();

        // Store many embeddings using VectorEngine API
        for i in 0..1500 {
            let key = format!("bulk_{i}");
            router.vector().store_embedding(&key, vec![1.0]).unwrap();
        }

        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_hnsw(&ctx);

        assert_eq!(result.status, CheckStatus::Healthy);
        assert!(result.message.contains("1.5K"));
    }

    #[test]
    fn test_check_hnsw_uses_vector_count() {
        use query_router::QueryRouter;

        let router = QueryRouter::new();

        // Verify that count() returns 0 initially
        assert_eq!(router.vector().count(), 0);

        // Store embeddings
        router
            .vector()
            .store_embedding("test1", vec![1.0, 2.0])
            .unwrap();
        router
            .vector()
            .store_embedding("test2", vec![3.0, 4.0])
            .unwrap();

        // Verify count is updated
        assert_eq!(router.vector().count(), 2);

        let ctx = DiagnosticContext::<crate::wal::Wal>::new(&router, None);
        let result = check_hnsw(&ctx);

        assert_eq!(result.status, CheckStatus::Healthy);
        assert!(result.message.contains("2 vectors"));
    }
}
