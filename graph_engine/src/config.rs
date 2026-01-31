//! Configuration for the graph engine.

/// Configuration for `GraphEngine` runtime behavior.
#[derive(Debug, Clone)]
pub struct GraphEngineConfig {
    /// Default limit for pattern matching results.
    pub default_match_limit: usize,
    /// Threshold for parallel processing of pattern matches.
    pub pattern_parallel_threshold: usize,
    /// Maximum hops for variable-length edge patterns.
    pub max_variable_length_hops: usize,
    /// Default damping factor for `PageRank`.
    pub pagerank_default_damping: f64,
    /// Default convergence tolerance for `PageRank`.
    pub pagerank_default_tolerance: f64,
    /// Default max iterations for `PageRank`.
    pub pagerank_default_max_iterations: usize,
    /// Threshold for parallel centrality computation.
    pub centrality_parallel_threshold: usize,
    /// Maximum passes for community detection.
    pub community_max_passes: usize,
    /// Maximum iterations for label propagation.
    pub label_propagation_max_iterations: usize,
    /// Number of striped locks for index operations.
    pub index_lock_count: usize,
    /// Maximum memory (bytes) for path search before truncation.
    pub max_path_search_memory_bytes: usize,
    /// Threshold for parallel edge operations during node deletion.
    pub parallel_threshold: usize,
    /// Threshold for parallel numeric aggregations.
    pub aggregate_parallel_threshold: usize,
    /// Maximum results returned without pagination (safety limit).
    pub max_unpaginated_results: usize,
}

impl Default for GraphEngineConfig {
    fn default() -> Self {
        Self {
            default_match_limit: 1000,
            pattern_parallel_threshold: 100,
            max_variable_length_hops: 20,
            pagerank_default_damping: 0.85,
            pagerank_default_tolerance: 1e-6,
            pagerank_default_max_iterations: 100,
            centrality_parallel_threshold: 100,
            community_max_passes: 10,
            label_propagation_max_iterations: 100,
            index_lock_count: 64,
            max_path_search_memory_bytes: 100 * 1024 * 1024,
            parallel_threshold: 100,
            aggregate_parallel_threshold: 1000,
            max_unpaginated_results: 100_000,
        }
    }
}

impl GraphEngineConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn default_match_limit(mut self, limit: usize) -> Self {
        self.default_match_limit = limit;
        self
    }

    #[must_use]
    pub const fn pattern_parallel_threshold(mut self, threshold: usize) -> Self {
        self.pattern_parallel_threshold = threshold;
        self
    }

    #[must_use]
    pub const fn max_variable_length_hops(mut self, max_hops: usize) -> Self {
        self.max_variable_length_hops = max_hops;
        self
    }

    #[must_use]
    pub const fn pagerank_default_damping(mut self, damping: f64) -> Self {
        self.pagerank_default_damping = damping;
        self
    }

    #[must_use]
    pub const fn pagerank_default_tolerance(mut self, tolerance: f64) -> Self {
        self.pagerank_default_tolerance = tolerance;
        self
    }

    #[must_use]
    pub const fn pagerank_default_max_iterations(mut self, max_iterations: usize) -> Self {
        self.pagerank_default_max_iterations = max_iterations;
        self
    }

    #[must_use]
    pub const fn centrality_parallel_threshold(mut self, threshold: usize) -> Self {
        self.centrality_parallel_threshold = threshold;
        self
    }

    #[must_use]
    pub const fn community_max_passes(mut self, max_passes: usize) -> Self {
        self.community_max_passes = max_passes;
        self
    }

    #[must_use]
    pub const fn label_propagation_max_iterations(mut self, max_iterations: usize) -> Self {
        self.label_propagation_max_iterations = max_iterations;
        self
    }

    #[must_use]
    pub fn index_lock_count(mut self, count: usize) -> Self {
        self.index_lock_count = count.max(1);
        self
    }

    #[must_use]
    pub const fn max_path_search_memory_bytes(mut self, bytes: usize) -> Self {
        self.max_path_search_memory_bytes = bytes;
        self
    }

    #[must_use]
    pub const fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    #[must_use]
    pub const fn aggregate_parallel_threshold(mut self, threshold: usize) -> Self {
        self.aggregate_parallel_threshold = threshold;
        self
    }

    #[must_use]
    pub const fn max_unpaginated_results(mut self, max: usize) -> Self {
        self.max_unpaginated_results = max;
        self
    }
}
