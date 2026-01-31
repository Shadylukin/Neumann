// SPDX-License-Identifier: MIT OR Apache-2.0
//! Byzantine Fault Tolerance and Geometry Experiments
//!
//! This crate provides:
//!
//! 1. **BFT Experiments**: Verification of detection completeness and attribution thresholds
//! 2. **Geometry Benchmarks**: Hyperbolic vs Euclidean HNSW comparison
//! 3. **Lock Optimization Prototypes**: SingleLockEntityIndex for benchmarking
//!
//! # BFT Key Insight
//!
//! Detection is **structural** (geometric) - it works at any network size because
//! equivocation creates observable inconsistencies.
//!
//! Attribution is **counting** (arithmetic) - it requires enough honest nodes to
//! outvote the Byzantine coalition (3f+1).
//!
//! The 3f+1 bound is about WHO, not THAT.
//!
//! # Geometry Key Insight
//!
//! Hyperbolic space has exponential volume growth (V(r) ~ e^(d*r)) vs polynomial
//! for Euclidean (V(r) ~ r^d). This means:
//! - Trees embed with zero distortion in hyperbolic space
//! - 5D hyperbolic can match 200D Euclidean for hierarchical data
//!
//! # Benchmark Modules
//!
//! - `bench_utils`: Data generation for benchmarks (trees, clusters, random)
//! - `single_lock_entity_index`: Lock optimization prototype

pub mod adversary;
pub mod attribution;
pub mod bench_utils;
pub mod curvature_byzantine;
pub mod detection_benchmark;
pub mod hyperbolic;
pub mod hyperbolic_consensus;
pub mod hyperbolic_hash;
pub mod hyperbolic_hnsw;
pub mod network;
pub mod proofs;
pub mod rank2;
pub mod report;
pub mod report_adversary;
pub mod report_detection;
pub mod report_network;
pub mod single_lock_entity_index;
pub mod tensor_attribution;
pub mod types;

pub use adversary::{
    AdaptiveAdversary, AdversaryStrategy, CoordinatedFraming, OptimalBelow3f1, RandomEquivocation,
    SplitEquivocation, SubtleEquivocation,
};
pub use attribution::{
    AttributionAlgorithm, EigenvalueTrust, GraphVertexCover, IterativeElimination, MajorityVoting,
    OptimalAbove3f1,
};
pub use network::NetworkModel;
pub use proofs::{
    prove_attribution_threshold, prove_detection_complete, prove_detection_completeness,
    prove_detection_soundness, smoking_gun, test_attribution_fails_below_threshold,
    test_attribution_in_direct_observation, test_attribution_works_at_threshold,
};
pub use rank2::{
    identify_inconsistent_senders, measure_rank2, measure_rank2_by_sender, Rank2Measurement,
};
pub use types::{
    AttributionResult, Counterexample, NetworkMessage, NodeId, ProofResult, SmokingGunResult,
    ThresholdTest,
};

// Report-based BFT exports
pub use report::{DeliveredMessage, InconsistencyEvidence, Report, ReportLie};
pub use report_adversary::{
    FrameHonestSender, MaximalConfusion, OptimalReportBelow3f1, ReportLyingStrategy, SplitReports,
    TruthfulReporting,
};
pub use report_detection::{
    attribution_correct, check_framing, AttributionDetector, DetectionOutput,
    MajorityReportDetector, QuorumReportDetector, ReportDetector,
};
pub use report_network::ReportBasedNetwork;

// Tensor attribution (the real theorem)
pub use tensor_attribution::{EvidenceTensor, NodeEvidence, TensorSimulation};

// Hyperbolic geometry experiments
pub use curvature_byzantine::{
    CurvatureByzantineDetector, CurvatureSimulation, HybridByzantineDetector, NodeCurvature,
};
pub use hyperbolic::{AdaptiveGeometry, CurvatureEstimator, LorentzPoint, MixedCurvatureSpace};
pub use hyperbolic_consensus::{
    AdaptiveConflictDetector, ConflictDelta, EuclideanConflictDetector, HyperbolicConflictDetector,
};
pub use hyperbolic_hnsw::{
    run_hnsw_benchmark_suite, EuclideanHNSW, HNSWBenchmarkResults, HyperbolicHNSW,
    HyperbolicHNSWConfig, LorentzVector,
};
