//! Byzantine Fault Tolerance Experiments
//!
//! This crate provides rigorous experimental verification of two key BFT theorems:
//!
//! 1. **Detection Completeness**: Rank 2 density > 0 iff Byzantine equivocation exists.
//!    No threshold is needed - the presence of ANY inconsistency proves Byzantine behavior.
//!
//! 2. **Attribution Threshold**: Identifying WHO is Byzantine requires n >= 3f+1.
//!    This is a tight bound - below it, an optimal adversary can defeat all algorithms.
//!
//! # Key Insight
//!
//! Detection is **structural** (geometric) - it works at any network size because
//! equivocation creates observable inconsistencies.
//!
//! Attribution is **counting** (arithmetic) - it requires enough honest nodes to
//! outvote the Byzantine coalition (3f+1).
//!
//! The 3f+1 bound is about WHO, not THAT.
//!
//! # Report-Based BFT (The Hard Problem)
//!
//! The `report_*` modules model the REAL BFT problem:
//! - Nodes only see messages sent TO them (no global visibility)
//! - Nodes must REPORT what they received (claims, not ground truth)
//! - Byzantine nodes can LIE about reports
//! - Attribution becomes ambiguous: "Did sender equivocate, OR is reporter lying?"
//!
//! This is why 3f+1 emerges: honest nodes must outvote lies in reports.

pub mod adversary;
pub mod attribution;
pub mod network;
pub mod proofs;
pub mod rank2;
pub mod report;
pub mod report_adversary;
pub mod report_detection;
pub mod report_network;
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
