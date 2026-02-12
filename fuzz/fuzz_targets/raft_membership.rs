// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use tensor_chain::{ConfigChange, JointConfig, RaftMembershipConfig};

#[derive(Arbitrary, Debug)]
struct RaftMembershipInput {
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    /// Test RaftMembershipConfig serialization roundtrip
    ConfigRoundtrip {
        voters: Vec<String>,
        learners: Vec<String>,
        config_index: u64,
        has_joint: bool,
        joint_old: Vec<String>,
        joint_new: Vec<String>,
    },

    /// Test JointConfig quorum calculation
    JointQuorum {
        old_voters: Vec<String>,
        new_voters: Vec<String>,
        votes: Vec<String>,
    },

    /// Test ConfigChange serialization
    ConfigChangeRoundtrip { change_type: ConfigChangeType },

    /// Test membership operations sequence
    MembershipOps { operations: Vec<MembershipOp> },
}

#[derive(Arbitrary, Debug)]
enum ConfigChangeType {
    AddLearner {
        node_id: String,
    },
    PromoteLearner {
        node_id: String,
    },
    RemoveNode {
        node_id: String,
    },
    JointChange {
        additions: Vec<String>,
        removals: Vec<String>,
    },
}

#[derive(Arbitrary, Debug)]
enum MembershipOp {
    AddLearner(String),
    PromoteLearner(String),
    RemoveNode(String),
}

fn truncate_strings(strings: Vec<String>, max_len: usize, max_items: usize) -> Vec<String> {
    strings
        .into_iter()
        .take(max_items)
        .map(|s| s.chars().take(max_len).collect())
        .collect()
}

fuzz_target!(|input: RaftMembershipInput| {
    match input.test_case {
        TestCase::ConfigRoundtrip {
            voters,
            learners,
            config_index,
            has_joint,
            joint_old,
            joint_new,
        } => {
            // Limit sizes
            let voters = truncate_strings(voters, 32, 10);
            let learners = truncate_strings(learners, 32, 10);

            let mut config = RaftMembershipConfig::new(voters.clone());
            for learner in learners {
                config.add_learner(learner);
            }
            config.config_index = config_index;

            if has_joint {
                let old = truncate_strings(joint_old, 32, 10);
                let new = truncate_strings(joint_new, 32, 10);
                config.joint = Some(JointConfig {
                    old_voters: old,
                    new_voters: new,
                });
            }

            // Bincode roundtrip
            if let Ok(bytes) = bitcode::serialize(&config) {
                let decoded: Result<RaftMembershipConfig, _> = bitcode::deserialize(&bytes);
                assert!(
                    decoded.is_ok(),
                    "Failed to deserialize RaftMembershipConfig"
                );

                let decoded = decoded.unwrap();
                assert_eq!(decoded.voters, config.voters);
                assert_eq!(decoded.learners, config.learners);
                assert_eq!(decoded.config_index, config.config_index);
            }
        },

        TestCase::JointQuorum {
            old_voters,
            new_voters,
            votes,
        } => {
            let old = truncate_strings(old_voters, 32, 10);
            let new = truncate_strings(new_voters, 32, 10);
            let votes = truncate_strings(votes, 32, 10);

            if old.is_empty() || new.is_empty() {
                return;
            }

            let joint = JointConfig {
                old_voters: old.clone(),
                new_voters: new.clone(),
            };

            let vote_set: HashSet<String> = votes.into_iter().collect();

            // Calculate expected quorum
            let old_quorum = (old.len() / 2) + 1;
            let new_quorum = (new.len() / 2) + 1;
            let old_votes = old.iter().filter(|n| vote_set.contains(*n)).count();
            let new_votes = new.iter().filter(|n| vote_set.contains(*n)).count();
            let expected = old_votes >= old_quorum && new_votes >= new_quorum;

            // Verify joint quorum calculation
            assert_eq!(joint.has_joint_quorum(&vote_set), expected);
        },

        TestCase::ConfigChangeRoundtrip { change_type } => {
            let change = match change_type {
                ConfigChangeType::AddLearner { node_id } => ConfigChange::AddLearner {
                    node_id: node_id.chars().take(32).collect(),
                },
                ConfigChangeType::PromoteLearner { node_id } => ConfigChange::PromoteLearner {
                    node_id: node_id.chars().take(32).collect(),
                },
                ConfigChangeType::RemoveNode { node_id } => ConfigChange::RemoveNode {
                    node_id: node_id.chars().take(32).collect(),
                },
                ConfigChangeType::JointChange {
                    additions,
                    removals,
                } => ConfigChange::JointChange {
                    additions: truncate_strings(additions, 32, 10),
                    removals: truncate_strings(removals, 32, 10),
                },
            };

            // Bincode roundtrip
            if let Ok(bytes) = bitcode::serialize(&change) {
                let decoded: Result<ConfigChange, _> = bitcode::deserialize(&bytes);
                assert!(decoded.is_ok(), "Failed to deserialize ConfigChange");
                assert_eq!(decoded.unwrap(), change);
            }
        },

        TestCase::MembershipOps { operations } => {
            let mut config = RaftMembershipConfig::new(vec![
                "node1".to_string(),
                "node2".to_string(),
                "node3".to_string(),
            ]);

            // Apply operations (limit to 20)
            for op in operations.into_iter().take(20) {
                match op {
                    MembershipOp::AddLearner(id) => {
                        let id: String = id.chars().take(32).collect();
                        if !config.voters.contains(&id) && !config.learners.contains(&id) {
                            config.add_learner(id);
                        }
                    },
                    MembershipOp::PromoteLearner(id) => {
                        let id: String = id.chars().take(32).collect();
                        config.promote_learner(&id);
                    },
                    MembershipOp::RemoveNode(id) => {
                        let id: String = id.chars().take(32).collect();
                        config.remove_node(&id);
                    },
                }
            }

            // After any sequence of operations, config should be valid
            // (voters + learners should have no duplicates)
            let all_nodes: Vec<&String> =
                config.voters.iter().chain(config.learners.iter()).collect();
            let unique: HashSet<&String> = all_nodes.iter().copied().collect();
            assert_eq!(all_nodes.len(), unique.len(), "No duplicates allowed");

            // Roundtrip should work
            if let Ok(bytes) = bitcode::serialize(&config) {
                let decoded: Result<RaftMembershipConfig, _> = bitcode::deserialize(&bytes);
                assert!(decoded.is_ok());
            }
        },
    }
});
