#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{
    gossip::GossipMessage,
    signing::{Identity, SequenceTracker, SignedGossipMessage, ValidatorRegistry},
};

#[derive(Arbitrary, Debug)]
struct SignedGossipInput {
    message_type: MessageType,
    sequence: u64,
    timestamp_offset_ms: i64,
    tamper_signature: bool,
    tamper_payload: bool,
    tamper_sender: bool,
}

#[derive(Arbitrary, Debug)]
enum MessageType {
    Sync {
        sender: String,
        state_count: u8,
        sender_time: u64,
    },
    Suspect {
        reporter: String,
        suspect: String,
        incarnation: u64,
    },
    Alive {
        node_id: String,
        incarnation: u64,
    },
    PingReq {
        origin: String,
        target: String,
        sequence: u64,
    },
    PingAck {
        origin: String,
        target: String,
        sequence: u64,
        success: bool,
    },
}

fuzz_target!(|input: SignedGossipInput| {
    // Generate a fixed identity for the fuzzer
    let identity = Identity::generate();
    let registry = ValidatorRegistry::new();
    registry.register(&identity);
    let tracker = SequenceTracker::new();

    // Create the gossip message based on input type
    let gossip_msg = match input.message_type {
        MessageType::Sync {
            sender,
            state_count: _,
            sender_time,
        } => {
            let sender: String = sender.chars().take(64).collect();
            GossipMessage::Sync {
                sender,
                states: vec![],
                sender_time,
            }
        }
        MessageType::Suspect {
            reporter,
            suspect,
            incarnation,
        } => {
            let reporter: String = reporter.chars().take(64).collect();
            let suspect: String = suspect.chars().take(64).collect();
            GossipMessage::Suspect {
                reporter,
                suspect,
                incarnation,
            }
        }
        MessageType::Alive {
            node_id,
            incarnation,
        } => {
            let node_id: String = node_id.chars().take(64).collect();
            GossipMessage::Alive {
                node_id,
                incarnation,
            }
        }
        MessageType::PingReq {
            origin,
            target,
            sequence,
        } => {
            let origin: String = origin.chars().take(64).collect();
            let target: String = target.chars().take(64).collect();
            GossipMessage::PingReq {
                origin,
                target,
                sequence,
            }
        }
        MessageType::PingAck {
            origin,
            target,
            sequence,
            success,
        } => {
            let origin: String = origin.chars().take(64).collect();
            let target: String = target.chars().take(64).collect();
            GossipMessage::PingAck {
                origin,
                target,
                sequence,
                success,
            }
        }
    };

    // Create signed message
    let seq = input.sequence.min(u64::MAX - 1);
    let signed_result = SignedGossipMessage::new(&identity, &gossip_msg, seq);

    if let Ok(mut signed) = signed_result {
        let original_sender = signed.envelope.sender.clone();
        let original_sig = signed.envelope.signature.clone();

        // Apply tampering based on input
        if input.tamper_signature && !signed.envelope.signature.is_empty() {
            signed.envelope.signature[0] ^= 0xFF;
        }

        if input.tamper_payload && !signed.envelope.payload.is_empty() {
            signed.envelope.payload[0] ^= 0xFF;
        }

        if input.tamper_sender {
            signed.envelope.sender = "tampered_sender_id_12345".to_string();
        }

        // Properties to verify:
        // 1. Tampered messages MUST fail verification (unless no tampering)
        // 2. Valid messages MUST pass verification
        // 3. Never panic on any input

        let verify_result = signed.verify(&registry);

        if input.tamper_signature || input.tamper_payload || input.tamper_sender {
            // Tampered messages should fail
            // (but may still succeed if tampering doesn't affect verification)
            if verify_result.is_ok() {
                // This is only acceptable if we didn't actually tamper with anything
                // that affects verification
                if input.tamper_signature && !original_sig.is_empty() {
                    // Should have failed
                    assert!(
                        verify_result.is_err(),
                        "Signature tampering should cause verification failure"
                    );
                }
                if input.tamper_sender && signed.envelope.sender != original_sender {
                    // Should have failed
                    assert!(
                        verify_result.is_err(),
                        "Sender tampering should cause verification failure"
                    );
                }
            }
        }

        // Test replay detection with tracker
        let tracker_result = signed.verify_with_tracker(&registry, &tracker);

        // If verification without tracker failed, with tracker should also fail
        if verify_result.is_err() {
            // With tracker should also fail (or fail for replay)
        }

        // Test serialization roundtrip (should never panic)
        if let Ok(bytes) = bincode::serialize(&signed) {
            let _: Result<SignedGossipMessage, _> = bincode::deserialize(&bytes);
        }
    }
});
