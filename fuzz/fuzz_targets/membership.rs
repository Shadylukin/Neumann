// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tensor_chain::membership::{ClusterConfig, HealthConfig, LocalNodeConfig, PeerNodeConfig};

#[derive(Arbitrary, Debug)]
struct MembershipInput {
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    JsonRoundtrip {
        cluster_id: String,
        local_node_id: String,
        local_port: u16,
        peers: Vec<PeerInput>,
        health: HealthInput,
    },
    BincodeRoundtrip {
        cluster_id: String,
        local_node_id: String,
        local_port: u16,
        peers: Vec<PeerInput>,
    },
    ParseHealthConfig {
        ping_interval_ms: u64,
        failure_threshold: u8,
        ping_timeout_ms: u64,
        startup_grace_ms: u64,
    },
}

#[derive(Arbitrary, Debug)]
struct PeerInput {
    node_id: String,
    port: u16,
    ip_octets: [u8; 4],
}

#[derive(Arbitrary, Debug)]
struct HealthInput {
    ping_interval_ms: u64,
    failure_threshold: u8,
    ping_timeout_ms: u64,
    startup_grace_ms: u64,
}

fuzz_target!(|input: MembershipInput| {
    match input.test_case {
        TestCase::JsonRoundtrip {
            cluster_id,
            local_node_id,
            local_port,
            peers,
            health,
        } => {
            // Limit string lengths
            let cluster_id: String = cluster_id.chars().take(64).collect();
            let local_node_id: String = local_node_id.chars().take(64).collect();

            // Create config
            let local = LocalNodeConfig {
                node_id: local_node_id,
                bind_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), local_port),
            };

            let mut config = ClusterConfig::new(cluster_id, local);

            // Add peers (limit to 10)
            for peer in peers.into_iter().take(10) {
                let peer_id: String = peer.node_id.chars().take(64).collect();
                let addr = SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::new(
                        peer.ip_octets[0],
                        peer.ip_octets[1],
                        peer.ip_octets[2],
                        peer.ip_octets[3],
                    )),
                    peer.port,
                );
                config.peers.push(PeerNodeConfig {
                    node_id: peer_id,
                    address: addr,
                });
            }

            // Set health config (ensure reasonable values)
            config.health = HealthConfig {
                ping_interval_ms: health.ping_interval_ms.max(1),
                failure_threshold: (health.failure_threshold as usize).max(1),
                ping_timeout_ms: health.ping_timeout_ms.max(1),
                startup_grace_ms: health.startup_grace_ms,
            };

            // Serialize to JSON
            if let Ok(json) = serde_json::to_string(&config) {
                // Deserialize back
                let decoded: Result<ClusterConfig, _> = serde_json::from_str(&json);
                assert!(decoded.is_ok(), "Failed to parse valid JSON config");

                let decoded = decoded.unwrap();
                assert_eq!(decoded.cluster_id, config.cluster_id);
                assert_eq!(decoded.local.node_id, config.local.node_id);
                assert_eq!(decoded.peers.len(), config.peers.len());
            }
        },

        TestCase::BincodeRoundtrip {
            cluster_id,
            local_node_id,
            local_port,
            peers,
        } => {
            let cluster_id: String = cluster_id.chars().take(64).collect();
            let local_node_id: String = local_node_id.chars().take(64).collect();

            let local = LocalNodeConfig {
                node_id: local_node_id,
                bind_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), local_port),
            };

            let mut config = ClusterConfig::new(cluster_id, local);

            for peer in peers.into_iter().take(10) {
                let peer_id: String = peer.node_id.chars().take(64).collect();
                let addr = SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::new(
                        peer.ip_octets[0],
                        peer.ip_octets[1],
                        peer.ip_octets[2],
                        peer.ip_octets[3],
                    )),
                    peer.port,
                );
                config.peers.push(PeerNodeConfig {
                    node_id: peer_id,
                    address: addr,
                });
            }

            // Serialize to bincode
            if let Ok(bytes) = bitcode::serialize(&config) {
                // Deserialize back
                let decoded: Result<ClusterConfig, _> = bitcode::deserialize(&bytes);
                assert!(decoded.is_ok(), "Failed to parse valid bincode config");

                let decoded = decoded.unwrap();
                assert_eq!(decoded.cluster_id, config.cluster_id);
            }
        },

        TestCase::ParseHealthConfig {
            ping_interval_ms,
            failure_threshold,
            ping_timeout_ms,
            startup_grace_ms,
        } => {
            let config = HealthConfig {
                ping_interval_ms: ping_interval_ms.max(1),
                failure_threshold: (failure_threshold as usize).max(1),
                ping_timeout_ms: ping_timeout_ms.max(1),
                startup_grace_ms,
            };

            // JSON roundtrip
            if let Ok(json) = serde_json::to_string(&config) {
                let decoded: Result<HealthConfig, _> = serde_json::from_str(&json);
                assert!(decoded.is_ok(), "Failed to parse valid health config");

                let decoded = decoded.unwrap();
                assert_eq!(decoded.ping_interval_ms, config.ping_interval_ms);
                assert_eq!(decoded.failure_threshold, config.failure_threshold);
            }

            // Bincode roundtrip
            if let Ok(bytes) = bitcode::serialize(&config) {
                let decoded: Result<HealthConfig, _> = bitcode::deserialize(&bytes);
                assert!(
                    decoded.is_ok(),
                    "Failed to parse valid bincode health config"
                );
            }
        },
    }
});
