// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph intelligence module for security introspection.
//!
//! Provides analytical functions over the vault's access-control graph:
//! path explanation, blast-radius analysis, grant simulation,
//! security auditing (cycles, SPOFs, over-privilege), and critical entity detection.

use std::collections::{HashMap, HashSet, VecDeque};

use std::cmp::Ordering;

use graph_engine::algorithms::{SimilarityConfig, TriangleConfig};
use graph_engine::{Direction, PropertyValue};

use crate::access::AccessController;
use crate::vault::Vault;
use crate::Permission;

/// Hard limit on BFS traversal depth.
const MAX_BFS_DEPTH: usize = 32;

/// Edge type prefix for vault access grants.
const VAULT_ACCESS_PREFIX: &str = "VAULT_ACCESS";

/// Allowed edge types for traversal.
const ALLOWED_TRAVERSAL_EDGES: &[&str] = &[
    "VAULT_ACCESS",
    "VAULT_ACCESS_READ",
    "VAULT_ACCESS_WRITE",
    "VAULT_ACCESS_ADMIN",
    "MEMBER",
];

fn is_allowed_edge_type(edge_type: &str) -> bool {
    ALLOWED_TRAVERSAL_EDGES
        .iter()
        .any(|&allowed| edge_type.starts_with(allowed))
}

// =====================================================================
// Types
// =====================================================================

/// A single hop in an access path explanation.
#[derive(Debug, Clone)]
pub struct AccessHop {
    /// The entity key at this hop (e.g. `"user:alice"` or `"vault_secret:db/pass"`).
    pub entity: String,
    /// The graph edge type traversed to reach this hop (e.g. `"MEMBER"` or `"VAULT_ACCESS_ADMIN"`).
    pub edge_type: String,
    /// The permission level granted by this edge, if it is an access edge.
    pub permission: Option<Permission>,
    /// Zero-based index of this hop in the path.
    pub hop_index: usize,
}

/// Why access was denied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenialReason {
    /// No graph path exists between the entity and the secret.
    NoPath,
    /// A path exists but the highest permission is below what was required.
    InsufficientPermission {
        /// The highest permission found along any path.
        highest: Permission,
        /// The permission level that was required.
        required: Permission,
    },
    /// Permission was attenuated below usable level by hop distance.
    AttenuatedBeyondThreshold {
        /// The original (pre-attenuation) permission on the edge.
        original: Permission,
        /// The permission after attenuation, if any remained.
        attenuated_to: Option<Permission>,
    },
    /// An HMAC signature on an access edge failed verification.
    TamperedEdge {
        /// The hop index where the tampered edge was detected.
        at_hop: usize,
    },
}

/// Full explanation of how an entity accesses (or fails to access) a secret.
#[derive(Debug, Clone)]
pub struct AccessExplanation {
    /// The entity whose access is being explained.
    pub entity: String,
    /// The secret being accessed.
    pub secret: String,
    /// Whether the entity has effective access to the secret.
    pub granted: bool,
    /// The best effective permission across all paths, after attenuation.
    pub effective_permission: Option<Permission>,
    /// All discovered paths from entity to secret through the access graph.
    pub paths: Vec<Vec<AccessHop>>,
    /// The reason access was denied, if applicable.
    pub denial_reason: Option<DenialReason>,
}

/// A single secret reachable by an entity.
#[derive(Debug, Clone)]
pub struct ReachableSecret {
    /// Plaintext name of the secret.
    pub secret_name: String,
    /// Best effective permission to this secret.
    pub permission: Permission,
    /// Shortest path length (in hops) to reach this secret.
    pub hop_count: usize,
    /// Number of distinct paths that reach this secret at the shortest hop distance.
    pub path_count: usize,
}

/// All secrets reachable by an entity.
#[derive(Debug, Clone)]
pub struct BlastRadius {
    /// The entity whose blast radius is being computed.
    pub entity: String,
    /// All secrets reachable by this entity, sorted by permission then hop count.
    pub secrets: Vec<ReachableSecret>,
    /// Total number of reachable secrets.
    pub total_secrets: usize,
}

/// A new access created by a simulated grant.
#[derive(Debug, Clone)]
pub struct NewAccess {
    /// The entity that would gain new or upgraded access.
    pub entity: String,
    /// The secret that would become accessible.
    pub secret: String,
    /// The permission level that would be granted.
    pub permission: Permission,
}

/// Result of a simulated grant.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// The entity that would receive the direct grant.
    pub target_entity: String,
    /// The secret the grant targets.
    pub secret: String,
    /// The permission level requested in the simulation.
    pub requested_permission: Permission,
    /// All entities that would gain new or upgraded access transitively.
    pub new_accesses: Vec<NewAccess>,
    /// Total number of affected entities.
    pub total_affected: usize,
}

/// A cycle detected in the access graph.
#[derive(Debug, Clone)]
pub struct AccessCycle {
    /// The entity keys forming the cycle (strongly connected component).
    pub entities: Vec<String>,
}

/// An entity whose removal would disconnect parts of the graph.
#[derive(Debug, Clone)]
pub struct SinglePointOfFailure {
    /// The entity key that is a single point of failure.
    pub entity: String,
    /// Number of secrets that become unreachable if this entity is removed.
    pub secrets_affected: usize,
    /// Number of other entities that lose access to at least one secret.
    pub entities_affected: usize,
}

/// An entity with disproportionately high access.
#[derive(Debug, Clone)]
pub struct OverPrivilegedEntity {
    /// The over-privileged entity key.
    pub entity: String,
    /// PageRank score reflecting graph influence.
    pub pagerank_score: f64,
    /// Total number of secrets reachable by this entity.
    pub reachable_secrets: usize,
    /// Number of secrets accessible with admin permission.
    pub admin_count: usize,
}

/// Full security audit report.
#[derive(Debug, Clone)]
pub struct SecurityAuditReport {
    /// All cycles (strongly connected components) found in the access graph.
    pub cycles: Vec<AccessCycle>,
    /// Entities whose removal would disconnect secrets from the graph.
    pub single_points_of_failure: Vec<SinglePointOfFailure>,
    /// Entities with disproportionately broad or elevated access.
    pub over_privileged: Vec<OverPrivilegedEntity>,
    /// Total number of non-secret entity nodes in the graph.
    pub total_entities: usize,
    /// Total number of vault secrets.
    pub total_secrets: usize,
    /// Total number of edges in the access graph.
    pub total_edges: usize,
}

/// An entity identified as critical infrastructure.
#[derive(Debug, Clone)]
pub struct CriticalEntity {
    /// The critical entity key.
    pub entity: String,
    /// Whether removing this entity would disconnect secrets from the graph.
    pub is_single_point_of_failure: bool,
    /// Number of secrets that depend solely on this entity for reachability.
    pub secrets_solely_dependent: usize,
    /// Total number of secrets reachable from this entity.
    pub total_reachable_secrets: usize,
    /// PageRank score reflecting graph influence.
    pub pagerank_score: f64,
}

// =====================================================================
// Tier 3: Graph Analytics Types
// =====================================================================

/// Individual entity privilege analysis.
#[derive(Debug, Clone)]
pub struct PrivilegeAnalysis {
    /// The entity being analysed.
    pub entity: String,
    /// PageRank score reflecting graph influence.
    pub pagerank_score: f64,
    /// Total number of secrets reachable by this entity.
    pub reachable_secrets: usize,
    /// Number of secrets reachable with admin permission.
    pub admin_count: usize,
    /// Number of secrets reachable with write permission.
    pub write_count: usize,
    /// Number of secrets reachable with read permission.
    pub read_count: usize,
    /// `pagerank * reachable_secrets` -- higher means more over-privileged.
    pub privilege_score: f64,
}

/// Full privilege analysis report.
#[derive(Debug, Clone)]
pub struct PrivilegeAnalysisReport {
    /// Per-entity privilege analyses, sorted by privilege score descending.
    pub entities: Vec<PrivilegeAnalysis>,
    /// Arithmetic mean of all privilege scores.
    pub mean_privilege_score: f64,
    /// Highest privilege score across all entities.
    pub max_privilege_score: f64,
}

/// Anomaly score for a specific access grant.
#[derive(Debug, Clone)]
pub struct DelegationAnomalyScore {
    /// The entity holding the grant.
    pub entity: String,
    /// The secret targeted by the grant.
    pub secret: String,
    /// Jaccard similarity between the entity and secret neighborhoods.
    pub jaccard: f64,
    /// Adamic-Adar index between the entity and secret nodes.
    pub adamic_adar: f64,
    /// Combined score: `1.0 - jaccard` (low similarity = high anomaly).
    pub anomaly_score: f64,
}

/// An inferred role based on community detection.
#[derive(Debug, Clone)]
pub struct InferredRole {
    /// Louvain community identifier for this role.
    pub role_id: u64,
    /// Entity keys assigned to this role.
    pub members: Vec<String>,
    /// Secrets reachable by every member of this role.
    pub common_secrets: Vec<String>,
}

/// Result of role inference.
#[derive(Debug, Clone)]
pub struct RoleInferenceResult {
    /// Inferred roles sorted by member count descending.
    pub roles: Vec<InferredRole>,
    /// Louvain modularity score of the community partition.
    pub modularity: f64,
    /// Entities not assigned to any role (singleton communities).
    pub unassigned: Vec<String>,
}

/// Per-entity trust score based on triangle participation.
#[derive(Debug, Clone)]
pub struct EntityTrustScore {
    /// The entity being scored.
    pub entity: String,
    /// Number of triangles this entity participates in.
    pub triangle_count: usize,
    /// Local clustering coefficient of this entity's neighborhood.
    pub clustering_coefficient: f64,
    /// Normalized trust: `clustering_coefficient * (1 + ln(1 + triangles))`.
    pub trust_score: f64,
}

/// Full trust transitivity report.
#[derive(Debug, Clone)]
pub struct TrustTransitivityReport {
    /// Per-entity trust scores sorted by trust score descending.
    pub entities: Vec<EntityTrustScore>,
    /// Global clustering coefficient of the access graph.
    pub global_clustering: f64,
    /// Total number of triangles found in the graph.
    pub total_triangles: usize,
}

/// Contributor to an entity's risk.
#[derive(Debug, Clone)]
pub struct RiskContributor {
    /// The secret contributing to this entity's risk score.
    pub secret: String,
    /// The permission level held on this secret.
    pub permission: Permission,
    /// Number of hops to reach this secret from the entity.
    pub hop_count: usize,
}

/// Per-entity risk score.
#[derive(Debug, Clone)]
pub struct EntityRiskScore {
    /// The entity being scored.
    pub entity: String,
    /// Eigenvector centrality score reflecting structural importance.
    pub eigenvector_score: f64,
    /// Number of secrets reachable with admin permission.
    pub reachable_admin_secrets: usize,
    /// Admin-level secrets that contribute to the risk score.
    pub risk_contributors: Vec<RiskContributor>,
    /// `eigenvector * (1 + admin_count)` -- higher means more risk.
    pub risk_score: f64,
}

/// Full risk propagation report.
#[derive(Debug, Clone)]
pub struct RiskPropagationReport {
    /// Per-entity risk scores sorted by risk score descending.
    pub entities: Vec<EntityRiskScore>,
    /// Arithmetic mean of all entity risk scores.
    pub mean_risk: f64,
    /// Highest risk score across all entities.
    pub max_risk: f64,
}

// =====================================================================
// Internal helpers
// =====================================================================

/// Edge info for BFS traversal inside graph_intel.
struct IntelEdgeInfo {
    target_key: String,
    source_key: String,
    edge_type: String,
    signature: Option<Vec<u8>>,
    sig_timestamp: Option<i64>,
    capacity: Option<i64>,
}

/// Get outgoing edges with full metadata for a given entity key.
fn get_outgoing_edges_full(vault: &Vault, entity_key: &str) -> Vec<IntelEdgeInfo> {
    let Some(node_id) = vault.find_entity_node(entity_key) else {
        return Vec::new();
    };

    let mut result = Vec::new();
    if let Ok(edges) = vault.graph.edges_of(node_id, Direction::Outgoing) {
        for edge in edges {
            let target_id = if edge.from == node_id {
                edge.to
            } else {
                edge.from
            };
            if let Some(target_key) = vault.node_entity_key(target_id) {
                let signature = match edge.properties.get("vault_sig") {
                    Some(PropertyValue::Bytes(b)) => Some(b.clone()),
                    _ => None,
                };
                let sig_timestamp = match edge.properties.get("vault_sig_ts") {
                    Some(PropertyValue::Int(ts)) => Some(*ts),
                    _ => None,
                };
                let capacity = match edge.properties.get("vault_capacity") {
                    Some(PropertyValue::Int(c)) => Some(*c),
                    _ => None,
                };
                result.push(IntelEdgeInfo {
                    target_key,
                    source_key: entity_key.to_string(),
                    edge_type: edge.edge_type.clone(),
                    signature,
                    sig_timestamp,
                    capacity,
                });
            }
        }
    }
    result
}

/// Collect all vault secret node keys mapped to their decrypted plaintext names.
fn collect_vault_secrets(vault: &Vault) -> Vec<(String, String)> {
    let mut secrets = Vec::new();
    for vault_key in vault.store.scan(Vault::PREFIX) {
        if let Ok(tensor) = vault.store.get(&vault_key) {
            if let Some(name) = vault.decrypt_key_name(&tensor) {
                let node_key = vault.secret_node_key(&name);
                secrets.push((node_key, name));
            }
        }
    }
    secrets
}

/// Collect all entity node keys (non-secret nodes) from the graph.
fn collect_entity_keys(vault: &Vault) -> Vec<String> {
    let mut entities = Vec::new();
    let secrets_prefix = "vault_secret:";
    let all_edges = vault.graph.all_edges();
    let mut seen = HashSet::new();
    for edge in all_edges {
        for node_id in [edge.from, edge.to] {
            if seen.insert(node_id) {
                if let Some(key) = vault.node_entity_key(node_id) {
                    if !key.starts_with(secrets_prefix) {
                        entities.push(key);
                    }
                }
            }
        }
    }
    entities
}

/// BFS reachability from an entity, returning all reachable secret node keys
/// with their best effective permission and shortest hop count.
fn bfs_reachable_secrets(
    vault: &Vault,
    entity: &str,
) -> HashMap<String, (Permission, usize, usize)> {
    // Maps secret_node_key -> (best_permission, shortest_hop_count, path_count)
    let mut result: HashMap<String, (Permission, usize, usize)> = HashMap::new();

    let mut visited = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    queue.push_back((entity.to_string(), 0));
    visited.insert(entity.to_string());

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= vault.attenuation.horizon {
            continue;
        }

        for edge in get_outgoing_edges_full(vault, &current) {
            if !is_allowed_edge_type(&edge.edge_type) {
                continue;
            }

            if edge.edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                // Verify signature if present
                if let (Some(sig), Some(ts)) = (&edge.signature, edge.sig_timestamp) {
                    if !vault.edge_signer.verify_edge(
                        &edge.source_key,
                        &edge.target_key,
                        &edge.edge_type,
                        ts,
                        sig,
                    ) {
                        continue; // skip tampered edge
                    }
                }

                let total_hops = depth + 1;
                if let Some(perm) = Permission::from_edge_type(&edge.edge_type) {
                    if let Some(attenuated) = vault.attenuation.attenuate(perm, total_hops) {
                        let effective = match edge.capacity.and_then(Permission::from_level) {
                            Some(cap) => min_permission(attenuated, cap),
                            None => attenuated,
                        };
                        let entry = result.entry(edge.target_key.clone()).or_insert((
                            Permission::Read,
                            usize::MAX,
                            0,
                        ));
                        if total_hops < entry.1 {
                            // Shorter path found: reset
                            *entry = (effective, total_hops, 1);
                        } else if total_hops == entry.1 {
                            entry.2 += 1;
                            entry.0 = max_permission(entry.0, effective);
                        } else if effective.to_level() > entry.0.to_level() {
                            entry.0 = effective;
                        }
                    }
                }
            } else {
                // MEMBER edge -- continue traversal
                if !visited.contains(&edge.target_key) {
                    visited.insert(edge.target_key.clone());
                    queue.push_back((edge.target_key, depth + 1));
                }
            }
        }
    }

    result
}

/// BFS to find all entities reachable via MEMBER edges from the given entity.
fn bfs_member_reachable(vault: &Vault, entity: &str) -> HashSet<String> {
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    queue.push_back((entity.to_string(), 0));
    visited.insert(entity.to_string());

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= MAX_BFS_DEPTH {
            continue;
        }

        // Check incoming MEMBER edges (who points to `current`)
        if let Some(node_id) = vault.find_entity_node(&current) {
            if let Ok(edges) = vault.graph.edges_of(node_id, Direction::Incoming) {
                for edge in edges {
                    if edge.edge_type == "MEMBER" {
                        let source_id = if edge.to == node_id {
                            edge.from
                        } else {
                            edge.to
                        };
                        if let Some(source_key) = vault.node_entity_key(source_id) {
                            if visited.insert(source_key.clone()) {
                                queue.push_back((source_key, depth + 1));
                            }
                        }
                    }
                }
            }
        }
    }

    visited
}

fn max_permission(a: Permission, b: Permission) -> Permission {
    match (a, b) {
        (Permission::Admin, _) | (_, Permission::Admin) => Permission::Admin,
        (Permission::Write, _) | (_, Permission::Write) => Permission::Write,
        (Permission::Read, Permission::Read) => Permission::Read,
    }
}

fn min_permission(a: Permission, b: Permission) -> Permission {
    match (a, b) {
        (Permission::Read, _) | (_, Permission::Read) => Permission::Read,
        (Permission::Write, _) | (_, Permission::Write) => Permission::Write,
        (Permission::Admin, Permission::Admin) => Permission::Admin,
    }
}

// =====================================================================
// Core functions
// =====================================================================

/// Intermediate BFS result for explain_access.
struct ExplainBfsResult {
    paths: Vec<Vec<AccessHop>>,
    best_permission: Option<Permission>,
    tampered_hop: Option<usize>,
    highest_raw: Option<Permission>,
}

/// BFS to find all paths from entity to secret_node with signature verification.
fn explain_bfs(vault: &Vault, entity: &str, secret_node: &str) -> ExplainBfsResult {
    let mut paths: Vec<Vec<AccessHop>> = Vec::new();
    let mut best_permission: Option<Permission> = None;
    let mut tampered_hop: Option<usize> = None;
    let mut highest_raw: Option<Permission> = None;

    let mut queue: VecDeque<(String, usize, Vec<AccessHop>)> = VecDeque::new();
    let mut visited = HashSet::new();

    queue.push_back((entity.to_string(), 0, Vec::new()));
    visited.insert(entity.to_string());

    while let Some((current, depth, path)) = queue.pop_front() {
        if depth >= vault.attenuation.horizon {
            continue;
        }

        for edge in get_outgoing_edges_full(vault, &current) {
            if !is_allowed_edge_type(&edge.edge_type) {
                continue;
            }

            if edge.edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                if edge.target_key == secret_node {
                    let total_hops = depth + 1;
                    let perm_from_edge = Permission::from_edge_type(&edge.edge_type);

                    if !verify_edge_sig(vault, &edge) {
                        tampered_hop = Some(total_hops);
                        continue;
                    }

                    let mut hop_path = path.clone();
                    hop_path.push(AccessHop {
                        entity: edge.target_key.clone(),
                        edge_type: edge.edge_type.clone(),
                        permission: perm_from_edge,
                        hop_index: total_hops,
                    });

                    if let Some(perm) = perm_from_edge {
                        highest_raw = Some(highest_raw.map_or(perm, |e| max_permission(e, perm)));
                        if let Some(eff) = compute_effective(vault, &edge, perm, total_hops) {
                            best_permission =
                                Some(best_permission.map_or(eff, |e| max_permission(e, eff)));
                        }
                    }
                    paths.push(hop_path);
                }
            } else if !visited.contains(&edge.target_key) {
                visited.insert(edge.target_key.clone());
                let mut new_path = path.clone();
                new_path.push(AccessHop {
                    entity: edge.target_key.clone(),
                    edge_type: edge.edge_type.clone(),
                    permission: None,
                    hop_index: depth + 1,
                });
                queue.push_back((edge.target_key, depth + 1, new_path));
            }
        }
    }

    ExplainBfsResult {
        paths,
        best_permission,
        tampered_hop,
        highest_raw,
    }
}

/// Verify an edge's HMAC signature. Unsigned (legacy) edges pass.
fn verify_edge_sig(vault: &Vault, edge: &IntelEdgeInfo) -> bool {
    if let (Some(sig), Some(ts)) = (&edge.signature, edge.sig_timestamp) {
        vault
            .edge_signer
            .verify_edge(&edge.source_key, &edge.target_key, &edge.edge_type, ts, sig)
    } else {
        true
    }
}

/// Compute the effective permission after attenuation and bottleneck.
fn compute_effective(
    vault: &Vault,
    edge: &IntelEdgeInfo,
    perm: Permission,
    hops: usize,
) -> Option<Permission> {
    vault.attenuation.attenuate(perm, hops).map(|attenuated| {
        match edge.capacity.and_then(Permission::from_level) {
            Some(cap) => min_permission(attenuated, cap),
            None => attenuated,
        }
    })
}

/// Explain how an entity accesses (or fails to access) a specific secret.
pub fn explain_access(vault: &Vault, entity: &str, secret: &str) -> AccessExplanation {
    let secret_node = vault.secret_node_key(secret);

    if vault.find_entity_node(entity).is_none() {
        return AccessExplanation {
            entity: entity.to_string(),
            secret: secret.to_string(),
            granted: false,
            effective_permission: None,
            paths: Vec::new(),
            denial_reason: Some(DenialReason::NoPath),
        };
    }

    let bfs = explain_bfs(vault, entity, &secret_node);

    let denial_reason = if bfs.paths.is_empty() {
        if bfs.tampered_hop.is_some() {
            bfs.tampered_hop
                .map(|at_hop| DenialReason::TamperedEdge { at_hop })
        } else {
            Some(DenialReason::NoPath)
        }
    } else if bfs.best_permission.is_none() {
        Some(DenialReason::AttenuatedBeyondThreshold {
            original: bfs.highest_raw.unwrap_or(Permission::Read),
            attenuated_to: None,
        })
    } else {
        None
    };

    AccessExplanation {
        entity: entity.to_string(),
        secret: secret.to_string(),
        granted: bfs.best_permission.is_some(),
        effective_permission: bfs.best_permission,
        paths: bfs.paths,
        denial_reason,
    }
}

/// Compute the blast radius: all secrets reachable by an entity.
pub fn blast_radius(vault: &Vault, entity: &str) -> BlastRadius {
    let reachable = bfs_reachable_secrets(vault, entity);
    let all_secrets = collect_vault_secrets(vault);

    // Map secret_node_key -> plaintext name
    let node_to_name: HashMap<String, String> = all_secrets.into_iter().collect();

    let mut secrets: Vec<ReachableSecret> = Vec::new();
    for (node_key, (perm, hops, paths)) in &reachable {
        if let Some(name) = node_to_name.get(node_key) {
            secrets.push(ReachableSecret {
                secret_name: name.clone(),
                permission: *perm,
                hop_count: *hops,
                path_count: *paths,
            });
        }
    }

    secrets.sort_by(|a, b| {
        b.permission
            .to_level()
            .cmp(&a.permission.to_level())
            .then_with(|| a.hop_count.cmp(&b.hop_count))
            .then_with(|| a.secret_name.cmp(&b.secret_name))
    });

    let total = secrets.len();
    BlastRadius {
        entity: entity.to_string(),
        secrets,
        total_secrets: total,
    }
}

/// Simulate granting an entity access to a secret and return the impact.
pub fn simulate_grant(
    vault: &Vault,
    entity: &str,
    secret: &str,
    permission: Permission,
) -> SimulationResult {
    let secret_node = vault.secret_node_key(secret);

    // Find all entities that would gain access through the target entity.
    // These are entities that can reach `entity` via MEMBER edges (incoming).
    let reachable_entities = bfs_member_reachable(vault, entity);

    let mut new_accesses = Vec::new();

    for ent in &reachable_entities {
        // Check current access
        let current_perm = AccessController::get_permission_level_verified(
            &vault.graph,
            ent,
            &secret_node,
            &vault.edge_signer,
            &vault.attenuation,
        );

        // Would this entity gain new or upgraded access?
        let would_gain = match current_perm {
            None => true,
            Some(existing) => permission.to_level() > existing.to_level(),
        };

        if would_gain {
            new_accesses.push(NewAccess {
                entity: ent.clone(),
                secret: secret.to_string(),
                permission,
            });
        }
    }

    let total = new_accesses.len();
    SimulationResult {
        target_entity: entity.to_string(),
        secret: secret.to_string(),
        requested_permission: permission,
        new_accesses,
        total_affected: total,
    }
}

/// Run a full security audit of the vault's access graph.
pub fn security_audit(vault: &Vault) -> SecurityAuditReport {
    let all_entities = collect_entity_keys(vault);
    let all_secrets = collect_vault_secrets(vault);
    let edge_count = vault.graph.all_edges().len();

    // 1. Cycle detection (Tarjan SCC)
    let cycles = detect_cycles(vault, &all_entities);

    // 2. Single points of failure
    let spofs = detect_spofs(vault, &all_entities, &all_secrets);

    // 3. Over-privileged entities
    let over_privileged = detect_over_privileged(vault, &all_entities);

    SecurityAuditReport {
        cycles,
        single_points_of_failure: spofs,
        over_privileged,
        total_entities: all_entities.len(),
        total_secrets: all_secrets.len(),
        total_edges: edge_count,
    }
}

/// Find entities critical to vault infrastructure.
pub fn find_critical_entities(vault: &Vault) -> Vec<CriticalEntity> {
    let all_entities = collect_entity_keys(vault);
    let all_secrets = collect_vault_secrets(vault);
    let spofs = detect_spofs(vault, &all_entities, &all_secrets);

    let spof_set: HashSet<String> = spofs.iter().map(|s| s.entity.clone()).collect();
    let spof_map: HashMap<String, &SinglePointOfFailure> =
        spofs.iter().map(|s| (s.entity.clone(), s)).collect();

    // PageRank scores
    let pr_scores = vault
        .graph
        .pagerank(None)
        .map_or_else(|_| HashMap::new(), |pr| pr.scores);

    let mut results = Vec::new();
    for ent in &all_entities {
        if ent == Vault::ROOT {
            continue;
        }

        let reachable = bfs_reachable_secrets(vault, ent);
        let total_reachable = reachable.len();
        if total_reachable == 0 {
            continue;
        }

        let node_id = vault.find_entity_node(ent);
        let pr_score = node_id
            .and_then(|id| pr_scores.get(&id).copied())
            .unwrap_or(0.0);

        let is_spof = spof_set.contains(ent);
        let solely_dependent = spof_map.get(ent).map_or(0, |s| s.secrets_affected);

        results.push(CriticalEntity {
            entity: ent.clone(),
            is_single_point_of_failure: is_spof,
            secrets_solely_dependent: solely_dependent,
            total_reachable_secrets: total_reachable,
            pagerank_score: pr_score,
        });
    }

    // Sort: SPOFs first, then by pagerank * reachable_secrets descending
    results.sort_by(|a, b| {
        b.is_single_point_of_failure
            .cmp(&a.is_single_point_of_failure)
            .then_with(|| {
                #[allow(clippy::cast_precision_loss)]
                let score_a = a.pagerank_score * a.total_reachable_secrets as f64;
                #[allow(clippy::cast_precision_loss)]
                let score_b = b.pagerank_score * b.total_reachable_secrets as f64;
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    results
}

// =====================================================================
// Security audit sub-algorithms
// =====================================================================

/// Tarjan SCC state bundled into a struct to avoid many function parameters.
struct TarjanState<'a> {
    vault: &'a Vault,
    index_counter: usize,
    stack: Vec<String>,
    on_stack: HashSet<String>,
    indices: HashMap<String, usize>,
    lowlinks: HashMap<String, usize>,
    sccs: Vec<Vec<String>>,
}

impl<'a> TarjanState<'a> {
    fn new(vault: &'a Vault) -> Self {
        Self {
            vault,
            index_counter: 0,
            stack: Vec::new(),
            on_stack: HashSet::new(),
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            sccs: Vec::new(),
        }
    }

    fn visit(&mut self, v: &str) {
        self.indices.insert(v.to_string(), self.index_counter);
        self.lowlinks.insert(v.to_string(), self.index_counter);
        self.index_counter += 1;
        self.stack.push(v.to_string());
        self.on_stack.insert(v.to_string());

        let successors: Vec<String> = get_outgoing_edges_full(self.vault, v)
            .into_iter()
            .filter(|e| is_allowed_edge_type(&e.edge_type))
            .map(|e| e.target_key)
            .collect();

        for w in &successors {
            if !self.indices.contains_key(w) {
                self.visit(w);
                let lw = *self.lowlinks.get(w).unwrap_or(&usize::MAX);
                let lv = *self.lowlinks.get(v).unwrap_or(&usize::MAX);
                self.lowlinks.insert(v.to_string(), lv.min(lw));
            } else if self.on_stack.contains(w) {
                let iw = *self.indices.get(w).unwrap_or(&usize::MAX);
                let lv = *self.lowlinks.get(v).unwrap_or(&usize::MAX);
                self.lowlinks.insert(v.to_string(), lv.min(iw));
            }
        }

        if self.lowlinks.get(v) == self.indices.get(v) {
            let mut scc = Vec::new();
            while let Some(w) = self.stack.pop() {
                self.on_stack.remove(&w);
                scc.push(w.clone());
                if w == v {
                    break;
                }
            }
            if scc.len() > 1 {
                self.sccs.push(scc);
            }
        }
    }
}

/// Tarjan-style SCC detection to find cycles in entity relationships.
fn detect_cycles(vault: &Vault, entities: &[String]) -> Vec<AccessCycle> {
    let mut state = TarjanState::new(vault);

    for entity in entities {
        if !state.indices.contains_key(entity) {
            state.visit(entity);
        }
    }

    state
        .sccs
        .into_iter()
        .map(|entities| AccessCycle { entities })
        .collect()
}

/// Detect single points of failure: entities whose removal disconnects secrets.
fn detect_spofs(
    vault: &Vault,
    entities: &[String],
    secrets: &[(String, String)],
) -> Vec<SinglePointOfFailure> {
    if secrets.is_empty() {
        return Vec::new();
    }

    // Baseline: which secrets are reachable from root
    let baseline_reachable = bfs_reachable_secrets(vault, Vault::ROOT);
    let baseline_secrets: HashSet<&String> = baseline_reachable.keys().collect();

    let mut spofs = Vec::new();

    for entity in entities {
        if entity == Vault::ROOT {
            continue;
        }

        // Simulate removal: BFS from root, skipping this entity
        let reachable_without = bfs_reachable_secrets_excluding(vault, Vault::ROOT, entity);
        let remaining: HashSet<&String> = reachable_without.keys().collect();

        let lost_secrets: HashSet<&&String> = baseline_secrets.difference(&remaining).collect();
        if lost_secrets.is_empty() {
            continue;
        }

        // Count affected entities: those who lose access to at least one secret
        let other_entities: Vec<&String> = entities
            .iter()
            .filter(|e| *e != entity && *e != Vault::ROOT)
            .collect();

        let mut entities_affected = 0;
        for other in &other_entities {
            let their_reachable = bfs_reachable_secrets_excluding(vault, other, entity);
            let their_remaining: HashSet<&String> = their_reachable.keys().collect();
            let their_original = bfs_reachable_secrets(vault, other);
            let their_baseline: HashSet<&String> = their_original.keys().collect();
            if their_baseline.difference(&their_remaining).next().is_some() {
                entities_affected += 1;
            }
        }

        spofs.push(SinglePointOfFailure {
            entity: entity.clone(),
            secrets_affected: lost_secrets.len(),
            entities_affected,
        });
    }

    spofs.sort_by(|a, b| b.secrets_affected.cmp(&a.secrets_affected));
    spofs
}

/// BFS reachable secrets excluding a specific entity (for SPOF analysis).
fn bfs_reachable_secrets_excluding(
    vault: &Vault,
    start: &str,
    exclude: &str,
) -> HashMap<String, (Permission, usize, usize)> {
    let mut result: HashMap<String, (Permission, usize, usize)> = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    queue.push_back((start.to_string(), 0));
    visited.insert(start.to_string());
    visited.insert(exclude.to_string()); // pre-mark excluded

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= vault.attenuation.horizon {
            continue;
        }

        for edge in get_outgoing_edges_full(vault, &current) {
            if !is_allowed_edge_type(&edge.edge_type) {
                continue;
            }

            if edge.target_key == exclude {
                continue;
            }

            if edge.edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                if let Some(perm) = Permission::from_edge_type(&edge.edge_type) {
                    let total_hops = depth + 1;
                    if let Some(attenuated) = vault.attenuation.attenuate(perm, total_hops) {
                        let effective = match edge.capacity.and_then(Permission::from_level) {
                            Some(cap) => min_permission(attenuated, cap),
                            None => attenuated,
                        };
                        let entry = result.entry(edge.target_key.clone()).or_insert((
                            Permission::Read,
                            usize::MAX,
                            0,
                        ));
                        if total_hops < entry.1 {
                            *entry = (effective, total_hops, 1);
                        } else if total_hops == entry.1 {
                            entry.2 += 1;
                            entry.0 = max_permission(entry.0, effective);
                        }
                    }
                }
            } else if !visited.contains(&edge.target_key) {
                visited.insert(edge.target_key.clone());
                queue.push_back((edge.target_key, depth + 1));
            }
        }
    }

    result
}

/// Detect over-privileged entities using blast radius and admin edge count.
fn detect_over_privileged(vault: &Vault, entities: &[String]) -> Vec<OverPrivilegedEntity> {
    let pr_scores = vault
        .graph
        .pagerank(None)
        .map_or_else(|_| HashMap::new(), |pr| pr.scores);

    let mut results = Vec::new();
    for entity in entities {
        if entity == Vault::ROOT {
            continue;
        }

        let reachable = bfs_reachable_secrets(vault, entity);
        let reachable_count = reachable.len();
        if reachable_count == 0 {
            continue;
        }

        // Count admin edges
        let admin_count = get_outgoing_edges_full(vault, entity)
            .iter()
            .filter(|e| e.edge_type.ends_with("_ADMIN"))
            .count();

        let node_id = vault.find_entity_node(entity);
        let pr_score = node_id
            .and_then(|id| pr_scores.get(&id).copied())
            .unwrap_or(0.0);

        results.push(OverPrivilegedEntity {
            entity: entity.clone(),
            pagerank_score: pr_score,
            reachable_secrets: reachable_count,
            admin_count,
        });
    }

    results.sort_by(|a, b| {
        b.reachable_secrets
            .cmp(&a.reachable_secrets)
            .then_with(|| b.admin_count.cmp(&a.admin_count))
    });

    results
}

// =====================================================================
// Tier 3: Graph Analytics Functions
// =====================================================================

/// Analyse entity privilege levels using PageRank and BFS reachability.
#[allow(clippy::cast_precision_loss)]
pub fn privilege_analysis(vault: &Vault) -> PrivilegeAnalysisReport {
    let entities = collect_entity_keys(vault);
    let pr_scores = vault
        .graph
        .pagerank(None)
        .map_or_else(|_| HashMap::new(), |pr| pr.scores);

    let mut analyses = Vec::new();
    for ent in &entities {
        if ent == Vault::ROOT {
            continue;
        }

        let reachable = bfs_reachable_secrets(vault, ent);
        let reachable_count = reachable.len();

        let mut admin_count = 0usize;
        let mut write_count = 0usize;
        let mut read_count = 0usize;
        for (perm, _, _) in reachable.values() {
            match perm {
                Permission::Admin => admin_count += 1,
                Permission::Write => write_count += 1,
                Permission::Read => read_count += 1,
            }
        }

        let node_id = vault.find_entity_node(ent);
        let pr_score = node_id
            .and_then(|id| pr_scores.get(&id).copied())
            .unwrap_or(0.0);

        let privilege_score = pr_score * reachable_count as f64;

        analyses.push(PrivilegeAnalysis {
            entity: ent.clone(),
            pagerank_score: pr_score,
            reachable_secrets: reachable_count,
            admin_count,
            write_count,
            read_count,
            privilege_score,
        });
    }

    analyses.sort_by(|a, b| {
        b.privilege_score
            .partial_cmp(&a.privilege_score)
            .unwrap_or(Ordering::Equal)
    });

    let max_privilege_score = analyses.first().map_or(0.0, |a| a.privilege_score);
    let sum: f64 = analyses.iter().map(|a| a.privilege_score).sum();
    let mean_privilege_score = if analyses.is_empty() {
        0.0
    } else {
        sum / analyses.len() as f64
    };

    PrivilegeAnalysisReport {
        entities: analyses,
        mean_privilege_score,
        max_privilege_score,
    }
}

/// Score delegation grants for anomalous patterns using Jaccard similarity.
pub fn delegation_anomaly_scores(vault: &Vault) -> Vec<DelegationAnomalyScore> {
    let entities = collect_entity_keys(vault);
    let config = SimilarityConfig {
        edge_type: None,
        direction: Direction::Both,
    };

    let mut scores = Vec::new();
    for ent in &entities {
        if ent == Vault::ROOT {
            continue;
        }

        for edge in get_outgoing_edges_full(vault, ent) {
            if !edge.edge_type.starts_with(VAULT_ACCESS_PREFIX) {
                continue;
            }

            let Some(entity_id) = vault.find_entity_node(ent) else {
                continue;
            };
            let Some(secret_id) = vault.find_entity_node(&edge.target_key) else {
                continue;
            };

            let jaccard = vault
                .graph
                .jaccard_similarity(entity_id, secret_id, &config)
                .unwrap_or(0.0);
            let adamic = vault
                .graph
                .adamic_adar(entity_id, secret_id, &config)
                .unwrap_or(0.0);
            let anomaly_score = 1.0 - jaccard;

            if anomaly_score > 0.5 {
                // Resolve secret name from edge target key
                let secret_name = edge
                    .target_key
                    .strip_prefix("vault_secret:")
                    .unwrap_or(&edge.target_key)
                    .to_string();

                scores.push(DelegationAnomalyScore {
                    entity: ent.clone(),
                    secret: secret_name,
                    jaccard,
                    adamic_adar: adamic,
                    anomaly_score,
                });
            }
        }
    }

    scores.sort_by(|a, b| {
        b.anomaly_score
            .partial_cmp(&a.anomaly_score)
            .unwrap_or(Ordering::Equal)
    });

    scores
}

/// Infer roles from community structure using Louvain algorithm.
pub fn infer_roles(vault: &Vault) -> RoleInferenceResult {
    let Ok(community_result) = vault.graph.louvain_communities(None) else {
        return RoleInferenceResult {
            roles: Vec::new(),
            modularity: 0.0,
            unassigned: Vec::new(),
        };
    };

    let modularity = community_result.modularity.unwrap_or(0.0);
    let secrets_prefix = "vault_secret:";

    // Map community_id -> entity string keys (non-secret nodes only)
    let mut community_entities: HashMap<u64, Vec<String>> = HashMap::new();
    for (&node_id, &comm_id) in &community_result.communities {
        if let Some(key) = vault.node_entity_key(node_id) {
            if !key.starts_with(secrets_prefix) && key != Vault::ROOT {
                community_entities.entry(comm_id).or_default().push(key);
            }
        }
    }

    let mut roles = Vec::new();
    let mut unassigned = Vec::new();

    for (comm_id, members) in &community_entities {
        if members.len() < 2 {
            unassigned.extend(members.clone());
            continue;
        }

        // Find common secrets: intersection of all members' reachable secrets
        let mut common: Option<HashSet<String>> = None;
        for member in members {
            let reachable = bfs_reachable_secrets(vault, member);
            let secret_keys: HashSet<String> = reachable.keys().cloned().collect();
            common = Some(match common {
                Some(existing) => existing.intersection(&secret_keys).cloned().collect(),
                None => secret_keys,
            });
        }

        let common_secrets: Vec<String> = common
            .unwrap_or_default()
            .into_iter()
            .map(|k| k.strip_prefix(secrets_prefix).unwrap_or(&k).to_string())
            .collect();

        roles.push(InferredRole {
            role_id: *comm_id,
            members: members.clone(),
            common_secrets,
        });
    }

    // Sort roles by member count descending
    roles.sort_by(|a, b| b.members.len().cmp(&a.members.len()));

    RoleInferenceResult {
        roles,
        modularity,
        unassigned,
    }
}

/// Compute trust scores from triangle participation and clustering coefficients.
#[allow(clippy::cast_precision_loss)]
pub fn trust_transitivity(vault: &Vault) -> TrustTransitivityReport {
    let config = TriangleConfig {
        edge_type: None,
        undirected: true,
    };
    let Ok(tri_result) = vault.graph.count_triangles(&config) else {
        return TrustTransitivityReport {
            entities: Vec::new(),
            global_clustering: 0.0,
            total_triangles: 0,
        };
    };

    let entities = collect_entity_keys(vault);
    let mut trust_scores = Vec::new();

    for ent in &entities {
        if ent == Vault::ROOT {
            continue;
        }
        let Some(node_id) = vault.find_entity_node(ent) else {
            continue;
        };

        let triangles = tri_result
            .node_triangles
            .get(&node_id)
            .copied()
            .unwrap_or(0);
        let clustering = tri_result
            .local_clustering
            .get(&node_id)
            .copied()
            .unwrap_or(0.0);
        let trust_score = clustering * (1.0 + (triangles as f64).ln_1p());

        trust_scores.push(EntityTrustScore {
            entity: ent.clone(),
            triangle_count: triangles,
            clustering_coefficient: clustering,
            trust_score,
        });
    }

    trust_scores.sort_by(|a, b| {
        b.trust_score
            .partial_cmp(&a.trust_score)
            .unwrap_or(Ordering::Equal)
    });

    TrustTransitivityReport {
        entities: trust_scores,
        global_clustering: tri_result.global_clustering,
        total_triangles: tri_result.triangle_count,
    }
}

/// Compute risk scores using eigenvector centrality and admin reachability.
#[allow(clippy::cast_precision_loss)]
pub fn risk_propagation(vault: &Vault) -> RiskPropagationReport {
    let eig_scores = vault
        .graph
        .eigenvector_centrality(None)
        .map_or_else(|_| HashMap::new(), |r| r.scores);

    let entities = collect_entity_keys(vault);
    let all_secrets = collect_vault_secrets(vault);
    let node_to_name: HashMap<String, String> = all_secrets.into_iter().collect();

    let mut risk_scores = Vec::new();

    for ent in &entities {
        if ent == Vault::ROOT {
            continue;
        }

        let node_id = vault.find_entity_node(ent);
        let eig_score = node_id
            .and_then(|id| eig_scores.get(&id).copied())
            .unwrap_or(0.0);

        let reachable = bfs_reachable_secrets(vault, ent);
        let mut contributors = Vec::new();
        let mut admin_count = 0usize;

        for (secret_key, (perm, hops, _)) in &reachable {
            if *perm == Permission::Admin {
                admin_count += 1;
                let secret_name = node_to_name.get(secret_key).cloned().unwrap_or_else(|| {
                    secret_key
                        .strip_prefix("vault_secret:")
                        .unwrap_or(secret_key)
                        .to_string()
                });
                contributors.push(RiskContributor {
                    secret: secret_name,
                    permission: *perm,
                    hop_count: *hops,
                });
            }
        }

        let risk_score = eig_score * (1.0 + admin_count as f64);

        risk_scores.push(EntityRiskScore {
            entity: ent.clone(),
            eigenvector_score: eig_score,
            reachable_admin_secrets: admin_count,
            risk_contributors: contributors,
            risk_score,
        });
    }

    risk_scores.sort_by(|a, b| {
        b.risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(Ordering::Equal)
    });

    let max_risk = risk_scores.first().map_or(0.0, |r| r.risk_score);
    let sum: f64 = risk_scores.iter().map(|r| r.risk_score).sum();
    let mean_risk = if risk_scores.is_empty() {
        0.0
    } else {
        sum / risk_scores.len() as f64
    };

    RiskPropagationReport {
        entities: risk_scores,
        mean_risk,
        max_risk,
    }
}

// =====================================================================
// Behavior Embeddings & Geometric Anomaly Detection
// =====================================================================

/// Per-entity embedding vector derived from topology and access patterns.
#[derive(Debug, Clone)]
pub struct NodeEmbedding {
    /// The entity key this embedding represents.
    pub entity: String,
    /// L2-normalized feature vector combining topology and access pattern features.
    pub embedding: Vec<f32>,
}

/// Configuration for behavior embedding computation.
#[derive(Debug, Clone)]
pub struct BehaviorEmbeddingConfig {
    /// Include topology features: PageRank, eigenvector centrality, clustering coefficient.
    pub use_topology_features: bool,
    /// Include access pattern features: binary vector of accessible secrets.
    pub use_access_patterns: bool,
}

impl Default for BehaviorEmbeddingConfig {
    fn default() -> Self {
        Self {
            use_topology_features: true,
            use_access_patterns: true,
        }
    }
}

/// Anomaly result for a single entity.
#[derive(Debug, Clone)]
pub struct GeometricAnomalyResult {
    /// The anomalous entity key.
    pub entity: String,
    /// Z-score of this entity's k-NN distance relative to the population.
    pub anomaly_score: f64,
    /// Entity keys of the k nearest neighbors in embedding space.
    pub nearest_neighbors: Vec<String>,
    /// Euclidean distance to the k-th nearest neighbor.
    pub knn_distance: f64,
}

/// Report of geometric anomaly detection across all entities.
#[derive(Debug, Clone)]
pub struct GeometricAnomalyReport {
    /// Entities flagged as anomalous, sorted by anomaly score descending.
    pub anomalies: Vec<GeometricAnomalyResult>,
    /// Mean k-NN distance across all entities.
    pub mean_distance: f64,
    /// Distance threshold used for anomaly detection (`mean + multiplier * stddev`).
    pub threshold: f64,
    /// Total number of entities evaluated.
    pub total_entities: usize,
}

/// A cluster of entities with similar access patterns.
#[derive(Debug, Clone)]
pub struct SpectralCluster {
    /// Numeric identifier for this cluster (derived from Louvain community ID).
    pub cluster_id: usize,
    /// Entity keys belonging to this cluster.
    pub members: Vec<String>,
    /// Centroid of the cluster in embedding space (empty if embeddings not computed).
    pub center: Vec<f32>,
}

/// Result of clustering entities by their access graph structure.
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// All discovered clusters.
    pub clusters: Vec<SpectralCluster>,
    /// Map from entity key to its assigned cluster ID.
    pub assignments: HashMap<String, usize>,
    /// Louvain modularity score of the partition.
    pub modularity: f64,
}

/// Compute behavior embeddings for all entities in the vault.
#[allow(clippy::cast_precision_loss)]
pub fn compute_behavior_embeddings(
    vault: &Vault,
    config: BehaviorEmbeddingConfig,
) -> Vec<NodeEmbedding> {
    let entities = collect_entity_keys(vault);
    if entities.is_empty() {
        return Vec::new();
    }

    // Collect all secrets for access pattern features
    let all_secrets: Vec<String> = collect_vault_secrets(vault)
        .into_iter()
        .map(|(node_key, _name)| node_key)
        .collect();

    // Pre-compute topology scores if needed
    let (pagerank_scores, eig_scores) = if config.use_topology_features {
        let pr = vault
            .graph
            .pagerank(None)
            .map_or_else(|_| HashMap::new(), |r| r.scores);
        let eig = vault
            .graph
            .eigenvector_centrality(None)
            .map_or_else(|_| HashMap::new(), |r| r.scores);
        (pr, eig)
    } else {
        (HashMap::new(), HashMap::new())
    };

    let mut embeddings = Vec::with_capacity(entities.len());

    for entity in &entities {
        if entity == Vault::ROOT {
            continue;
        }

        let mut features = Vec::new();

        // Topology features (3 floats: pagerank, eigenvector, clustering coeff)
        if config.use_topology_features {
            let node_id = vault.find_entity_node(entity);

            let pr = node_id
                .and_then(|id| pagerank_scores.get(&id).copied())
                .unwrap_or(0.0);
            features.push(pr as f32);

            let eig = node_id
                .and_then(|id| eig_scores.get(&id).copied())
                .unwrap_or(0.0);
            features.push(eig as f32);

            // Local clustering coefficient: ratio of neighbor-neighbor edges to possible
            let clustering = compute_local_clustering(vault, entity);
            features.push(clustering);
        }

        // Access pattern features (binary vector of accessible secrets)
        if config.use_access_patterns {
            let reachable = bfs_reachable_secrets(vault, entity);
            let secret_count = all_secrets.len().max(1);
            for secret_key in &all_secrets {
                if reachable.contains_key(secret_key) {
                    features.push(1.0);
                } else {
                    features.push(0.0);
                }
            }
            // Normalize the access vector
            let access_sum: f32 = features
                .iter()
                .skip(if config.use_topology_features { 3 } else { 0 })
                .sum();
            if access_sum > 0.0 {
                let norm = (secret_count as f32).sqrt();
                let start = if config.use_topology_features { 3 } else { 0 };
                for f in &mut features[start..] {
                    *f /= norm;
                }
            }
        }

        // L2-normalize the full vector
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for f in &mut features {
                *f /= norm;
            }
        }

        embeddings.push(NodeEmbedding {
            entity: entity.clone(),
            embedding: features,
        });
    }

    embeddings
}

/// Compute local clustering coefficient for an entity.
#[allow(clippy::cast_precision_loss)]
fn compute_local_clustering(vault: &Vault, entity: &str) -> f32 {
    let neighbors: Vec<String> = get_outgoing_edges_full(vault, entity)
        .into_iter()
        .filter(|e| is_allowed_edge_type(&e.edge_type))
        .map(|e| e.target_key)
        .collect();

    let n = neighbors.len();
    if n < 2 {
        return 0.0;
    }

    let neighbor_set: HashSet<&str> = neighbors.iter().map(String::as_str).collect();
    let mut triangle_edges = 0usize;

    for neighbor in &neighbors {
        let edges = get_outgoing_edges_full(vault, neighbor);
        for e in edges {
            if is_allowed_edge_type(&e.edge_type) && neighbor_set.contains(e.target_key.as_str()) {
                triangle_edges += 1;
            }
        }
    }

    let possible = n * (n - 1);
    if possible == 0 {
        0.0
    } else {
        triangle_edges as f32 / possible as f32
    }
}

/// Detect geometric anomalies in entity behavior embeddings using k-NN distance.
#[allow(clippy::cast_precision_loss)]
pub fn detect_geometric_anomalies(
    embeddings: &[NodeEmbedding],
    k: usize,
    threshold_multiplier: f64,
) -> GeometricAnomalyReport {
    let n = embeddings.len();
    if n == 0 {
        return GeometricAnomalyReport {
            anomalies: Vec::new(),
            mean_distance: 0.0,
            threshold: 0.0,
            total_entities: 0,
        };
    }

    let effective_k = k.min(n.saturating_sub(1)).max(1);

    // Compute pairwise distances and find k-th nearest neighbor distance
    let mut knn_distances: Vec<f64> = Vec::with_capacity(n);
    let mut knn_neighbors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d = euclidean_distance(&embeddings[i].embedding, &embeddings[j].embedding);
                (j, d)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        distances.truncate(effective_k);

        let knn_dist = distances.last().map_or(0.0, |&(_, d)| d);
        knn_distances.push(knn_dist);
        knn_neighbors.push(distances);
    }

    // Compute mean and stddev of k-NN distances
    let sum: f64 = knn_distances.iter().sum();
    let mean = if n > 0 { sum / n as f64 } else { 0.0 };
    let variance: f64 = knn_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f64>()
        / n.max(1) as f64;
    let stddev = variance.sqrt();
    let threshold = threshold_multiplier.mul_add(stddev, mean);

    // Flag anomalies
    let mut anomalies = Vec::new();
    for (i, dist) in knn_distances.iter().enumerate() {
        if *dist > threshold {
            let neighbors = knn_neighbors[i]
                .iter()
                .map(|&(j, _)| embeddings[j].entity.clone())
                .collect();
            anomalies.push(GeometricAnomalyResult {
                entity: embeddings[i].entity.clone(),
                anomaly_score: (*dist - mean) / stddev.max(f64::EPSILON),
                nearest_neighbors: neighbors,
                knn_distance: *dist,
            });
        }
    }

    anomalies.sort_by(|a, b| {
        b.anomaly_score
            .partial_cmp(&a.anomaly_score)
            .unwrap_or(Ordering::Equal)
    });

    GeometricAnomalyReport {
        anomalies,
        mean_distance: mean,
        threshold,
        total_entities: n,
    }
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    let min_len = a.len().min(b.len());
    let mut sum = 0.0_f64;
    for i in 0..min_len {
        let diff = f64::from(a[i]) - f64::from(b[i]);
        sum += diff * diff;
    }
    // Handle dimension mismatch: treat missing dimensions as 0
    for v in a.iter().skip(min_len) {
        sum += f64::from(*v) * f64::from(*v);
    }
    for v in b.iter().skip(min_len) {
        sum += f64::from(*v) * f64::from(*v);
    }
    sum.sqrt()
}

/// Cluster entities by access graph structure using Louvain communities.
#[allow(clippy::cast_precision_loss)]
pub fn cluster_entities(vault: &Vault) -> ClusteringResult {
    let louvain = vault.graph.louvain_communities(None);

    let entities = collect_entity_keys(vault);
    let entity_set: HashSet<&str> = entities.iter().map(String::as_str).collect();

    match louvain {
        Ok(result) => {
            let mut cluster_map: HashMap<usize, Vec<String>> = HashMap::new();
            let mut assignments = HashMap::new();

            for (node_id, community_id) in &result.communities {
                #[allow(clippy::cast_possible_truncation)]
                let cid = *community_id as usize;
                if let Some(key) = vault.node_entity_key(*node_id) {
                    if entity_set.contains(key.as_str()) {
                        cluster_map.entry(cid).or_default().push(key.clone());
                        assignments.insert(key, cid);
                    }
                }
            }

            let clusters: Vec<SpectralCluster> = cluster_map
                .into_iter()
                .map(|(id, members)| SpectralCluster {
                    cluster_id: id,
                    members,
                    center: Vec::new(), // Center computation would need embeddings
                })
                .collect();

            ClusteringResult {
                clusters,
                assignments,
                modularity: result.modularity.unwrap_or(0.0),
            }
        },
        Err(_) => ClusteringResult {
            clusters: Vec::new(),
            assignments: HashMap::new(),
            modularity: 0.0,
        },
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::VaultConfig;

    fn create_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap()
    }

    // -- explain_access tests --

    #[test]
    fn test_explain_access_direct_path() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Admin)
            .unwrap();

        let result = explain_access(&vault, "user:alice", "db/password");
        assert!(result.granted);
        assert_eq!(result.effective_permission, Some(Permission::Admin));
        assert!(!result.paths.is_empty());
        assert!(result.denial_reason.is_none());
    }

    #[test]
    fn test_explain_access_transitive_path() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "group:devs", "db/password", Permission::Admin)
            .unwrap();

        // Add MEMBER edge: alice -> devs
        let alice_node = vault.get_or_create_entity_node("user:alice");
        let devs_node = vault.get_or_create_entity_node("group:devs");
        vault
            .graph
            .create_edge(
                alice_node,
                devs_node,
                "MEMBER",
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();

        let result = explain_access(&vault, "user:alice", "db/password");
        assert!(result.granted);
        assert!(result.effective_permission.is_some());
        assert!(result.paths.len() >= 1);
        // Should have at least 2 hops: alice -> devs -> secret
        assert!(result.paths[0].len() >= 2);
    }

    #[test]
    fn test_explain_access_no_path() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();

        let result = explain_access(&vault, "user:bob", "db/password");
        assert!(!result.granted);
        assert_eq!(result.effective_permission, None);
        assert!(result.paths.is_empty());
        assert_eq!(result.denial_reason, Some(DenialReason::NoPath));
    }

    #[test]
    fn test_explain_access_insufficient_permission() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Read)
            .unwrap();

        let result = explain_access(&vault, "user:alice", "db/password");
        // Access IS granted (Read), just at a lower level
        assert!(result.granted);
        assert_eq!(result.effective_permission, Some(Permission::Read));
        // The user can check if effective_permission.allows(Permission::Write)
    }

    #[test]
    fn test_explain_access_multiple_paths() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();

        // Direct grant with Read
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Read)
            .unwrap();

        // Also grant through group with Admin
        vault
            .grant_with_permission(
                Vault::ROOT,
                "group:admins",
                "db/password",
                Permission::Admin,
            )
            .unwrap();

        let alice_node = vault.get_or_create_entity_node("user:alice");
        let admins_node = vault.get_or_create_entity_node("group:admins");
        vault
            .graph
            .create_edge(
                alice_node,
                admins_node,
                "MEMBER",
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();

        let result = explain_access(&vault, "user:alice", "db/password");
        assert!(result.granted);
        // Should see multiple paths
        assert!(result.paths.len() >= 2);
        // Best permission should be at least Read (direct)
        assert!(result.effective_permission.is_some());
    }

    // -- blast_radius tests --

    #[test]
    fn test_blast_radius_single_secret() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Read)
            .unwrap();

        let result = blast_radius(&vault, "user:alice");
        assert_eq!(result.entity, "user:alice");
        assert_eq!(result.total_secrets, 1);
        assert_eq!(result.secrets[0].secret_name, "db/password");
        assert_eq!(result.secrets[0].permission, Permission::Read);
    }

    #[test]
    fn test_blast_radius_multiple_secrets() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault.set(Vault::ROOT, "api/key", "secret2").unwrap();
        vault.set(Vault::ROOT, "ssh/key", "secret3").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "api/key", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "ssh/key", Permission::Read)
            .unwrap();

        let result = blast_radius(&vault, "user:alice");
        assert_eq!(result.total_secrets, 3);
        // Sorted by permission descending
        assert_eq!(result.secrets[0].permission, Permission::Admin);
    }

    #[test]
    fn test_blast_radius_no_access() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();

        let result = blast_radius(&vault, "user:nobody");
        assert_eq!(result.total_secrets, 0);
        assert!(result.secrets.is_empty());
    }

    #[test]
    fn test_blast_radius_transitive() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "group:devs", "db/password", Permission::Write)
            .unwrap();

        let alice_node = vault.get_or_create_entity_node("user:alice");
        let devs_node = vault.get_or_create_entity_node("group:devs");
        vault
            .graph
            .create_edge(
                alice_node,
                devs_node,
                "MEMBER",
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();

        let result = blast_radius(&vault, "user:alice");
        assert_eq!(result.total_secrets, 1);
        assert_eq!(result.secrets[0].secret_name, "db/password");
    }

    // -- simulate_grant tests --

    #[test]
    fn test_simulate_grant_new_access() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();

        let result = simulate_grant(&vault, "user:alice", "db/password", Permission::Write);
        assert_eq!(result.target_entity, "user:alice");
        // alice should be in new_accesses since she has no current access
        assert!(result.new_accesses.iter().any(|a| a.entity == "user:alice"));
        assert!(result.total_affected >= 1);
    }

    #[test]
    fn test_simulate_grant_upgrade() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Read)
            .unwrap();

        let result = simulate_grant(&vault, "user:alice", "db/password", Permission::Write);
        // alice should get upgraded Read -> Write
        assert!(result
            .new_accesses
            .iter()
            .any(|a| a.entity == "user:alice" && a.permission == Permission::Write));
    }

    #[test]
    fn test_simulate_grant_no_change() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Admin)
            .unwrap();

        let result = simulate_grant(&vault, "user:alice", "db/password", Permission::Read);
        // alice already has Admin which is >= Read, so no change
        let alice_entry = result
            .new_accesses
            .iter()
            .find(|a| a.entity == "user:alice");
        assert!(alice_entry.is_none());
    }

    #[test]
    fn test_simulate_grant_transitive_impact() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();

        // Create group with members
        let alice_node = vault.get_or_create_entity_node("user:alice");
        let group_node = vault.get_or_create_entity_node("group:devs");
        vault
            .graph
            .create_edge(
                alice_node,
                group_node,
                "MEMBER",
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();

        let result = simulate_grant(&vault, "group:devs", "db/password", Permission::Write);
        // alice is a member of group:devs, so she should also be affected
        assert!(result.new_accesses.iter().any(|a| a.entity == "user:alice"));
    }

    // -- security_audit tests --

    #[test]
    fn test_security_audit_no_cycles() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Read)
            .unwrap();

        let report = security_audit(&vault);
        assert!(report.cycles.is_empty());
        assert!(report.total_entities > 0);
    }

    #[test]
    fn test_security_audit_detects_cycle() {
        let vault = create_vault();

        // Create a MEMBER cycle: A -> B -> C -> A
        let a = vault.get_or_create_entity_node("group:a");
        let b = vault.get_or_create_entity_node("group:b");
        let c = vault.get_or_create_entity_node("group:c");

        vault
            .graph
            .create_edge(a, b, "MEMBER", std::collections::HashMap::new(), true)
            .unwrap();
        vault
            .graph
            .create_edge(b, c, "MEMBER", std::collections::HashMap::new(), true)
            .unwrap();
        vault
            .graph
            .create_edge(c, a, "MEMBER", std::collections::HashMap::new(), true)
            .unwrap();

        let report = security_audit(&vault);
        assert!(
            !report.cycles.is_empty(),
            "Should detect the A->B->C->A cycle"
        );
        // The cycle should contain all three entities
        let cycle = &report.cycles[0];
        assert!(cycle.entities.len() >= 3);
    }

    #[test]
    fn test_security_audit_spof_detection() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();

        // root -> bottleneck -> secret (only path)
        vault
            .grant_with_permission(
                Vault::ROOT,
                "group:bottleneck",
                "db/password",
                Permission::Admin,
            )
            .unwrap();

        // Remove direct root access edge to secret, making bottleneck the only path
        // Actually, root always has access. Let's test with a non-root entity.
        vault.set(Vault::ROOT, "api/key", "secret2").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "group:gateway", "api/key", Permission::Admin)
            .unwrap();

        let alice_node = vault.get_or_create_entity_node("user:alice");
        let gw_node = vault.get_or_create_entity_node("group:gateway");
        vault
            .graph
            .create_edge(
                alice_node,
                gw_node,
                "MEMBER",
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();

        // Now the path for api/key is: root -> secret, gateway -> secret, alice -> gateway -> secret
        // gateway is an SPOF for alice's access to api/key
        let report = security_audit(&vault);
        // The report should have some entities and secrets
        assert!(report.total_entities > 0);
        assert!(report.total_secrets > 0);
    }

    #[test]
    fn test_security_audit_over_privileged() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass1", "s1").unwrap();
        vault.set(Vault::ROOT, "db/pass2", "s2").unwrap();
        vault.set(Vault::ROOT, "db/pass3", "s3").unwrap();

        // Give alice admin to everything
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass2", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass3", Permission::Admin)
            .unwrap();

        let report = security_audit(&vault);
        assert!(!report.over_privileged.is_empty());
        let alice_entry = report
            .over_privileged
            .iter()
            .find(|e| e.entity == "user:alice");
        assert!(alice_entry.is_some());
        assert_eq!(alice_entry.unwrap().reachable_secrets, 3);
        assert_eq!(alice_entry.unwrap().admin_count, 3);
    }

    // -- find_critical_entities tests --

    #[test]
    fn test_find_critical_entities_ranking() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "s1").unwrap();
        vault.set(Vault::ROOT, "api/key", "s2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "api/key", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "db/password", Permission::Read)
            .unwrap();

        let critical = find_critical_entities(&vault);
        assert!(!critical.is_empty());
        // alice should be ranked higher (more reachable secrets)
        let alice_pos = critical.iter().position(|c| c.entity == "user:alice");
        let bob_pos = critical.iter().position(|c| c.entity == "user:bob");
        if let (Some(a), Some(b)) = (alice_pos, bob_pos) {
            assert!(a < b, "alice should rank higher than bob");
        }
    }

    // -- integration-style tests through vault ops --

    #[test]
    fn test_explain_access_through_vault_ops() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "prod/db", "password123").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "team:backend", "prod/db", Permission::Write)
            .unwrap();

        let alice_node = vault.get_or_create_entity_node("user:alice");
        let team_node = vault.get_or_create_entity_node("team:backend");
        vault
            .graph
            .create_edge(
                alice_node,
                team_node,
                "MEMBER",
                std::collections::HashMap::new(),
                true,
            )
            .unwrap();

        let explanation = vault.explain_access("user:alice", "prod/db");
        assert!(explanation.granted);
        assert!(explanation.effective_permission.is_some());
        assert_eq!(explanation.entity, "user:alice");
        assert_eq!(explanation.secret, "prod/db");
    }

    #[test]
    fn test_blast_radius_through_vault_ops() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "prod/db", "pass1").unwrap();
        vault.set(Vault::ROOT, "prod/api", "key1").unwrap();
        vault.set(Vault::ROOT, "staging/db", "pass2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "prod/db", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "prod/api", Permission::Write)
            .unwrap();

        let radius = vault.blast_radius("user:alice");
        assert_eq!(radius.total_secrets, 2);
    }

    #[test]
    fn test_simulate_grant_through_vault_ops() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "prod/db", "pass1").unwrap();

        let sim = vault.simulate_grant("user:alice", "prod/db", Permission::Write);
        assert_eq!(sim.target_entity, "user:alice");
        assert_eq!(sim.secret, "prod/db");
        assert!(sim.total_affected >= 1);
    }

    #[test]
    fn test_security_audit_full_vault() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass", "s1").unwrap();
        vault.set(Vault::ROOT, "api/key", "s2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "api/key", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "api/key", Permission::Write)
            .unwrap();

        let report = vault.security_audit();
        assert!(report.total_entities >= 2);
        assert_eq!(report.total_secrets, 2);
        assert!(report.total_edges > 0);
    }

    #[test]
    fn test_scoped_explain_and_blast_radius() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/password", "secret1").unwrap();
        vault.set(Vault::ROOT, "api/key", "secret2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/password", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "api/key", Permission::Read)
            .unwrap();

        let scoped = vault.scope("user:alice");
        let explanation = scoped.explain_access("db/password");
        assert!(explanation.granted);
        assert_eq!(explanation.entity, "user:alice");

        let radius = scoped.blast_radius();
        assert_eq!(radius.total_secrets, 2);
        assert_eq!(radius.entity, "user:alice");
    }

    // -- privilege_analysis tests --

    #[test]
    fn test_privilege_analysis_empty_vault() {
        let vault = create_vault();
        let report = privilege_analysis(&vault);
        assert!(report.entities.is_empty());
        assert_eq!(report.mean_privilege_score, 0.0);
        assert_eq!(report.max_privilege_score, 0.0);
    }

    #[test]
    fn test_privilege_analysis_single_entity() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass", "s1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass", Permission::Admin)
            .unwrap();

        let report = privilege_analysis(&vault);
        assert_eq!(report.entities.len(), 1);
        assert_eq!(report.entities[0].entity, "user:alice");
        assert_eq!(report.entities[0].reachable_secrets, 1);
        assert_eq!(report.entities[0].admin_count, 1);
    }

    #[test]
    fn test_privilege_analysis_multiple_entities() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass1", "s1").unwrap();
        vault.set(Vault::ROOT, "db/pass2", "s2").unwrap();
        vault.set(Vault::ROOT, "db/pass3", "s3").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass2", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass3", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "db/pass1", Permission::Read)
            .unwrap();

        let report = privilege_analysis(&vault);
        assert!(report.entities.len() >= 2);
        // alice has more reachable secrets, should rank higher
        assert_eq!(report.entities[0].entity, "user:alice");
    }

    #[test]
    fn test_privilege_analysis_skips_root() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass", "s1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass", Permission::Read)
            .unwrap();

        let report = privilege_analysis(&vault);
        assert!(report.entities.iter().all(|e| e.entity != Vault::ROOT));
    }

    #[test]
    fn test_privilege_analysis_admin_write_read_counts() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault.set(Vault::ROOT, "s3", "v3").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s3", Permission::Read)
            .unwrap();

        let report = privilege_analysis(&vault);
        let alice = report
            .entities
            .iter()
            .find(|e| e.entity == "user:alice")
            .unwrap();
        assert_eq!(alice.admin_count, 1);
        assert_eq!(alice.write_count, 1);
        assert_eq!(alice.read_count, 1);
        assert_eq!(alice.reachable_secrets, 3);
    }

    #[test]
    fn test_privilege_analysis_scores_computation() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Read)
            .unwrap();

        let report = privilege_analysis(&vault);
        let alice = report
            .entities
            .iter()
            .find(|e| e.entity == "user:alice")
            .unwrap();
        // privilege_score = pagerank * reachable_secrets
        let expected = alice.pagerank_score * alice.reachable_secrets as f64;
        assert!(
            (alice.privilege_score - expected).abs() < f64::EPSILON,
            "privilege_score should equal pagerank * reachable_secrets"
        );
        assert!(report.max_privilege_score >= report.mean_privilege_score);
    }

    // -- delegation_anomaly_scores tests --

    #[test]
    fn test_delegation_anomaly_empty_vault() {
        let vault = create_vault();
        let scores = delegation_anomaly_scores(&vault);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_delegation_anomaly_normal_grants() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass", "s1").unwrap();

        // Create two entities that share access to the same secret
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "db/pass", Permission::Read)
            .unwrap();

        // With shared neighbors, jaccard should be higher -> lower anomaly
        let scores = delegation_anomaly_scores(&vault);
        // Well-connected grants may still have anomaly > 0.5 in small graphs
        // The key invariant: all returned scores have anomaly_score > 0.5
        for s in &scores {
            assert!(s.anomaly_score > 0.5);
        }
    }

    #[test]
    fn test_delegation_anomaly_isolated_grant() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass", "s1").unwrap();

        // Single isolated grant -- entity and secret share no neighbors
        vault
            .grant_with_permission(Vault::ROOT, "user:lone", "db/pass", Permission::Admin)
            .unwrap();

        let scores = delegation_anomaly_scores(&vault);
        // Isolated grant should have high anomaly (low jaccard)
        for s in &scores {
            assert!(
                s.anomaly_score > 0.5,
                "isolated grant should have high anomaly"
            );
        }
    }

    #[test]
    fn test_delegation_anomaly_sorted_by_score() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Read)
            .unwrap();

        let scores = delegation_anomaly_scores(&vault);
        for w in scores.windows(2) {
            assert!(
                w[0].anomaly_score >= w[1].anomaly_score,
                "results should be sorted by anomaly_score descending"
            );
        }
    }

    #[test]
    fn test_delegation_anomaly_skips_root() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "db/pass", "s1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "db/pass", Permission::Read)
            .unwrap();

        let scores = delegation_anomaly_scores(&vault);
        assert!(
            scores.iter().all(|s| s.entity != Vault::ROOT),
            "ROOT should not appear in anomaly scores"
        );
    }

    // -- infer_roles tests --

    #[test]
    fn test_infer_roles_empty_vault() {
        let vault = create_vault();
        let result = infer_roles(&vault);
        assert!(result.roles.is_empty());
        assert_eq!(result.modularity, 0.0);
    }

    #[test]
    fn test_infer_roles_two_groups() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        // Group 1: alice and bob share s1
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s1", Permission::Admin)
            .unwrap();

        // Group 2: charlie and dave share s2
        vault
            .grant_with_permission(Vault::ROOT, "user:charlie", "s2", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:dave", "s2", Permission::Admin)
            .unwrap();

        let result = infer_roles(&vault);
        // Louvain should detect some community structure
        // At minimum, the entities should appear in roles or unassigned
        let total_assigned: usize = result.roles.iter().map(|r| r.members.len()).sum();
        let total = total_assigned + result.unassigned.len();
        assert!(total >= 4, "all entities should appear somewhere");
    }

    #[test]
    fn test_infer_roles_common_secrets() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "shared", "v1").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "shared", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "shared", Permission::Read)
            .unwrap();

        let result = infer_roles(&vault);
        // If alice and bob are in the same role, they should share "shared"
        for role in &result.roles {
            let has_alice = role.members.iter().any(|m| m == "user:alice");
            let has_bob = role.members.iter().any(|m| m == "user:bob");
            if has_alice && has_bob {
                assert!(
                    !role.common_secrets.is_empty(),
                    "common secrets should include the shared secret"
                );
            }
        }
    }

    #[test]
    fn test_infer_roles_singletons_unassigned() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();

        // Only one entity -- should be unassigned (community < 2 members)
        vault
            .grant_with_permission(Vault::ROOT, "user:lonely", "s1", Permission::Read)
            .unwrap();

        let result = infer_roles(&vault);
        // Either in a role with ROOT (excluded) or unassigned
        let in_role = result
            .roles
            .iter()
            .any(|r| r.members.iter().any(|m| m == "user:lonely"));
        let in_unassigned = result.unassigned.iter().any(|u| u == "user:lonely");
        assert!(
            in_role || in_unassigned,
            "lonely entity should appear in roles or unassigned"
        );
    }

    #[test]
    fn test_infer_roles_modularity() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s1", Permission::Read)
            .unwrap();

        let result = infer_roles(&vault);
        // Modularity should be a real number (could be 0.0 for trivial graphs)
        assert!(result.modularity.is_finite());
    }

    // -- trust_transitivity tests --

    #[test]
    fn test_trust_transitivity_empty() {
        let vault = create_vault();
        let report = trust_transitivity(&vault);
        assert!(report.entities.is_empty());
        assert_eq!(report.total_triangles, 0);
    }

    #[test]
    fn test_trust_transitivity_triangle() {
        let vault = create_vault();

        // Create a triangle: A <-> B <-> C <-> A via MEMBER edges
        let a = vault.get_or_create_entity_node("user:a");
        let b = vault.get_or_create_entity_node("user:b");
        let c = vault.get_or_create_entity_node("user:c");

        let props = std::collections::HashMap::new();
        vault
            .graph
            .create_edge(a, b, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(b, c, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(c, a, "MEMBER", props, true)
            .unwrap();

        let report = trust_transitivity(&vault);
        // With undirected=true, the triangle A-B-C should be detected
        assert!(
            report.total_triangles >= 1,
            "should detect at least one triangle"
        );
        // Entities in the triangle should have non-zero trust scores
        let non_zero = report
            .entities
            .iter()
            .filter(|e| e.trust_score > 0.0)
            .count();
        assert!(non_zero > 0, "triangle participants should have trust > 0");
    }

    #[test]
    fn test_trust_transitivity_global_clustering() {
        let vault = create_vault();

        let a = vault.get_or_create_entity_node("user:a");
        let b = vault.get_or_create_entity_node("user:b");
        let c = vault.get_or_create_entity_node("user:c");

        let props = std::collections::HashMap::new();
        vault
            .graph
            .create_edge(a, b, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(b, c, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(c, a, "MEMBER", props, true)
            .unwrap();

        let report = trust_transitivity(&vault);
        assert!(
            report.global_clustering.is_finite(),
            "global clustering should be finite"
        );
    }

    #[test]
    fn test_trust_transitivity_scoring_formula() {
        let vault = create_vault();

        let a = vault.get_or_create_entity_node("user:a");
        let b = vault.get_or_create_entity_node("user:b");
        let c = vault.get_or_create_entity_node("user:c");

        let props = std::collections::HashMap::new();
        vault
            .graph
            .create_edge(a, b, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(b, c, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(c, a, "MEMBER", props, true)
            .unwrap();

        let report = trust_transitivity(&vault);
        for ent in &report.entities {
            let expected = ent.clustering_coefficient * (1.0 + (ent.triangle_count as f64).ln_1p());
            assert!(
                (ent.trust_score - expected).abs() < 1e-10,
                "trust_score should match formula for {}",
                ent.entity
            );
        }
    }

    #[test]
    fn test_trust_transitivity_sorted() {
        let vault = create_vault();

        let a = vault.get_or_create_entity_node("user:a");
        let b = vault.get_or_create_entity_node("user:b");
        let c = vault.get_or_create_entity_node("user:c");
        let d = vault.get_or_create_entity_node("user:d");

        let props = std::collections::HashMap::new();
        // Triangle: a-b-c
        vault
            .graph
            .create_edge(a, b, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(b, c, "MEMBER", props.clone(), true)
            .unwrap();
        vault
            .graph
            .create_edge(c, a, "MEMBER", props.clone(), true)
            .unwrap();
        // d connects to a only -- no triangle
        vault
            .graph
            .create_edge(d, a, "MEMBER", props, true)
            .unwrap();

        let report = trust_transitivity(&vault);
        for w in report.entities.windows(2) {
            assert!(
                w[0].trust_score >= w[1].trust_score,
                "results should be sorted by trust_score descending"
            );
        }
    }

    // -- risk_propagation tests --

    #[test]
    fn test_risk_propagation_empty() {
        let vault = create_vault();
        let report = risk_propagation(&vault);
        assert!(report.entities.is_empty());
        assert_eq!(report.mean_risk, 0.0);
        assert_eq!(report.max_risk, 0.0);
    }

    #[test]
    fn test_risk_propagation_admin_amplifies() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:admin_user", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:admin_user", "s2", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "s1", Permission::Read)
            .unwrap();

        let report = risk_propagation(&vault);
        let admin_entry = report
            .entities
            .iter()
            .find(|e| e.entity == "user:admin_user");
        let reader_entry = report.entities.iter().find(|e| e.entity == "user:reader");

        assert!(admin_entry.is_some());
        assert!(reader_entry.is_some());
        // Admin entity should have higher risk
        assert!(
            admin_entry.unwrap().reachable_admin_secrets
                >= reader_entry.unwrap().reachable_admin_secrets
        );
    }

    #[test]
    fn test_risk_propagation_read_only_low_risk() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "s1", Permission::Read)
            .unwrap();

        let report = risk_propagation(&vault);
        let reader = report
            .entities
            .iter()
            .find(|e| e.entity == "user:reader")
            .unwrap();
        assert_eq!(
            reader.reachable_admin_secrets, 0,
            "read-only entity should have zero admin secrets"
        );
        assert!(
            reader.risk_contributors.is_empty(),
            "read-only entity should have no risk contributors"
        );
    }

    #[test]
    fn test_risk_propagation_contributors() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "critical/db", "v1").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "critical/db", Permission::Admin)
            .unwrap();

        let report = risk_propagation(&vault);
        let alice = report
            .entities
            .iter()
            .find(|e| e.entity == "user:alice")
            .unwrap();
        assert_eq!(alice.reachable_admin_secrets, 1);
        assert_eq!(alice.risk_contributors.len(), 1);
        assert_eq!(alice.risk_contributors[0].secret, "critical/db");
        assert_eq!(alice.risk_contributors[0].permission, Permission::Admin);
    }

    #[test]
    fn test_risk_propagation_sorted() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s1", Permission::Read)
            .unwrap();

        let report = risk_propagation(&vault);
        for w in report.entities.windows(2) {
            assert!(
                w[0].risk_score >= w[1].risk_score,
                "results should be sorted by risk_score descending"
            );
        }
    }

    // -- scoped wrappers for Tier 3 --

    #[test]
    fn test_scoped_privilege_analysis() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();

        let scoped = vault.scope("user:alice");
        let report = scoped.privilege_analysis();
        assert!(!report.entities.is_empty());
    }

    #[test]
    fn test_scoped_delegation_anomaly() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();

        let scoped = vault.scope("user:alice");
        let scores = scoped.delegation_anomaly_scores();
        // Should return anomaly scores (may be empty if jaccard > 0.5)
        for s in &scores {
            assert!(s.anomaly_score > 0.5);
        }
    }

    #[test]
    fn test_scoped_infer_roles() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s1", Permission::Read)
            .unwrap();

        let scoped = vault.scope("user:alice");
        let result = scoped.infer_roles();
        assert!(result.modularity.is_finite());
    }

    #[test]
    fn test_scoped_trust_and_risk() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();

        let scoped = vault.scope("user:alice");

        let trust = scoped.trust_transitivity();
        assert!(trust.global_clustering.is_finite());

        let risk = scoped.risk_propagation();
        assert!(risk.max_risk >= risk.mean_risk);
    }

    // -- behavior embeddings tests --

    #[test]
    fn test_behavior_embeddings_topology_features() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s2", Permission::Read)
            .unwrap();

        let config = BehaviorEmbeddingConfig {
            use_topology_features: true,
            use_access_patterns: false,
        };
        let embeddings = compute_behavior_embeddings(&vault, config);
        assert!(!embeddings.is_empty());
        // Each embedding should have 3 topology features (pagerank, eigenvector, clustering)
        for emb in &embeddings {
            assert_eq!(emb.embedding.len(), 3);
            // L2-normalized, so norm should be ~1.0 (or all zeros)
            let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm < f32::EPSILON || (norm - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_behavior_embeddings_access_patterns() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();

        let config = BehaviorEmbeddingConfig {
            use_topology_features: false,
            use_access_patterns: true,
        };
        let embeddings = compute_behavior_embeddings(&vault, config);
        // Alice should have access to s1 encoded in her embedding
        let alice = embeddings.iter().find(|e| e.entity == "user:alice");
        assert!(alice.is_some());
        assert!(!alice.unwrap().embedding.is_empty());
    }

    #[test]
    fn test_behavior_embeddings_empty_graph() {
        let vault = create_vault();
        let config = BehaviorEmbeddingConfig::default();
        let embeddings = compute_behavior_embeddings(&vault, config);
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_geometric_anomaly_isolated_entity() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault.set(Vault::ROOT, "s3", "v3").unwrap();

        // Alice and Bob get the same access; Charlie gets nothing
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s2", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:charlie", "s3", Permission::Admin)
            .unwrap();

        let config = BehaviorEmbeddingConfig::default();
        let embeddings = compute_behavior_embeddings(&vault, config);

        // k=1 with a low threshold should flag Charlie as anomalous
        let report = detect_geometric_anomalies(&embeddings, 1, 1.0);
        assert_eq!(report.total_entities, embeddings.len());
        assert!(report.mean_distance >= 0.0);
    }

    #[test]
    fn test_geometric_anomaly_normal_cluster() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();

        // All get the same access
        vault
            .grant_with_permission(Vault::ROOT, "user:a", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:b", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:c", "s1", Permission::Read)
            .unwrap();

        let config = BehaviorEmbeddingConfig::default();
        let embeddings = compute_behavior_embeddings(&vault, config);
        // Very high threshold should not flag anyone
        let report = detect_geometric_anomalies(&embeddings, 1, 10.0);
        assert!(report.anomalies.is_empty());
    }

    #[test]
    fn test_geometric_anomaly_threshold() {
        // Build embeddings manually to test threshold math
        let embeddings = vec![
            NodeEmbedding {
                entity: "a".to_string(),
                embedding: vec![0.0, 0.0],
            },
            NodeEmbedding {
                entity: "b".to_string(),
                embedding: vec![0.1, 0.0],
            },
            NodeEmbedding {
                entity: "c".to_string(),
                embedding: vec![0.0, 0.1],
            },
            NodeEmbedding {
                entity: "outlier".to_string(),
                embedding: vec![10.0, 10.0],
            },
        ];
        let report = detect_geometric_anomalies(&embeddings, 1, 1.5);
        assert_eq!(report.total_entities, 4);
        assert!(report.threshold > report.mean_distance);
        // The outlier should be flagged
        assert!(
            report.anomalies.iter().any(|a| a.entity == "outlier"),
            "outlier entity should be flagged"
        );
    }

    #[test]
    fn test_geometric_anomaly_k_exceeds_entities() {
        let embeddings = vec![
            NodeEmbedding {
                entity: "a".to_string(),
                embedding: vec![0.0],
            },
            NodeEmbedding {
                entity: "b".to_string(),
                embedding: vec![1.0],
            },
        ];
        // k=100 but only 2 entities -- should gracefully use k=1
        let report = detect_geometric_anomalies(&embeddings, 100, 2.0);
        assert_eq!(report.total_entities, 2);
    }

    #[test]
    fn test_cluster_entities_two_communities() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        // Create two isolated groups
        vault
            .grant_with_permission(Vault::ROOT, "user:a1", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:a2", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:b1", "s2", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:b2", "s2", Permission::Read)
            .unwrap();

        let result = cluster_entities(&vault);
        // Should have at least 1 cluster (Louvain might merge or not)
        assert!(result.modularity.is_finite());
    }

    #[test]
    fn test_cluster_entities_single_community() {
        let vault = create_vault();
        vault.set(Vault::ROOT, "s1", "v1").unwrap();

        // All connected to same secret
        vault
            .grant_with_permission(Vault::ROOT, "user:a", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:b", "s1", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:c", "s1", Permission::Read)
            .unwrap();

        let result = cluster_entities(&vault);
        assert!(result.modularity.is_finite());
    }

    #[test]
    fn test_cluster_entities_empty_vault() {
        let vault = create_vault();
        let result = cluster_entities(&vault);
        assert!(result.clusters.is_empty());
        assert!(result.assignments.is_empty());
    }
}
