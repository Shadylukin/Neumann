//! CSR-based graph storage with append log.
//!
//! `GraphTensor` stores graph edges using Compressed Sparse Row (CSR) format
//! for efficient traversal, combined with an append log for O(1) edge insertion.
//! Background merging combines pending edges into the CSR structure.
//!
//! # Design Philosophy
//!
//! - Writes go to append log: O(1)
//! - Reads check both CSR and pending log
//! - Background merge rebuilds CSR with pending edges
//! - No resize stalls from hash table growth

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
};

use bitvec::prelude::*;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::{entity_index::EntityId, metadata_slab::MetadataSlab, TensorData};

/// Edge identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

impl EdgeId {
    /// Creates a new edge ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying u64 value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for EdgeId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

/// Interned edge type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct EdgeTypeId(u32);

impl EdgeTypeId {
    /// Create from array index. Edge type count is practically limited.
    #[allow(clippy::cast_possible_truncation)]
    const fn from_index(idx: usize) -> Self {
        Self(idx as u32)
    }
}

/// Interned edge type registry.
///
/// Consolidates edge type strings and their IDs into a single structure
/// to reduce lock contention (one lock instead of two).
#[derive(Debug, Clone, Default)]
struct EdgeTypeRegistry {
    types: Vec<String>,
    ids: HashMap<String, EdgeTypeId>,
}

impl EdgeTypeRegistry {
    fn new() -> Self {
        let mut registry = Self::default();
        registry.types.push("default".to_string());
        registry.ids.insert("default".to_string(), EdgeTypeId(0));
        registry
    }

    fn intern(&mut self, edge_type: &str) -> EdgeTypeId {
        if let Some(&id) = self.ids.get(edge_type) {
            return id;
        }
        let id = EdgeTypeId::from_index(self.types.len());
        self.types.push(edge_type.to_string());
        self.ids.insert(edge_type.to_string(), id);
        id
    }

    fn get_id(&self, edge_type: &str) -> EdgeTypeId {
        self.ids
            .get(edge_type)
            .copied()
            .unwrap_or(EdgeTypeId(u32::MAX))
    }
}

/// A pending edge entry in the append log.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeEntry {
    edge_id: EdgeId,
    from: EntityId,
    to: EntityId,
    edge_type: EdgeTypeId,
    directed: bool,
}

/// Compressed Sparse Row graph representation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CsrGraph {
    /// Row pointers: node i's outgoing edges start at `row_ptr[i]` and end at `row_ptr[i+1]`.
    row_ptr: Vec<u32>,

    /// Column indices: target nodes for each edge.
    col_idx: Vec<EntityId>,

    /// Edge IDs for each edge (parallel to `col_idx`).
    edge_ids: Vec<EdgeId>,

    /// Edge types for each edge (parallel to `col_idx`).
    edge_types: Vec<EdgeTypeId>,

    /// Direction flags packed as bits (true = directed).
    directions: BitVec,

    /// Maximum node ID in the graph (for sizing `row_ptr`).
    max_node_id: u64,
}

impl CsrGraph {
    fn new() -> Self {
        Self::default()
    }

    /// Build a CSR graph from a list of edges.
    fn build(edges: &[EdgeEntry], max_node_id: u64) -> Self {
        if edges.is_empty() {
            return Self::new();
        }

        // Count outgoing edges per node
        #[allow(clippy::cast_possible_truncation)] // Node count bounded by memory
        let num_nodes = (max_node_id + 1) as usize;
        let mut out_degree = vec![0u32; num_nodes];
        for edge in edges {
            let from_idx = edge.from.as_index();
            if from_idx < num_nodes {
                out_degree[from_idx] += 1;
            }
        }

        // Build row_ptr (prefix sum)
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        row_ptr.push(0);
        for &deg in &out_degree {
            row_ptr.push(row_ptr.last().unwrap() + deg);
        }

        let total_edges = *row_ptr.last().unwrap() as usize;
        let mut col_idx = vec![EntityId::new(0); total_edges];
        let mut edge_ids = vec![EdgeId::new(0); total_edges];
        let mut edge_types = vec![EdgeTypeId(0); total_edges];
        let mut directions = bitvec![0; total_edges];

        // Fill edges
        let mut write_pos = vec![0u32; num_nodes];
        for edge in edges {
            let from_idx = edge.from.as_index();
            if from_idx < num_nodes {
                let pos = (row_ptr[from_idx] + write_pos[from_idx]) as usize;
                col_idx[pos] = edge.to;
                edge_ids[pos] = edge.edge_id;
                edge_types[pos] = edge.edge_type;
                directions.set(pos, edge.directed);
                write_pos[from_idx] += 1;
            }
        }

        Self {
            row_ptr,
            col_idx,
            edge_ids,
            edge_types,
            directions,
            max_node_id,
        }
    }

    /// Get outgoing edges for a node.
    fn outgoing(&self, node: EntityId) -> Vec<(EntityId, EdgeId)> {
        let node_idx = node.as_index();
        if node_idx >= self.row_ptr.len().saturating_sub(1) {
            return Vec::new();
        }

        let start = self.row_ptr[node_idx] as usize;
        let end = self.row_ptr[node_idx + 1] as usize;

        (start..end)
            .map(|i| (self.col_idx[i], self.edge_ids[i]))
            .collect()
    }

    const fn edge_count(&self) -> usize {
        self.col_idx.len()
    }

    const fn node_count(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }
}

/// Graph storage with CSR format and append log.
///
/// # Thread Safety
///
/// Uses `parking_lot` locks for concurrent access. The append log allows
/// O(1) edge insertion without blocking reads on the CSR structure.
///
/// ## Lock Ordering (prevents deadlock)
///
/// When multiple locks are needed, acquire in this order:
/// 1. `merge_lock` (outermost, exclusive merge coordination)
/// 2. `edge_types` (edge type registry)
/// 3. `incoming_index` (incoming edge index)
/// 4. `csr` (main CSR structure)
/// 5. `pending` (pending edge log)
/// 6. `deleted` (deleted edge set, innermost)
///
/// Most operations only need 1-2 locks. The `merge()` operation needs
/// the `merge_lock` to ensure atomicity.
pub struct GraphTensor {
    /// Merge lock to prevent concurrent merges and ensure atomic state transitions.
    merge_lock: Mutex<()>,

    /// Immutable CSR structure (rebuilt on merge).
    csr: RwLock<CsrGraph>,

    /// Pending edges not yet in CSR.
    pending: Mutex<Vec<EdgeEntry>>,

    /// Deleted edge IDs (actual removal on rebuild).
    deleted: Mutex<BTreeSet<EdgeId>>,

    /// Edge metadata (properties).
    edge_data: MetadataSlab,

    /// Edge type string interning (consolidated registry).
    edge_types: RwLock<EdgeTypeRegistry>,

    /// Next edge ID.
    next_edge_id: AtomicU64,

    /// Maximum node ID seen.
    max_node_id: AtomicU64,

    /// Merge threshold (number of pending edges before auto-merge).
    merge_threshold: usize,

    /// Incoming edge index (for reverse traversal).
    incoming_index: RwLock<BTreeMap<EntityId, Vec<(EntityId, EdgeId)>>>,
}

impl GraphTensor {
    /// Create a new `GraphTensor`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_merge_threshold(10_000)
    }

    /// Create a `GraphTensor` with a custom merge threshold.
    #[must_use]
    pub fn with_merge_threshold(threshold: usize) -> Self {
        Self {
            merge_lock: Mutex::new(()),
            csr: RwLock::new(CsrGraph::new()),
            pending: Mutex::new(Vec::new()),
            deleted: Mutex::new(BTreeSet::new()),
            edge_data: MetadataSlab::new(),
            edge_types: RwLock::new(EdgeTypeRegistry::new()),
            next_edge_id: AtomicU64::new(0),
            max_node_id: AtomicU64::new(0),
            merge_threshold: threshold,
            incoming_index: RwLock::new(BTreeMap::new()),
        }
    }

    /// Add an edge to the graph.
    ///
    /// Returns the new `EdgeId`. The edge is added to the pending log
    /// and will be merged into the CSR on the next merge.
    pub fn add_edge(
        &self,
        from: EntityId,
        to: EntityId,
        edge_type: &str,
        directed: bool,
    ) -> EdgeId {
        let edge_id = EdgeId::new(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let edge_type_id = self.intern_edge_type(edge_type);

        // Update max node ID
        self.update_max_node_id(from);
        self.update_max_node_id(to);

        // Add to pending log
        {
            let entry = EdgeEntry {
                edge_id,
                from,
                to,
                edge_type: edge_type_id,
                directed,
            };
            self.pending.lock().push(entry);
        }

        // Update incoming index
        {
            let mut incoming = self.incoming_index.write();
            incoming.entry(to).or_default().push((from, edge_id));
        }

        // Auto-merge if threshold reached
        if self.pending.lock().len() >= self.merge_threshold {
            self.merge();
        }

        edge_id
    }

    /// Get outgoing edges from a node.
    ///
    /// Checks both CSR and pending edges.
    pub fn outgoing(&self, node: EntityId) -> Vec<(EntityId, EdgeId)> {
        let deleted = self.deleted.lock();
        let mut result = Vec::new();

        // Check CSR
        {
            let csr = self.csr.read();
            for (to, edge_id) in csr.outgoing(node) {
                if !deleted.contains(&edge_id) {
                    result.push((to, edge_id));
                }
            }
        }

        // Check pending
        {
            let pending = self.pending.lock();
            for entry in pending.iter() {
                if entry.from == node && !deleted.contains(&entry.edge_id) {
                    result.push((entry.to, entry.edge_id));
                }
            }
        }

        result
    }

    /// Get incoming edges to a node.
    pub fn incoming(&self, node: EntityId) -> Vec<(EntityId, EdgeId)> {
        let deleted = self.deleted.lock();
        let incoming = self.incoming_index.read();

        incoming
            .get(&node)
            .map(|edges| {
                edges
                    .iter()
                    .filter(|(_, edge_id)| !deleted.contains(edge_id))
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if an edge exists between two nodes.
    pub fn edge_exists(&self, from: EntityId, to: EntityId, edge_type: Option<&str>) -> bool {
        let deleted = self.deleted.lock();
        let edge_type_id = edge_type.map(|t| self.get_edge_type_id(t));

        // Check CSR
        {
            let csr = self.csr.read();
            let node_idx = from.as_index();
            if node_idx < csr.row_ptr.len().saturating_sub(1) {
                let start = csr.row_ptr[node_idx] as usize;
                let end = csr.row_ptr[node_idx + 1] as usize;

                for i in start..end {
                    if csr.col_idx[i] == to && !deleted.contains(&csr.edge_ids[i]) {
                        if let Some(type_id) = edge_type_id {
                            if csr.edge_types[i] == type_id {
                                return true;
                            }
                        } else {
                            return true;
                        }
                    }
                }
            }
        }

        // Check pending
        {
            let pending = self.pending.lock();
            for entry in pending.iter() {
                if entry.from == from && entry.to == to && !deleted.contains(&entry.edge_id) {
                    if let Some(type_id) = edge_type_id {
                        if entry.edge_type == type_id {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Delete an edge by ID.
    ///
    /// The edge is marked as deleted; actual removal happens on merge.
    pub fn delete_edge(&self, edge_id: EdgeId) -> bool {
        // Check if edge exists in CSR or pending
        let exists = {
            let csr = self.csr.read();
            csr.edge_ids.contains(&edge_id)
        } || {
            let pending = self.pending.lock();
            pending.iter().any(|e| e.edge_id == edge_id)
        };

        if exists {
            self.deleted.lock().insert(edge_id);
            true
        } else {
            false
        }
    }

    /// Merge pending edges into CSR.
    ///
    /// This rebuilds the CSR structure with all non-deleted edges.
    /// Uses `merge_lock` to ensure atomic state transitions and prevent
    /// race conditions where edges could be deleted after snapshot
    /// but before CSR rebuild.
    pub fn merge(&self) {
        // Acquire merge lock first - prevents concurrent merges and
        // ensures no modifications to pending/deleted during rebuild
        let _merge_guard = self.merge_lock.lock();

        // Atomically take both pending and deleted
        // This prevents race where edge is deleted after snapshot but before clear
        let pending: Vec<EdgeEntry> = self.pending.lock().drain(..).collect();
        let deleted: BTreeSet<EdgeId> = std::mem::take(&mut *self.deleted.lock());

        if pending.is_empty() && deleted.is_empty() {
            return;
        }

        // Collect all edges from CSR
        let mut all_edges: Vec<EdgeEntry> = {
            let csr = self.csr.read();
            let mut edges = Vec::with_capacity(csr.edge_count());

            for node_idx in 0..csr.node_count() {
                let start = csr.row_ptr[node_idx] as usize;
                let end = csr.row_ptr[node_idx + 1] as usize;

                for i in start..end {
                    if !deleted.contains(&csr.edge_ids[i]) {
                        edges.push(EdgeEntry {
                            edge_id: csr.edge_ids[i],
                            from: EntityId::new(node_idx as u64),
                            to: csr.col_idx[i],
                            edge_type: csr.edge_types[i],
                            directed: csr.directions[i],
                        });
                    }
                }
            }

            edges
        };

        // Add non-deleted pending edges
        for entry in pending {
            if !deleted.contains(&entry.edge_id) {
                all_edges.push(entry);
            }
        }

        // Build new CSR
        let max_node_id = self.max_node_id.load(Ordering::Relaxed);
        let new_csr = CsrGraph::build(&all_edges, max_node_id);

        // Swap in new CSR
        *self.csr.write() = new_csr;
    }

    /// Set edge metadata.
    pub fn set_edge_data(&self, edge_id: EdgeId, data: TensorData) {
        let key = format!("edge:{}", edge_id.as_u64());
        self.edge_data.set(&key, data);
    }

    /// Get edge metadata.
    pub fn get_edge_data(&self, edge_id: EdgeId) -> Option<TensorData> {
        let key = format!("edge:{}", edge_id.as_u64());
        self.edge_data.get(&key)
    }

    /// BFS traversal from a starting node.
    pub fn bfs(&self, start: EntityId, max_depth: usize) -> Vec<EntityId> {
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(start);
        queue.push_back((start, 0));

        while let Some((node, depth)) = queue.pop_front() {
            result.push(node);

            if depth >= max_depth {
                continue;
            }

            for (neighbor, _) in self.outgoing(node) {
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        result
    }

    /// Shortest path between two nodes using BFS.
    pub fn shortest_path(&self, from: EntityId, to: EntityId) -> Option<Vec<EntityId>> {
        if from == to {
            return Some(vec![from]);
        }

        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        let mut parent: BTreeMap<EntityId, EntityId> = BTreeMap::new();

        visited.insert(from);
        queue.push_back(from);

        while let Some(node) = queue.pop_front() {
            for (neighbor, _) in self.outgoing(node) {
                if visited.insert(neighbor) {
                    parent.insert(neighbor, node);

                    if neighbor == to {
                        // Reconstruct path
                        let mut path = vec![to];
                        let mut current = to;
                        while let Some(&p) = parent.get(&current) {
                            path.push(p);
                            current = p;
                        }
                        path.reverse();
                        return Some(path);
                    }

                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Get the total number of edges (including pending, excluding deleted).
    pub fn edge_count(&self) -> usize {
        let deleted = self.deleted.lock().clone();

        let csr_count = {
            let csr = self.csr.read();
            csr.edge_ids
                .iter()
                .filter(|id| !deleted.contains(id))
                .count()
        };

        let pending_count = {
            let pending = self.pending.lock();
            pending
                .iter()
                .filter(|e| !deleted.contains(&e.edge_id))
                .count()
        };

        csr_count + pending_count
    }

    /// Get the number of pending edges.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Clear all edges.
    pub fn clear(&self) {
        *self.csr.write() = CsrGraph::new();
        self.pending.lock().clear();
        self.deleted.lock().clear();
        self.incoming_index.write().clear();
        self.edge_data.clear();
        self.next_edge_id.store(0, Ordering::Relaxed);
        self.max_node_id.store(0, Ordering::Relaxed);
    }

    /// Get serializable state for snapshots.
    pub fn snapshot(&self) -> GraphTensorSnapshot {
        // Force merge before snapshot
        self.merge();

        let edge_types = self.edge_types.read().types.clone();

        // Collect all edges
        let edges = {
            let csr = self.csr.read();
            let mut edges = Vec::new();
            for node_idx in 0..csr.node_count() {
                let start = csr.row_ptr[node_idx] as usize;
                let end = csr.row_ptr[node_idx + 1] as usize;

                for i in start..end {
                    edges.push(GraphEdgeSnapshot {
                        edge_id: csr.edge_ids[i],
                        from: EntityId::new(node_idx as u64),
                        to: csr.col_idx[i],
                        edge_type_idx: csr.edge_types[i].0,
                        directed: csr.directions[i],
                    });
                }
            }
            edges
        };

        GraphTensorSnapshot {
            edges,
            edge_types,
            next_edge_id: self.next_edge_id.load(Ordering::Relaxed),
            max_node_id: self.max_node_id.load(Ordering::Relaxed),
            edge_data: self.edge_data.snapshot(),
        }
    }

    /// Restore from a snapshot.
    #[must_use]
    pub fn restore(snapshot: GraphTensorSnapshot) -> Self {
        let graph = Self::new();

        // Restore edge types using consolidated registry
        {
            let mut registry = graph.edge_types.write();
            registry.types.clear();
            registry.ids.clear();
            for (idx, type_name) in snapshot.edge_types.iter().enumerate() {
                registry.types.push(type_name.clone());
                registry
                    .ids
                    .insert(type_name.clone(), EdgeTypeId::from_index(idx));
            }
        }

        // Restore edges
        for edge in snapshot.edges {
            let edge_type = &snapshot.edge_types[edge.edge_type_idx as usize];
            graph.add_edge(edge.from, edge.to, edge_type, edge.directed);
        }

        // Restore counters
        graph
            .next_edge_id
            .store(snapshot.next_edge_id, Ordering::Relaxed);
        graph
            .max_node_id
            .store(snapshot.max_node_id, Ordering::Relaxed);

        // Restore edge data
        for (key, data) in snapshot.edge_data.iter() {
            graph.edge_data.set(key, data.clone());
        }

        graph
    }

    /// Intern an edge type string, returning its ID.
    fn intern_edge_type(&self, edge_type: &str) -> EdgeTypeId {
        // Fast path: read-only check
        {
            let registry = self.edge_types.read();
            if let Some(&id) = registry.ids.get(edge_type) {
                return id;
            }
        }

        // Slow path: write to intern new type
        let mut registry = self.edge_types.write();
        registry.intern(edge_type)
    }

    /// Get edge type ID (for lookups).
    fn get_edge_type_id(&self, edge_type: &str) -> EdgeTypeId {
        self.edge_types.read().get_id(edge_type)
    }

    /// Update max node ID.
    fn update_max_node_id(&self, node: EntityId) {
        let id = node.as_u64();
        loop {
            let current = self.max_node_id.load(Ordering::Relaxed);
            if id <= current {
                break;
            }
            if self
                .max_node_id
                .compare_exchange(current, id, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

impl Default for GraphTensor {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable edge for snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphEdgeSnapshot {
    edge_id: EdgeId,
    from: EntityId,
    to: EntityId,
    edge_type_idx: u32,
    directed: bool,
}

/// Serializable snapshot of `GraphTensor` state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTensorSnapshot {
    edges: Vec<GraphEdgeSnapshot>,
    edge_types: Vec<String>,
    next_edge_id: u64,
    max_node_id: u64,
    edge_data: crate::metadata_slab::MetadataSlabSnapshot,
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread, time::Instant};

    use super::*;

    #[test]
    fn test_new() {
        let graph = GraphTensor::new();
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_edge() {
        let graph = GraphTensor::new();
        let from = EntityId::new(1);
        let to = EntityId::new(2);

        let edge_id = graph.add_edge(from, to, "follows", true);
        assert_eq!(edge_id.as_u64(), 0);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_outgoing() {
        let graph = GraphTensor::new();
        let n1 = EntityId::new(1);
        let n2 = EntityId::new(2);
        let n3 = EntityId::new(3);

        graph.add_edge(n1, n2, "follows", true);
        graph.add_edge(n1, n3, "follows", true);

        let outgoing = graph.outgoing(n1);
        assert_eq!(outgoing.len(), 2);
    }

    #[test]
    fn test_incoming() {
        let graph = GraphTensor::new();
        let n1 = EntityId::new(1);
        let n2 = EntityId::new(2);
        let n3 = EntityId::new(3);

        graph.add_edge(n1, n2, "follows", true);
        graph.add_edge(n3, n2, "follows", true);

        let incoming = graph.incoming(n2);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_edge_exists() {
        let graph = GraphTensor::new();
        let n1 = EntityId::new(1);
        let n2 = EntityId::new(2);
        let n3 = EntityId::new(3);

        graph.add_edge(n1, n2, "follows", true);

        assert!(graph.edge_exists(n1, n2, None));
        assert!(!graph.edge_exists(n1, n3, None));
        assert!(graph.edge_exists(n1, n2, Some("follows")));
        assert!(!graph.edge_exists(n1, n2, Some("likes")));
    }

    #[test]
    fn test_delete_edge() {
        let graph = GraphTensor::new();
        let n1 = EntityId::new(1);
        let n2 = EntityId::new(2);

        let edge_id = graph.add_edge(n1, n2, "follows", true);
        assert_eq!(graph.edge_count(), 1);

        let deleted = graph.delete_edge(edge_id);
        assert!(deleted);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_delete_nonexistent() {
        let graph = GraphTensor::new();
        let deleted = graph.delete_edge(EdgeId::new(999));
        assert!(!deleted);
    }

    #[test]
    fn test_merge() {
        let graph = GraphTensor::with_merge_threshold(1000); // High threshold to control merge

        for i in 0..100 {
            graph.add_edge(EntityId::new(i), EntityId::new(i + 1), "next", true);
        }

        assert!(graph.pending_count() > 0);

        graph.merge();

        assert_eq!(graph.pending_count(), 0);
        assert_eq!(graph.edge_count(), 100);
    }

    #[test]
    fn test_auto_merge() {
        let graph = GraphTensor::with_merge_threshold(10);

        for i in 0..15 {
            graph.add_edge(EntityId::new(i), EntityId::new(i + 1), "next", true);
        }

        // Should have auto-merged
        assert!(graph.pending_count() < 10);
    }

    #[test]
    fn test_bfs() {
        let graph = GraphTensor::new();

        // Create a tree: 0 -> (1, 2), 1 -> (3, 4), 2 -> (5, 6)
        graph.add_edge(EntityId::new(0), EntityId::new(1), "child", true);
        graph.add_edge(EntityId::new(0), EntityId::new(2), "child", true);
        graph.add_edge(EntityId::new(1), EntityId::new(3), "child", true);
        graph.add_edge(EntityId::new(1), EntityId::new(4), "child", true);
        graph.add_edge(EntityId::new(2), EntityId::new(5), "child", true);
        graph.add_edge(EntityId::new(2), EntityId::new(6), "child", true);

        let visited = graph.bfs(EntityId::new(0), 1);
        assert_eq!(visited.len(), 3); // 0, 1, 2

        let visited = graph.bfs(EntityId::new(0), 2);
        assert_eq!(visited.len(), 7); // All nodes
    }

    #[test]
    fn test_shortest_path() {
        let graph = GraphTensor::new();

        // Create a chain: 0 -> 1 -> 2 -> 3
        graph.add_edge(EntityId::new(0), EntityId::new(1), "next", true);
        graph.add_edge(EntityId::new(1), EntityId::new(2), "next", true);
        graph.add_edge(EntityId::new(2), EntityId::new(3), "next", true);

        let path = graph.shortest_path(EntityId::new(0), EntityId::new(3));
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], EntityId::new(0));
        assert_eq!(path[3], EntityId::new(3));
    }

    #[test]
    fn test_shortest_path_no_path() {
        let graph = GraphTensor::new();

        graph.add_edge(EntityId::new(0), EntityId::new(1), "next", true);
        graph.add_edge(EntityId::new(2), EntityId::new(3), "next", true);

        let path = graph.shortest_path(EntityId::new(0), EntityId::new(3));
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_same_node() {
        let graph = GraphTensor::new();
        let path = graph.shortest_path(EntityId::new(0), EntityId::new(0));
        assert_eq!(path, Some(vec![EntityId::new(0)]));
    }

    #[test]
    fn test_edge_data() {
        let graph = GraphTensor::new();
        let edge_id = graph.add_edge(EntityId::new(0), EntityId::new(1), "follows", true);

        let mut data = TensorData::new();
        data.set(
            "weight",
            crate::TensorValue::Scalar(crate::ScalarValue::Float(0.5)),
        );
        graph.set_edge_data(edge_id, data);

        let retrieved = graph.get_edge_data(edge_id);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_clear() {
        let graph = GraphTensor::new();

        for i in 0..10 {
            graph.add_edge(EntityId::new(i), EntityId::new(i + 1), "next", true);
        }

        graph.clear();

        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.pending_count(), 0);
    }

    #[test]
    fn test_snapshot_restore() {
        let graph = GraphTensor::new();

        graph.add_edge(EntityId::new(0), EntityId::new(1), "follows", true);
        graph.add_edge(EntityId::new(1), EntityId::new(2), "likes", true);

        let snapshot = graph.snapshot();
        let restored = GraphTensor::restore(snapshot);

        assert_eq!(restored.edge_count(), 2);
        assert!(restored.edge_exists(EntityId::new(0), EntityId::new(1), None));
        assert!(restored.edge_exists(EntityId::new(1), EntityId::new(2), None));
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let graph = Arc::new(GraphTensor::with_merge_threshold(10000));
        let mut handles = vec![];

        // Writer threads
        for t in 0..4u64 {
            let g = Arc::clone(&graph);
            handles.push(thread::spawn(move || {
                for i in 0..100u64 {
                    let from = EntityId::new(t * 1000 + i);
                    let to = EntityId::new(t * 1000 + i + 1);
                    g.add_edge(from, to, "next", true);
                }
            }));
        }

        // Reader threads
        for _ in 0..4 {
            let g = Arc::clone(&graph);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = g.outgoing(EntityId::new(0));
                    let _ = g.edge_count();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(graph.edge_count(), 400);
    }

    #[test]
    fn test_no_resize_stall() {
        let graph = GraphTensor::with_merge_threshold(100_000);
        let count = 10_000;

        let start = Instant::now();
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count {
            let op_start = Instant::now();
            graph.add_edge(EntityId::new(i), EntityId::new(i + 1), "next", true);
            let op_time = op_start.elapsed();
            if op_time > max_op_time {
                max_op_time = op_time;
            }
        }

        let total_time = start.elapsed();

        // No single operation should take more than 100ms (accounts for coverage overhead)
        assert!(
            max_op_time.as_millis() < 100,
            "Max operation time {:?} exceeded 100ms threshold",
            max_op_time
        );

        // Verify throughput is reasonable
        let ops_per_sec = count as f64 / total_time.as_secs_f64();
        assert!(
            ops_per_sec > 50_000.0,
            "Throughput {:.0} ops/sec too low",
            ops_per_sec
        );
    }

    #[test]
    fn test_multiple_edge_types() {
        let graph = GraphTensor::new();

        graph.add_edge(EntityId::new(0), EntityId::new(1), "follows", true);
        graph.add_edge(EntityId::new(0), EntityId::new(1), "likes", true);
        graph.add_edge(EntityId::new(0), EntityId::new(1), "blocks", true);

        assert_eq!(graph.edge_count(), 3);
        assert!(graph.edge_exists(EntityId::new(0), EntityId::new(1), Some("follows")));
        assert!(graph.edge_exists(EntityId::new(0), EntityId::new(1), Some("likes")));
        assert!(graph.edge_exists(EntityId::new(0), EntityId::new(1), Some("blocks")));
    }

    #[test]
    fn test_csr_build_empty() {
        let csr = CsrGraph::build(&[], 0);
        assert_eq!(csr.edge_count(), 0);
        assert_eq!(csr.node_count(), 0);
    }

    #[test]
    fn test_default() {
        let graph = GraphTensor::default();
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_edge_id_from_u64() {
        let edge_id: EdgeId = 42u64.into();
        assert_eq!(edge_id.as_u64(), 42);
    }

    #[test]
    fn test_outgoing_after_merge() {
        // Create graph with merge threshold low enough to trigger CSR build
        let graph = GraphTensor::with_merge_threshold(5);

        // Add enough edges to trigger merge
        for i in 0..10 {
            graph.add_edge(EntityId::new(0), EntityId::new(i + 1), "knows", true);
        }

        // Force merge
        graph.merge();

        // Query outgoing edges from CSR
        let outgoing = graph.outgoing(EntityId::new(0));
        assert_eq!(outgoing.len(), 10);
    }

    #[test]
    fn test_outgoing_after_merge_with_deletion() {
        let graph = GraphTensor::with_merge_threshold(5);

        // Add edges
        let mut edge_ids = vec![];
        for i in 0..10 {
            edge_ids.push(graph.add_edge(EntityId::new(0), EntityId::new(i + 1), "knows", true));
        }

        // Force merge
        graph.merge();

        // Delete one edge
        graph.delete_edge(edge_ids[5]);

        // Query outgoing - should exclude deleted edge
        let outgoing = graph.outgoing(EntityId::new(0));
        assert_eq!(outgoing.len(), 9);
    }

    #[test]
    fn test_edge_exists_after_merge() {
        let graph = GraphTensor::with_merge_threshold(3);

        // Add edges
        graph.add_edge(EntityId::new(0), EntityId::new(1), "knows", true);
        graph.add_edge(EntityId::new(0), EntityId::new(2), "knows", true);
        graph.add_edge(EntityId::new(0), EntityId::new(3), "knows", true);
        graph.add_edge(EntityId::new(0), EntityId::new(4), "follows", true);

        // Force merge
        graph.merge();

        // Check edge_exists in CSR
        assert!(graph.edge_exists(EntityId::new(0), EntityId::new(1), None));
        assert!(graph.edge_exists(EntityId::new(0), EntityId::new(2), Some("knows")));
        assert!(graph.edge_exists(EntityId::new(0), EntityId::new(4), Some("follows")));
        assert!(!graph.edge_exists(EntityId::new(0), EntityId::new(4), Some("knows")));
    }

    #[test]
    fn test_merge_empty_pending() {
        let graph = GraphTensor::new();

        // Merge with empty pending list - should be no-op
        graph.merge();

        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_merge_preserves_existing_csr() {
        let graph = GraphTensor::with_merge_threshold(2);

        // Add edges and merge
        graph.add_edge(EntityId::new(0), EntityId::new(1), "a", true);
        graph.add_edge(EntityId::new(0), EntityId::new(2), "a", true);
        graph.merge();

        // Add more and merge again
        graph.add_edge(EntityId::new(0), EntityId::new(3), "b", true);
        graph.merge();

        // Should have all edges
        assert_eq!(graph.edge_count(), 3);
        let outgoing = graph.outgoing(EntityId::new(0));
        assert_eq!(outgoing.len(), 3);
    }

    #[test]
    fn test_edge_exists_nonexistent_node() {
        let graph = GraphTensor::with_merge_threshold(2);

        graph.add_edge(EntityId::new(0), EntityId::new(1), "knows", true);
        graph.add_edge(EntityId::new(0), EntityId::new(2), "knows", true);
        graph.merge();

        // Check nonexistent source node
        assert!(!graph.edge_exists(EntityId::new(999), EntityId::new(1), None));

        // Check nonexistent target
        assert!(!graph.edge_exists(EntityId::new(0), EntityId::new(999), None));
    }

    #[test]
    fn test_outgoing_nonexistent_node() {
        let graph = GraphTensor::with_merge_threshold(2);

        graph.add_edge(EntityId::new(0), EntityId::new(1), "knows", true);
        graph.add_edge(EntityId::new(0), EntityId::new(2), "knows", true);
        graph.merge();

        // Query nonexistent node
        let outgoing = graph.outgoing(EntityId::new(999));
        assert!(outgoing.is_empty());
    }

    #[test]
    fn test_snapshot_restore_with_edge_data() {
        let graph = GraphTensor::with_merge_threshold(2);

        // Add nodes and edges
        graph.add_edge(EntityId::new(0), EntityId::new(1), "friend", true);

        // Add edge data
        let mut edge_data = TensorData::new();
        edge_data.set(
            "weight",
            crate::TensorValue::Scalar(crate::ScalarValue::Float(0.5)),
        );
        graph.edge_data.set("edge:0:1:friend", edge_data);

        // Create snapshot
        let snapshot = graph.snapshot();

        // Restore to new graph
        let restored = GraphTensor::restore(snapshot);

        // Verify edge data was restored
        let restored_data = restored.edge_data.get("edge:0:1:friend");
        assert!(restored_data.is_some());
    }
}
