// Pedantic lint configuration for graph_engine
#![allow(clippy::cast_possible_wrap)] // u64 IDs won't exceed i64::MAX
#![allow(clippy::cast_sign_loss)] // i64 values from store are always non-negative IDs
#![allow(clippy::needless_pass_by_value)] // HashMap ownership is intentional for API design
#![allow(clippy::missing_errors_doc)] // Error conditions are self-evident from Result types
#![allow(clippy::uninlined_format_args)] // Keep format strings readable

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
};

use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensor_store::{fields, ScalarValue, TensorData, TensorStore, TensorStoreError, TensorValue};
use tracing::{instrument, warn};

/// Range comparison operator for property queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// Target of an index (node or edge properties).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexTarget {
    Node,
    Edge,
}

/// Wrapper for f64 that provides total ordering (NaN sorts first).
#[derive(Debug, Clone, Copy)]
pub struct OrderedFloat(pub f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NaN sorts first (smallest), then normal ordering
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self
                .0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

impl Hash for OrderedFloat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Property value with total ordering for `BTreeMap` indexes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OrderedPropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(OrderedFloat),
    String(String),
}

impl From<&PropertyValue> for OrderedPropertyValue {
    fn from(v: &PropertyValue) -> Self {
        match v {
            PropertyValue::Null => Self::Null,
            PropertyValue::Bool(b) => Self::Bool(*b),
            PropertyValue::Int(i) => Self::Int(*i),
            PropertyValue::Float(f) => Self::Float(OrderedFloat(*f)),
            PropertyValue::String(s) => Self::String(s.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl PropertyValue {
    fn to_scalar(&self) -> ScalarValue {
        match self {
            Self::Null => ScalarValue::Null,
            Self::Int(v) => ScalarValue::Int(*v),
            Self::Float(v) => ScalarValue::Float(*v),
            Self::String(v) => ScalarValue::String(v.clone()),
            Self::Bool(v) => ScalarValue::Bool(*v),
        }
    }

    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Int(v) => Self::Int(*v),
            ScalarValue::Float(v) => Self::Float(*v),
            ScalarValue::String(v) => Self::String(v.clone()),
            ScalarValue::Bool(v) => Self::Bool(*v),
            ScalarValue::Null | ScalarValue::Bytes(_) => Self::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub label: String,
    pub properties: HashMap<String, PropertyValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub properties: HashMap<String, PropertyValue>,
    pub directed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Path {
    pub nodes: Vec<u64>,
    pub edges: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphError {
    NodeNotFound(u64),
    EdgeNotFound(u64),
    StorageError(String),
    PathNotFound,
    IndexAlreadyExists { target: String, property: String },
    IndexNotFound { target: String, property: String },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node not found: {id}"),
            Self::EdgeNotFound(id) => write!(f, "Edge not found: {id}"),
            Self::StorageError(e) => write!(f, "Storage error: {e}"),
            Self::PathNotFound => write!(f, "No path found between nodes"),
            Self::IndexAlreadyExists { target, property } => {
                write!(f, "Index already exists: {target}.{property}")
            },
            Self::IndexNotFound { target, property } => {
                write!(f, "Index not found: {target}.{property}")
            },
        }
    }
}

impl std::error::Error for GraphError {}

impl From<TensorStoreError> for GraphError {
    fn from(e: TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, GraphError>;

/// Number of striped locks for index operations.
const INDEX_LOCK_COUNT: usize = 64;

/// Type alias for the `BTreeMap` index structure.
type PropertyIndex = BTreeMap<OrderedPropertyValue, Vec<u64>>;

pub struct GraphEngine {
    store: TensorStore,
    node_counter: AtomicU64,
    edge_counter: AtomicU64,
    /// In-memory `BTreeMap` indexes for O(log n) property lookups.
    btree_indexes: RwLock<HashMap<(IndexTarget, String), PropertyIndex>>,
    /// Striped locks for concurrent index updates.
    #[allow(clippy::type_complexity)]
    index_locks: [RwLock<()>; INDEX_LOCK_COUNT],
}

impl std::fmt::Debug for GraphEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let index_count = self.btree_indexes.read().len();
        f.debug_struct("GraphEngine")
            .field("node_counter", &self.node_counter.load(Ordering::Relaxed))
            .field("edge_counter", &self.edge_counter.load(Ordering::Relaxed))
            .field("index_count", &index_count)
            .finish_non_exhaustive()
    }
}

/// Creates array of `RwLock`s for striped locking.
fn create_index_locks() -> [RwLock<()>; INDEX_LOCK_COUNT] {
    std::array::from_fn(|_| RwLock::new(()))
}

impl GraphEngine {
    const PARALLEL_THRESHOLD: usize = 100;

    #[must_use]
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            node_counter: AtomicU64::new(0),
            edge_counter: AtomicU64::new(0),
            btree_indexes: RwLock::new(HashMap::new()),
            index_locks: create_index_locks(),
        }
    }

    /// Create a `GraphEngine` with an existing store.
    ///
    /// This scans the store to initialize node/edge counters correctly,
    /// avoiding ID collisions with existing data. Also rebuilds any
    /// persisted indexes from stored metadata.
    #[must_use]
    pub fn with_store(store: TensorStore) -> Self {
        let mut max_node_id = 0u64;
        let mut max_edge_id = 0u64;

        // Scan for existing nodes to find max ID
        for key in store.scan("node:") {
            // Skip edge list keys like "node:1:out" and "node:1:in"
            if key.contains(":out") || key.contains(":in") {
                continue;
            }
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    max_node_id = max_node_id.max(id);
                }
            }
        }

        // Scan for existing edges to find max ID
        for key in store.scan("edge:") {
            if let Some(id_str) = key.strip_prefix("edge:") {
                // Handle both "edge:123" and "edge:type:123" formats
                let id_part = id_str.rsplit(':').next().unwrap_or(id_str);
                if let Ok(id) = id_part.parse::<u64>() {
                    max_edge_id = max_edge_id.max(id);
                }
            }
        }

        // Rebuild indexes from persistent metadata
        let btree_indexes = Self::rebuild_indexes_from_store(&store);

        Self {
            store,
            node_counter: AtomicU64::new(max_node_id),
            edge_counter: AtomicU64::new(max_edge_id),
            btree_indexes: RwLock::new(btree_indexes),
            index_locks: create_index_locks(),
        }
    }

    /// Rebuild in-memory indexes from persistent store metadata.
    fn rebuild_indexes_from_store(
        store: &TensorStore,
    ) -> HashMap<(IndexTarget, String), PropertyIndex> {
        let mut indexes = HashMap::new();

        // Scan for index metadata keys: _graph_idx:node:{property} or _graph_idx:edge:{property}
        for key in store.scan("_graph_idx:") {
            let parts: Vec<&str> = key.splitn(3, ':').collect();
            if parts.len() < 3 {
                continue;
            }

            let target = match parts[1] {
                "node" => IndexTarget::Node,
                "edge" => IndexTarget::Edge,
                _ => continue,
            };
            let property = parts[2].to_string();

            // Rebuild the BTreeMap index by scanning all nodes/edges
            let mut btree: PropertyIndex = BTreeMap::new();

            match target {
                IndexTarget::Node => {
                    for node_key in store.scan("node:") {
                        if node_key.contains(":out") || node_key.contains(":in") {
                            continue;
                        }
                        if let Some(id_str) = node_key.strip_prefix("node:") {
                            if let Ok(id) = id_str.parse::<u64>() {
                                if let Ok(tensor) = store.get(&node_key) {
                                    // Handle special properties (_label)
                                    let value = if property == "_label" {
                                        match tensor.get("_label") {
                                            Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                                                Some(OrderedPropertyValue::String(s.clone()))
                                            },
                                            _ => None,
                                        }
                                    } else if let Some(TensorValue::Scalar(scalar)) =
                                        tensor.get(&property)
                                    {
                                        Some(OrderedPropertyValue::from(
                                            &PropertyValue::from_scalar(scalar),
                                        ))
                                    } else {
                                        None
                                    };

                                    if let Some(ordered_val) = value {
                                        btree.entry(ordered_val).or_default().push(id);
                                    }
                                }
                            }
                        }
                    }
                },
                IndexTarget::Edge => {
                    for edge_key in store.scan("edge:") {
                        if let Some(id_str) = edge_key.strip_prefix("edge:") {
                            // Only handle "edge:N" format (classic mode)
                            if let Ok(id) = id_str.parse::<u64>() {
                                if let Ok(tensor) = store.get(&edge_key) {
                                    // Handle special properties (_edge_type)
                                    let value = if property == "_edge_type" {
                                        match tensor.get("_edge_type") {
                                            Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                                                Some(OrderedPropertyValue::String(s.clone()))
                                            },
                                            _ => None,
                                        }
                                    } else if let Some(TensorValue::Scalar(scalar)) =
                                        tensor.get(&property)
                                    {
                                        Some(OrderedPropertyValue::from(
                                            &PropertyValue::from_scalar(scalar),
                                        ))
                                    } else {
                                        None
                                    };

                                    if let Some(ordered_val) = value {
                                        btree.entry(ordered_val).or_default().push(id);
                                    }
                                }
                            }
                        }
                    }
                },
            }

            indexes.insert((target, property), btree);
        }

        indexes
    }

    /// Access the underlying store.
    #[must_use]
    pub const fn store(&self) -> &TensorStore {
        &self.store
    }

    #[inline]
    fn node_key(id: u64) -> String {
        format!("node:{id}")
    }

    #[inline]
    fn edge_key(id: u64) -> String {
        format!("edge:{id}")
    }

    #[inline]
    fn outgoing_edges_key(node_id: u64) -> String {
        format!("node:{node_id}:out")
    }

    #[inline]
    fn incoming_edges_key(node_id: u64) -> String {
        format!("node:{node_id}:in")
    }

    #[inline]
    fn index_metadata_key(target: IndexTarget, property: &str) -> String {
        let target_str = match target {
            IndexTarget::Node => "node",
            IndexTarget::Edge => "edge",
        };
        format!("_graph_idx:{target_str}:{property}")
    }

    /// Get the striped lock index for a given id.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn lock_index(id: u64) -> usize {
        (id as usize) % INDEX_LOCK_COUNT
    }

    // ========== Index CRUD Methods ==========

    /// Create an index on a node property for O(log n) lookups.
    ///
    /// # Errors
    ///
    /// Returns `IndexAlreadyExists` if an index already exists for this property.
    pub fn create_node_property_index(&self, property: &str) -> Result<()> {
        self.create_property_index(IndexTarget::Node, property)
    }

    /// Create an index on an edge property for O(log n) lookups.
    ///
    /// # Errors
    ///
    /// Returns `IndexAlreadyExists` if an index already exists for this property.
    pub fn create_edge_property_index(&self, property: &str) -> Result<()> {
        self.create_property_index(IndexTarget::Edge, property)
    }

    /// Create an index on node labels for fast label-based lookups.
    ///
    /// # Errors
    ///
    /// Returns `IndexAlreadyExists` if a label index already exists.
    pub fn create_label_index(&self) -> Result<()> {
        self.create_property_index(IndexTarget::Node, "_label")
    }

    /// Create an index on edge types for fast type-based lookups.
    ///
    /// # Errors
    ///
    /// Returns `IndexAlreadyExists` if an edge type index already exists.
    pub fn create_edge_type_index(&self) -> Result<()> {
        self.create_property_index(IndexTarget::Edge, "_edge_type")
    }

    /// Drop a node property index.
    ///
    /// # Errors
    ///
    /// Returns `IndexNotFound` if no index exists for this property.
    pub fn drop_node_index(&self, property: &str) -> Result<()> {
        self.drop_property_index(IndexTarget::Node, property)
    }

    /// Drop an edge property index.
    ///
    /// # Errors
    ///
    /// Returns `IndexNotFound` if no index exists for this property.
    pub fn drop_edge_index(&self, property: &str) -> Result<()> {
        self.drop_property_index(IndexTarget::Edge, property)
    }

    /// Check if a node property index exists.
    #[must_use]
    pub fn has_node_index(&self, property: &str) -> bool {
        self.btree_indexes
            .read()
            .contains_key(&(IndexTarget::Node, property.to_string()))
    }

    /// Check if an edge property index exists.
    #[must_use]
    pub fn has_edge_index(&self, property: &str) -> bool {
        self.btree_indexes
            .read()
            .contains_key(&(IndexTarget::Edge, property.to_string()))
    }

    /// Get list of indexed node properties.
    #[must_use]
    pub fn get_indexed_node_properties(&self) -> Vec<String> {
        self.btree_indexes
            .read()
            .keys()
            .filter_map(|(target, prop)| {
                if *target == IndexTarget::Node {
                    Some(prop.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get list of indexed edge properties.
    #[must_use]
    pub fn get_indexed_edge_properties(&self) -> Vec<String> {
        self.btree_indexes
            .read()
            .keys()
            .filter_map(|(target, prop)| {
                if *target == IndexTarget::Edge {
                    Some(prop.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    fn create_property_index(&self, target: IndexTarget, property: &str) -> Result<()> {
        let key = (target, property.to_string());

        // Check if index already exists
        {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&key) {
                return Err(GraphError::IndexAlreadyExists {
                    target: format!("{target:?}"),
                    property: property.to_string(),
                });
            }
        }

        // Build the index by scanning all nodes/edges
        let mut btree: PropertyIndex = BTreeMap::new();

        match target {
            IndexTarget::Node => {
                for node_key in self.store.scan("node:") {
                    if node_key.contains(":out") || node_key.contains(":in") {
                        continue;
                    }
                    if let Some(id_str) = node_key.strip_prefix("node:") {
                        if let Ok(id) = id_str.parse::<u64>() {
                            if let Ok(tensor) = self.store.get(&node_key) {
                                let value = if property == "_label" {
                                    match tensor.get("_label") {
                                        Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                                            Some(OrderedPropertyValue::String(s.clone()))
                                        },
                                        _ => None,
                                    }
                                } else if let Some(TensorValue::Scalar(scalar)) =
                                    tensor.get(property)
                                {
                                    Some(OrderedPropertyValue::from(&PropertyValue::from_scalar(
                                        scalar,
                                    )))
                                } else {
                                    None
                                };

                                if let Some(ordered_val) = value {
                                    btree.entry(ordered_val).or_default().push(id);
                                }
                            }
                        }
                    }
                }
            },
            IndexTarget::Edge => {
                for edge_key in self.store.scan("edge:") {
                    if let Some(id_str) = edge_key.strip_prefix("edge:") {
                        if let Ok(id) = id_str.parse::<u64>() {
                            if let Ok(tensor) = self.store.get(&edge_key) {
                                let value = if property == "_edge_type" {
                                    match tensor.get("_edge_type") {
                                        Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                                            Some(OrderedPropertyValue::String(s.clone()))
                                        },
                                        _ => None,
                                    }
                                } else if let Some(TensorValue::Scalar(scalar)) =
                                    tensor.get(property)
                                {
                                    Some(OrderedPropertyValue::from(&PropertyValue::from_scalar(
                                        scalar,
                                    )))
                                } else {
                                    None
                                };

                                if let Some(ordered_val) = value {
                                    btree.entry(ordered_val).or_default().push(id);
                                }
                            }
                        }
                    }
                }
            },
        }

        // Persist index metadata
        let metadata_key = Self::index_metadata_key(target, property);
        let mut metadata = TensorData::new();
        metadata.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("index".into())),
        );
        self.store.put(metadata_key, metadata)?;

        // Insert into in-memory index
        self.btree_indexes.write().insert(key, btree);

        Ok(())
    }

    fn drop_property_index(&self, target: IndexTarget, property: &str) -> Result<()> {
        let key = (target, property.to_string());

        // Remove from in-memory index
        let removed = self.btree_indexes.write().remove(&key).is_some();

        if !removed {
            return Err(GraphError::IndexNotFound {
                target: format!("{target:?}"),
                property: property.to_string(),
            });
        }

        // Remove persistent metadata
        let metadata_key = Self::index_metadata_key(target, property);
        self.store.delete(&metadata_key)?;

        Ok(())
    }

    // ========== Internal Index Maintenance ==========

    fn index_add(
        &self,
        target: IndexTarget,
        property: &str,
        value: &OrderedPropertyValue,
        id: u64,
    ) {
        let key = (target, property.to_string());
        let _lock = self.index_locks[Self::lock_index(id)].write();

        let mut indexes = self.btree_indexes.write();
        if let Some(btree) = indexes.get_mut(&key) {
            btree.entry(value.clone()).or_default().push(id);
        }
    }

    fn index_remove(
        &self,
        target: IndexTarget,
        property: &str,
        value: &OrderedPropertyValue,
        id: u64,
    ) {
        let key = (target, property.to_string());
        let _lock = self.index_locks[Self::lock_index(id)].write();

        let mut indexes = self.btree_indexes.write();
        if let Some(btree) = indexes.get_mut(&key) {
            if let Some(ids) = btree.get_mut(value) {
                ids.retain(|&x| x != id);
                if ids.is_empty() {
                    btree.remove(value);
                }
            }
        }
    }

    fn index_node_properties(
        &self,
        id: u64,
        label: &str,
        properties: &HashMap<String, PropertyValue>,
    ) {
        let indexes = self.btree_indexes.read();

        // Index label if _label index exists
        if indexes.contains_key(&(IndexTarget::Node, "_label".to_string())) {
            drop(indexes);
            self.index_add(
                IndexTarget::Node,
                "_label",
                &OrderedPropertyValue::String(label.to_string()),
                id,
            );
        } else {
            drop(indexes);
        }

        // Index each property that has an index
        for (prop_name, prop_value) in properties {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&(IndexTarget::Node, prop_name.clone())) {
                drop(indexes);
                self.index_add(
                    IndexTarget::Node,
                    prop_name,
                    &OrderedPropertyValue::from(prop_value),
                    id,
                );
            }
        }
    }

    fn unindex_node_properties(
        &self,
        id: u64,
        label: &str,
        properties: &HashMap<String, PropertyValue>,
    ) {
        let indexes = self.btree_indexes.read();

        // Unindex label if _label index exists
        if indexes.contains_key(&(IndexTarget::Node, "_label".to_string())) {
            drop(indexes);
            self.index_remove(
                IndexTarget::Node,
                "_label",
                &OrderedPropertyValue::String(label.to_string()),
                id,
            );
        } else {
            drop(indexes);
        }

        // Unindex each property that has an index
        for (prop_name, prop_value) in properties {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&(IndexTarget::Node, prop_name.clone())) {
                drop(indexes);
                self.index_remove(
                    IndexTarget::Node,
                    prop_name,
                    &OrderedPropertyValue::from(prop_value),
                    id,
                );
            }
        }
    }

    fn index_edge_properties(
        &self,
        id: u64,
        edge_type: &str,
        properties: &HashMap<String, PropertyValue>,
    ) {
        let indexes = self.btree_indexes.read();

        // Index edge_type if _edge_type index exists
        if indexes.contains_key(&(IndexTarget::Edge, "_edge_type".to_string())) {
            drop(indexes);
            self.index_add(
                IndexTarget::Edge,
                "_edge_type",
                &OrderedPropertyValue::String(edge_type.to_string()),
                id,
            );
        } else {
            drop(indexes);
        }

        // Index each property that has an index
        for (prop_name, prop_value) in properties {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&(IndexTarget::Edge, prop_name.clone())) {
                drop(indexes);
                self.index_add(
                    IndexTarget::Edge,
                    prop_name,
                    &OrderedPropertyValue::from(prop_value),
                    id,
                );
            }
        }
    }

    fn unindex_edge_properties(
        &self,
        id: u64,
        edge_type: &str,
        properties: &HashMap<String, PropertyValue>,
    ) {
        let indexes = self.btree_indexes.read();

        // Unindex edge_type if _edge_type index exists
        if indexes.contains_key(&(IndexTarget::Edge, "_edge_type".to_string())) {
            drop(indexes);
            self.index_remove(
                IndexTarget::Edge,
                "_edge_type",
                &OrderedPropertyValue::String(edge_type.to_string()),
                id,
            );
        } else {
            drop(indexes);
        }

        // Unindex each property that has an index
        for (prop_name, prop_value) in properties {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&(IndexTarget::Edge, prop_name.clone())) {
                drop(indexes);
                self.index_remove(
                    IndexTarget::Edge,
                    prop_name,
                    &OrderedPropertyValue::from(prop_value),
                    id,
                );
            }
        }
    }

    // ========== Index-Accelerated Queries ==========

    /// Find nodes by exact property value match. Uses index if available.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_property(
        &self,
        property: &str,
        value: &PropertyValue,
    ) -> Result<Vec<Node>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Node, property.to_string());

        let indexes = self.btree_indexes.read();
        if let Some(btree) = indexes.get(&key) {
            // Use index
            let ids = btree.get(&ordered_value).cloned().unwrap_or_default();
            drop(indexes);

            let mut nodes = Vec::with_capacity(ids.len());
            for id in ids {
                if let Ok(node) = self.get_node(id) {
                    nodes.push(node);
                }
            }
            nodes.sort_by_key(|n| n.id);
            return Ok(nodes);
        }
        drop(indexes);

        // Fallback to scan
        Ok(self.scan_nodes_by_property(property, value))
    }

    /// Find nodes using a range comparison. Uses index if available.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_where(
        &self,
        property: &str,
        op: RangeOp,
        value: &PropertyValue,
    ) -> Result<Vec<Node>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Node, property.to_string());

        let indexes = self.btree_indexes.read();
        if let Some(btree) = indexes.get(&key) {
            // Use index range query
            let ids: Vec<u64> = match op {
                RangeOp::Lt => btree
                    .range(..ordered_value)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
                RangeOp::Le => btree
                    .range(..=ordered_value)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
                RangeOp::Gt => btree
                    .range((
                        std::ops::Bound::Excluded(ordered_value),
                        std::ops::Bound::Unbounded,
                    ))
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
                RangeOp::Ge => btree
                    .range(ordered_value..)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
            };
            drop(indexes);

            let mut nodes = Vec::with_capacity(ids.len());
            for id in ids {
                if let Ok(node) = self.get_node(id) {
                    nodes.push(node);
                }
            }
            nodes.sort_by_key(|n| n.id);
            return Ok(nodes);
        }
        drop(indexes);

        // Fallback to scan
        Ok(self.scan_nodes_where(property, op, value))
    }

    /// Find nodes by label. Uses label index if available.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_label(&self, label: &str) -> Result<Vec<Node>> {
        self.find_nodes_by_property("_label", &PropertyValue::String(label.to_string()))
    }

    /// Find edges by exact property value match. Uses index if available.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_edges_by_property(
        &self,
        property: &str,
        value: &PropertyValue,
    ) -> Result<Vec<Edge>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Edge, property.to_string());

        let indexes = self.btree_indexes.read();
        if let Some(btree) = indexes.get(&key) {
            // Use index
            let ids = btree.get(&ordered_value).cloned().unwrap_or_default();
            drop(indexes);

            let mut edges = Vec::with_capacity(ids.len());
            for id in ids {
                if let Ok(edge) = self.get_edge(id) {
                    edges.push(edge);
                }
            }
            edges.sort_by_key(|e| e.id);
            return Ok(edges);
        }
        drop(indexes);

        // Fallback to scan
        Ok(self.scan_edges_by_property(property, value))
    }

    /// Find edges using a range comparison. Uses index if available.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_edges_where(
        &self,
        property: &str,
        op: RangeOp,
        value: &PropertyValue,
    ) -> Result<Vec<Edge>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Edge, property.to_string());

        let indexes = self.btree_indexes.read();
        if let Some(btree) = indexes.get(&key) {
            // Use index range query
            let ids: Vec<u64> = match op {
                RangeOp::Lt => btree
                    .range(..ordered_value)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
                RangeOp::Le => btree
                    .range(..=ordered_value)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
                RangeOp::Gt => btree
                    .range((
                        std::ops::Bound::Excluded(ordered_value),
                        std::ops::Bound::Unbounded,
                    ))
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
                RangeOp::Ge => btree
                    .range(ordered_value..)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect(),
            };
            drop(indexes);

            let mut edges = Vec::with_capacity(ids.len());
            for id in ids {
                if let Ok(edge) = self.get_edge(id) {
                    edges.push(edge);
                }
            }
            edges.sort_by_key(|e| e.id);
            return Ok(edges);
        }
        drop(indexes);

        // Fallback to scan
        Ok(self.scan_edges_where(property, op, value))
    }

    /// Find edges by type. Uses edge type index if available.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_edges_by_type(&self, edge_type: &str) -> Result<Vec<Edge>> {
        self.find_edges_by_property("_edge_type", &PropertyValue::String(edge_type.to_string()))
    }

    // ========== Scan Fallbacks ==========

    fn scan_nodes_by_property(&self, property: &str, value: &PropertyValue) -> Vec<Node> {
        let mut nodes = Vec::new();

        for key in self.store.scan("node:") {
            if key.contains(":out") || key.contains(":in") {
                continue;
            }
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(node) = self.get_node(id) {
                        let matches = if property == "_label" {
                            match value {
                                PropertyValue::String(s) => node.label == *s,
                                _ => false,
                            }
                        } else {
                            node.properties.get(property) == Some(value)
                        };
                        if matches {
                            nodes.push(node);
                        }
                    }
                }
            }
        }

        nodes.sort_by_key(|n| n.id);
        nodes
    }

    fn scan_nodes_where(&self, property: &str, op: RangeOp, value: &PropertyValue) -> Vec<Node> {
        let ordered_value = OrderedPropertyValue::from(value);
        let mut nodes = Vec::new();

        for key in self.store.scan("node:") {
            if key.contains(":out") || key.contains(":in") {
                continue;
            }
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(node) = self.get_node(id) {
                        let prop_value = if property == "_label" {
                            Some(PropertyValue::String(node.label.clone()))
                        } else {
                            node.properties.get(property).cloned()
                        };

                        if let Some(pv) = prop_value {
                            let ordered_pv = OrderedPropertyValue::from(&pv);
                            let matches = match op {
                                RangeOp::Lt => ordered_pv < ordered_value,
                                RangeOp::Le => ordered_pv <= ordered_value,
                                RangeOp::Gt => ordered_pv > ordered_value,
                                RangeOp::Ge => ordered_pv >= ordered_value,
                            };
                            if matches {
                                nodes.push(node);
                            }
                        }
                    }
                }
            }
        }

        nodes.sort_by_key(|n| n.id);
        nodes
    }

    fn scan_edges_by_property(&self, property: &str, value: &PropertyValue) -> Vec<Edge> {
        let mut edges = Vec::new();

        for key in self.store.scan("edge:") {
            if let Some(id_str) = key.strip_prefix("edge:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(edge) = self.get_edge(id) {
                        let matches = if property == "_edge_type" {
                            match value {
                                PropertyValue::String(s) => edge.edge_type == *s,
                                _ => false,
                            }
                        } else {
                            edge.properties.get(property) == Some(value)
                        };
                        if matches {
                            edges.push(edge);
                        }
                    }
                }
            }
        }

        edges.sort_by_key(|e| e.id);
        edges
    }

    fn scan_edges_where(&self, property: &str, op: RangeOp, value: &PropertyValue) -> Vec<Edge> {
        let ordered_value = OrderedPropertyValue::from(value);
        let mut edges = Vec::new();

        for key in self.store.scan("edge:") {
            if let Some(id_str) = key.strip_prefix("edge:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(edge) = self.get_edge(id) {
                        let prop_value = if property == "_edge_type" {
                            Some(PropertyValue::String(edge.edge_type.clone()))
                        } else {
                            edge.properties.get(property).cloned()
                        };

                        if let Some(pv) = prop_value {
                            let ordered_pv = OrderedPropertyValue::from(&pv);
                            let matches = match op {
                                RangeOp::Lt => ordered_pv < ordered_value,
                                RangeOp::Le => ordered_pv <= ordered_value,
                                RangeOp::Gt => ordered_pv > ordered_value,
                                RangeOp::Ge => ordered_pv >= ordered_value,
                            };
                            if matches {
                                edges.push(edge);
                            }
                        }
                    }
                }
            }
        }

        edges.sort_by_key(|e| e.id);
        edges
    }

    #[instrument(skip(self, label, properties))]
    pub fn create_node(
        &self,
        label: impl Into<String>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<u64> {
        let id = self.node_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let label = label.into();

        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(id as i64)));
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("node".into())),
        );
        tensor.set(
            "_label",
            TensorValue::Scalar(ScalarValue::String(label.clone())),
        );

        for (key, value) in &properties {
            tensor.set(key, TensorValue::Scalar(value.to_scalar()));
        }

        self.store.put(Self::node_key(id), tensor)?;

        // Initialize empty edge lists
        let out_tensor = TensorData::new();
        let in_tensor = TensorData::new();
        self.store.put(Self::outgoing_edges_key(id), out_tensor)?;
        self.store.put(Self::incoming_edges_key(id), in_tensor)?;

        // Update indexes
        self.index_node_properties(id, &label, &properties);

        Ok(id)
    }

    #[instrument(skip(self, edge_type, properties), fields(from = from, to = to))]
    pub fn create_edge(
        &self,
        from: u64,
        to: u64,
        edge_type: impl Into<String>,
        properties: HashMap<String, PropertyValue>,
        directed: bool,
    ) -> Result<u64> {
        // Verify both nodes exist
        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        let id = self.edge_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let edge_type = edge_type.into();

        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(id as i64)));
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("edge".into())),
        );
        tensor.set("_from", TensorValue::Scalar(ScalarValue::Int(from as i64)));
        tensor.set("_to", TensorValue::Scalar(ScalarValue::Int(to as i64)));
        tensor.set(
            "_edge_type",
            TensorValue::Scalar(ScalarValue::String(edge_type.clone())),
        );
        tensor.set(
            "_directed",
            TensorValue::Scalar(ScalarValue::Bool(directed)),
        );

        for (key, value) in &properties {
            tensor.set(key, TensorValue::Scalar(value.to_scalar()));
        }

        self.store.put(Self::edge_key(id), tensor)?;

        // Add to outgoing edges of 'from' node
        self.add_edge_to_list(Self::outgoing_edges_key(from), id)?;
        // Add to incoming edges of 'to' node
        self.add_edge_to_list(Self::incoming_edges_key(to), id)?;

        // For undirected edges, also add reverse connections
        if !directed {
            self.add_edge_to_list(Self::outgoing_edges_key(to), id)?;
            self.add_edge_to_list(Self::incoming_edges_key(from), id)?;
        }

        // Update indexes
        self.index_edge_properties(id, &edge_type, &properties);

        Ok(id)
    }

    fn add_edge_to_list(&self, key: String, edge_id: u64) -> Result<()> {
        let mut tensor = self.store.get(&key).unwrap_or_else(|_| TensorData::new());

        let edge_key = format!("e{}", edge_id);
        tensor.set(
            &edge_key,
            TensorValue::Scalar(ScalarValue::Int(edge_id as i64)),
        );

        self.store.put(key, tensor)?;
        Ok(())
    }

    fn get_edge_list(&self, key: &str) -> Vec<u64> {
        let Ok(tensor) = self.store.get(key) else {
            return Vec::new();
        };

        let mut edges = Vec::new();
        for k in tensor.keys() {
            if k.starts_with('e') {
                if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = tensor.get(k) {
                    edges.push(*id as u64);
                }
            }
        }
        edges
    }

    pub fn node_exists(&self, id: u64) -> bool {
        self.store.exists(&Self::node_key(id))
    }

    #[instrument(skip(self), fields(node_id = id))]
    pub fn get_node(&self, id: u64) -> Result<Node> {
        let tensor = self
            .store
            .get(&Self::node_key(id))
            .map_err(|_| GraphError::NodeNotFound(id))?;

        let label = match tensor.get("_label") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => String::new(),
        };

        let mut properties = HashMap::new();
        for key in tensor.keys() {
            if key.starts_with('_') {
                continue;
            }
            if let Some(TensorValue::Scalar(scalar)) = tensor.get(key) {
                properties.insert(key.clone(), PropertyValue::from_scalar(scalar));
            }
        }

        Ok(Node {
            id,
            label,
            properties,
        })
    }

    pub fn get_edge(&self, id: u64) -> Result<Edge> {
        let tensor = self
            .store
            .get(&Self::edge_key(id))
            .map_err(|_| GraphError::EdgeNotFound(id))?;

        let from = match tensor.get("_from") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v as u64,
            _ => 0,
        };

        let to = match tensor.get("_to") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v as u64,
            _ => 0,
        };

        let edge_type = match tensor.get("_edge_type") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => String::new(),
        };

        let directed = match tensor.get("_directed") {
            Some(TensorValue::Scalar(ScalarValue::Bool(b))) => *b,
            _ => true,
        };

        let mut properties = HashMap::new();
        for key in tensor.keys() {
            if key.starts_with('_') {
                continue;
            }
            if let Some(TensorValue::Scalar(scalar)) = tensor.get(key) {
                properties.insert(key.clone(), PropertyValue::from_scalar(scalar));
            }
        }

        Ok(Edge {
            id,
            from,
            to,
            edge_type,
            properties,
            directed,
        })
    }

    /// Update a node's label and/or properties.
    ///
    /// Pass `None` to leave the label unchanged. Properties are merged with
    /// existing properties; pass `PropertyValue::Null` to remove a property.
    pub fn update_node(
        &self,
        id: u64,
        label: Option<&str>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<()> {
        // Get old node for index maintenance
        let old_node = self.get_node(id)?;

        let key = Self::node_key(id);
        let mut tensor = self
            .store
            .get(&key)
            .map_err(|_| GraphError::NodeNotFound(id))?;

        // Unindex old values for changed properties
        let old_label = old_node.label.clone();
        let mut changed_props: HashMap<String, PropertyValue> = HashMap::new();

        if let Some(new_label) = label {
            if new_label != old_label {
                // Unindex old label
                self.index_remove(
                    IndexTarget::Node,
                    "_label",
                    &OrderedPropertyValue::String(old_label.clone()),
                    id,
                );
            }
            tensor.set(
                "_label",
                TensorValue::Scalar(ScalarValue::String(new_label.to_string())),
            );
        }

        for (prop_key, value) in &properties {
            // Unindex old value if it exists
            if let Some(old_value) = old_node.properties.get(prop_key) {
                self.index_remove(
                    IndexTarget::Node,
                    prop_key,
                    &OrderedPropertyValue::from(old_value),
                    id,
                );
            }

            if value == &PropertyValue::Null {
                tensor.remove(prop_key);
            } else {
                tensor.set(prop_key, TensorValue::Scalar(value.to_scalar()));
                changed_props.insert(prop_key.clone(), value.clone());
            }
        }

        self.store.put(key, tensor)?;

        // Index new values
        let new_label = label.unwrap_or(&old_label);
        if label.is_some() && new_label != old_label {
            self.index_add(
                IndexTarget::Node,
                "_label",
                &OrderedPropertyValue::String(new_label.to_string()),
                id,
            );
        }

        for (prop_key, prop_value) in &changed_props {
            self.index_add(
                IndexTarget::Node,
                prop_key,
                &OrderedPropertyValue::from(prop_value),
                id,
            );
        }

        Ok(())
    }

    /// Update an edge's properties.
    ///
    /// Properties are merged with existing properties; pass `PropertyValue::Null`
    /// to remove a property. The edge type, from/to nodes, and directedness cannot
    /// be changed after creation.
    pub fn update_edge(&self, id: u64, properties: HashMap<String, PropertyValue>) -> Result<()> {
        // Get old edge for index maintenance
        let old_edge = self.get_edge(id)?;

        let key = Self::edge_key(id);
        let mut tensor = self
            .store
            .get(&key)
            .map_err(|_| GraphError::EdgeNotFound(id))?;

        let mut changed_props: HashMap<String, PropertyValue> = HashMap::new();

        for (prop_key, value) in &properties {
            // Unindex old value if it exists
            if let Some(old_value) = old_edge.properties.get(prop_key) {
                self.index_remove(
                    IndexTarget::Edge,
                    prop_key,
                    &OrderedPropertyValue::from(old_value),
                    id,
                );
            }

            if value == &PropertyValue::Null {
                tensor.remove(prop_key);
            } else {
                tensor.set(prop_key, TensorValue::Scalar(value.to_scalar()));
                changed_props.insert(prop_key.clone(), value.clone());
            }
        }

        self.store.put(key, tensor)?;

        // Index new values
        for (prop_key, prop_value) in &changed_props {
            self.index_add(
                IndexTarget::Edge,
                prop_key,
                &OrderedPropertyValue::from(prop_value),
                id,
            );
        }

        Ok(())
    }

    /// Returns all edges connected to a node.
    pub fn edges_of(&self, node_id: u64, direction: Direction) -> Result<Vec<Edge>> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }

        let mut edge_ids = HashSet::new();

        if direction == Direction::Outgoing || direction == Direction::Both {
            for id in self.get_edge_list(&Self::outgoing_edges_key(node_id)) {
                edge_ids.insert(id);
            }
        }

        if direction == Direction::Incoming || direction == Direction::Both {
            for id in self.get_edge_list(&Self::incoming_edges_key(node_id)) {
                edge_ids.insert(id);
            }
        }

        let mut edges = Vec::with_capacity(edge_ids.len());
        for id in edge_ids {
            if let Ok(edge) = self.get_edge(id) {
                edges.push(edge);
            } else {
                warn!(edge_id = id, "orphaned edge ID in edge list");
            }
        }
        edges.sort_by_key(|e| e.id);
        Ok(edges)
    }

    #[instrument(skip(self), fields(node_id = node_id))]
    pub fn neighbors(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<Node>> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }

        let mut neighbor_ids = HashSet::new();

        // Get outgoing neighbors
        if direction == Direction::Outgoing || direction == Direction::Both {
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(node_id));
            for edge_id in out_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                        // For outgoing, the neighbor is the 'to' node (unless it's us in
                        // undirected)
                        if edge.from == node_id && edge.to != node_id {
                            neighbor_ids.insert(edge.to);
                        } else if edge.to == node_id && edge.from != node_id {
                            neighbor_ids.insert(edge.from);
                        }
                    }
                }
            }
        }

        // Get incoming neighbors
        if direction == Direction::Incoming || direction == Direction::Both {
            let in_edges = self.get_edge_list(&Self::incoming_edges_key(node_id));
            for edge_id in in_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                        if edge.to == node_id && edge.from != node_id {
                            neighbor_ids.insert(edge.from);
                        } else if edge.from == node_id && edge.to != node_id {
                            neighbor_ids.insert(edge.to);
                        }
                    }
                }
            }
        }

        let mut neighbors = Vec::new();
        for id in neighbor_ids {
            if let Ok(node) = self.get_node(id) {
                neighbors.push(node);
            }
        }

        neighbors.sort_by_key(|n| n.id);
        Ok(neighbors)
    }

    pub fn traverse(
        &self,
        start: u64,
        direction: Direction,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> Result<Vec<Node>> {
        if !self.node_exists(start) {
            return Err(GraphError::NodeNotFound(start));
        }

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((start, 0usize));
        visited.insert(start);

        while let Some((current_id, depth)) = queue.pop_front() {
            if let Ok(node) = self.get_node(current_id) {
                result.push(node);
            }

            if depth >= max_depth {
                continue;
            }

            let neighbors = self.get_neighbor_ids(current_id, edge_type, direction);
            for neighbor_id in neighbors {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    queue.push_back((neighbor_id, depth + 1));
                }
            }
        }

        Ok(result)
    }

    fn get_neighbor_ids(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<u64> {
        let mut neighbor_ids = HashSet::new();

        if direction == Direction::Outgoing || direction == Direction::Both {
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(node_id));
            for edge_id in out_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                        if edge.from == node_id {
                            neighbor_ids.insert(edge.to);
                        }
                        if !edge.directed && edge.to == node_id {
                            neighbor_ids.insert(edge.from);
                        }
                    }
                }
            }
        }

        if direction == Direction::Incoming || direction == Direction::Both {
            let in_edges = self.get_edge_list(&Self::incoming_edges_key(node_id));
            for edge_id in in_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                        if edge.to == node_id {
                            neighbor_ids.insert(edge.from);
                        }
                        if !edge.directed && edge.from == node_id {
                            neighbor_ids.insert(edge.to);
                        }
                    }
                }
            }
        }

        neighbor_ids.remove(&node_id);
        neighbor_ids.into_iter().collect()
    }

    pub fn find_path(&self, from: u64, to: u64) -> Result<Path> {
        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        if from == to {
            return Ok(Path {
                nodes: vec![from],
                edges: vec![],
            });
        }

        // BFS for shortest path
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<u64, (u64, u64)> = HashMap::new(); // node -> (parent_node, edge_id)

        queue.push_back(from);
        visited.insert(from);

        while let Some(current) = queue.pop_front() {
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(current));

            for edge_id in out_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    let neighbor = if edge.from == current {
                        edge.to
                    } else if !edge.directed && edge.to == current {
                        edge.from
                    } else {
                        continue;
                    };

                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, (current, edge_id));

                        if neighbor == to {
                            return Ok(self.reconstruct_path(from, to, &parent));
                        }

                        queue.push_back(neighbor);
                    }
                }
            }
        }

        Err(GraphError::PathNotFound)
    }

    #[allow(clippy::unused_self)]
    fn reconstruct_path(&self, from: u64, to: u64, parent: &HashMap<u64, (u64, u64)>) -> Path {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut current = to;

        while current != from {
            nodes.push(current);
            if let Some((p, edge_id)) = parent.get(&current) {
                edges.push(*edge_id);
                current = *p;
            } else {
                break;
            }
        }
        nodes.push(from);

        nodes.reverse();
        edges.reverse();

        Path { nodes, edges }
    }

    /// Returns the number of nodes in the graph.
    ///
    /// Each node creates 3 keys: `node:N`, `node:N:out`, `node:N:in`.
    /// We count only keys matching `node:N` (no colons after the ID).
    pub fn node_count(&self) -> usize {
        self.store
            .scan("node:")
            .into_iter()
            .filter(|k| {
                // Only count "node:N" keys, not "node:N:out" or "node:N:in"
                let suffix = k.strip_prefix("node:").unwrap_or("");
                !suffix.contains(':')
            })
            .count()
    }

    /// Returns the number of edges in the graph (classic edge mode only).
    pub fn edge_count(&self) -> usize {
        self.store
            .scan("edge:")
            .into_iter()
            .filter(|k| {
                // Only count "edge:N" keys (classic mode), not "edge:type:N" (entity mode)
                let suffix = k.strip_prefix("edge:").unwrap_or("");
                suffix.parse::<u64>().is_ok()
            })
            .count()
    }

    /// Delete an edge by ID, cleaning up edge lists on both connected nodes.
    pub fn delete_edge(&self, edge_id: u64) -> Result<()> {
        let edge = self.get_edge(edge_id)?;

        // Unindex edge properties
        self.unindex_edge_properties(edge_id, &edge.edge_type, &edge.properties);

        // Remove from 'from' node's outgoing list
        self.remove_edge_from_list(&Self::outgoing_edges_key(edge.from), edge_id)?;

        // Remove from 'to' node's incoming list
        self.remove_edge_from_list(&Self::incoming_edges_key(edge.to), edge_id)?;

        // For undirected edges, also remove reverse connections
        if !edge.directed {
            self.remove_edge_from_list(&Self::outgoing_edges_key(edge.to), edge_id)?;
            self.remove_edge_from_list(&Self::incoming_edges_key(edge.from), edge_id)?;
        }

        // Delete the edge itself
        self.store.delete(&Self::edge_key(edge_id))?;
        Ok(())
    }

    fn remove_edge_from_list(&self, key: &str, edge_id: u64) -> Result<()> {
        if let Ok(mut tensor) = self.store.get(key) {
            let edge_key = format!("e{}", edge_id);
            tensor.remove(&edge_key);
            self.store.put(key, tensor)?;
        }
        Ok(())
    }

    pub fn delete_node(&self, id: u64) -> Result<()> {
        // Get node for index cleanup before deletion
        let node = self.get_node(id)?;

        // Get all edges connected to this node
        let out_edges = self.get_edge_list(&Self::outgoing_edges_key(id));
        let in_edges = self.get_edge_list(&Self::incoming_edges_key(id));

        // Collect unique edge IDs
        let mut all_edge_ids: HashSet<u64> = out_edges.into_iter().collect();
        all_edge_ids.extend(in_edges);

        // Delete each edge properly (removes from other nodes' edge lists)
        if all_edge_ids.len() >= Self::PARALLEL_THRESHOLD {
            // For high-degree nodes, batch the cleanup
            let edges_to_delete: Vec<_> = all_edge_ids.iter().copied().collect();
            edges_to_delete.par_iter().for_each(|edge_id| {
                if let Ok(edge) = self.get_edge(*edge_id) {
                    // Unindex edge properties
                    self.unindex_edge_properties(*edge_id, &edge.edge_type, &edge.properties);

                    // Clean up the OTHER node's edge list (not the one we're deleting)
                    let other_node = if edge.from == id { edge.to } else { edge.from };

                    if edge.from == id {
                        // We're the 'from' node, clean up 'to' node's incoming list
                        self.remove_edge_from_list(&Self::incoming_edges_key(other_node), *edge_id)
                            .ok();
                    }
                    if edge.to == id {
                        // We're the 'to' node, clean up 'from' node's outgoing list
                        self.remove_edge_from_list(&Self::outgoing_edges_key(other_node), *edge_id)
                            .ok();
                    }
                    // For undirected edges, also clean up reverse
                    if !edge.directed && other_node != id {
                        self.remove_edge_from_list(&Self::outgoing_edges_key(other_node), *edge_id)
                            .ok();
                        self.remove_edge_from_list(&Self::incoming_edges_key(other_node), *edge_id)
                            .ok();
                    }
                }
                // Delete the edge record
                self.store.delete(&Self::edge_key(*edge_id)).ok();
            });
        } else {
            for edge_id in all_edge_ids {
                if let Ok(edge) = self.get_edge(edge_id) {
                    // Unindex edge properties
                    self.unindex_edge_properties(edge_id, &edge.edge_type, &edge.properties);

                    let other_node = if edge.from == id { edge.to } else { edge.from };

                    if edge.from == id {
                        self.remove_edge_from_list(&Self::incoming_edges_key(other_node), edge_id)?;
                    }
                    if edge.to == id {
                        self.remove_edge_from_list(&Self::outgoing_edges_key(other_node), edge_id)?;
                    }
                    if !edge.directed && other_node != id {
                        self.remove_edge_from_list(&Self::outgoing_edges_key(other_node), edge_id)?;
                        self.remove_edge_from_list(&Self::incoming_edges_key(other_node), edge_id)?;
                    }
                }
                self.store.delete(&Self::edge_key(edge_id)).ok();
            }
        }

        // Unindex node properties
        self.unindex_node_properties(id, &node.label, &node.properties);

        // Delete the node itself and its edge lists
        self.store.delete(&Self::node_key(id))?;
        self.store.delete(&Self::outgoing_edges_key(id))?;
        self.store.delete(&Self::incoming_edges_key(id))?;

        Ok(())
    }

    // ========== Unified Entity Mode ==========
    // These methods work with entity keys directly (e.g., "user:1") and use
    // _out/_in fields for graph edges, enabling cross-engine queries.

    /// Get or create an entity for graph operations.
    fn get_or_create_entity(&self, key: &str) -> TensorData {
        self.store.get(key).unwrap_or_else(|_| TensorData::new())
    }

    /// Add an outgoing edge to an entity's _out field.
    pub fn add_entity_edge(&self, from_key: &str, to_key: &str, edge_type: &str) -> Result<String> {
        let edge_id = self.edge_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let edge_key = format!("edge:{}:{}", edge_type, edge_id);

        let mut edge_data = TensorData::new();
        edge_data.set(
            fields::TYPE,
            TensorValue::Scalar(ScalarValue::String("edge".into())),
        );
        edge_data.set(
            fields::FROM,
            TensorValue::Scalar(ScalarValue::String(from_key.into())),
        );
        edge_data.set(
            fields::TO,
            TensorValue::Scalar(ScalarValue::String(to_key.into())),
        );
        edge_data.set(
            fields::EDGE_TYPE,
            TensorValue::Scalar(ScalarValue::String(edge_type.into())),
        );
        edge_data.set(
            fields::DIRECTED,
            TensorValue::Scalar(ScalarValue::Bool(true)),
        );

        self.store.put(&edge_key, edge_data)?;

        let mut from_entity = self.get_or_create_entity(from_key);
        from_entity.add_outgoing_edge(edge_key.clone());
        self.store.put(from_key, from_entity)?;

        let mut to_entity = self.get_or_create_entity(to_key);
        to_entity.add_incoming_edge(edge_key.clone());
        self.store.put(to_key, to_entity)?;

        Ok(edge_key)
    }

    /// Add an undirected edge between two entities.
    pub fn add_entity_edge_undirected(
        &self,
        key1: &str,
        key2: &str,
        edge_type: &str,
    ) -> Result<String> {
        let edge_id = self.edge_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let edge_key = format!("edge:{}:{}", edge_type, edge_id);

        let mut edge_data = TensorData::new();
        edge_data.set(
            fields::TYPE,
            TensorValue::Scalar(ScalarValue::String("edge".into())),
        );
        edge_data.set(
            fields::FROM,
            TensorValue::Scalar(ScalarValue::String(key1.into())),
        );
        edge_data.set(
            fields::TO,
            TensorValue::Scalar(ScalarValue::String(key2.into())),
        );
        edge_data.set(
            fields::EDGE_TYPE,
            TensorValue::Scalar(ScalarValue::String(edge_type.into())),
        );
        edge_data.set(
            fields::DIRECTED,
            TensorValue::Scalar(ScalarValue::Bool(false)),
        );

        self.store.put(&edge_key, edge_data)?;

        let mut entity1 = self.get_or_create_entity(key1);
        entity1.add_outgoing_edge(edge_key.clone());
        entity1.add_incoming_edge(edge_key.clone());
        self.store.put(key1, entity1)?;

        let mut entity2 = self.get_or_create_entity(key2);
        entity2.add_outgoing_edge(edge_key.clone());
        entity2.add_incoming_edge(edge_key.clone());
        self.store.put(key2, entity2)?;

        Ok(edge_key)
    }

    /// Get outgoing edge keys for an entity.
    pub fn get_entity_outgoing(&self, key: &str) -> Result<Vec<String>> {
        let entity = self
            .store
            .get(key)
            .map_err(|_| GraphError::StorageError(format!("Entity not found: {}", key)))?;

        Ok(entity.outgoing_edges().cloned().unwrap_or_default())
    }

    /// Get incoming edge keys for an entity.
    pub fn get_entity_incoming(&self, key: &str) -> Result<Vec<String>> {
        let entity = self
            .store
            .get(key)
            .map_err(|_| GraphError::StorageError(format!("Entity not found: {}", key)))?;

        Ok(entity.incoming_edges().cloned().unwrap_or_default())
    }

    /// Get edge data by edge key.
    pub fn get_entity_edge(&self, edge_key: &str) -> Result<(String, String, String, bool)> {
        let edge = self
            .store
            .get(edge_key)
            .map_err(|_| GraphError::StorageError(format!("Edge not found: {}", edge_key)))?;

        let from = match edge.get(fields::FROM) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => String::new(),
        };
        let to = match edge.get(fields::TO) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => String::new(),
        };
        let edge_type = match edge.get(fields::EDGE_TYPE) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => String::new(),
        };
        let directed = match edge.get(fields::DIRECTED) {
            Some(TensorValue::Scalar(ScalarValue::Bool(b))) => *b,
            _ => true,
        };

        Ok((from, to, edge_type, directed))
    }

    /// Get outgoing neighbor entity keys.
    pub fn get_entity_neighbors_out(&self, key: &str) -> Result<Vec<String>> {
        let edges = self.get_entity_outgoing(key)?;
        let mut neighbors = Vec::new();

        for edge_key in edges {
            if let Ok((from, to, _, _)) = self.get_entity_edge(&edge_key) {
                if from == key {
                    neighbors.push(to);
                } else {
                    neighbors.push(from);
                }
            }
        }

        Ok(neighbors)
    }

    /// Get incoming neighbor entity keys.
    pub fn get_entity_neighbors_in(&self, key: &str) -> Result<Vec<String>> {
        let edges = self.get_entity_incoming(key)?;
        let mut neighbors = Vec::new();

        for edge_key in edges {
            if let Ok((from, to, _, _)) = self.get_entity_edge(&edge_key) {
                if to == key {
                    neighbors.push(from);
                } else {
                    neighbors.push(to);
                }
            }
        }

        Ok(neighbors)
    }

    /// Get all neighbor entity keys (both directions).
    pub fn get_entity_neighbors(&self, key: &str) -> Result<Vec<String>> {
        let mut neighbors = HashSet::new();

        for n in self.get_entity_neighbors_out(key)? {
            neighbors.insert(n);
        }
        for n in self.get_entity_neighbors_in(key)? {
            neighbors.insert(n);
        }

        neighbors.remove(key);
        Ok(neighbors.into_iter().collect())
    }

    /// Check if an entity has graph edges.
    pub fn entity_has_edges(&self, key: &str) -> bool {
        self.store.get(key).map(|e| e.has_edges()).unwrap_or(false)
    }

    /// Delete an edge by key, updating connected entities.
    pub fn delete_entity_edge(&self, edge_key: &str) -> Result<()> {
        let (from, to, _, _) = self.get_entity_edge(edge_key)?;

        if let Ok(mut from_entity) = self.store.get(&from) {
            if let Some(edges) = from_entity.outgoing_edges() {
                let filtered: Vec<String> =
                    edges.iter().filter(|e| *e != edge_key).cloned().collect();
                from_entity.set_outgoing_edges(filtered);
            }
            if let Some(edges) = from_entity.incoming_edges() {
                let filtered: Vec<String> =
                    edges.iter().filter(|e| *e != edge_key).cloned().collect();
                from_entity.set_incoming_edges(filtered);
            }
            self.store.put(&from, from_entity)?;
        }

        if from != to {
            if let Ok(mut to_entity) = self.store.get(&to) {
                if let Some(edges) = to_entity.outgoing_edges() {
                    let filtered: Vec<String> =
                        edges.iter().filter(|e| *e != edge_key).cloned().collect();
                    to_entity.set_outgoing_edges(filtered);
                }
                if let Some(edges) = to_entity.incoming_edges() {
                    let filtered: Vec<String> =
                        edges.iter().filter(|e| *e != edge_key).cloned().collect();
                    to_entity.set_incoming_edges(filtered);
                }
                self.store.put(&to, to_entity)?;
            }
        }

        self.store.delete(edge_key)?;
        Ok(())
    }

    /// Scan for entities with graph edges.
    pub fn scan_entities_with_edges(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| self.entity_has_edges(key))
            .collect()
    }
}

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_node_and_retrieve() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Alice".into()));
        props.insert("age".to_string(), PropertyValue::Int(30));

        let id = engine.create_node("Person", props).unwrap();
        assert_eq!(id, 1);

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.label, "Person");
        assert_eq!(
            node.properties.get("name"),
            Some(&PropertyValue::String("Alice".into()))
        );
    }

    #[test]
    fn create_edge_between_nodes() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("since".to_string(), PropertyValue::Int(2020));

        let edge_id = engine.create_edge(n1, n2, "KNOWS", props, true).unwrap();
        assert_eq!(edge_id, 1);

        let edge = engine.get_edge(edge_id).unwrap();
        assert_eq!(edge.from, n1);
        assert_eq!(edge.to, n2);
        assert_eq!(edge.edge_type, "KNOWS");
        assert!(edge.directed);
    }

    #[test]
    fn create_edge_fails_for_nonexistent_node() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();

        let result = engine.create_edge(n1, 999, "KNOWS", HashMap::new(), true);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));

        let result = engine.create_edge(999, n1, "KNOWS", HashMap::new(), true);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn neighbors_directed_edge() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "KNOWS", HashMap::new(), true)
            .unwrap();

        let out_neighbors = engine.neighbors(n1, None, Direction::Outgoing).unwrap();
        assert_eq!(out_neighbors.len(), 2);

        let in_neighbors = engine.neighbors(n1, None, Direction::Incoming).unwrap();
        assert_eq!(in_neighbors.len(), 0);

        let n2_in = engine.neighbors(n2, None, Direction::Incoming).unwrap();
        assert_eq!(n2_in.len(), 1);
        assert_eq!(n2_in[0].id, n1);
    }

    #[test]
    fn neighbors_undirected_edge() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "FRIENDS", HashMap::new(), false)
            .unwrap();

        let n1_neighbors = engine.neighbors(n1, None, Direction::Both).unwrap();
        assert_eq!(n1_neighbors.len(), 1);
        assert_eq!(n1_neighbors[0].id, n2);

        let n2_neighbors = engine.neighbors(n2, None, Direction::Both).unwrap();
        assert_eq!(n2_neighbors.len(), 1);
        assert_eq!(n2_neighbors[0].id, n1);
    }

    #[test]
    fn neighbors_by_edge_type() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Company", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "WORKS_AT", HashMap::new(), true)
            .unwrap();

        let knows = engine
            .neighbors(n1, Some("KNOWS"), Direction::Outgoing)
            .unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].id, n2);

        let works = engine
            .neighbors(n1, Some("WORKS_AT"), Direction::Outgoing)
            .unwrap();
        assert_eq!(works.len(), 1);
        assert_eq!(works[0].id, n3);
    }

    #[test]
    fn traverse_bfs() {
        let engine = GraphEngine::new();

        // Create a chain: n1 -> n2 -> n3 -> n4
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();
        let n4 = engine.create_node("D", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n4, "NEXT", HashMap::new(), true)
            .unwrap();

        // Depth 0: only start node
        let result = engine.traverse(n1, Direction::Outgoing, 0, None).unwrap();
        assert_eq!(result.len(), 1);

        // Depth 1: start + direct neighbors
        let result = engine.traverse(n1, Direction::Outgoing, 1, None).unwrap();
        assert_eq!(result.len(), 2);

        // Depth 3: all nodes
        let result = engine.traverse(n1, Direction::Outgoing, 3, None).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn traverse_handles_cycles() {
        let engine = GraphEngine::new();

        // Create a cycle: n1 -> n2 -> n3 -> n1
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n1, "NEXT", HashMap::new(), true)
            .unwrap();

        // Should not infinite loop, should visit each node once
        let result = engine.traverse(n1, Direction::Outgoing, 10, None).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn find_path_simple() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "NEXT", HashMap::new(), true)
            .unwrap();

        let path = engine.find_path(n1, n3).unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]);
        assert_eq!(path.edges.len(), 2);
    }

    #[test]
    fn find_path_same_node() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        let path = engine.find_path(n1, n1).unwrap();
        assert_eq!(path.nodes, vec![n1]);
        assert!(path.edges.is_empty());
    }

    #[test]
    fn find_path_not_found() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let result = engine.find_path(n1, n2);
        assert!(matches!(result, Err(GraphError::PathNotFound)));
    }

    #[test]
    fn find_path_shortest() {
        let engine = GraphEngine::new();

        // Create graph: n1 -> n2 -> n4 (short path)
        //               n1 -> n3 -> n2 -> n4 (long path)
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();
        let n4 = engine.create_node("D", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n4, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n2, "NEXT", HashMap::new(), true)
            .unwrap();

        let path = engine.find_path(n1, n4).unwrap();
        // BFS should find shortest path: n1 -> n2 -> n4
        assert_eq!(path.nodes.len(), 3);
        assert_eq!(path.nodes, vec![n1, n2, n4]);
    }

    #[test]
    fn create_1000_nodes_with_edges_traverse() {
        let engine = GraphEngine::new();

        // Create 1000 nodes
        let mut node_ids = Vec::new();
        for i in 0..1000 {
            let mut props = HashMap::new();
            props.insert("index".to_string(), PropertyValue::Int(i));
            let id = engine.create_node("Node", props).unwrap();
            node_ids.push(id);
        }

        // Create chain of edges: 0 -> 1 -> 2 -> ... -> 999
        for i in 0..999 {
            engine
                .create_edge(node_ids[i], node_ids[i + 1], "NEXT", HashMap::new(), true)
                .unwrap();
        }

        // Traverse from node 0 with depth 10 should get 11 nodes
        let result = engine
            .traverse(node_ids[0], Direction::Outgoing, 10, None)
            .unwrap();
        assert_eq!(result.len(), 11);

        // Traverse full chain
        let result = engine
            .traverse(node_ids[0], Direction::Outgoing, 1000, None)
            .unwrap();
        assert_eq!(result.len(), 1000);

        // Find path from 0 to 50
        let path = engine.find_path(node_ids[0], node_ids[50]).unwrap();
        assert_eq!(path.nodes.len(), 51);
    }

    #[test]
    fn directed_vs_undirected_edges() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        // Directed edge: n1 -> n2
        engine
            .create_edge(n1, n2, "DIRECTED", HashMap::new(), true)
            .unwrap();

        // Undirected edge: n1 -- n3
        engine
            .create_edge(n1, n3, "UNDIRECTED", HashMap::new(), false)
            .unwrap();

        // From n1: can reach both n2 (directed out) and n3 (undirected)
        let from_n1 = engine.neighbors(n1, None, Direction::Outgoing).unwrap();
        assert_eq!(from_n1.len(), 2);

        // From n2: cannot reach n1 (directed edge goes other way)
        let from_n2 = engine.neighbors(n2, None, Direction::Outgoing).unwrap();
        assert_eq!(from_n2.len(), 0);

        // From n3: can reach n1 (undirected)
        let from_n3 = engine.neighbors(n3, None, Direction::Outgoing).unwrap();
        assert_eq!(from_n3.len(), 1);
        assert_eq!(from_n3[0].id, n1);
    }

    #[test]
    fn delete_node() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        assert!(engine.node_exists(n1));
        engine.delete_node(n1).unwrap();
        assert!(!engine.node_exists(n1));

        // Edge should also be deleted
        let result = engine.get_edge(1);
        assert!(matches!(result, Err(GraphError::EdgeNotFound(_))));
    }

    #[test]
    fn error_display() {
        let e1 = GraphError::NodeNotFound(1);
        assert!(format!("{}", e1).contains("1"));

        let e2 = GraphError::EdgeNotFound(2);
        assert!(format!("{}", e2).contains("2"));

        let e3 = GraphError::StorageError("disk".into());
        assert!(format!("{}", e3).contains("disk"));

        let e4 = GraphError::PathNotFound;
        assert!(format!("{}", e4).contains("path"));
    }

    #[test]
    fn engine_default_trait() {
        let engine = GraphEngine::default();
        assert!(!engine.node_exists(1));
    }

    #[test]
    fn property_value_conversions() {
        let values = vec![
            PropertyValue::Null,
            PropertyValue::Int(42),
            PropertyValue::Float(3.14),
            PropertyValue::String("test".into()),
            PropertyValue::Bool(true),
        ];

        for v in values {
            let scalar = v.to_scalar();
            let back = PropertyValue::from_scalar(&scalar);
            assert_eq!(v, back);
        }

        // Bytes converts to Null
        let bytes = ScalarValue::Bytes(vec![1, 2, 3]);
        assert_eq!(PropertyValue::from_scalar(&bytes), PropertyValue::Null);
    }

    #[test]
    fn direction_equality() {
        assert_eq!(Direction::Outgoing, Direction::Outgoing);
        assert_eq!(Direction::Incoming, Direction::Incoming);
        assert_eq!(Direction::Both, Direction::Both);
        assert_ne!(Direction::Outgoing, Direction::Incoming);
    }

    #[test]
    fn neighbors_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.neighbors(999, None, Direction::Both);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn traverse_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.traverse(999, Direction::Both, 5, None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn find_path_nonexistent_node() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.find_path(999, n1);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));

        let result = engine.find_path(n1, 999);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn self_loop_edge() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n1, "SELF", HashMap::new(), true)
            .unwrap();

        // Self-loop shouldn't appear in neighbors (we filter self)
        let neighbors = engine.neighbors(n1, None, Direction::Both).unwrap();
        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn with_store_constructor() {
        let store = TensorStore::new();
        let engine = GraphEngine::with_store(store);
        assert!(!engine.node_exists(1));
    }

    #[test]
    fn error_from_tensor_store() {
        let err = TensorStoreError::NotFound("key".into());
        let graph_err: GraphError = err.into();
        assert!(matches!(graph_err, GraphError::StorageError(_)));
    }

    #[test]
    fn error_is_error_trait() {
        let err: &dyn std::error::Error = &GraphError::PathNotFound;
        assert!(err.to_string().contains("path"));
    }

    #[test]
    fn clone_types() {
        let node = Node {
            id: 1,
            label: "Test".into(),
            properties: HashMap::new(),
        };
        let _ = node.clone();

        let edge = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "TEST".into(),
            properties: HashMap::new(),
            directed: true,
        };
        let _ = edge.clone();

        let path = Path {
            nodes: vec![1, 2],
            edges: vec![1],
        };
        let _ = path.clone();

        let err = GraphError::PathNotFound;
        let _ = err.clone();
    }

    #[test]
    fn node_count() {
        let engine = GraphEngine::new();

        // Empty graph
        assert_eq!(engine.node_count(), 0);

        // Add some nodes
        engine.create_node("A", HashMap::new()).unwrap();
        engine.create_node("B", HashMap::new()).unwrap();
        engine.create_node("C", HashMap::new()).unwrap();

        assert_eq!(engine.node_count(), 3);
    }

    #[test]
    fn traverse_incoming_direction() {
        let engine = GraphEngine::new();

        // Create: n1 -> n2 -> n3
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "NEXT", HashMap::new(), true)
            .unwrap();

        // Traverse incoming from n3 should find n2 and n1
        let result = engine.traverse(n3, Direction::Incoming, 10, None).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn neighbors_incoming_with_edge_type() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n3, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "WORKS_WITH", HashMap::new(), true)
            .unwrap();

        // Get only KNOWS incoming neighbors of n3
        let knows = engine
            .neighbors(n3, Some("KNOWS"), Direction::Incoming)
            .unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].id, n1);

        // Get only WORKS_WITH incoming neighbors of n3
        let works = engine
            .neighbors(n3, Some("WORKS_WITH"), Direction::Incoming)
            .unwrap();
        assert_eq!(works.len(), 1);
        assert_eq!(works[0].id, n2);
    }

    #[test]
    fn find_path_through_undirected() {
        let engine = GraphEngine::new();

        // Create: n1 -- n2 -- n3 (undirected)
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "CONN", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(n2, n3, "CONN", HashMap::new(), false)
            .unwrap();

        // Should find path n1 -> n2 -> n3
        let path = engine.find_path(n1, n3).unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]);

        // Should also find reverse path n3 -> n2 -> n1
        let path_rev = engine.find_path(n3, n1).unwrap();
        assert_eq!(path_rev.nodes, vec![n3, n2, n1]);
    }

    #[test]
    fn delete_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.delete_node(999);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn traverse_incoming_only() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        // n1 -> n2, n3 -> n2 (both point to n2)
        engine
            .create_edge(n1, n2, "POINTS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n2, "POINTS", HashMap::new(), true)
            .unwrap();

        // Traverse incoming from n2 with depth 1
        let result = engine.traverse(n2, Direction::Incoming, 1, None).unwrap();
        assert_eq!(result.len(), 3); // n2 + n1 + n3
    }

    #[test]
    fn get_neighbor_ids_incoming_undirected() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Undirected edge from n1 to n2
        engine
            .create_edge(n1, n2, "LINK", HashMap::new(), false)
            .unwrap();

        // From n2's perspective with Incoming direction,
        // should still see n1 via the undirected edge
        let neighbors = engine.neighbors(n2, None, Direction::Incoming).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, n1);
    }

    #[test]
    fn delete_high_degree_node_parallel() {
        let engine = GraphEngine::new();

        // Create a hub node with >100 edges to trigger parallel deletion
        let hub = engine.create_node("hub", HashMap::new()).unwrap();

        // Create 150 leaf nodes connected to the hub
        for i in 0..150 {
            let leaf = engine
                .create_node(&format!("leaf{}", i), HashMap::new())
                .unwrap();
            engine
                .create_edge(hub, leaf, "CONNECTS", HashMap::new(), true)
                .unwrap();
        }

        // Verify hub has 150 outgoing edges
        let neighbors = engine.neighbors(hub, None, Direction::Outgoing).unwrap();
        assert_eq!(neighbors.len(), 150);

        // Delete hub node (should use parallel edge deletion)
        engine.delete_node(hub).unwrap();
        assert!(!engine.node_exists(hub));
    }

    // Unified Entity Mode tests

    #[test]
    fn entity_edge_directed() {
        let engine = GraphEngine::new();

        let edge_key = engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();

        assert!(edge_key.starts_with("edge:follows:"));

        let outgoing = engine.get_entity_outgoing("user:1").unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0], edge_key);

        let incoming = engine.get_entity_incoming("user:2").unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0], edge_key);
    }

    #[test]
    fn entity_edge_undirected() {
        let engine = GraphEngine::new();

        let edge_key = engine
            .add_entity_edge_undirected("user:1", "user:2", "friend")
            .unwrap();

        let out1 = engine.get_entity_outgoing("user:1").unwrap();
        let in1 = engine.get_entity_incoming("user:1").unwrap();
        let out2 = engine.get_entity_outgoing("user:2").unwrap();
        let in2 = engine.get_entity_incoming("user:2").unwrap();

        assert!(out1.contains(&edge_key));
        assert!(in1.contains(&edge_key));
        assert!(out2.contains(&edge_key));
        assert!(in2.contains(&edge_key));
    }

    #[test]
    fn entity_get_edge() {
        let engine = GraphEngine::new();

        let edge_key = engine
            .add_entity_edge("user:1", "post:1", "created")
            .unwrap();

        let (from, to, edge_type, directed) = engine.get_entity_edge(&edge_key).unwrap();
        assert_eq!(from, "user:1");
        assert_eq!(to, "post:1");
        assert_eq!(edge_type, "created");
        assert!(directed);
    }

    #[test]
    fn entity_neighbors_out() {
        let engine = GraphEngine::new();

        engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();
        engine
            .add_entity_edge("user:1", "user:3", "follows")
            .unwrap();

        let neighbors = engine.get_entity_neighbors_out("user:1").unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"user:2".to_string()));
        assert!(neighbors.contains(&"user:3".to_string()));
    }

    #[test]
    fn entity_neighbors_in() {
        let engine = GraphEngine::new();

        engine
            .add_entity_edge("user:1", "user:3", "follows")
            .unwrap();
        engine
            .add_entity_edge("user:2", "user:3", "follows")
            .unwrap();

        let neighbors = engine.get_entity_neighbors_in("user:3").unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"user:1".to_string()));
        assert!(neighbors.contains(&"user:2".to_string()));
    }

    #[test]
    fn entity_neighbors_both() {
        let engine = GraphEngine::new();

        engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();
        engine
            .add_entity_edge("user:3", "user:2", "follows")
            .unwrap();

        let neighbors = engine.get_entity_neighbors("user:2").unwrap();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn entity_has_edges() {
        let engine = GraphEngine::new();

        assert!(!engine.entity_has_edges("user:1"));

        engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();
        assert!(engine.entity_has_edges("user:1"));
        assert!(engine.entity_has_edges("user:2"));
    }

    #[test]
    fn entity_delete_edge() {
        let engine = GraphEngine::new();

        let edge_key = engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();

        engine.delete_entity_edge(&edge_key).unwrap();

        let outgoing = engine.get_entity_outgoing("user:1").unwrap();
        assert!(outgoing.is_empty());

        let incoming = engine.get_entity_incoming("user:2").unwrap();
        assert!(incoming.is_empty());
    }

    #[test]
    fn entity_preserves_other_fields() {
        let store = TensorStore::new();

        let mut user = TensorData::new();
        user.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        store.put("user:1", user).unwrap();

        let engine = GraphEngine::with_store(store);
        engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();

        let entity = engine.store.get("user:1").unwrap();
        assert!(entity.has("name"));
        assert!(entity.has(fields::OUT));
    }

    #[test]
    fn entity_scan_with_edges() {
        let engine = GraphEngine::new();

        engine
            .add_entity_edge("user:1", "user:2", "follows")
            .unwrap();
        engine
            .add_entity_edge("user:3", "user:4", "follows")
            .unwrap();

        let with_edges = engine.scan_entities_with_edges();
        assert_eq!(with_edges.len(), 4);
    }

    #[test]
    fn entity_edge_nonexistent_returns_error() {
        let engine = GraphEngine::new();
        let result = engine.get_entity_edge("nonexistent:edge");
        assert!(result.is_err());
    }

    #[test]
    fn entity_outgoing_nonexistent_returns_error() {
        let engine = GraphEngine::new();
        let result = engine.get_entity_outgoing("nonexistent:entity");
        assert!(result.is_err());
    }

    // New tests for production readiness fixes

    #[test]
    fn with_store_initializes_counters() {
        let store = TensorStore::new();

        // Pre-populate with nodes and edges
        let engine1 = GraphEngine::with_store(store);
        let n1 = engine1.create_node("A", HashMap::new()).unwrap();
        let n2 = engine1.create_node("B", HashMap::new()).unwrap();
        engine1
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        assert_eq!(n1, 1);
        assert_eq!(n2, 2);

        // Create a new engine with the same store
        let store2 = engine1.store().clone();
        let engine2 = GraphEngine::with_store(store2);

        // New IDs should not collide
        let n3 = engine2.create_node("C", HashMap::new()).unwrap();
        assert_eq!(n3, 3);

        let edge2 = engine2
            .create_edge(n2, n3, "KNOWS", HashMap::new(), true)
            .unwrap();
        assert_eq!(edge2, 2);
    }

    #[test]
    fn update_node_label() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .update_node(id, Some("Employee"), HashMap::new())
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.label, "Employee");
    }

    #[test]
    fn update_node_properties() {
        let engine = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Alice".into()));
        props.insert("age".to_string(), PropertyValue::Int(30));

        let id = engine.create_node("Person", props).unwrap();

        // Update and add properties
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), PropertyValue::Int(31));
        updates.insert("city".to_string(), PropertyValue::String("NYC".into()));
        engine.update_node(id, None, updates).unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.label, "Person"); // unchanged
        assert_eq!(
            node.properties.get("name"),
            Some(&PropertyValue::String("Alice".into()))
        );
        assert_eq!(node.properties.get("age"), Some(&PropertyValue::Int(31)));
        assert_eq!(
            node.properties.get("city"),
            Some(&PropertyValue::String("NYC".into()))
        );
    }

    #[test]
    fn update_node_remove_property() {
        let engine = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Alice".into()));
        props.insert(
            "temp".to_string(),
            PropertyValue::String("remove me".into()),
        );

        let id = engine.create_node("Person", props).unwrap();

        // Remove property by setting to Null
        let mut updates = HashMap::new();
        updates.insert("temp".to_string(), PropertyValue::Null);
        engine.update_node(id, None, updates).unwrap();

        let node = engine.get_node(id).unwrap();
        assert!(node.properties.get("name").is_some());
        assert!(node.properties.get("temp").is_none());
    }

    #[test]
    fn update_node_nonexistent() {
        let engine = GraphEngine::new();
        let result = engine.update_node(999, None, HashMap::new());
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn update_edge_properties() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        let edge_id = engine.create_edge(n1, n2, "CONNECTS", props, true).unwrap();

        // Update edge properties
        let mut updates = HashMap::new();
        updates.insert("weight".to_string(), PropertyValue::Float(2.5));
        updates.insert(
            "label".to_string(),
            PropertyValue::String("important".into()),
        );
        engine.update_edge(edge_id, updates).unwrap();

        let edge = engine.get_edge(edge_id).unwrap();
        assert_eq!(
            edge.properties.get("weight"),
            Some(&PropertyValue::Float(2.5))
        );
        assert_eq!(
            edge.properties.get("label"),
            Some(&PropertyValue::String("important".into()))
        );
    }

    #[test]
    fn update_edge_nonexistent() {
        let engine = GraphEngine::new();
        let result = engine.update_edge(999, HashMap::new());
        assert!(matches!(result, Err(GraphError::EdgeNotFound(999))));
    }

    #[test]
    fn delete_edge_directed() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let edge_id = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        // Verify edge exists
        assert!(engine.get_edge(edge_id).is_ok());
        assert_eq!(
            engine
                .neighbors(n1, None, Direction::Outgoing)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            engine
                .neighbors(n2, None, Direction::Incoming)
                .unwrap()
                .len(),
            1
        );

        // Delete edge
        engine.delete_edge(edge_id).unwrap();

        // Verify edge is gone
        assert!(matches!(
            engine.get_edge(edge_id),
            Err(GraphError::EdgeNotFound(_))
        ));
        assert_eq!(
            engine
                .neighbors(n1, None, Direction::Outgoing)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            engine
                .neighbors(n2, None, Direction::Incoming)
                .unwrap()
                .len(),
            0
        );

        // Nodes should still exist
        assert!(engine.node_exists(n1));
        assert!(engine.node_exists(n2));
    }

    #[test]
    fn delete_edge_undirected() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let edge_id = engine
            .create_edge(n1, n2, "FRIENDS", HashMap::new(), false)
            .unwrap();

        // Both directions should work
        assert_eq!(
            engine.neighbors(n1, None, Direction::Both).unwrap().len(),
            1
        );
        assert_eq!(
            engine.neighbors(n2, None, Direction::Both).unwrap().len(),
            1
        );

        // Delete edge
        engine.delete_edge(edge_id).unwrap();

        // Both directions should be empty
        assert_eq!(
            engine.neighbors(n1, None, Direction::Both).unwrap().len(),
            0
        );
        assert_eq!(
            engine.neighbors(n2, None, Direction::Both).unwrap().len(),
            0
        );
    }

    #[test]
    fn delete_edge_nonexistent() {
        let engine = GraphEngine::new();
        let result = engine.delete_edge(999);
        assert!(matches!(result, Err(GraphError::EdgeNotFound(999))));
    }

    #[test]
    fn edge_count() {
        let engine = GraphEngine::new();

        assert_eq!(engine.edge_count(), 0);

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "E1", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "E2", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "E3", HashMap::new(), false)
            .unwrap();

        assert_eq!(engine.edge_count(), 3);
    }

    #[test]
    fn edges_of_node() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        let e1 = engine
            .create_edge(n1, n2, "OUT", HashMap::new(), true)
            .unwrap();
        let e2 = engine
            .create_edge(n3, n1, "IN", HashMap::new(), true)
            .unwrap();

        // Outgoing only
        let out_edges = engine.edges_of(n1, Direction::Outgoing).unwrap();
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].id, e1);

        // Incoming only
        let in_edges = engine.edges_of(n1, Direction::Incoming).unwrap();
        assert_eq!(in_edges.len(), 1);
        assert_eq!(in_edges[0].id, e2);

        // Both
        let all_edges = engine.edges_of(n1, Direction::Both).unwrap();
        assert_eq!(all_edges.len(), 2);
    }

    #[test]
    fn edges_of_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.edges_of(999, Direction::Both);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn delete_node_cleans_up_other_nodes_edge_lists() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        // n1 -> n2 -> n3
        engine
            .create_edge(n1, n2, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "E", HashMap::new(), true)
            .unwrap();

        // Delete n2 (the middle node)
        engine.delete_node(n2).unwrap();

        // n1 should have no outgoing neighbors now
        assert_eq!(
            engine
                .neighbors(n1, None, Direction::Outgoing)
                .unwrap()
                .len(),
            0
        );

        // n3 should have no incoming neighbors now
        assert_eq!(
            engine
                .neighbors(n3, None, Direction::Incoming)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn graph_engine_debug() {
        let engine = GraphEngine::new();
        let debug_str = format!("{:?}", engine);
        assert!(debug_str.contains("GraphEngine"));
        assert!(debug_str.contains("node_counter"));
    }

    #[test]
    fn node_edge_equality() {
        let node1 = Node {
            id: 1,
            label: "Test".into(),
            properties: HashMap::new(),
        };
        let node2 = Node {
            id: 1,
            label: "Test".into(),
            properties: HashMap::new(),
        };
        assert_eq!(node1, node2);

        let edge1 = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "E".into(),
            properties: HashMap::new(),
            directed: true,
        };
        let edge2 = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "E".into(),
            properties: HashMap::new(),
            directed: true,
        };
        assert_eq!(edge1, edge2);
    }

    #[test]
    fn path_equality() {
        let path1 = Path {
            nodes: vec![1, 2, 3],
            edges: vec![1, 2],
        };
        let path2 = Path {
            nodes: vec![1, 2, 3],
            edges: vec![1, 2],
        };
        assert_eq!(path1, path2);
    }

    #[test]
    fn graph_error_hash() {
        use std::collections::HashSet;
        let mut errors = HashSet::new();
        errors.insert(GraphError::NodeNotFound(1));
        errors.insert(GraphError::NodeNotFound(1));
        errors.insert(GraphError::EdgeNotFound(2));
        assert_eq!(errors.len(), 2);
    }

    // ========== Property Index Tests ==========

    #[test]
    fn create_node_property_index() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        engine.create_node("Person", props).unwrap();

        let mut props2 = HashMap::new();
        props2.insert("age".to_string(), PropertyValue::Int(25));
        engine.create_node("Person", props2).unwrap();

        // Create index
        engine.create_node_property_index("age").unwrap();
        assert!(engine.has_node_index("age"));

        // Query using index
        let nodes = engine
            .find_nodes_by_property("age", &PropertyValue::Int(30))
            .unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].id, 1);
    }

    #[test]
    fn create_edge_property_index() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.5));
        engine.create_edge(n1, n2, "CONNECTS", props, true).unwrap();

        // Create index
        engine.create_edge_property_index("weight").unwrap();
        assert!(engine.has_edge_index("weight"));

        // Query using index
        let edges = engine
            .find_edges_by_property("weight", &PropertyValue::Float(1.5))
            .unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn create_label_index() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Company", HashMap::new()).unwrap();

        engine.create_label_index().unwrap();
        assert!(engine.has_node_index("_label"));

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 2);

        let companies = engine.find_nodes_by_label("Company").unwrap();
        assert_eq!(companies.len(), 1);
    }

    #[test]
    fn create_edge_type_index() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        engine.create_edge_type_index().unwrap();
        assert!(engine.has_edge_index("_edge_type"));

        let knows = engine.find_edges_by_type("KNOWS").unwrap();
        assert_eq!(knows.len(), 1);

        let follows = engine.find_edges_by_type("FOLLOWS").unwrap();
        assert_eq!(follows.len(), 1);
    }

    #[test]
    fn drop_node_index() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_label_index().unwrap();

        assert!(engine.has_node_index("_label"));
        engine.drop_node_index("_label").unwrap();
        assert!(!engine.has_node_index("_label"));
    }

    #[test]
    fn drop_edge_index() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        engine.create_edge_type_index().unwrap();
        assert!(engine.has_edge_index("_edge_type"));

        engine.drop_edge_index("_edge_type").unwrap();
        assert!(!engine.has_edge_index("_edge_type"));
    }

    #[test]
    fn index_already_exists_error() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();
        let result = engine.create_label_index();

        assert!(matches!(result, Err(GraphError::IndexAlreadyExists { .. })));
    }

    #[test]
    fn drop_nonexistent_index_error() {
        let engine = GraphEngine::new();

        let result = engine.drop_node_index("nonexistent");
        assert!(matches!(result, Err(GraphError::IndexNotFound { .. })));
    }

    #[test]
    fn find_nodes_by_property_int() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        engine.create_node("Person", props).unwrap();

        engine.create_node_property_index("age").unwrap();

        let nodes = engine
            .find_nodes_by_property("age", &PropertyValue::Int(30))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn find_nodes_by_property_string() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        engine.create_node("Person", props).unwrap();

        engine.create_node_property_index("name").unwrap();

        let nodes = engine
            .find_nodes_by_property("name", &PropertyValue::String("Alice".to_string()))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn find_nodes_by_property_float() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("score".to_string(), PropertyValue::Float(3.14));
        engine.create_node("Person", props).unwrap();

        engine.create_node_property_index("score").unwrap();

        let nodes = engine
            .find_nodes_by_property("score", &PropertyValue::Float(3.14))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn find_nodes_by_property_bool() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("active".to_string(), PropertyValue::Bool(true));
        engine.create_node("Person", props).unwrap();

        engine.create_node_property_index("active").unwrap();

        let nodes = engine
            .find_nodes_by_property("active", &PropertyValue::Bool(true))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn find_nodes_where_lt() {
        let engine = GraphEngine::new();

        for i in 1..=10 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Node", props).unwrap();
        }

        engine.create_node_property_index("value").unwrap();

        let nodes = engine
            .find_nodes_where("value", RangeOp::Lt, &PropertyValue::Int(5))
            .unwrap();
        assert_eq!(nodes.len(), 4); // 1, 2, 3, 4
    }

    #[test]
    fn find_nodes_where_le() {
        let engine = GraphEngine::new();

        for i in 1..=10 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Node", props).unwrap();
        }

        engine.create_node_property_index("value").unwrap();

        let nodes = engine
            .find_nodes_where("value", RangeOp::Le, &PropertyValue::Int(5))
            .unwrap();
        assert_eq!(nodes.len(), 5); // 1, 2, 3, 4, 5
    }

    #[test]
    fn find_nodes_where_gt() {
        let engine = GraphEngine::new();

        for i in 1..=10 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Node", props).unwrap();
        }

        engine.create_node_property_index("value").unwrap();

        let nodes = engine
            .find_nodes_where("value", RangeOp::Gt, &PropertyValue::Int(5))
            .unwrap();
        assert_eq!(nodes.len(), 5); // 6, 7, 8, 9, 10
    }

    #[test]
    fn find_nodes_where_ge() {
        let engine = GraphEngine::new();

        for i in 1..=10 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Node", props).unwrap();
        }

        engine.create_node_property_index("value").unwrap();

        let nodes = engine
            .find_nodes_where("value", RangeOp::Ge, &PropertyValue::Int(5))
            .unwrap();
        assert_eq!(nodes.len(), 6); // 5, 6, 7, 8, 9, 10
    }

    #[test]
    fn index_updated_on_create_node() {
        let engine = GraphEngine::new();

        // Create index first
        engine.create_label_index().unwrap();

        // Create node after index exists
        engine.create_node("Person", HashMap::new()).unwrap();

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 1);
    }

    #[test]
    fn index_updated_on_update_node() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        let id = engine.create_node("Person", props).unwrap();

        engine.create_node_property_index("age").unwrap();

        // Verify initial state
        let nodes = engine
            .find_nodes_by_property("age", &PropertyValue::Int(30))
            .unwrap();
        assert_eq!(nodes.len(), 1);

        // Update the property
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), PropertyValue::Int(31));
        engine.update_node(id, None, updates).unwrap();

        // Old value should not be found
        let nodes = engine
            .find_nodes_by_property("age", &PropertyValue::Int(30))
            .unwrap();
        assert_eq!(nodes.len(), 0);

        // New value should be found
        let nodes = engine
            .find_nodes_by_property("age", &PropertyValue::Int(31))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn index_updated_on_delete_node() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 1);

        engine.delete_node(id).unwrap();

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 0);
    }

    #[test]
    fn index_updated_on_create_edge() {
        let engine = GraphEngine::new();

        engine.create_edge_type_index().unwrap();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        let edges = engine.find_edges_by_type("KNOWS").unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn index_updated_on_update_edge() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        let edge_id = engine.create_edge(n1, n2, "CONN", props, true).unwrap();

        engine.create_edge_property_index("weight").unwrap();

        // Verify initial
        let edges = engine
            .find_edges_by_property("weight", &PropertyValue::Float(1.0))
            .unwrap();
        assert_eq!(edges.len(), 1);

        // Update
        let mut updates = HashMap::new();
        updates.insert("weight".to_string(), PropertyValue::Float(2.0));
        engine.update_edge(edge_id, updates).unwrap();

        // Old value gone
        let edges = engine
            .find_edges_by_property("weight", &PropertyValue::Float(1.0))
            .unwrap();
        assert_eq!(edges.len(), 0);

        // New value present
        let edges = engine
            .find_edges_by_property("weight", &PropertyValue::Float(2.0))
            .unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn index_updated_on_delete_edge() {
        let engine = GraphEngine::new();

        engine.create_edge_type_index().unwrap();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let edge_id = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        let edges = engine.find_edges_by_type("KNOWS").unwrap();
        assert_eq!(edges.len(), 1);

        engine.delete_edge(edge_id).unwrap();

        let edges = engine.find_edges_by_type("KNOWS").unwrap();
        assert_eq!(edges.len(), 0);
    }

    #[test]
    fn index_rebuilt_from_store() {
        let store = TensorStore::new();
        let engine1 = GraphEngine::with_store(store);

        // Create data and index
        engine1.create_node("Person", HashMap::new()).unwrap();
        engine1.create_node("Person", HashMap::new()).unwrap();
        engine1.create_label_index().unwrap();

        // Verify index works
        let persons = engine1.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 2);

        // Create new engine from same store (simulates restart)
        let store2 = engine1.store().clone();
        let engine2 = GraphEngine::with_store(store2);

        // Index should be rebuilt
        assert!(engine2.has_node_index("_label"));
        let persons = engine2.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn query_without_index_falls_back_to_scan() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        engine.create_node("Person", props).unwrap();

        // No index created - should fall back to scan
        let nodes = engine
            .find_nodes_by_property("age", &PropertyValue::Int(30))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn range_query_without_index_falls_back_to_scan() {
        let engine = GraphEngine::new();

        for i in 1..=5 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Node", props).unwrap();
        }

        // No index - should fall back to scan
        let nodes = engine
            .find_nodes_where("value", RangeOp::Lt, &PropertyValue::Int(3))
            .unwrap();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn find_no_match_returns_empty() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();
        engine.create_node("Person", HashMap::new()).unwrap();

        let companies = engine.find_nodes_by_label("Company").unwrap();
        assert!(companies.is_empty());
    }

    #[test]
    fn find_multiple_matches() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();

        for _ in 0..5 {
            engine.create_node("Person", HashMap::new()).unwrap();
        }

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 5);
    }

    #[test]
    fn float_nan_handling() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("value".to_string(), PropertyValue::Float(f64::NAN));
        engine.create_node("Node", props).unwrap();

        engine.create_node_property_index("value").unwrap();

        let nodes = engine
            .find_nodes_by_property("value", &PropertyValue::Float(f64::NAN))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn float_infinity_handling() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("value".to_string(), PropertyValue::Float(f64::INFINITY));
        engine.create_node("Node", props).unwrap();

        engine.create_node_property_index("value").unwrap();

        let nodes = engine
            .find_nodes_by_property("value", &PropertyValue::Float(f64::INFINITY))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn empty_property_value() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String(String::new()));
        engine.create_node("Node", props).unwrap();

        engine.create_node_property_index("name").unwrap();

        let nodes = engine
            .find_nodes_by_property("name", &PropertyValue::String(String::new()))
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn unicode_string_handling() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Hello Unicode Test".to_string()),
        );
        engine.create_node("Node", props).unwrap();

        engine.create_node_property_index("name").unwrap();

        let nodes = engine
            .find_nodes_by_property(
                "name",
                &PropertyValue::String("Hello Unicode Test".to_string()),
            )
            .unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn get_indexed_properties() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();
        engine.create_node_property_index("age").unwrap();
        engine.create_edge_type_index().unwrap();

        let node_props = engine.get_indexed_node_properties();
        assert_eq!(node_props.len(), 2);
        assert!(node_props.contains(&"_label".to_string()));
        assert!(node_props.contains(&"age".to_string()));

        let edge_props = engine.get_indexed_edge_properties();
        assert_eq!(edge_props.len(), 1);
        assert!(edge_props.contains(&"_edge_type".to_string()));
    }

    #[test]
    fn ordered_float_ordering() {
        // Test that NaN sorts first
        let nan = OrderedFloat(f64::NAN);
        let one = OrderedFloat(1.0);
        let two = OrderedFloat(2.0);

        assert!(nan < one);
        assert!(one < two);
        assert!(nan < two);
    }

    #[test]
    fn index_error_display() {
        let e1 = GraphError::IndexAlreadyExists {
            target: "Node".to_string(),
            property: "age".to_string(),
        };
        assert!(format!("{}", e1).contains("Node"));
        assert!(format!("{}", e1).contains("age"));

        let e2 = GraphError::IndexNotFound {
            target: "Edge".to_string(),
            property: "weight".to_string(),
        };
        assert!(format!("{}", e2).contains("Edge"));
        assert!(format!("{}", e2).contains("weight"));
    }

    #[test]
    fn concurrent_index_reads() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(GraphEngine::new());

        // Create nodes and index
        for i in 0..100 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Node", props).unwrap();
        }
        engine.create_node_property_index("value").unwrap();

        // Spawn multiple readers
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    let nodes = eng
                        .find_nodes_by_property("value", &PropertyValue::Int(i * 10))
                        .unwrap();
                    assert_eq!(nodes.len(), 1);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_index_writes() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(GraphEngine::new());
        engine.create_label_index().unwrap();

        // Spawn multiple writers
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    for _ in 0..10 {
                        eng.create_node("Person", HashMap::new()).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should have 100 persons
        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 100);
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn ordered_float_equality() {
        // Test PartialEq implementation
        let a = OrderedFloat(1.5);
        let b = OrderedFloat(1.5);
        let c = OrderedFloat(2.5);
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Test NaN equality (NaN == NaN for OrderedFloat)
        let nan1 = OrderedFloat(f64::NAN);
        let nan2 = OrderedFloat(f64::NAN);
        assert_eq!(nan1, nan2);
    }

    #[test]
    fn ordered_float_greater_than_nan() {
        // Test that regular float is Greater than NaN
        let nan = OrderedFloat(f64::NAN);
        let regular = OrderedFloat(1.0);
        assert!(regular > nan); // This tests the (false, true) => Greater case
    }

    #[test]
    fn ordered_float_hash() {
        use std::collections::HashMap;
        // Test Hash implementation by using OrderedFloat as HashMap key
        let mut map: HashMap<OrderedFloat, i32> = HashMap::new();
        map.insert(OrderedFloat(1.5), 1);
        map.insert(OrderedFloat(2.5), 2);
        map.insert(OrderedFloat(f64::NAN), 3);

        assert_eq!(map.get(&OrderedFloat(1.5)), Some(&1));
        assert_eq!(map.get(&OrderedFloat(2.5)), Some(&2));
        assert_eq!(map.get(&OrderedFloat(f64::NAN)), Some(&3));
    }

    #[test]
    fn ordered_property_value_null() {
        // Test PropertyValue::Null conversion
        let null = PropertyValue::Null;
        let ordered = OrderedPropertyValue::from(&null);
        assert_eq!(ordered, OrderedPropertyValue::Null);
    }

    #[test]
    fn with_store_rebuilds_edge_indexes() {
        // Test that edge indexes are rebuilt from store
        let store = TensorStore::new();

        // Create engine, add data, create edge index
        let engine = GraphEngine::with_store(store.clone());
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Int(10));
        engine
            .create_edge(n1, n2, "CONNECTS", props.clone(), true)
            .unwrap();

        props.insert("weight".to_string(), PropertyValue::Int(20));
        engine.create_edge(n2, n1, "CONNECTS", props, true).unwrap();

        // Create edge property index
        engine.create_edge_property_index("weight").unwrap();

        // Verify index works
        let edges = engine
            .find_edges_by_property("weight", &PropertyValue::Int(10))
            .unwrap();
        assert_eq!(edges.len(), 1);

        // Create new engine from same store - should rebuild indexes
        let engine2 = GraphEngine::with_store(store);

        // Verify index was rebuilt
        let edges2 = engine2
            .find_edges_by_property("weight", &PropertyValue::Int(10))
            .unwrap();
        assert_eq!(edges2.len(), 1);

        let edges3 = engine2
            .find_edges_by_property("weight", &PropertyValue::Int(20))
            .unwrap();
        assert_eq!(edges3.len(), 1);
    }

    #[test]
    fn with_store_rebuilds_edge_type_index() {
        // Test edge type index rebuilding
        let store = TensorStore::new();

        let engine = GraphEngine::with_store(store.clone());
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "LIKES", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n1, "KNOWS", HashMap::new(), true)
            .unwrap();

        engine.create_edge_type_index().unwrap();

        // Verify
        let likes = engine.find_edges_by_type("LIKES").unwrap();
        assert_eq!(likes.len(), 1);

        // Rebuild from store
        let engine2 = GraphEngine::with_store(store);
        let likes2 = engine2.find_edges_by_type("LIKES").unwrap();
        assert_eq!(likes2.len(), 1);
    }

    #[test]
    fn create_node_after_property_index_exists() {
        // Test that creating nodes after index exists updates the index
        let engine = GraphEngine::new();

        // Create property index FIRST
        engine.create_node_property_index("age").unwrap();

        // Now create nodes with that property
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(25));
        engine.create_node("Person", props).unwrap();

        props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        engine.create_node("Person", props).unwrap();

        // Find using index
        let found = engine
            .find_nodes_by_property("age", &PropertyValue::Int(25))
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].properties.get("age"),
            Some(&PropertyValue::Int(25))
        );
    }

    #[test]
    fn delete_node_removes_from_property_index() {
        // Test that deleting node removes from property index
        let engine = GraphEngine::new();

        // Create property index first
        engine.create_node_property_index("name").unwrap();

        // Create nodes
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        let id1 = engine.create_node("Person", props).unwrap();

        props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Bob".to_string()));
        engine.create_node("Person", props).unwrap();

        // Verify both are indexed
        let alice = engine
            .find_nodes_by_property("name", &PropertyValue::String("Alice".to_string()))
            .unwrap();
        assert_eq!(alice.len(), 1);

        // Delete Alice
        engine.delete_node(id1).unwrap();

        // Verify Alice is no longer found
        let alice_after = engine
            .find_nodes_by_property("name", &PropertyValue::String("Alice".to_string()))
            .unwrap();
        assert!(alice_after.is_empty());

        // Bob should still be there
        let bob = engine
            .find_nodes_by_property("name", &PropertyValue::String("Bob".to_string()))
            .unwrap();
        assert_eq!(bob.len(), 1);
    }

    #[test]
    fn create_edge_after_property_index_exists() {
        // Test creating edges after edge property index exists
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create edge property index FIRST
        engine.create_edge_property_index("weight").unwrap();

        // Now create edges with that property
        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Int(100));
        engine.create_edge(n1, n2, "CONNECTS", props, true).unwrap();

        // Find using index
        let found = engine
            .find_edges_by_property("weight", &PropertyValue::Int(100))
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].properties.get("weight"),
            Some(&PropertyValue::Int(100))
        );
    }

    #[test]
    fn delete_edge_removes_from_property_index() {
        // Test that deleting edge removes from property index
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create edge property index first
        engine.create_edge_property_index("priority").unwrap();

        // Create edges
        let mut props = HashMap::new();
        props.insert("priority".to_string(), PropertyValue::Int(1));
        let e1 = engine.create_edge(n1, n2, "LINK", props, true).unwrap();

        props = HashMap::new();
        props.insert("priority".to_string(), PropertyValue::Int(2));
        engine.create_edge(n2, n1, "LINK", props, true).unwrap();

        // Verify both indexed
        let p1 = engine
            .find_edges_by_property("priority", &PropertyValue::Int(1))
            .unwrap();
        assert_eq!(p1.len(), 1);

        // Delete first edge
        engine.delete_edge(e1).unwrap();

        // Verify it's gone from index
        let p1_after = engine
            .find_edges_by_property("priority", &PropertyValue::Int(1))
            .unwrap();
        assert!(p1_after.is_empty());

        // Second edge should still be there
        let p2 = engine
            .find_edges_by_property("priority", &PropertyValue::Int(2))
            .unwrap();
        assert_eq!(p2.len(), 1);
    }

    #[test]
    fn find_edges_where_with_index() {
        // Test range queries on edge properties with index
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        // Create edge property index first
        engine.create_edge_property_index("cost").unwrap();

        // Create edges with different costs
        let mut props = HashMap::new();
        props.insert("cost".to_string(), PropertyValue::Int(10));
        engine.create_edge(n1, n2, "PATH", props, true).unwrap();

        props = HashMap::new();
        props.insert("cost".to_string(), PropertyValue::Int(20));
        engine.create_edge(n2, n3, "PATH", props, true).unwrap();

        props = HashMap::new();
        props.insert("cost".to_string(), PropertyValue::Int(30));
        engine.create_edge(n1, n3, "PATH", props, true).unwrap();

        // Test Lt
        let lt20 = engine
            .find_edges_where("cost", RangeOp::Lt, &PropertyValue::Int(20))
            .unwrap();
        assert_eq!(lt20.len(), 1);
        assert_eq!(
            lt20[0].properties.get("cost"),
            Some(&PropertyValue::Int(10))
        );

        // Test Le
        let le20 = engine
            .find_edges_where("cost", RangeOp::Le, &PropertyValue::Int(20))
            .unwrap();
        assert_eq!(le20.len(), 2);

        // Test Gt
        let gt20 = engine
            .find_edges_where("cost", RangeOp::Gt, &PropertyValue::Int(20))
            .unwrap();
        assert_eq!(gt20.len(), 1);
        assert_eq!(
            gt20[0].properties.get("cost"),
            Some(&PropertyValue::Int(30))
        );

        // Test Ge
        let ge20 = engine
            .find_edges_where("cost", RangeOp::Ge, &PropertyValue::Int(20))
            .unwrap();
        assert_eq!(ge20.len(), 2);
    }

    #[test]
    fn find_edges_where_scan_fallback() {
        // Test range queries on edge properties without index (scan fallback)
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create edges WITHOUT index
        let mut props = HashMap::new();
        props.insert("score".to_string(), PropertyValue::Int(5));
        engine.create_edge(n1, n2, "RATED", props, true).unwrap();

        props = HashMap::new();
        props.insert("score".to_string(), PropertyValue::Int(8));
        engine.create_edge(n2, n1, "RATED", props, true).unwrap();

        // Query without index should still work via scan
        let high = engine
            .find_edges_where("score", RangeOp::Gt, &PropertyValue::Int(6))
            .unwrap();
        assert_eq!(high.len(), 1);
        assert_eq!(
            high[0].properties.get("score"),
            Some(&PropertyValue::Int(8))
        );
    }

    #[test]
    fn find_edges_by_property_with_index() {
        // Test exact match on edge property with index
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create index first
        engine.create_edge_property_index("status").unwrap();

        // Create edges
        let mut props = HashMap::new();
        props.insert(
            "status".to_string(),
            PropertyValue::String("active".to_string()),
        );
        engine.create_edge(n1, n2, "CONN", props, true).unwrap();

        props = HashMap::new();
        props.insert(
            "status".to_string(),
            PropertyValue::String("inactive".to_string()),
        );
        engine.create_edge(n2, n1, "CONN", props, true).unwrap();

        // Find with index
        let active = engine
            .find_edges_by_property("status", &PropertyValue::String("active".to_string()))
            .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(
            active[0].properties.get("status"),
            Some(&PropertyValue::String("active".to_string()))
        );
    }

    #[test]
    fn find_edges_by_property_scan_fallback() {
        // Test exact match without index (scan)
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create edges without index
        let mut props = HashMap::new();
        props.insert(
            "color".to_string(),
            PropertyValue::String("red".to_string()),
        );
        engine.create_edge(n1, n2, "HAS", props, true).unwrap();

        // Find without index
        let red = engine
            .find_edges_by_property("color", &PropertyValue::String("red".to_string()))
            .unwrap();
        assert_eq!(red.len(), 1);
    }

    #[test]
    fn with_store_rebuilds_node_property_index() {
        // Test that node property indexes are rebuilt from store
        let store = TensorStore::new();

        let engine = GraphEngine::with_store(store.clone());

        // Create property index and add nodes
        engine.create_node_property_index("score").unwrap();

        let mut props = HashMap::new();
        props.insert("score".to_string(), PropertyValue::Int(100));
        engine.create_node("Player", props).unwrap();

        props = HashMap::new();
        props.insert("score".to_string(), PropertyValue::Int(200));
        engine.create_node("Player", props).unwrap();

        // Rebuild from store
        let engine2 = GraphEngine::with_store(store);

        // Verify index works
        let high_score = engine2
            .find_nodes_by_property("score", &PropertyValue::Int(200))
            .unwrap();
        assert_eq!(high_score.len(), 1);
    }

    #[test]
    fn with_store_no_indexes() {
        // Test with_store when there are no indexes
        let store = TensorStore::new();

        let engine = GraphEngine::with_store(store.clone());
        engine.create_node("Test", HashMap::new()).unwrap();

        // Rebuild - should work fine with no indexes
        let engine2 = GraphEngine::with_store(store);
        assert_eq!(engine2.node_count(), 1);
    }

    #[test]
    fn update_node_with_indexed_property() {
        // Test that updating a node updates the property index
        let engine = GraphEngine::new();

        // Create property index first
        engine.create_node_property_index("level").unwrap();

        // Create node
        let mut props = HashMap::new();
        props.insert("level".to_string(), PropertyValue::Int(1));
        let id = engine.create_node("Player", props).unwrap();

        // Verify initial index
        let level1 = engine
            .find_nodes_by_property("level", &PropertyValue::Int(1))
            .unwrap();
        assert_eq!(level1.len(), 1);

        // Update node
        let mut new_props = HashMap::new();
        new_props.insert("level".to_string(), PropertyValue::Int(2));
        engine.update_node(id, None, new_props).unwrap();

        // Old value should not be found
        let level1_after = engine
            .find_nodes_by_property("level", &PropertyValue::Int(1))
            .unwrap();
        assert!(level1_after.is_empty());

        // New value should be found
        let level2 = engine
            .find_nodes_by_property("level", &PropertyValue::Int(2))
            .unwrap();
        assert_eq!(level2.len(), 1);
    }

    #[test]
    fn update_edge_with_indexed_property() {
        // Test that updating an edge updates the property index
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create property index first
        engine.create_edge_property_index("version").unwrap();

        // Create edge
        let mut props = HashMap::new();
        props.insert("version".to_string(), PropertyValue::Int(1));
        let eid = engine.create_edge(n1, n2, "LINK", props, true).unwrap();

        // Verify initial index
        let v1 = engine
            .find_edges_by_property("version", &PropertyValue::Int(1))
            .unwrap();
        assert_eq!(v1.len(), 1);

        // Update edge
        let mut new_props = HashMap::new();
        new_props.insert("version".to_string(), PropertyValue::Int(2));
        engine.update_edge(eid, new_props).unwrap();

        // Old value should not be found
        let v1_after = engine
            .find_edges_by_property("version", &PropertyValue::Int(1))
            .unwrap();
        assert!(v1_after.is_empty());

        // New value should be found
        let v2 = engine
            .find_edges_by_property("version", &PropertyValue::Int(2))
            .unwrap();
        assert_eq!(v2.len(), 1);
    }

    #[test]
    fn find_nodes_by_null_property() {
        // Test finding nodes with Null property value
        let engine = GraphEngine::new();

        engine.create_node_property_index("optional").unwrap();

        let mut props = HashMap::new();
        props.insert("optional".to_string(), PropertyValue::Null);
        engine.create_node("Item", props).unwrap();

        let found = engine
            .find_nodes_by_property("optional", &PropertyValue::Null)
            .unwrap();
        assert_eq!(found.len(), 1);
    }
}
