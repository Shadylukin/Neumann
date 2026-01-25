// Pedantic lint configuration for graph_engine
#![allow(clippy::cast_possible_wrap)] // u64 IDs won't exceed i64::MAX
#![allow(clippy::cast_sign_loss)] // i64 values from store are always non-negative IDs
#![allow(clippy::needless_pass_by_value)] // HashMap ownership is intentional for API design
#![allow(clippy::missing_errors_doc)] // Error conditions are self-evident from Result types
#![allow(clippy::uninlined_format_args)] // Keep format strings readable

use std::{
    cmp::Ordering as CmpOrdering,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[allow(clippy::cast_possible_truncation)] // Millis since epoch won't overflow u64 until year 584 million
fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

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

/// Comparison operator for property conditions during traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,
    Ne,
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

/// A single property condition for filtering nodes or edges during traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyCondition {
    pub property: String,
    pub op: CompareOp,
    pub value: PropertyValue,
}

impl PropertyCondition {
    #[must_use]
    pub fn new(property: impl Into<String>, op: CompareOp, value: PropertyValue) -> Self {
        Self {
            property: property.into(),
            op,
            value,
        }
    }

    #[must_use]
    pub fn matches_node(&self, node: &Node) -> bool {
        self.evaluate_properties(&node.properties)
    }

    #[must_use]
    pub fn matches_edge(&self, edge: &Edge) -> bool {
        self.evaluate_properties(&edge.properties)
    }

    fn evaluate_properties(&self, props: &HashMap<String, PropertyValue>) -> bool {
        match props.get(&self.property) {
            None => {
                // Missing property: only Ne matches (value is "not equal" to missing)
                self.op == CompareOp::Ne
            },
            Some(prop_val) => self.compare_values(prop_val),
        }
    }

    fn compare_values(&self, actual: &PropertyValue) -> bool {
        match (&self.value, actual) {
            // Null comparisons: null == null is true for Eq/Le/Ge
            (PropertyValue::Null, PropertyValue::Null) => {
                matches!(self.op, CompareOp::Eq | CompareOp::Le | CompareOp::Ge)
            },

            // Int comparisons
            (PropertyValue::Int(expected), PropertyValue::Int(actual)) => {
                self.compare_ordered(actual, expected)
            },

            // Float comparisons
            (PropertyValue::Float(expected), PropertyValue::Float(actual)) => {
                self.compare_floats(*actual, *expected)
            },

            // Cross-numeric: int vs float
            (PropertyValue::Int(expected), PropertyValue::Float(actual)) =>
            {
                #[allow(clippy::cast_precision_loss)]
                self.compare_floats(*actual, *expected as f64)
            },
            (PropertyValue::Float(expected), PropertyValue::Int(actual)) =>
            {
                #[allow(clippy::cast_precision_loss)]
                self.compare_floats(*actual as f64, *expected)
            },

            // String comparisons
            (PropertyValue::String(expected), PropertyValue::String(actual)) => {
                self.compare_ordered(actual, expected)
            },

            // Bool comparisons (only Eq/Ne make sense)
            (PropertyValue::Bool(expected), PropertyValue::Bool(actual)) => match self.op {
                CompareOp::Eq => actual == expected,
                CompareOp::Ne => actual != expected,
                _ => false, // Lt/Le/Gt/Ge don't make sense for bools
            },

            // Type mismatch: only Ne returns true
            _ => self.op == CompareOp::Ne,
        }
    }

    fn compare_ordered<T: Ord>(&self, actual: &T, expected: &T) -> bool {
        match self.op {
            CompareOp::Eq => actual == expected,
            CompareOp::Ne => actual != expected,
            CompareOp::Lt => actual < expected,
            CompareOp::Le => actual <= expected,
            CompareOp::Gt => actual > expected,
            CompareOp::Ge => actual >= expected,
        }
    }

    fn compare_floats(&self, actual: f64, expected: f64) -> bool {
        // Handle NaN: NaN is not equal to anything including itself
        if actual.is_nan() || expected.is_nan() {
            return self.op == CompareOp::Ne;
        }
        match self.op {
            CompareOp::Eq => (actual - expected).abs() < f64::EPSILON,
            CompareOp::Ne => (actual - expected).abs() >= f64::EPSILON,
            CompareOp::Lt => actual < expected,
            CompareOp::Le => actual <= expected,
            CompareOp::Gt => actual > expected,
            CompareOp::Ge => actual >= expected,
        }
    }
}

/// Filter configuration for graph traversal operations.
#[derive(Debug, Clone, Default)]
pub struct TraversalFilter {
    node_conditions: Vec<PropertyCondition>,
    edge_conditions: Vec<PropertyCondition>,
}

impl TraversalFilter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn node_eq(self, property: &str, value: PropertyValue) -> Self {
        self.node_where(property, CompareOp::Eq, value)
    }

    #[must_use]
    pub fn node_ne(self, property: &str, value: PropertyValue) -> Self {
        self.node_where(property, CompareOp::Ne, value)
    }

    #[must_use]
    pub fn node_where(mut self, property: &str, op: CompareOp, value: PropertyValue) -> Self {
        self.node_conditions
            .push(PropertyCondition::new(property, op, value));
        self
    }

    #[must_use]
    pub fn edge_eq(self, property: &str, value: PropertyValue) -> Self {
        self.edge_where(property, CompareOp::Eq, value)
    }

    #[must_use]
    pub fn edge_ne(self, property: &str, value: PropertyValue) -> Self {
        self.edge_where(property, CompareOp::Ne, value)
    }

    #[must_use]
    pub fn edge_where(mut self, property: &str, op: CompareOp, value: PropertyValue) -> Self {
        self.edge_conditions
            .push(PropertyCondition::new(property, op, value));
        self
    }

    #[must_use]
    pub fn matches_node(&self, node: &Node) -> bool {
        self.node_conditions.iter().all(|c| c.matches_node(node))
    }

    #[must_use]
    pub fn matches_edge(&self, edge: &Edge) -> bool {
        self.edge_conditions.iter().all(|c| c.matches_edge(edge))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_conditions.is_empty() && self.edge_conditions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub labels: Vec<String>,
    pub properties: HashMap<String, PropertyValue>,
    #[serde(default)]
    pub created_at: Option<u64>,
    #[serde(default)]
    pub updated_at: Option<u64>,
}

impl Node {
    #[must_use]
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l == label)
    }

    #[must_use]
    pub const fn created_at_millis(&self) -> Option<u64> {
        self.created_at
    }

    #[must_use]
    pub const fn updated_at_millis(&self) -> Option<u64> {
        self.updated_at
    }

    #[must_use]
    pub const fn last_modified_millis(&self) -> Option<u64> {
        match self.updated_at {
            Some(u) => Some(u),
            None => self.created_at,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub properties: HashMap<String, PropertyValue>,
    pub directed: bool,
    #[serde(default)]
    pub created_at: Option<u64>,
    #[serde(default)]
    pub updated_at: Option<u64>,
}

impl Edge {
    #[must_use]
    pub const fn created_at_millis(&self) -> Option<u64> {
        self.created_at
    }

    #[must_use]
    pub const fn updated_at_millis(&self) -> Option<u64> {
        self.updated_at
    }

    #[must_use]
    pub const fn last_modified_millis(&self) -> Option<u64> {
        match self.updated_at {
            Some(u) => Some(u),
            None => self.created_at,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Path {
    pub nodes: Vec<u64>,
    pub edges: Vec<u64>,
}

/// A path with accumulated edge weights from Dijkstra's algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedPath {
    pub nodes: Vec<u64>,
    pub edges: Vec<u64>,
    pub total_weight: f64,
}

/// Configuration for all-paths search to prevent memory explosion.
#[derive(Debug, Clone, Copy)]
pub struct AllPathsConfig {
    /// Maximum number of paths to return (default: 1000).
    pub max_paths: usize,
    /// Maximum parent entries per node (default: 100).
    pub max_parents_per_node: usize,
}

impl Default for AllPathsConfig {
    fn default() -> Self {
        Self {
            max_paths: 1000,
            max_parents_per_node: 100,
        }
    }
}

/// Collection of all shortest unweighted paths between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllPaths {
    /// All shortest paths (same hop count).
    pub paths: Vec<Path>,
    /// Number of hops in each path.
    pub hop_count: usize,
}

/// Collection of all shortest weighted paths between two nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllWeightedPaths {
    /// All shortest paths (same total weight).
    pub paths: Vec<WeightedPath>,
    /// The minimum total weight.
    pub total_weight: f64,
}

/// Configuration for variable-length path traversal.
#[derive(Debug, Clone)]
pub struct VariableLengthConfig {
    /// Minimum number of hops (inclusive). Default: 1.
    pub min_hops: usize,
    /// Maximum number of hops (inclusive). Default: 5.
    pub max_hops: usize,
    /// Direction of traversal. Default: Outgoing.
    pub direction: Direction,
    /// Edge type filter. None means all types.
    pub edge_types: Option<Vec<String>>,
    /// Maximum paths to return. Default: 1000.
    pub max_paths: usize,
    /// Whether to allow cycles in paths. Default: false.
    pub allow_cycles: bool,
    /// Property filter for nodes and edges. Default: None (no filtering).
    pub filter: Option<TraversalFilter>,
}

impl Default for VariableLengthConfig {
    fn default() -> Self {
        Self {
            min_hops: 1,
            max_hops: 5,
            direction: Direction::Outgoing,
            edge_types: None,
            max_paths: 1000,
            allow_cycles: false,
            filter: None,
        }
    }
}

impl VariableLengthConfig {
    #[must_use]
    pub fn with_hops(min: usize, max: usize) -> Self {
        Self {
            min_hops: min,
            max_hops: max.min(20), // Safety cap
            ..Self::default()
        }
    }

    #[must_use]
    pub fn direction(mut self, dir: Direction) -> Self {
        self.direction = dir;
        self
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: &str) -> Self {
        self.edge_types = Some(vec![edge_type.to_string()]);
        self
    }

    #[must_use]
    pub fn edge_types(mut self, types: &[&str]) -> Self {
        self.edge_types = Some(types.iter().map(|s| (*s).to_string()).collect());
        self
    }

    #[must_use]
    pub const fn max_paths(mut self, max: usize) -> Self {
        self.max_paths = max;
        self
    }

    #[must_use]
    pub const fn allow_cycles(mut self, allow: bool) -> Self {
        self.allow_cycles = allow;
        self
    }

    #[must_use]
    pub fn with_filter(mut self, filter: TraversalFilter) -> Self {
        self.filter = Some(filter);
        self
    }
}

/// Statistics from path search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PathSearchStats {
    pub paths_found: usize,
    pub nodes_explored: usize,
    pub edges_traversed: usize,
    pub truncated: bool,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
}

/// Result of variable-length path search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableLengthPaths {
    pub paths: Vec<Path>,
    pub stats: PathSearchStats,
}

impl VariableLengthPaths {
    #[must_use]
    pub fn paths_of_length(&self, len: usize) -> Vec<&Path> {
        self.paths.iter().filter(|p| p.edges.len() == len).collect()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

/// Pagination configuration for graph queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pagination {
    /// Number of items to skip.
    pub skip: usize,
    /// Maximum items to return.
    pub limit: Option<usize>,
    /// Whether to compute total count (expensive).
    pub count_total: bool,
}

impl Pagination {
    #[must_use]
    pub const fn new(skip: usize, limit: usize) -> Self {
        Self {
            skip,
            limit: Some(limit),
            count_total: false,
        }
    }

    #[must_use]
    pub const fn limit(limit: usize) -> Self {
        Self {
            skip: 0,
            limit: Some(limit),
            count_total: false,
        }
    }

    #[must_use]
    pub const fn with_total_count(mut self) -> Self {
        self.count_total = true;
        self
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.skip == 0 && self.limit.is_none()
    }
}

/// Result of a paginated graph query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PagedResult<T> {
    pub items: Vec<T>,
    pub total_count: Option<usize>,
    pub has_more: bool,
}

impl<T> PagedResult<T> {
    #[must_use]
    pub fn new(items: Vec<T>, total_count: Option<usize>, has_more: bool) -> Self {
        Self {
            items,
            total_count,
            has_more,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> Default for PagedResult<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            total_count: None,
            has_more: false,
        }
    }
}

/// Result of an aggregation operation on graph data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateResult {
    /// Number of items that matched the criteria.
    pub count: u64,
    /// Sum of numeric property values (None if no numeric values).
    pub sum: Option<f64>,
    /// Average of numeric property values (None if no numeric values).
    pub avg: Option<f64>,
    /// Minimum property value found (None if no values).
    pub min: Option<PropertyValue>,
    /// Maximum property value found (None if no values).
    pub max: Option<PropertyValue>,
}

impl AggregateResult {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            count: 0,
            sum: None,
            avg: None,
            min: None,
            max: None,
        }
    }

    #[must_use]
    pub fn count_only(count: u64) -> Self {
        Self {
            count,
            sum: None,
            avg: None,
            min: None,
            max: None,
        }
    }
}

impl Default for AggregateResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Min-heap entry for Dijkstra (smallest distance first).
#[derive(Copy, Clone)]
struct DijkstraEntry {
    cost: f64,
    node_id: u64,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for DijkstraEntry {}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Flip ordering for min-heap (smallest cost first)
        // Use total_cmp for NaN-safe f64 comparison
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphError {
    NodeNotFound(u64),
    EdgeNotFound(u64),
    StorageError(String),
    PathNotFound,
    IndexAlreadyExists { target: String, property: String },
    IndexNotFound { target: String, property: String },
    NegativeWeight { edge_id: u64, weight: f64 },
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
            Self::NegativeWeight { edge_id, weight } => {
                write!(f, "Edge {edge_id} has negative weight: {weight}")
            },
        }
    }
}

impl std::error::Error for GraphError {}

impl Eq for GraphError {}

impl Hash for GraphError {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::NodeNotFound(id) | Self::EdgeNotFound(id) => id.hash(state),
            Self::StorageError(s) => s.hash(state),
            Self::PathNotFound => {},
            Self::IndexAlreadyExists { target, property }
            | Self::IndexNotFound { target, property } => {
                target.hash(state);
                property.hash(state);
            },
            Self::NegativeWeight { edge_id, weight } => {
                edge_id.hash(state);
                weight.to_bits().hash(state);
            },
        }
    }
}

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
    /// Whether the label index has been initialized (for lazy auto-creation).
    label_index_initialized: AtomicBool,
    /// Whether the edge type index has been initialized (for lazy auto-creation).
    edge_type_index_initialized: AtomicBool,
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
    const AGGREGATE_PARALLEL_THRESHOLD: usize = 1000;

    #[must_use]
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            node_counter: AtomicU64::new(0),
            edge_counter: AtomicU64::new(0),
            btree_indexes: RwLock::new(HashMap::new()),
            index_locks: create_index_locks(),
            label_index_initialized: AtomicBool::new(false),
            edge_type_index_initialized: AtomicBool::new(false),
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

        // Check if label index was rebuilt
        let label_index_exists =
            btree_indexes.contains_key(&(IndexTarget::Node, "_label".to_string()));
        // Check if edge type index was rebuilt
        let edge_type_index_exists =
            btree_indexes.contains_key(&(IndexTarget::Edge, "_edge_type".to_string()));

        Self {
            store,
            node_counter: AtomicU64::new(max_node_id),
            edge_counter: AtomicU64::new(max_edge_id),
            btree_indexes: RwLock::new(btree_indexes),
            index_locks: create_index_locks(),
            label_index_initialized: AtomicBool::new(label_index_exists),
            edge_type_index_initialized: AtomicBool::new(edge_type_index_exists),
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
                                    // Handle special properties (_label) - support both formats
                                    if property == "_label" {
                                        // Try new format first, then fall back to legacy
                                        let labels = match tensor.get("_labels") {
                                            Some(TensorValue::Pointers(labels)) => labels.clone(),
                                            _ => match tensor.get("_label") {
                                                Some(TensorValue::Scalar(ScalarValue::String(
                                                    s,
                                                ))) => {
                                                    vec![s.clone()]
                                                },
                                                _ => Vec::new(),
                                            },
                                        };
                                        for label in labels {
                                            btree
                                                .entry(OrderedPropertyValue::String(label))
                                                .or_default()
                                                .push(id);
                                        }
                                    } else if let Some(TensorValue::Scalar(scalar)) =
                                        tensor.get(&property)
                                    {
                                        let ordered_val = OrderedPropertyValue::from(
                                            &PropertyValue::from_scalar(scalar),
                                        );
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
    const fn lock_index(id: u64) -> usize {
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

    /// Ensure the label index exists, creating it if necessary.
    ///
    /// Uses double-checked locking for thread-safe lazy initialization.
    fn ensure_label_index(&self) {
        // Fast path: already initialized
        if self.label_index_initialized.load(Ordering::Acquire) {
            return;
        }

        // Slow path: check and create
        let key = (IndexTarget::Node, "_label".to_string());
        {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&key) {
                self.label_index_initialized.store(true, Ordering::Release);
                return;
            }
        }

        // Create the index (ignore AlreadyExists error from race)
        let _ = self.create_label_index();
        self.label_index_initialized.store(true, Ordering::Release);
    }

    /// Ensure the edge type index exists, creating it if necessary.
    ///
    /// Uses double-checked locking for thread-safe lazy initialization.
    fn ensure_edge_type_index(&self) {
        // Fast path: already initialized
        if self.edge_type_index_initialized.load(Ordering::Acquire) {
            return;
        }

        // Slow path: check and create
        let key = (IndexTarget::Edge, "_edge_type".to_string());
        {
            let indexes = self.btree_indexes.read();
            if indexes.contains_key(&key) {
                self.edge_type_index_initialized
                    .store(true, Ordering::Release);
                return;
            }
        }

        // Create the index (ignore AlreadyExists error from race)
        let _ = self.create_edge_type_index();
        self.edge_type_index_initialized
            .store(true, Ordering::Release);
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
                                // Handle special properties (_label) - support both formats
                                if property == "_label" {
                                    let labels = match tensor.get("_labels") {
                                        Some(TensorValue::Pointers(labels)) => labels.clone(),
                                        _ => match tensor.get("_label") {
                                            Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                                                vec![s.clone()]
                                            },
                                            _ => Vec::new(),
                                        },
                                    };
                                    for label in labels {
                                        btree
                                            .entry(OrderedPropertyValue::String(label))
                                            .or_default()
                                            .push(id);
                                    }
                                } else if let Some(TensorValue::Scalar(scalar)) =
                                    tensor.get(property)
                                {
                                    let ordered_val = OrderedPropertyValue::from(
                                        &PropertyValue::from_scalar(scalar),
                                    );
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
        labels: &[String],
        properties: &HashMap<String, PropertyValue>,
    ) {
        let indexes = self.btree_indexes.read();

        // Index each label if _label index exists
        if indexes.contains_key(&(IndexTarget::Node, "_label".to_string())) {
            drop(indexes);
            for label in labels {
                self.index_add(
                    IndexTarget::Node,
                    "_label",
                    &OrderedPropertyValue::String(label.clone()),
                    id,
                );
            }
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
        labels: &[String],
        properties: &HashMap<String, PropertyValue>,
    ) {
        let indexes = self.btree_indexes.read();

        // Unindex each label if _label index exists
        if indexes.contains_key(&(IndexTarget::Node, "_label".to_string())) {
            drop(indexes);
            for label in labels {
                self.index_remove(
                    IndexTarget::Node,
                    "_label",
                    &OrderedPropertyValue::String(label.clone()),
                    id,
                );
            }
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
    /// For nodes with multiple labels, returns any node that has the specified label.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_label(&self, label: &str) -> Result<Vec<Node>> {
        self.find_nodes_by_property("_label", &PropertyValue::String(label.to_string()))
    }

    /// Find nodes that have ALL specified labels (intersection).
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_all_labels(&self, labels: &[&str]) -> Result<Vec<Node>> {
        if labels.is_empty() {
            return Ok(Vec::new());
        }

        // Get nodes with the first label
        let mut result_ids: HashSet<u64> = self
            .find_nodes_by_label(labels[0])?
            .into_iter()
            .map(|n| n.id)
            .collect();

        // Intersect with nodes having each additional label
        for label in &labels[1..] {
            let label_ids: HashSet<u64> = self
                .find_nodes_by_label(label)?
                .into_iter()
                .map(|n| n.id)
                .collect();
            result_ids.retain(|id| label_ids.contains(id));
            if result_ids.is_empty() {
                return Ok(Vec::new());
            }
        }

        // Collect and sort results
        let mut nodes: Vec<Node> = result_ids
            .into_iter()
            .filter_map(|id| self.get_node(id).ok())
            .collect();
        nodes.sort_by_key(|n| n.id);
        Ok(nodes)
    }

    /// Find nodes that have ANY of the specified labels (union).
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_any_label(&self, labels: &[&str]) -> Result<Vec<Node>> {
        if labels.is_empty() {
            return Ok(Vec::new());
        }

        let mut result_ids: HashSet<u64> = HashSet::new();

        for label in labels {
            let label_ids: Vec<u64> = self
                .find_nodes_by_label(label)?
                .into_iter()
                .map(|n| n.id)
                .collect();
            result_ids.extend(label_ids);
        }

        // Collect and sort results
        let mut nodes: Vec<Node> = result_ids
            .into_iter()
            .filter_map(|id| self.get_node(id).ok())
            .collect();
        nodes.sort_by_key(|n| n.id);
        Ok(nodes)
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
                                PropertyValue::String(s) => node.has_label(s),
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
                        // For labels, check if any label matches the range condition
                        if property == "_label" {
                            for label in &node.labels {
                                let ordered_pv = OrderedPropertyValue::String(label.clone());
                                let matches = match op {
                                    RangeOp::Lt => ordered_pv < ordered_value,
                                    RangeOp::Le => ordered_pv <= ordered_value,
                                    RangeOp::Gt => ordered_pv > ordered_value,
                                    RangeOp::Ge => ordered_pv >= ordered_value,
                                };
                                if matches {
                                    nodes.push(node);
                                    break;
                                }
                            }
                        } else if let Some(pv) = node.properties.get(property).cloned() {
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
        self.create_node_with_labels(vec![label.into()], properties)
    }

    #[instrument(skip(self, labels, properties))]
    pub fn create_node_with_labels(
        &self,
        labels: Vec<String>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<u64> {
        // Ensure label index exists (lazy init on first node creation)
        self.ensure_label_index();

        let id = self.node_counter.fetch_add(1, Ordering::SeqCst) + 1;

        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(id as i64)));
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("node".into())),
        );
        tensor.set("_labels", TensorValue::Pointers(labels.clone()));
        tensor.set(
            "_created_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis() as i64)),
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
        self.index_node_properties(id, &labels, &properties);

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
        // Ensure edge type index exists (lazy init on first edge creation)
        self.ensure_edge_type_index();

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
        tensor.set(
            "_created_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis() as i64)),
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

        // Read labels - support both new format (_labels) and legacy format (_label)
        let labels = match tensor.get("_labels") {
            Some(TensorValue::Pointers(labels)) => labels.clone(),
            _ => match tensor.get("_label") {
                Some(TensorValue::Scalar(ScalarValue::String(s))) => vec![s.clone()],
                _ => Vec::new(),
            },
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

        let created_at = match tensor.get("_created_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some(*v as u64),
            _ => None,
        };
        let updated_at = match tensor.get("_updated_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some(*v as u64),
            _ => None,
        };

        Ok(Node {
            id,
            labels,
            properties,
            created_at,
            updated_at,
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

        let created_at = match tensor.get("_created_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some(*v as u64),
            _ => None,
        };
        let updated_at = match tensor.get("_updated_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some(*v as u64),
            _ => None,
        };

        Ok(Edge {
            id,
            from,
            to,
            edge_type,
            properties,
            directed,
            created_at,
            updated_at,
        })
    }

    /// Update a node's labels and/or properties.
    ///
    /// Pass `None` to leave labels unchanged. Properties are merged with
    /// existing properties; pass `PropertyValue::Null` to remove a property.
    pub fn update_node(
        &self,
        id: u64,
        labels: Option<Vec<String>>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<()> {
        // Get old node for index maintenance
        let old_node = self.get_node(id)?;

        let key = Self::node_key(id);
        let mut tensor = self
            .store
            .get(&key)
            .map_err(|_| GraphError::NodeNotFound(id))?;

        // Handle label changes
        let old_labels = old_node.labels.clone();
        let mut changed_props: HashMap<String, PropertyValue> = HashMap::new();

        if let Some(ref new_labels) = labels {
            // Unindex old labels
            for old_label in &old_labels {
                self.index_remove(
                    IndexTarget::Node,
                    "_label",
                    &OrderedPropertyValue::String(old_label.clone()),
                    id,
                );
            }
            tensor.set("_labels", TensorValue::Pointers(new_labels.clone()));
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

        tensor.set(
            "_updated_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis() as i64)),
        );

        self.store.put(key, tensor)?;

        // Index new labels
        if let Some(ref new_labels) = labels {
            for new_label in new_labels {
                self.index_add(
                    IndexTarget::Node,
                    "_label",
                    &OrderedPropertyValue::String(new_label.clone()),
                    id,
                );
            }
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

    /// Add a label to an existing node.
    pub fn add_label(&self, id: u64, label: &str) -> Result<()> {
        let node = self.get_node(id)?;

        // Check if label already exists
        if node.has_label(label) {
            return Ok(());
        }

        let key = Self::node_key(id);
        let mut tensor = self
            .store
            .get(&key)
            .map_err(|_| GraphError::NodeNotFound(id))?;

        let mut new_labels = node.labels;
        new_labels.push(label.to_string());
        tensor.set("_labels", TensorValue::Pointers(new_labels));
        tensor.set(
            "_updated_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis() as i64)),
        );
        self.store.put(key, tensor)?;

        // Update index
        self.index_add(
            IndexTarget::Node,
            "_label",
            &OrderedPropertyValue::String(label.to_string()),
            id,
        );

        Ok(())
    }

    /// Remove a label from an existing node.
    pub fn remove_label(&self, id: u64, label: &str) -> Result<()> {
        let node = self.get_node(id)?;

        // Check if label exists
        if !node.has_label(label) {
            return Ok(());
        }

        let key = Self::node_key(id);
        let mut tensor = self
            .store
            .get(&key)
            .map_err(|_| GraphError::NodeNotFound(id))?;

        let new_labels: Vec<String> = node
            .labels
            .iter()
            .filter(|l| *l != label)
            .cloned()
            .collect();
        tensor.set("_labels", TensorValue::Pointers(new_labels));
        tensor.set(
            "_updated_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis() as i64)),
        );
        self.store.put(key, tensor)?;

        // Update index
        self.index_remove(
            IndexTarget::Node,
            "_label",
            &OrderedPropertyValue::String(label.to_string()),
            id,
        );

        Ok(())
    }

    /// Get all labels for a node.
    pub fn get_node_labels(&self, id: u64) -> Result<Vec<String>> {
        let node = self.get_node(id)?;
        Ok(node.labels)
    }

    /// Check if a node has a specific label.
    pub fn node_has_label(&self, id: u64, label: &str) -> Result<bool> {
        let node = self.get_node(id)?;
        Ok(node.has_label(label))
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

        tensor.set(
            "_updated_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis() as i64)),
        );

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

    /// Returns the number of outgoing edges from a node.
    ///
    /// Equivalent to Neo4j's `size((n)-->())`.
    pub fn out_degree(&self, node_id: u64) -> Result<usize> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        Ok(self.get_edge_list(&Self::outgoing_edges_key(node_id)).len())
    }

    /// Returns the number of incoming edges to a node.
    ///
    /// Equivalent to Neo4j's `size((n)<--())`.
    pub fn in_degree(&self, node_id: u64) -> Result<usize> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        Ok(self.get_edge_list(&Self::incoming_edges_key(node_id)).len())
    }

    /// Returns the total degree of a node (in-degree + out-degree).
    ///
    /// For undirected edges, each edge is counted once in outgoing and once in incoming,
    /// so total degree counts each undirected edge twice (consistent with graph theory).
    /// Equivalent to Neo4j's `size((n)--())`.
    pub fn degree(&self, node_id: u64) -> Result<usize> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        let out = self.get_edge_list(&Self::outgoing_edges_key(node_id)).len();
        let in_ = self.get_edge_list(&Self::incoming_edges_key(node_id)).len();
        Ok(out + in_)
    }

    /// Returns the number of outgoing edges of a specific type from a node.
    ///
    /// Equivalent to Neo4j's `size((n)-[:TYPE]->())`.
    pub fn out_degree_by_type(&self, node_id: u64, edge_type: &str) -> Result<usize> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        let count = self
            .get_edge_list(&Self::outgoing_edges_key(node_id))
            .into_iter()
            .filter(|&edge_id| {
                self.get_edge(edge_id)
                    .map(|e| e.edge_type == edge_type)
                    .unwrap_or(false)
            })
            .count();
        Ok(count)
    }

    /// Returns the number of incoming edges of a specific type to a node.
    ///
    /// Equivalent to Neo4j's `size((n)<-[:TYPE]-())`.
    pub fn in_degree_by_type(&self, node_id: u64, edge_type: &str) -> Result<usize> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        let count = self
            .get_edge_list(&Self::incoming_edges_key(node_id))
            .into_iter()
            .filter(|&edge_id| {
                self.get_edge(edge_id)
                    .map(|e| e.edge_type == edge_type)
                    .unwrap_or(false)
            })
            .count();
        Ok(count)
    }

    /// Returns the total degree of a node for a specific edge type.
    ///
    /// Equivalent to Neo4j's `size((n)-[:TYPE]-())`.
    pub fn degree_by_type(&self, node_id: u64, edge_type: &str) -> Result<usize> {
        let out = self.out_degree_by_type(node_id, edge_type)?;
        let in_ = self.in_degree_by_type(node_id, edge_type)?;
        Ok(out + in_)
    }

    /// Returns all nodes in the graph, sorted by ID.
    ///
    /// Equivalent to Neo4j's `MATCH (n) RETURN n`.
    pub fn all_nodes(&self) -> Vec<Node> {
        let mut nodes = Vec::new();
        for key in self.store.scan("node:") {
            // Skip edge list keys (node:N:out, node:N:in)
            let Some(suffix) = key.strip_prefix("node:") else {
                continue;
            };
            if suffix.contains(':') {
                continue;
            }
            if let Ok(id) = suffix.parse::<u64>() {
                if let Ok(node) = self.get_node(id) {
                    nodes.push(node);
                }
            }
        }
        nodes.sort_by_key(|n| n.id);
        nodes
    }

    /// Returns all edges in the graph, sorted by ID.
    ///
    /// Equivalent to Neo4j's `MATCH ()-[r]->() RETURN r`.
    pub fn all_edges(&self) -> Vec<Edge> {
        let mut edges = Vec::new();
        for key in self.store.scan("edge:") {
            let Some(id_str) = key.strip_prefix("edge:") else {
                continue;
            };
            if let Ok(id) = id_str.parse::<u64>() {
                if let Ok(edge) = self.get_edge(id) {
                    edges.push(edge);
                }
            }
        }
        edges.sort_by_key(|e| e.id);
        edges
    }

    #[instrument(skip(self, filter), fields(node_id = node_id))]
    pub fn neighbors(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
        filter: Option<&TraversalFilter>,
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
                        // Apply edge filter
                        if let Some(f) = filter {
                            if !f.matches_edge(&edge) {
                                continue;
                            }
                        }
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
                        // Apply edge filter
                        if let Some(f) = filter {
                            if !f.matches_edge(&edge) {
                                continue;
                            }
                        }
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
                // Apply node filter
                if let Some(f) = filter {
                    if !f.matches_node(&node) {
                        continue;
                    }
                }
                neighbors.push(node);
            }
        }

        neighbors.sort_by_key(|n| n.id);
        Ok(neighbors)
    }

    // ========== Paginated Queries ==========

    /// Returns all nodes in the graph with pagination.
    #[must_use]
    pub fn all_nodes_paginated(&self, pagination: Pagination) -> PagedResult<Node> {
        let mut node_ids: Vec<u64> = self
            .store
            .scan("node:")
            .iter()
            .filter_map(|k| {
                let suffix = k.strip_prefix("node:")?;
                if suffix.contains(':') {
                    return None;
                }
                suffix.parse().ok()
            })
            .collect();
        node_ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(node_ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = node_ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Node> = node_ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_node(id).ok())
            .collect();

        PagedResult::new(items, total_count, has_more)
    }

    /// Returns all edges in the graph with pagination.
    #[must_use]
    pub fn all_edges_paginated(&self, pagination: Pagination) -> PagedResult<Edge> {
        let mut edge_ids: Vec<u64> = self
            .store
            .scan("edge:")
            .iter()
            .filter_map(|k| {
                let suffix = k.strip_prefix("edge:")?;
                suffix.parse().ok()
            })
            .collect();
        edge_ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(edge_ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = edge_ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Edge> = edge_ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_edge(id).ok())
            .collect();

        PagedResult::new(items, total_count, has_more)
    }

    /// Find nodes by label with pagination.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_label_paginated(
        &self,
        label: &str,
        pagination: Pagination,
    ) -> Result<PagedResult<Node>> {
        self.find_nodes_by_property_paginated(
            "_label",
            &PropertyValue::String(label.to_string()),
            pagination,
        )
    }

    /// Find nodes by exact property value match with pagination.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_by_property_paginated(
        &self,
        property: &str,
        value: &PropertyValue,
        pagination: Pagination,
    ) -> Result<PagedResult<Node>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Node, property.to_string());

        let indexes = self.btree_indexes.read();
        let mut ids: Vec<u64> = if let Some(btree) = indexes.get(&key) {
            btree.get(&ordered_value).cloned().unwrap_or_default()
        } else {
            drop(indexes);
            self.scan_node_ids_by_property(property, value)
        };
        ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Node> = ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_node(id).ok())
            .collect();

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Find nodes using a range comparison with pagination.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_nodes_where_paginated(
        &self,
        property: &str,
        op: RangeOp,
        value: &PropertyValue,
        pagination: Pagination,
    ) -> Result<PagedResult<Node>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Node, property.to_string());

        let indexes = self.btree_indexes.read();
        let mut ids: Vec<u64> = if let Some(btree) = indexes.get(&key) {
            match op {
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
            }
        } else {
            drop(indexes);
            self.scan_node_ids_where(property, op, value)
        };
        ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Node> = ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_node(id).ok())
            .collect();

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Find edges by type with pagination.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_edges_by_type_paginated(
        &self,
        edge_type: &str,
        pagination: Pagination,
    ) -> Result<PagedResult<Edge>> {
        self.find_edges_by_property_paginated(
            "_edge_type",
            &PropertyValue::String(edge_type.to_string()),
            pagination,
        )
    }

    /// Find edges by exact property value match with pagination.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_edges_by_property_paginated(
        &self,
        property: &str,
        value: &PropertyValue,
        pagination: Pagination,
    ) -> Result<PagedResult<Edge>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Edge, property.to_string());

        let indexes = self.btree_indexes.read();
        let mut ids: Vec<u64> = if let Some(btree) = indexes.get(&key) {
            btree.get(&ordered_value).cloned().unwrap_or_default()
        } else {
            drop(indexes);
            self.scan_edge_ids_by_property(property, value)
        };
        ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Edge> = ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_edge(id).ok())
            .collect();

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Find edges using a range comparison with pagination.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn find_edges_where_paginated(
        &self,
        property: &str,
        op: RangeOp,
        value: &PropertyValue,
        pagination: Pagination,
    ) -> Result<PagedResult<Edge>> {
        let ordered_value = OrderedPropertyValue::from(value);
        let key = (IndexTarget::Edge, property.to_string());

        let indexes = self.btree_indexes.read();
        let mut ids: Vec<u64> = if let Some(btree) = indexes.get(&key) {
            match op {
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
            }
        } else {
            drop(indexes);
            self.scan_edge_ids_where(property, op, value)
        };
        ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Edge> = ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_edge(id).ok())
            .collect();

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Returns edges connected to a node with pagination.
    ///
    /// # Errors
    ///
    /// Returns `NodeNotFound` if the node does not exist.
    pub fn edges_of_paginated(
        &self,
        node_id: u64,
        direction: Direction,
        pagination: Pagination,
    ) -> Result<PagedResult<Edge>> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }

        let mut edge_ids: Vec<u64> = {
            let mut ids = HashSet::new();
            if direction == Direction::Outgoing || direction == Direction::Both {
                for id in self.get_edge_list(&Self::outgoing_edges_key(node_id)) {
                    ids.insert(id);
                }
            }
            if direction == Direction::Incoming || direction == Direction::Both {
                for id in self.get_edge_list(&Self::incoming_edges_key(node_id)) {
                    ids.insert(id);
                }
            }
            ids.into_iter().collect()
        };
        edge_ids.sort_unstable();

        let total_count = if pagination.count_total {
            Some(edge_ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = edge_ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Edge> = edge_ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_edge(id).ok())
            .collect();

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Returns neighbors of a node with pagination.
    ///
    /// # Errors
    ///
    /// Returns `NodeNotFound` if the node does not exist.
    #[instrument(skip(self, filter), fields(node_id = node_id))]
    pub fn neighbors_paginated(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
        filter: Option<&TraversalFilter>,
        pagination: Pagination,
    ) -> Result<PagedResult<Node>> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }

        let mut neighbor_ids: Vec<u64> = {
            let mut ids = HashSet::new();

            if direction == Direction::Outgoing || direction == Direction::Both {
                let out_edges = self.get_edge_list(&Self::outgoing_edges_key(node_id));
                for edge_id in out_edges {
                    if let Ok(edge) = self.get_edge(edge_id) {
                        if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                            if let Some(f) = filter {
                                if !f.matches_edge(&edge) {
                                    continue;
                                }
                            }
                            if edge.from == node_id && edge.to != node_id {
                                ids.insert(edge.to);
                            } else if edge.to == node_id && edge.from != node_id {
                                ids.insert(edge.from);
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
                            if let Some(f) = filter {
                                if !f.matches_edge(&edge) {
                                    continue;
                                }
                            }
                            if edge.to == node_id && edge.from != node_id {
                                ids.insert(edge.from);
                            } else if edge.from == node_id && edge.to != node_id {
                                ids.insert(edge.to);
                            }
                        }
                    }
                }
            }

            ids.into_iter().collect()
        };
        neighbor_ids.sort_unstable();

        // Collect all neighbors that pass the node filter (pre-pagination)
        let filtered_ids: Vec<u64> = neighbor_ids
            .into_iter()
            .filter(|&id| {
                if let Ok(node) = self.get_node(id) {
                    if let Some(f) = filter {
                        f.matches_node(&node)
                    } else {
                        true
                    }
                } else {
                    false
                }
            })
            .collect();

        let total_count = if pagination.count_total {
            Some(filtered_ids.len())
        } else {
            None
        };
        let effective_limit = pagination.limit.unwrap_or(usize::MAX);
        let has_more = filtered_ids.len() > pagination.skip.saturating_add(effective_limit);

        let items: Vec<Node> = filtered_ids
            .into_iter()
            .skip(pagination.skip)
            .take(effective_limit)
            .filter_map(|id| self.get_node(id).ok())
            .collect();

        Ok(PagedResult::new(items, total_count, has_more))
    }

    // ========== Paginated Query Helpers ==========

    fn scan_node_ids_by_property(&self, property: &str, value: &PropertyValue) -> Vec<u64> {
        let mut ids = Vec::new();

        for key in self.store.scan("node:") {
            if key.contains(":out") || key.contains(":in") {
                continue;
            }
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(node) = self.get_node(id) {
                        let matches = if property == "_label" {
                            match value {
                                PropertyValue::String(s) => node.has_label(s),
                                _ => false,
                            }
                        } else {
                            node.properties.get(property) == Some(value)
                        };
                        if matches {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    fn scan_node_ids_where(&self, property: &str, op: RangeOp, value: &PropertyValue) -> Vec<u64> {
        let ordered_value = OrderedPropertyValue::from(value);
        let mut ids = Vec::new();

        for key in self.store.scan("node:") {
            if key.contains(":out") || key.contains(":in") {
                continue;
            }
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(node) = self.get_node(id) {
                        if let Some(prop_val) = node.properties.get(property).cloned() {
                            let ordered_pv = OrderedPropertyValue::from(&prop_val);
                            let matches = match op {
                                RangeOp::Lt => ordered_pv < ordered_value,
                                RangeOp::Le => ordered_pv <= ordered_value,
                                RangeOp::Gt => ordered_pv > ordered_value,
                                RangeOp::Ge => ordered_pv >= ordered_value,
                            };
                            if matches {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
        ids
    }

    fn scan_edge_ids_by_property(&self, property: &str, value: &PropertyValue) -> Vec<u64> {
        let mut ids = Vec::new();

        for key in self.store.scan("edge:") {
            if let Some(id_str) = key.strip_prefix("edge:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(edge) = self.get_edge(id) {
                        let matches = if property == "_edge_type" {
                            match value {
                                PropertyValue::String(s) => &edge.edge_type == s,
                                _ => false,
                            }
                        } else {
                            edge.properties.get(property) == Some(value)
                        };
                        if matches {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    fn scan_edge_ids_where(&self, property: &str, op: RangeOp, value: &PropertyValue) -> Vec<u64> {
        let ordered_value = OrderedPropertyValue::from(value);
        let mut ids = Vec::new();

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
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
        ids
    }

    pub fn traverse(
        &self,
        start: u64,
        direction: Direction,
        max_depth: usize,
        edge_type: Option<&str>,
        filter: Option<&TraversalFilter>,
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
                // Apply node filter (except for start node which is always included)
                let include = if current_id == start {
                    true
                } else if let Some(f) = filter {
                    f.matches_node(&node)
                } else {
                    true
                };
                if include {
                    result.push(node);
                }
            }

            if depth >= max_depth {
                continue;
            }

            let neighbors =
                self.get_neighbor_ids_filtered(current_id, edge_type, direction, filter);
            for neighbor_id in neighbors {
                if visited.insert(neighbor_id) {
                    queue.push_back((neighbor_id, depth + 1));
                }
            }
        }

        Ok(result)
    }

    #[allow(dead_code)]
    fn get_neighbor_ids(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Vec<u64> {
        self.get_neighbor_ids_filtered(node_id, edge_type, direction, None)
    }

    fn get_neighbor_ids_filtered(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
        filter: Option<&TraversalFilter>,
    ) -> Vec<u64> {
        let mut neighbor_ids = HashSet::new();

        if direction == Direction::Outgoing || direction == Direction::Both {
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(node_id));
            for edge_id in out_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                        // Apply edge filter
                        if let Some(f) = filter {
                            if !f.matches_edge(&edge) {
                                continue;
                            }
                        }
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
                        // Apply edge filter
                        if let Some(f) = filter {
                            if !f.matches_edge(&edge) {
                                continue;
                            }
                        }
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

    pub fn find_path(&self, from: u64, to: u64, filter: Option<&TraversalFilter>) -> Result<Path> {
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
                    // Apply edge filter
                    if let Some(f) = filter {
                        if !f.matches_edge(&edge) {
                            continue;
                        }
                    }

                    let neighbor = if edge.from == current {
                        edge.to
                    } else if !edge.directed && edge.to == current {
                        edge.from
                    } else {
                        continue;
                    };

                    // Apply node filter to neighbor (except target which we always want to reach)
                    if neighbor != to {
                        if let Some(f) = filter {
                            if let Ok(node) = self.get_node(neighbor) {
                                if !f.matches_node(&node) {
                                    continue;
                                }
                            }
                        }
                    }

                    if visited.insert(neighbor) {
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

    #[allow(clippy::unused_self)]
    #[allow(clippy::cast_precision_loss)]
    fn extract_edge_weight(&self, edge: &Edge, weight_property: &str) -> Result<f64> {
        match edge.properties.get(weight_property) {
            Some(PropertyValue::Float(w)) => {
                if *w < 0.0 {
                    return Err(GraphError::NegativeWeight {
                        edge_id: edge.id,
                        weight: *w,
                    });
                }
                Ok(*w)
            },
            Some(PropertyValue::Int(w)) => {
                if *w < 0 {
                    return Err(GraphError::NegativeWeight {
                        edge_id: edge.id,
                        weight: *w as f64,
                    });
                }
                Ok(*w as f64)
            },
            _ => Ok(1.0), // Default weight for missing/non-numeric
        }
    }

    /// Find shortest weighted path using Dijkstra's algorithm.
    ///
    /// Uses the specified edge property as weight. Missing properties default to 1.0.
    /// Returns error if any traversed edge has negative weight.
    #[instrument(skip(self, weight_property), fields(from = from, to = to))]
    pub fn find_weighted_path(
        &self,
        from: u64,
        to: u64,
        weight_property: &str,
    ) -> Result<WeightedPath> {
        // Validate nodes exist
        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        // Same node: zero-cost path
        if from == to {
            return Ok(WeightedPath {
                nodes: vec![from],
                edges: vec![],
                total_weight: 0.0,
            });
        }

        // Dijkstra's algorithm
        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut parent: HashMap<u64, (u64, u64)> = HashMap::new(); // node -> (parent, edge_id)
        let mut heap = BinaryHeap::new();

        dist.insert(from, 0.0);
        heap.push(DijkstraEntry {
            cost: 0.0,
            node_id: from,
        });

        while let Some(DijkstraEntry { cost, node_id }) = heap.pop() {
            // Early termination
            if node_id == to {
                return Ok(self.reconstruct_weighted_path(from, to, &parent, cost));
            }

            // Skip if we've found a better path
            if cost > *dist.get(&node_id).unwrap_or(&f64::INFINITY) {
                continue;
            }

            // Process outgoing edges
            for edge_id in self.get_edge_list(&Self::outgoing_edges_key(node_id)) {
                let edge = self.get_edge(edge_id)?;

                // Determine neighbor (handle directed/undirected)
                let neighbor = if edge.from == node_id {
                    edge.to
                } else if !edge.directed && edge.to == node_id {
                    edge.from
                } else {
                    continue;
                };

                let weight = self.extract_edge_weight(&edge, weight_property)?;
                let new_cost = cost + weight;

                if new_cost < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    dist.insert(neighbor, new_cost);
                    parent.insert(neighbor, (node_id, edge_id));
                    heap.push(DijkstraEntry {
                        cost: new_cost,
                        node_id: neighbor,
                    });
                }
            }

            // Also check incoming edges for undirected traversal
            for edge_id in self.get_edge_list(&Self::incoming_edges_key(node_id)) {
                let edge = self.get_edge(edge_id)?;

                if edge.directed {
                    continue; // Can't traverse directed edge backwards
                }

                let neighbor = if edge.to == node_id {
                    edge.from
                } else {
                    continue;
                };

                let weight = self.extract_edge_weight(&edge, weight_property)?;
                let new_cost = cost + weight;

                if new_cost < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    dist.insert(neighbor, new_cost);
                    parent.insert(neighbor, (node_id, edge_id));
                    heap.push(DijkstraEntry {
                        cost: new_cost,
                        node_id: neighbor,
                    });
                }
            }
        }

        Err(GraphError::PathNotFound)
    }

    #[allow(clippy::unused_self)]
    fn reconstruct_weighted_path(
        &self,
        from: u64,
        to: u64,
        parent: &HashMap<u64, (u64, u64)>,
        total_weight: f64,
    ) -> WeightedPath {
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

        WeightedPath {
            nodes,
            edges,
            total_weight,
        }
    }

    /// Find all shortest paths using BFS with multi-parent tracking.
    ///
    /// Returns all paths with the minimum hop count between two nodes.
    /// Use `config` to limit memory usage for graphs with many equal-length paths.
    #[instrument(skip(self, config), fields(from = from, to = to))]
    pub fn find_all_paths(
        &self,
        from: u64,
        to: u64,
        config: Option<AllPathsConfig>,
    ) -> Result<AllPaths> {
        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        if from == to {
            return Ok(AllPaths {
                paths: vec![Path {
                    nodes: vec![from],
                    edges: vec![],
                }],
                hop_count: 0,
            });
        }

        let config = config.unwrap_or_default();

        // Level-based BFS with multi-parent tracking
        let mut visited_level: HashMap<u64, usize> = HashMap::new();
        let mut parents: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
        let mut current_level = VecDeque::new();
        let mut next_level = VecDeque::new();
        let mut level = 0usize;
        let mut destination_level: Option<usize> = None;

        visited_level.insert(from, 0);
        current_level.push_back(from);

        while !current_level.is_empty() && destination_level.is_none() {
            level += 1;

            while let Some(current) = current_level.pop_front() {
                // Process outgoing edges
                for edge_id in self.get_edge_list(&Self::outgoing_edges_key(current)) {
                    if let Ok(edge) = self.get_edge(edge_id) {
                        let neighbor = if edge.from == current {
                            edge.to
                        } else if !edge.directed && edge.to == current {
                            edge.from
                        } else {
                            continue;
                        };

                        match visited_level.get(&neighbor) {
                            None => {
                                // First time reaching this node
                                visited_level.insert(neighbor, level);
                                parents.insert(neighbor, vec![(current, edge_id)]);
                                next_level.push_back(neighbor);

                                if neighbor == to {
                                    destination_level = Some(level);
                                }
                            },
                            Some(&node_level) if node_level == level => {
                                // Same level - add as additional parent
                                if let Some(p) = parents.get_mut(&neighbor) {
                                    if p.len() < config.max_parents_per_node {
                                        p.push((current, edge_id));
                                    }
                                }
                            },
                            _ => {}, // Earlier level - skip
                        }
                    }
                }
            }

            std::mem::swap(&mut current_level, &mut next_level);
        }

        let hop_count = destination_level.ok_or(GraphError::PathNotFound)?;
        let paths = self.enumerate_paths(from, to, &parents, config.max_paths);

        Ok(AllPaths { paths, hop_count })
    }

    #[allow(clippy::unused_self)]
    fn enumerate_paths(
        &self,
        from: u64,
        to: u64,
        parents: &HashMap<u64, Vec<(u64, u64)>>,
        max_paths: usize,
    ) -> Vec<Path> {
        let mut paths = Vec::new();
        let mut stack: Vec<(u64, Vec<u64>, Vec<u64>)> = vec![(to, vec![to], vec![])];

        while let Some((current, nodes, edges)) = stack.pop() {
            if paths.len() >= max_paths {
                break;
            }

            if current == from {
                let mut final_nodes = nodes;
                let mut final_edges = edges;
                final_nodes.reverse();
                final_edges.reverse();
                paths.push(Path {
                    nodes: final_nodes,
                    edges: final_edges,
                });
            } else if let Some(parent_list) = parents.get(&current) {
                for (parent, edge_id) in parent_list {
                    let mut new_nodes = nodes.clone();
                    new_nodes.push(*parent);
                    let mut new_edges = edges.clone();
                    new_edges.push(*edge_id);
                    stack.push((*parent, new_nodes, new_edges));
                }
            }
        }

        paths
    }

    /// Find all shortest weighted paths using Dijkstra with multi-parent tracking.
    ///
    /// Returns all paths with the minimum total weight between two nodes.
    /// Use `config` to limit memory usage for graphs with many equal-weight paths.
    #[allow(clippy::too_many_lines)]
    #[instrument(skip(self, weight_property, config), fields(from = from, to = to))]
    pub fn find_all_weighted_paths(
        &self,
        from: u64,
        to: u64,
        weight_property: &str,
        config: Option<AllPathsConfig>,
    ) -> Result<AllWeightedPaths> {
        const EPSILON: f64 = 1e-10;

        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        if from == to {
            return Ok(AllWeightedPaths {
                paths: vec![WeightedPath {
                    nodes: vec![from],
                    edges: vec![],
                    total_weight: 0.0,
                }],
                total_weight: 0.0,
            });
        }

        let config = config.unwrap_or_default();

        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut parents: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
        let mut heap = BinaryHeap::new();
        let mut destination_cost: Option<f64> = None;

        dist.insert(from, 0.0);
        heap.push(DijkstraEntry {
            cost: 0.0,
            node_id: from,
        });

        while let Some(DijkstraEntry { cost, node_id }) = heap.pop() {
            // Stop if we've exceeded destination cost
            if let Some(dc) = destination_cost {
                if cost > dc + EPSILON {
                    break;
                }
            }

            // Skip outdated entries
            if cost > *dist.get(&node_id).unwrap_or(&f64::INFINITY) + EPSILON {
                continue;
            }

            // Process outgoing edges
            for edge_id in self.get_edge_list(&Self::outgoing_edges_key(node_id)) {
                let edge = self.get_edge(edge_id)?;

                let neighbor = if edge.from == node_id {
                    edge.to
                } else if !edge.directed && edge.to == node_id {
                    edge.from
                } else {
                    continue;
                };

                let weight = self.extract_edge_weight(&edge, weight_property)?;
                let new_cost = cost + weight;
                let current_dist = *dist.get(&neighbor).unwrap_or(&f64::INFINITY);

                if new_cost < current_dist - EPSILON {
                    // Strictly better path
                    dist.insert(neighbor, new_cost);
                    parents.insert(neighbor, vec![(node_id, edge_id)]);
                    heap.push(DijkstraEntry {
                        cost: new_cost,
                        node_id: neighbor,
                    });

                    if neighbor == to {
                        destination_cost = Some(new_cost);
                    }
                } else if (new_cost - current_dist).abs() < EPSILON {
                    // Equal cost - add parent
                    if let Some(p) = parents.get_mut(&neighbor) {
                        if p.len() < config.max_parents_per_node {
                            p.push((node_id, edge_id));
                        }
                    }
                }
            }

            // Process incoming edges for undirected
            for edge_id in self.get_edge_list(&Self::incoming_edges_key(node_id)) {
                let edge = self.get_edge(edge_id)?;

                if edge.directed {
                    continue;
                }

                let neighbor = if edge.to == node_id {
                    edge.from
                } else {
                    continue;
                };

                let weight = self.extract_edge_weight(&edge, weight_property)?;
                let new_cost = cost + weight;
                let current_dist = *dist.get(&neighbor).unwrap_or(&f64::INFINITY);

                if new_cost < current_dist - EPSILON {
                    dist.insert(neighbor, new_cost);
                    parents.insert(neighbor, vec![(node_id, edge_id)]);
                    heap.push(DijkstraEntry {
                        cost: new_cost,
                        node_id: neighbor,
                    });

                    if neighbor == to {
                        destination_cost = Some(new_cost);
                    }
                } else if (new_cost - current_dist).abs() < EPSILON {
                    if let Some(p) = parents.get_mut(&neighbor) {
                        if p.len() < config.max_parents_per_node {
                            p.push((node_id, edge_id));
                        }
                    }
                }
            }
        }

        let total_weight = destination_cost.ok_or(GraphError::PathNotFound)?;
        let paths =
            self.enumerate_weighted_paths(from, to, &parents, total_weight, config.max_paths);

        Ok(AllWeightedPaths {
            paths,
            total_weight,
        })
    }

    #[allow(clippy::unused_self)]
    fn enumerate_weighted_paths(
        &self,
        from: u64,
        to: u64,
        parents: &HashMap<u64, Vec<(u64, u64)>>,
        total_weight: f64,
        max_paths: usize,
    ) -> Vec<WeightedPath> {
        let mut paths = Vec::new();
        let mut stack: Vec<(u64, Vec<u64>, Vec<u64>)> = vec![(to, vec![to], vec![])];

        while let Some((current, nodes, edges)) = stack.pop() {
            if paths.len() >= max_paths {
                break;
            }

            if current == from {
                let mut final_nodes = nodes;
                let mut final_edges = edges;
                final_nodes.reverse();
                final_edges.reverse();
                paths.push(WeightedPath {
                    nodes: final_nodes,
                    edges: final_edges,
                    total_weight,
                });
            } else if let Some(parent_list) = parents.get(&current) {
                for (parent, edge_id) in parent_list {
                    let mut new_nodes = nodes.clone();
                    new_nodes.push(*parent);
                    let mut new_edges = edges.clone();
                    new_edges.push(*edge_id);
                    stack.push((*parent, new_nodes, new_edges));
                }
            }
        }

        paths
    }

    /// Find all paths between two nodes within a hop range.
    ///
    /// Unlike `find_path` (shortest path) or `find_all_paths` (all shortest paths),
    /// this method finds all paths within a specified hop range, supporting
    /// Neo4j-style variable-length patterns like `[:KNOWS*1..5]`.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::type_complexity)]
    #[instrument(skip(self, config), fields(from = from, to = to))]
    pub fn find_variable_paths(
        &self,
        from: u64,
        to: u64,
        config: VariableLengthConfig,
    ) -> Result<VariableLengthPaths> {
        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        let mut paths = Vec::new();
        let mut stats = PathSearchStats::default();

        // Handle from == to with min_hops == 0
        if from == to && config.min_hops == 0 {
            paths.push(Path {
                nodes: vec![from],
                edges: vec![],
            });
            stats.paths_found = 1;
            stats.min_length = Some(0);
            stats.max_length = Some(0);
            if config.max_hops == 0 {
                return Ok(VariableLengthPaths { paths, stats });
            }
        }

        // DFS with explicit stack: (current_node, path_nodes, path_edges, visited)
        let mut stack: Vec<(u64, Vec<u64>, Vec<u64>, HashSet<u64>)> = Vec::new();

        let mut initial_visited = HashSet::new();
        if !config.allow_cycles {
            initial_visited.insert(from);
        }
        stack.push((from, vec![from], vec![], initial_visited));

        let edge_type_set: Option<HashSet<&str>> = config
            .edge_types
            .as_ref()
            .map(|types| types.iter().map(String::as_str).collect());

        while let Some((current, path_nodes, path_edges, visited)) = stack.pop() {
            if paths.len() >= config.max_paths {
                stats.truncated = true;
                break;
            }

            let current_depth = path_edges.len();
            if current_depth >= config.max_hops {
                continue;
            }

            stats.nodes_explored += 1;

            let neighbors = self.get_variable_path_neighbors_filtered(
                current,
                edge_type_set.as_ref(),
                config.direction,
                config.filter.as_ref(),
            );

            for (neighbor_id, edge_id) in neighbors {
                stats.edges_traversed += 1;

                if !config.allow_cycles && visited.contains(&neighbor_id) {
                    continue;
                }

                // Apply node filter (except for destination which we always want to reach)
                if neighbor_id != to {
                    if let Some(ref f) = config.filter {
                        if let Ok(node) = self.get_node(neighbor_id) {
                            if !f.matches_node(&node) {
                                continue;
                            }
                        }
                    }
                }

                let new_depth = current_depth + 1;
                let mut new_nodes = path_nodes.clone();
                let mut new_edges = path_edges.clone();
                new_nodes.push(neighbor_id);
                new_edges.push(edge_id);

                // Check if path reaches destination within hop range
                if neighbor_id == to && new_depth >= config.min_hops {
                    paths.push(Path {
                        nodes: new_nodes.clone(),
                        edges: new_edges.clone(),
                    });
                    stats.paths_found += 1;
                    stats.min_length =
                        Some(stats.min_length.map_or(new_depth, |m| m.min(new_depth)));
                    stats.max_length =
                        Some(stats.max_length.map_or(new_depth, |m| m.max(new_depth)));
                }

                // Continue exploring if under max_hops
                if new_depth < config.max_hops {
                    let mut new_visited = visited.clone();
                    if !config.allow_cycles {
                        new_visited.insert(neighbor_id);
                    }
                    stack.push((neighbor_id, new_nodes, new_edges, new_visited));
                }
            }
        }

        Ok(VariableLengthPaths { paths, stats })
    }

    #[allow(dead_code)]
    fn get_variable_path_neighbors(
        &self,
        node_id: u64,
        edge_types: Option<&HashSet<&str>>,
        direction: Direction,
    ) -> Vec<(u64, u64)> {
        self.get_variable_path_neighbors_filtered(node_id, edge_types, direction, None)
    }

    fn get_variable_path_neighbors_filtered(
        &self,
        node_id: u64,
        edge_types: Option<&HashSet<&str>>,
        direction: Direction,
        filter: Option<&TraversalFilter>,
    ) -> Vec<(u64, u64)> {
        let mut results = Vec::new();

        if direction == Direction::Outgoing || direction == Direction::Both {
            for edge_id in self.get_edge_list(&Self::outgoing_edges_key(node_id)) {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if let Some(types) = edge_types {
                        if !types.contains(edge.edge_type.as_str()) {
                            continue;
                        }
                    }
                    // Apply edge filter
                    if let Some(f) = filter {
                        if !f.matches_edge(&edge) {
                            continue;
                        }
                    }
                    let neighbor = if edge.from == node_id {
                        edge.to
                    } else if !edge.directed && edge.to == node_id {
                        edge.from
                    } else {
                        continue;
                    };
                    results.push((neighbor, edge_id));
                }
            }
        }

        if direction == Direction::Incoming || direction == Direction::Both {
            for edge_id in self.get_edge_list(&Self::incoming_edges_key(node_id)) {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if let Some(types) = edge_types {
                        if !types.contains(edge.edge_type.as_str()) {
                            continue;
                        }
                    }
                    // Apply edge filter
                    if let Some(f) = filter {
                        if !f.matches_edge(&edge) {
                            continue;
                        }
                    }
                    // For incoming direction, we can traverse incoming edges
                    let neighbor = if edge.to == node_id {
                        edge.from
                    } else if !edge.directed && edge.from == node_id {
                        edge.to
                    } else {
                        continue;
                    };
                    // Avoid duplicates when direction is Both and edge is undirected
                    if direction == Direction::Both && !edge.directed {
                        continue;
                    }
                    results.push((neighbor, edge_id));
                }
            }
        }

        results
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

    // ========== Aggregation Methods ==========

    /// Returns the total count of nodes.
    #[must_use]
    pub fn count_nodes(&self) -> u64 {
        self.node_count() as u64
    }

    /// Returns the count of nodes with a specific label.
    pub fn count_nodes_by_label(&self, label: &str) -> Result<u64> {
        Ok(self.find_nodes_by_label(label)?.len() as u64)
    }

    /// Returns the total count of edges.
    #[must_use]
    pub fn count_edges(&self) -> u64 {
        self.edge_count() as u64
    }

    /// Returns the count of edges with a specific type.
    pub fn count_edges_by_type(&self, edge_type: &str) -> Result<u64> {
        Ok(self.find_edges_by_type(edge_type)?.len() as u64)
    }

    /// Aggregates a property across all nodes.
    #[must_use]
    pub fn aggregate_node_property(&self, property: &str) -> AggregateResult {
        let nodes = self.all_nodes();
        Self::aggregate_property_values(nodes.iter().filter_map(|n| n.properties.get(property)))
    }

    /// Aggregates a property across nodes with a specific label.
    pub fn aggregate_node_property_by_label(
        &self,
        label: &str,
        property: &str,
    ) -> Result<AggregateResult> {
        let nodes = self.find_nodes_by_label(label)?;
        Ok(Self::aggregate_property_values(
            nodes.iter().filter_map(|n| n.properties.get(property)),
        ))
    }

    /// Aggregates a property across nodes matching a filter condition.
    pub fn aggregate_node_property_where(
        &self,
        filter_prop: &str,
        op: RangeOp,
        value: &PropertyValue,
        agg_prop: &str,
    ) -> Result<AggregateResult> {
        let nodes = self.find_nodes_where(filter_prop, op, value)?;
        Ok(Self::aggregate_property_values(
            nodes.iter().filter_map(|n| n.properties.get(agg_prop)),
        ))
    }

    /// Aggregates a property across all edges.
    #[must_use]
    pub fn aggregate_edge_property(&self, property: &str) -> AggregateResult {
        let edges = self.all_edges();
        Self::aggregate_property_values(edges.iter().filter_map(|e| e.properties.get(property)))
    }

    /// Aggregates a property across edges with a specific type.
    pub fn aggregate_edge_property_by_type(
        &self,
        edge_type: &str,
        property: &str,
    ) -> Result<AggregateResult> {
        let edges = self.find_edges_by_type(edge_type)?;
        Ok(Self::aggregate_property_values(
            edges.iter().filter_map(|e| e.properties.get(property)),
        ))
    }

    /// Aggregates a property across edges matching a filter condition.
    pub fn aggregate_edge_property_where(
        &self,
        filter_prop: &str,
        op: RangeOp,
        value: &PropertyValue,
        agg_prop: &str,
    ) -> Result<AggregateResult> {
        let edges = self.find_edges_where(filter_prop, op, value)?;
        Ok(Self::aggregate_property_values(
            edges.iter().filter_map(|e| e.properties.get(agg_prop)),
        ))
    }

    /// Returns the sum of a numeric property across all nodes.
    #[must_use]
    pub fn sum_node_property(&self, property: &str) -> Option<f64> {
        self.aggregate_node_property(property).sum
    }

    /// Returns the average of a numeric property across all nodes.
    #[must_use]
    pub fn avg_node_property(&self, property: &str) -> Option<f64> {
        self.aggregate_node_property(property).avg
    }

    /// Returns the sum of a numeric property across all edges.
    #[must_use]
    pub fn sum_edge_property(&self, property: &str) -> Option<f64> {
        self.aggregate_edge_property(property).sum
    }

    /// Returns the average of a numeric property across all edges.
    #[must_use]
    pub fn avg_edge_property(&self, property: &str) -> Option<f64> {
        self.aggregate_edge_property(property).avg
    }

    /// Core aggregation logic over property values.
    #[allow(clippy::cast_precision_loss)] // Intentional for aggregation: values near i64::MAX are uncommon
    fn aggregate_property_values<'a>(
        values: impl Iterator<Item = &'a PropertyValue>,
    ) -> AggregateResult {
        let values: Vec<&PropertyValue> = values.collect();

        if values.is_empty() {
            return AggregateResult::empty();
        }

        let count = values.len() as u64;

        // Extract numeric values for sum/avg
        let numeric: Vec<f64> = values
            .iter()
            .filter_map(|v| match v {
                PropertyValue::Int(i) => Some(*i as f64),
                PropertyValue::Float(f) => Some(*f),
                _ => None,
            })
            .collect();

        let (sum, avg) = if numeric.is_empty() {
            (None, None)
        } else if numeric.len() >= Self::AGGREGATE_PARALLEL_THRESHOLD {
            // Parallel aggregation for large datasets
            let total: f64 = numeric.par_iter().sum();
            (Some(total), Some(total / numeric.len() as f64))
        } else {
            // Sequential aggregation
            let total: f64 = numeric.iter().sum();
            (Some(total), Some(total / numeric.len() as f64))
        };

        // Find min/max across all comparable values
        let min = values
            .iter()
            .copied()
            .min_by(Self::compare_property_values)
            .cloned();
        let max = values
            .iter()
            .copied()
            .max_by(Self::compare_property_values)
            .cloned();

        AggregateResult {
            count,
            sum,
            avg,
            min,
            max,
        }
    }

    /// Comparison function for `PropertyValue` ordering.
    #[allow(clippy::trivially_copy_pass_by_ref)] // Required by min_by/max_by iterator adapters
    #[allow(clippy::cast_precision_loss)] // Intentional for mixed Int/Float comparison
    fn compare_property_values(a: &&PropertyValue, b: &&PropertyValue) -> CmpOrdering {
        match (a, b) {
            (PropertyValue::Null, PropertyValue::Null) => CmpOrdering::Equal,
            (PropertyValue::Null, _) => CmpOrdering::Less,
            (_, PropertyValue::Null) => CmpOrdering::Greater,
            (PropertyValue::Int(x), PropertyValue::Int(y)) => x.cmp(y),
            (PropertyValue::Float(x), PropertyValue::Float(y)) => {
                x.partial_cmp(y).unwrap_or(CmpOrdering::Equal)
            },
            (PropertyValue::Int(x), PropertyValue::Float(y)) => {
                (*x as f64).partial_cmp(y).unwrap_or(CmpOrdering::Equal)
            },
            (PropertyValue::Float(x), PropertyValue::Int(y)) => {
                x.partial_cmp(&(*y as f64)).unwrap_or(CmpOrdering::Equal)
            },
            (PropertyValue::String(x), PropertyValue::String(y)) => x.cmp(y),
            (PropertyValue::Bool(x), PropertyValue::Bool(y)) => x.cmp(y),
            // Different types: use type ordinal for consistent ordering
            _ => Self::property_value_type_ordinal(a).cmp(&Self::property_value_type_ordinal(b)),
        }
    }

    /// Returns a numeric ordinal for property value types for consistent ordering.
    const fn property_value_type_ordinal(v: &PropertyValue) -> u8 {
        match v {
            PropertyValue::Null => 0,
            PropertyValue::Bool(_) => 1,
            PropertyValue::Int(_) => 2,
            PropertyValue::Float(_) => 3,
            PropertyValue::String(_) => 4,
        }
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
        self.unindex_node_properties(id, &node.labels, &node.properties);

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
        assert_eq!(node.labels, vec!["Person"]);
        assert!(node.has_label("Person"));
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

        let out_neighbors = engine
            .neighbors(n1, None, Direction::Outgoing, None)
            .unwrap();
        assert_eq!(out_neighbors.len(), 2);

        let in_neighbors = engine
            .neighbors(n1, None, Direction::Incoming, None)
            .unwrap();
        assert_eq!(in_neighbors.len(), 0);

        let n2_in = engine
            .neighbors(n2, None, Direction::Incoming, None)
            .unwrap();
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

        let n1_neighbors = engine.neighbors(n1, None, Direction::Both, None).unwrap();
        assert_eq!(n1_neighbors.len(), 1);
        assert_eq!(n1_neighbors[0].id, n2);

        let n2_neighbors = engine.neighbors(n2, None, Direction::Both, None).unwrap();
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
            .neighbors(n1, Some("KNOWS"), Direction::Outgoing, None)
            .unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].id, n2);

        let works = engine
            .neighbors(n1, Some("WORKS_AT"), Direction::Outgoing, None)
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
        let result = engine
            .traverse(n1, Direction::Outgoing, 0, None, None)
            .unwrap();
        assert_eq!(result.len(), 1);

        // Depth 1: start + direct neighbors
        let result = engine
            .traverse(n1, Direction::Outgoing, 1, None, None)
            .unwrap();
        assert_eq!(result.len(), 2);

        // Depth 3: all nodes
        let result = engine
            .traverse(n1, Direction::Outgoing, 3, None, None)
            .unwrap();
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
        let result = engine
            .traverse(n1, Direction::Outgoing, 10, None, None)
            .unwrap();
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

        let path = engine.find_path(n1, n3, None).unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]);
        assert_eq!(path.edges.len(), 2);
    }

    #[test]
    fn find_path_same_node() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        let path = engine.find_path(n1, n1, None).unwrap();
        assert_eq!(path.nodes, vec![n1]);
        assert!(path.edges.is_empty());
    }

    #[test]
    fn find_path_not_found() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let result = engine.find_path(n1, n2, None);
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

        let path = engine.find_path(n1, n4, None).unwrap();
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
            .traverse(node_ids[0], Direction::Outgoing, 10, None, None)
            .unwrap();
        assert_eq!(result.len(), 11);

        // Traverse full chain
        let result = engine
            .traverse(node_ids[0], Direction::Outgoing, 1000, None, None)
            .unwrap();
        assert_eq!(result.len(), 1000);

        // Find path from 0 to 50
        let path = engine.find_path(node_ids[0], node_ids[50], None).unwrap();
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
        let from_n1 = engine
            .neighbors(n1, None, Direction::Outgoing, None)
            .unwrap();
        assert_eq!(from_n1.len(), 2);

        // From n2: cannot reach n1 (directed edge goes other way)
        let from_n2 = engine
            .neighbors(n2, None, Direction::Outgoing, None)
            .unwrap();
        assert_eq!(from_n2.len(), 0);

        // From n3: can reach n1 (undirected)
        let from_n3 = engine
            .neighbors(n3, None, Direction::Outgoing, None)
            .unwrap();
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
        let result = engine.neighbors(999, None, Direction::Both, None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn traverse_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.traverse(999, Direction::Both, 5, None, None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn find_path_nonexistent_node() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.find_path(999, n1, None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));

        let result = engine.find_path(n1, 999, None);
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
        let neighbors = engine.neighbors(n1, None, Direction::Both, None).unwrap();
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
            labels: vec!["Test".into()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };
        let _ = node.clone();

        let edge = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "TEST".into(),
            properties: HashMap::new(),
            directed: true,
            created_at: None,
            updated_at: None,
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
        let result = engine
            .traverse(n3, Direction::Incoming, 10, None, None)
            .unwrap();
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
            .neighbors(n3, Some("KNOWS"), Direction::Incoming, None)
            .unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].id, n1);

        // Get only WORKS_WITH incoming neighbors of n3
        let works = engine
            .neighbors(n3, Some("WORKS_WITH"), Direction::Incoming, None)
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
        let path = engine.find_path(n1, n3, None).unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]);

        // Should also find reverse path n3 -> n2 -> n1
        let path_rev = engine.find_path(n3, n1, None).unwrap();
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
        let result = engine
            .traverse(n2, Direction::Incoming, 1, None, None)
            .unwrap();
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
        let neighbors = engine
            .neighbors(n2, None, Direction::Incoming, None)
            .unwrap();
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
        let neighbors = engine
            .neighbors(hub, None, Direction::Outgoing, None)
            .unwrap();
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
            .update_node(id, Some(vec!["Employee".to_string()]), HashMap::new())
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Employee"]);
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
        assert_eq!(node.labels, vec!["Person"]); // unchanged
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
                .neighbors(n1, None, Direction::Outgoing, None)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            engine
                .neighbors(n2, None, Direction::Incoming, None)
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
                .neighbors(n1, None, Direction::Outgoing, None)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            engine
                .neighbors(n2, None, Direction::Incoming, None)
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
            engine
                .neighbors(n1, None, Direction::Both, None)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            engine
                .neighbors(n2, None, Direction::Both, None)
                .unwrap()
                .len(),
            1
        );

        // Delete edge
        engine.delete_edge(edge_id).unwrap();

        // Both directions should be empty
        assert_eq!(
            engine
                .neighbors(n1, None, Direction::Both, None)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            engine
                .neighbors(n2, None, Direction::Both, None)
                .unwrap()
                .len(),
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
                .neighbors(n1, None, Direction::Outgoing, None)
                .unwrap()
                .len(),
            0
        );

        // n3 should have no incoming neighbors now
        assert_eq!(
            engine
                .neighbors(n3, None, Direction::Incoming, None)
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
            labels: vec!["Test".into()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };
        let node2 = Node {
            id: 1,
            labels: vec!["Test".into()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };
        assert_eq!(node1, node2);

        let edge1 = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "E".into(),
            properties: HashMap::new(),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        let edge2 = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "E".into(),
            properties: HashMap::new(),
            directed: true,
            created_at: None,
            updated_at: None,
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

        // With auto-index, label index is created on first node
        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Company", HashMap::new()).unwrap();

        // Index already exists from auto-creation
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

        // Edge type index is auto-created on first edge
        assert!(!engine.has_edge_index("_edge_type"));
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        assert!(engine.has_edge_index("_edge_type"));

        engine
            .create_edge(n2, n3, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        // Manual creation after auto-init returns error
        assert!(engine.create_edge_type_index().is_err());

        let knows = engine.find_edges_by_type("KNOWS").unwrap();
        assert_eq!(knows.len(), 1);

        let follows = engine.find_edges_by_type("FOLLOWS").unwrap();
        assert_eq!(follows.len(), 1);
    }

    #[test]
    fn drop_node_index() {
        let engine = GraphEngine::new();

        // Auto-creates label index on first node
        engine.create_node("Person", HashMap::new()).unwrap();

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

        // Edge type index is auto-created on first edge
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

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Edge type index is auto-created on first edge
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

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Edge type index is auto-created on first edge
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

        // Create data (auto-creates label index)
        engine1.create_node("Person", HashMap::new()).unwrap();
        engine1.create_node("Person", HashMap::new()).unwrap();

        // Verify index works
        assert!(engine1.has_node_index("_label"));
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

        // Edge type index is auto-created on first edge
        engine
            .create_edge(n1, n2, "LIKES", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n1, "KNOWS", HashMap::new(), true)
            .unwrap();

        // Verify index exists
        assert!(engine.has_edge_index("_edge_type"));
        let likes = engine.find_edges_by_type("LIKES").unwrap();
        assert_eq!(likes.len(), 1);

        // Rebuild from store
        let engine2 = GraphEngine::with_store(store);
        assert!(engine2.has_edge_index("_edge_type"));
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

    // ========== Multiple Labels Tests ==========

    #[test]
    fn create_node_with_multiple_labels() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(
                vec!["Person".into(), "Employee".into(), "Manager".into()],
                HashMap::new(),
            )
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Person", "Employee", "Manager"]);
        assert!(node.has_label("Person"));
        assert!(node.has_label("Employee"));
        assert!(node.has_label("Manager"));
        assert!(!node.has_label("Admin"));
    }

    #[test]
    fn create_node_with_empty_labels() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(Vec::new(), HashMap::new())
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert!(node.labels.is_empty());
        assert!(!node.has_label("Anything"));
    }

    #[test]
    fn create_node_single_label_backwards_compat() {
        let engine = GraphEngine::new();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Person"]);
        assert!(node.has_label("Person"));
    }

    #[test]
    fn get_node_returns_all_labels() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(vec!["A".into(), "B".into(), "C".into()], HashMap::new())
            .unwrap();

        let labels = engine.get_node_labels(id).unwrap();
        assert_eq!(labels, vec!["A", "B", "C"]);
    }

    #[test]
    fn add_label_to_existing_node() {
        let engine = GraphEngine::new();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        engine.add_label(id, "Employee").unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Person", "Employee"]);
    }

    #[test]
    fn add_label_already_present_idempotent() {
        let engine = GraphEngine::new();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        engine.add_label(id, "Person").unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Person"]); // not duplicated
    }

    #[test]
    fn remove_label_from_node() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();

        engine.remove_label(id, "Employee").unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Person"]);
        assert!(!node.has_label("Employee"));
    }

    #[test]
    fn remove_label_not_present_ok() {
        let engine = GraphEngine::new();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        engine.remove_label(id, "NonExistent").unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Person"]); // unchanged
    }

    #[test]
    fn remove_last_label_leaves_empty() {
        let engine = GraphEngine::new();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        engine.remove_label(id, "Person").unwrap();

        let node = engine.get_node(id).unwrap();
        assert!(node.labels.is_empty());
    }

    #[test]
    fn update_node_replaces_all_labels() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();

        engine
            .update_node(
                id,
                Some(vec!["Admin".into(), "Manager".into()]),
                HashMap::new(),
            )
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.labels, vec!["Admin", "Manager"]);
        assert!(!node.has_label("Person"));
        assert!(!node.has_label("Employee"));
    }

    #[test]
    fn find_nodes_by_label_returns_multi_label_nodes() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();
        let multi_id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();
        engine.create_node("Company", HashMap::new()).unwrap();

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 2);

        let employees = engine.find_nodes_by_label("Employee").unwrap();
        assert_eq!(employees.len(), 1);
        assert_eq!(employees[0].id, multi_id);
    }

    #[test]
    fn find_nodes_by_all_labels_intersection() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();
        let all_three = engine
            .create_node_with_labels(
                vec!["Person".into(), "Employee".into(), "Manager".into()],
                HashMap::new(),
            )
            .unwrap();

        let result = engine
            .find_nodes_by_all_labels(&["Person", "Employee", "Manager"])
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, all_three);

        let result = engine
            .find_nodes_by_all_labels(&["Person", "Employee"])
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn find_nodes_by_all_labels_empty_returns_empty() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();

        let result = engine.find_nodes_by_all_labels(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn find_nodes_by_all_labels_no_match() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Employee", HashMap::new()).unwrap();

        let result = engine
            .find_nodes_by_all_labels(&["Person", "Employee"])
            .unwrap();
        assert!(result.is_empty()); // no node has both
    }

    #[test]
    fn find_nodes_by_any_label_union() {
        let engine = GraphEngine::new();

        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Employee", HashMap::new()).unwrap();
        engine.create_node("Company", HashMap::new()).unwrap();

        let result = engine
            .find_nodes_by_any_label(&["Person", "Employee"])
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn node_has_label_true() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(vec!["A".into(), "B".into()], HashMap::new())
            .unwrap();

        assert!(engine.node_has_label(id, "A").unwrap());
        assert!(engine.node_has_label(id, "B").unwrap());
    }

    #[test]
    fn node_has_label_false() {
        let engine = GraphEngine::new();

        let id = engine.create_node("A", HashMap::new()).unwrap();

        assert!(!engine.node_has_label(id, "B").unwrap());
    }

    #[test]
    fn multi_label_node_indexed_under_each_label() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();

        let id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 1);
        assert_eq!(persons[0].id, id);

        let employees = engine.find_nodes_by_label("Employee").unwrap();
        assert_eq!(employees.len(), 1);
        assert_eq!(employees[0].id, id);
    }

    #[test]
    fn add_label_updates_index() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();

        let id = engine.create_node("Person", HashMap::new()).unwrap();

        // Before adding label
        let employees = engine.find_nodes_by_label("Employee").unwrap();
        assert!(employees.is_empty());

        engine.add_label(id, "Employee").unwrap();

        // After adding label
        let employees = engine.find_nodes_by_label("Employee").unwrap();
        assert_eq!(employees.len(), 1);
    }

    #[test]
    fn remove_label_updates_index() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();

        let id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();

        engine.remove_label(id, "Employee").unwrap();

        let employees = engine.find_nodes_by_label("Employee").unwrap();
        assert!(employees.is_empty());

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 1);
    }

    #[test]
    fn delete_node_removes_all_labels_from_index() {
        let engine = GraphEngine::new();

        engine.create_label_index().unwrap();

        let id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();

        engine.delete_node(id).unwrap();

        let persons = engine.find_nodes_by_label("Person").unwrap();
        assert!(persons.is_empty());

        let employees = engine.find_nodes_by_label("Employee").unwrap();
        assert!(employees.is_empty());
    }

    #[test]
    fn with_store_rebuilds_multi_label_index() {
        let store = TensorStore::new();

        let engine = GraphEngine::with_store(store.clone());
        engine.create_label_index().unwrap();

        engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();
        drop(engine);

        // Rebuild from store
        let engine2 = GraphEngine::with_store(store);

        let persons = engine2.find_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 1);

        let employees = engine2.find_nodes_by_label("Employee").unwrap();
        assert_eq!(employees.len(), 1);
    }

    #[test]
    fn unicode_labels() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(vec!["Gebruiker".into(), "Benutzer".into()], HashMap::new())
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert!(node.has_label("Gebruiker"));
        assert!(node.has_label("Benutzer"));
    }

    #[test]
    fn empty_string_label() {
        let engine = GraphEngine::new();

        let id = engine
            .create_node_with_labels(vec!["".into(), "Valid".into()], HashMap::new())
            .unwrap();

        let node = engine.get_node(id).unwrap();
        assert!(node.has_label(""));
        assert!(node.has_label("Valid"));
    }

    #[test]
    fn node_has_label_helper_method() {
        let node = Node {
            id: 1,
            labels: vec!["A".into(), "B".into(), "C".into()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };

        assert!(node.has_label("A"));
        assert!(node.has_label("B"));
        assert!(node.has_label("C"));
        assert!(!node.has_label("D"));
        assert!(!node.has_label(""));
    }

    #[test]
    fn label_index_auto_created_on_first_node() {
        let engine = GraphEngine::new();

        // No index initially
        assert!(!engine.has_node_index("_label"));

        // Create first node
        engine.create_node("Person", HashMap::new()).unwrap();

        // Index now exists
        assert!(engine.has_node_index("_label"));
    }

    #[test]
    fn find_nodes_by_label_uses_auto_index() {
        let engine = GraphEngine::new();

        // Create nodes (triggers auto-index)
        for i in 0..100 {
            let label = if i % 2 == 0 { "Even" } else { "Odd" };
            engine.create_node(label, HashMap::new()).unwrap();
        }

        // Query uses index (not scan)
        assert!(engine.has_node_index("_label"));
        let evens = engine.find_nodes_by_label("Even").unwrap();
        assert_eq!(evens.len(), 50);
    }

    #[test]
    fn manual_label_index_still_works() {
        let engine = GraphEngine::new();

        // Manual creation before any nodes
        engine.create_label_index().unwrap();
        assert!(engine.has_node_index("_label"));

        // Creating nodes still works
        engine.create_node("Test", HashMap::new()).unwrap();

        // Second manual call returns error
        assert!(engine.create_label_index().is_err());
    }

    #[test]
    fn label_index_survives_reload() {
        let store = TensorStore::new();

        {
            let engine = GraphEngine::with_store(store.clone());
            engine.create_node("Person", HashMap::new()).unwrap();
            assert!(engine.has_node_index("_label"));
        }

        // Reload
        let engine2 = GraphEngine::with_store(store);
        assert!(engine2.has_node_index("_label"));
    }

    #[test]
    fn edge_type_index_auto_created_on_first_edge() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // No edge type index initially
        assert!(!engine.has_edge_index("_edge_type"));

        // Create first edge
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        // Index now exists
        assert!(engine.has_edge_index("_edge_type"));
    }

    #[test]
    fn find_edges_by_type_uses_auto_index() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create edges (triggers auto-index)
        for i in 0..50 {
            let edge_type = if i % 2 == 0 { "KNOWS" } else { "FOLLOWS" };
            engine
                .create_edge(n1, n2, edge_type, HashMap::new(), true)
                .unwrap();
        }

        // Query uses index
        assert!(engine.has_edge_index("_edge_type"));
        let knows = engine.find_edges_by_type("KNOWS").unwrap();
        assert_eq!(knows.len(), 25);
    }

    #[test]
    fn manual_edge_type_index_still_works() {
        let engine = GraphEngine::new();

        // Manual creation before any edges
        engine.create_edge_type_index().unwrap();
        assert!(engine.has_edge_index("_edge_type"));

        // Creating edges still works
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        // Second manual call returns error
        assert!(engine.create_edge_type_index().is_err());
    }

    #[test]
    fn edge_type_index_survives_reload() {
        let store = TensorStore::new();

        {
            let engine = GraphEngine::with_store(store.clone());
            let n1 = engine.create_node("A", HashMap::new()).unwrap();
            let n2 = engine.create_node("B", HashMap::new()).unwrap();
            engine
                .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
                .unwrap();
            assert!(engine.has_edge_index("_edge_type"));
        }

        // Reload
        let engine2 = GraphEngine::with_store(store);
        assert!(engine2.has_edge_index("_edge_type"));
    }

    #[test]
    fn find_weighted_path_simple() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("weight".to_string(), PropertyValue::Float(1.5));
        engine.create_edge(n1, n2, "ROAD", props1, true).unwrap();

        let mut props2 = HashMap::new();
        props2.insert("weight".to_string(), PropertyValue::Float(2.5));
        engine.create_edge(n2, n3, "ROAD", props2, true).unwrap();

        let path = engine.find_weighted_path(n1, n3, "weight").unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]);
        assert_eq!(path.edges.len(), 2);
        assert!((path.total_weight - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_weighted_path_same_node() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        let path = engine.find_weighted_path(n1, n1, "weight").unwrap();
        assert_eq!(path.nodes, vec![n1]);
        assert!(path.edges.is_empty());
        assert!((path.total_weight - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_weighted_path_chooses_lighter_route() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        // Direct heavy path: n1 -> n3 (weight 10)
        let mut heavy = HashMap::new();
        heavy.insert("weight".to_string(), PropertyValue::Float(10.0));
        engine.create_edge(n1, n3, "DIRECT", heavy, true).unwrap();

        // Indirect light path: n1 -> n2 -> n3 (weight 1 + 2 = 3)
        let mut light1 = HashMap::new();
        light1.insert("weight".to_string(), PropertyValue::Float(1.0));
        engine.create_edge(n1, n2, "ROAD", light1, true).unwrap();

        let mut light2 = HashMap::new();
        light2.insert("weight".to_string(), PropertyValue::Float(2.0));
        engine.create_edge(n2, n3, "ROAD", light2, true).unwrap();

        let path = engine.find_weighted_path(n1, n3, "weight").unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]); // Takes lighter route
        assert!((path.total_weight - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_weighted_path_default_weight() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        // No weight properties - should default to 1.0 each
        engine
            .create_edge(n1, n2, "ROAD", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "ROAD", HashMap::new(), true)
            .unwrap();

        let path = engine.find_weighted_path(n1, n3, "weight").unwrap();
        assert_eq!(path.nodes.len(), 3);
        assert!((path.total_weight - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_weighted_path_int_weight() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("cost".to_string(), PropertyValue::Int(5));
        engine.create_edge(n1, n2, "ROAD", props, true).unwrap();

        let path = engine.find_weighted_path(n1, n2, "cost").unwrap();
        assert!((path.total_weight - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_weighted_path_negative_weight_error() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(-1.0));
        engine.create_edge(n1, n2, "ROAD", props, true).unwrap();

        let result = engine.find_weighted_path(n1, n2, "weight");
        assert!(matches!(result, Err(GraphError::NegativeWeight { .. })));
    }

    #[test]
    fn find_weighted_path_not_found() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        // No edge between them

        let result = engine.find_weighted_path(n1, n2, "weight");
        assert!(matches!(result, Err(GraphError::PathNotFound)));
    }

    #[test]
    fn find_weighted_path_node_not_found() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.find_weighted_path(n1, 999, "weight");
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn find_weighted_path_undirected() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        // Undirected edge n2 <-> n1
        engine
            .create_edge(n2, n1, "ROAD", props.clone(), false)
            .unwrap();
        // Directed edge n2 -> n3
        engine.create_edge(n2, n3, "ROAD", props, true).unwrap();

        // Can traverse n1 -> n2 (backwards on undirected) -> n3
        let path = engine.find_weighted_path(n1, n3, "weight").unwrap();
        assert_eq!(path.nodes, vec![n1, n2, n3]);
    }

    #[test]
    fn find_weighted_path_zero_weight() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(0.0));
        engine.create_edge(n1, n2, "FREE", props, true).unwrap();

        let path = engine.find_weighted_path(n1, n2, "weight").unwrap();
        assert!((path.total_weight - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_weighted_path_large_graph() {
        let engine = GraphEngine::new();
        let mut nodes = Vec::new();

        // Create chain of 100 nodes
        for i in 0..100 {
            let label = format!("N{i}");
            nodes.push(engine.create_node(&label, HashMap::new()).unwrap());
        }

        // Connect with increasing weights
        for i in 0..99 {
            let mut props = HashMap::new();
            props.insert("weight".to_string(), PropertyValue::Float((i + 1) as f64));
            engine
                .create_edge(nodes[i], nodes[i + 1], "NEXT", props, true)
                .unwrap();
        }

        let path = engine
            .find_weighted_path(nodes[0], nodes[99], "weight")
            .unwrap();
        assert_eq!(path.nodes.len(), 100);
        // Sum of 1 + 2 + ... + 99 = 99 * 100 / 2 = 4950
        assert!((path.total_weight - 4950.0).abs() < f64::EPSILON);
    }

    #[test]
    fn weighted_path_equality() {
        let p1 = WeightedPath {
            nodes: vec![1, 2, 3],
            edges: vec![10, 20],
            total_weight: 5.0,
        };
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }

    // ==================== find_all_paths tests ====================

    #[test]
    fn find_all_paths_simple() {
        // Linear chain: A -> B -> C (only 1 path)
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "NEXT", HashMap::new(), true)
            .unwrap();

        let result = engine.find_all_paths(a, c, None).unwrap();
        assert_eq!(result.hop_count, 2);
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a, b, c]);
    }

    #[test]
    fn find_all_paths_diamond() {
        // Diamond: A -> B1 -> C
        //          A -> B2 -> C
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b1 = engine.create_node("B1", HashMap::new()).unwrap();
        let b2 = engine.create_node("B2", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b1, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(a, b2, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b1, c, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b2, c, "E", HashMap::new(), true)
            .unwrap();

        let result = engine.find_all_paths(a, c, None).unwrap();
        assert_eq!(result.hop_count, 2);
        assert_eq!(result.paths.len(), 2);

        // Both paths should have length 3 nodes
        for path in &result.paths {
            assert_eq!(path.nodes.len(), 3);
            assert_eq!(path.nodes[0], a);
            assert_eq!(path.nodes[2], c);
        }

        // Check both middle nodes are present
        let middles: Vec<u64> = result.paths.iter().map(|p| p.nodes[1]).collect();
        assert!(middles.contains(&b1));
        assert!(middles.contains(&b2));
    }

    #[test]
    fn find_all_paths_same_node() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.find_all_paths(a, a, None).unwrap();
        assert_eq!(result.hop_count, 0);
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a]);
        assert!(result.paths[0].edges.is_empty());
    }

    #[test]
    fn find_all_paths_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        // No edge between them

        let result = engine.find_all_paths(a, b, None);
        assert!(matches!(result, Err(GraphError::PathNotFound)));
    }

    #[test]
    fn find_all_paths_node_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.find_all_paths(a, 999, None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));

        let result = engine.find_all_paths(999, a, None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn find_all_paths_three_parallel() {
        // A -> B1/B2/B3 -> C (3 paths)
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b1 = engine.create_node("B1", HashMap::new()).unwrap();
        let b2 = engine.create_node("B2", HashMap::new()).unwrap();
        let b3 = engine.create_node("B3", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b1, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(a, b2, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(a, b3, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b1, c, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b2, c, "E", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b3, c, "E", HashMap::new(), true)
            .unwrap();

        let result = engine.find_all_paths(a, c, None).unwrap();
        assert_eq!(result.hop_count, 2);
        assert_eq!(result.paths.len(), 3);
    }

    #[test]
    fn find_all_paths_prefers_shorter() {
        // A -> B -> C (short path, 2 hops)
        // A -> D -> E -> C (long path, 3 hops)
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();
        let e = engine.create_node("E", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        engine.create_edge(a, d, "E", HashMap::new(), true).unwrap();
        engine.create_edge(d, e, "E", HashMap::new(), true).unwrap();
        engine.create_edge(e, c, "E", HashMap::new(), true).unwrap();

        let result = engine.find_all_paths(a, c, None).unwrap();
        assert_eq!(result.hop_count, 2);
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a, b, c]);
    }

    #[test]
    fn find_all_paths_undirected() {
        // A -- B -- C (undirected)
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "E", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "E", HashMap::new(), false)
            .unwrap();

        // Forward
        let result = engine.find_all_paths(a, c, None).unwrap();
        assert_eq!(result.hop_count, 2);
        assert_eq!(result.paths.len(), 1);

        // Backward
        let result = engine.find_all_paths(c, a, None).unwrap();
        assert_eq!(result.hop_count, 2);
        assert_eq!(result.paths.len(), 1);
    }

    #[test]
    fn find_all_paths_with_cycle() {
        // A -> B -> C with A -> C direct
        //      ^-------|
        // The cycle shouldn't cause infinite loop
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        engine.create_edge(c, a, "E", HashMap::new(), true).unwrap(); // cycle
        engine.create_edge(a, c, "E", HashMap::new(), true).unwrap(); // direct

        let result = engine.find_all_paths(a, c, None).unwrap();
        // Shortest path is direct: A -> C (1 hop)
        assert_eq!(result.hop_count, 1);
        assert_eq!(result.paths.len(), 1);
    }

    #[test]
    fn find_all_paths_max_paths_limit() {
        // Create a graph with many paths
        // A -> B1..B10 -> C (10 paths, but limit to 3)
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        for i in 0..10 {
            let b = engine
                .create_node(&format!("B{i}"), HashMap::new())
                .unwrap();
            engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
            engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        }

        let config = AllPathsConfig {
            max_paths: 3,
            max_parents_per_node: 100,
        };
        let result = engine.find_all_paths(a, c, Some(config)).unwrap();
        assert_eq!(result.paths.len(), 3);
    }

    #[test]
    fn find_all_paths_max_parents_limit() {
        // A -> B1..B10 -> C, but limit parents per node to 2
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        for i in 0..10 {
            let b = engine
                .create_node(&format!("B{i}"), HashMap::new())
                .unwrap();
            engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
            engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        }

        let config = AllPathsConfig {
            max_paths: 1000,
            max_parents_per_node: 2,
        };
        let result = engine.find_all_paths(a, c, Some(config)).unwrap();
        // Only 2 parents tracked at C, so only 2 paths
        assert_eq!(result.paths.len(), 2);
    }

    // ==================== find_all_weighted_paths tests ====================

    #[test]
    fn find_all_weighted_paths_simple() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        engine.create_edge(a, b, "E", props.clone(), true).unwrap();
        engine.create_edge(b, c, "E", props, true).unwrap();

        let result = engine
            .find_all_weighted_paths(a, c, "weight", None)
            .unwrap();
        assert!((result.total_weight - 2.0).abs() < f64::EPSILON);
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a, b, c]);
    }

    #[test]
    fn find_all_weighted_paths_diamond_equal() {
        // Diamond with equal weights: 2 paths
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b1 = engine.create_node("B1", HashMap::new()).unwrap();
        let b2 = engine.create_node("B2", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        engine.create_edge(a, b1, "E", props.clone(), true).unwrap();
        engine.create_edge(a, b2, "E", props.clone(), true).unwrap();
        engine.create_edge(b1, c, "E", props.clone(), true).unwrap();
        engine.create_edge(b2, c, "E", props, true).unwrap();

        let result = engine
            .find_all_weighted_paths(a, c, "weight", None)
            .unwrap();
        assert!((result.total_weight - 2.0).abs() < f64::EPSILON);
        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn find_all_weighted_paths_one_lighter() {
        // Diamond with different weights: only 1 lighter path
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b1 = engine.create_node("B1", HashMap::new()).unwrap();
        let b2 = engine.create_node("B2", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        let mut light = HashMap::new();
        light.insert("weight".to_string(), PropertyValue::Float(1.0));
        let mut heavy = HashMap::new();
        heavy.insert("weight".to_string(), PropertyValue::Float(5.0));

        engine.create_edge(a, b1, "E", light.clone(), true).unwrap();
        engine.create_edge(b1, c, "E", light, true).unwrap();
        engine.create_edge(a, b2, "E", heavy.clone(), true).unwrap();
        engine.create_edge(b2, c, "E", heavy, true).unwrap();

        let result = engine
            .find_all_weighted_paths(a, c, "weight", None)
            .unwrap();
        assert!((result.total_weight - 2.0).abs() < f64::EPSILON);
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes[1], b1);
    }

    #[test]
    fn find_all_weighted_paths_epsilon() {
        // Test float precision handling
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b1 = engine.create_node("B1", HashMap::new()).unwrap();
        let b2 = engine.create_node("B2", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("weight".to_string(), PropertyValue::Float(1.0));
        let mut props2 = HashMap::new();
        // Very close but not exactly equal
        props2.insert("weight".to_string(), PropertyValue::Float(1.000_000_000_01));

        engine
            .create_edge(a, b1, "E", props1.clone(), true)
            .unwrap();
        engine.create_edge(b1, c, "E", props1, true).unwrap();
        engine
            .create_edge(a, b2, "E", props2.clone(), true)
            .unwrap();
        engine.create_edge(b2, c, "E", props2, true).unwrap();

        let result = engine
            .find_all_weighted_paths(a, c, "weight", None)
            .unwrap();
        // Both paths should be considered equal weight (within epsilon)
        assert_eq!(result.paths.len(), 2);
    }

    #[test]
    fn find_all_weighted_paths_negative_error() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(-1.0));
        engine.create_edge(a, b, "E", props, true).unwrap();

        let result = engine.find_all_weighted_paths(a, b, "weight", None);
        assert!(matches!(result, Err(GraphError::NegativeWeight { .. })));
    }

    #[test]
    fn find_all_weighted_paths_same_node() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine
            .find_all_weighted_paths(a, a, "weight", None)
            .unwrap();
        assert!((result.total_weight - 0.0).abs() < f64::EPSILON);
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a]);
    }

    #[test]
    fn find_all_weighted_paths_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        let result = engine.find_all_weighted_paths(a, b, "weight", None);
        assert!(matches!(result, Err(GraphError::PathNotFound)));
    }

    #[test]
    fn find_all_weighted_paths_node_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.find_all_weighted_paths(a, 999, "weight", None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));

        let result = engine.find_all_weighted_paths(999, a, "weight", None);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn find_all_weighted_paths_undirected() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        engine.create_edge(a, b, "E", props.clone(), false).unwrap();
        engine.create_edge(b, c, "E", props, false).unwrap();

        // Forward
        let result = engine
            .find_all_weighted_paths(a, c, "weight", None)
            .unwrap();
        assert!((result.total_weight - 2.0).abs() < f64::EPSILON);

        // Backward
        let result = engine
            .find_all_weighted_paths(c, a, "weight", None)
            .unwrap();
        assert!((result.total_weight - 2.0).abs() < f64::EPSILON);
    }

    // ==================== Struct tests ====================

    #[test]
    fn all_paths_equality() {
        let p1 = AllPaths {
            paths: vec![Path {
                nodes: vec![1, 2],
                edges: vec![10],
            }],
            hop_count: 1,
        };
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }

    #[test]
    fn all_weighted_paths_equality() {
        let p1 = AllWeightedPaths {
            paths: vec![WeightedPath {
                nodes: vec![1, 2],
                edges: vec![10],
                total_weight: 5.0,
            }],
            total_weight: 5.0,
        };
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }

    #[test]
    fn all_paths_config_default() {
        let config = AllPathsConfig::default();
        assert_eq!(config.max_paths, 1000);
        assert_eq!(config.max_parents_per_node, 100);
    }

    #[test]
    fn all_paths_complex_grid() {
        // 3x3 grid with multiple paths from top-left to bottom-right
        // Each step right or down is one hop
        let engine = GraphEngine::new();
        let mut nodes = [[0u64; 3]; 3];

        // Create nodes
        for i in 0..3 {
            for j in 0..3 {
                nodes[i][j] = engine
                    .create_node(&format!("N{i}{j}"), HashMap::new())
                    .unwrap();
            }
        }

        // Create horizontal edges
        for i in 0..3 {
            for j in 0..2 {
                engine
                    .create_edge(nodes[i][j], nodes[i][j + 1], "E", HashMap::new(), true)
                    .unwrap();
            }
        }

        // Create vertical edges
        for i in 0..2 {
            for j in 0..3 {
                engine
                    .create_edge(nodes[i][j], nodes[i + 1][j], "E", HashMap::new(), true)
                    .unwrap();
            }
        }

        let result = engine
            .find_all_paths(nodes[0][0], nodes[2][2], None)
            .unwrap();
        // Shortest path is 4 hops (2 right + 2 down in any order)
        assert_eq!(result.hop_count, 4);
        // Number of paths = C(4,2) = 6 (choose 2 positions for "right" out of 4 moves)
        assert_eq!(result.paths.len(), 6);
    }

    // ==================== Variable-length path tests ====================

    #[test]
    fn variable_paths_simple_chain() {
        // A -> B -> C -> D
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(c, d, "NEXT", HashMap::new(), true)
            .unwrap();

        let config = VariableLengthConfig::with_hops(1, 5);
        let result = engine.find_variable_paths(a, d, config).unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a, b, c, d]);
        assert_eq!(result.paths[0].edges.len(), 3);
        assert_eq!(result.stats.paths_found, 1);
        assert_eq!(result.stats.min_length, Some(3));
        assert_eq!(result.stats.max_length, Some(3));
        assert!(!result.stats.truncated);
    }

    #[test]
    fn variable_paths_exact_hops() {
        // A -> B -> C, looking for exactly 2 hops
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(2, 2);
        let result = engine.find_variable_paths(a, c, config).unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.stats.min_length, Some(2));
        assert_eq!(result.stats.max_length, Some(2));
    }

    #[test]
    fn variable_paths_hop_range() {
        // A -> B -> C, looking for 1-3 hops from A to C
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        // Direct edge from A to C
        engine.create_edge(a, c, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(1, 3);
        let result = engine.find_variable_paths(a, c, config).unwrap();

        // Should find both paths: A->C (1 hop) and A->B->C (2 hops)
        assert_eq!(result.paths.len(), 2);
        assert_eq!(result.stats.min_length, Some(1));
        assert_eq!(result.stats.max_length, Some(2));
    }

    #[test]
    fn variable_paths_no_path_in_range() {
        // A -> B -> C, looking for exactly 1 hop from A to C
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(1, 1);
        let result = engine.find_variable_paths(a, c, config).unwrap();

        assert!(result.is_empty());
        assert_eq!(result.stats.paths_found, 0);
    }

    #[test]
    fn variable_paths_same_node_zero_hops() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let config = VariableLengthConfig::with_hops(0, 0);
        let result = engine.find_variable_paths(a, a, config).unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![a]);
        assert!(result.paths[0].edges.is_empty());
        assert_eq!(result.stats.min_length, Some(0));
    }

    #[test]
    fn variable_paths_diamond() {
        // A -> B -> D
        //  \-> C -/
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(a, c, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, d, "E", HashMap::new(), true).unwrap();
        engine.create_edge(c, d, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(1, 3);
        let result = engine.find_variable_paths(a, d, config).unwrap();

        // Should find 2 paths: A->B->D and A->C->D
        assert_eq!(result.paths.len(), 2);
        assert_eq!(result.stats.min_length, Some(2));
        assert_eq!(result.stats.max_length, Some(2));
    }

    #[test]
    fn variable_paths_multiple_lengths() {
        // A -> B -> C -> D with direct A -> D edge
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        engine.create_edge(c, d, "E", HashMap::new(), true).unwrap();
        engine.create_edge(a, d, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(1, 5);
        let result = engine.find_variable_paths(a, d, config).unwrap();

        // Should find 2 paths: A->D (1 hop) and A->B->C->D (3 hops)
        assert_eq!(result.paths.len(), 2);
        assert_eq!(result.stats.min_length, Some(1));
        assert_eq!(result.stats.max_length, Some(3));

        let short_paths = result.paths_of_length(1);
        assert_eq!(short_paths.len(), 1);

        let long_paths = result.paths_of_length(3);
        assert_eq!(long_paths.len(), 1);
    }

    #[test]
    fn variable_paths_edge_type_filter() {
        // A -> B (KNOWS) -> C (KNOWS)
        // A -> B (WORKS_WITH) -> C
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(a, c, "WORKS_WITH", HashMap::new(), true)
            .unwrap();

        let config = VariableLengthConfig::with_hops(1, 3).edge_type("KNOWS");
        let result = engine.find_variable_paths(a, c, config).unwrap();

        // Should only find A->B->C via KNOWS edges
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].edges.len(), 2);
    }

    #[test]
    fn variable_paths_multiple_edge_types() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "WORKS_WITH", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(a, c, "LIVES_WITH", HashMap::new(), true)
            .unwrap();

        let config = VariableLengthConfig::with_hops(1, 3).edge_types(&["KNOWS", "WORKS_WITH"]);
        let result = engine.find_variable_paths(a, c, config).unwrap();

        // Should find A->B->C via KNOWS and WORKS_WITH, but not A->C via LIVES_WITH
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].edges.len(), 2);
    }

    #[test]
    fn variable_paths_direction_incoming() {
        // A -> B -> C, search from C to A with Incoming direction
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(1, 3).direction(Direction::Incoming);
        let result = engine.find_variable_paths(c, a, config).unwrap();

        // Should find path C->B->A following edges backwards
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].nodes, vec![c, b, a]);
    }

    #[test]
    fn variable_paths_max_paths_limit() {
        // Create a graph with many paths
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let z = engine.create_node("Z", HashMap::new()).unwrap();

        // Create 10 intermediate nodes, each connecting A to Z
        for i in 0..10 {
            let n = engine
                .create_node(&format!("N{i}"), HashMap::new())
                .unwrap();
            engine.create_edge(a, n, "E", HashMap::new(), true).unwrap();
            engine.create_edge(n, z, "E", HashMap::new(), true).unwrap();
        }

        let config = VariableLengthConfig::with_hops(1, 3).max_paths(5);
        let result = engine.find_variable_paths(a, z, config).unwrap();

        assert_eq!(result.paths.len(), 5);
        assert!(result.stats.truncated);
    }

    #[test]
    fn variable_paths_cycle_detection() {
        // A -> B -> C -> A (cycle), looking for paths from A to C
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        engine.create_edge(c, a, "E", HashMap::new(), true).unwrap();

        // Without allowing cycles
        let config = VariableLengthConfig::with_hops(1, 5).allow_cycles(false);
        let result = engine.find_variable_paths(a, c, config).unwrap();
        assert_eq!(result.paths.len(), 1); // Only A->B->C

        // With allowing cycles, we might find more paths via the cycle
        let config = VariableLengthConfig::with_hops(1, 5).allow_cycles(true);
        let result = engine.find_variable_paths(a, c, config).unwrap();
        // Could find A->B->C and A->B->C->A->B->C, etc.
        assert!(result.paths.len() >= 1);
    }

    #[test]
    fn variable_paths_node_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let config = VariableLengthConfig::default();

        let result = engine.find_variable_paths(a, 999, config.clone());
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));

        let result = engine.find_variable_paths(999, a, config);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn variable_paths_undirected() {
        // A -- B -- C (undirected edges)
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "E", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "E", HashMap::new(), false)
            .unwrap();

        // Can traverse forward
        let config = VariableLengthConfig::with_hops(1, 3);
        let result = engine.find_variable_paths(a, c, config).unwrap();
        assert_eq!(result.paths.len(), 1);

        // Can also traverse backward
        let config = VariableLengthConfig::with_hops(1, 3);
        let result = engine.find_variable_paths(c, a, config).unwrap();
        assert_eq!(result.paths.len(), 1);
    }

    #[test]
    fn variable_paths_stats_accuracy() {
        // A -> B -> C
        //  \-> D -/
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
        engine.create_edge(a, d, "E", HashMap::new(), true).unwrap();
        engine.create_edge(b, c, "E", HashMap::new(), true).unwrap();
        engine.create_edge(d, c, "E", HashMap::new(), true).unwrap();

        let config = VariableLengthConfig::with_hops(1, 3);
        let result = engine.find_variable_paths(a, c, config).unwrap();

        assert_eq!(result.stats.paths_found, 2);
        assert!(result.stats.nodes_explored > 0);
        assert!(result.stats.edges_traversed > 0);
        assert!(!result.stats.truncated);
    }

    #[test]
    fn variable_length_config_default() {
        let config = VariableLengthConfig::default();
        assert_eq!(config.min_hops, 1);
        assert_eq!(config.max_hops, 5);
        assert_eq!(config.direction, Direction::Outgoing);
        assert!(config.edge_types.is_none());
        assert_eq!(config.max_paths, 1000);
        assert!(!config.allow_cycles);
    }

    #[test]
    fn variable_length_config_safety_cap() {
        // max_hops is capped at 20
        let config = VariableLengthConfig::with_hops(1, 100);
        assert_eq!(config.max_hops, 20);
    }

    #[test]
    fn variable_length_paths_is_empty() {
        let empty = VariableLengthPaths {
            paths: vec![],
            stats: PathSearchStats::default(),
        };
        assert!(empty.is_empty());

        let non_empty = VariableLengthPaths {
            paths: vec![Path {
                nodes: vec![1],
                edges: vec![],
            }],
            stats: PathSearchStats::default(),
        };
        assert!(!non_empty.is_empty());
    }

    // Property filtering tests

    #[test]
    fn property_condition_eq_match() {
        let cond = PropertyCondition::new("age", CompareOp::Eq, PropertyValue::Int(30));
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: props,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node));
    }

    #[test]
    fn property_condition_eq_no_match() {
        let cond = PropertyCondition::new("age", CompareOp::Eq, PropertyValue::Int(30));
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(25));
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: props,
            created_at: None,
            updated_at: None,
        };
        assert!(!cond.matches_node(&node));
    }

    #[test]
    fn property_condition_ne_match() {
        let cond = PropertyCondition::new(
            "status",
            CompareOp::Ne,
            PropertyValue::String("inactive".to_string()),
        );
        let mut props = HashMap::new();
        props.insert(
            "status".to_string(),
            PropertyValue::String("active".to_string()),
        );
        let node = Node {
            id: 1,
            labels: vec!["User".to_string()],
            properties: props,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node));
    }

    #[test]
    fn property_condition_ne_missing_property() {
        let cond = PropertyCondition::new("missing", CompareOp::Ne, PropertyValue::Int(10));
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };
        // Missing property should return true for Ne
        assert!(cond.matches_node(&node));
    }

    #[test]
    fn property_condition_lt_int() {
        let cond = PropertyCondition::new("age", CompareOp::Lt, PropertyValue::Int(30));
        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(25));
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: props,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node));
    }

    #[test]
    fn property_condition_le_float() {
        let cond = PropertyCondition::new("weight", CompareOp::Le, PropertyValue::Float(5.0));
        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(5.0));
        let edge = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "CONNECTS".to_string(),
            properties: props,
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_edge(&edge));
    }

    #[test]
    fn property_condition_gt_string() {
        let cond = PropertyCondition::new(
            "name",
            CompareOp::Gt,
            PropertyValue::String("Alice".to_string()),
        );
        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Bob".to_string()));
        let node = Node {
            id: 1,
            labels: vec!["Person".to_string()],
            properties: props,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node));
    }

    #[test]
    fn property_condition_ge_comparison() {
        let cond = PropertyCondition::new("score", CompareOp::Ge, PropertyValue::Int(100));
        let mut props1 = HashMap::new();
        props1.insert("score".to_string(), PropertyValue::Int(100));
        let node1 = Node {
            id: 1,
            labels: vec![],
            properties: props1,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node1));

        let mut props2 = HashMap::new();
        props2.insert("score".to_string(), PropertyValue::Int(150));
        let node2 = Node {
            id: 2,
            labels: vec![],
            properties: props2,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node2));
    }

    #[test]
    fn traversal_filter_empty_matches_all() {
        let filter = TraversalFilter::new();
        assert!(filter.is_empty());

        let node = Node {
            id: 1,
            labels: vec!["Any".to_string()],
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        };
        assert!(filter.matches_node(&node));

        let edge = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "ANY".to_string(),
            properties: HashMap::new(),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!(filter.matches_edge(&edge));
    }

    #[test]
    fn traversal_filter_single_node_condition() {
        let filter = TraversalFilter::new().node_eq("active", PropertyValue::Bool(true));

        let mut props_active = HashMap::new();
        props_active.insert("active".to_string(), PropertyValue::Bool(true));
        let active_node = Node {
            id: 1,
            labels: vec![],
            properties: props_active,
            created_at: None,
            updated_at: None,
        };
        assert!(filter.matches_node(&active_node));

        let mut props_inactive = HashMap::new();
        props_inactive.insert("active".to_string(), PropertyValue::Bool(false));
        let inactive_node = Node {
            id: 2,
            labels: vec![],
            properties: props_inactive,
            created_at: None,
            updated_at: None,
        };
        assert!(!filter.matches_node(&inactive_node));
    }

    #[test]
    fn traversal_filter_multiple_conditions_and() {
        let filter = TraversalFilter::new()
            .node_where("age", CompareOp::Ge, PropertyValue::Int(18))
            .node_where("age", CompareOp::Lt, PropertyValue::Int(65));

        let mut props = HashMap::new();
        props.insert("age".to_string(), PropertyValue::Int(30));
        let node = Node {
            id: 1,
            labels: vec![],
            properties: props,
            created_at: None,
            updated_at: None,
        };
        assert!(filter.matches_node(&node));

        let mut props_too_young = HashMap::new();
        props_too_young.insert("age".to_string(), PropertyValue::Int(16));
        let too_young = Node {
            id: 2,
            labels: vec![],
            properties: props_too_young,
            created_at: None,
            updated_at: None,
        };
        assert!(!filter.matches_node(&too_young));
    }

    #[test]
    fn traversal_filter_edge_condition() {
        let filter =
            TraversalFilter::new().edge_where("weight", CompareOp::Lt, PropertyValue::Float(10.0));

        let mut light_props = HashMap::new();
        light_props.insert("weight".to_string(), PropertyValue::Float(5.0));
        let light_edge = Edge {
            id: 1,
            from: 1,
            to: 2,
            edge_type: "CONNECTS".to_string(),
            properties: light_props,
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!(filter.matches_edge(&light_edge));

        let mut heavy_props = HashMap::new();
        heavy_props.insert("weight".to_string(), PropertyValue::Float(15.0));
        let heavy_edge = Edge {
            id: 2,
            from: 2,
            to: 3,
            edge_type: "CONNECTS".to_string(),
            properties: heavy_props,
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!(!filter.matches_edge(&heavy_edge));
    }

    #[test]
    fn traversal_filter_builder_pattern() {
        let filter = TraversalFilter::new()
            .node_eq("type", PropertyValue::String("user".to_string()))
            .node_ne("banned", PropertyValue::Bool(true))
            .edge_eq("active", PropertyValue::Bool(true));

        assert!(!filter.is_empty());
    }

    #[test]
    fn traverse_filtered_by_node_property() {
        let engine = GraphEngine::new();

        // Create nodes with different ages
        let mut props30 = HashMap::new();
        props30.insert("age".to_string(), PropertyValue::Int(30));
        let n1 = engine.create_node("Person", props30).unwrap();

        let mut props25 = HashMap::new();
        props25.insert("age".to_string(), PropertyValue::Int(25));
        let n2 = engine.create_node("Person", props25).unwrap();

        let mut props35 = HashMap::new();
        props35.insert("age".to_string(), PropertyValue::Int(35));
        let n3 = engine.create_node("Person", props35).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "KNOWS", HashMap::new(), true)
            .unwrap();

        // Filter for age > 28
        let filter =
            TraversalFilter::new().node_where("age", CompareOp::Gt, PropertyValue::Int(28));

        let result = engine
            .traverse(n1, Direction::Outgoing, 2, None, Some(&filter))
            .unwrap();

        // Should include n1 (start node always included) and n3 (age 35 > 28)
        // n2 (age 25) should be excluded from results but still traversed through
        assert!(result.iter().any(|n| n.id == n1));
        assert!(result.iter().any(|n| n.id == n3));
        assert!(!result.iter().any(|n| n.id == n2)); // n2 filtered out
    }

    #[test]
    fn traverse_filtered_by_edge_property() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();
        let n3 = engine.create_node("Node", HashMap::new()).unwrap();

        let mut light = HashMap::new();
        light.insert("weight".to_string(), PropertyValue::Float(2.0));
        engine.create_edge(n1, n2, "EDGE", light, true).unwrap();

        let mut heavy = HashMap::new();
        heavy.insert("weight".to_string(), PropertyValue::Float(10.0));
        engine.create_edge(n1, n3, "EDGE", heavy, true).unwrap();

        // Filter for weight < 5.0
        let filter =
            TraversalFilter::new().edge_where("weight", CompareOp::Lt, PropertyValue::Float(5.0));

        let result = engine
            .traverse(n1, Direction::Outgoing, 1, None, Some(&filter))
            .unwrap();

        // Should include n1 and n2, but not n3 (edge too heavy)
        assert!(result.iter().any(|n| n.id == n1));
        assert!(result.iter().any(|n| n.id == n2));
        assert!(!result.iter().any(|n| n.id == n3));
    }

    #[test]
    fn traverse_filtered_combined() {
        let engine = GraphEngine::new();

        let mut props30 = HashMap::new();
        props30.insert("age".to_string(), PropertyValue::Int(30));
        let n1 = engine.create_node("Person", props30).unwrap();

        let mut props40 = HashMap::new();
        props40.insert("age".to_string(), PropertyValue::Int(40));
        let n2 = engine.create_node("Person", props40).unwrap();

        let mut props20 = HashMap::new();
        props20.insert("age".to_string(), PropertyValue::Int(20));
        let n3 = engine.create_node("Person", props20).unwrap();

        let mut close = HashMap::new();
        close.insert("distance".to_string(), PropertyValue::Float(1.0));
        engine.create_edge(n1, n2, "KNOWS", close, true).unwrap();

        let mut far = HashMap::new();
        far.insert("distance".to_string(), PropertyValue::Float(100.0));
        engine.create_edge(n1, n3, "KNOWS", far, true).unwrap();

        // Filter for age > 25 AND distance < 50
        let filter = TraversalFilter::new()
            .node_where("age", CompareOp::Gt, PropertyValue::Int(25))
            .edge_where("distance", CompareOp::Lt, PropertyValue::Float(50.0));

        let result = engine
            .traverse(n1, Direction::Outgoing, 1, None, Some(&filter))
            .unwrap();

        // n1 is start (always included), n2 passes both filters
        // n3 fails node filter (age 20 < 25) AND edge is filtered out
        assert!(result.iter().any(|n| n.id == n1));
        assert!(result.iter().any(|n| n.id == n2));
        assert!(!result.iter().any(|n| n.id == n3));
    }

    #[test]
    fn traverse_filtered_no_matches() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert("level".to_string(), PropertyValue::Int(5));
        let n1 = engine.create_node("Node", props.clone()).unwrap();
        let n2 = engine.create_node("Node", props).unwrap();

        engine
            .create_edge(n1, n2, "EDGE", HashMap::new(), true)
            .unwrap();

        // Filter that matches nothing (level > 100)
        let filter =
            TraversalFilter::new().node_where("level", CompareOp::Gt, PropertyValue::Int(100));

        let result = engine
            .traverse(n1, Direction::Outgoing, 1, None, Some(&filter))
            .unwrap();

        // Only start node included
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, n1);
    }

    #[test]
    fn traverse_filtered_with_edge_type() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();
        let n3 = engine.create_node("Node", HashMap::new()).unwrap();

        let mut active = HashMap::new();
        active.insert("active".to_string(), PropertyValue::Bool(true));
        engine
            .create_edge(n1, n2, "KNOWS", active.clone(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "WORKS_WITH", active, true)
            .unwrap();

        let filter = TraversalFilter::new().edge_eq("active", PropertyValue::Bool(true));

        // Only traverse KNOWS edges with active=true
        let result = engine
            .traverse(n1, Direction::Outgoing, 1, Some("KNOWS"), Some(&filter))
            .unwrap();

        assert!(result.iter().any(|n| n.id == n1));
        assert!(result.iter().any(|n| n.id == n2));
        assert!(!result.iter().any(|n| n.id == n3)); // filtered by edge type
    }

    #[test]
    fn find_path_filtered_by_edge() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();
        let n3 = engine.create_node("Node", HashMap::new()).unwrap();
        let n4 = engine.create_node("Node", HashMap::new()).unwrap();

        // Create two paths: n1 -> n2 -> n4 (blocked) and n1 -> n3 -> n4 (open)
        let mut blocked = HashMap::new();
        blocked.insert("blocked".to_string(), PropertyValue::Bool(true));
        engine.create_edge(n1, n2, "PATH", blocked, true).unwrap();
        engine
            .create_edge(n2, n4, "PATH", HashMap::new(), true)
            .unwrap();

        let mut open = HashMap::new();
        open.insert("blocked".to_string(), PropertyValue::Bool(false));
        engine
            .create_edge(n1, n3, "PATH", open.clone(), true)
            .unwrap();
        engine.create_edge(n3, n4, "PATH", open, true).unwrap();

        // Filter out blocked edges
        let filter = TraversalFilter::new().edge_ne("blocked", PropertyValue::Bool(true));

        let path = engine.find_path(n1, n4, Some(&filter)).unwrap();

        // Path should go through n3, not n2
        assert!(path.nodes.contains(&n3));
        assert!(!path.nodes.contains(&n2));
    }

    #[test]
    fn find_path_filtered_no_path() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();

        let mut blocked = HashMap::new();
        blocked.insert("passable".to_string(), PropertyValue::Bool(false));
        engine.create_edge(n1, n2, "PATH", blocked, true).unwrap();

        // Filter requires passable=true
        let filter = TraversalFilter::new().edge_eq("passable", PropertyValue::Bool(true));

        let result = engine.find_path(n1, n2, Some(&filter));
        assert!(matches!(result, Err(GraphError::PathNotFound)));
    }

    #[test]
    fn find_path_filtered_alternate_route() {
        let engine = GraphEngine::new();

        let start = engine.create_node("Node", HashMap::new()).unwrap();
        let end = engine.create_node("Node", HashMap::new()).unwrap();
        let mid1 = engine.create_node("Node", HashMap::new()).unwrap();
        let mid2 = engine.create_node("Node", HashMap::new()).unwrap();

        // Short path through mid1 with high cost
        let mut high_cost = HashMap::new();
        high_cost.insert("cost".to_string(), PropertyValue::Int(100));
        engine
            .create_edge(start, mid1, "PATH", high_cost.clone(), true)
            .unwrap();
        engine
            .create_edge(mid1, end, "PATH", high_cost, true)
            .unwrap();

        // Longer path through mid2 with low cost
        let mut low_cost = HashMap::new();
        low_cost.insert("cost".to_string(), PropertyValue::Int(5));
        engine
            .create_edge(start, mid2, "PATH", low_cost.clone(), true)
            .unwrap();
        engine
            .create_edge(mid2, end, "PATH", low_cost, true)
            .unwrap();

        // Filter for cost < 50
        let filter =
            TraversalFilter::new().edge_where("cost", CompareOp::Lt, PropertyValue::Int(50));

        let path = engine.find_path(start, end, Some(&filter)).unwrap();

        // Path should go through mid2 (low cost), not mid1 (high cost blocked)
        assert!(path.nodes.contains(&mid2));
        assert!(!path.nodes.contains(&mid1));
    }

    #[test]
    fn variable_paths_with_filter() {
        let engine = GraphEngine::new();

        let mut active = HashMap::new();
        active.insert("active".to_string(), PropertyValue::Bool(true));
        let n1 = engine.create_node("Node", active.clone()).unwrap();
        let n2 = engine.create_node("Node", active.clone()).unwrap();

        let mut inactive = HashMap::new();
        inactive.insert("active".to_string(), PropertyValue::Bool(false));
        let n3 = engine.create_node("Node", inactive).unwrap();

        let n4 = engine.create_node("Node", active).unwrap();

        // n1 -> n2 -> n4 (all active)
        // n1 -> n3 -> n4 (n3 is inactive)
        engine
            .create_edge(n1, n2, "PATH", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n4, "PATH", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "PATH", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n4, "PATH", HashMap::new(), true)
            .unwrap();

        let filter = TraversalFilter::new().node_eq("active", PropertyValue::Bool(true));

        let config = VariableLengthConfig::with_hops(1, 3).with_filter(filter);

        let result = engine.find_variable_paths(n1, n4, config).unwrap();

        // Should only find path through n2 (active), not n3 (inactive)
        assert!(!result.paths.is_empty());
        for path in &result.paths {
            assert!(!path.nodes.contains(&n3));
        }
    }

    #[test]
    fn variable_paths_filter_range_ops() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();
        let n3 = engine.create_node("Node", HashMap::new()).unwrap();

        let mut light = HashMap::new();
        light.insert("weight".to_string(), PropertyValue::Float(1.0));
        engine
            .create_edge(n1, n2, "E", light.clone(), true)
            .unwrap();
        engine.create_edge(n2, n3, "E", light, true).unwrap();

        let mut heavy = HashMap::new();
        heavy.insert("weight".to_string(), PropertyValue::Float(100.0));
        engine.create_edge(n1, n3, "E", heavy, true).unwrap();

        let filter =
            TraversalFilter::new().edge_where("weight", CompareOp::Le, PropertyValue::Float(10.0));

        let config = VariableLengthConfig::with_hops(1, 2).with_filter(filter);

        let result = engine.find_variable_paths(n1, n3, config).unwrap();

        // Should find path n1 -> n2 -> n3, but not direct n1 -> n3 (too heavy)
        assert!(!result.paths.is_empty());
        for path in &result.paths {
            // All paths should be length 2 (through n2)
            assert_eq!(path.nodes.len(), 3);
            assert!(path.nodes.contains(&n2));
        }
    }

    #[test]
    fn filter_null_property_value() {
        let cond = PropertyCondition::new("field", CompareOp::Eq, PropertyValue::Null);

        let mut props_null = HashMap::new();
        props_null.insert("field".to_string(), PropertyValue::Null);
        let node_null = Node {
            id: 1,
            labels: vec![],
            properties: props_null,
            created_at: None,
            updated_at: None,
        };
        assert!(cond.matches_node(&node_null));

        let mut props_int = HashMap::new();
        props_int.insert("field".to_string(), PropertyValue::Int(5));
        let node_int = Node {
            id: 2,
            labels: vec![],
            properties: props_int,
            created_at: None,
            updated_at: None,
        };
        assert!(!cond.matches_node(&node_int));
    }

    #[test]
    fn filter_empty_is_noop() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "EDGE", HashMap::new(), true)
            .unwrap();

        let empty_filter = TraversalFilter::new();

        // Traverse with empty filter should behave same as no filter
        let with_filter = engine
            .traverse(n1, Direction::Outgoing, 1, None, Some(&empty_filter))
            .unwrap();
        let without_filter = engine
            .traverse(n1, Direction::Outgoing, 1, None, None)
            .unwrap();

        assert_eq!(with_filter.len(), without_filter.len());
    }

    #[test]
    fn neighbors_with_filter() {
        let engine = GraphEngine::new();

        let mut props30 = HashMap::new();
        props30.insert("age".to_string(), PropertyValue::Int(30));
        let n1 = engine.create_node("Person", props30).unwrap();

        let mut props25 = HashMap::new();
        props25.insert("age".to_string(), PropertyValue::Int(25));
        let n2 = engine.create_node("Person", props25).unwrap();

        let mut props40 = HashMap::new();
        props40.insert("age".to_string(), PropertyValue::Int(40));
        let n3 = engine.create_node("Person", props40).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "KNOWS", HashMap::new(), true)
            .unwrap();

        let filter =
            TraversalFilter::new().node_where("age", CompareOp::Ge, PropertyValue::Int(30));

        let neighbors = engine
            .neighbors(n1, None, Direction::Outgoing, Some(&filter))
            .unwrap();

        // Should only include n3 (age 40), not n2 (age 25)
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, n3);
    }

    #[test]
    fn variable_length_config_with_filter() {
        let filter = TraversalFilter::new().node_eq("active", PropertyValue::Bool(true));

        let config = VariableLengthConfig::with_hops(1, 5)
            .direction(Direction::Both)
            .edge_type("KNOWS")
            .with_filter(filter);

        assert!(config.filter.is_some());
        assert!(!config.filter.as_ref().unwrap().is_empty());
    }

    // Degree calculation tests

    #[test]
    fn out_degree_no_edges() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();

        assert_eq!(engine.out_degree(n1).unwrap(), 0);
    }

    #[test]
    fn in_degree_no_edges() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();

        assert_eq!(engine.in_degree(n1).unwrap(), 0);
    }

    #[test]
    fn degree_no_edges() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();

        assert_eq!(engine.degree(n1).unwrap(), 0);
    }

    #[test]
    fn out_degree_with_edges() {
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

        assert_eq!(engine.out_degree(n1).unwrap(), 2);
        assert_eq!(engine.out_degree(n2).unwrap(), 0);
        assert_eq!(engine.out_degree(n3).unwrap(), 0);
    }

    #[test]
    fn in_degree_with_edges() {
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

        assert_eq!(engine.in_degree(n1).unwrap(), 0);
        assert_eq!(engine.in_degree(n2).unwrap(), 1);
        assert_eq!(engine.in_degree(n3).unwrap(), 1);
    }

    #[test]
    fn degree_with_edges() {
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
        engine
            .create_edge(n3, n1, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        // n1: 2 outgoing + 1 incoming = 3
        assert_eq!(engine.degree(n1).unwrap(), 3);
        // n2: 0 outgoing + 1 incoming = 1
        assert_eq!(engine.degree(n2).unwrap(), 1);
        // n3: 1 outgoing + 1 incoming = 2
        assert_eq!(engine.degree(n3).unwrap(), 2);
    }

    #[test]
    fn out_degree_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.out_degree(999);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn in_degree_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.in_degree(999);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn degree_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.degree(999);
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn out_degree_by_type_matches() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n3, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        assert_eq!(engine.out_degree_by_type(n1, "KNOWS").unwrap(), 1);
        assert_eq!(engine.out_degree_by_type(n1, "FOLLOWS").unwrap(), 1);
    }

    #[test]
    fn out_degree_by_type_no_match() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        assert_eq!(engine.out_degree_by_type(n1, "FOLLOWS").unwrap(), 0);
    }

    #[test]
    fn in_degree_by_type_matches() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .create_edge(n2, n1, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n1, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        assert_eq!(engine.in_degree_by_type(n1, "KNOWS").unwrap(), 1);
        assert_eq!(engine.in_degree_by_type(n1, "FOLLOWS").unwrap(), 1);
    }

    #[test]
    fn in_degree_by_type_no_match() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        engine
            .create_edge(n2, n1, "KNOWS", HashMap::new(), true)
            .unwrap();

        assert_eq!(engine.in_degree_by_type(n1, "FOLLOWS").unwrap(), 0);
    }

    #[test]
    fn degree_by_type_combined() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Person", HashMap::new()).unwrap();

        // n1 -> n2 (KNOWS)
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        // n3 -> n1 (KNOWS)
        engine
            .create_edge(n3, n1, "KNOWS", HashMap::new(), true)
            .unwrap();
        // n1 -> n3 (FOLLOWS)
        engine
            .create_edge(n1, n3, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        // n1 has 1 outgoing KNOWS + 1 incoming KNOWS = 2
        assert_eq!(engine.degree_by_type(n1, "KNOWS").unwrap(), 2);
        // n1 has 1 outgoing FOLLOWS + 0 incoming FOLLOWS = 1
        assert_eq!(engine.degree_by_type(n1, "FOLLOWS").unwrap(), 1);
    }

    #[test]
    fn degree_undirected_edge() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        // Undirected edge: stored in both outgoing and incoming lists for both nodes
        engine
            .create_edge(n1, n2, "FRIENDS", HashMap::new(), false)
            .unwrap();

        // Each node sees the edge in both outgoing and incoming
        assert_eq!(engine.out_degree(n1).unwrap(), 1);
        assert_eq!(engine.in_degree(n1).unwrap(), 1);
        assert_eq!(engine.degree(n1).unwrap(), 2);

        assert_eq!(engine.out_degree(n2).unwrap(), 1);
        assert_eq!(engine.in_degree(n2).unwrap(), 1);
        assert_eq!(engine.degree(n2).unwrap(), 2);
    }

    #[test]
    fn degree_hub_node() {
        let engine = GraphEngine::new();
        let hub = engine.create_node("Hub", HashMap::new()).unwrap();

        // Create 100 nodes connected to the hub
        for _ in 0..100 {
            let node = engine.create_node("Spoke", HashMap::new()).unwrap();
            engine
                .create_edge(hub, node, "CONNECTS", HashMap::new(), true)
                .unwrap();
        }

        assert_eq!(engine.out_degree(hub).unwrap(), 100);
        assert_eq!(engine.in_degree(hub).unwrap(), 0);
        assert_eq!(engine.degree(hub).unwrap(), 100);
    }

    #[test]
    fn degree_self_loop() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();

        // Self-loop: same node as source and target
        engine
            .create_edge(n1, n1, "SELF", HashMap::new(), true)
            .unwrap();

        // Self-loop appears in both outgoing and incoming lists
        assert_eq!(engine.out_degree(n1).unwrap(), 1);
        assert_eq!(engine.in_degree(n1).unwrap(), 1);
        assert_eq!(engine.degree(n1).unwrap(), 2);
    }

    #[test]
    fn degree_by_type_mixed_types() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        // Create multiple edges of different types
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n2, "FOLLOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n1, "LIKES", HashMap::new(), true)
            .unwrap();

        assert_eq!(engine.out_degree_by_type(n1, "KNOWS").unwrap(), 2);
        assert_eq!(engine.out_degree_by_type(n1, "FOLLOWS").unwrap(), 1);
        assert_eq!(engine.out_degree_by_type(n1, "LIKES").unwrap(), 0);

        assert_eq!(engine.in_degree_by_type(n1, "KNOWS").unwrap(), 0);
        assert_eq!(engine.in_degree_by_type(n1, "LIKES").unwrap(), 1);

        assert_eq!(engine.degree_by_type(n1, "KNOWS").unwrap(), 2);
        assert_eq!(engine.degree_by_type(n1, "LIKES").unwrap(), 1);

        // Total degree for n1: 3 outgoing + 1 incoming = 4
        assert_eq!(engine.degree(n1).unwrap(), 4);
    }

    #[test]
    fn all_nodes_empty_graph() {
        let engine = GraphEngine::new();
        let nodes = engine.all_nodes();
        assert!(nodes.is_empty());
    }

    #[test]
    fn all_edges_empty_graph() {
        let engine = GraphEngine::new();
        let edges = engine.all_edges();
        assert!(edges.is_empty());
    }

    #[test]
    fn all_nodes_returns_all() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Company", HashMap::new()).unwrap();

        let nodes = engine.all_nodes();
        assert_eq!(nodes.len(), 3);

        let ids: Vec<u64> = nodes.iter().map(|n| n.id).collect();
        assert!(ids.contains(&n1));
        assert!(ids.contains(&n2));
        assert!(ids.contains(&n3));
    }

    #[test]
    fn all_edges_returns_all() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let n3 = engine.create_node("Person", HashMap::new()).unwrap();

        let e1 = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        let e2 = engine
            .create_edge(n2, n3, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        let edges = engine.all_edges();
        assert_eq!(edges.len(), 2);

        let ids: Vec<u64> = edges.iter().map(|e| e.id).collect();
        assert!(ids.contains(&e1));
        assert!(ids.contains(&e2));
    }

    #[test]
    fn all_nodes_sorted_by_id() {
        let engine = GraphEngine::new();

        // Create nodes in various order
        let _n1 = engine.create_node("A", HashMap::new()).unwrap();
        let _n2 = engine.create_node("B", HashMap::new()).unwrap();
        let _n3 = engine.create_node("C", HashMap::new()).unwrap();

        let nodes = engine.all_nodes();
        assert_eq!(nodes.len(), 3);

        // Verify sorted by ID
        for i in 1..nodes.len() {
            assert!(nodes[i - 1].id < nodes[i].id);
        }
    }

    #[test]
    fn all_edges_sorted_by_id() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        let _e1 = engine
            .create_edge(n1, n2, "R1", HashMap::new(), true)
            .unwrap();
        let _e2 = engine
            .create_edge(n2, n3, "R2", HashMap::new(), true)
            .unwrap();
        let _e3 = engine
            .create_edge(n1, n3, "R3", HashMap::new(), true)
            .unwrap();

        let edges = engine.all_edges();
        assert_eq!(edges.len(), 3);

        // Verify sorted by ID
        for i in 1..edges.len() {
            assert!(edges[i - 1].id < edges[i].id);
        }
    }

    #[test]
    fn all_nodes_after_deletion() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine.delete_node(n2).unwrap();

        let nodes = engine.all_nodes();
        assert_eq!(nodes.len(), 2);

        let ids: Vec<u64> = nodes.iter().map(|n| n.id).collect();
        assert!(ids.contains(&n1));
        assert!(!ids.contains(&n2));
        assert!(ids.contains(&n3));
    }

    #[test]
    fn all_edges_after_deletion() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        let e1 = engine
            .create_edge(n1, n2, "R1", HashMap::new(), true)
            .unwrap();
        let e2 = engine
            .create_edge(n2, n3, "R2", HashMap::new(), true)
            .unwrap();
        let e3 = engine
            .create_edge(n1, n3, "R3", HashMap::new(), true)
            .unwrap();

        engine.delete_edge(e2).unwrap();

        let edges = engine.all_edges();
        assert_eq!(edges.len(), 2);

        let ids: Vec<u64> = edges.iter().map(|e| e.id).collect();
        assert!(ids.contains(&e1));
        assert!(!ids.contains(&e2));
        assert!(ids.contains(&e3));
    }

    #[test]
    fn all_nodes_includes_properties() {
        let engine = GraphEngine::new();

        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        props.insert("age".to_string(), PropertyValue::Int(30));

        let n1 = engine.create_node("Person", props).unwrap();

        let nodes = engine.all_nodes();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].id, n1);
        assert!(nodes[0].has_label("Person"));
        assert_eq!(
            nodes[0].properties.get("name"),
            Some(&PropertyValue::String("Alice".to_string()))
        );
        assert_eq!(
            nodes[0].properties.get("age"),
            Some(&PropertyValue::Int(30))
        );
    }

    #[test]
    fn all_edges_includes_properties() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("since".to_string(), PropertyValue::Int(2020));
        props.insert(
            "status".to_string(),
            PropertyValue::String("active".to_string()),
        );

        let e1 = engine.create_edge(n1, n2, "KNOWS", props, true).unwrap();

        let edges = engine.all_edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].id, e1);
        assert_eq!(edges[0].edge_type, "KNOWS");
        assert_eq!(edges[0].from, n1);
        assert_eq!(edges[0].to, n2);
        assert_eq!(
            edges[0].properties.get("since"),
            Some(&PropertyValue::Int(2020))
        );
        assert_eq!(
            edges[0].properties.get("status"),
            Some(&PropertyValue::String("active".to_string()))
        );
    }

    #[test]
    fn all_edges_includes_directed_and_undirected() {
        let engine = GraphEngine::new();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Create directed edge
        let e1 = engine
            .create_edge(n1, n2, "DIRECTED", HashMap::new(), true)
            .unwrap();

        // Create undirected edge
        let e2 = engine
            .create_edge(n1, n2, "UNDIRECTED", HashMap::new(), false)
            .unwrap();

        let edges = engine.all_edges();
        assert_eq!(edges.len(), 2);

        let directed = edges.iter().find(|e| e.id == e1).unwrap();
        assert!(directed.directed);

        let undirected = edges.iter().find(|e| e.id == e2).unwrap();
        assert!(!undirected.directed);
    }

    #[test]
    fn all_nodes_large_graph() {
        let engine = GraphEngine::new();

        let count = 1000;
        let mut created_ids = Vec::with_capacity(count);

        for i in 0..count {
            let mut props = HashMap::new();
            props.insert("index".to_string(), PropertyValue::Int(i as i64));
            let id = engine.create_node("Node", props).unwrap();
            created_ids.push(id);
        }

        let nodes = engine.all_nodes();
        assert_eq!(nodes.len(), count);

        // Verify all created nodes are present
        let node_ids: std::collections::HashSet<u64> = nodes.iter().map(|n| n.id).collect();
        for id in &created_ids {
            assert!(node_ids.contains(id));
        }

        // Verify sorted
        for i in 1..nodes.len() {
            assert!(nodes[i - 1].id < nodes[i].id);
        }
    }

    // ========== Timestamp Tests ==========

    #[test]
    fn node_created_at_is_set() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();
        let node = engine.get_node(id).unwrap();
        assert!(node.created_at.is_some());
    }

    #[test]
    fn edge_created_at_is_set() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let eid = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        let edge = engine.get_edge(eid).unwrap();
        assert!(edge.created_at.is_some());
    }

    #[test]
    fn node_created_at_is_recent() {
        let before = current_timestamp_millis();
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();
        let after = current_timestamp_millis();
        let node = engine.get_node(id).unwrap();
        let ts = node.created_at.unwrap();
        assert!(ts >= before && ts <= after);
    }

    #[test]
    fn edge_created_at_is_recent() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let before = current_timestamp_millis();
        let eid = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        let after = current_timestamp_millis();
        let edge = engine.get_edge(eid).unwrap();
        let ts = edge.created_at.unwrap();
        assert!(ts >= before && ts <= after);
    }

    #[test]
    fn node_update_sets_updated_at() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();
        let node_before = engine.get_node(id).unwrap();
        assert!(node_before.updated_at.is_none());

        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Alice".into()));
        engine.update_node(id, None, props).unwrap();

        let node_after = engine.get_node(id).unwrap();
        assert!(node_after.updated_at.is_some());
    }

    #[test]
    fn edge_update_sets_updated_at() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let eid = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        let edge_before = engine.get_edge(eid).unwrap();
        assert!(edge_before.updated_at.is_none());

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Int(42));
        engine.update_edge(eid, props).unwrap();

        let edge_after = engine.get_edge(eid).unwrap();
        assert!(edge_after.updated_at.is_some());
    }

    #[test]
    fn node_add_label_sets_updated_at() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();
        let node_before = engine.get_node(id).unwrap();
        assert!(node_before.updated_at.is_none());

        engine.add_label(id, "Employee").unwrap();

        let node_after = engine.get_node(id).unwrap();
        assert!(node_after.updated_at.is_some());
    }

    #[test]
    fn node_remove_label_sets_updated_at() {
        let engine = GraphEngine::new();
        let id = engine
            .create_node_with_labels(vec!["Person".into(), "Employee".into()], HashMap::new())
            .unwrap();
        let node_before = engine.get_node(id).unwrap();
        assert!(node_before.updated_at.is_none());

        engine.remove_label(id, "Employee").unwrap();

        let node_after = engine.get_node(id).unwrap();
        assert!(node_after.updated_at.is_some());
    }

    #[test]
    fn updated_at_greater_than_created_at() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Bob".into()));
        engine.update_node(id, None, props).unwrap();

        let node = engine.get_node(id).unwrap();
        assert!(node.updated_at.unwrap() > node.created_at.unwrap());
    }

    #[test]
    fn node_last_modified_prefers_updated() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String("Alice".into()));
        engine.update_node(id, None, props).unwrap();

        let node = engine.get_node(id).unwrap();
        assert_eq!(node.last_modified_millis(), node.updated_at);
    }

    #[test]
    fn edge_last_modified_prefers_updated() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let eid = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Int(10));
        engine.update_edge(eid, props).unwrap();

        let edge = engine.get_edge(eid).unwrap();
        assert_eq!(edge.last_modified_millis(), edge.updated_at);
    }

    #[test]
    fn last_modified_returns_created_when_no_update() {
        let engine = GraphEngine::new();
        let id = engine.create_node("Person", HashMap::new()).unwrap();
        let node = engine.get_node(id).unwrap();
        assert_eq!(node.last_modified_millis(), node.created_at);
    }

    #[test]
    fn legacy_node_without_timestamps_returns_none() {
        // Simulate a legacy node by directly putting data without timestamps
        let store = TensorStore::new();
        let engine = GraphEngine::with_store(store);

        // Manually create a node without timestamps (simulating legacy data)
        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(999)));
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("node".into())),
        );
        tensor.set("_labels", TensorValue::Pointers(vec!["Legacy".into()]));
        engine.store.put("node:999".to_string(), tensor).unwrap();

        let node = engine.get_node(999).unwrap();
        assert!(node.created_at.is_none());
        assert!(node.updated_at.is_none());
    }

    #[test]
    fn legacy_edge_without_timestamps_returns_none() {
        let store = TensorStore::new();
        let engine = GraphEngine::with_store(store);

        // Create nodes first
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        // Manually create an edge without timestamps (simulating legacy data)
        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(888)));
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("edge".into())),
        );
        tensor.set("_from", TensorValue::Scalar(ScalarValue::Int(n1 as i64)));
        tensor.set("_to", TensorValue::Scalar(ScalarValue::Int(n2 as i64)));
        tensor.set(
            "_edge_type",
            TensorValue::Scalar(ScalarValue::String("LEGACY".into())),
        );
        tensor.set("_directed", TensorValue::Scalar(ScalarValue::Bool(true)));
        engine.store.put("edge:888".to_string(), tensor).unwrap();

        let edge = engine.get_edge(888).unwrap();
        assert!(edge.created_at.is_none());
        assert!(edge.updated_at.is_none());
    }

    #[test]
    fn concurrent_creates_have_timestamps() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(GraphEngine::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let eng = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                let id = eng.create_node("Concurrent", HashMap::new()).unwrap();
                let node = eng.get_node(id).unwrap();
                assert!(node.created_at.is_some());
                id
            }));
        }

        let ids: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(ids.len(), 10);

        for id in ids {
            let node = engine.get_node(id).unwrap();
            assert!(node.created_at.is_some());
        }
    }

    // ========== Pagination Tests ==========

    #[test]
    fn pagination_new() {
        let p = Pagination::new(5, 10);
        assert_eq!(p.skip, 5);
        assert_eq!(p.limit, Some(10));
        assert!(!p.count_total);
    }

    #[test]
    fn pagination_limit_only() {
        let p = Pagination::limit(20);
        assert_eq!(p.skip, 0);
        assert_eq!(p.limit, Some(20));
        assert!(!p.count_total);
    }

    #[test]
    fn pagination_with_total_count() {
        let p = Pagination::new(0, 10).with_total_count();
        assert!(p.count_total);
    }

    #[test]
    fn pagination_is_empty() {
        assert!(Pagination::default().is_empty());
        assert!(!Pagination::limit(10).is_empty());
        assert!(!Pagination::new(5, 10).is_empty());
    }

    #[test]
    fn pagination_default() {
        let p = Pagination::default();
        assert_eq!(p.skip, 0);
        assert!(p.limit.is_none());
        assert!(!p.count_total);
    }

    #[test]
    fn paged_result_new() {
        let result: PagedResult<i32> = PagedResult::new(vec![1, 2, 3], Some(10), true);
        assert_eq!(result.items, vec![1, 2, 3]);
        assert_eq!(result.total_count, Some(10));
        assert!(result.has_more);
    }

    #[test]
    fn paged_result_is_empty() {
        let empty: PagedResult<i32> = PagedResult::default();
        assert!(empty.is_empty());

        let non_empty: PagedResult<i32> = PagedResult::new(vec![1], None, false);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn paged_result_len() {
        let result: PagedResult<i32> = PagedResult::new(vec![1, 2, 3], None, false);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn paged_result_default() {
        let result: PagedResult<i32> = PagedResult::default();
        assert!(result.items.is_empty());
        assert!(result.total_count.is_none());
        assert!(!result.has_more);
    }

    #[test]
    fn all_nodes_paginated_basic() {
        let engine = GraphEngine::new();
        for i in 0..5 {
            engine
                .create_node(&format!("Label{i}"), HashMap::new())
                .unwrap();
        }

        let result = engine.all_nodes_paginated(Pagination::limit(3));
        assert_eq!(result.len(), 3);
        assert!(result.has_more);
        assert!(result.total_count.is_none());
    }

    #[test]
    fn all_nodes_paginated_skip() {
        let engine = GraphEngine::new();
        for i in 0..5 {
            engine
                .create_node(&format!("Label{i}"), HashMap::new())
                .unwrap();
        }

        let result = engine.all_nodes_paginated(Pagination::new(2, 10));
        assert_eq!(result.len(), 3);
        assert!(!result.has_more);
    }

    #[test]
    fn all_nodes_paginated_has_more() {
        let engine = GraphEngine::new();
        for _ in 0..10 {
            engine.create_node("Test", HashMap::new()).unwrap();
        }

        let result = engine.all_nodes_paginated(Pagination::new(0, 5));
        assert!(result.has_more);

        let result = engine.all_nodes_paginated(Pagination::new(5, 5));
        assert!(!result.has_more);

        let result = engine.all_nodes_paginated(Pagination::new(8, 5));
        assert!(!result.has_more);
    }

    #[test]
    fn all_nodes_paginated_skip_beyond_total() {
        let engine = GraphEngine::new();
        for _ in 0..5 {
            engine.create_node("Test", HashMap::new()).unwrap();
        }

        let result = engine.all_nodes_paginated(Pagination::new(10, 5));
        assert!(result.is_empty());
        assert!(!result.has_more);
    }

    #[test]
    fn all_nodes_paginated_total_count() {
        let engine = GraphEngine::new();
        for _ in 0..5 {
            engine.create_node("Test", HashMap::new()).unwrap();
        }

        let result = engine.all_nodes_paginated(Pagination::limit(2).with_total_count());
        assert_eq!(result.len(), 2);
        assert_eq!(result.total_count, Some(5));
        assert!(result.has_more);
    }

    #[test]
    fn all_nodes_paginated_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine.all_nodes_paginated(Pagination::limit(10));
        assert!(result.is_empty());
        assert!(!result.has_more);
        assert!(result.total_count.is_none());
    }

    #[test]
    fn all_edges_paginated_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for i in 0..5 {
            engine
                .create_edge(n1, n2, &format!("REL{i}"), HashMap::new(), true)
                .unwrap();
        }

        let result = engine.all_edges_paginated(Pagination::limit(3));
        assert_eq!(result.len(), 3);
        assert!(result.has_more);
    }

    #[test]
    fn all_edges_paginated_skip() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for _ in 0..5 {
            engine
                .create_edge(n1, n2, "REL", HashMap::new(), true)
                .unwrap();
        }

        let result = engine.all_edges_paginated(Pagination::new(2, 10));
        assert_eq!(result.len(), 3);
        assert!(!result.has_more);
    }

    #[test]
    fn find_nodes_by_label_paginated_basic() {
        let engine = GraphEngine::new();
        for _ in 0..10 {
            engine.create_node("Person", HashMap::new()).unwrap();
        }
        for _ in 0..5 {
            engine.create_node("Company", HashMap::new()).unwrap();
        }

        let result = engine
            .find_nodes_by_label_paginated("Person", Pagination::limit(5).with_total_count())
            .unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.total_count, Some(10));
        assert!(result.has_more);
    }

    #[test]
    fn find_nodes_by_label_paginated_with_index() {
        let engine = GraphEngine::new();
        engine.create_label_index().unwrap();

        for _ in 0..10 {
            engine.create_node("Person", HashMap::new()).unwrap();
        }

        let result = engine
            .find_nodes_by_label_paginated("Person", Pagination::new(3, 4).with_total_count())
            .unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.total_count, Some(10));
        assert!(result.has_more);
    }

    #[test]
    fn find_edges_by_type_paginated_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for _ in 0..8 {
            engine
                .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
                .unwrap();
        }
        for _ in 0..4 {
            engine
                .create_edge(n1, n2, "LIKES", HashMap::new(), true)
                .unwrap();
        }

        let result = engine
            .find_edges_by_type_paginated("KNOWS", Pagination::limit(5).with_total_count())
            .unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.total_count, Some(8));
        assert!(result.has_more);
    }

    #[test]
    fn edges_of_paginated_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for _ in 0..6 {
            engine
                .create_edge(n1, n2, "REL", HashMap::new(), true)
                .unwrap();
        }

        let result = engine
            .edges_of_paginated(
                n1,
                Direction::Outgoing,
                Pagination::limit(4).with_total_count(),
            )
            .unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.total_count, Some(6));
        assert!(result.has_more);
    }

    #[test]
    fn edges_of_paginated_direction() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        let n3 = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(n1, n2, "OUT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n2, "OUT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n3, n1, "IN", HashMap::new(), true)
            .unwrap();

        let out_result = engine
            .edges_of_paginated(
                n1,
                Direction::Outgoing,
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(out_result.total_count, Some(2));

        let in_result = engine
            .edges_of_paginated(
                n1,
                Direction::Incoming,
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(in_result.total_count, Some(1));

        let both_result = engine
            .edges_of_paginated(
                n1,
                Direction::Both,
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(both_result.total_count, Some(3));
    }

    #[test]
    fn neighbors_paginated_basic() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();

        for i in 0..8 {
            let neighbor = engine
                .create_node(&format!("Neighbor{i}"), HashMap::new())
                .unwrap();
            engine
                .create_edge(center, neighbor, "CONNECTED", HashMap::new(), true)
                .unwrap();
        }

        let result = engine
            .neighbors_paginated(
                center,
                None,
                Direction::Outgoing,
                None,
                Pagination::limit(5).with_total_count(),
            )
            .unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.total_count, Some(8));
        assert!(result.has_more);
    }

    #[test]
    fn neighbors_paginated_with_filter() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();

        for i in 0..10 {
            let mut props = HashMap::new();
            props.insert("active".to_string(), PropertyValue::Bool(i % 2 == 0));
            let neighbor = engine.create_node("Neighbor", props).unwrap();
            engine
                .create_edge(center, neighbor, "CONNECTED", HashMap::new(), true)
                .unwrap();
        }

        let filter = TraversalFilter::new().node_eq("active", PropertyValue::Bool(true));

        let result = engine
            .neighbors_paginated(
                center,
                None,
                Direction::Outgoing,
                Some(&filter),
                Pagination::limit(3).with_total_count(),
            )
            .unwrap();
        assert_eq!(result.total_count, Some(5));
        assert_eq!(result.len(), 3);
        assert!(result.has_more);
    }

    #[test]
    fn paginated_results_match_unpaginated_order() {
        let engine = GraphEngine::new();
        for i in 0..20 {
            let mut props = HashMap::new();
            props.insert("order".to_string(), PropertyValue::Int(i));
            engine.create_node("Test", props).unwrap();
        }

        let all = engine.all_nodes();
        let page1 = engine.all_nodes_paginated(Pagination::new(0, 5));
        let page2 = engine.all_nodes_paginated(Pagination::new(5, 5));
        let page3 = engine.all_nodes_paginated(Pagination::new(10, 5));
        let page4 = engine.all_nodes_paginated(Pagination::new(15, 5));

        let mut reconstructed = Vec::new();
        reconstructed.extend(page1.items);
        reconstructed.extend(page2.items);
        reconstructed.extend(page3.items);
        reconstructed.extend(page4.items);

        assert_eq!(all.len(), reconstructed.len());
        for (a, b) in all.iter().zip(reconstructed.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn find_nodes_by_property_paginated_basic() {
        let engine = GraphEngine::new();

        for i in 0..10 {
            let mut props = HashMap::new();
            props.insert("status".to_string(), PropertyValue::String("active".into()));
            props.insert("index".to_string(), PropertyValue::Int(i));
            engine.create_node("Item", props).unwrap();
        }

        for i in 0..5 {
            let mut props = HashMap::new();
            props.insert(
                "status".to_string(),
                PropertyValue::String("inactive".into()),
            );
            props.insert("index".to_string(), PropertyValue::Int(i));
            engine.create_node("Item", props).unwrap();
        }

        let result = engine
            .find_nodes_by_property_paginated(
                "status",
                &PropertyValue::String("active".into()),
                Pagination::limit(5).with_total_count(),
            )
            .unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.total_count, Some(10));
        assert!(result.has_more);
    }

    #[test]
    fn find_nodes_where_paginated_basic() {
        let engine = GraphEngine::new();

        for i in 0..15 {
            let mut props = HashMap::new();
            props.insert("age".to_string(), PropertyValue::Int(i * 10));
            engine.create_node("Person", props).unwrap();
        }

        let result = engine
            .find_nodes_where_paginated(
                "age",
                RangeOp::Ge,
                &PropertyValue::Int(50),
                Pagination::limit(5).with_total_count(),
            )
            .unwrap();

        assert_eq!(result.total_count, Some(10));
        assert_eq!(result.len(), 5);
        assert!(result.has_more);
    }

    #[test]
    fn find_edges_by_property_paginated_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for i in 0..8 {
            let mut props = HashMap::new();
            props.insert("weight".to_string(), PropertyValue::Int(i));
            engine.create_edge(n1, n2, "REL", props, true).unwrap();
        }

        let result = engine
            .find_edges_by_property_paginated(
                "weight",
                &PropertyValue::Int(3),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();

        assert_eq!(result.total_count, Some(1));
        assert_eq!(result.len(), 1);
        assert!(!result.has_more);
    }

    #[test]
    fn find_edges_where_paginated_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for i in 0..10 {
            let mut props = HashMap::new();
            props.insert("score".to_string(), PropertyValue::Int(i * 10));
            engine.create_edge(n1, n2, "REL", props, true).unwrap();
        }

        let result = engine
            .find_edges_where_paginated(
                "score",
                RangeOp::Lt,
                &PropertyValue::Int(50),
                Pagination::limit(3).with_total_count(),
            )
            .unwrap();

        assert_eq!(result.total_count, Some(5));
        assert_eq!(result.len(), 3);
        assert!(result.has_more);
    }

    #[test]
    fn all_nodes_paginated_limit_zero() {
        let engine = GraphEngine::new();
        for _ in 0..5 {
            engine.create_node("Test", HashMap::new()).unwrap();
        }

        let result = engine.all_nodes_paginated(Pagination::new(0, 0).with_total_count());
        assert!(result.is_empty());
        assert_eq!(result.total_count, Some(5));
        assert!(result.has_more);
    }

    #[test]
    fn edges_paginated_results_match_unpaginated_order() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for _ in 0..10 {
            engine
                .create_edge(n1, n2, "REL", HashMap::new(), true)
                .unwrap();
        }

        let all = engine.all_edges();
        let page1 = engine.all_edges_paginated(Pagination::new(0, 5));
        let page2 = engine.all_edges_paginated(Pagination::new(5, 5));

        let mut reconstructed = Vec::new();
        reconstructed.extend(page1.items);
        reconstructed.extend(page2.items);

        assert_eq!(all.len(), reconstructed.len());
        for (a, b) in all.iter().zip(reconstructed.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn find_nodes_where_paginated_all_range_ops() {
        let engine = GraphEngine::new();

        for i in 0..10 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i * 10));
            engine.create_node("Test", props).unwrap();
        }

        let lt = engine
            .find_nodes_where_paginated(
                "value",
                RangeOp::Lt,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(lt.total_count, Some(5));

        let le = engine
            .find_nodes_where_paginated(
                "value",
                RangeOp::Le,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(le.total_count, Some(6));

        let gt = engine
            .find_nodes_where_paginated(
                "value",
                RangeOp::Gt,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(gt.total_count, Some(4));

        let ge = engine
            .find_nodes_where_paginated(
                "value",
                RangeOp::Ge,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(ge.total_count, Some(5));
    }

    #[test]
    fn find_edges_where_paginated_all_range_ops() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        for i in 0..10 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i * 10));
            engine.create_edge(n1, n2, "REL", props, true).unwrap();
        }

        let lt = engine
            .find_edges_where_paginated(
                "value",
                RangeOp::Lt,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(lt.total_count, Some(5));

        let le = engine
            .find_edges_where_paginated(
                "value",
                RangeOp::Le,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(le.total_count, Some(6));

        let gt = engine
            .find_edges_where_paginated(
                "value",
                RangeOp::Gt,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(gt.total_count, Some(4));

        let ge = engine
            .find_edges_where_paginated(
                "value",
                RangeOp::Ge,
                &PropertyValue::Int(50),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(ge.total_count, Some(5));
    }

    #[test]
    fn edges_of_paginated_nonexistent_node() {
        let engine = GraphEngine::new();
        let result = engine.edges_of_paginated(999, Direction::Both, Pagination::limit(10));
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn neighbors_paginated_nonexistent_node() {
        let engine = GraphEngine::new();
        let result =
            engine.neighbors_paginated(999, None, Direction::Both, None, Pagination::limit(10));
        assert!(matches!(result, Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn neighbors_paginated_edge_type_filter() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();

        for i in 0..5 {
            let neighbor = engine
                .create_node(&format!("N{i}"), HashMap::new())
                .unwrap();
            engine
                .create_edge(center, neighbor, "KNOWS", HashMap::new(), true)
                .unwrap();
        }
        for i in 5..10 {
            let neighbor = engine
                .create_node(&format!("N{i}"), HashMap::new())
                .unwrap();
            engine
                .create_edge(center, neighbor, "LIKES", HashMap::new(), true)
                .unwrap();
        }

        let result = engine
            .neighbors_paginated(
                center,
                Some("KNOWS"),
                Direction::Outgoing,
                None,
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(result.total_count, Some(5));
    }

    #[test]
    fn neighbors_paginated_incoming_direction() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();

        for i in 0..5 {
            let neighbor = engine
                .create_node(&format!("N{i}"), HashMap::new())
                .unwrap();
            engine
                .create_edge(neighbor, center, "KNOWS", HashMap::new(), true)
                .unwrap();
        }

        let result = engine
            .neighbors_paginated(
                center,
                None,
                Direction::Incoming,
                None,
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert_eq!(result.total_count, Some(5));
    }

    #[test]
    fn find_nodes_by_property_paginated_empty_result() {
        let engine = GraphEngine::new();
        engine.create_node("Test", HashMap::new()).unwrap();

        let result = engine
            .find_nodes_by_property_paginated(
                "nonexistent",
                &PropertyValue::String("value".into()),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert!(result.is_empty());
        assert_eq!(result.total_count, Some(0));
        assert!(!result.has_more);
    }

    #[test]
    fn find_edges_by_property_paginated_empty_result() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "REL", HashMap::new(), true)
            .unwrap();

        let result = engine
            .find_edges_by_property_paginated(
                "nonexistent",
                &PropertyValue::String("value".into()),
                Pagination::limit(10).with_total_count(),
            )
            .unwrap();
        assert!(result.is_empty());
        assert_eq!(result.total_count, Some(0));
        assert!(!result.has_more);
    }

    #[test]
    fn all_edges_paginated_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine.all_edges_paginated(Pagination::limit(10));
        assert!(result.is_empty());
        assert!(!result.has_more);
    }

    #[test]
    fn find_nodes_by_label_paginated_empty_result() {
        let engine = GraphEngine::new();
        engine.create_node("Other", HashMap::new()).unwrap();

        let result = engine
            .find_nodes_by_label_paginated("NonExistent", Pagination::limit(10).with_total_count())
            .unwrap();
        assert!(result.is_empty());
        assert_eq!(result.total_count, Some(0));
    }

    #[test]
    fn find_edges_by_type_paginated_empty_result() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "OTHER", HashMap::new(), true)
            .unwrap();

        let result = engine
            .find_edges_by_type_paginated("NonExistent", Pagination::limit(10).with_total_count())
            .unwrap();
        assert!(result.is_empty());
        assert_eq!(result.total_count, Some(0));
    }

    #[test]
    fn edges_of_paginated_empty_edges() {
        let engine = GraphEngine::new();
        let n = engine.create_node("Lonely", HashMap::new()).unwrap();

        let result = engine
            .edges_of_paginated(n, Direction::Both, Pagination::limit(10).with_total_count())
            .unwrap();
        assert!(result.is_empty());
        assert_eq!(result.total_count, Some(0));
    }

    // ========== Aggregation Tests ==========

    #[test]
    fn aggregate_result_empty() {
        let result = AggregateResult::empty();
        assert_eq!(result.count, 0);
        assert!(result.sum.is_none());
        assert!(result.avg.is_none());
        assert!(result.min.is_none());
        assert!(result.max.is_none());
    }

    #[test]
    fn aggregate_result_count_only() {
        let result = AggregateResult::count_only(42);
        assert_eq!(result.count, 42);
        assert!(result.sum.is_none());
        assert!(result.avg.is_none());
        assert!(result.min.is_none());
        assert!(result.max.is_none());
    }

    #[test]
    fn aggregate_result_default() {
        let result = AggregateResult::default();
        assert_eq!(result, AggregateResult::empty());
    }

    #[test]
    fn count_nodes_empty_graph() {
        let engine = GraphEngine::new();
        assert_eq!(engine.count_nodes(), 0);
    }

    #[test]
    fn count_nodes_basic() {
        let engine = GraphEngine::new();
        engine.create_node("A", HashMap::new()).unwrap();
        engine.create_node("B", HashMap::new()).unwrap();
        engine.create_node("A", HashMap::new()).unwrap();
        assert_eq!(engine.count_nodes(), 3);
    }

    #[test]
    fn count_nodes_by_label_basic() {
        let engine = GraphEngine::new();
        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Company", HashMap::new()).unwrap();

        assert_eq!(engine.count_nodes_by_label("Person").unwrap(), 2);
        assert_eq!(engine.count_nodes_by_label("Company").unwrap(), 1);
    }

    #[test]
    fn count_nodes_by_label_no_match() {
        let engine = GraphEngine::new();
        engine.create_node("Person", HashMap::new()).unwrap();
        assert_eq!(engine.count_nodes_by_label("NonExistent").unwrap(), 0);
    }

    #[test]
    fn count_edges_empty_graph() {
        let engine = GraphEngine::new();
        assert_eq!(engine.count_edges(), 0);
    }

    #[test]
    fn count_edges_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n1, "FOLLOWS", HashMap::new(), true)
            .unwrap();
        assert_eq!(engine.count_edges(), 2);
    }

    #[test]
    fn count_edges_by_type_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n1, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n1, n2, "FOLLOWS", HashMap::new(), true)
            .unwrap();

        assert_eq!(engine.count_edges_by_type("KNOWS").unwrap(), 2);
        assert_eq!(engine.count_edges_by_type("FOLLOWS").unwrap(), 1);
    }

    #[test]
    fn count_edges_by_type_no_match() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        assert_eq!(engine.count_edges_by_type("NonExistent").unwrap(), 0);
    }

    #[test]
    fn aggregate_node_property_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine.aggregate_node_property("age");
        assert_eq!(result, AggregateResult::empty());
    }

    #[test]
    fn aggregate_node_property_int_values() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("age".to_string(), PropertyValue::Int(20));
        let mut props2 = HashMap::new();
        props2.insert("age".to_string(), PropertyValue::Int(30));
        let mut props3 = HashMap::new();
        props3.insert("age".to_string(), PropertyValue::Int(40));

        engine.create_node("Person", props1).unwrap();
        engine.create_node("Person", props2).unwrap();
        engine.create_node("Person", props3).unwrap();

        let result = engine.aggregate_node_property("age");
        assert_eq!(result.count, 3);
        assert_eq!(result.sum, Some(90.0));
        assert_eq!(result.avg, Some(30.0));
        assert_eq!(result.min, Some(PropertyValue::Int(20)));
        assert_eq!(result.max, Some(PropertyValue::Int(40)));
    }

    #[test]
    fn aggregate_node_property_float_values() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("score".to_string(), PropertyValue::Float(1.5));
        let mut props2 = HashMap::new();
        props2.insert("score".to_string(), PropertyValue::Float(2.5));

        engine.create_node("Item", props1).unwrap();
        engine.create_node("Item", props2).unwrap();

        let result = engine.aggregate_node_property("score");
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(4.0));
        assert_eq!(result.avg, Some(2.0));
        assert_eq!(result.min, Some(PropertyValue::Float(1.5)));
        assert_eq!(result.max, Some(PropertyValue::Float(2.5)));
    }

    #[test]
    fn aggregate_node_property_mixed_numeric() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("value".to_string(), PropertyValue::Int(10));
        let mut props2 = HashMap::new();
        props2.insert("value".to_string(), PropertyValue::Float(20.5));

        engine.create_node("Data", props1).unwrap();
        engine.create_node("Data", props2).unwrap();

        let result = engine.aggregate_node_property("value");
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(30.5));
        assert_eq!(result.avg, Some(15.25));
    }

    #[test]
    fn aggregate_node_property_non_numeric() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        let mut props2 = HashMap::new();
        props2.insert("name".to_string(), PropertyValue::String("Bob".to_string()));

        engine.create_node("Person", props1).unwrap();
        engine.create_node("Person", props2).unwrap();

        let result = engine.aggregate_node_property("name");
        assert_eq!(result.count, 2);
        assert!(result.sum.is_none());
        assert!(result.avg.is_none());
        assert_eq!(result.min, Some(PropertyValue::String("Alice".to_string())));
        assert_eq!(result.max, Some(PropertyValue::String("Bob".to_string())));
    }

    #[test]
    fn aggregate_node_property_with_nulls() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("value".to_string(), PropertyValue::Int(10));
        let mut props2 = HashMap::new();
        props2.insert("value".to_string(), PropertyValue::Null);
        let mut props3 = HashMap::new();
        props3.insert("value".to_string(), PropertyValue::Int(20));

        engine.create_node("Data", props1).unwrap();
        engine.create_node("Data", props2).unwrap();
        engine.create_node("Data", props3).unwrap();

        let result = engine.aggregate_node_property("value");
        assert_eq!(result.count, 3);
        assert_eq!(result.sum, Some(30.0));
        assert_eq!(result.avg, Some(15.0));
        assert_eq!(result.min, Some(PropertyValue::Null));
        assert_eq!(result.max, Some(PropertyValue::Int(20)));
    }

    #[test]
    fn aggregate_node_property_by_label() {
        let engine = GraphEngine::new();
        let mut person1 = HashMap::new();
        person1.insert("age".to_string(), PropertyValue::Int(25));
        let mut person2 = HashMap::new();
        person2.insert("age".to_string(), PropertyValue::Int(35));
        let mut company = HashMap::new();
        company.insert("age".to_string(), PropertyValue::Int(100));

        engine.create_node("Person", person1).unwrap();
        engine.create_node("Person", person2).unwrap();
        engine.create_node("Company", company).unwrap();

        let result = engine
            .aggregate_node_property_by_label("Person", "age")
            .unwrap();
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(60.0));
        assert_eq!(result.avg, Some(30.0));
    }

    #[test]
    fn aggregate_node_property_where() {
        let engine = GraphEngine::new();
        engine.create_node_property_index("age").unwrap();

        let mut props1 = HashMap::new();
        props1.insert("age".to_string(), PropertyValue::Int(20));
        props1.insert("score".to_string(), PropertyValue::Int(80));
        let mut props2 = HashMap::new();
        props2.insert("age".to_string(), PropertyValue::Int(30));
        props2.insert("score".to_string(), PropertyValue::Int(90));
        let mut props3 = HashMap::new();
        props3.insert("age".to_string(), PropertyValue::Int(40));
        props3.insert("score".to_string(), PropertyValue::Int(70));

        engine.create_node("Person", props1).unwrap();
        engine.create_node("Person", props2).unwrap();
        engine.create_node("Person", props3).unwrap();

        let result = engine
            .aggregate_node_property_where("age", RangeOp::Gt, &PropertyValue::Int(25), "score")
            .unwrap();
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(160.0));
        assert_eq!(result.avg, Some(80.0));
    }

    #[test]
    fn aggregate_edge_property_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("weight".to_string(), PropertyValue::Int(5));
        let mut props2 = HashMap::new();
        props2.insert("weight".to_string(), PropertyValue::Int(10));

        engine.create_edge(n1, n2, "CONN", props1, true).unwrap();
        engine.create_edge(n2, n1, "CONN", props2, true).unwrap();

        let result = engine.aggregate_edge_property("weight");
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(15.0));
        assert_eq!(result.avg, Some(7.5));
    }

    #[test]
    fn aggregate_edge_property_by_type() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("strength".to_string(), PropertyValue::Int(10));
        let mut props2 = HashMap::new();
        props2.insert("strength".to_string(), PropertyValue::Int(20));
        let mut props3 = HashMap::new();
        props3.insert("strength".to_string(), PropertyValue::Int(100));

        engine.create_edge(n1, n2, "KNOWS", props1, true).unwrap();
        engine.create_edge(n2, n1, "KNOWS", props2, true).unwrap();
        engine.create_edge(n1, n2, "FOLLOWS", props3, true).unwrap();

        let result = engine
            .aggregate_edge_property_by_type("KNOWS", "strength")
            .unwrap();
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(30.0));
        assert_eq!(result.avg, Some(15.0));
    }

    #[test]
    fn aggregate_edge_property_where() {
        let engine = GraphEngine::new();
        engine.create_edge_property_index("weight").unwrap();

        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("weight".to_string(), PropertyValue::Int(5));
        props1.insert("cost".to_string(), PropertyValue::Int(100));
        let mut props2 = HashMap::new();
        props2.insert("weight".to_string(), PropertyValue::Int(15));
        props2.insert("cost".to_string(), PropertyValue::Int(200));
        let mut props3 = HashMap::new();
        props3.insert("weight".to_string(), PropertyValue::Int(25));
        props3.insert("cost".to_string(), PropertyValue::Int(300));

        engine.create_edge(n1, n2, "CONN", props1, true).unwrap();
        engine.create_edge(n2, n1, "CONN", props2, true).unwrap();
        engine.create_edge(n1, n2, "CONN", props3, true).unwrap();

        let result = engine
            .aggregate_edge_property_where("weight", RangeOp::Ge, &PropertyValue::Int(15), "cost")
            .unwrap();
        assert_eq!(result.count, 2);
        assert_eq!(result.sum, Some(500.0));
        assert_eq!(result.avg, Some(250.0));
    }

    #[test]
    fn sum_node_property_basic() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("value".to_string(), PropertyValue::Int(10));
        let mut props2 = HashMap::new();
        props2.insert("value".to_string(), PropertyValue::Int(20));

        engine.create_node("N", props1).unwrap();
        engine.create_node("N", props2).unwrap();

        assert_eq!(engine.sum_node_property("value"), Some(30.0));
    }

    #[test]
    fn avg_node_property_basic() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("value".to_string(), PropertyValue::Int(10));
        let mut props2 = HashMap::new();
        props2.insert("value".to_string(), PropertyValue::Int(30));

        engine.create_node("N", props1).unwrap();
        engine.create_node("N", props2).unwrap();

        assert_eq!(engine.avg_node_property("value"), Some(20.0));
    }

    #[test]
    fn sum_edge_property_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("weight".to_string(), PropertyValue::Float(1.5));
        let mut props2 = HashMap::new();
        props2.insert("weight".to_string(), PropertyValue::Float(2.5));

        engine.create_edge(n1, n2, "E", props1, true).unwrap();
        engine.create_edge(n2, n1, "E", props2, true).unwrap();

        assert_eq!(engine.sum_edge_property("weight"), Some(4.0));
    }

    #[test]
    fn avg_edge_property_basic() {
        let engine = GraphEngine::new();
        let n1 = engine.create_node("A", HashMap::new()).unwrap();
        let n2 = engine.create_node("B", HashMap::new()).unwrap();

        let mut props1 = HashMap::new();
        props1.insert("weight".to_string(), PropertyValue::Float(2.0));
        let mut props2 = HashMap::new();
        props2.insert("weight".to_string(), PropertyValue::Float(4.0));

        engine.create_edge(n1, n2, "E", props1, true).unwrap();
        engine.create_edge(n2, n1, "E", props2, true).unwrap();

        assert_eq!(engine.avg_edge_property("weight"), Some(3.0));
    }

    #[test]
    fn aggregate_finds_min_int() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("value".to_string(), PropertyValue::Int(50));
        let mut props2 = HashMap::new();
        props2.insert("value".to_string(), PropertyValue::Int(10));
        let mut props3 = HashMap::new();
        props3.insert("value".to_string(), PropertyValue::Int(30));

        engine.create_node("N", props1).unwrap();
        engine.create_node("N", props2).unwrap();
        engine.create_node("N", props3).unwrap();

        let result = engine.aggregate_node_property("value");
        assert_eq!(result.min, Some(PropertyValue::Int(10)));
    }

    #[test]
    fn aggregate_finds_max_int() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("value".to_string(), PropertyValue::Int(50));
        let mut props2 = HashMap::new();
        props2.insert("value".to_string(), PropertyValue::Int(10));
        let mut props3 = HashMap::new();
        props3.insert("value".to_string(), PropertyValue::Int(30));

        engine.create_node("N", props1).unwrap();
        engine.create_node("N", props2).unwrap();
        engine.create_node("N", props3).unwrap();

        let result = engine.aggregate_node_property("value");
        assert_eq!(result.max, Some(PropertyValue::Int(50)));
    }

    #[test]
    fn aggregate_finds_min_string() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert(
            "name".to_string(),
            PropertyValue::String("Zebra".to_string()),
        );
        let mut props2 = HashMap::new();
        props2.insert(
            "name".to_string(),
            PropertyValue::String("Apple".to_string()),
        );
        let mut props3 = HashMap::new();
        props3.insert(
            "name".to_string(),
            PropertyValue::String("Mango".to_string()),
        );

        engine.create_node("Item", props1).unwrap();
        engine.create_node("Item", props2).unwrap();
        engine.create_node("Item", props3).unwrap();

        let result = engine.aggregate_node_property("name");
        assert_eq!(result.min, Some(PropertyValue::String("Apple".to_string())));
        assert_eq!(result.max, Some(PropertyValue::String("Zebra".to_string())));
    }

    #[test]
    fn aggregate_finds_max_float() {
        let engine = GraphEngine::new();
        let mut props1 = HashMap::new();
        props1.insert("score".to_string(), PropertyValue::Float(1.1));
        let mut props2 = HashMap::new();
        props2.insert("score".to_string(), PropertyValue::Float(9.9));
        let mut props3 = HashMap::new();
        props3.insert("score".to_string(), PropertyValue::Float(5.5));

        engine.create_node("Item", props1).unwrap();
        engine.create_node("Item", props2).unwrap();
        engine.create_node("Item", props3).unwrap();

        let result = engine.aggregate_node_property("score");
        assert_eq!(result.max, Some(PropertyValue::Float(9.9)));
    }

    #[test]
    fn aggregate_empty_property_name() {
        let engine = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert("".to_string(), PropertyValue::Int(42));
        engine.create_node("N", props).unwrap();

        let result = engine.aggregate_node_property("");
        assert_eq!(result.count, 1);
        assert_eq!(result.sum, Some(42.0));
    }

    #[test]
    fn aggregate_nonexistent_property() {
        let engine = GraphEngine::new();
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("test".to_string()),
        );
        engine.create_node("N", props).unwrap();

        let result = engine.aggregate_node_property("nonexistent");
        assert_eq!(result, AggregateResult::empty());
    }

    #[test]
    fn aggregate_large_dataset_parallel() {
        let engine = GraphEngine::new();

        // Create more than AGGREGATE_PARALLEL_THRESHOLD nodes
        for i in 0..1500 {
            let mut props = HashMap::new();
            props.insert("value".to_string(), PropertyValue::Int(i));
            engine.create_node("Data", props).unwrap();
        }

        let result = engine.aggregate_node_property("value");
        assert_eq!(result.count, 1500);
        // Sum of 0..1500 = n*(n-1)/2 = 1500*1499/2 = 1124250
        assert_eq!(result.sum, Some(1_124_250.0));
        assert_eq!(result.avg, Some(1_124_250.0 / 1500.0));
        assert_eq!(result.min, Some(PropertyValue::Int(0)));
        assert_eq!(result.max, Some(PropertyValue::Int(1499)));
    }
}
