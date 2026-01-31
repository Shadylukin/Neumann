//! Graph engine for property graphs with nodes, edges, and traversal algorithms.
//!
//! This crate provides a thread-safe graph database engine supporting:
//! - Labeled nodes with arbitrary properties
//! - Directed and undirected edges with properties
//! - Property indexes for efficient lookups
//! - Unique and existence constraints
//! - Graph algorithms (`PageRank`, betweenness centrality, connected components)
//! - BFS/DFS traversal with filtering
//! - Batch operations for high-throughput scenarios
//!
//! # Thread Safety
//!
//! `GraphEngine` is thread-safe and can be shared across threads using `Arc`.
//! All operations use fine-grained locking via `parking_lot::RwLock`.

// Module declarations
pub mod algorithms;
mod config;
pub mod distributed;
mod error;
mod fulltext;
mod geo;
pub mod partitioning;

// Re-exports from modules
pub use algorithms::{
    AStarConfig, AStarResult, BiconnectedConfig, BiconnectedResult, HeuristicFn, KCoreConfig,
    KCoreResult, MstConfig, MstEdge, MstResult, SccConfig, SccResult, SimilarityConfig,
    SimilarityMetric, SimilarityResult, TriangleConfig, TriangleResult,
};
pub use config::GraphEngineConfig;
pub use distributed::{
    CrossShardQuery, DistributedConfig, DistributedError, DistributedGraphEngine,
    DistributedResult, DistributedStats, DistributedStatsSnapshot, DistributedTransaction,
    GraphOperation,
};
pub use error::{GraphError, Result};
pub use fulltext::{FullTextConfig, FullTextIndex};
pub use geo::{GeoConfig, GeoIndex, GeoPoint};
pub use partitioning::{
    GraphPartitioner, PartitionAssignment, PartitionConfig, PartitionStats, PartitionStrategy,
    ShardId,
};

use std::{
    cmp::Ordering as CmpOrdering,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
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
use tensor_store::{fields, ScalarValue, TensorData, TensorStore, TensorValue};
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
    DateTime(i64),
    Bytes(Vec<u8>),
    /// For complex types (List, Map, Point), we use JSON string for ordering.
    Complex(String),
}

impl From<&PropertyValue> for OrderedPropertyValue {
    fn from(v: &PropertyValue) -> Self {
        match v {
            PropertyValue::Null => Self::Null,
            PropertyValue::Bool(b) => Self::Bool(*b),
            PropertyValue::Int(i) => Self::Int(*i),
            PropertyValue::Float(f) => Self::Float(OrderedFloat(*f)),
            PropertyValue::String(s) => Self::String(s.clone()),
            PropertyValue::DateTime(dt) => Self::DateTime(*dt),
            PropertyValue::Bytes(b) => Self::Bytes(b.clone()),
            PropertyValue::List(_) | PropertyValue::Map(_) | PropertyValue::Point { .. } => {
                Self::Complex(serde_json::to_string(v).unwrap_or_default())
            },
        }
    }
}

/// A typed property value for nodes and edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::use_self)] // Self cannot be used in recursive enum variants
pub enum PropertyValue {
    /// Absence of value.
    Null,
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point.
    Float(f64),
    /// UTF-8 string.
    String(String),
    /// Boolean value.
    Bool(bool),
    /// Unix timestamp in milliseconds.
    DateTime(i64),
    /// Ordered list of values.
    List(Vec<PropertyValue>),
    /// Key-value map.
    Map(HashMap<String, PropertyValue>),
    /// Raw bytes.
    Bytes(Vec<u8>),
    /// Geospatial point (latitude, longitude in degrees).
    Point { lat: f64, lon: f64 },
}

#[allow(clippy::use_self)] // Self cannot be used in recursive type return positions
impl PropertyValue {
    fn to_scalar(&self) -> ScalarValue {
        match self {
            Self::Null => ScalarValue::Null,
            Self::Int(v) | Self::DateTime(v) => ScalarValue::Int(*v),
            Self::Float(v) => ScalarValue::Float(*v),
            Self::String(v) => ScalarValue::String(v.clone()),
            Self::Bool(v) => ScalarValue::Bool(*v),
            Self::Bytes(v) => ScalarValue::Bytes(v.clone()),
            // Complex types serialize to JSON string for storage
            Self::List(_) | Self::Map(_) | Self::Point { .. } => {
                ScalarValue::String(serde_json::to_string(self).unwrap_or_default())
            },
        }
    }

    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Int(v) => Self::Int(*v),
            ScalarValue::Float(v) => Self::Float(*v),
            ScalarValue::String(v) => {
                // Try to deserialize as JSON-encoded complex type first
                if v.starts_with('{') || v.starts_with('[') {
                    if let Ok(pv) = serde_json::from_str::<Self>(v) {
                        return pv;
                    }
                }
                Self::String(v.clone())
            },
            ScalarValue::Bool(v) => Self::Bool(*v),
            ScalarValue::Bytes(v) => Self::Bytes(v.clone()),
            ScalarValue::Null => Self::Null,
        }
    }

    #[must_use]
    pub const fn value_type(&self) -> PropertyValueType {
        match self {
            Self::Null => PropertyValueType::Null,
            Self::Int(_) => PropertyValueType::Int,
            Self::Float(_) => PropertyValueType::Float,
            Self::String(_) => PropertyValueType::String,
            Self::Bool(_) => PropertyValueType::Bool,
            Self::DateTime(_) => PropertyValueType::DateTime,
            Self::List(_) => PropertyValueType::List,
            Self::Map(_) => PropertyValueType::Map,
            Self::Bytes(_) => PropertyValueType::Bytes,
            Self::Point { .. } => PropertyValueType::Point,
        }
    }

    /// Returns the value as a list, if it is one.
    #[must_use]
    pub fn as_list(&self) -> Option<&[PropertyValue]> {
        match self {
            Self::List(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the value as a map, if it is one.
    #[must_use]
    pub const fn as_map(&self) -> Option<&HashMap<String, PropertyValue>> {
        match self {
            Self::Map(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the value as bytes, if it is one.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the value as a point (lat, lon), if it is one.
    #[must_use]
    pub const fn as_point(&self) -> Option<(f64, f64)> {
        match self {
            Self::Point { lat, lon } => Some((*lat, *lon)),
            _ => None,
        }
    }

    /// Returns the value as a datetime (Unix millis), if it is one.
    #[must_use]
    pub const fn as_datetime(&self) -> Option<i64> {
        match self {
            Self::DateTime(v) => Some(*v),
            _ => None,
        }
    }

    /// Checks if a list contains a value.
    #[must_use]
    pub fn contains(&self, value: &PropertyValue) -> bool {
        match self {
            Self::List(list) => list.contains(value),
            _ => false,
        }
    }

    /// Calculates the distance in kilometers between two points using the Haversine formula.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_km(&self, other: &Self) -> Option<f64> {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let (lat1, lon1) = self.as_point()?;
        let (lat2, lon2) = other.as_point()?;

        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let delta_lat = (lat2 - lat1).to_radians();
        let delta_lon = (lon2 - lon1).to_radians();

        let sin_delta_lat_half = (delta_lat / 2.0).sin();
        let sin_delta_lon_half = (delta_lon / 2.0).sin();
        let a = (lat1_rad.cos() * lat2_rad.cos())
            .mul_add(sin_delta_lon_half.powi(2), sin_delta_lat_half.powi(2));
        let c = 2.0 * a.sqrt().asin();

        Some(EARTH_RADIUS_KM * c)
    }
}

/// Property value type for type constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyValueType {
    Null,
    Int,
    Float,
    String,
    Bool,
    DateTime,
    List,
    Map,
    Bytes,
    Point,
}

/// Type of constraint enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Property must be unique across all nodes/edges of given label/type.
    Unique,
    /// Property must exist (not null).
    Exists,
    /// Property must match a specific type.
    PropertyType(PropertyValueType),
}

/// Target scope of a constraint.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintTarget {
    /// Applies to nodes with specific label.
    NodeLabel(String),
    /// Applies to edges with specific type.
    EdgeType(String),
    /// Applies to all nodes.
    AllNodes,
    /// Applies to all edges.
    AllEdges,
}

/// A constraint definition for property validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub target: ConstraintTarget,
    pub property: String,
    pub constraint_type: ConstraintType,
}

/// Input for batch node creation.
#[derive(Debug, Clone)]
pub struct NodeInput {
    pub labels: Vec<String>,
    pub properties: HashMap<String, PropertyValue>,
}

impl NodeInput {
    #[must_use]
    pub const fn new(labels: Vec<String>, properties: HashMap<String, PropertyValue>) -> Self {
        Self { labels, properties }
    }
}

/// Input for batch edge creation.
#[derive(Debug, Clone)]
pub struct EdgeInput {
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub properties: HashMap<String, PropertyValue>,
    pub directed: bool,
}

impl EdgeInput {
    #[must_use]
    pub fn new(
        from: u64,
        to: u64,
        edge_type: impl Into<String>,
        properties: HashMap<String, PropertyValue>,
        directed: bool,
    ) -> Self {
        Self {
            from,
            to,
            edge_type: edge_type.into(),
            properties,
            directed,
        }
    }
}

/// Result of a batch operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchResult {
    pub created_ids: Vec<u64>,
    pub count: usize,
}

/// Error detail for a failed item in a graph batch operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBatchItemError {
    /// Index of the failed item in the original batch.
    pub index: usize,
    /// ID of the failed item (if known).
    pub id: Option<u64>,
    /// Description of the failure cause.
    pub cause: String,
}

/// Batch delete result with failure tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchDeleteResult {
    /// IDs successfully deleted.
    pub deleted_ids: Vec<u64>,
    /// Count of successful deletions.
    pub count: usize,
    /// Failed deletions with reasons.
    pub failed: Vec<GraphBatchItemError>,
}

impl BatchDeleteResult {
    #[must_use]
    pub fn new(deleted_ids: Vec<u64>) -> Self {
        let count = deleted_ids.len();
        Self {
            deleted_ids,
            count,
            failed: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_failures(deleted_ids: Vec<u64>, failed: Vec<GraphBatchItemError>) -> Self {
        let count = deleted_ids.len();
        Self {
            deleted_ids,
            count,
            failed,
        }
    }

    #[must_use]
    pub fn has_failures(&self) -> bool {
        !self.failed.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Direction {
    Outgoing,
    Incoming,
    #[default]
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

    #[allow(clippy::option_if_let_else)] // match is clearer for this logic
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
    pub const fn is_empty(&self) -> bool {
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
    pub const fn direction(mut self, dir: Direction) -> Self {
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
    pub const fn is_empty(&self) -> bool {
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedResult<T> {
    pub items: Vec<T>,
    pub total_count: Option<usize>,
    pub has_more: bool,
}

impl<T> PagedResult<T> {
    #[must_use]
    pub const fn new(items: Vec<T>, total_count: Option<usize>, has_more: bool) -> Self {
        Self {
            items,
            total_count,
            has_more,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
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
    pub(crate) const fn empty() -> Self {
        Self {
            count: 0,
            sum: None,
            avg: None,
            min: None,
            max: None,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn count_only(count: u64) -> Self {
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

// ========== Pattern Matching Types ==========

/// Maximum hops for variable-length edge patterns (safety cap for `VariableLengthSpec`).
pub const MAX_VARIABLE_LENGTH_HOPS: usize = 20;

/// Variable-length path specification (e.g., *1..5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariableLengthSpec {
    pub min_hops: usize,
    pub max_hops: usize,
}

impl VariableLengthSpec {
    #[must_use]
    pub fn new(min_hops: usize, max_hops: usize) -> Self {
        Self {
            min_hops,
            max_hops: max_hops.min(MAX_VARIABLE_LENGTH_HOPS),
        }
    }
}

/// A node pattern that matches nodes by label and/or property conditions.
#[derive(Debug, Clone, Default)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub conditions: Vec<PropertyCondition>,
}

impl NodePattern {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn variable(mut self, name: &str) -> Self {
        self.variable = Some(name.to_string());
        self
    }

    #[must_use]
    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    #[must_use]
    pub fn where_eq(self, property: &str, value: PropertyValue) -> Self {
        self.where_cond(property, CompareOp::Eq, value)
    }

    #[must_use]
    pub fn where_cond(mut self, property: &str, op: CompareOp, value: PropertyValue) -> Self {
        self.conditions
            .push(PropertyCondition::new(property, op, value));
        self
    }

    #[must_use]
    pub fn matches(&self, node: &Node) -> bool {
        // Check label if specified
        if let Some(ref label) = self.label {
            if !node.has_label(label) {
                return false;
            }
        }
        // Check all property conditions
        self.conditions.iter().all(|c| c.matches_node(node))
    }
}

/// An edge pattern that matches edges by type and/or property conditions.
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub variable: Option<String>,
    pub edge_type: Option<String>,
    pub direction: Direction,
    pub conditions: Vec<PropertyCondition>,
    pub variable_length: Option<VariableLengthSpec>,
}

impl Default for EdgePattern {
    fn default() -> Self {
        Self {
            variable: None,
            edge_type: None,
            direction: Direction::Outgoing,
            conditions: Vec::new(),
            variable_length: None,
        }
    }
}

impl EdgePattern {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn variable(mut self, name: &str) -> Self {
        self.variable = Some(name.to_string());
        self
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: &str) -> Self {
        self.edge_type = Some(edge_type.to_string());
        self
    }

    #[must_use]
    pub const fn direction(mut self, dir: Direction) -> Self {
        self.direction = dir;
        self
    }

    #[must_use]
    pub fn variable_length(mut self, min: usize, max: usize) -> Self {
        self.variable_length = Some(VariableLengthSpec::new(min, max));
        self
    }

    #[must_use]
    pub fn where_eq(self, property: &str, value: PropertyValue) -> Self {
        self.where_cond(property, CompareOp::Eq, value)
    }

    #[must_use]
    pub fn where_cond(mut self, property: &str, op: CompareOp, value: PropertyValue) -> Self {
        self.conditions
            .push(PropertyCondition::new(property, op, value));
        self
    }

    #[must_use]
    pub fn matches(&self, edge: &Edge) -> bool {
        // Check edge type if specified
        if let Some(ref et) = self.edge_type {
            if edge.edge_type != *et {
                return false;
            }
        }
        // Check all property conditions
        self.conditions.iter().all(|c| c.matches_edge(edge))
    }
}

/// An element in a path pattern.
#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern),
}

/// A path pattern: `(a)-[r]->(b)-[s]->(c)`
#[derive(Debug, Clone)]
pub struct PathPattern {
    pub elements: Vec<PatternElement>,
}

impl PathPattern {
    #[must_use]
    pub fn new(start: NodePattern, edge: EdgePattern, end: NodePattern) -> Self {
        Self {
            elements: vec![
                PatternElement::Node(start),
                PatternElement::Edge(edge),
                PatternElement::Node(end),
            ],
        }
    }

    #[must_use]
    pub fn extend(mut self, edge: EdgePattern, node: NodePattern) -> Self {
        self.elements.push(PatternElement::Edge(edge));
        self.elements.push(PatternElement::Node(node));
        self
    }

    #[allow(dead_code)]
    pub(crate) fn node_patterns(&self) -> impl Iterator<Item = &NodePattern> {
        self.elements.iter().filter_map(|e| {
            if let PatternElement::Node(np) = e {
                Some(np)
            } else {
                None
            }
        })
    }

    #[allow(dead_code)]
    pub(crate) fn edge_patterns(&self) -> impl Iterator<Item = &EdgePattern> {
        self.elements.iter().filter_map(|e| {
            if let PatternElement::Edge(ep) = e {
                Some(ep)
            } else {
                None
            }
        })
    }

    fn node_pattern_at(&self, element_idx: usize) -> Option<&NodePattern> {
        self.elements.get(element_idx).and_then(|e| {
            if let PatternElement::Node(np) = e {
                Some(np)
            } else {
                None
            }
        })
    }

    fn edge_pattern_at(&self, element_idx: usize) -> Option<&EdgePattern> {
        self.elements.get(element_idx).and_then(|e| {
            if let PatternElement::Edge(ep) = e {
                Some(ep)
            } else {
                None
            }
        })
    }
}

/// A complete pattern query.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub path: PathPattern,
    pub limit: Option<usize>,
}

impl Pattern {
    #[must_use]
    pub const fn new(path: PathPattern) -> Self {
        Self { path, limit: None }
    }

    #[must_use]
    pub const fn limit(mut self, max: usize) -> Self {
        self.limit = Some(max);
        self
    }
}

/// A binding of a variable to a graph element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Binding {
    Node(Node),
    Edge(Edge),
    Path(Path),
}

/// A single match with all variable bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternMatch {
    pub bindings: HashMap<String, Binding>,
}

impl PatternMatch {
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    #[must_use]
    pub fn get_node(&self, var: &str) -> Option<&Node> {
        self.bindings.get(var).and_then(|b| {
            if let Binding::Node(n) = b {
                Some(n)
            } else {
                None
            }
        })
    }

    #[must_use]
    pub fn get_edge(&self, var: &str) -> Option<&Edge> {
        self.bindings.get(var).and_then(|b| {
            if let Binding::Edge(e) = b {
                Some(e)
            } else {
                None
            }
        })
    }

    #[must_use]
    pub fn get_path(&self, var: &str) -> Option<&Path> {
        self.bindings.get(var).and_then(|b| {
            if let Binding::Path(p) = b {
                Some(p)
            } else {
                None
            }
        })
    }
}

impl Default for PatternMatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from pattern matching.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PatternMatchStats {
    pub matches_found: usize,
    pub nodes_evaluated: usize,
    pub edges_evaluated: usize,
    pub truncated: bool,
}

/// Result of pattern matching operation.
#[derive(Debug, Clone)]
pub struct PatternMatchResult {
    pub matches: Vec<PatternMatch>,
    pub stats: PatternMatchStats,
}

impl PatternMatchResult {
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            matches: Vec::new(),
            stats: PatternMatchStats::default(),
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.matches.len()
    }
}

impl Default for PatternMatchResult {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// Graph Algorithm Types
// ============================================================================

const PAGERANK_DEFAULT_DAMPING: f64 = 0.85;
const PAGERANK_DEFAULT_TOLERANCE: f64 = 1e-6;
const PAGERANK_DEFAULT_MAX_ITERATIONS: usize = 100;
const COMMUNITY_MAX_PASSES: usize = 10;
const LABEL_PROPAGATION_MAX_ITERATIONS: usize = 100;

/// Configuration for `PageRank` algorithm.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    pub damping: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub direction: Direction,
    pub edge_type: Option<String>,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: PAGERANK_DEFAULT_DAMPING,
            tolerance: PAGERANK_DEFAULT_TOLERANCE,
            max_iterations: PAGERANK_DEFAULT_MAX_ITERATIONS,
            direction: Direction::Outgoing,
            edge_type: None,
        }
    }
}

impl PageRankConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    #[must_use]
    pub const fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    #[must_use]
    pub const fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    #[must_use]
    pub const fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }
}

/// Result of `PageRank` computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageRankResult {
    pub scores: HashMap<u64, f64>,
    pub iterations: usize,
    pub convergence: f64,
    pub converged: bool,
}

impl PageRankResult {
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            scores: HashMap::new(),
            iterations: 0,
            convergence: 0.0,
            converged: true,
        }
    }

    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<(u64, f64)> {
        let mut sorted: Vec<_> = self.scores.iter().map(|(&id, &s)| (id, s)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));
        sorted.truncate(k);
        sorted
    }
}

impl Default for PageRankResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Type of centrality measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CentralityType {
    Betweenness,
    Closeness,
    Eigenvector,
}

/// Configuration for centrality algorithms.
#[derive(Debug, Clone)]
pub struct CentralityConfig {
    pub direction: Direction,
    pub edge_type: Option<String>,
    pub sampling_ratio: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for CentralityConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Both,
            edge_type: None,
            sampling_ratio: 1.0,
            max_iterations: PAGERANK_DEFAULT_MAX_ITERATIONS,
            tolerance: PAGERANK_DEFAULT_TOLERANCE,
        }
    }
}

impl CentralityConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // clamp() is not const
    pub fn sampling_ratio(mut self, ratio: f64) -> Self {
        self.sampling_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    #[must_use]
    pub const fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    #[must_use]
    pub const fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Result of centrality computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentralityResult {
    pub scores: HashMap<u64, f64>,
    pub centrality_type: CentralityType,
    pub iterations: Option<usize>,
    pub converged: Option<bool>,
    pub sample_count: Option<usize>,
}

impl CentralityResult {
    #[must_use]
    pub(crate) fn empty(centrality_type: CentralityType) -> Self {
        Self {
            scores: HashMap::new(),
            centrality_type,
            iterations: None,
            converged: None,
            sample_count: None,
        }
    }

    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<(u64, f64)> {
        let mut sorted: Vec<_> = self.scores.iter().map(|(&id, &s)| (id, s)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));
        sorted.truncate(k);
        sorted
    }
}

/// Configuration for community detection algorithms.
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    pub direction: Direction,
    pub edge_type: Option<String>,
    pub resolution: f64,
    pub max_passes: usize,
    pub max_iterations: usize,
    pub seed: Option<u64>,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Both,
            edge_type: None,
            resolution: 1.0,
            max_passes: COMMUNITY_MAX_PASSES,
            max_iterations: LABEL_PROPAGATION_MAX_ITERATIONS,
            seed: None,
        }
    }
}

impl CommunityConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }

    #[must_use]
    pub const fn resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    #[must_use]
    pub const fn max_passes(mut self, max_passes: usize) -> Self {
        self.max_passes = max_passes;
        self
    }

    #[must_use]
    pub const fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of community detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityResult {
    pub communities: HashMap<u64, u64>,
    pub members: HashMap<u64, Vec<u64>>,
    pub community_count: usize,
    pub modularity: Option<f64>,
    pub passes: Option<usize>,
    pub iterations: Option<usize>,
}

impl CommunityResult {
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            communities: HashMap::new(),
            members: HashMap::new(),
            community_count: 0,
            modularity: None,
            passes: None,
            iterations: None,
        }
    }

    #[must_use]
    pub fn communities_by_size(&self) -> Vec<(u64, usize)> {
        let mut sizes: Vec<_> = self.members.iter().map(|(&c, m)| (c, m.len())).collect();
        sizes.sort_by(|a, b| b.1.cmp(&a.1));
        sizes
    }
}

impl Default for CommunityResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Union-Find data structure for connected components.
struct UnionFind {
    parent: HashMap<u64, u64>,
    rank: HashMap<u64, usize>,
}

impl UnionFind {
    fn new(nodes: &[u64]) -> Self {
        let parent = nodes.iter().map(|&n| (n, n)).collect();
        let rank = nodes.iter().map(|&n| (n, 0)).collect();
        Self { parent, rank }
    }

    fn find(&mut self, x: u64) -> u64 {
        let p = self.parent[&x];
        if p == x {
            x
        } else {
            let root = self.find(p);
            self.parent.insert(x, root);
            root
        }
    }

    fn union(&mut self, x: u64, y: u64) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            let rank_x = self.rank[&rx];
            let rank_y = self.rank[&ry];
            match rank_x.cmp(&rank_y) {
                CmpOrdering::Less => {
                    self.parent.insert(rx, ry);
                },
                CmpOrdering::Greater => {
                    self.parent.insert(ry, rx);
                },
                CmpOrdering::Equal => {
                    self.parent.insert(ry, rx);
                    self.rank.insert(rx, rank_x + 1);
                },
            }
        }
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

/// Type alias for the `BTreeMap` index structure.
type PropertyIndex = BTreeMap<OrderedPropertyValue, Vec<u64>>;

/// Compound index for multi-property lookups.
/// Key is a vector of ordered property values matching the index definition order.
type CompoundIndex = BTreeMap<Vec<OrderedPropertyValue>, Vec<u64>>;

/// Lock ordering (acquire in this order to prevent deadlocks):
/// 1. `batch_unique_lock` - Serializes unique constraint operations
/// 2. `constraints` - Constraint definitions (`RwLock`)
/// 3. `btree_indexes` - In-memory indexes (`RwLock`)
/// 4. Store operations - Internal `TensorStore` locking
pub struct GraphEngine {
    store: TensorStore,
    node_counter: AtomicU64,
    edge_counter: AtomicU64,
    /// In-memory `BTreeMap` indexes for O(log n) property lookups.
    btree_indexes: RwLock<HashMap<(IndexTarget, String), PropertyIndex>>,
    /// Compound indexes for multi-property lookups.
    compound_indexes: RwLock<HashMap<Vec<String>, CompoundIndex>>,
    /// Full-text indexes for text search.
    fulltext_indexes: RwLock<HashMap<String, fulltext::FullTextIndex>>,
    /// Geospatial indexes for location queries.
    geo_indexes: RwLock<HashMap<String, geo::GeoIndex>>,
    /// Striped locks for concurrent index updates.
    index_locks: Vec<RwLock<()>>,
    /// Whether the label index has been initialized (for lazy auto-creation).
    label_index_initialized: AtomicBool,
    /// Whether the edge type index has been initialized (for lazy auto-creation).
    edge_type_index_initialized: AtomicBool,
    /// Cached constraints for fast validation.
    constraints: RwLock<HashMap<String, Constraint>>,
    /// Serializes batch operations involving unique constraints to prevent TOCTOU races.
    batch_unique_lock: RwLock<()>,
    /// Runtime configuration.
    config: GraphEngineConfig,
}

impl std::fmt::Debug for GraphEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let index_count = self.btree_indexes.read().len();
        let constraint_count = self.constraints.read().len();
        f.debug_struct("GraphEngine")
            .field("node_counter", &self.node_counter.load(Ordering::Relaxed))
            .field("edge_counter", &self.edge_counter.load(Ordering::Relaxed))
            .field("index_count", &index_count)
            .field("constraint_count", &constraint_count)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Creates vector of `RwLock`s for striped locking.
fn create_index_locks(count: usize) -> Vec<RwLock<()>> {
    (0..count).map(|_| RwLock::new(())).collect()
}

impl GraphEngine {
    const PARALLEL_THRESHOLD: usize = 100;
    const AGGREGATE_PARALLEL_THRESHOLD: usize = 1000;
    const MAX_UNPAGINATED_RESULTS: usize = 100_000;

    #[must_use]
    pub fn new() -> Self {
        Self::with_config(GraphEngineConfig::default())
    }

    #[must_use]
    pub fn with_config(config: GraphEngineConfig) -> Self {
        let lock_count = config.index_lock_count.max(1);
        Self {
            store: TensorStore::new(),
            node_counter: AtomicU64::new(0),
            edge_counter: AtomicU64::new(0),
            btree_indexes: RwLock::new(HashMap::new()),
            compound_indexes: RwLock::new(HashMap::new()),
            fulltext_indexes: RwLock::new(HashMap::new()),
            geo_indexes: RwLock::new(HashMap::new()),
            index_locks: create_index_locks(lock_count),
            label_index_initialized: AtomicBool::new(false),
            edge_type_index_initialized: AtomicBool::new(false),
            constraints: RwLock::new(HashMap::new()),
            batch_unique_lock: RwLock::new(()),
            config,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &GraphEngineConfig {
        &self.config
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

        // Load existing constraints from store
        let constraints = Self::load_constraints_from_store(&store);

        let config = GraphEngineConfig::default();
        Self {
            store,
            node_counter: AtomicU64::new(max_node_id),
            edge_counter: AtomicU64::new(max_edge_id),
            btree_indexes: RwLock::new(btree_indexes),
            compound_indexes: RwLock::new(HashMap::new()),
            fulltext_indexes: RwLock::new(HashMap::new()),
            geo_indexes: RwLock::new(HashMap::new()),
            index_locks: create_index_locks(config.index_lock_count),
            label_index_initialized: AtomicBool::new(label_index_exists),
            edge_type_index_initialized: AtomicBool::new(edge_type_index_exists),
            constraints: RwLock::new(constraints),
            batch_unique_lock: RwLock::new(()),
            config,
        }
    }

    fn load_constraints_from_store(store: &TensorStore) -> HashMap<String, Constraint> {
        let mut constraints = HashMap::new();
        for key in store.scan("_graph_constraint:") {
            if let Ok(tensor) = store.get(&key) {
                let name = Self::extract_string(&tensor, "name").unwrap_or_default();
                let target_json = Self::extract_string(&tensor, "target").unwrap_or_default();
                let property = Self::extract_string(&tensor, "property").unwrap_or_default();
                let type_json =
                    Self::extract_string(&tensor, "constraint_type").unwrap_or_default();

                if let (Ok(target), Ok(constraint_type)) = (
                    serde_json::from_str::<ConstraintTarget>(&target_json),
                    serde_json::from_str::<ConstraintType>(&type_json),
                ) {
                    constraints.insert(
                        name.clone(),
                        Constraint {
                            name,
                            target,
                            property,
                            constraint_type,
                        },
                    );
                }
            }
        }
        constraints
    }

    fn extract_string(tensor: &TensorData, key: &str) -> Option<String> {
        match tensor.get(key) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.clone()),
            _ => None,
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
    const fn lock_index(&self, id: u64) -> usize {
        (id as usize) % self.index_locks.len()
    }

    // ========== Index CRUD Methods ==========

    /// Create an index on a node property for O(log n) lookups.
    ///
    fn validate_property_name(name: &str) -> Result<()> {
        if name.contains(':') {
            return Err(GraphError::InvalidPropertyName {
                name: name.to_string(),
            });
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns `IndexAlreadyExists` if an index already exists for this property.
    /// Returns `InvalidPropertyName` if the property name contains ':'.
    pub fn create_node_property_index(&self, property: &str) -> Result<()> {
        Self::validate_property_name(property)?;
        self.create_property_index(IndexTarget::Node, property)
    }

    /// Create an index on an edge property for O(log n) lookups.
    ///
    /// # Errors
    ///
    /// Returns `IndexAlreadyExists` if an index already exists for this property.
    /// Returns `InvalidPropertyName` if the property name contains ':'.
    pub fn create_edge_property_index(&self, property: &str) -> Result<()> {
        Self::validate_property_name(property)?;
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

    // ========== Compound Index Methods ==========

    /// Create a compound index on multiple node properties.
    ///
    /// Compound indexes enable efficient lookups when filtering on multiple properties
    /// simultaneously. Properties are indexed in the order provided.
    ///
    /// # Errors
    ///
    /// Returns an error if the index already exists or if property names are invalid.
    pub fn create_compound_index(&self, properties: &[&str]) -> Result<()> {
        if properties.is_empty() {
            return Ok(());
        }

        for prop in properties {
            Self::validate_property_name(prop)?;
        }

        let key: Vec<String> = properties.iter().map(|s| (*s).to_string()).collect();

        // Check if index already exists
        if self.compound_indexes.read().contains_key(&key) {
            return Err(GraphError::IndexAlreadyExists {
                target: "compound".to_string(),
                property: properties.join(","),
            });
        }

        // Build the index by scanning all nodes
        let mut btree: CompoundIndex = BTreeMap::new();

        for node_key in self.store.scan("node:") {
            if node_key.contains(":out") || node_key.contains(":in") {
                continue;
            }

            if let Some(id_str) = node_key.strip_prefix("node:") {
                if let Ok(node_id) = id_str.parse::<u64>() {
                    if let Ok(node) = self.get_node(node_id) {
                        let compound_key: Vec<OrderedPropertyValue> = properties
                            .iter()
                            .map(|p| {
                                node.properties
                                    .get(*p)
                                    .map_or(OrderedPropertyValue::Null, OrderedPropertyValue::from)
                            })
                            .collect();

                        btree.entry(compound_key).or_default().push(node_id);
                    }
                }
            }
        }

        self.compound_indexes.write().insert(key, btree);
        Ok(())
    }

    /// Find nodes by compound index lookup.
    ///
    /// Values must be provided in the same order as the index was created.
    ///
    /// # Errors
    ///
    /// Returns an error if the compound index doesn't exist.
    pub fn find_by_compound(&self, values: &[(&str, &PropertyValue)]) -> Result<Vec<Node>> {
        let key: Vec<String> = values.iter().map(|(k, _)| (*k).to_string()).collect();
        let compound_key: Vec<OrderedPropertyValue> = values
            .iter()
            .map(|(_, v)| OrderedPropertyValue::from(*v))
            .collect();

        let ids = self
            .compound_indexes
            .read()
            .get(&key)
            .ok_or_else(|| GraphError::IndexNotFound {
                target: "compound".to_string(),
                property: key.join(","),
            })?
            .get(&compound_key)
            .cloned()
            .unwrap_or_default();

        ids.into_iter().map(|id| self.get_node(id)).collect()
    }

    /// Check if a compound index exists for the given properties.
    #[must_use]
    pub fn has_compound_index(&self, properties: &[&str]) -> bool {
        let key: Vec<String> = properties.iter().map(|s| (*s).to_string()).collect();
        self.compound_indexes.read().contains_key(&key)
    }

    /// Drop a compound index.
    ///
    /// # Errors
    ///
    /// Returns an error if the compound index doesn't exist.
    pub fn drop_compound_index(&self, properties: &[&str]) -> Result<()> {
        let key: Vec<String> = properties.iter().map(|s| (*s).to_string()).collect();

        self.compound_indexes
            .write()
            .remove(&key)
            .map(|_| ())
            .ok_or_else(|| GraphError::IndexNotFound {
                target: "compound".to_string(),
                property: properties.join(","),
            })
    }

    /// Get list of compound indexes.
    #[must_use]
    pub fn get_compound_indexes(&self) -> Vec<Vec<String>> {
        self.compound_indexes.read().keys().cloned().collect()
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
        let _lock = self.index_locks[self.lock_index(id)].write();

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
        let _lock = self.index_locks[self.lock_index(id)].write();

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

    /// # Errors
    /// Returns `ConstraintViolation` if a unique/exists constraint is violated.
    /// Returns `StorageError` if the underlying store operation fails.
    #[instrument(skip(self, label, properties))]
    pub fn create_node(
        &self,
        label: impl Into<String>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<u64> {
        self.create_node_with_labels(vec![label.into()], properties)
    }

    /// # Errors
    /// Returns `ConstraintViolation` if a unique/exists constraint is violated.
    /// Returns `StorageError` if the underlying store operation fails.
    #[allow(clippy::needless_pass_by_value)] // ownership avoids caller clones
    #[instrument(skip(self, labels, properties))]
    pub fn create_node_with_labels(
        &self,
        labels: Vec<String>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<u64> {
        // Acquire lock to prevent TOCTOU race when unique constraints exist.
        let has_unique = self.has_any_unique_node_constraint();
        let _guard = if has_unique {
            Some(self.batch_unique_lock.write())
        } else {
            None
        };

        // Validate constraints before creating
        self.validate_node_constraints(&labels, &properties, None)?;

        // Ensure label index exists (lazy init on first node creation)
        self.ensure_label_index();

        let id = self.node_counter.fetch_add(1, Ordering::SeqCst) + 1;

        let mut tensor = TensorData::new();
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::Int(id.cast_signed())),
        );
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("node".into())),
        );
        tensor.set("_labels", TensorValue::Pointers(labels.clone()));
        tensor.set(
            "_created_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
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

    /// # Errors
    /// Returns `NodeNotFound` if source or target node doesn't exist.
    /// Returns `ConstraintViolation` if a unique/exists constraint is violated.
    /// Returns `StorageError` if the underlying store operation fails.
    #[allow(clippy::needless_pass_by_value)] // ownership avoids caller clones
    #[instrument(skip(self, edge_type, properties), fields(from = from, to = to))]
    pub fn create_edge(
        &self,
        from: u64,
        to: u64,
        edge_type: impl Into<String>,
        properties: HashMap<String, PropertyValue>,
        directed: bool,
    ) -> Result<u64> {
        let edge_type = edge_type.into();

        // Acquire lock to prevent TOCTOU race when unique constraints exist.
        let has_unique = self.has_any_unique_edge_constraint();
        let _guard = if has_unique {
            Some(self.batch_unique_lock.write())
        } else {
            None
        };

        // Ensure edge type index exists (lazy init on first edge creation)
        self.ensure_edge_type_index();

        // Verify both nodes exist
        if !self.node_exists(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.node_exists(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        // Validate constraints before creating
        self.validate_edge_constraints(&edge_type, &properties, None)?;

        let id = self.edge_counter.fetch_add(1, Ordering::SeqCst) + 1;

        let mut tensor = TensorData::new();
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::Int(id.cast_signed())),
        );
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("edge".into())),
        );
        tensor.set(
            "_from",
            TensorValue::Scalar(ScalarValue::Int(from.cast_signed())),
        );
        tensor.set(
            "_to",
            TensorValue::Scalar(ScalarValue::Int(to.cast_signed())),
        );
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
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
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

        let edge_key = format!("e{edge_id}");
        tensor.set(
            &edge_key,
            TensorValue::Scalar(ScalarValue::Int(edge_id.cast_signed())),
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
                    edges.push((*id).cast_unsigned());
                }
            }
        }
        edges
    }

    pub fn node_exists(&self, id: u64) -> bool {
        self.store.exists(&Self::node_key(id))
    }

    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some((*v).cast_unsigned()),
            _ => None,
        };
        let updated_at = match tensor.get("_updated_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some((*v).cast_unsigned()),
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

    /// # Errors
    /// Returns `EdgeNotFound` if the edge doesn't exist.
    /// Returns `CorruptedEdge` if required fields are missing.
    pub fn get_edge(&self, id: u64) -> Result<Edge> {
        let tensor = self
            .store
            .get(&Self::edge_key(id))
            .map_err(|_| GraphError::EdgeNotFound(id))?;

        let from = match tensor.get("_from") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => (*v).cast_unsigned(),
            _ => {
                return Err(GraphError::CorruptedEdge {
                    edge_id: id,
                    field: "_from".to_string(),
                })
            },
        };

        let to = match tensor.get("_to") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => (*v).cast_unsigned(),
            _ => {
                return Err(GraphError::CorruptedEdge {
                    edge_id: id,
                    field: "_to".to_string(),
                })
            },
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
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some((*v).cast_unsigned()),
            _ => None,
        };
        let updated_at = match tensor.get("_updated_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some((*v).cast_unsigned()),
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
    /// Returns `ConstraintViolation` if the update violates a constraint.
    #[allow(clippy::needless_pass_by_value)] // ownership avoids caller clones
    pub fn update_node(
        &self,
        id: u64,
        labels: Option<Vec<String>>,
        properties: HashMap<String, PropertyValue>,
    ) -> Result<()> {
        // Get old node for index maintenance
        let old_node = self.get_node(id)?;

        // Validate constraints before updating (exclude self from unique checks)
        let effective_labels = labels.as_ref().unwrap_or(&old_node.labels);
        self.validate_node_constraints(effective_labels, &properties, Some(id))?;

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
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
    pub fn get_node_labels(&self, id: u64) -> Result<Vec<String>> {
        let node = self.get_node(id)?;
        Ok(node.labels)
    }

    /// Check if a node has a specific label.
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
    pub fn node_has_label(&self, id: u64, label: &str) -> Result<bool> {
        let node = self.get_node(id)?;
        Ok(node.has_label(label))
    }

    /// Update an edge's properties.
    ///
    /// Properties are merged with existing properties; pass `PropertyValue::Null`
    /// to remove a property. The edge type, from/to nodes, and directedness cannot
    /// be changed after creation.
    ///
    /// # Errors
    /// Returns `EdgeNotFound` if the edge doesn't exist.
    #[allow(clippy::needless_pass_by_value)] // ownership avoids caller clones
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
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
    pub fn out_degree(&self, node_id: u64) -> Result<usize> {
        if !self.node_exists(node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        Ok(self.get_edge_list(&Self::outgoing_edges_key(node_id)).len())
    }

    /// Returns the number of incoming edges to a node.
    ///
    /// Equivalent to Neo4j's `size((n)<--())`.
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
    pub fn degree_by_type(&self, node_id: u64, edge_type: &str) -> Result<usize> {
        let out = self.out_degree_by_type(node_id, edge_type)?;
        let in_ = self.in_degree_by_type(node_id, edge_type)?;
        Ok(out + in_)
    }

    /// Returns all nodes in the graph, sorted by ID.
    ///
    /// Equivalent to Neo4j's `MATCH (n) RETURN n`.
    ///
    /// **Warning:** For large graphs, use `all_nodes_paginated()` instead.
    /// This method has a safety limit to prevent OOM.
    pub fn all_nodes(&self) -> Vec<Node> {
        let mut nodes = Vec::new();
        for key in self.store.scan("node:") {
            if nodes.len() >= Self::MAX_UNPAGINATED_RESULTS {
                warn!(
                    limit = Self::MAX_UNPAGINATED_RESULTS,
                    "all_nodes() hit safety limit, use all_nodes_paginated() for large graphs"
                );
                break;
            }
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

    /// Returns all node IDs in the graph.
    ///
    /// More efficient than `all_nodes()` when only IDs are needed.
    ///
    /// # Errors
    ///
    /// Returns an error on internal failure (should not happen in practice).
    pub fn get_all_node_ids(&self) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        for key in self.store.scan("node:") {
            if key.contains(":out") || key.contains(":in") {
                continue;
            }
            if let Some(id_str) = key.strip_prefix("node:") {
                if let Ok(id) = id_str.parse::<u64>() {
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// Returns all edges in the graph, sorted by ID.
    ///
    /// Equivalent to Neo4j's `MATCH ()-[r]->() RETURN r`.
    ///
    /// **Warning:** For large graphs, use `all_edges_paginated()` instead.
    /// This method has a safety limit to prevent OOM.
    pub fn all_edges(&self) -> Vec<Edge> {
        let mut edges = Vec::new();
        for key in self.store.scan("edge:") {
            if edges.len() >= Self::MAX_UNPAGINATED_RESULTS {
                warn!(
                    limit = Self::MAX_UNPAGINATED_RESULTS,
                    "all_edges() hit safety limit, use all_edges_paginated() for large graphs"
                );
                break;
            }
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

    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist.
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
    #[allow(clippy::option_if_let_else)]
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
    #[allow(clippy::option_if_let_else)]
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
    #[allow(clippy::option_if_let_else)]
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
    #[allow(clippy::option_if_let_else)]
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
    #[allow(clippy::option_if_let_else)]
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

    /// # Errors
    /// Returns `NodeNotFound` if the start node doesn't exist.
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

    /// # Errors
    /// Returns `NodeNotFound` if either node doesn't exist.
    /// Returns `PathNotFound` if no path exists between the nodes.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if either node doesn't exist.
    /// Returns `NegativeWeight` if an edge has a negative weight.
    /// Returns `PathNotFound` if no path exists.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if either node doesn't exist.
    /// Returns `PathNotFound` if no path exists.
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
    ///
    /// # Errors
    /// Returns `NodeNotFound` if either node doesn't exist.
    /// Returns `NegativeWeight` if an edge has a negative weight.
    /// Returns `PathNotFound` if no path exists.
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
    ///
    /// Uses iterative deepening DFS with backtracking for O(D) memory instead of O(B^D).
    ///
    /// # Errors
    /// Returns `NodeNotFound` if either node doesn't exist.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::needless_pass_by_value)] // config is consumed during path finding
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
        let max_memory = self.config.max_path_search_memory_bytes;
        let mut memory_used = 0usize;

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

        let edge_type_set: Option<HashSet<&str>> = config
            .edge_types
            .as_ref()
            .map(|types| types.iter().map(String::as_str).collect());

        // Iterative deepening: search depth by depth to find shorter paths first
        // and to avoid cloning paths per branch (backtracking approach)
        for target_depth in config.min_hops.max(1)..=config.max_hops {
            if paths.len() >= config.max_paths || memory_used > max_memory {
                stats.truncated = true;
                break;
            }

            // Use backtracking DFS for this depth level
            let mut path_nodes = Vec::with_capacity(target_depth + 1);
            let mut path_edges = Vec::with_capacity(target_depth);
            let mut visited = HashSet::new();

            path_nodes.push(from);
            if !config.allow_cycles {
                visited.insert(from);
            }

            self.find_paths_dfs_backtrack(
                from,
                to,
                target_depth,
                &config,
                edge_type_set.as_ref(),
                &mut path_nodes,
                &mut path_edges,
                &mut visited,
                &mut paths,
                &mut stats,
                &mut memory_used,
                max_memory,
            );
        }

        Ok(VariableLengthPaths { paths, stats })
    }

    /// DFS with backtracking for memory-efficient path finding.
    /// Uses O(D) memory by modifying vectors in place rather than cloning.
    #[allow(clippy::too_many_arguments)]
    fn find_paths_dfs_backtrack(
        &self,
        current: u64,
        to: u64,
        target_depth: usize,
        config: &VariableLengthConfig,
        edge_type_set: Option<&HashSet<&str>>,
        path_nodes: &mut Vec<u64>,
        path_edges: &mut Vec<u64>,
        visited: &mut HashSet<u64>,
        paths: &mut Vec<Path>,
        stats: &mut PathSearchStats,
        memory_used: &mut usize,
        max_memory: usize,
    ) {
        // Check termination conditions
        if paths.len() >= config.max_paths || *memory_used > max_memory {
            stats.truncated = true;
            return;
        }

        let current_depth = path_edges.len();

        // At target depth, check if we reached destination
        if current_depth == target_depth {
            if current == to {
                // Clone only when we find a valid path
                let path_memory =
                    (path_nodes.len() + path_edges.len()) * std::mem::size_of::<u64>();
                *memory_used += path_memory;

                paths.push(Path {
                    nodes: path_nodes.clone(),
                    edges: path_edges.clone(),
                });
                stats.paths_found += 1;
                stats.min_length = Some(
                    stats
                        .min_length
                        .map_or(target_depth, |m| m.min(target_depth)),
                );
                stats.max_length = Some(
                    stats
                        .max_length
                        .map_or(target_depth, |m| m.max(target_depth)),
                );
            }
            return;
        }

        stats.nodes_explored += 1;

        // Get neighbors
        let neighbors = self.get_variable_path_neighbors_filtered(
            current,
            edge_type_set,
            config.direction,
            config.filter.as_ref(),
        );

        for (neighbor_id, edge_id) in neighbors {
            stats.edges_traversed += 1;

            // Skip if already visited (unless cycles allowed)
            if !config.allow_cycles && visited.contains(&neighbor_id) {
                continue;
            }

            // Apply node filter (except for destination)
            if neighbor_id != to {
                if let Some(ref f) = config.filter {
                    if let Ok(node) = self.get_node(neighbor_id) {
                        if !f.matches_node(&node) {
                            continue;
                        }
                    }
                }
            }

            // Push onto path (backtracking)
            path_nodes.push(neighbor_id);
            path_edges.push(edge_id);
            if !config.allow_cycles {
                visited.insert(neighbor_id);
            }

            // Recurse
            self.find_paths_dfs_backtrack(
                neighbor_id,
                to,
                target_depth,
                config,
                edge_type_set,
                path_nodes,
                path_edges,
                visited,
                paths,
                stats,
                memory_used,
                max_memory,
            );

            // Backtrack
            path_nodes.pop();
            path_edges.pop();
            if !config.allow_cycles {
                visited.remove(&neighbor_id);
            }
        }
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
    ///
    /// # Errors
    /// Returns an error if the label lookup fails.
    pub fn count_nodes_by_label(&self, label: &str) -> Result<u64> {
        Ok(self.find_nodes_by_label(label)?.len() as u64)
    }

    /// Returns the total count of edges.
    #[must_use]
    pub fn count_edges(&self) -> u64 {
        self.edge_count() as u64
    }

    /// Returns the count of edges with a specific type.
    ///
    /// # Errors
    /// Returns an error if the edge type lookup fails.
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
    ///
    /// # Errors
    /// Returns an error if the label lookup fails.
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
    ///
    /// # Errors
    /// Returns an error if the filter condition lookup fails.
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
    ///
    /// # Errors
    /// Returns an error if the edge type lookup fails.
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
    ///
    /// # Errors
    /// Returns an error if the filter condition lookup fails.
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
            PropertyValue::DateTime(_) => 5,
            PropertyValue::List(_) => 6,
            PropertyValue::Map(_) => 7,
            PropertyValue::Bytes(_) => 8,
            PropertyValue::Point { .. } => 9,
        }
    }

    // ========== Pattern Matching Methods ==========

    /// Match a pattern against the graph.
    ///
    /// This method implements declarative pattern matching similar to Neo4j's MATCH clause.
    /// It supports:
    /// - Node patterns with label and property filters
    /// - Edge patterns with type, direction, and property filters
    /// - Variable-length edge patterns (e.g., *1..5)
    /// - Variable bindings for extracting matched elements
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    #[instrument(skip(self, pattern))]
    pub fn match_pattern(&self, pattern: &Pattern) -> Result<PatternMatchResult> {
        let limit = pattern.limit.unwrap_or(self.config.default_match_limit);
        let mut stats = PatternMatchStats::default();
        let mut matches = Vec::new();

        // Get candidates for the first node pattern
        let candidates = self.find_pattern_candidates(pattern.path.node_pattern_at(0))?;

        // Check if we should use parallel processing
        let use_parallel = candidates.len() >= self.config.pattern_parallel_threshold;

        if use_parallel {
            // Parallel processing for large candidate sets
            let collected: Vec<_> = candidates
                .into_par_iter()
                .filter_map(|start_node| {
                    let mut local_stats = PatternMatchStats::default();
                    let mut local_matches = Vec::new();
                    local_stats.nodes_evaluated += 1;

                    let mut bindings = HashMap::new();
                    if let Some(var) = pattern
                        .path
                        .node_pattern_at(0)
                        .and_then(|np| np.variable.clone())
                    {
                        bindings.insert(var, Binding::Node(start_node.clone()));
                    }

                    self.dfs_extend_match(
                        &pattern.path,
                        0,
                        start_node.id,
                        bindings,
                        &mut local_matches,
                        &mut local_stats,
                        limit,
                    )
                    .ok()?;

                    Some((local_matches, local_stats))
                })
                .collect();

            // Merge results
            for (local_matches, local_stats) in collected {
                if matches.len() >= limit {
                    stats.truncated = true;
                    break;
                }
                let remaining = limit - matches.len();
                let to_add = local_matches.len().min(remaining);
                matches.extend(local_matches.into_iter().take(to_add));
                stats.nodes_evaluated += local_stats.nodes_evaluated;
                stats.edges_evaluated += local_stats.edges_evaluated;
            }
        } else {
            // Sequential processing for small candidate sets
            for start_node in candidates {
                if matches.len() >= limit {
                    stats.truncated = true;
                    break;
                }
                stats.nodes_evaluated += 1;

                let mut bindings = HashMap::new();
                if let Some(var) = pattern
                    .path
                    .node_pattern_at(0)
                    .and_then(|np| np.variable.clone())
                {
                    bindings.insert(var, Binding::Node(start_node.clone()));
                }

                self.dfs_extend_match(
                    &pattern.path,
                    0,
                    start_node.id,
                    bindings,
                    &mut matches,
                    &mut stats,
                    limit,
                )?;
            }
        }

        stats.matches_found = matches.len();
        Ok(PatternMatchResult { matches, stats })
    }

    /// Convenience method for matching a simple two-node pattern.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn match_simple(
        &self,
        start: NodePattern,
        edge: EdgePattern,
        end: NodePattern,
    ) -> Result<PatternMatchResult> {
        let path = PathPattern::new(start, edge, end);
        self.match_pattern(&Pattern::new(path))
    }

    /// Count the number of matches for a pattern without returning bindings.
    ///
    /// This is more efficient than `match_pattern` when only the count is needed.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn count_pattern_matches(&self, pattern: &Pattern) -> Result<u64> {
        // Use a high limit to count all matches
        let mut counting_pattern = pattern.clone();
        counting_pattern.limit = Some(usize::MAX);
        let result = self.match_pattern(&counting_pattern)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(result.matches.len() as u64)
    }

    /// Check if any match exists for a pattern.
    ///
    /// Short-circuits as soon as one match is found.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the underlying store operation fails.
    pub fn pattern_exists(&self, pattern: &Pattern) -> Result<bool> {
        let mut single_pattern = pattern.clone();
        single_pattern.limit = Some(1);
        let result = self.match_pattern(&single_pattern)?;
        Ok(!result.matches.is_empty())
    }

    /// Find candidate nodes for a node pattern using indexes.
    fn find_pattern_candidates(&self, np: Option<&NodePattern>) -> Result<Vec<Node>> {
        let Some(np) = np else {
            return Ok(Vec::new());
        };

        // If we have a label, use the label index
        if let Some(ref label) = np.label {
            self.ensure_label_index();
            let mut nodes = self.find_nodes_by_label(label)?;
            // Apply additional property conditions
            if !np.conditions.is_empty() {
                nodes.retain(|n| np.conditions.iter().all(|c| c.matches_node(n)));
            }
            return Ok(nodes);
        }

        // If we have property conditions, try to use property indexes
        for cond in &np.conditions {
            if cond.op == CompareOp::Eq {
                if let Ok(mut nodes) = self.find_nodes_by_property(&cond.property, &cond.value) {
                    // Apply remaining conditions
                    nodes.retain(|n| np.matches(n));
                    return Ok(nodes);
                }
            }
        }

        // Fallback to scanning all nodes
        let mut nodes = self.all_nodes();
        nodes.retain(|n| np.matches(n));
        Ok(nodes)
    }

    /// Get neighbors matching an edge pattern.
    fn get_pattern_neighbors(&self, node_id: u64, ep: &EdgePattern) -> Vec<(Node, Edge)> {
        let mut results = Vec::new();

        // Get edge lists based on direction
        let (out_edges, in_edges) = match ep.direction {
            Direction::Outgoing => (
                self.get_edge_list(&Self::outgoing_edges_key(node_id)),
                vec![],
            ),
            Direction::Incoming => (
                vec![],
                self.get_edge_list(&Self::incoming_edges_key(node_id)),
            ),
            Direction::Both => (
                self.get_edge_list(&Self::outgoing_edges_key(node_id)),
                self.get_edge_list(&Self::incoming_edges_key(node_id)),
            ),
        };

        // Process outgoing edges
        for edge_id in out_edges {
            if let Ok(edge) = self.get_edge(edge_id) {
                if ep.matches(&edge) {
                    let neighbor_id = if edge.from == node_id {
                        edge.to
                    } else {
                        edge.from
                    };
                    if let Ok(neighbor) = self.get_node(neighbor_id) {
                        results.push((neighbor, edge));
                    }
                }
            }
        }

        // Process incoming edges
        for edge_id in in_edges {
            if let Ok(edge) = self.get_edge(edge_id) {
                if ep.matches(&edge) {
                    let neighbor_id = if edge.to == node_id {
                        edge.from
                    } else {
                        edge.to
                    };
                    if let Ok(neighbor) = self.get_node(neighbor_id) {
                        // Avoid duplicates for undirected edges
                        if !results.iter().any(|(n, _)| n.id == neighbor.id) {
                            results.push((neighbor, edge));
                        }
                    }
                }
            }
        }

        results
    }

    /// DFS extension of a partial match.
    #[allow(clippy::too_many_arguments)]
    fn dfs_extend_match(
        &self,
        path: &PathPattern,
        current_element_idx: usize,
        current_node_id: u64,
        bindings: HashMap<String, Binding>,
        matches: &mut Vec<PatternMatch>,
        stats: &mut PatternMatchStats,
        limit: usize,
    ) -> Result<()> {
        // Check if we've completed the pattern
        if current_element_idx >= path.elements.len() - 1 {
            matches.push(PatternMatch { bindings });
            return Ok(());
        }

        // Get the next edge and node patterns
        let edge_pattern = path.edge_pattern_at(current_element_idx + 1);
        let next_node_pattern = path.node_pattern_at(current_element_idx + 2);

        let Some(ep) = edge_pattern else {
            return Ok(());
        };
        let Some(np) = next_node_pattern else {
            return Ok(());
        };

        // Handle variable-length patterns
        if let Some(var_len) = ep.variable_length {
            return self.extend_variable_length_match(
                path,
                current_element_idx,
                current_node_id,
                bindings,
                matches,
                stats,
                limit,
                ep,
                np,
                var_len,
            );
        }

        // Get matching neighbors
        let neighbors = self.get_pattern_neighbors(current_node_id, ep);

        for (neighbor, edge) in neighbors {
            if matches.len() >= limit {
                return Ok(());
            }
            stats.edges_evaluated += 1;

            // Check if the neighbor matches the next node pattern
            if !np.matches(&neighbor) {
                continue;
            }
            stats.nodes_evaluated += 1;

            // Create new bindings
            let mut new_bindings = bindings.clone();
            if let Some(ref var) = ep.variable {
                new_bindings.insert(var.clone(), Binding::Edge(edge));
            }
            if let Some(ref var) = np.variable {
                new_bindings.insert(var.clone(), Binding::Node(neighbor.clone()));
            }

            // Recursively extend the match
            self.dfs_extend_match(
                path,
                current_element_idx + 2,
                neighbor.id,
                new_bindings,
                matches,
                stats,
                limit,
            )?;
        }

        Ok(())
    }

    /// Extend match for variable-length edge patterns.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::needless_pass_by_value)] // bindings is extended during recursion
    fn extend_variable_length_match(
        &self,
        path: &PathPattern,
        current_element_idx: usize,
        start_node_id: u64,
        bindings: HashMap<String, Binding>,
        matches: &mut Vec<PatternMatch>,
        stats: &mut PatternMatchStats,
        limit: usize,
        ep: &EdgePattern,
        end_np: &NodePattern,
        var_len: VariableLengthSpec,
    ) -> Result<()> {
        // Use DFS with explicit stack: (current_node_id, path_nodes, path_edges, depth, visited)
        // Uses Arc<HashSet> for copy-on-write to avoid O(B^D) memory explosion
        #[allow(clippy::type_complexity)]
        let mut stack: Vec<(u64, Vec<u64>, Vec<u64>, usize, Arc<HashSet<u64>>)> = Vec::new();

        let mut initial_visited = HashSet::new();
        initial_visited.insert(start_node_id);
        stack.push((
            start_node_id,
            vec![start_node_id],
            vec![],
            0,
            Arc::new(initial_visited),
        ));

        // Handle min_hops == 0 case (match without traversing)
        if var_len.min_hops == 0 {
            if let Ok(start_node) = self.get_node(start_node_id) {
                if end_np.matches(&start_node) {
                    let mut new_bindings = bindings.clone();
                    if let Some(ref var) = ep.variable {
                        new_bindings.insert(
                            var.clone(),
                            Binding::Path(Path {
                                nodes: vec![start_node_id],
                                edges: vec![],
                            }),
                        );
                    }
                    if let Some(ref var) = end_np.variable {
                        new_bindings.insert(var.clone(), Binding::Node(start_node));
                    }

                    // Continue to next part of pattern or complete match
                    if current_element_idx + 2 >= path.elements.len() - 1 {
                        matches.push(PatternMatch {
                            bindings: new_bindings,
                        });
                    } else {
                        self.dfs_extend_match(
                            path,
                            current_element_idx + 2,
                            start_node_id,
                            new_bindings,
                            matches,
                            stats,
                            limit,
                        )?;
                    }
                }
            }
        }

        while let Some((current_id, path_nodes, path_edges, depth, visited)) = stack.pop() {
            if matches.len() >= limit {
                return Ok(());
            }

            if depth >= var_len.max_hops {
                continue;
            }

            // Get neighbors matching the edge pattern (without variable length consideration)
            let single_hop_ep = EdgePattern {
                variable: None,
                edge_type: ep.edge_type.clone(),
                direction: ep.direction,
                conditions: ep.conditions.clone(),
                variable_length: None,
            };
            let neighbors = self.get_pattern_neighbors(current_id, &single_hop_ep);

            for (neighbor, edge) in neighbors {
                stats.edges_evaluated += 1;

                // Skip visited nodes to prevent cycles
                if visited.contains(&neighbor.id) {
                    continue;
                }

                let new_depth = depth + 1;
                let mut new_path_nodes = path_nodes.clone();
                let mut new_path_edges = path_edges.clone();
                new_path_nodes.push(neighbor.id);
                new_path_edges.push(edge.id);

                // Check if this is a valid end point
                if new_depth >= var_len.min_hops && end_np.matches(&neighbor) {
                    stats.nodes_evaluated += 1;

                    let mut new_bindings = bindings.clone();
                    if let Some(ref var) = ep.variable {
                        new_bindings.insert(
                            var.clone(),
                            Binding::Path(Path {
                                nodes: new_path_nodes.clone(),
                                edges: new_path_edges.clone(),
                            }),
                        );
                    }
                    if let Some(ref var) = end_np.variable {
                        new_bindings.insert(var.clone(), Binding::Node(neighbor.clone()));
                    }

                    // Continue to next part of pattern or complete match
                    if current_element_idx + 2 >= path.elements.len() - 1 {
                        matches.push(PatternMatch {
                            bindings: new_bindings,
                        });
                    } else {
                        self.dfs_extend_match(
                            path,
                            current_element_idx + 2,
                            neighbor.id,
                            new_bindings,
                            matches,
                            stats,
                            limit,
                        )?;
                    }
                }

                // Continue exploring if under max_hops
                if new_depth < var_len.max_hops {
                    let mut new_visited = Arc::clone(&visited);
                    Arc::make_mut(&mut new_visited).insert(neighbor.id);
                    stack.push((
                        neighbor.id,
                        new_path_nodes,
                        new_path_edges,
                        new_depth,
                        new_visited,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Delete an edge by ID, cleaning up edge lists on both connected nodes.
    ///
    /// # Errors
    /// Returns `EdgeNotFound` if the edge doesn't exist, or a storage error on failure.
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
            let edge_key = format!("e{edge_id}");
            tensor.remove(&edge_key);
            self.store.put(key, tensor)?;
        }
        Ok(())
    }

    /// # Errors
    /// Returns `NodeNotFound` if the node doesn't exist, or `PartialDeletionError`
    /// if some connected edges fail to delete.
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
            let failed_edges: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

            edges_to_delete.par_iter().for_each(|edge_id| {
                let result: Result<()> = (|| {
                    let edge = self.get_edge(*edge_id)?;
                    // Unindex edge properties
                    self.unindex_edge_properties(*edge_id, &edge.edge_type, &edge.properties);

                    // Clean up the OTHER node's edge list (not the one we're deleting)
                    let other_node = if edge.from == id { edge.to } else { edge.from };

                    if edge.from == id {
                        // We're the 'from' node, clean up 'to' node's incoming list
                        self.remove_edge_from_list(
                            &Self::incoming_edges_key(other_node),
                            *edge_id,
                        )?;
                    }
                    if edge.to == id {
                        // We're the 'to' node, clean up 'from' node's outgoing list
                        self.remove_edge_from_list(
                            &Self::outgoing_edges_key(other_node),
                            *edge_id,
                        )?;
                    }
                    // For undirected edges, also clean up reverse
                    if !edge.directed && other_node != id {
                        self.remove_edge_from_list(
                            &Self::outgoing_edges_key(other_node),
                            *edge_id,
                        )?;
                        self.remove_edge_from_list(
                            &Self::incoming_edges_key(other_node),
                            *edge_id,
                        )?;
                    }

                    // Delete the edge record
                    self.store.delete(&Self::edge_key(*edge_id))?;
                    Ok(())
                })();

                if result.is_err() {
                    if let Ok(mut guard) = failed_edges.lock() {
                        guard.push(*edge_id);
                    }
                }
            });

            let failed = failed_edges.into_inner().unwrap_or_default();
            if !failed.is_empty() {
                return Err(GraphError::PartialDeletionError {
                    node_id: id,
                    failed_edges: failed,
                });
            }
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
    // DEPRECATED: Use node ID-based API methods instead.

    /// Get or create an entity for graph operations.
    fn get_or_create_entity(&self, key: &str) -> TensorData {
        self.store.get(key).unwrap_or_else(|_| TensorData::new())
    }

    /// Add an outgoing edge to an entity's _out field.
    ///
    /// # Errors
    /// Returns a storage error if the edge or entity cannot be written.
    #[deprecated(since = "0.2.0", note = "Use node ID-based API instead")]
    pub fn add_entity_edge(&self, from_key: &str, to_key: &str, edge_type: &str) -> Result<String> {
        let edge_id = self.edge_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let edge_key = format!("edge:{edge_type}:{edge_id}");

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
    ///
    /// # Errors
    /// Returns a storage error if the edge or entities cannot be written.
    #[deprecated(since = "0.2.0", note = "Use node ID-based API instead")]
    pub fn add_entity_edge_undirected(
        &self,
        key1: &str,
        key2: &str,
        edge_type: &str,
    ) -> Result<String> {
        let edge_id = self.edge_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let edge_key = format!("edge:{edge_type}:{edge_id}");

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
    ///
    /// # Errors
    /// Returns a storage error if the entity is not found.
    #[deprecated(since = "0.2.0", note = "Use edges_of() instead")]
    pub fn get_entity_outgoing(&self, key: &str) -> Result<Vec<String>> {
        let entity = self
            .store
            .get(key)
            .map_err(|_| GraphError::StorageError(format!("Entity not found: {key}")))?;

        Ok(entity.outgoing_edges().cloned().unwrap_or_default())
    }

    /// Get incoming edge keys for an entity.
    ///
    /// # Errors
    /// Returns a storage error if the entity is not found.
    #[deprecated(since = "0.2.0", note = "Use edges_of() instead")]
    pub fn get_entity_incoming(&self, key: &str) -> Result<Vec<String>> {
        let entity = self
            .store
            .get(key)
            .map_err(|_| GraphError::StorageError(format!("Entity not found: {key}")))?;

        Ok(entity.incoming_edges().cloned().unwrap_or_default())
    }

    /// Get edge data by edge key.
    ///
    /// # Errors
    /// Returns a storage error if the edge is not found.
    #[deprecated(since = "0.2.0", note = "Use get_edge() instead")]
    pub fn get_entity_edge(&self, edge_key: &str) -> Result<(String, String, String, bool)> {
        let edge = self
            .store
            .get(edge_key)
            .map_err(|_| GraphError::StorageError(format!("Edge not found: {edge_key}")))?;

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
    ///
    /// # Errors
    /// Returns a storage error if the entity is not found.
    #[deprecated(
        since = "0.2.0",
        note = "Use neighbors() with Direction::Outgoing instead"
    )]
    #[allow(deprecated)]
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
    ///
    /// # Errors
    /// Returns a storage error if the entity is not found.
    #[deprecated(
        since = "0.2.0",
        note = "Use neighbors() with Direction::Incoming instead"
    )]
    #[allow(deprecated)]
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
    ///
    /// # Errors
    /// Returns a storage error if the entity is not found.
    #[deprecated(since = "0.2.0", note = "Use neighbors() with Direction::Both instead")]
    #[allow(deprecated)]
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
    #[deprecated(since = "0.2.0", note = "Use node ID-based API instead")]
    pub fn entity_has_edges(&self, key: &str) -> bool {
        self.store.get(key).map(|e| e.has_edges()).unwrap_or(false)
    }

    /// Delete an edge by key, updating connected entities.
    ///
    /// # Errors
    /// Returns a storage error if the edge is not found or cannot be deleted.
    #[deprecated(since = "0.2.0", note = "Use delete_edge() instead")]
    #[allow(deprecated)]
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
    #[deprecated(since = "0.2.0", note = "Use node ID-based API instead")]
    #[allow(deprecated)]
    pub fn scan_entities_with_edges(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| self.entity_has_edges(key))
            .collect()
    }

    // ========================================================================
    // Graph Algorithm Methods
    // ========================================================================

    fn all_node_ids(&self) -> Vec<u64> {
        self.all_nodes().into_iter().map(|n| n.id).collect()
    }

    fn algo_bfs_distances(
        &self,
        source: u64,
        direction: Direction,
        edge_type: Option<&str>,
    ) -> HashMap<u64, usize> {
        let mut distances = HashMap::new();
        distances.insert(source, 0);

        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[&current];
            for neighbor in self.get_neighbor_ids(current, edge_type, direction) {
                if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor) {
                    e.insert(current_dist + 1);
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }

    /// Find connected components using Union-Find.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    pub fn connected_components(&self, config: Option<CommunityConfig>) -> Result<CommunityResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();

        if nodes.is_empty() {
            return Ok(CommunityResult::empty());
        }

        let mut uf = UnionFind::new(&nodes);

        for edge in self.all_edges() {
            if let Some(ref et) = cfg.edge_type {
                if edge.edge_type != *et {
                    continue;
                }
            }
            uf.union(edge.from, edge.to);
        }

        let mut communities: HashMap<u64, u64> = HashMap::new();
        let mut members: HashMap<u64, Vec<u64>> = HashMap::new();

        for &node in &nodes {
            let root = uf.find(node);
            communities.insert(node, root);
            members.entry(root).or_default().push(node);
        }

        let community_count = members.len();

        Ok(CommunityResult {
            communities,
            members,
            community_count,
            modularity: None,
            passes: None,
            iterations: None,
        })
    }

    /// Compute `PageRank` scores using power iteration.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    #[allow(clippy::cast_precision_loss)]
    pub fn pagerank(&self, config: Option<PageRankConfig>) -> Result<PageRankResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(PageRankResult::empty());
        }

        let mut out_degree: HashMap<u64, usize> = HashMap::new();
        let mut in_neighbors: HashMap<u64, Vec<u64>> = HashMap::new();

        for &node in &nodes {
            let out_edges: Vec<Edge> = self
                .edges_of(node, cfg.direction)
                .unwrap_or_default()
                .into_iter()
                .filter(|e| {
                    cfg.edge_type.is_none() || cfg.edge_type.as_deref() == Some(&e.edge_type)
                })
                .collect();
            out_degree.insert(node, out_edges.len());
            for edge in out_edges {
                let target = if edge.from == node {
                    edge.to
                } else {
                    edge.from
                };
                in_neighbors.entry(target).or_default().push(node);
            }
        }

        let n_f64 = n as f64;
        let mut scores: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 1.0 / n_f64)).collect();
        let teleport = (1.0 - cfg.damping) / n_f64;

        let mut convergence = f64::MAX;
        let mut iterations = 0;

        for iter in 0..cfg.max_iterations {
            iterations = iter + 1;

            let dangling_sum: f64 = nodes
                .iter()
                .filter(|&&n| out_degree.get(&n).copied().unwrap_or(0) == 0)
                .map(|n| scores.get(n).copied().unwrap_or(0.0))
                .sum();
            let dangling_contrib = cfg.damping * dangling_sum / n_f64;

            let new_scores: HashMap<u64, f64> = if n >= self.config.centrality_parallel_threshold {
                nodes
                    .par_iter()
                    .map(|&node| {
                        let sum: f64 = in_neighbors.get(&node).map_or(0.0, |ins| {
                            ins.iter()
                                .map(|&src| {
                                    let deg = out_degree.get(&src).copied().unwrap_or(1).max(1);
                                    scores.get(&src).copied().unwrap_or(0.0) / deg as f64
                                })
                                .sum()
                        });
                        (node, cfg.damping.mul_add(sum, teleport + dangling_contrib))
                    })
                    .collect()
            } else {
                nodes
                    .iter()
                    .map(|&node| {
                        let sum: f64 = in_neighbors.get(&node).map_or(0.0, |ins| {
                            ins.iter()
                                .map(|&src| {
                                    let deg = out_degree.get(&src).copied().unwrap_or(1).max(1);
                                    scores.get(&src).copied().unwrap_or(0.0) / deg as f64
                                })
                                .sum()
                        });
                        (node, cfg.damping.mul_add(sum, teleport + dangling_contrib))
                    })
                    .collect()
            };

            convergence = nodes
                .iter()
                .map(|n| {
                    (new_scores.get(n).copied().unwrap_or(0.0)
                        - scores.get(n).copied().unwrap_or(0.0))
                    .abs()
                })
                .sum();

            scores = new_scores;

            if convergence < cfg.tolerance {
                return Ok(PageRankResult {
                    scores,
                    iterations,
                    convergence,
                    converged: true,
                });
            }
        }

        Ok(PageRankResult {
            scores,
            iterations,
            convergence,
            converged: false,
        })
    }

    /// Compute betweenness centrality using Brandes' algorithm.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn betweenness_centrality(
        &self,
        config: Option<CentralityConfig>,
    ) -> Result<CentralityResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(CentralityResult::empty(CentralityType::Betweenness));
        }

        #[allow(clippy::cast_sign_loss)] // sampling_ratio is clamped to [0.0, 1.0]
        let sample_count = ((n as f64) * cfg.sampling_ratio).ceil() as usize;
        let sources: Vec<u64> = if cfg.sampling_ratio < 1.0 && sample_count < n {
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            cfg.sampling_ratio.to_bits().hash(&mut hasher);
            let seed = hasher.finish();

            let mut indices: Vec<usize> = (0..n).collect();
            let mut lcg = seed;
            for i in (1..n).rev() {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let j = (lcg as usize) % (i + 1);
                indices.swap(i, j);
            }
            indices.truncate(sample_count);
            indices.into_iter().map(|i| nodes[i]).collect()
        } else {
            nodes.clone()
        };

        let partial_scores: Vec<HashMap<u64, f64>> =
            if n >= self.config.centrality_parallel_threshold {
                sources
                    .par_iter()
                    .map(|&source| {
                        self.brandes_single_source(
                            source,
                            &nodes,
                            cfg.direction,
                            cfg.edge_type.as_deref(),
                        )
                    })
                    .collect()
            } else {
                sources
                    .iter()
                    .map(|&source| {
                        self.brandes_single_source(
                            source,
                            &nodes,
                            cfg.direction,
                            cfg.edge_type.as_deref(),
                        )
                    })
                    .collect()
            };

        let mut scores: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();
        for partial in partial_scores {
            for (node, score) in partial {
                *scores.entry(node).or_insert(0.0) += score;
            }
        }

        if cfg.sampling_ratio < 1.0 {
            let scale = n as f64 / sample_count as f64;
            for score in scores.values_mut() {
                *score *= scale;
            }
        }

        for score in scores.values_mut() {
            *score /= 2.0;
        }

        Ok(CentralityResult {
            scores,
            centrality_type: CentralityType::Betweenness,
            iterations: None,
            converged: None,
            sample_count: Some(sources.len()),
        })
    }

    fn brandes_single_source(
        &self,
        source: u64,
        nodes: &[u64],
        direction: Direction,
        edge_type: Option<&str>,
    ) -> HashMap<u64, f64> {
        // Create node set for O(1) membership checks - neighbors may be outside sampling set
        let node_set: HashSet<u64> = nodes.iter().copied().collect();

        let mut sigma: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();
        let mut dist: HashMap<u64, i64> = nodes.iter().map(|&n| (n, -1)).collect();
        let mut pred: HashMap<u64, Vec<u64>> = nodes.iter().map(|&n| (n, Vec::new())).collect();

        sigma.insert(source, 1.0);
        dist.insert(source, 0);

        let mut queue = VecDeque::new();
        queue.push_back(source);
        let mut stack = Vec::new();

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let v_dist = dist.get(&v).copied().unwrap_or(-1);
            if v_dist < 0 {
                continue;
            }

            for neighbor in self.get_neighbor_ids(v, edge_type, direction) {
                // Skip neighbors not in our node set (sampling may exclude them)
                if !node_set.contains(&neighbor) {
                    continue;
                }

                let neighbor_dist = dist.get(&neighbor).copied().unwrap_or(-1);
                if neighbor_dist < 0 {
                    queue.push_back(neighbor);
                    dist.insert(neighbor, v_dist + 1);
                }
                let neighbor_dist = dist.get(&neighbor).copied().unwrap_or(-1);
                if neighbor_dist == v_dist + 1 {
                    let v_sigma = sigma.get(&v).copied().unwrap_or(0.0);
                    let n_sigma = sigma.get(&neighbor).copied().unwrap_or(0.0);
                    sigma.insert(neighbor, n_sigma + v_sigma);
                    if let Some(p) = pred.get_mut(&neighbor) {
                        p.push(v);
                    }
                }
            }
        }

        let mut delta: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();

        while let Some(w) = stack.pop() {
            if let Some(predecessors) = pred.get(&w) {
                for &v in predecessors {
                    let v_sigma = sigma.get(&v).copied().unwrap_or(0.0);
                    let w_sigma = sigma.get(&w).copied().unwrap_or(1.0);
                    let w_delta = delta.get(&w).copied().unwrap_or(0.0);

                    if w_sigma > 0.0 {
                        let contrib = (v_sigma / w_sigma) * (1.0 + w_delta);
                        let v_delta = delta.get(&v).copied().unwrap_or(0.0);
                        delta.insert(v, v_delta + contrib);
                    }
                }
            }
        }

        delta
    }

    /// Compute closeness centrality using harmonic variant.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    #[allow(clippy::cast_precision_loss)]
    pub fn closeness_centrality(
        &self,
        config: Option<CentralityConfig>,
    ) -> Result<CentralityResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(CentralityResult::empty(CentralityType::Closeness));
        }

        let scores: HashMap<u64, f64> = if n >= self.config.centrality_parallel_threshold {
            nodes
                .par_iter()
                .map(|&node| {
                    let distances =
                        self.algo_bfs_distances(node, cfg.direction, cfg.edge_type.as_deref());
                    let harmonic_sum: f64 = distances
                        .iter()
                        .filter(|(&n, _)| n != node)
                        .map(|(_, &d)| if d > 0 { 1.0 / d as f64 } else { 0.0 })
                        .sum();
                    (node, harmonic_sum / (n - 1).max(1) as f64)
                })
                .collect()
        } else {
            nodes
                .iter()
                .map(|&node| {
                    let distances =
                        self.algo_bfs_distances(node, cfg.direction, cfg.edge_type.as_deref());
                    let harmonic_sum: f64 = distances
                        .iter()
                        .filter(|(&n, _)| n != node)
                        .map(|(_, &d)| if d > 0 { 1.0 / d as f64 } else { 0.0 })
                        .sum();
                    (node, harmonic_sum / (n - 1).max(1) as f64)
                })
                .collect()
        };

        Ok(CentralityResult {
            scores,
            centrality_type: CentralityType::Closeness,
            iterations: None,
            converged: None,
            sample_count: None,
        })
    }

    /// Compute eigenvector centrality using power iteration.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    #[allow(clippy::cast_precision_loss)]
    pub fn eigenvector_centrality(
        &self,
        config: Option<CentralityConfig>,
    ) -> Result<CentralityResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(CentralityResult::empty(CentralityType::Eigenvector));
        }

        let n_f64 = n as f64;
        let mut scores: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 1.0 / n_f64)).collect();
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..cfg.max_iterations {
            iterations = iter + 1;

            let new_scores: HashMap<u64, f64> = if n >= self.config.centrality_parallel_threshold {
                nodes
                    .par_iter()
                    .map(|&node| {
                        let sum: f64 = self
                            .get_neighbor_ids(node, cfg.edge_type.as_deref(), cfg.direction)
                            .iter()
                            .map(|&neighbor| scores.get(&neighbor).copied().unwrap_or(0.0))
                            .sum();
                        (node, sum)
                    })
                    .collect()
            } else {
                nodes
                    .iter()
                    .map(|&node| {
                        let sum: f64 = self
                            .get_neighbor_ids(node, cfg.edge_type.as_deref(), cfg.direction)
                            .iter()
                            .map(|&neighbor| scores.get(&neighbor).copied().unwrap_or(0.0))
                            .sum();
                        (node, sum)
                    })
                    .collect()
            };

            let norm: f64 = new_scores.values().map(|&v| v * v).sum::<f64>().sqrt();
            let norm = if norm > 0.0 { norm } else { 1.0 };

            let normalized: HashMap<u64, f64> =
                new_scores.into_iter().map(|(n, v)| (n, v / norm)).collect();

            let delta: f64 = nodes
                .iter()
                .map(|n| {
                    (normalized.get(n).copied().unwrap_or(0.0)
                        - scores.get(n).copied().unwrap_or(0.0))
                    .abs()
                })
                .sum();

            scores = normalized;

            if delta < cfg.tolerance {
                converged = true;
                break;
            }
        }

        Ok(CentralityResult {
            scores,
            centrality_type: CentralityType::Eigenvector,
            iterations: Some(iterations),
            converged: Some(converged),
            sample_count: None,
        })
    }

    /// Detect communities using label propagation.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    #[allow(clippy::cast_possible_truncation)]
    pub fn label_propagation(&self, config: Option<CommunityConfig>) -> Result<CommunityResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(CommunityResult::empty());
        }

        let mut labels: HashMap<u64, u64> = nodes.iter().map(|&n| (n, n)).collect();

        let mut order: Vec<u64> = nodes;
        let seed = cfg.seed.unwrap_or(42);
        let mut lcg = seed;
        for i in (1..n).rev() {
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let j = (lcg as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut iterations = 0;

        for iter in 0..cfg.max_iterations {
            iterations = iter + 1;
            let mut changed = false;

            for &node in &order {
                let neighbors =
                    self.get_neighbor_ids(node, cfg.edge_type.as_deref(), cfg.direction);
                if neighbors.is_empty() {
                    continue;
                }

                let mut label_counts: HashMap<u64, usize> = HashMap::new();
                for &neighbor in &neighbors {
                    let label = labels[&neighbor];
                    *label_counts.entry(label).or_insert(0) += 1;
                }

                let max_count = label_counts.values().copied().max().unwrap_or(0);
                let mut candidates: Vec<u64> = label_counts
                    .into_iter()
                    .filter(|&(_, c)| c == max_count)
                    .map(|(l, _)| l)
                    .collect();
                candidates.sort_unstable();

                let new_label = candidates[0];
                if labels[&node] != new_label {
                    labels.insert(node, new_label);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let mut members: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&node, &label) in &labels {
            members.entry(label).or_default().push(node);
        }

        let community_count = members.len();

        Ok(CommunityResult {
            communities: labels,
            members,
            community_count,
            modularity: None,
            passes: None,
            iterations: Some(iterations),
        })
    }

    /// Detect communities using Louvain algorithm.
    ///
    /// # Errors
    /// This method currently doesn't fail, but returns `Result` for API consistency.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn louvain_communities(&self, config: Option<CommunityConfig>) -> Result<CommunityResult> {
        let cfg = config.unwrap_or_default();
        let nodes = self.all_node_ids();
        let n = nodes.len();

        if n == 0 {
            return Ok(CommunityResult::empty());
        }

        let mut adj: HashMap<u64, HashMap<u64, f64>> = HashMap::new();
        let mut total_weight = 0.0;

        for edge in self.all_edges() {
            if let Some(ref et) = cfg.edge_type {
                if edge.edge_type != *et {
                    continue;
                }
            }
            let weight = 1.0;
            total_weight += weight;

            adj.entry(edge.from)
                .or_default()
                .entry(edge.to)
                .and_modify(|w| *w += weight)
                .or_insert(weight);

            if !edge.directed {
                adj.entry(edge.to)
                    .or_default()
                    .entry(edge.from)
                    .and_modify(|w| *w += weight)
                    .or_insert(weight);
            }
        }

        if total_weight == 0.0 {
            let communities: HashMap<u64, u64> = nodes.iter().map(|&n| (n, n)).collect();
            let members: HashMap<u64, Vec<u64>> = nodes.iter().map(|&n| (n, vec![n])).collect();
            return Ok(CommunityResult {
                communities,
                members,
                community_count: n,
                modularity: Some(0.0),
                passes: Some(0),
                iterations: None,
            });
        }

        let mut communities: HashMap<u64, u64> = nodes.iter().map(|&n| (n, n)).collect();
        let mut passes = 0;

        for pass in 0..cfg.max_passes {
            passes = pass + 1;
            let mut improved = false;

            let mut order: Vec<u64> = nodes.clone();
            let seed = cfg.seed.unwrap_or(42).wrapping_add(pass as u64);
            let mut lcg = seed;
            for i in (1..n).rev() {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let j = (lcg as usize) % (i + 1);
                order.swap(i, j);
            }

            for &node in &order {
                let current_comm = communities[&node];

                let mut comm_weights: HashMap<u64, f64> = HashMap::new();
                if let Some(neighbors) = adj.get(&node) {
                    for (&neighbor, &weight) in neighbors {
                        let neighbor_comm = communities[&neighbor];
                        *comm_weights.entry(neighbor_comm).or_insert(0.0) += weight;
                    }
                }

                let mut best_comm = current_comm;
                let mut best_delta = 0.0;

                for (&comm, &weight_to_comm) in &comm_weights {
                    if comm == current_comm {
                        continue;
                    }
                    let delta =
                        weight_to_comm - comm_weights.get(&current_comm).copied().unwrap_or(0.0);
                    let delta = delta * cfg.resolution;

                    if delta > best_delta {
                        best_delta = delta;
                        best_comm = comm;
                    }
                }

                if best_comm != current_comm {
                    communities.insert(node, best_comm);
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        let modularity = self.compute_modularity_from_adj(&communities, &adj, total_weight);

        let mut members: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&node, &comm) in &communities {
            members.entry(comm).or_default().push(node);
        }

        let community_count = members.len();

        Ok(CommunityResult {
            communities,
            members,
            community_count,
            modularity: Some(modularity),
            passes: Some(passes),
            iterations: None,
        })
    }

    #[allow(clippy::unused_self)]
    fn compute_modularity_from_adj(
        &self,
        communities: &HashMap<u64, u64>,
        adj: &HashMap<u64, HashMap<u64, f64>>,
        total_weight: f64,
    ) -> f64 {
        if total_weight == 0.0 {
            return 0.0;
        }

        let mut degree: HashMap<u64, f64> = HashMap::new();
        for (node, neighbors) in adj {
            let sum: f64 = neighbors.values().sum();
            degree.insert(*node, sum);
        }

        let mut q = 0.0;
        let m2 = 2.0 * total_weight;

        for (node, neighbors) in adj {
            let node_comm = communities.get(node).copied().unwrap_or(*node);
            let node_deg = degree.get(node).copied().unwrap_or(0.0);

            for (&neighbor, &weight) in neighbors {
                let neighbor_comm = communities.get(&neighbor).copied().unwrap_or(neighbor);
                if node_comm == neighbor_comm {
                    let neighbor_deg = degree.get(&neighbor).copied().unwrap_or(0.0);
                    q += weight - (node_deg * neighbor_deg) / m2;
                }
            }
        }

        q / m2
    }
}

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Constraint management methods
impl GraphEngine {
    /// # Errors
    /// Returns `ConstraintAlreadyExists` if a constraint with the same name exists,
    /// or `ConstraintViolation` if existing data violates the constraint.
    pub fn create_constraint(&self, constraint: Constraint) -> Result<()> {
        let constraint_name = &constraint.name;
        let key = format!("_graph_constraint:{constraint_name}");
        if self.store.exists(&key) {
            return Err(GraphError::ConstraintAlreadyExists(constraint.name));
        }

        // For unique constraints, create backing index if not exists
        if constraint.constraint_type == ConstraintType::Unique {
            match &constraint.target {
                ConstraintTarget::NodeLabel(_) | ConstraintTarget::AllNodes => {
                    let _ = self.create_node_property_index(&constraint.property);
                },
                ConstraintTarget::EdgeType(_) | ConstraintTarget::AllEdges => {
                    let _ = self.create_edge_property_index(&constraint.property);
                },
            }

            // Validate existing data doesn't violate constraint
            self.validate_unique_constraint_on_existing_data(&constraint)?;
        }

        // For exists constraints, validate existing data
        if constraint.constraint_type == ConstraintType::Exists {
            self.validate_exists_constraint_on_existing_data(&constraint)?;
        }

        // Store constraint definition
        let mut tensor = TensorData::new();
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("constraint".into())),
        );
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String(constraint.name.clone())),
        );
        tensor.set(
            "target",
            TensorValue::Scalar(ScalarValue::String(
                serde_json::to_string(&constraint.target).unwrap_or_default(),
            )),
        );
        tensor.set(
            "property",
            TensorValue::Scalar(ScalarValue::String(constraint.property.clone())),
        );
        tensor.set(
            "constraint_type",
            TensorValue::Scalar(ScalarValue::String(
                serde_json::to_string(&constraint.constraint_type).unwrap_or_default(),
            )),
        );

        self.store.put(key, tensor)?;

        // Cache in memory for fast lookup
        self.constraints
            .write()
            .insert(constraint.name.clone(), constraint);

        Ok(())
    }

    /// # Errors
    /// Returns `ConstraintNotFound` if the constraint doesn't exist.
    pub fn drop_constraint(&self, name: &str) -> Result<()> {
        let key = format!("_graph_constraint:{name}");
        if !self.store.exists(&key) {
            return Err(GraphError::ConstraintNotFound(name.to_string()));
        }

        self.store.delete(&key)?;
        self.constraints.write().remove(name);
        Ok(())
    }

    #[must_use]
    pub fn list_constraints(&self) -> Vec<Constraint> {
        self.constraints.read().values().cloned().collect()
    }

    #[must_use]
    pub fn get_constraint(&self, name: &str) -> Option<Constraint> {
        self.constraints.read().get(name).cloned()
    }

    fn validate_unique_constraint_on_existing_data(&self, constraint: &Constraint) -> Result<()> {
        let mut seen_values: HashMap<OrderedPropertyValue, u64> = HashMap::new();

        match &constraint.target {
            ConstraintTarget::NodeLabel(label) => {
                if let Ok(nodes) = self.find_nodes_by_label(label) {
                    for node in nodes {
                        if let Some(value) = node.properties.get(&constraint.property) {
                            let ordered = OrderedPropertyValue::from(value);
                            if let Some(&existing_id) = seen_values.get(&ordered) {
                                return Err(GraphError::ConstraintViolation {
                                    constraint_name: constraint.name.clone(),
                                    message: format!(
                                        "Duplicate value for property '{}' on nodes {} and {}",
                                        constraint.property, existing_id, node.id
                                    ),
                                });
                            }
                            seen_values.insert(ordered, node.id);
                        }
                    }
                }
            },
            ConstraintTarget::AllNodes => {
                for node in self.all_nodes() {
                    if let Some(value) = node.properties.get(&constraint.property) {
                        let ordered = OrderedPropertyValue::from(value);
                        if let Some(&existing_id) = seen_values.get(&ordered) {
                            return Err(GraphError::ConstraintViolation {
                                constraint_name: constraint.name.clone(),
                                message: format!(
                                    "Duplicate value for property '{}' on nodes {} and {}",
                                    constraint.property, existing_id, node.id
                                ),
                            });
                        }
                        seen_values.insert(ordered, node.id);
                    }
                }
            },
            ConstraintTarget::EdgeType(edge_type) => {
                if let Ok(edges) = self.find_edges_by_type(edge_type) {
                    for edge in edges {
                        if let Some(value) = edge.properties.get(&constraint.property) {
                            let ordered = OrderedPropertyValue::from(value);
                            if let Some(&existing_id) = seen_values.get(&ordered) {
                                return Err(GraphError::ConstraintViolation {
                                    constraint_name: constraint.name.clone(),
                                    message: format!(
                                        "Duplicate value for property '{}' on edges {} and {}",
                                        constraint.property, existing_id, edge.id
                                    ),
                                });
                            }
                            seen_values.insert(ordered, edge.id);
                        }
                    }
                }
            },
            ConstraintTarget::AllEdges => {
                for edge in self.all_edges() {
                    if let Some(value) = edge.properties.get(&constraint.property) {
                        let ordered = OrderedPropertyValue::from(value);
                        if let Some(&existing_id) = seen_values.get(&ordered) {
                            return Err(GraphError::ConstraintViolation {
                                constraint_name: constraint.name.clone(),
                                message: format!(
                                    "Duplicate value for property '{}' on edges {} and {}",
                                    constraint.property, existing_id, edge.id
                                ),
                            });
                        }
                        seen_values.insert(ordered, edge.id);
                    }
                }
            },
        }
        Ok(())
    }

    fn validate_exists_constraint_on_existing_data(&self, constraint: &Constraint) -> Result<()> {
        match &constraint.target {
            ConstraintTarget::NodeLabel(label) => {
                if let Ok(nodes) = self.find_nodes_by_label(label) {
                    for node in nodes {
                        if !node.properties.contains_key(&constraint.property) {
                            return Err(GraphError::ConstraintViolation {
                                constraint_name: constraint.name.clone(),
                                message: format!(
                                    "Node {} missing required property '{}'",
                                    node.id, constraint.property
                                ),
                            });
                        }
                    }
                }
            },
            ConstraintTarget::AllNodes => {
                for node in self.all_nodes() {
                    if !node.properties.contains_key(&constraint.property) {
                        return Err(GraphError::ConstraintViolation {
                            constraint_name: constraint.name.clone(),
                            message: format!(
                                "Node {} missing required property '{}'",
                                node.id, constraint.property
                            ),
                        });
                    }
                }
            },
            ConstraintTarget::EdgeType(edge_type) => {
                if let Ok(edges) = self.find_edges_by_type(edge_type) {
                    for edge in edges {
                        if !edge.properties.contains_key(&constraint.property) {
                            return Err(GraphError::ConstraintViolation {
                                constraint_name: constraint.name.clone(),
                                message: format!(
                                    "Edge {} missing required property '{}'",
                                    edge.id, constraint.property
                                ),
                            });
                        }
                    }
                }
            },
            ConstraintTarget::AllEdges => {
                for edge in self.all_edges() {
                    if !edge.properties.contains_key(&constraint.property) {
                        return Err(GraphError::ConstraintViolation {
                            constraint_name: constraint.name.clone(),
                            message: format!(
                                "Edge {} missing required property '{}'",
                                edge.id, constraint.property
                            ),
                        });
                    }
                }
            },
        }
        Ok(())
    }

    #[allow(
        clippy::significant_drop_in_scrutinee,
        clippy::significant_drop_tightening
    )]
    fn validate_node_constraints(
        &self,
        labels: &[String],
        properties: &HashMap<String, PropertyValue>,
        exclude_id: Option<u64>,
    ) -> Result<()> {
        let constraints = self.constraints.read();

        for constraint in constraints.values() {
            // Check if constraint applies to this node
            let applies = match &constraint.target {
                ConstraintTarget::AllNodes => true,
                ConstraintTarget::NodeLabel(label) => labels.contains(label),
                _ => false,
            };

            if !applies {
                continue;
            }

            match &constraint.constraint_type {
                ConstraintType::Exists => match properties.get(&constraint.property) {
                    None | Some(PropertyValue::Null) => {
                        return Err(GraphError::ConstraintViolation {
                            constraint_name: constraint.name.clone(),
                            message: format!(
                                "Property '{}' is required and cannot be null",
                                constraint.property
                            ),
                        });
                    },
                    Some(_) => {},
                },
                ConstraintType::Unique => {
                    if let Some(value) = properties.get(&constraint.property) {
                        // Use index to check uniqueness
                        if let Ok(existing) =
                            self.find_nodes_by_property(&constraint.property, value)
                        {
                            for node in &existing {
                                // Skip self during update
                                if exclude_id != Some(node.id) {
                                    return Err(GraphError::ConstraintViolation {
                                        constraint_name: constraint.name.clone(),
                                        message: format!(
                                            "Property '{}' value already exists on node {}",
                                            constraint.property, node.id
                                        ),
                                    });
                                }
                            }
                        }
                    }
                },
                ConstraintType::PropertyType(expected_type) => {
                    if let Some(value) = properties.get(&constraint.property) {
                        if value.value_type() != *expected_type {
                            return Err(GraphError::ConstraintViolation {
                                constraint_name: constraint.name.clone(),
                                message: format!(
                                    "Property '{}' must be type {:?}, got {:?}",
                                    constraint.property,
                                    expected_type,
                                    value.value_type()
                                ),
                            });
                        }
                    }
                },
            }
        }

        Ok(())
    }

    #[allow(
        clippy::significant_drop_in_scrutinee,
        clippy::significant_drop_tightening
    )]
    fn validate_edge_constraints(
        &self,
        edge_type: &str,
        properties: &HashMap<String, PropertyValue>,
        exclude_id: Option<u64>,
    ) -> Result<()> {
        let constraints = self.constraints.read();

        for constraint in constraints.values() {
            // Check if constraint applies to this edge
            let applies = match &constraint.target {
                ConstraintTarget::AllEdges => true,
                ConstraintTarget::EdgeType(t) => t == edge_type,
                _ => false,
            };

            if !applies {
                continue;
            }

            match &constraint.constraint_type {
                ConstraintType::Exists => match properties.get(&constraint.property) {
                    None | Some(PropertyValue::Null) => {
                        return Err(GraphError::ConstraintViolation {
                            constraint_name: constraint.name.clone(),
                            message: format!(
                                "Property '{}' is required and cannot be null",
                                constraint.property
                            ),
                        });
                    },
                    Some(_) => {},
                },
                ConstraintType::Unique => {
                    if let Some(value) = properties.get(&constraint.property) {
                        // Use index to check uniqueness
                        if let Ok(existing) =
                            self.find_edges_by_property(&constraint.property, value)
                        {
                            for edge in &existing {
                                // Skip self during update
                                if exclude_id != Some(edge.id) {
                                    return Err(GraphError::ConstraintViolation {
                                        constraint_name: constraint.name.clone(),
                                        message: format!(
                                            "Property '{}' value already exists on edge {}",
                                            constraint.property, edge.id
                                        ),
                                    });
                                }
                            }
                        }
                    }
                },
                ConstraintType::PropertyType(expected_type) => {
                    if let Some(value) = properties.get(&constraint.property) {
                        if value.value_type() != *expected_type {
                            return Err(GraphError::ConstraintViolation {
                                constraint_name: constraint.name.clone(),
                                message: format!(
                                    "Property '{}' must be type {:?}, got {:?}",
                                    constraint.property,
                                    expected_type,
                                    value.value_type()
                                ),
                            });
                        }
                    }
                },
            }
        }

        Ok(())
    }

    fn has_any_unique_node_constraint(&self) -> bool {
        self.constraints.read().values().any(|c| {
            matches!(c.constraint_type, ConstraintType::Unique)
                && matches!(
                    c.target,
                    ConstraintTarget::AllNodes | ConstraintTarget::NodeLabel(_)
                )
        })
    }

    fn has_any_unique_edge_constraint(&self) -> bool {
        self.constraints.read().values().any(|c| {
            matches!(c.constraint_type, ConstraintType::Unique)
                && matches!(
                    c.target,
                    ConstraintTarget::AllEdges | ConstraintTarget::EdgeType(_)
                )
        })
    }
}

// Batch operations
impl GraphEngine {
    /// # Errors
    /// Returns `BatchValidationError` if any node violates constraints, or
    /// `BatchCreationError` if storage fails.
    #[allow(clippy::needless_pass_by_value)] // batch input is consumed
    pub fn batch_create_nodes(&self, nodes: Vec<NodeInput>) -> Result<BatchResult> {
        if nodes.is_empty() {
            return Ok(BatchResult {
                created_ids: vec![],
                count: 0,
            });
        }

        // Acquire lock to prevent TOCTOU race when unique constraints exist.
        // The lock is held for the entire validation + creation phase.
        let has_unique = self.has_any_unique_node_constraint();
        let _guard = if has_unique {
            Some(self.batch_unique_lock.write())
        } else {
            None
        };

        // Phase 1: Validate all inputs (check constraints if any)
        for (idx, node) in nodes.iter().enumerate() {
            self.validate_node_constraints(&node.labels, &node.properties, None)
                .map_err(|e| GraphError::BatchValidationError {
                    index: idx,
                    cause: Box::new(e),
                })?;
        }

        // Ensure label index exists
        self.ensure_label_index();

        // Phase 2: Pre-allocate IDs atomically
        let count = nodes.len();
        // Check for ID space exhaustion before allocating
        let current = self.node_counter.load(Ordering::SeqCst);
        if current > u64::MAX - count as u64 {
            return Err(GraphError::IdSpaceExhausted {
                entity_type: "node",
            });
        }
        let start_id = self.node_counter.fetch_add(count as u64, Ordering::SeqCst) + 1;
        let ids: Vec<u64> = (start_id..start_id + count as u64).collect();

        // Phase 3: Create all nodes (parallelize if count >= threshold)
        let results: Vec<Result<()>> = if count >= Self::PARALLEL_THRESHOLD {
            nodes
                .par_iter()
                .zip(ids.par_iter())
                .map(|(node, &id)| self.create_node_internal(id, &node.labels, &node.properties))
                .collect()
        } else {
            nodes
                .iter()
                .zip(ids.iter())
                .map(|(node, &id)| self.create_node_internal(id, &node.labels, &node.properties))
                .collect()
        };

        // Phase 4: Check for failures
        for (idx, result) in results.into_iter().enumerate() {
            result.map_err(|e| GraphError::BatchCreationError {
                index: idx,
                cause: Box::new(e),
            })?;
        }

        Ok(BatchResult {
            created_ids: ids,
            count,
        })
    }

    fn create_node_internal(
        &self,
        id: u64,
        labels: &[String],
        properties: &HashMap<String, PropertyValue>,
    ) -> Result<()> {
        let mut tensor = TensorData::new();
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::Int(id.cast_signed())),
        );
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("node".into())),
        );
        tensor.set("_labels", TensorValue::Pointers(labels.to_vec()));
        tensor.set(
            "_created_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
        );

        for (key, value) in properties {
            tensor.set(key, TensorValue::Scalar(value.to_scalar()));
        }

        self.store.put(Self::node_key(id), tensor)?;

        // Initialize empty edge lists
        let out_tensor = TensorData::new();
        let in_tensor = TensorData::new();
        self.store.put(Self::outgoing_edges_key(id), out_tensor)?;
        self.store.put(Self::incoming_edges_key(id), in_tensor)?;

        // Update indexes
        self.index_node_properties(id, labels, properties);

        Ok(())
    }

    /// Creates multiple edges in a single batch operation.
    ///
    /// # Concurrency Note
    /// Node existence is validated before edge creation. If nodes are deleted
    /// concurrently by another thread, edge creation will fail with `NodeNotFound`.
    /// This is by design - the caller should retry if this race is possible.
    ///
    /// # Errors
    /// Returns `BatchValidationError` if nodes don't exist or constraints are violated.
    #[allow(clippy::needless_pass_by_value)] // batch input is consumed
    pub fn batch_create_edges(&self, edges: Vec<EdgeInput>) -> Result<BatchResult> {
        if edges.is_empty() {
            return Ok(BatchResult {
                created_ids: vec![],
                count: 0,
            });
        }

        // Acquire lock to prevent TOCTOU race when unique constraints exist.
        let has_unique = self.has_any_unique_edge_constraint();
        let _guard = if has_unique {
            Some(self.batch_unique_lock.write())
        } else {
            None
        };

        // Phase 1: Validate all source/target nodes exist and constraints
        for (idx, edge) in edges.iter().enumerate() {
            if !self.node_exists(edge.from) {
                return Err(GraphError::BatchValidationError {
                    index: idx,
                    cause: Box::new(GraphError::NodeNotFound(edge.from)),
                });
            }
            if !self.node_exists(edge.to) {
                return Err(GraphError::BatchValidationError {
                    index: idx,
                    cause: Box::new(GraphError::NodeNotFound(edge.to)),
                });
            }
            self.validate_edge_constraints(&edge.edge_type, &edge.properties, None)
                .map_err(|e| GraphError::BatchValidationError {
                    index: idx,
                    cause: Box::new(e),
                })?;
        }

        // Ensure edge type index exists
        self.ensure_edge_type_index();

        // Phase 2: Pre-allocate IDs
        let count = edges.len();
        // Check for ID space exhaustion before allocating
        let current = self.edge_counter.load(Ordering::SeqCst);
        if current > u64::MAX - count as u64 {
            return Err(GraphError::IdSpaceExhausted {
                entity_type: "edge",
            });
        }
        let start_id = self.edge_counter.fetch_add(count as u64, Ordering::SeqCst) + 1;
        let ids: Vec<u64> = (start_id..start_id + count as u64).collect();

        // Phase 3: Create all edges (cannot fully parallelize due to edge list updates)
        for (edge, &id) in edges.iter().zip(ids.iter()) {
            self.create_edge_internal(
                id,
                edge.from,
                edge.to,
                &edge.edge_type,
                &edge.properties,
                edge.directed,
            )?;
        }

        Ok(BatchResult {
            created_ids: ids,
            count,
        })
    }

    fn create_edge_internal(
        &self,
        id: u64,
        from: u64,
        to: u64,
        edge_type: &str,
        properties: &HashMap<String, PropertyValue>,
        directed: bool,
    ) -> Result<()> {
        let mut tensor = TensorData::new();
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::Int(id.cast_signed())),
        );
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("edge".into())),
        );
        tensor.set(
            "_from",
            TensorValue::Scalar(ScalarValue::Int(from.cast_signed())),
        );
        tensor.set(
            "_to",
            TensorValue::Scalar(ScalarValue::Int(to.cast_signed())),
        );
        tensor.set(
            "_edge_type",
            TensorValue::Scalar(ScalarValue::String(edge_type.to_string())),
        );
        tensor.set(
            "_directed",
            TensorValue::Scalar(ScalarValue::Bool(directed)),
        );
        tensor.set(
            "_created_at",
            TensorValue::Scalar(ScalarValue::Int(current_timestamp_millis().cast_signed())),
        );

        for (key, value) in properties {
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
        self.index_edge_properties(id, edge_type, properties);

        Ok(())
    }

    /// Deletes multiple nodes, reporting both successes and failures.
    ///
    /// Unlike fail-fast batch operations, this method continues processing
    /// after individual failures and reports which IDs failed and why.
    ///
    /// # Errors
    ///
    /// This method always returns `Ok`. Individual failures are reported
    /// in `BatchDeleteResult::failed`.
    pub fn batch_delete_nodes(&self, ids: Vec<u64>) -> Result<BatchDeleteResult> {
        let mut deleted_ids = Vec::with_capacity(ids.len());
        let mut failed = Vec::new();

        for (idx, id) in ids.into_iter().enumerate() {
            match self.delete_node(id) {
                Ok(()) => deleted_ids.push(id),
                Err(e) => {
                    failed.push(GraphBatchItemError {
                        index: idx,
                        id: Some(id),
                        cause: e.to_string(),
                    });
                },
            }
        }

        Ok(BatchDeleteResult::with_failures(deleted_ids, failed))
    }

    /// Deletes multiple edges, reporting both successes and failures.
    ///
    /// Unlike fail-fast batch operations, this method continues processing
    /// after individual failures and reports which IDs failed and why.
    ///
    /// # Errors
    ///
    /// This method always returns `Ok`. Individual failures are reported
    /// in `BatchDeleteResult::failed`.
    pub fn batch_delete_edges(&self, ids: Vec<u64>) -> Result<BatchDeleteResult> {
        let mut deleted_ids = Vec::with_capacity(ids.len());
        let mut failed = Vec::new();

        for (idx, id) in ids.into_iter().enumerate() {
            match self.delete_edge(id) {
                Ok(()) => deleted_ids.push(id),
                Err(e) => {
                    failed.push(GraphBatchItemError {
                        index: idx,
                        id: Some(id),
                        cause: e.to_string(),
                    });
                },
            }
        }

        Ok(BatchDeleteResult::with_failures(deleted_ids, failed))
    }

    /// # Errors
    /// Returns `BatchValidationError` if any node doesn't exist or violates constraints.
    #[allow(clippy::type_complexity)]
    pub fn batch_update_nodes(
        &self,
        updates: Vec<(u64, Option<Vec<String>>, HashMap<String, PropertyValue>)>,
    ) -> Result<usize> {
        // Validate all updates first
        for (idx, (id, labels, properties)) in updates.iter().enumerate() {
            let node = self
                .get_node(*id)
                .map_err(|e| GraphError::BatchValidationError {
                    index: idx,
                    cause: Box::new(e),
                })?;
            let effective_labels = labels.as_ref().unwrap_or(&node.labels);
            self.validate_node_constraints(effective_labels, properties, Some(*id))
                .map_err(|e| GraphError::BatchValidationError {
                    index: idx,
                    cause: Box::new(e),
                })?;
        }

        // Apply updates
        let mut update_count = 0;
        for (id, labels, properties) in updates {
            if self.update_node(id, labels, properties).is_ok() {
                update_count += 1;
            }
        }
        Ok(update_count)
    }
}

#[cfg(test)]
mod tests;
