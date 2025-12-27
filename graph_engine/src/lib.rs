use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use tensor_store::{fields, ScalarValue, TensorData, TensorStore, TensorStoreError, TensorValue};

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
            PropertyValue::Null => ScalarValue::Null,
            PropertyValue::Int(v) => ScalarValue::Int(*v),
            PropertyValue::Float(v) => ScalarValue::Float(*v),
            PropertyValue::String(v) => ScalarValue::String(v.clone()),
            PropertyValue::Bool(v) => ScalarValue::Bool(*v),
        }
    }

    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Null => PropertyValue::Null,
            ScalarValue::Int(v) => PropertyValue::Int(*v),
            ScalarValue::Float(v) => PropertyValue::Float(*v),
            ScalarValue::String(v) => PropertyValue::String(v.clone()),
            ScalarValue::Bool(v) => PropertyValue::Bool(*v),
            ScalarValue::Bytes(_) => PropertyValue::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub label: String,
    pub properties: HashMap<String, PropertyValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub properties: HashMap<String, PropertyValue>,
    pub directed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    pub nodes: Vec<u64>,
    pub edges: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphError {
    NodeNotFound(u64),
    EdgeNotFound(u64),
    StorageError(String),
    PathNotFound,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            GraphError::EdgeNotFound(id) => write!(f, "Edge not found: {}", id),
            GraphError::StorageError(e) => write!(f, "Storage error: {}", e),
            GraphError::PathNotFound => write!(f, "No path found between nodes"),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<TensorStoreError> for GraphError {
    fn from(e: TensorStoreError) -> Self {
        GraphError::StorageError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, GraphError>;

pub struct GraphEngine {
    store: TensorStore,
    node_counter: AtomicU64,
    edge_counter: AtomicU64,
}

impl GraphEngine {
    const PARALLEL_THRESHOLD: usize = 100;

    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            node_counter: AtomicU64::new(0),
            edge_counter: AtomicU64::new(0),
        }
    }

    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store,
            node_counter: AtomicU64::new(0),
            edge_counter: AtomicU64::new(0),
        }
    }

    /// Access the underlying store.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    fn node_key(id: u64) -> String {
        format!("node:{}", id)
    }

    fn edge_key(id: u64) -> String {
        format!("edge:{}", id)
    }

    fn outgoing_edges_key(node_id: u64) -> String {
        format!("node:{}:out", node_id)
    }

    fn incoming_edges_key(node_id: u64) -> String {
        format!("node:{}:in", node_id)
    }

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
        tensor.set("_label", TensorValue::Scalar(ScalarValue::String(label)));

        for (key, value) in &properties {
            tensor.set(key, TensorValue::Scalar(value.to_scalar()));
        }

        self.store.put(Self::node_key(id), tensor)?;

        // Initialize empty edge lists
        let out_tensor = TensorData::new();
        let in_tensor = TensorData::new();
        self.store.put(Self::outgoing_edges_key(id), out_tensor)?;
        self.store.put(Self::incoming_edges_key(id), in_tensor)?;

        Ok(id)
    }

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
            TensorValue::Scalar(ScalarValue::String(edge_type)),
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

    fn get_edge_list(&self, key: &str) -> Result<Vec<u64>> {
        let tensor = match self.store.get(key) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut edges = Vec::new();
        for k in tensor.keys() {
            if k.starts_with('e') {
                if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = tensor.get(k) {
                    edges.push(*id as u64);
                }
            }
        }
        Ok(edges)
    }

    pub fn node_exists(&self, id: u64) -> bool {
        self.store.exists(&Self::node_key(id))
    }

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
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(node_id))?;
            for edge_id in out_edges {
                if let Ok(edge) = self.get_edge(edge_id) {
                    if edge_type.is_none() || edge_type == Some(&edge.edge_type) {
                        // For outgoing, the neighbor is the 'to' node (unless it's us in undirected)
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
            let in_edges = self.get_edge_list(&Self::incoming_edges_key(node_id))?;
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

            let neighbors = self.get_neighbor_ids(current_id, edge_type, direction)?;
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
    ) -> Result<Vec<u64>> {
        let mut neighbor_ids = HashSet::new();

        if direction == Direction::Outgoing || direction == Direction::Both {
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(node_id))?;
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
            let in_edges = self.get_edge_list(&Self::incoming_edges_key(node_id))?;
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
        Ok(neighbor_ids.into_iter().collect())
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
            let out_edges = self.get_edge_list(&Self::outgoing_edges_key(current))?;

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

    pub fn node_count(&self) -> usize {
        self.store.scan_count("node:") - self.store.scan_count("node:") / 3 * 2
    }

    pub fn delete_node(&self, id: u64) -> Result<()> {
        if !self.node_exists(id) {
            return Err(GraphError::NodeNotFound(id));
        }

        // Delete edges connected to this node
        let out_edges = self.get_edge_list(&Self::outgoing_edges_key(id))?;
        let in_edges = self.get_edge_list(&Self::incoming_edges_key(id))?;

        // Collect all edge IDs to delete
        let all_edges: Vec<u64> = out_edges.into_iter().chain(in_edges).collect();

        // Parallel deletion for high-degree nodes
        if all_edges.len() >= Self::PARALLEL_THRESHOLD {
            all_edges.par_iter().for_each(|edge_id| {
                let _ = self.store.delete(&Self::edge_key(*edge_id));
            });
        } else {
            for edge_id in all_edges {
                let _ = self.store.delete(&Self::edge_key(edge_id));
            }
        }

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
}
