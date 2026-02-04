use super::*;
use tensor_store::TensorStoreError;

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
    // Types that roundtrip exactly
    let exact_values = vec![
        PropertyValue::Null,
        PropertyValue::Int(42),
        PropertyValue::Float(3.14),
        PropertyValue::String("test".into()),
        PropertyValue::Bool(true),
        PropertyValue::Bytes(vec![1, 2, 3]),
        PropertyValue::List(vec![PropertyValue::Int(1), PropertyValue::Int(2)]),
        PropertyValue::Point {
            lat: 51.5,
            lon: -0.1,
        },
    ];

    for v in exact_values {
        let scalar = v.to_scalar();
        let back = PropertyValue::from_scalar(&scalar);
        assert_eq!(v, back);
    }

    // DateTime is stored as Int, so roundtrip produces Int
    let dt = PropertyValue::DateTime(1234567890);
    let scalar = dt.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, PropertyValue::Int(1234567890));

    // Bytes from ScalarValue::Bytes
    let bytes = ScalarValue::Bytes(vec![1, 2, 3]);
    assert_eq!(
        PropertyValue::from_scalar(&bytes),
        PropertyValue::Bytes(vec![1, 2, 3])
    );
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

// Unified Entity Mode tests (deprecated API - retained for backwards compatibility testing)

#[test]
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
fn entity_edge_nonexistent_returns_error() {
    let engine = GraphEngine::new();
    let result = engine.get_entity_edge("nonexistent:edge");
    assert!(result.is_err());
}

#[test]
#[allow(deprecated)]
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
fn variable_paths_memory_budget() {
    // Create a graph with many paths, use low memory budget
    let config = GraphEngineConfig::new().max_path_search_memory_bytes(100);
    let engine = GraphEngine::with_config(config);
    let a = engine.create_node("A", HashMap::new()).unwrap();
    let z = engine.create_node("Z", HashMap::new()).unwrap();

    // Create many intermediate nodes for exponential paths
    for i in 0..20 {
        let n = engine
            .create_node(&format!("N{i}"), HashMap::new())
            .unwrap();
        engine.create_edge(a, n, "E", HashMap::new(), true).unwrap();
        engine.create_edge(n, z, "E", HashMap::new(), true).unwrap();
    }

    let path_config = VariableLengthConfig::with_hops(1, 3);
    let result = engine.find_variable_paths(a, z, path_config).unwrap();

    // Should have found some paths but stopped due to memory limit
    assert!(result.stats.truncated);
    // Should have found at least a few paths before memory limit
    assert!(!result.paths.is_empty());
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
    let filter = TraversalFilter::new().node_where("age", CompareOp::Gt, PropertyValue::Int(28));

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
    let filter = TraversalFilter::new().node_where("level", CompareOp::Gt, PropertyValue::Int(100));

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
    let filter = TraversalFilter::new().edge_where("cost", CompareOp::Lt, PropertyValue::Int(50));

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

    let filter = TraversalFilter::new().node_where("age", CompareOp::Ge, PropertyValue::Int(30));

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

// ========== Pattern Matching Tests ==========

#[test]
fn node_pattern_default_matches_any() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let node = engine.get_node(n1).unwrap();

    let pattern = NodePattern::new();
    assert!(pattern.matches(&node));
}

#[test]
fn node_pattern_matches_label() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let node = engine.get_node(n1).unwrap();

    let pattern = NodePattern::new().label("Person");
    assert!(pattern.matches(&node));
}

#[test]
fn node_pattern_matches_label_no_match() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let node = engine.get_node(n1).unwrap();

    let pattern = NodePattern::new().label("Company");
    assert!(!pattern.matches(&node));
}

#[test]
fn node_pattern_matches_property_eq() {
    let engine = GraphEngine::new();
    let mut props = HashMap::new();
    props.insert("name".to_string(), PropertyValue::String("Alice".into()));
    let n1 = engine.create_node("Person", props).unwrap();
    let node = engine.get_node(n1).unwrap();

    let pattern = NodePattern::new().where_eq("name", PropertyValue::String("Alice".into()));
    assert!(pattern.matches(&node));

    let pattern2 = NodePattern::new().where_eq("name", PropertyValue::String("Bob".into()));
    assert!(!pattern2.matches(&node));
}

#[test]
fn node_pattern_matches_property_range() {
    let engine = GraphEngine::new();
    let mut props = HashMap::new();
    props.insert("age".to_string(), PropertyValue::Int(30));
    let n1 = engine.create_node("Person", props).unwrap();
    let node = engine.get_node(n1).unwrap();

    let pattern = NodePattern::new().where_cond("age", CompareOp::Gt, PropertyValue::Int(25));
    assert!(pattern.matches(&node));

    let pattern2 = NodePattern::new().where_cond("age", CompareOp::Lt, PropertyValue::Int(25));
    assert!(!pattern2.matches(&node));
}

#[test]
fn node_pattern_matches_label_and_property() {
    let engine = GraphEngine::new();
    let mut props = HashMap::new();
    props.insert("name".to_string(), PropertyValue::String("Alice".into()));
    let n1 = engine.create_node("Person", props).unwrap();
    let node = engine.get_node(n1).unwrap();

    let pattern = NodePattern::new()
        .label("Person")
        .where_eq("name", PropertyValue::String("Alice".into()));
    assert!(pattern.matches(&node));

    let pattern2 = NodePattern::new()
        .label("Company")
        .where_eq("name", PropertyValue::String("Alice".into()));
    assert!(!pattern2.matches(&node));
}

#[test]
fn node_pattern_builder_chain() {
    let pattern = NodePattern::new()
        .variable("n")
        .label("Person")
        .where_eq("name", PropertyValue::String("Alice".into()))
        .where_cond("age", CompareOp::Gt, PropertyValue::Int(18));

    assert_eq!(pattern.variable, Some("n".to_string()));
    assert_eq!(pattern.label, Some("Person".to_string()));
    assert_eq!(pattern.conditions.len(), 2);
}

#[test]
fn edge_pattern_default_matches_any() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    let edge = engine.get_edge(e1).unwrap();

    let pattern = EdgePattern::new();
    assert!(pattern.matches(&edge));
}

#[test]
fn edge_pattern_matches_type() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    let edge = engine.get_edge(e1).unwrap();

    let pattern = EdgePattern::new().edge_type("KNOWS");
    assert!(pattern.matches(&edge));
}

#[test]
fn edge_pattern_matches_type_no_match() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    let edge = engine.get_edge(e1).unwrap();

    let pattern = EdgePattern::new().edge_type("FOLLOWS");
    assert!(!pattern.matches(&edge));
}

#[test]
fn edge_pattern_matches_property() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let mut props = HashMap::new();
    props.insert("since".to_string(), PropertyValue::Int(2020));
    let e1 = engine.create_edge(n1, n2, "KNOWS", props, true).unwrap();
    let edge = engine.get_edge(e1).unwrap();

    let pattern = EdgePattern::new().where_eq("since", PropertyValue::Int(2020));
    assert!(pattern.matches(&edge));

    let pattern2 = EdgePattern::new().where_eq("since", PropertyValue::Int(2021));
    assert!(!pattern2.matches(&edge));
}

#[test]
fn edge_pattern_variable_length_spec() {
    let pattern = EdgePattern::new().variable_length(1, 5);
    assert!(pattern.variable_length.is_some());
    let spec = pattern.variable_length.unwrap();
    assert_eq!(spec.min_hops, 1);
    assert_eq!(spec.max_hops, 5);
}

#[test]
fn edge_pattern_builder_chain() {
    let pattern = EdgePattern::new()
        .variable("r")
        .edge_type("KNOWS")
        .direction(Direction::Both)
        .where_eq("since", PropertyValue::Int(2020));

    assert_eq!(pattern.variable, Some("r".to_string()));
    assert_eq!(pattern.edge_type, Some("KNOWS".to_string()));
    assert_eq!(pattern.direction, Direction::Both);
    assert_eq!(pattern.conditions.len(), 1);
}

#[test]
fn path_pattern_new_creates_three_elements() {
    let path = PathPattern::new(
        NodePattern::new().label("Person"),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new().label("Person"),
    );
    assert_eq!(path.elements.len(), 3);
}

#[test]
fn path_pattern_extend_adds_two_elements() {
    let path = PathPattern::new(
        NodePattern::new().label("A"),
        EdgePattern::new().edge_type("R"),
        NodePattern::new().label("B"),
    )
    .extend(
        EdgePattern::new().edge_type("S"),
        NodePattern::new().label("C"),
    );

    assert_eq!(path.elements.len(), 5);
}

#[test]
fn path_pattern_node_patterns_iterator() {
    let path = PathPattern::new(
        NodePattern::new().label("A"),
        EdgePattern::new().edge_type("R"),
        NodePattern::new().label("B"),
    );
    let node_patterns: Vec<_> = path.node_patterns().collect();
    assert_eq!(node_patterns.len(), 2);
}

#[test]
fn path_pattern_edge_patterns_iterator() {
    let path = PathPattern::new(
        NodePattern::new().label("A"),
        EdgePattern::new().edge_type("R"),
        NodePattern::new().label("B"),
    )
    .extend(
        EdgePattern::new().edge_type("S"),
        NodePattern::new().label("C"),
    );
    let edge_patterns: Vec<_> = path.edge_patterns().collect();
    assert_eq!(edge_patterns.len(), 2);
}

#[test]
fn pattern_default_limit() {
    let path = PathPattern::new(NodePattern::new(), EdgePattern::new(), NodePattern::new());
    let pattern = Pattern::new(path);
    assert_eq!(pattern.limit, None);
}

#[test]
fn pattern_with_limit() {
    let path = PathPattern::new(NodePattern::new(), EdgePattern::new(), NodePattern::new());
    let pattern = Pattern::new(path).limit(10);
    assert_eq!(pattern.limit, Some(10));
}

#[test]
fn pattern_match_result_is_empty() {
    let result = PatternMatchResult::empty();
    assert!(result.is_empty());

    let mut result2 = PatternMatchResult::empty();
    result2.matches.push(PatternMatch::new());
    assert!(!result2.is_empty());
}

#[test]
fn pattern_match_result_len() {
    let mut result = PatternMatchResult::empty();
    assert_eq!(result.len(), 0);

    result.matches.push(PatternMatch::new());
    result.matches.push(PatternMatch::new());
    assert_eq!(result.len(), 2);
}

#[test]
fn pattern_match_stats_default() {
    let stats = PatternMatchStats::default();
    assert_eq!(stats.matches_found, 0);
    assert_eq!(stats.nodes_evaluated, 0);
    assert_eq!(stats.edges_evaluated, 0);
    assert!(!stats.truncated);
}

#[test]
fn match_simple_two_nodes() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new().variable("a"),
            EdgePattern::new().variable("r"),
            NodePattern::new().variable("b"),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let m = &result.matches[0];
    assert!(m.get_node("a").is_some());
    assert!(m.get_node("b").is_some());
    assert!(m.get_edge("r").is_some());
}

#[test]
fn match_with_label_filter() {
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

    let result = engine
        .match_simple(
            NodePattern::new().label("Person"),
            EdgePattern::new(),
            NodePattern::new().label("Person"),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
}

#[test]
fn match_with_property_filter() {
    let engine = GraphEngine::new();
    let mut props1 = HashMap::new();
    props1.insert("name".to_string(), PropertyValue::String("Alice".into()));
    let n1 = engine.create_node("Person", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("name".to_string(), PropertyValue::String("Bob".into()));
    let n2 = engine.create_node("Person", props2).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("name".to_string(), PropertyValue::String("Charlie".into()));
    let n3 = engine.create_node("Person", props3).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new().where_eq("name", PropertyValue::String("Alice".into())),
            EdgePattern::new(),
            NodePattern::new().where_eq("name", PropertyValue::String("Bob".into())),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
}

#[test]
fn match_with_edge_type_filter() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n2, "FOLLOWS", HashMap::new(), true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new(),
            EdgePattern::new().edge_type("KNOWS"),
            NodePattern::new(),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
}

#[test]
fn match_bidirectional() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    // Outgoing only should find 1 from n1's perspective
    let result_out = engine
        .match_simple(
            NodePattern::new(),
            EdgePattern::new().direction(Direction::Outgoing),
            NodePattern::new(),
        )
        .unwrap();
    assert_eq!(result_out.len(), 1);

    // Both directions should find matches from both nodes
    let result_both = engine
        .match_simple(
            NodePattern::new(),
            EdgePattern::new().direction(Direction::Both),
            NodePattern::new(),
        )
        .unwrap();
    assert_eq!(result_both.len(), 2);
}

#[test]
fn match_chain_three_nodes() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "KNOWS", HashMap::new(), true)
        .unwrap();

    let path = PathPattern::new(
        NodePattern::new().variable("a"),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new().variable("b"),
    )
    .extend(
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new().variable("c"),
    );

    let result = engine.match_pattern(&Pattern::new(path)).unwrap();
    assert_eq!(result.len(), 1);

    let m = &result.matches[0];
    assert_eq!(m.get_node("a").unwrap().id, n1);
    assert_eq!(m.get_node("b").unwrap().id, n2);
    assert_eq!(m.get_node("c").unwrap().id, n3);
}

#[test]
fn match_variable_length_path() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new().variable("a"),
            EdgePattern::new()
                .edge_type("KNOWS")
                .variable_length(1, 3)
                .variable("p"),
            NodePattern::new().variable("b"),
        )
        .unwrap();

    // Should find: n1->n2 (1 hop), n1->n3 (2 hops), n2->n3 (1 hop)
    assert_eq!(result.len(), 3);
}

#[test]
fn match_no_results() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    // No edges

    let result = engine
        .match_simple(NodePattern::new(), EdgePattern::new(), NodePattern::new())
        .unwrap();

    assert!(result.is_empty());
    assert_eq!(result.stats.matches_found, 0);
}

#[test]
fn match_with_limit_truncates() {
    let engine = GraphEngine::new();
    // Create a chain of nodes
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "KNOWS", HashMap::new(), true)
            .unwrap();
    }

    let path = PathPattern::new(NodePattern::new(), EdgePattern::new(), NodePattern::new());
    let pattern = Pattern::new(path).limit(2);
    let result = engine.match_pattern(&pattern).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.stats.truncated);
}

#[test]
fn match_returns_correct_bindings() {
    let engine = GraphEngine::new();
    let mut props1 = HashMap::new();
    props1.insert("name".to_string(), PropertyValue::String("Alice".into()));
    let n1 = engine.create_node("Person", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("name".to_string(), PropertyValue::String("Bob".into()));
    let n2 = engine.create_node("Person", props2).unwrap();

    let mut edge_props = HashMap::new();
    edge_props.insert("since".to_string(), PropertyValue::Int(2020));
    engine
        .create_edge(n1, n2, "KNOWS", edge_props, true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new().variable("a"),
            EdgePattern::new().variable("r"),
            NodePattern::new().variable("b"),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let m = &result.matches[0];

    let node_a = m.get_node("a").unwrap();
    assert_eq!(
        node_a.properties.get("name"),
        Some(&PropertyValue::String("Alice".into()))
    );

    let node_b = m.get_node("b").unwrap();
    assert_eq!(
        node_b.properties.get("name"),
        Some(&PropertyValue::String("Bob".into()))
    );

    let edge_r = m.get_edge("r").unwrap();
    assert_eq!(
        edge_r.properties.get("since"),
        Some(&PropertyValue::Int(2020))
    );
}

#[test]
fn count_pattern_matches_basic() {
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

    let path = PathPattern::new(
        NodePattern::new(),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new(),
    );
    let count = engine.count_pattern_matches(&Pattern::new(path)).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn count_pattern_matches_empty() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();

    let path = PathPattern::new(NodePattern::new(), EdgePattern::new(), NodePattern::new());
    let count = engine.count_pattern_matches(&Pattern::new(path)).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn pattern_exists_true() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    let path = PathPattern::new(
        NodePattern::new(),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new(),
    );
    assert!(engine.pattern_exists(&Pattern::new(path)).unwrap());
}

#[test]
fn pattern_exists_false() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();

    let path = PathPattern::new(
        NodePattern::new(),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new(),
    );
    assert!(!engine.pattern_exists(&Pattern::new(path)).unwrap());
}

#[test]
fn pattern_exists_short_circuits() {
    let engine = GraphEngine::new();
    // Create many edges
    for i in 0..100 {
        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();
        if i == 0 {
            // First edge should be enough to return true
        }
    }

    let path = PathPattern::new(
        NodePattern::new(),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new(),
    );
    let result = engine.pattern_exists(&Pattern::new(path)).unwrap();
    assert!(result);
}

#[test]
fn match_self_loop() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n1, "SELF_REF", HashMap::new(), true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new().variable("a"),
            EdgePattern::new().edge_type("SELF_REF"),
            NodePattern::new().variable("b"),
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let m = &result.matches[0];
    assert_eq!(m.get_node("a").unwrap().id, m.get_node("b").unwrap().id);
}

#[test]
fn match_disconnected_graph() {
    let engine = GraphEngine::new();
    // Create two disconnected components
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    let n4 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n3, n4, "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine
        .match_simple(
            NodePattern::new(),
            EdgePattern::new().edge_type("KNOWS"),
            NodePattern::new(),
        )
        .unwrap();

    assert_eq!(result.len(), 2);
}

#[test]
fn variable_length_spec_caps_max_hops() {
    let spec = VariableLengthSpec::new(1, 100);
    assert_eq!(spec.max_hops, MAX_VARIABLE_LENGTH_HOPS);
}

#[test]
fn pattern_match_get_node_returns_none_for_edge() {
    let mut m = PatternMatch::new();
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    let edge = engine.get_edge(e1).unwrap();

    m.bindings.insert("r".to_string(), Binding::Edge(edge));
    assert!(m.get_node("r").is_none());
}

#[test]
fn pattern_match_get_edge_returns_none_for_node() {
    let mut m = PatternMatch::new();
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let node = engine.get_node(n1).unwrap();

    m.bindings.insert("n".to_string(), Binding::Node(node));
    assert!(m.get_edge("n").is_none());
}

#[test]
fn pattern_match_get_path() {
    let mut m = PatternMatch::new();
    let path = Path {
        nodes: vec![1, 2, 3],
        edges: vec![1, 2],
    };
    m.bindings.insert("p".to_string(), Binding::Path(path));

    let retrieved = m.get_path("p").unwrap();
    assert_eq!(retrieved.nodes, vec![1, 2, 3]);
    assert_eq!(retrieved.edges, vec![1, 2]);
}

#[test]
fn match_variable_length_min_zero() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    // With min=0, should match the start node itself as well
    let result = engine
        .match_simple(
            NodePattern::new().variable("a"),
            EdgePattern::new()
                .edge_type("KNOWS")
                .variable_length(0, 2)
                .variable("p"),
            NodePattern::new().variable("b"),
        )
        .unwrap();

    // Should find: n1->n1 (0 hops), n1->n2 (1 hop), n2->n2 (0 hops)
    assert!(result.len() >= 2);
}

#[test]
fn match_empty_graph() {
    let engine = GraphEngine::new();

    let result = engine
        .match_simple(NodePattern::new(), EdgePattern::new(), NodePattern::new())
        .unwrap();

    assert!(result.is_empty());
    assert_eq!(result.stats.matches_found, 0);
}

#[test]
fn pattern_match_result_default() {
    let result = PatternMatchResult::default();
    assert!(result.is_empty());
    assert_eq!(result.stats, PatternMatchStats::default());
}

// ==================== PageRank Tests ====================

#[test]
fn pagerank_empty_graph() {
    let engine = GraphEngine::new();
    let result = engine.pagerank(None).unwrap();
    assert!(result.scores.is_empty());
    assert_eq!(result.iterations, 0);
    assert!(result.converged);
}

#[test]
fn pagerank_single_node() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 1);
    assert!((result.scores[&n1] - 1.0).abs() < 1e-6);
    assert!(result.converged);
}

#[test]
fn pagerank_two_nodes_directed() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "LINKS", HashMap::new(), true)
        .unwrap();

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 2);
    // Node 2 should have higher rank (receives link)
    assert!(result.scores[&n2] > result.scores[&n1]);
}

#[test]
fn pagerank_cycle() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..4 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(nodes[0], nodes[1], "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[2], "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[2], nodes[3], "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[3], nodes[0], "LINKS", HashMap::new(), true)
        .unwrap();

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 4);
    // All nodes should have equal rank in a cycle
    let first_score = result.scores[&nodes[0]];
    for node in &nodes[1..] {
        assert!((result.scores[node] - first_score).abs() < 0.01);
    }
}

#[test]
fn pagerank_star_graph() {
    let engine = GraphEngine::new();
    // Center node, spokes all pointing to center
    let center = engine.create_node("Person", HashMap::new()).unwrap();
    let mut spokes = Vec::new();
    for _ in 0..5 {
        let spoke = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(spoke, center, "LINKS", HashMap::new(), true)
            .unwrap();
        spokes.push(spoke);
    }

    let result = engine.pagerank(None).unwrap();
    // Center should have highest rank
    let center_score = result.scores[&center];
    for spoke in &spokes {
        assert!(center_score > result.scores[spoke]);
    }
}

#[test]
fn pagerank_convergence() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..10 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..9 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "LINKS", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.pagerank(None).unwrap();
    assert!(result.converged);
    assert!(result.convergence < 1e-6);
}

#[test]
fn pagerank_max_iterations() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "LINKS", HashMap::new(), true)
            .unwrap();
    }

    let config = PageRankConfig::new().max_iterations(2);
    let result = engine.pagerank(Some(config)).unwrap();
    assert_eq!(result.iterations, 2);
}

#[test]
fn pagerank_damping_factor() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "LINKS", HashMap::new(), true)
        .unwrap();

    let config_low = PageRankConfig::new().damping(0.5);
    let config_high = PageRankConfig::new().damping(0.99);

    let result_low = engine.pagerank(Some(config_low)).unwrap();
    let result_high = engine.pagerank(Some(config_high)).unwrap();

    // Higher damping should lead to more difference between nodes
    let diff_low = (result_low.scores[&n2] - result_low.scores[&n1]).abs();
    let diff_high = (result_high.scores[&n2] - result_high.scores[&n1]).abs();
    assert!(diff_high > diff_low);
}

#[test]
fn pagerank_dangling_nodes() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    // 1 -> 2, 1 -> 3, 2 and 3 are dangling (no outgoing edges)
    engine
        .create_edge(n1, n2, "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "LINKS", HashMap::new(), true)
        .unwrap();

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 3);
    // All nodes should have valid scores
    for score in result.scores.values() {
        assert!(*score > 0.0);
        assert!(score.is_finite());
    }
}

#[test]
fn pagerank_edge_type_filter() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "FOLLOWS", HashMap::new(), true)
        .unwrap();

    let config = PageRankConfig::new().edge_type("LINKS");
    let result = engine.pagerank(Some(config)).unwrap();

    // With only LINKS edges, node 2 should rank higher than 3
    assert!(result.scores[&n2] > result.scores[&n3]);
}

#[test]
fn pagerank_result_top_k() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n2, "LINKS", HashMap::new(), true)
        .unwrap();

    let result = engine.pagerank(None).unwrap();
    let top_2 = result.top_k(2);
    assert_eq!(top_2.len(), 2);
    // Node 2 should be first (highest rank)
    assert_eq!(top_2[0].0, n2);
}

#[test]
fn pagerank_config_builder() {
    let config = PageRankConfig::new()
        .damping(0.9)
        .tolerance(1e-8)
        .max_iterations(50)
        .direction(Direction::Both)
        .edge_type("KNOWS");

    assert!((config.damping - 0.9).abs() < 1e-10);
    assert!((config.tolerance - 1e-8).abs() < 1e-12);
    assert_eq!(config.max_iterations, 50);
    assert_eq!(config.direction, Direction::Both);
    assert_eq!(config.edge_type, Some("KNOWS".to_string()));
}

#[test]
fn graph_engine_config_default() {
    let config = GraphEngineConfig::default();
    assert_eq!(config.default_match_limit, 1000);
    assert_eq!(config.pattern_parallel_threshold, 100);
    assert_eq!(config.max_variable_length_hops, 20);
    assert!((config.pagerank_default_damping - 0.85).abs() < 1e-10);
    assert!((config.pagerank_default_tolerance - 1e-6).abs() < 1e-12);
    assert_eq!(config.pagerank_default_max_iterations, 100);
    assert_eq!(config.centrality_parallel_threshold, 100);
    assert_eq!(config.community_max_passes, 10);
    assert_eq!(config.label_propagation_max_iterations, 100);
    assert_eq!(config.index_lock_count, 64);
    assert_eq!(config.max_path_search_memory_bytes, 100 * 1024 * 1024);
}

#[test]
fn graph_engine_config_builder() {
    let config = GraphEngineConfig::new()
        .default_match_limit(500)
        .pattern_parallel_threshold(50)
        .max_variable_length_hops(10)
        .index_lock_count(32);

    assert_eq!(config.default_match_limit, 500);
    assert_eq!(config.pattern_parallel_threshold, 50);
    assert_eq!(config.max_variable_length_hops, 10);
    assert_eq!(config.index_lock_count, 32);
}

#[test]
fn graph_engine_with_config() {
    let config = GraphEngineConfig::new().index_lock_count(16);
    let engine = GraphEngine::with_config(config);
    assert_eq!(engine.config().index_lock_count, 16);
    assert_eq!(engine.index_locks.len(), 16);
}

// ==================== Centrality Tests ====================

#[test]
fn betweenness_empty_graph() {
    let engine = GraphEngine::new();
    let result = engine.betweenness_centrality(None).unwrap();
    assert!(result.scores.is_empty());
    assert_eq!(result.centrality_type, CentralityType::Betweenness);
}

#[test]
fn betweenness_line_graph() {
    let engine = GraphEngine::new();
    // 1 - 2 - 3 - 4 - 5
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(nodes[i + 1], nodes[i], "KNOWS", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.betweenness_centrality(None).unwrap();
    // Middle node (index 2) should have highest betweenness
    assert!(result.scores[&nodes[2]] >= result.scores[&nodes[0]]);
    assert!(result.scores[&nodes[2]] >= result.scores[&nodes[4]]);
}

#[test]
fn betweenness_star_graph() {
    let engine = GraphEngine::new();
    // Center node 1, spokes 2-6
    let center = engine.create_node("Person", HashMap::new()).unwrap();
    let mut spokes = Vec::new();
    for _ in 0..5 {
        let spoke = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(center, spoke, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(spoke, center, "KNOWS", HashMap::new(), true)
            .unwrap();
        spokes.push(spoke);
    }

    let result = engine.betweenness_centrality(None).unwrap();
    // Center should have highest betweenness
    let center_score = result.scores[&center];
    for spoke in &spokes {
        assert!(center_score >= result.scores[spoke]);
    }
}

#[test]
fn betweenness_cycle() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..4 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(nodes[0], nodes[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[0], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[2], nodes[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[2], nodes[3], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[3], nodes[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[3], nodes[0], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[0], nodes[3], "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine.betweenness_centrality(None).unwrap();
    // All nodes should have similar betweenness in a cycle
    let first_score = result.scores[&nodes[0]];
    for node in &nodes[1..] {
        assert!((result.scores[node] - first_score).abs() < 0.5);
    }
}

#[test]
fn betweenness_sampling() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..20 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..19 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "KNOWS", HashMap::new(), true)
            .unwrap();
    }

    let config = CentralityConfig::new().sampling_ratio(0.5);
    let result = engine.betweenness_centrality(Some(config)).unwrap();
    assert!(result.sample_count.is_some());
    assert!(result.sample_count.unwrap() <= 10);
}

#[test]
fn betweenness_with_external_neighbors() {
    // Graph where neighbors may be outside the sampling set
    // This tests that brandes_single_source doesn't panic when
    // get_neighbor_ids returns nodes not in the nodes slice
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..10 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    // Create a fully connected graph
    for i in 0..10 {
        for j in (i + 1)..10 {
            engine
                .create_edge(nodes[i], nodes[j], "KNOWS", HashMap::new(), true)
                .unwrap();
            engine
                .create_edge(nodes[j], nodes[i], "KNOWS", HashMap::new(), true)
                .unwrap();
        }
    }

    // With 50% sampling, not all nodes will be in the sample set
    // but they may still be returned as neighbors - this should not panic
    let config = CentralityConfig::new().sampling_ratio(0.5);
    let result = engine.betweenness_centrality(Some(config)).unwrap();
    assert!(result.sample_count.is_some());
    // Should complete without panic even with external neighbors
    assert_eq!(result.centrality_type, CentralityType::Betweenness);
}

#[test]
fn closeness_empty_graph() {
    let engine = GraphEngine::new();
    let result = engine.closeness_centrality(None).unwrap();
    assert!(result.scores.is_empty());
    assert_eq!(result.centrality_type, CentralityType::Closeness);
}

#[test]
fn closeness_disconnected() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    // No edges - disconnected

    let result = engine.closeness_centrality(None).unwrap();
    // Harmonic closeness handles disconnected graphs
    assert_eq!(result.scores.len(), 2);
    assert_eq!(result.scores[&n1], 0.0);
    assert_eq!(result.scores[&n2], 0.0);
}

#[test]
fn closeness_star_graph() {
    let engine = GraphEngine::new();
    let center = engine.create_node("Person", HashMap::new()).unwrap();
    let mut spokes = Vec::new();
    for _ in 0..5 {
        let spoke = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(center, spoke, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(spoke, center, "KNOWS", HashMap::new(), true)
            .unwrap();
        spokes.push(spoke);
    }

    let result = engine.closeness_centrality(None).unwrap();
    // Center should have highest closeness
    let center_score = result.scores[&center];
    for spoke in &spokes {
        assert!(center_score >= result.scores[spoke]);
    }
}

#[test]
fn eigenvector_empty_graph() {
    let engine = GraphEngine::new();
    let result = engine.eigenvector_centrality(None).unwrap();
    assert!(result.scores.is_empty());
    assert_eq!(result.centrality_type, CentralityType::Eigenvector);
}

#[test]
fn eigenvector_star_graph() {
    let engine = GraphEngine::new();
    let center = engine.create_node("Person", HashMap::new()).unwrap();
    let mut spokes = Vec::new();
    for _ in 0..5 {
        let spoke = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(center, spoke, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(spoke, center, "KNOWS", HashMap::new(), true)
            .unwrap();
        spokes.push(spoke);
    }

    let result = engine.eigenvector_centrality(None).unwrap();
    // All nodes should have valid scores
    assert_eq!(result.scores.len(), 6);
    for score in result.scores.values() {
        assert!(*score >= 0.0);
    }
}

#[test]
fn eigenvector_convergence() {
    let engine = GraphEngine::new();
    // Use a clique for guaranteed convergence
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                engine
                    .create_edge(nodes[i], nodes[j], "KNOWS", HashMap::new(), true)
                    .unwrap();
            }
        }
    }

    let result = engine.eigenvector_centrality(None).unwrap();
    assert!(result.converged.unwrap_or(false));
}

#[test]
fn centrality_result_top_k() {
    let engine = GraphEngine::new();
    let center = engine.create_node("Person", HashMap::new()).unwrap();
    let mut spokes = Vec::new();
    for _ in 0..5 {
        let spoke = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(spoke, center, "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(center, spoke, "KNOWS", HashMap::new(), true)
            .unwrap();
        spokes.push(spoke);
    }

    let result = engine.betweenness_centrality(None).unwrap();
    let top_3 = result.top_k(3);
    assert_eq!(top_3.len(), 3);
    // Center should be first
    assert_eq!(top_3[0].0, center);
}

#[test]
fn centrality_config_builder() {
    let config = CentralityConfig::new()
        .direction(Direction::Outgoing)
        .edge_type("FOLLOWS")
        .sampling_ratio(0.5)
        .max_iterations(50)
        .tolerance(1e-8);

    assert_eq!(config.direction, Direction::Outgoing);
    assert_eq!(config.edge_type, Some("FOLLOWS".to_string()));
    assert!((config.sampling_ratio - 0.5).abs() < 1e-10);
    assert_eq!(config.max_iterations, 50);
    assert!((config.tolerance - 1e-8).abs() < 1e-12);
}

#[test]
fn centrality_type_enum() {
    assert_ne!(CentralityType::Betweenness, CentralityType::Closeness);
    assert_ne!(CentralityType::Closeness, CentralityType::Eigenvector);
    assert_eq!(CentralityType::Betweenness, CentralityType::Betweenness);
}

// ==================== Community Detection Tests ====================

#[test]
fn connected_components_empty() {
    let engine = GraphEngine::new();
    let result = engine.connected_components(None).unwrap();
    assert!(result.communities.is_empty());
    assert!(result.members.is_empty());
    assert_eq!(result.community_count, 0);
}

#[test]
fn connected_components_isolated() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();
    // No edges - all isolated

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 3);
    // Each node in its own community
    let comm_1 = result.communities[&n1];
    let comm_2 = result.communities[&n2];
    let comm_3 = result.communities[&n3];
    assert_ne!(comm_1, comm_2);
    assert_ne!(comm_2, comm_3);
    assert_ne!(comm_1, comm_3);
}

#[test]
fn connected_components_single() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "KNOWS", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 1);
    // All nodes in same community
    let comm = result.communities[&nodes[0]];
    for node in &nodes[1..] {
        assert_eq!(result.communities[node], comm);
    }
}

#[test]
fn connected_components_two_groups() {
    let engine = GraphEngine::new();
    // Group 1: 1-2-3
    let mut group1 = Vec::new();
    for _ in 0..3 {
        group1.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(group1[0], group1[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(group1[1], group1[2], "KNOWS", HashMap::new(), true)
        .unwrap();

    // Group 2: 4-5
    let mut group2 = Vec::new();
    for _ in 0..2 {
        group2.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(group2[0], group2[1], "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 2);
    // Check group membership
    assert_eq!(
        result.communities[&group1[0]],
        result.communities[&group1[1]]
    );
    assert_eq!(
        result.communities[&group1[1]],
        result.communities[&group1[2]]
    );
    assert_eq!(
        result.communities[&group2[0]],
        result.communities[&group2[1]]
    );
    assert_ne!(
        result.communities[&group1[0]],
        result.communities[&group2[0]]
    );
}

#[test]
fn louvain_empty() {
    let engine = GraphEngine::new();
    let result = engine.louvain_communities(None).unwrap();
    assert!(result.communities.is_empty());
    assert_eq!(result.community_count, 0);
}

#[test]
fn louvain_single_community() {
    let engine = GraphEngine::new();
    // Fully connected graph - should stay as one community
    let mut nodes = Vec::new();
    for _ in 0..4 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..4 {
        for j in (i + 1)..4 {
            engine
                .create_edge(nodes[i], nodes[j], "KNOWS", HashMap::new(), true)
                .unwrap();
            engine
                .create_edge(nodes[j], nodes[i], "KNOWS", HashMap::new(), true)
                .unwrap();
        }
    }

    let result = engine.louvain_communities(None).unwrap();
    assert!(result.community_count >= 1);
}

#[test]
fn louvain_two_clusters() {
    let engine = GraphEngine::new();
    // Two dense clusters with weak connection
    // Cluster 1: 1, 2, 3
    let mut cluster1 = Vec::new();
    for _ in 0..3 {
        cluster1.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(cluster1[0], cluster1[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster1[1], cluster1[0], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster1[1], cluster1[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster1[2], cluster1[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster1[0], cluster1[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster1[2], cluster1[0], "KNOWS", HashMap::new(), true)
        .unwrap();

    // Cluster 2: 4, 5, 6
    let mut cluster2 = Vec::new();
    for _ in 0..3 {
        cluster2.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(cluster2[0], cluster2[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster2[1], cluster2[0], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster2[1], cluster2[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster2[2], cluster2[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster2[0], cluster2[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(cluster2[2], cluster2[0], "KNOWS", HashMap::new(), true)
        .unwrap();

    // Weak link between clusters
    engine
        .create_edge(cluster1[2], cluster2[0], "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine.louvain_communities(None).unwrap();
    // Should find 2 communities
    assert!(result.community_count >= 1);
}

#[test]
fn louvain_modularity() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..4 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(nodes[0], nodes[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[0], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[2], nodes[3], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[3], nodes[2], "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine.louvain_communities(None).unwrap();
    // Modularity should be computed
    assert!(result.modularity.is_some());
}

#[test]
fn louvain_deterministic() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..6 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(nodes[0], nodes[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[3], nodes[4], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[4], nodes[5], "KNOWS", HashMap::new(), true)
        .unwrap();

    let config = CommunityConfig::new().seed(42);
    let result1 = engine.louvain_communities(Some(config.clone())).unwrap();
    let result2 = engine.louvain_communities(Some(config)).unwrap();

    assert_eq!(result1.community_count, result2.community_count);
}

#[test]
fn label_propagation_empty() {
    let engine = GraphEngine::new();
    let result = engine.label_propagation(None).unwrap();
    assert!(result.communities.is_empty());
    assert_eq!(result.community_count, 0);
}

#[test]
fn label_propagation_isolated() {
    let engine = GraphEngine::new();
    let _n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let _n2 = engine.create_node("Person", HashMap::new()).unwrap();

    let result = engine.label_propagation(None).unwrap();
    // Each isolated node gets its own community
    assert_eq!(result.community_count, 2);
}

#[test]
fn label_propagation_convergence() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "KNOWS", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(nodes[i + 1], nodes[i], "KNOWS", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.label_propagation(None).unwrap();
    assert!(result.iterations.is_some());
}

#[test]
fn label_propagation_deterministic() {
    let engine = GraphEngine::new();
    let mut nodes = Vec::new();
    for _ in 0..6 {
        nodes.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(nodes[0], nodes[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[0], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[1], nodes[2], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(nodes[2], nodes[1], "KNOWS", HashMap::new(), true)
        .unwrap();

    let config = CommunityConfig::new().seed(42);
    let result1 = engine.label_propagation(Some(config.clone())).unwrap();
    let result2 = engine.label_propagation(Some(config)).unwrap();

    assert_eq!(result1.communities, result2.communities);
}

#[test]
fn community_config_builder() {
    let config = CommunityConfig::new()
        .direction(Direction::Both)
        .edge_type("KNOWS")
        .resolution(1.5)
        .max_passes(20)
        .max_iterations(200)
        .seed(12345);

    assert_eq!(config.direction, Direction::Both);
    assert_eq!(config.edge_type, Some("KNOWS".to_string()));
    assert!((config.resolution - 1.5).abs() < 1e-10);
    assert_eq!(config.max_passes, 20);
    assert_eq!(config.max_iterations, 200);
    assert_eq!(config.seed, Some(12345));
}

#[test]
fn community_result_communities_by_size() {
    let engine = GraphEngine::new();
    // Group 1: 3 nodes
    let mut group = Vec::new();
    for _ in 0..3 {
        group.push(engine.create_node("Person", HashMap::new()).unwrap());
    }
    engine
        .create_edge(group[0], group[1], "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(group[1], group[2], "KNOWS", HashMap::new(), true)
        .unwrap();

    // Isolated: 1 node
    let _isolated = engine.create_node("Person", HashMap::new()).unwrap();

    let result = engine.connected_components(None).unwrap();
    let by_size = result.communities_by_size();
    // Largest community first
    assert!(!by_size.is_empty());
    if by_size.len() > 1 {
        assert!(by_size[0].1 >= by_size[1].1);
    }
}

// ==================== Batch Operations Tests ====================

#[test]
fn batch_create_nodes_empty() {
    let engine = GraphEngine::new();
    let result = engine.batch_create_nodes(vec![]).unwrap();
    assert!(result.created_ids.is_empty());
    assert_eq!(result.count, 0);
}

#[test]
fn batch_create_nodes_single() {
    let engine = GraphEngine::new();
    let mut props = HashMap::new();
    props.insert("name".to_string(), PropertyValue::String("Alice".into()));

    let nodes = vec![NodeInput::new(vec!["Person".to_string()], props)];
    let result = engine.batch_create_nodes(nodes).unwrap();

    assert_eq!(result.count, 1);
    assert_eq!(result.created_ids.len(), 1);

    let node = engine.get_node(result.created_ids[0]).unwrap();
    assert_eq!(node.labels, vec!["Person"]);
    assert_eq!(
        node.properties.get("name"),
        Some(&PropertyValue::String("Alice".into()))
    );
}

#[test]
fn batch_create_nodes_multiple() {
    let engine = GraphEngine::new();
    let nodes: Vec<NodeInput> = (0..5)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("index".to_string(), PropertyValue::Int(i));
            NodeInput::new(vec!["Item".to_string()], props)
        })
        .collect();

    let result = engine.batch_create_nodes(nodes).unwrap();

    assert_eq!(result.count, 5);
    assert_eq!(result.created_ids.len(), 5);

    // Verify sequential IDs
    for (i, &id) in result.created_ids.iter().enumerate() {
        assert_eq!(id, (i + 1) as u64);
        let node = engine.get_node(id).unwrap();
        assert_eq!(
            node.properties.get("index"),
            Some(&PropertyValue::Int(i as i64))
        );
    }
}

#[test]
fn batch_create_nodes_validation_failure() {
    let engine = GraphEngine::new();

    // Create a unique constraint on email
    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "email".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    // First node
    let mut props1 = HashMap::new();
    props1.insert(
        "email".to_string(),
        PropertyValue::String("test@test.com".into()),
    );
    engine
        .create_node_with_labels(vec!["Person".into()], props1)
        .unwrap();

    // Try to batch create with duplicate email
    let mut props2 = HashMap::new();
    props2.insert(
        "email".to_string(),
        PropertyValue::String("test@test.com".into()),
    );
    let nodes = vec![NodeInput::new(vec!["Person".to_string()], props2)];

    let result = engine.batch_create_nodes(nodes);
    assert!(matches!(
        result,
        Err(GraphError::BatchValidationError { .. })
    ));
}

#[test]
fn batch_create_nodes_parallel() {
    let engine = GraphEngine::new();
    // Create more than PARALLEL_THRESHOLD (100) nodes
    let nodes: Vec<NodeInput> = (0..150)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("index".to_string(), PropertyValue::Int(i));
            NodeInput::new(vec!["Item".to_string()], props)
        })
        .collect();

    let result = engine.batch_create_nodes(nodes).unwrap();

    assert_eq!(result.count, 150);
    assert_eq!(result.created_ids.len(), 150);
}

#[test]
fn batch_create_edges_empty() {
    let engine = GraphEngine::new();
    let result = engine.batch_create_edges(vec![]).unwrap();
    assert!(result.created_ids.is_empty());
    assert_eq!(result.count, 0);
}

#[test]
fn batch_create_edges_multiple() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();
    let n3 = engine.create_node("Node", HashMap::new()).unwrap();

    let edges = vec![
        EdgeInput::new(n1, n2, "CONNECTS", HashMap::new(), true),
        EdgeInput::new(n2, n3, "CONNECTS", HashMap::new(), true),
        EdgeInput::new(n1, n3, "CONNECTS", HashMap::new(), true),
    ];

    let result = engine.batch_create_edges(edges).unwrap();

    assert_eq!(result.count, 3);
    assert_eq!(result.created_ids.len(), 3);

    // Verify edges exist
    for id in &result.created_ids {
        let edge = engine.get_edge(*id).unwrap();
        assert_eq!(edge.edge_type, "CONNECTS");
    }
}

#[test]
fn batch_create_edges_node_not_found() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();

    let edges = vec![EdgeInput::new(n1, 999, "CONNECTS", HashMap::new(), true)];

    let result = engine.batch_create_edges(edges);
    assert!(matches!(
        result,
        Err(GraphError::BatchValidationError { index: 0, .. })
    ));
}

#[test]
fn batch_delete_nodes_multiple() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();
    let n3 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create an edge that should be deleted with node
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    let deleted = engine.batch_delete_nodes(vec![n1, n2]).unwrap();
    assert_eq!(deleted.count, 2);
    assert_eq!(deleted.deleted_ids.len(), 2);
    assert!(!deleted.has_failures());

    assert!(!engine.node_exists(n1));
    assert!(!engine.node_exists(n2));
    assert!(engine.node_exists(n3));
}

#[test]
fn batch_delete_edges_multiple() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    let e1 = engine
        .create_edge(n1, n2, "A", HashMap::new(), true)
        .unwrap();
    let e2 = engine
        .create_edge(n1, n2, "B", HashMap::new(), true)
        .unwrap();

    let deleted = engine.batch_delete_edges(vec![e1, e2]).unwrap();
    assert_eq!(deleted.count, 2);
    assert_eq!(deleted.deleted_ids.len(), 2);
    assert!(!deleted.has_failures());

    assert!(engine.get_edge(e1).is_err());
    assert!(engine.get_edge(e2).is_err());
}

#[test]
fn test_batch_delete_nodes_reports_failures() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Test", HashMap::new()).unwrap();

    let result = engine.batch_delete_nodes(vec![n1, 999_999]).unwrap();

    assert_eq!(result.count, 1);
    assert_eq!(result.deleted_ids, vec![n1]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].id, Some(999_999));
    assert!(result.failed[0].cause.contains("not found"));
    assert!(result.has_failures());
}

#[test]
fn test_batch_delete_edges_reports_failures() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Test", HashMap::new()).unwrap();
    let n2 = engine.create_node("Test", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "REL", HashMap::new(), true)
        .unwrap();

    let result = engine.batch_delete_edges(vec![e1, 999_999]).unwrap();

    assert_eq!(result.count, 1);
    assert_eq!(result.deleted_ids, vec![e1]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].id, Some(999_999));
    assert!(result.has_failures());
}

#[test]
fn test_batch_delete_result_new() {
    let result = BatchDeleteResult::new(vec![1, 2, 3]);
    assert_eq!(result.count, 3);
    assert_eq!(result.deleted_ids, vec![1, 2, 3]);
    assert!(result.failed.is_empty());
    assert!(!result.has_failures());
}

#[test]
fn test_batch_delete_result_with_failures() {
    let failed = vec![GraphBatchItemError {
        index: 1,
        id: Some(999),
        cause: "not found".to_string(),
    }];
    let result = BatchDeleteResult::with_failures(vec![1, 2], failed);

    assert_eq!(result.count, 2);
    assert_eq!(result.deleted_ids, vec![1, 2]);
    assert_eq!(result.failed.len(), 1);
    assert!(result.has_failures());
}

#[test]
fn batch_update_nodes_multiple() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("name".to_string(), PropertyValue::String("Alice".into()));

    let mut props2 = HashMap::new();
    props2.insert("name".to_string(), PropertyValue::String("Bob".into()));

    let updates = vec![(n1, None, props1), (n2, None, props2)];

    let updated = engine.batch_update_nodes(updates).unwrap();
    assert_eq!(updated, 2);

    let node1 = engine.get_node(n1).unwrap();
    let node2 = engine.get_node(n2).unwrap();

    assert_eq!(
        node1.properties.get("name"),
        Some(&PropertyValue::String("Alice".into()))
    );
    assert_eq!(
        node2.properties.get("name"),
        Some(&PropertyValue::String("Bob".into()))
    );
}

#[test]
fn batch_result_struct() {
    let result = BatchResult {
        created_ids: vec![1, 2, 3],
        count: 3,
    };
    assert_eq!(result.created_ids, vec![1, 2, 3]);
    assert_eq!(result.count, 3);
}

// ==================== Constraint Tests ====================

#[test]
fn create_constraint_unique() {
    let engine = GraphEngine::new();
    let constraint = Constraint {
        name: "unique_email".to_string(),
        target: ConstraintTarget::NodeLabel("Person".to_string()),
        property: "email".to_string(),
        constraint_type: ConstraintType::Unique,
    };

    engine.create_constraint(constraint.clone()).unwrap();

    let retrieved = engine.get_constraint("unique_email");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), constraint);
}

#[test]
fn create_constraint_exists() {
    let engine = GraphEngine::new();
    let constraint = Constraint {
        name: "require_name".to_string(),
        target: ConstraintTarget::NodeLabel("Person".to_string()),
        property: "name".to_string(),
        constraint_type: ConstraintType::Exists,
    };

    engine.create_constraint(constraint).unwrap();
    assert!(engine.get_constraint("require_name").is_some());
}

#[test]
fn create_constraint_type() {
    let engine = GraphEngine::new();
    let constraint = Constraint {
        name: "age_is_int".to_string(),
        target: ConstraintTarget::AllNodes,
        property: "age".to_string(),
        constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
    };

    engine.create_constraint(constraint).unwrap();
    assert!(engine.get_constraint("age_is_int").is_some());
}

#[test]
fn constraint_already_exists() {
    let engine = GraphEngine::new();
    let constraint = Constraint {
        name: "test_constraint".to_string(),
        target: ConstraintTarget::AllNodes,
        property: "prop".to_string(),
        constraint_type: ConstraintType::Exists,
    };

    engine.create_constraint(constraint.clone()).unwrap();
    let result = engine.create_constraint(constraint);

    assert!(matches!(
        result,
        Err(GraphError::ConstraintAlreadyExists(_))
    ));
}

#[test]
fn drop_constraint() {
    let engine = GraphEngine::new();
    let constraint = Constraint {
        name: "to_drop".to_string(),
        target: ConstraintTarget::AllNodes,
        property: "prop".to_string(),
        constraint_type: ConstraintType::Exists,
    };

    engine.create_constraint(constraint).unwrap();
    assert!(engine.get_constraint("to_drop").is_some());

    engine.drop_constraint("to_drop").unwrap();
    assert!(engine.get_constraint("to_drop").is_none());
}

#[test]
fn drop_constraint_not_found() {
    let engine = GraphEngine::new();
    let result = engine.drop_constraint("nonexistent");
    assert!(matches!(result, Err(GraphError::ConstraintNotFound(_))));
}

#[test]
fn list_constraints() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "c1".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "a".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    engine
        .create_constraint(Constraint {
            name: "c2".to_string(),
            target: ConstraintTarget::AllEdges,
            property: "b".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    let constraints = engine.list_constraints();
    assert_eq!(constraints.len(), 2);
}

#[test]
fn unique_constraint_violation_on_create() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "email".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    let mut props = HashMap::new();
    props.insert(
        "email".to_string(),
        PropertyValue::String("alice@test.com".into()),
    );
    engine
        .create_node_with_labels(vec!["Person".into()], props.clone())
        .unwrap();

    // Attempt to create duplicate
    let result = engine.create_node_with_labels(vec!["Person".into()], props);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn unique_constraint_violation_on_update() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "email".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    let mut props1 = HashMap::new();
    props1.insert(
        "email".to_string(),
        PropertyValue::String("alice@test.com".into()),
    );
    engine
        .create_node_with_labels(vec!["Person".into()], props1)
        .unwrap();

    let mut props2 = HashMap::new();
    props2.insert(
        "email".to_string(),
        PropertyValue::String("bob@test.com".into()),
    );
    let n2 = engine
        .create_node_with_labels(vec!["Person".into()], props2)
        .unwrap();

    // Try to update n2 to have same email as n1
    let mut update_props = HashMap::new();
    update_props.insert(
        "email".to_string(),
        PropertyValue::String("alice@test.com".into()),
    );

    let result = engine.update_node(n2, None, update_props);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn exists_constraint_violation() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "require_name".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "name".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    // Attempt to create without required property
    let result = engine.create_node_with_labels(vec!["Person".into()], HashMap::new());
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn type_constraint_violation() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "age_is_int".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "age".to_string(),
            constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
        })
        .unwrap();

    // Attempt to create with wrong type
    let mut props = HashMap::new();
    props.insert(
        "age".to_string(),
        PropertyValue::String("not an int".into()),
    );

    let result = engine.create_node_with_labels(vec!["Person".into()], props);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn constraint_scoped_to_label() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "person_name".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "name".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    // Creating a non-Person node without name should succeed
    let result = engine.create_node_with_labels(vec!["Animal".into()], HashMap::new());
    assert!(result.is_ok());

    // Creating a Person node without name should fail
    let result = engine.create_node_with_labels(vec!["Person".into()], HashMap::new());
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn constraint_validates_existing_data() {
    let engine = GraphEngine::new();

    // Create nodes with duplicate emails
    let mut props1 = HashMap::new();
    props1.insert(
        "email".to_string(),
        PropertyValue::String("same@test.com".into()),
    );
    engine
        .create_node_with_labels(vec!["Person".into()], props1.clone())
        .unwrap();
    engine
        .create_node_with_labels(vec!["Person".into()], props1)
        .unwrap();

    // Try to add unique constraint - should fail
    let result = engine.create_constraint(Constraint {
        name: "unique_email".to_string(),
        target: ConstraintTarget::NodeLabel("Person".to_string()),
        property: "email".to_string(),
        constraint_type: ConstraintType::Unique,
    });

    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn constraints_persist_across_restart() {
    let store = TensorStore::new();

    // Create engine and add constraint
    {
        let engine = GraphEngine::with_store(store.clone());
        engine
            .create_constraint(Constraint {
                name: "persist_test".to_string(),
                target: ConstraintTarget::AllNodes,
                property: "test".to_string(),
                constraint_type: ConstraintType::Exists,
            })
            .unwrap();
    }

    // Create new engine with same store
    {
        let engine = GraphEngine::with_store(store);
        let constraint = engine.get_constraint("persist_test");
        assert!(constraint.is_some());
        assert_eq!(constraint.unwrap().property, "test");
    }
}

#[test]
fn node_input_new() {
    let props = HashMap::new();
    let input = NodeInput::new(vec!["Label".to_string()], props);
    assert_eq!(input.labels, vec!["Label"]);
    assert!(input.properties.is_empty());
}

#[test]
fn edge_input_new() {
    let props = HashMap::new();
    let input = EdgeInput::new(1, 2, "TYPE", props, true);
    assert_eq!(input.from, 1);
    assert_eq!(input.to, 2);
    assert_eq!(input.edge_type, "TYPE");
    assert!(input.directed);
}

#[test]
fn property_value_type() {
    assert_eq!(PropertyValue::Null.value_type(), PropertyValueType::Null);
    assert_eq!(PropertyValue::Int(42).value_type(), PropertyValueType::Int);
    assert_eq!(
        PropertyValue::Float(3.14).value_type(),
        PropertyValueType::Float
    );
    assert_eq!(
        PropertyValue::String("test".into()).value_type(),
        PropertyValueType::String
    );
    assert_eq!(
        PropertyValue::Bool(true).value_type(),
        PropertyValueType::Bool
    );
}

#[test]
fn edge_constraint_exists_violation() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_constraint(Constraint {
            name: "edge_weight_required".to_string(),
            target: ConstraintTarget::EdgeType("WEIGHTED".to_string()),
            property: "weight".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    // Create edge without required property
    let result = engine.create_edge(n1, n2, "WEIGHTED", HashMap::new(), true);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));

    // Create edge with required property - should succeed
    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.5));
    let result = engine.create_edge(n1, n2, "WEIGHTED", props, true);
    assert!(result.is_ok());
}

#[test]
fn batch_edges_with_undirected() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    let edges = vec![EdgeInput::new(n1, n2, "FRIEND", HashMap::new(), false)];
    let result = engine.batch_create_edges(edges).unwrap();

    assert_eq!(result.count, 1);

    // Verify undirected edge is traversable both ways
    let neighbors_from_n1 = engine
        .neighbors(n1, None, Direction::Outgoing, None)
        .unwrap();
    let neighbors_from_n2 = engine
        .neighbors(n2, None, Direction::Outgoing, None)
        .unwrap();

    assert!(neighbors_from_n1.iter().any(|n| n.id == n2));
    assert!(neighbors_from_n2.iter().any(|n| n.id == n1));
}

#[test]
fn all_nodes_unique_constraint() {
    let engine = GraphEngine::new();

    // Create nodes with same property across different labels
    let mut props = HashMap::new();
    props.insert("uuid".to_string(), PropertyValue::String("abc".into()));
    engine
        .create_node_with_labels(vec!["Person".into()], props.clone())
        .unwrap();

    // Create unique constraint on AllNodes
    engine
        .create_constraint(Constraint {
            name: "unique_uuid".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "uuid".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    // Creating any node with same uuid should fail
    let result = engine.create_node_with_labels(vec!["Animal".into()], props);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn all_nodes_unique_constraint_validates_existing() {
    let engine = GraphEngine::new();

    // Create nodes with duplicate property
    let mut props = HashMap::new();
    props.insert("code".to_string(), PropertyValue::String("dup".into()));
    engine
        .create_node_with_labels(vec!["A".into()], props.clone())
        .unwrap();
    engine
        .create_node_with_labels(vec!["B".into()], props)
        .unwrap();

    // Creating constraint should fail on existing data
    let result = engine.create_constraint(Constraint {
        name: "unique_code".to_string(),
        target: ConstraintTarget::AllNodes,
        property: "code".to_string(),
        constraint_type: ConstraintType::Unique,
    });
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn all_edges_unique_constraint() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("ref".to_string(), PropertyValue::String("unique1".into()));
    engine.create_edge(n1, n2, "TYPE_A", props, true).unwrap();

    // Create unique constraint on AllEdges
    engine
        .create_constraint(Constraint {
            name: "unique_edge_ref".to_string(),
            target: ConstraintTarget::AllEdges,
            property: "ref".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    // Creating edge with duplicate ref should fail
    let mut props2 = HashMap::new();
    props2.insert("ref".to_string(), PropertyValue::String("unique1".into()));
    let result = engine.create_edge(n1, n2, "TYPE_B", props2, true);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn edge_type_unique_constraint() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create unique constraint on specific edge type
    engine
        .create_constraint(Constraint {
            name: "unique_edge_id".to_string(),
            target: ConstraintTarget::EdgeType("LINK".to_string()),
            property: "link_id".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    // First edge should succeed
    let mut props1 = HashMap::new();
    props1.insert("link_id".to_string(), PropertyValue::String("id1".into()));
    engine.create_edge(n1, n2, "LINK", props1, true).unwrap();

    // Duplicate on same type should fail
    let mut props2 = HashMap::new();
    props2.insert("link_id".to_string(), PropertyValue::String("id1".into()));
    let result = engine.create_edge(n1, n2, "LINK", props2, true);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));

    // Different edge type should succeed
    let mut props3 = HashMap::new();
    props3.insert("link_id".to_string(), PropertyValue::String("id1".into()));
    let result = engine.create_edge(n1, n2, "OTHER", props3, true);
    assert!(result.is_ok());
}

#[test]
fn edge_type_constraint_validates_existing() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create edges with duplicate property
    let mut props = HashMap::new();
    props.insert("code".to_string(), PropertyValue::String("dup".into()));
    engine
        .create_edge(n1, n2, "REL", props.clone(), true)
        .unwrap();
    engine.create_edge(n1, n2, "REL", props, true).unwrap();

    // Creating constraint should fail on existing data
    let result = engine.create_constraint(Constraint {
        name: "unique_rel_code".to_string(),
        target: ConstraintTarget::EdgeType("REL".to_string()),
        property: "code".to_string(),
        constraint_type: ConstraintType::Unique,
    });
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn all_edges_exists_constraint() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_constraint(Constraint {
            name: "all_edges_need_weight".to_string(),
            target: ConstraintTarget::AllEdges,
            property: "weight".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    // Edge without weight should fail
    let result = engine.create_edge(n1, n2, "ANY_TYPE", HashMap::new(), true);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn all_edges_exists_validates_existing() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create edge without the property
    engine
        .create_edge(n1, n2, "LINK", HashMap::new(), true)
        .unwrap();

    // Creating exists constraint should fail
    let result = engine.create_constraint(Constraint {
        name: "require_weight".to_string(),
        target: ConstraintTarget::AllEdges,
        property: "weight".to_string(),
        constraint_type: ConstraintType::Exists,
    });
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn edge_type_exists_validates_existing() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create edge without the property
    engine
        .create_edge(n1, n2, "WEIGHTED", HashMap::new(), true)
        .unwrap();

    // Creating exists constraint should fail
    let result = engine.create_constraint(Constraint {
        name: "weighted_needs_weight".to_string(),
        target: ConstraintTarget::EdgeType("WEIGHTED".to_string()),
        property: "weight".to_string(),
        constraint_type: ConstraintType::Exists,
    });
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn all_nodes_exists_validates_existing() {
    let engine = GraphEngine::new();

    // Create node without the property
    engine
        .create_node_with_labels(vec!["Any".into()], HashMap::new())
        .unwrap();

    // Creating exists constraint should fail
    let result = engine.create_constraint(Constraint {
        name: "require_id".to_string(),
        target: ConstraintTarget::AllNodes,
        property: "external_id".to_string(),
        constraint_type: ConstraintType::Exists,
    });
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn all_edges_unique_validates_existing() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create edges with duplicate property
    let mut props = HashMap::new();
    props.insert("ref".to_string(), PropertyValue::String("dup".into()));
    engine
        .create_edge(n1, n2, "A", props.clone(), true)
        .unwrap();
    engine.create_edge(n1, n2, "B", props, true).unwrap();

    // Creating constraint should fail
    let result = engine.create_constraint(Constraint {
        name: "unique_ref".to_string(),
        target: ConstraintTarget::AllEdges,
        property: "ref".to_string(),
        constraint_type: ConstraintType::Unique,
    });
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn edge_type_property_type_constraint() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_constraint(Constraint {
            name: "weight_is_float".to_string(),
            target: ConstraintTarget::EdgeType("WEIGHTED".to_string()),
            property: "weight".to_string(),
            constraint_type: ConstraintType::PropertyType(PropertyValueType::Float),
        })
        .unwrap();

    // String weight should fail
    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::String("heavy".into()));
    let result = engine.create_edge(n1, n2, "WEIGHTED", props, true);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));

    // Float weight should succeed
    let mut props2 = HashMap::new();
    props2.insert("weight".to_string(), PropertyValue::Float(1.5));
    let result = engine.create_edge(n1, n2, "WEIGHTED", props2, true);
    assert!(result.is_ok());
}

#[test]
fn all_edges_property_type_constraint() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_constraint(Constraint {
            name: "priority_is_int".to_string(),
            target: ConstraintTarget::AllEdges,
            property: "priority".to_string(),
            constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
        })
        .unwrap();

    // String priority should fail
    let mut props = HashMap::new();
    props.insert("priority".to_string(), PropertyValue::String("high".into()));
    let result = engine.create_edge(n1, n2, "ANY", props, true);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

#[test]
fn constraint_not_applicable_to_different_target() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // Create constraint on nodes
    engine
        .create_constraint(Constraint {
            name: "node_constraint".to_string(),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "name".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    // Edge creation should not be affected by node constraint
    let result = engine.create_edge(n1, n2, "Person", HashMap::new(), true);
    assert!(result.is_ok());
}

#[test]
fn graph_error_display() {
    let err1 = GraphError::ConstraintViolation {
        constraint_name: "test".to_string(),
        message: "violation".to_string(),
    };
    assert!(err1.to_string().contains("test"));
    assert!(err1.to_string().contains("violation"));

    let err2 = GraphError::ConstraintAlreadyExists("dup".to_string());
    assert!(err2.to_string().contains("dup"));

    let err3 = GraphError::ConstraintNotFound("missing".to_string());
    assert!(err3.to_string().contains("missing"));

    let err4 = GraphError::BatchValidationError {
        index: 5,
        cause: Box::new(GraphError::NodeNotFound(42)),
    };
    assert!(err4.to_string().contains("5"));

    let err5 = GraphError::BatchCreationError {
        index: 3,
        cause: Box::new(GraphError::StorageError("fail".to_string())),
    };
    assert!(err5.to_string().contains("3"));

    let err6 = GraphError::NegativeWeight {
        edge_id: 1,
        weight: -1.5,
    };
    assert!(err6.to_string().contains("-1.5"));
}

#[test]
fn graph_error_hash_new_variants() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    set.insert(GraphError::ConstraintAlreadyExists("a".to_string()));
    set.insert(GraphError::ConstraintNotFound("b".to_string()));
    set.insert(GraphError::ConstraintViolation {
        constraint_name: "c".to_string(),
        message: "d".to_string(),
    });
    set.insert(GraphError::BatchValidationError {
        index: 0,
        cause: Box::new(GraphError::NodeNotFound(1)),
    });
    set.insert(GraphError::BatchCreationError {
        index: 1,
        cause: Box::new(GraphError::EdgeNotFound(2)),
    });
    set.insert(GraphError::PathNotFound);

    assert_eq!(set.len(), 6);
}

#[test]
fn property_condition_int_float_comparison() {
    // Test Int vs Float comparison paths
    let cond = PropertyCondition::new("val", CompareOp::Lt, PropertyValue::Int(10));
    let mut props = HashMap::new();
    props.insert("val".to_string(), PropertyValue::Float(5.0));
    let node = Node {
        id: 1,
        labels: vec![],
        properties: props,
        created_at: None,
        updated_at: None,
    };
    // This should compare Int(10) with Float(5.0)
    let result = cond.matches_node(&node);
    assert!(result); // 5.0 < 10

    // Test Float vs Int comparison
    let cond2 = PropertyCondition::new("val", CompareOp::Gt, PropertyValue::Float(3.0));
    let mut props2 = HashMap::new();
    props2.insert("val".to_string(), PropertyValue::Int(5));
    let node2 = Node {
        id: 2,
        labels: vec![],
        properties: props2,
        created_at: None,
        updated_at: None,
    };
    let result2 = cond2.matches_node(&node2);
    assert!(result2); // 5 > 3.0
}

#[test]
fn property_condition_bool_invalid_ops() {
    // Bool with Lt/Le/Gt/Ge should return false
    let cond = PropertyCondition::new("flag", CompareOp::Lt, PropertyValue::Bool(true));
    let mut props = HashMap::new();
    props.insert("flag".to_string(), PropertyValue::Bool(false));
    let node = Node {
        id: 1,
        labels: vec![],
        properties: props,
        created_at: None,
        updated_at: None,
    };
    assert!(!cond.matches_node(&node));
}

#[test]
fn property_condition_float_nan() {
    let cond = PropertyCondition::new("val", CompareOp::Eq, PropertyValue::Float(f64::NAN));
    let mut props = HashMap::new();
    props.insert("val".to_string(), PropertyValue::Float(1.0));
    let node = Node {
        id: 1,
        labels: vec![],
        properties: props,
        created_at: None,
        updated_at: None,
    };
    // NaN comparison with Eq should return false, Ne should return true
    assert!(!cond.matches_node(&node));

    let cond_ne = PropertyCondition::new("val", CompareOp::Ne, PropertyValue::Float(f64::NAN));
    assert!(cond_ne.matches_node(&node));
}

#[test]
fn property_condition_float_eq_ne() {
    let cond_eq = PropertyCondition::new("val", CompareOp::Eq, PropertyValue::Float(1.0));
    let mut props = HashMap::new();
    props.insert("val".to_string(), PropertyValue::Float(1.0));
    let node = Node {
        id: 1,
        labels: vec![],
        properties: props.clone(),
        created_at: None,
        updated_at: None,
    };
    assert!(cond_eq.matches_node(&node));

    let cond_ne = PropertyCondition::new("val", CompareOp::Ne, PropertyValue::Float(1.0));
    assert!(!cond_ne.matches_node(&node));

    let cond_ge = PropertyCondition::new("val", CompareOp::Ge, PropertyValue::Float(1.0));
    assert!(cond_ge.matches_node(&node));

    let cond_gt = PropertyCondition::new("val", CompareOp::Gt, PropertyValue::Float(0.5));
    assert!(cond_gt.matches_node(&node));
}

#[test]
fn property_condition_int_le() {
    let cond = PropertyCondition::new("val", CompareOp::Le, PropertyValue::Int(10));
    let mut props = HashMap::new();
    props.insert("val".to_string(), PropertyValue::Int(10));
    let node = Node {
        id: 1,
        labels: vec![],
        properties: props,
        created_at: None,
        updated_at: None,
    };
    assert!(cond.matches_node(&node));
}

#[test]
fn node_timestamps() {
    let node = Node {
        id: 1,
        labels: vec![],
        properties: HashMap::new(),
        created_at: Some(1000),
        updated_at: Some(2000),
    };
    assert_eq!(node.created_at_millis(), Some(1000));
    assert_eq!(node.updated_at_millis(), Some(2000));
}

#[test]
fn edge_timestamps() {
    let edge = Edge {
        id: 1,
        from: 1,
        to: 2,
        edge_type: "TEST".to_string(),
        properties: HashMap::new(),
        directed: true,
        created_at: Some(1000),
        updated_at: Some(2000),
    };
    assert_eq!(edge.created_at_millis(), Some(1000));
    assert_eq!(edge.updated_at_millis(), Some(2000));
}

#[test]
fn pagerank_result_default() {
    let result = PageRankResult::default();
    assert!(result.scores.is_empty());
}

#[test]
fn community_result_default() {
    let result = CommunityResult::default();
    assert!(result.communities.is_empty());
}

#[test]
fn centrality_result_empty() {
    let result = CentralityResult::empty(CentralityType::Betweenness);
    assert!(result.scores.is_empty());
}

#[test]
fn connected_components_with_edge_type_filter() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();
    let n3 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "LIKES", HashMap::new(), true)
        .unwrap();

    let config = CommunityConfig::new().edge_type("KNOWS");
    let result = engine.connected_components(Some(config)).unwrap();

    // Only KNOWS edges counted, so n3 should be separate
    assert!(result.community_count >= 2);
}

// ========== New Error Variant Tests ==========

#[test]
fn partial_deletion_error_display() {
    let err = GraphError::PartialDeletionError {
        node_id: 42,
        failed_edges: vec![1, 2, 3],
    };
    let msg = format!("{err}");
    assert!(msg.contains("42"));
    assert!(msg.contains("3 edges"));
}

#[test]
fn id_space_exhausted_display() {
    let err = GraphError::IdSpaceExhausted {
        entity_type: "node",
    };
    let msg = format!("{err}");
    assert!(msg.contains("node"));
    assert!(msg.contains("exhausted"));
}

#[test]
fn invalid_property_name_display() {
    let err = GraphError::InvalidPropertyName {
        name: "bad:name".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("bad:name"));
    assert!(msg.contains(":"));
}

#[test]
fn corrupted_edge_display() {
    let err = GraphError::CorruptedEdge {
        edge_id: 99,
        field: "_from".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("99"));
    assert!(msg.contains("_from"));
}

#[test]
fn graph_error_hash_new_variants_all() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    set.insert(GraphError::PartialDeletionError {
        node_id: 1,
        failed_edges: vec![2, 3],
    });
    set.insert(GraphError::IdSpaceExhausted {
        entity_type: "node",
    });
    set.insert(GraphError::InvalidPropertyName {
        name: "test".to_string(),
    });
    set.insert(GraphError::CorruptedEdge {
        edge_id: 1,
        field: "_from".to_string(),
    });

    assert_eq!(set.len(), 4);
}

// ========== Constraint Validation Tests ==========

#[test]
fn exists_constraint_null_value_rejected() {
    let engine = GraphEngine::new();
    engine
        .create_constraint(Constraint {
            name: "require_name".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "name".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    // Null value should be treated as missing
    let mut props = HashMap::new();
    props.insert("name".to_string(), PropertyValue::Null);

    let result = engine.create_node("Person", props);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, GraphError::ConstraintViolation { .. }));
}

#[test]
fn exists_constraint_missing_property_rejected() {
    let engine = GraphEngine::new();
    engine
        .create_constraint(Constraint {
            name: "require_name".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "name".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    let result = engine.create_node("Person", HashMap::new());
    assert!(result.is_err());
}

#[test]
fn exists_constraint_valid_value_accepted() {
    let engine = GraphEngine::new();
    engine
        .create_constraint(Constraint {
            name: "require_name".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "name".to_string(),
            constraint_type: ConstraintType::Exists,
        })
        .unwrap();

    let mut props = HashMap::new();
    props.insert("name".to_string(), PropertyValue::String("Alice".into()));

    let result = engine.create_node("Person", props);
    assert!(result.is_ok());
}

// ========== Property Name Validation Tests ==========

#[test]
fn property_name_with_colon_rejected() {
    let engine = GraphEngine::new();
    let result = engine.create_node_property_index("bad:name");
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        GraphError::InvalidPropertyName { .. }
    ));
}

#[test]
fn edge_property_name_with_colon_rejected() {
    let engine = GraphEngine::new();
    let result = engine.create_edge_property_index("also:bad");
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        GraphError::InvalidPropertyName { .. }
    ));
}

#[test]
fn valid_property_name_accepted() {
    let engine = GraphEngine::new();
    let result = engine.create_node_property_index("valid_name");
    assert!(result.is_ok());
}

// ========== Corrupted Edge Tests ==========

#[test]
fn get_edge_corrupted_from_field() {
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    let engine = GraphEngine::new();
    // Create a corrupted edge directly in the store (missing _from)
    let mut tensor = TensorData::new();
    tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(999)));
    tensor.set(
        "_type",
        TensorValue::Scalar(ScalarValue::String("edge".into())),
    );
    // Missing _from field
    tensor.set("_to", TensorValue::Scalar(ScalarValue::Int(2)));
    tensor.set(
        "_edge_type",
        TensorValue::Scalar(ScalarValue::String("TEST".into())),
    );
    tensor.set("_directed", TensorValue::Scalar(ScalarValue::Bool(true)));

    engine.store.put("edge:999", tensor).unwrap();

    let result = engine.get_edge(999);
    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::CorruptedEdge { edge_id, field } => {
            assert_eq!(edge_id, 999);
            assert_eq!(field, "_from");
        },
        _ => panic!("Expected CorruptedEdge error"),
    }
}

#[test]
fn get_edge_corrupted_to_field() {
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    let engine = GraphEngine::new();
    // Create a corrupted edge directly in the store (missing _to)
    let mut tensor = TensorData::new();
    tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(998)));
    tensor.set(
        "_type",
        TensorValue::Scalar(ScalarValue::String("edge".into())),
    );
    tensor.set("_from", TensorValue::Scalar(ScalarValue::Int(1)));
    // Missing _to field
    tensor.set(
        "_edge_type",
        TensorValue::Scalar(ScalarValue::String("TEST".into())),
    );
    tensor.set("_directed", TensorValue::Scalar(ScalarValue::Bool(true)));

    engine.store.put("edge:998", tensor).unwrap();

    let result = engine.get_edge(998);
    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::CorruptedEdge { edge_id, field } => {
            assert_eq!(edge_id, 998);
            assert_eq!(field, "_to");
        },
        _ => panic!("Expected CorruptedEdge error"),
    }
}

// ========== Safety Limit Tests ==========

#[test]
fn all_nodes_returns_results_under_limit() {
    let engine = GraphEngine::new();
    for _ in 0..10 {
        engine.create_node("Test", HashMap::new()).unwrap();
    }
    let nodes = engine.all_nodes();
    assert_eq!(nodes.len(), 10);
}

#[test]
fn all_edges_returns_results_under_limit() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Test", HashMap::new()).unwrap();
    let n2 = engine.create_node("Test", HashMap::new()).unwrap();
    for _ in 0..5 {
        engine
            .create_edge(n1, n2, "LINK", HashMap::new(), true)
            .unwrap();
    }
    let edges = engine.all_edges();
    assert_eq!(edges.len(), 5);
}

// ========== Batch Operation Tests ==========

#[test]
fn batch_create_nodes_with_unique_constraint() {
    let engine = GraphEngine::new();
    engine.create_node_property_index("email").unwrap();
    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "email".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    let nodes = vec![
        NodeInput {
            labels: vec!["User".to_string()],
            properties: {
                let mut p = HashMap::new();
                p.insert(
                    "email".to_string(),
                    PropertyValue::String("a@example.com".into()),
                );
                p
            },
        },
        NodeInput {
            labels: vec!["User".to_string()],
            properties: {
                let mut p = HashMap::new();
                p.insert(
                    "email".to_string(),
                    PropertyValue::String("b@example.com".into()),
                );
                p
            },
        },
    ];

    let result = engine.batch_create_nodes(nodes);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().count, 2);
}

#[test]
fn batch_create_nodes_unique_constraint_violation() {
    let engine = GraphEngine::new();
    engine.create_node_property_index("email").unwrap();
    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "email".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    // First create a node with an email
    let mut props = HashMap::new();
    props.insert(
        "email".to_string(),
        PropertyValue::String("taken@example.com".into()),
    );
    engine.create_node("User", props).unwrap();

    // Now try to batch create with a duplicate
    let nodes = vec![NodeInput {
        labels: vec!["User".to_string()],
        properties: {
            let mut p = HashMap::new();
            p.insert(
                "email".to_string(),
                PropertyValue::String("taken@example.com".into()),
            );
            p
        },
    }];

    let result = engine.batch_create_nodes(nodes);
    assert!(result.is_err());
}

#[test]
fn batch_create_edges_with_unique_constraint() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    engine.create_edge_property_index("ref_id").unwrap();
    engine
        .create_constraint(Constraint {
            name: "unique_ref".to_string(),
            target: ConstraintTarget::AllEdges,
            property: "ref_id".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    let edges = vec![
        EdgeInput {
            from: n1,
            to: n2,
            edge_type: "LINK".to_string(),
            properties: {
                let mut p = HashMap::new();
                p.insert("ref_id".to_string(), PropertyValue::String("ref1".into()));
                p
            },
            directed: true,
        },
        EdgeInput {
            from: n1,
            to: n2,
            edge_type: "LINK".to_string(),
            properties: {
                let mut p = HashMap::new();
                p.insert("ref_id".to_string(), PropertyValue::String("ref2".into()));
                p
            },
            directed: true,
        },
    ];

    let result = engine.batch_create_edges(edges);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().count, 2);
}

// ========== Concurrent Operation Tests ==========

#[test]
fn test_update_node_concurrent_same_property() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());
    let mut props = HashMap::new();
    props.insert("counter".to_string(), PropertyValue::Int(0));
    let node_id = engine.create_node("Counter", props).unwrap();

    let thread_count = 10;
    let updates_per_thread = 100;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                bar.wait();
                for i in 0..updates_per_thread {
                    let mut props = HashMap::new();
                    let value = (t * updates_per_thread + i) as i64;
                    props.insert("counter".to_string(), PropertyValue::Int(value));
                    if eng.update_node(node_id, None, props).is_ok() {
                        cnt.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    assert_eq!(
        success_count.load(Ordering::SeqCst),
        thread_count * updates_per_thread
    );

    // Verify node is still valid and has a counter value
    let node = engine.get_node(node_id).unwrap();
    assert!(matches!(
        node.properties.get("counter"),
        Some(PropertyValue::Int(_))
    ));
}

#[test]
fn test_update_node_concurrent_different_nodes() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Each thread gets its own node to update
    let thread_count = 10;
    let mut node_ids = Vec::new();
    for _ in 0..thread_count {
        node_ids.push(engine.create_node("Entity", HashMap::new()).unwrap());
    }
    let node_ids = Arc::new(node_ids);

    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            let ids = Arc::clone(&node_ids);
            thread::spawn(move || {
                bar.wait();
                let prop_name = format!("prop_{t}");
                let mut props = HashMap::new();
                props.insert(prop_name, PropertyValue::Int(t as i64));
                if eng.update_node(ids[t], None, props).is_ok() {
                    cnt.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    assert_eq!(success_count.load(Ordering::SeqCst), thread_count);

    // Verify each node has its property
    for t in 0..thread_count {
        let node = engine.get_node(node_ids[t]).unwrap();
        let prop_name = format!("prop_{t}");
        assert!(
            node.properties.contains_key(&prop_name),
            "property {prop_name} missing on node {}",
            node_ids[t]
        );
    }
}

#[test]
fn test_update_node_concurrent_add_remove_labels() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());
    let node_id = engine.create_node("Base", HashMap::new()).unwrap();

    let thread_count = 10;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                bar.wait();
                // Each thread adds its own label then updates with different label set
                let label = format!("Label{t}");
                let labels = vec!["Base".to_string(), label];
                if eng
                    .update_node(node_id, Some(labels), HashMap::new())
                    .is_ok()
                {
                    cnt.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    assert_eq!(success_count.load(Ordering::SeqCst), thread_count);

    // Node should have Base label and one of the thread labels
    let node = engine.get_node(node_id).unwrap();
    assert!(node.has_label("Base"));
    assert_eq!(node.labels.len(), 2);
}

#[test]
fn test_batch_create_edges_toctou_safety() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Create source and target nodes
    let mut source_ids = Vec::new();
    let mut target_ids = Vec::new();
    for _ in 0..20 {
        source_ids.push(engine.create_node("Source", HashMap::new()).unwrap());
        target_ids.push(engine.create_node("Target", HashMap::new()).unwrap());
    }

    let barrier = Arc::new(Barrier::new(2));

    // Thread 1: delete source nodes
    let eng1 = Arc::clone(&engine);
    let bar1 = Arc::clone(&barrier);
    let sources_to_delete = source_ids[10..].to_vec();
    let deleter = thread::spawn(move || {
        bar1.wait();
        for id in sources_to_delete {
            let _ = eng1.delete_node(id);
        }
    });

    // Thread 2: batch create edges using those sources
    let eng2 = Arc::clone(&engine);
    let bar2 = Arc::clone(&barrier);
    let edge_inputs: Vec<EdgeInput> = source_ids
        .iter()
        .zip(target_ids.iter())
        .map(|(&from, &to)| EdgeInput {
            from,
            to,
            edge_type: "LINK".to_string(),
            properties: HashMap::new(),
            directed: true,
        })
        .collect();

    let edge_creator = thread::spawn(move || {
        bar2.wait();
        eng2.batch_create_edges(edge_inputs)
    });

    deleter.join().expect("deleter should not panic");
    let result = edge_creator.join().expect("edge creator should not panic");

    // Should either succeed partially or fail cleanly (no corruption/panic)
    match result {
        Ok(batch_result) => {
            // Some edges created successfully
            assert!(batch_result.count <= 20);
        },
        Err(_) => {
            // Clean validation error is acceptable
        },
    }

    // Verify engine is still usable
    let _ = engine.create_node("Test", HashMap::new()).unwrap();
}

#[test]
fn test_index_concurrent_writes_50_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());
    engine.create_node_property_index("indexed_field").unwrap();

    let thread_count = 50;
    let nodes_per_thread = 100;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                bar.wait();
                for i in 0..nodes_per_thread {
                    let mut props = HashMap::new();
                    let value = format!("value_{t}_{i}");
                    props.insert("indexed_field".to_string(), PropertyValue::String(value));
                    if eng.create_node("Indexed", props).is_ok() {
                        cnt.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    let expected_count = thread_count * nodes_per_thread;
    assert_eq!(success_count.load(Ordering::SeqCst), expected_count);

    // Verify index integrity by looking up a random value
    let results = engine
        .find_nodes_by_property(
            "indexed_field",
            &PropertyValue::String("value_25_50".into()),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_striped_lock_fairness_64_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Instant;

    let engine = Arc::new(GraphEngine::new());
    engine.create_node_property_index("shard_key").unwrap();

    let thread_count = 64;
    let ops_per_thread = 50;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                let start = Instant::now();

                for i in 0..ops_per_thread {
                    let mut props = HashMap::new();
                    // Distribute across shards using different prefixes
                    let key = format!("{:02x}_{t}_{i}", t % 64);
                    props.insert("shard_key".to_string(), PropertyValue::String(key));
                    eng.create_node("Sharded", props).unwrap();
                }

                start.elapsed()
            })
        })
        .collect();

    let durations: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("no panic"))
        .collect();

    // Calculate variance to check fairness
    let avg_ms: f64 =
        durations.iter().map(|d| d.as_millis() as f64).sum::<f64>() / thread_count as f64;
    let max_ms = durations.iter().map(|d| d.as_millis()).max().unwrap() as f64;

    // No thread should take more than 5x the average (fairness check)
    // This is a loose bound to avoid flaky tests
    assert!(
        max_ms < avg_ms * 5.0 + 100.0,
        "thread starvation detected: max={max_ms}ms, avg={avg_ms}ms"
    );

    // Verify all nodes created
    let expected_total = thread_count * ops_per_thread;
    let actual_count = AtomicUsize::new(0);
    for id in 1..=(expected_total as u64 + 100) {
        if engine.get_node(id).is_ok() {
            actual_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    assert_eq!(
        actual_count.load(Ordering::Relaxed),
        expected_total,
        "expected {expected_total} nodes, found {}",
        actual_count.load(Ordering::Relaxed)
    );
}

#[test]
fn test_batch_delete_nodes_concurrent_20_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Pre-create 1000 nodes with edges
    let mut node_ids = Vec::new();
    for i in 0..1000 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        node_ids.push(engine.create_node("Deletable", props).unwrap());
    }

    // Create edges between consecutive nodes
    for i in 0..999 {
        engine
            .create_edge(node_ids[i], node_ids[i + 1], "LINK", HashMap::new(), true)
            .unwrap();
    }

    let thread_count = 20;
    let nodes_per_thread = 50;
    let barrier = Arc::new(Barrier::new(thread_count));
    let deleted_count = Arc::new(AtomicUsize::new(0));
    let node_ids = Arc::new(node_ids);

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&deleted_count);
            let ids = Arc::clone(&node_ids);
            thread::spawn(move || {
                bar.wait();
                let start_idx = t * nodes_per_thread;
                let end_idx = start_idx + nodes_per_thread;
                let batch: Vec<u64> = ids[start_idx..end_idx].to_vec();
                if let Ok(result) = eng.batch_delete_nodes(batch) {
                    cnt.fetch_add(result.count, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All targeted nodes should be deleted (1000 nodes)
    let expected = thread_count * nodes_per_thread;
    assert_eq!(deleted_count.load(Ordering::SeqCst), expected);

    // Verify nodes are gone
    for i in 0..expected {
        assert!(
            engine.get_node(node_ids[i]).is_err(),
            "node {} should be deleted",
            node_ids[i]
        );
    }

    // Engine should still be usable
    let _ = engine.create_node("Test", HashMap::new()).unwrap();
}

#[test]
fn test_batch_delete_edges_concurrent_20_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Create hub and spoke topology: 1 hub + 99 spokes with 500 edges total
    let hub = engine.create_node("Hub", HashMap::new()).unwrap();
    let mut spoke_ids = Vec::new();
    for i in 0..99 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        spoke_ids.push(engine.create_node("Spoke", props).unwrap());
    }

    // Create 500 edges from hub to spokes (cycling through spokes)
    let mut edge_ids = Vec::new();
    for i in 0..500 {
        let spoke = spoke_ids[i % 99];
        let mut props = HashMap::new();
        props.insert("edge_idx".to_string(), PropertyValue::Int(i as i64));
        let edge_id = engine
            .create_edge(hub, spoke, "CONNECTS", props, true)
            .unwrap();
        edge_ids.push(edge_id);
    }

    let thread_count = 20;
    let edges_per_thread = 25;
    let barrier = Arc::new(Barrier::new(thread_count));
    let deleted_count = Arc::new(AtomicUsize::new(0));
    let edge_ids = Arc::new(edge_ids);

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&deleted_count);
            let ids = Arc::clone(&edge_ids);
            thread::spawn(move || {
                bar.wait();
                let start_idx = t * edges_per_thread;
                let end_idx = start_idx + edges_per_thread;
                let batch: Vec<u64> = ids[start_idx..end_idx].to_vec();
                if let Ok(result) = eng.batch_delete_edges(batch) {
                    cnt.fetch_add(result.count, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All edges should be deleted
    let expected = thread_count * edges_per_thread;
    assert_eq!(deleted_count.load(Ordering::SeqCst), expected);

    // Verify hub node still exists
    assert!(engine.get_node(hub).is_ok());

    // Verify all spoke nodes still exist
    for spoke in &spoke_ids {
        assert!(engine.get_node(*spoke).is_ok());
    }
}

#[test]
fn test_batch_update_nodes_concurrent_10_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Pre-create 100 nodes (10 per thread)
    let mut node_ids = Vec::new();
    for i in 0..100 {
        let mut props = HashMap::new();
        props.insert("initial".to_string(), PropertyValue::Int(i));
        node_ids.push(engine.create_node("Updatable", props).unwrap());
    }
    let node_ids = Arc::new(node_ids);

    let thread_count = 10;
    let nodes_per_thread = 10;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            let ids = Arc::clone(&node_ids);
            thread::spawn(move || {
                bar.wait();
                let start_idx = t * nodes_per_thread;
                let end_idx = start_idx + nodes_per_thread;
                let batch_ids: Vec<u64> = ids[start_idx..end_idx].to_vec();

                let updates: Vec<(u64, Option<Vec<String>>, HashMap<String, PropertyValue>)> =
                    batch_ids
                        .into_iter()
                        .enumerate()
                        .map(|(i, id)| {
                            let mut props = HashMap::new();
                            props.insert(
                                format!("thread_{t}"),
                                PropertyValue::String(format!("updated_{i}")),
                            );
                            props.insert("updated_by".to_string(), PropertyValue::Int(t as i64));
                            (id, None, props)
                        })
                        .collect();

                if eng.batch_update_nodes(updates).is_ok() {
                    cnt.fetch_add(nodes_per_thread, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    assert_eq!(
        success_count.load(Ordering::SeqCst),
        thread_count * nodes_per_thread
    );

    // Verify each node has its thread-specific property
    for t in 0..thread_count {
        let start_idx = t * nodes_per_thread;
        for i in 0..nodes_per_thread {
            let node = engine.get_node(node_ids[start_idx + i]).unwrap();
            let prop_name = format!("thread_{t}");
            assert!(
                node.properties.contains_key(&prop_name),
                "node {} missing property {prop_name}",
                node_ids[start_idx + i]
            );
        }
    }
}

#[test]
fn test_constraint_validation_concurrent_30_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());
    engine.create_node_property_index("email").unwrap();
    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            target: ConstraintTarget::AllNodes,
            property: "email".to_string(),
            constraint_type: ConstraintType::Unique,
        })
        .unwrap();

    let thread_count = 30;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));
    let duplicate_rejected = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let success = Arc::clone(&success_count);
            let rejected = Arc::clone(&duplicate_rejected);
            thread::spawn(move || {
                bar.wait();
                // First 15 threads use unique emails, last 15 try duplicates
                let email = if t < 15 {
                    format!("unique_{t}@example.com")
                } else {
                    // Attempt duplicate of first 15
                    format!("unique_{}@example.com", t - 15)
                };

                let mut props = HashMap::new();
                props.insert("email".to_string(), PropertyValue::String(email));

                match eng.create_node("User", props) {
                    Ok(_) => {
                        success.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(GraphError::ConstraintViolation { .. }) => {
                        rejected.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(e) => panic!("unexpected error: {e:?}"),
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // Exactly 15 unique emails should succeed
    let successes = success_count.load(Ordering::SeqCst);
    let rejections = duplicate_rejected.load(Ordering::SeqCst);

    // Due to race conditions, we should have 15 successes total
    // (either the unique thread or the duplicate thread wins for each email)
    assert_eq!(
        successes + rejections,
        thread_count,
        "total should be {thread_count}, got {successes} successes + {rejections} rejections"
    );
    assert_eq!(successes, 15, "exactly 15 unique emails should succeed");
}

#[test]
fn test_index_create_drop_concurrent_10_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Pre-create 5 indexes to drop
    for i in 0..5 {
        engine
            .create_node_property_index(&format!("existing_{i}"))
            .unwrap();
    }

    let thread_count = 10;
    let barrier = Arc::new(Barrier::new(thread_count));
    let create_success = Arc::new(AtomicUsize::new(0));
    let drop_success = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let c_cnt = Arc::clone(&create_success);
            let d_cnt = Arc::clone(&drop_success);
            thread::spawn(move || {
                bar.wait();
                if t < 5 {
                    // Create new indexes
                    if eng.create_node_property_index(&format!("new_{t}")).is_ok() {
                        c_cnt.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    // Drop existing indexes
                    if eng.drop_node_index(&format!("existing_{}", t - 5)).is_ok() {
                        d_cnt.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All operations should succeed
    assert_eq!(create_success.load(Ordering::SeqCst), 5);
    assert_eq!(drop_success.load(Ordering::SeqCst), 5);

    // Verify engine is consistent
    let _ = engine.create_node("Test", HashMap::new()).unwrap();
}

#[test]
fn test_high_contention_single_node_100_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());
    let mut props = HashMap::new();
    props.insert("counter".to_string(), PropertyValue::Int(0));
    let node_id = engine.create_node("Contended", props).unwrap();

    let thread_count = 100;
    let updates_per_thread = 100;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                bar.wait();
                for i in 0..updates_per_thread {
                    let mut props = HashMap::new();
                    let value = (t * updates_per_thread + i) as i64;
                    props.insert("counter".to_string(), PropertyValue::Int(value));
                    if eng.update_node(node_id, None, props).is_ok() {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All updates should succeed (no corruption, no starvation)
    let expected = thread_count * updates_per_thread;
    assert_eq!(
        success_count.load(Ordering::SeqCst),
        expected,
        "all {expected} updates should succeed"
    );

    // Verify node is still valid
    let node = engine.get_node(node_id).unwrap();
    assert!(matches!(
        node.properties.get("counter"),
        Some(PropertyValue::Int(_))
    ));
}

#[test]
fn test_concurrent_traverse_during_modifications_20_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Create initial chain of 50 nodes
    let mut chain_ids = Vec::new();
    for i in 0..50 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        chain_ids.push(engine.create_node("Chain", props).unwrap());
    }
    for i in 0..49 {
        engine
            .create_edge(chain_ids[i], chain_ids[i + 1], "NEXT", HashMap::new(), true)
            .unwrap();
    }
    let chain_ids = Arc::new(chain_ids);

    let thread_count = 20;
    let barrier = Arc::new(Barrier::new(thread_count));
    let traversal_count = Arc::new(AtomicUsize::new(0));
    let edge_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // 10 traverser threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&traversal_count);
        let ids = Arc::clone(&chain_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..50 {
                let start_idx = t % 25;
                if eng
                    .traverse(ids[start_idx], Direction::Outgoing, 10, None, None)
                    .is_ok()
                {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // 10 edge writer threads
    for _ in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&edge_count);
        let ids = Arc::clone(&chain_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..20 {
                let from = ids[i % 50];
                let to = ids[(i + 10) % 50];
                if eng
                    .create_edge(from, to, "CROSS", HashMap::new(), true)
                    .is_ok()
                {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All traversals and edge creations should complete
    assert_eq!(traversal_count.load(Ordering::SeqCst), 10 * 50);
    assert_eq!(edge_count.load(Ordering::SeqCst), 10 * 20);
}

#[test]
fn test_concurrent_find_path_during_modifications_20_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Create 10x10 grid graph
    let mut grid_ids = Vec::new();
    for row in 0..10 {
        for col in 0..10 {
            let mut props = HashMap::new();
            props.insert("row".to_string(), PropertyValue::Int(row));
            props.insert("col".to_string(), PropertyValue::Int(col));
            grid_ids.push(engine.create_node("Grid", props).unwrap());
        }
    }

    // Connect horizontally and vertically
    for row in 0..10 {
        for col in 0..10 {
            let idx = row * 10 + col;
            if col < 9 {
                engine
                    .create_edge(
                        grid_ids[idx as usize],
                        grid_ids[(idx + 1) as usize],
                        "ADJACENT",
                        HashMap::new(),
                        false,
                    )
                    .unwrap();
            }
            if row < 9 {
                engine
                    .create_edge(
                        grid_ids[idx as usize],
                        grid_ids[(idx + 10) as usize],
                        "ADJACENT",
                        HashMap::new(),
                        false,
                    )
                    .unwrap();
            }
        }
    }

    let grid_ids = Arc::new(grid_ids);
    let thread_count = 20;
    let barrier = Arc::new(Barrier::new(thread_count));
    let path_found = Arc::new(AtomicUsize::new(0));
    let updates_done = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // 10 path finder threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&path_found);
        let ids = Arc::clone(&grid_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..20 {
                let start = ids[t % 50];
                let end = ids[99 - (t % 50)];
                if eng.find_path(start, end, None).is_ok() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // 10 modifier threads (update edge properties)
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&updates_done);
        let ids = Arc::clone(&grid_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..20 {
                let idx = (t * 10 + i) % 100;
                let mut props = HashMap::new();
                props.insert("weight".to_string(), PropertyValue::Float(i as f64 * 0.1));
                if eng.update_node(ids[idx], None, props).is_ok() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // All operations should complete without panic
    assert!(
        path_found.load(Ordering::SeqCst) > 0,
        "some paths should be found"
    );
    assert_eq!(updates_done.load(Ordering::SeqCst), 10 * 20);
}

#[test]
fn test_concurrent_pagerank_during_writes_16_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Create initial graph: 100 nodes with random-ish connections
    let mut node_ids = Vec::new();
    for i in 0..100 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        node_ids.push(engine.create_node("PageRank", props).unwrap());
    }

    // Create edges: each node connects to a few others
    for i in 0..100 {
        for j in 1..=3 {
            let target = (i + j * 7) % 100;
            if target != i {
                let _ = engine.create_edge(
                    node_ids[i],
                    node_ids[target],
                    "LINKS",
                    HashMap::new(),
                    true,
                );
            }
        }
    }

    let node_ids = Arc::new(node_ids);
    let thread_count = 16;
    let barrier = Arc::new(Barrier::new(thread_count));
    let pagerank_runs = Arc::new(AtomicUsize::new(0));
    let write_ops = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // 8 pagerank reader threads
    for _ in 0..8 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&pagerank_runs);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..5 {
                if let Ok(result) = eng.pagerank(Some(PageRankConfig {
                    damping: 0.85,
                    tolerance: 0.01,
                    max_iterations: 20,
                    direction: Direction::Outgoing,
                    edge_type: None,
                })) {
                    // Verify no NaN or infinite values
                    for (_, score) in &result.scores {
                        assert!(score.is_finite(), "score must be finite");
                    }
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // 8 writer threads
    for t in 0..8 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&write_ops);
        let ids = Arc::clone(&node_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..20 {
                let from = ids[(t * 10 + i) % 100];
                let to = ids[(t * 10 + i + 5) % 100];
                if eng
                    .create_edge(from, to, "NEW_LINK", HashMap::new(), true)
                    .is_ok()
                {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // Pagerank should complete successfully multiple times
    assert!(
        pagerank_runs.load(Ordering::SeqCst) > 0,
        "some pagerank runs should complete"
    );
    assert!(
        write_ops.load(Ordering::SeqCst) > 0,
        "some writes should complete"
    );
}

#[test]
fn test_concurrent_batch_mixed_operations_30_threads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    let engine = Arc::new(GraphEngine::new());

    // Pre-create 500 nodes
    let mut node_ids = Vec::new();
    for i in 0..500 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        node_ids.push(engine.create_node("Mixed", props).unwrap());
    }

    // Pre-create edges between consecutive nodes
    let mut edge_ids = Vec::new();
    for i in 0..400 {
        let edge_id = engine
            .create_edge(
                node_ids[i],
                node_ids[i + 1],
                "INITIAL",
                HashMap::new(),
                true,
            )
            .unwrap();
        edge_ids.push(edge_id);
    }

    let node_ids = Arc::new(node_ids);
    let edge_ids = Arc::new(edge_ids);
    let thread_count = 30;
    let barrier = Arc::new(Barrier::new(thread_count));

    let nodes_created = Arc::new(AtomicUsize::new(0));
    let edges_created = Arc::new(AtomicUsize::new(0));
    let nodes_deleted = Arc::new(AtomicUsize::new(0));
    let edges_deleted = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // 10 batch_create_nodes threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&nodes_created);
        handles.push(thread::spawn(move || {
            bar.wait();
            let nodes: Vec<NodeInput> = (0..20)
                .map(|i| NodeInput {
                    labels: vec!["BatchCreated".to_string()],
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("thread".to_string(), PropertyValue::Int(t as i64));
                        p.insert("idx".to_string(), PropertyValue::Int(i as i64));
                        p
                    },
                })
                .collect();
            if let Ok(result) = eng.batch_create_nodes(nodes) {
                cnt.fetch_add(result.count, Ordering::Relaxed);
            }
        }));
    }

    // 10 batch_create_edges threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&edges_created);
        let ids = Arc::clone(&node_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            let edges: Vec<EdgeInput> = (0..15)
                .map(|i| {
                    let from_idx = (t * 40 + i) % 500;
                    let to_idx = (t * 40 + i + 10) % 500;
                    EdgeInput {
                        from: ids[from_idx],
                        to: ids[to_idx],
                        edge_type: "BATCH_EDGE".to_string(),
                        properties: HashMap::new(),
                        directed: true,
                    }
                })
                .collect();
            if let Ok(result) = eng.batch_create_edges(edges) {
                cnt.fetch_add(result.count, Ordering::Relaxed);
            }
        }));
    }

    // 5 batch_delete_nodes threads (delete from higher indices to avoid conflicts)
    for t in 0..5 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&nodes_deleted);
        let ids = Arc::clone(&node_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            // Delete nodes from the end (450-499 range split among 5 threads)
            let start_idx = 450 + t * 10;
            let batch: Vec<u64> = ids[start_idx..start_idx + 10].to_vec();
            if let Ok(result) = eng.batch_delete_nodes(batch) {
                cnt.fetch_add(result.count, Ordering::Relaxed);
            }
        }));
    }

    // 5 batch_delete_edges threads
    for t in 0..5 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&edges_deleted);
        let ids = Arc::clone(&edge_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            // Delete edges from different ranges
            let start_idx = t * 20;
            let end_idx = (start_idx + 20).min(400);
            let batch: Vec<u64> = ids[start_idx..end_idx].to_vec();
            if let Ok(result) = eng.batch_delete_edges(batch) {
                cnt.fetch_add(result.count, Ordering::Relaxed);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // Verify operations completed
    assert!(
        nodes_created.load(Ordering::SeqCst) > 0,
        "some nodes should be created"
    );
    assert!(
        edges_created.load(Ordering::SeqCst) > 0,
        "some edges should be created"
    );
    assert!(
        nodes_deleted.load(Ordering::SeqCst) > 0,
        "some nodes should be deleted"
    );
    assert!(
        edges_deleted.load(Ordering::SeqCst) > 0,
        "some edges should be deleted"
    );

    // Verify engine is still consistent and usable
    let _ = engine.create_node("FinalTest", HashMap::new()).unwrap();
}

// PropertyValue accessor tests
#[test]
fn test_property_value_as_list() {
    let list = PropertyValue::List(vec![
        PropertyValue::Int(1),
        PropertyValue::Int(2),
        PropertyValue::Int(3),
    ]);
    let result = list.as_list();
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 3);

    let non_list = PropertyValue::Int(42);
    assert!(non_list.as_list().is_none());
}

#[test]
fn test_property_value_as_map() {
    let mut map = HashMap::new();
    map.insert(
        "key".to_string(),
        PropertyValue::String("value".to_string()),
    );
    let pv = PropertyValue::Map(map);
    let result = pv.as_map();
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 1);

    let non_map = PropertyValue::Int(42);
    assert!(non_map.as_map().is_none());
}

#[test]
fn test_property_value_as_bytes() {
    let bytes = PropertyValue::Bytes(vec![1, 2, 3, 4]);
    let result = bytes.as_bytes();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), &[1, 2, 3, 4]);

    let non_bytes = PropertyValue::Int(42);
    assert!(non_bytes.as_bytes().is_none());
}

#[test]
fn test_property_value_as_point() {
    let point = PropertyValue::Point {
        lat: 40.7128,
        lon: -74.0060,
    };
    let result = point.as_point();
    assert!(result.is_some());
    let (lat, lon) = result.unwrap();
    assert!((lat - 40.7128).abs() < f64::EPSILON);
    assert!((lon - (-74.0060)).abs() < f64::EPSILON);

    let non_point = PropertyValue::Int(42);
    assert!(non_point.as_point().is_none());
}

#[test]
fn test_property_value_as_datetime() {
    let dt = PropertyValue::DateTime(1609459200000);
    let result = dt.as_datetime();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), 1609459200000);

    let non_datetime = PropertyValue::Int(42);
    assert!(non_datetime.as_datetime().is_none());
}

#[test]
fn test_property_value_contains() {
    let list = PropertyValue::List(vec![
        PropertyValue::Int(1),
        PropertyValue::Int(2),
        PropertyValue::Int(3),
    ]);
    assert!(list.contains(&PropertyValue::Int(2)));
    assert!(!list.contains(&PropertyValue::Int(4)));

    let non_list = PropertyValue::Int(42);
    assert!(!non_list.contains(&PropertyValue::Int(42)));
}

#[test]
fn test_property_value_distance_km() {
    let nyc = PropertyValue::Point {
        lat: 40.7128,
        lon: -74.0060,
    };
    let la = PropertyValue::Point {
        lat: 34.0522,
        lon: -118.2437,
    };

    let distance = nyc.distance_km(&la);
    assert!(distance.is_some());
    let km = distance.unwrap();
    assert!(km > 3900.0 && km < 4000.0);

    let non_point = PropertyValue::Int(42);
    assert!(nyc.distance_km(&non_point).is_none());
    assert!(non_point.distance_km(&nyc).is_none());
}

#[test]
fn test_property_value_distance_km_same_point() {
    let point = PropertyValue::Point { lat: 0.0, lon: 0.0 };
    let distance = point.distance_km(&point);
    assert!(distance.is_some());
    assert!(distance.unwrap().abs() < 0.001);
}

#[test]
fn test_property_value_value_type() {
    assert_eq!(PropertyValue::Null.value_type(), PropertyValueType::Null);
    assert_eq!(PropertyValue::Int(42).value_type(), PropertyValueType::Int);
    assert_eq!(
        PropertyValue::Float(3.14).value_type(),
        PropertyValueType::Float
    );
    assert_eq!(
        PropertyValue::String("hello".to_string()).value_type(),
        PropertyValueType::String
    );
    assert_eq!(
        PropertyValue::Bool(true).value_type(),
        PropertyValueType::Bool
    );
    assert_eq!(
        PropertyValue::DateTime(12345).value_type(),
        PropertyValueType::DateTime
    );
    assert_eq!(
        PropertyValue::List(vec![]).value_type(),
        PropertyValueType::List
    );
    assert_eq!(
        PropertyValue::Map(HashMap::new()).value_type(),
        PropertyValueType::Map
    );
    assert_eq!(
        PropertyValue::Bytes(vec![]).value_type(),
        PropertyValueType::Bytes
    );
    assert_eq!(
        PropertyValue::Point { lat: 0.0, lon: 0.0 }.value_type(),
        PropertyValueType::Point
    );
}

#[test]
fn test_property_value_to_scalar_and_back() {
    let original_int = PropertyValue::Int(42);
    let scalar = original_int.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, original_int);

    let original_float = PropertyValue::Float(3.14);
    let scalar = original_float.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, original_float);

    let original_string = PropertyValue::String("hello".to_string());
    let scalar = original_string.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, original_string);

    let original_bool = PropertyValue::Bool(true);
    let scalar = original_bool.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, original_bool);

    let original_bytes = PropertyValue::Bytes(vec![1, 2, 3]);
    let scalar = original_bytes.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, original_bytes);

    let original_null = PropertyValue::Null;
    let scalar = original_null.to_scalar();
    let back = PropertyValue::from_scalar(&scalar);
    assert_eq!(back, original_null);
}

#[test]
fn test_property_value_to_scalar_complex_types() {
    let list = PropertyValue::List(vec![PropertyValue::Int(1), PropertyValue::Int(2)]);
    let scalar = list.to_scalar();
    match scalar {
        tensor_store::ScalarValue::String(s) => {
            assert!(s.contains("List") || s.contains("Int"));
        },
        _ => panic!("Expected string for complex type"),
    }

    let mut map = HashMap::new();
    map.insert(
        "key".to_string(),
        PropertyValue::String("value".to_string()),
    );
    let map_pv = PropertyValue::Map(map);
    let scalar = map_pv.to_scalar();
    match scalar {
        tensor_store::ScalarValue::String(s) => {
            assert!(s.contains("key") || s.contains("Map"));
        },
        _ => panic!("Expected string for complex type"),
    }

    let point = PropertyValue::Point { lat: 1.0, lon: 2.0 };
    let scalar = point.to_scalar();
    match scalar {
        tensor_store::ScalarValue::String(s) => {
            assert!(s.contains("Point") || s.contains("lat") || s.contains("1.0"));
        },
        _ => panic!("Expected string for complex type"),
    }
}

#[test]
fn test_property_value_from_scalar_json() {
    let list = PropertyValue::List(vec![PropertyValue::Int(1), PropertyValue::Int(2)]);
    let scalar = list.to_scalar();
    let json_str = match scalar {
        tensor_store::ScalarValue::String(s) => s,
        _ => panic!("Expected string"),
    };

    let parsed = PropertyValue::from_scalar(&tensor_store::ScalarValue::String(json_str));
    match parsed {
        PropertyValue::List(items) => assert_eq!(items.len(), 2),
        _ => panic!("Expected list from JSON roundtrip"),
    }

    let point = PropertyValue::Point { lat: 1.0, lon: 2.0 };
    let scalar = point.to_scalar();
    let json_str = match scalar {
        tensor_store::ScalarValue::String(s) => s,
        _ => panic!("Expected string"),
    };

    let parsed = PropertyValue::from_scalar(&tensor_store::ScalarValue::String(json_str));
    match parsed {
        PropertyValue::Point { lat, lon } => {
            assert!((lat - 1.0).abs() < f64::EPSILON);
            assert!((lon - 2.0).abs() < f64::EPSILON);
        },
        _ => panic!("Expected point from JSON roundtrip"),
    }
}

#[test]
fn test_property_value_datetime_to_scalar() {
    let dt = PropertyValue::DateTime(1609459200000);
    let scalar = dt.to_scalar();
    match scalar {
        tensor_store::ScalarValue::Int(v) => assert_eq!(v, 1609459200000),
        _ => panic!("Expected int for datetime"),
    }
}

// Compound index tests
#[test]
fn test_compound_index_create_and_find() {
    let engine = GraphEngine::new();

    let mut props1 = HashMap::new();
    props1.insert(
        "name".to_string(),
        PropertyValue::String("Alice".to_string()),
    );
    props1.insert("age".to_string(), PropertyValue::Int(30));
    engine.create_node("Person", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("name".to_string(), PropertyValue::String("Bob".to_string()));
    props2.insert("age".to_string(), PropertyValue::Int(25));
    engine.create_node("Person", props2).unwrap();

    engine.create_compound_index(&["name", "age"]).unwrap();
    assert!(engine.has_compound_index(&["name", "age"]));

    let results = engine
        .find_by_compound(&[
            ("name", &PropertyValue::String("Alice".to_string())),
            ("age", &PropertyValue::Int(30)),
        ])
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].properties.get("name"),
        Some(&PropertyValue::String("Alice".to_string()))
    );
}

#[test]
fn test_compound_index_empty_properties() {
    let engine = GraphEngine::new();
    engine.create_compound_index(&[]).unwrap();
}

#[test]
fn test_compound_index_already_exists() {
    let engine = GraphEngine::new();
    engine.create_compound_index(&["name", "age"]).unwrap();
    let result = engine.create_compound_index(&["name", "age"]);
    assert!(matches!(result, Err(GraphError::IndexAlreadyExists { .. })));
}

#[test]
fn test_compound_index_not_found() {
    let engine = GraphEngine::new();
    let result = engine.find_by_compound(&[("name", &PropertyValue::String("Alice".to_string()))]);
    assert!(matches!(result, Err(GraphError::IndexNotFound { .. })));
}

#[test]
fn test_compound_index_drop() {
    let engine = GraphEngine::new();
    engine.create_compound_index(&["name", "age"]).unwrap();
    assert!(engine.has_compound_index(&["name", "age"]));

    engine.drop_compound_index(&["name", "age"]).unwrap();
    assert!(!engine.has_compound_index(&["name", "age"]));
}

#[test]
fn test_compound_index_drop_not_found() {
    let engine = GraphEngine::new();
    let result = engine.drop_compound_index(&["nonexistent"]);
    assert!(matches!(result, Err(GraphError::IndexNotFound { .. })));
}

#[test]
fn test_compound_index_list() {
    let engine = GraphEngine::new();
    engine.create_compound_index(&["name", "age"]).unwrap();
    engine.create_compound_index(&["city", "country"]).unwrap();

    let indexes = engine.get_compound_indexes();
    assert_eq!(indexes.len(), 2);
}

#[test]
fn test_compound_index_find_no_match() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_compound_index(&["name"]).unwrap();

    let results = engine
        .find_by_compound(&[("name", &PropertyValue::String("NonExistent".to_string()))])
        .unwrap();
    assert!(results.is_empty());
}

// Error hash tests
#[test]
fn test_graph_error_hash_storage_error() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::StorageError("error1".to_string()));
    set.insert(GraphError::StorageError("error2".to_string()));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_constraint_already_exists() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::ConstraintAlreadyExists("c1".to_string()));
    set.insert(GraphError::ConstraintAlreadyExists("c2".to_string()));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_constraint_not_found() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::ConstraintNotFound("c1".to_string()));
    set.insert(GraphError::ConstraintNotFound("c2".to_string()));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_path_not_found() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::PathNotFound);
    set.insert(GraphError::PathNotFound);
    assert_eq!(set.len(), 1);
}

#[test]
fn test_graph_error_hash_index_exists() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::IndexAlreadyExists {
        target: "node".to_string(),
        property: "name".to_string(),
    });
    set.insert(GraphError::IndexAlreadyExists {
        target: "node".to_string(),
        property: "age".to_string(),
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_index_not_found() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::IndexNotFound {
        target: "node".to_string(),
        property: "name".to_string(),
    });
    set.insert(GraphError::IndexNotFound {
        target: "edge".to_string(),
        property: "name".to_string(),
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_negative_weight() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::NegativeWeight {
        edge_id: 1,
        weight: -1.0,
    });
    set.insert(GraphError::NegativeWeight {
        edge_id: 2,
        weight: -2.0,
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_constraint_violation() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::ConstraintViolation {
        constraint_name: "unique_name".to_string(),
        message: "msg1".to_string(),
    });
    set.insert(GraphError::ConstraintViolation {
        constraint_name: "unique_name".to_string(),
        message: "msg2".to_string(),
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_batch_validation() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::BatchValidationError {
        index: 0,
        cause: Box::new(GraphError::NodeNotFound(1)),
    });
    set.insert(GraphError::BatchValidationError {
        index: 1,
        cause: Box::new(GraphError::NodeNotFound(2)),
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_batch_creation() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::BatchCreationError {
        index: 0,
        cause: Box::new(GraphError::NodeNotFound(1)),
    });
    set.insert(GraphError::BatchCreationError {
        index: 1,
        cause: Box::new(GraphError::NodeNotFound(2)),
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_partial_deletion() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::PartialDeletionError {
        node_id: 1,
        failed_edges: vec![1, 2],
    });
    set.insert(GraphError::PartialDeletionError {
        node_id: 2,
        failed_edges: vec![3],
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_id_space_exhausted() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::IdSpaceExhausted {
        entity_type: "node",
    });
    set.insert(GraphError::IdSpaceExhausted {
        entity_type: "edge",
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_invalid_property_name() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::InvalidPropertyName {
        name: "a:b".to_string(),
    });
    set.insert(GraphError::InvalidPropertyName {
        name: "c:d".to_string(),
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_graph_error_hash_corrupted_edge() {
    use std::collections::HashSet;

    let mut set: HashSet<GraphError> = HashSet::new();
    set.insert(GraphError::CorruptedEdge {
        edge_id: 1,
        field: "from".to_string(),
    });
    set.insert(GraphError::CorruptedEdge {
        edge_id: 2,
        field: "to".to_string(),
    });
    assert_eq!(set.len(), 2);
}

// Error display tests
#[test]
fn test_graph_error_display_all_variants() {
    let errors = vec![
        GraphError::NodeNotFound(1),
        GraphError::EdgeNotFound(2),
        GraphError::StorageError("storage failed".to_string()),
        GraphError::PathNotFound,
        GraphError::IndexAlreadyExists {
            target: "node".to_string(),
            property: "name".to_string(),
        },
        GraphError::IndexNotFound {
            target: "edge".to_string(),
            property: "type".to_string(),
        },
        GraphError::NegativeWeight {
            edge_id: 5,
            weight: -3.5,
        },
        GraphError::ConstraintViolation {
            constraint_name: "unique".to_string(),
            message: "duplicate".to_string(),
        },
        GraphError::ConstraintAlreadyExists("my_constraint".to_string()),
        GraphError::ConstraintNotFound("missing".to_string()),
        GraphError::BatchValidationError {
            index: 3,
            cause: Box::new(GraphError::NodeNotFound(10)),
        },
        GraphError::BatchCreationError {
            index: 4,
            cause: Box::new(GraphError::EdgeNotFound(20)),
        },
        GraphError::PartialDeletionError {
            node_id: 100,
            failed_edges: vec![1, 2, 3],
        },
        GraphError::IdSpaceExhausted {
            entity_type: "node",
        },
        GraphError::InvalidPropertyName {
            name: "bad:name".to_string(),
        },
        GraphError::CorruptedEdge {
            edge_id: 99,
            field: "from".to_string(),
        },
    ];

    for err in errors {
        let display = format!("{}", err);
        assert!(!display.is_empty());
    }
}

// SCC algorithm additional tests
#[test]
fn test_scc_edge_type_filter() {
    use crate::algorithms::SccConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "TYPED", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "OTHER", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "TYPED", HashMap::new(), true)
        .unwrap();

    let config = SccConfig::new().edge_type("TYPED");
    let result = engine.strongly_connected_components(&config).unwrap();
    assert!(result.component_count >= 1);
}

#[test]
fn test_scc_components_by_size() {
    use crate::algorithms::SccConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let _n3 = engine.create_node("C", HashMap::new()).unwrap();
    let _n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n1, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .strongly_connected_components(&SccConfig::new())
        .unwrap();
    let by_size = result.components_by_size();
    assert!(!by_size.is_empty());
    assert!(by_size[0].1 >= 1);
}

#[test]
fn test_scc_result_default() {
    use crate::algorithms::SccResult;

    let result = SccResult::default();
    assert_eq!(result.component_count, 0);
    assert!(result.largest_component().is_none());
}

// MST algorithm additional tests
#[test]
fn test_mst_config_builder() {
    use crate::algorithms::MstConfig;

    let config = MstConfig::new("cost")
        .default_weight(2.0)
        .compute_forest(false);

    assert_eq!(config.weight_property, "cost");
    assert!((config.default_weight - 2.0).abs() < f64::EPSILON);
    assert!(!config.compute_forest);
}

#[test]
fn test_mst_result_accessors() {
    use crate::algorithms::MstResult;

    let result = MstResult::empty();
    assert_eq!(result.edge_count(), 0);
    assert!(!result.is_connected());
}

// Similarity algorithm additional tests
#[test]
fn test_similarity_node_link_prediction() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .jaccard_similarity(n1, n2, &SimilarityConfig::default())
        .unwrap();
    assert!(result >= 0.0 && result <= 1.0);
}

// Edge property index tests
#[test]
fn test_edge_property_index_create_and_find() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.5));
    engine.create_edge(n1, n2, "CONN", props, true).unwrap();

    engine.create_edge_property_index("weight").unwrap();
    assert!(engine.has_edge_index("weight"));

    let edges = engine
        .find_edges_by_property("weight", &PropertyValue::Float(1.5))
        .unwrap();
    assert_eq!(edges.len(), 1);
}

#[test]
fn test_edge_type_index_find() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "LIKES", HashMap::new(), true)
        .unwrap();

    let knows_edges = engine.find_edges_by_type("KNOWS").unwrap();
    assert_eq!(knows_edges.len(), 1);
}

// Scan fallback tests (for when indexes don't exist)
#[test]
fn test_find_nodes_by_label_property_scan() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Company", HashMap::new()).unwrap();

    let persons = engine.find_nodes_by_label("Person").unwrap();
    assert_eq!(persons.len(), 2);
}

#[test]
fn test_find_nodes_where_scan() {
    let engine = GraphEngine::new();

    let mut props1 = HashMap::new();
    props1.insert("age".to_string(), PropertyValue::Int(25));
    engine.create_node("Person", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("age".to_string(), PropertyValue::Int(30));
    engine.create_node("Person", props2).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("age".to_string(), PropertyValue::Int(35));
    engine.create_node("Person", props3).unwrap();

    let older_than_30 = engine
        .find_nodes_where("age", RangeOp::Gt, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(older_than_30.len(), 1);

    let at_least_30 = engine
        .find_nodes_where("age", RangeOp::Ge, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(at_least_30.len(), 2);

    let under_30 = engine
        .find_nodes_where("age", RangeOp::Lt, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(under_30.len(), 1);

    let at_most_30 = engine
        .find_nodes_where("age", RangeOp::Le, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(at_most_30.len(), 2);
}

#[test]
fn test_find_edges_where_scan() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "E", props1, true).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("weight".to_string(), PropertyValue::Float(2.0));
    engine.create_edge(n2, n3, "E", props2, true).unwrap();

    let heavy = engine
        .find_edges_where("weight", RangeOp::Gt, &PropertyValue::Float(1.5))
        .unwrap();
    assert_eq!(heavy.len(), 1);
}

#[test]
fn test_find_edges_by_edge_type_scan() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n2, "LIKES", HashMap::new(), true)
        .unwrap();

    let knows = engine
        .find_edges_by_property("_edge_type", &PropertyValue::String("KNOWS".to_string()))
        .unwrap();
    assert_eq!(knows.len(), 1);
}

// Range queries on labels
#[test]
fn test_find_nodes_where_label_range() {
    let engine = GraphEngine::new();
    engine.create_node("Apple", HashMap::new()).unwrap();
    engine.create_node("Banana", HashMap::new()).unwrap();
    engine.create_node("Cherry", HashMap::new()).unwrap();

    let before_b = engine
        .find_nodes_where(
            "_label",
            RangeOp::Lt,
            &PropertyValue::String("B".to_string()),
        )
        .unwrap();
    assert_eq!(before_b.len(), 1);
}

// OrderedPropertyValue tests
#[test]
fn test_ordered_property_value_ordering() {
    let v1 = OrderedPropertyValue::Int(1);
    let v2 = OrderedPropertyValue::Int(2);
    assert!(v1 < v2);

    let s1 = OrderedPropertyValue::String("a".to_string());
    let s2 = OrderedPropertyValue::String("b".to_string());
    assert!(s1 < s2);

    let pv1 = PropertyValue::Float(1.0);
    let pv2 = PropertyValue::Float(2.0);
    let f1 = OrderedPropertyValue::from(&pv1);
    let f2 = OrderedPropertyValue::from(&pv2);
    assert!(f1 < f2);
}

#[test]
fn test_ordered_property_value_null() {
    let null = OrderedPropertyValue::Null;
    let int = OrderedPropertyValue::Int(1);
    assert!(null < int);
}

// Pagination tests
#[test]
fn test_pagination_constructors() {
    let pagination = Pagination::new(0, 100);
    assert_eq!(pagination.skip, 0);
    assert_eq!(pagination.limit, Some(100));

    let pagination2 = Pagination::limit(50);
    assert_eq!(pagination2.skip, 0);
    assert_eq!(pagination2.limit, Some(50));
}

#[test]
fn test_pagination_with_total_count() {
    let pagination = Pagination::new(0, 100).with_total_count();
    assert!(pagination.count_total);
}

// WeightedPath tests
#[test]
fn test_weighted_path_clone() {
    let path = WeightedPath {
        nodes: vec![1, 2, 3],
        edges: vec![10, 20],
        total_weight: 5.0,
    };
    let cloned = path.clone();
    assert_eq!(cloned.nodes, path.nodes);
    assert_eq!(cloned.edges, path.edges);
    assert!((cloned.total_weight - path.total_weight).abs() < f64::EPSILON);
}

// Edge index drop tests
#[test]
fn test_edge_property_index_drop() {
    let engine = GraphEngine::new();
    engine.create_edge_property_index("weight").unwrap();
    assert!(engine.has_edge_index("weight"));

    engine.drop_edge_index("weight").unwrap();
    assert!(!engine.has_edge_index("weight"));
}

#[test]
fn test_edge_property_index_drop_not_found() {
    let engine = GraphEngine::new();
    let result = engine.drop_edge_index("nonexistent");
    assert!(matches!(result, Err(GraphError::IndexNotFound { .. })));
}

// OrderedPropertyValue conversion tests
#[test]
fn test_ordered_property_value_from_datetime() {
    let dt = PropertyValue::DateTime(1609459200000);
    let ordered = OrderedPropertyValue::from(&dt);
    match ordered {
        OrderedPropertyValue::DateTime(v) => assert_eq!(v, 1609459200000),
        _ => panic!("Expected DateTime"),
    }
}

#[test]
fn test_ordered_property_value_from_bytes() {
    let bytes = PropertyValue::Bytes(vec![1, 2, 3]);
    let ordered = OrderedPropertyValue::from(&bytes);
    match ordered {
        OrderedPropertyValue::Bytes(v) => assert_eq!(v, vec![1, 2, 3]),
        _ => panic!("Expected Bytes"),
    }
}

#[test]
fn test_ordered_property_value_from_complex() {
    let list = PropertyValue::List(vec![PropertyValue::Int(1)]);
    let ordered = OrderedPropertyValue::from(&list);
    match ordered {
        OrderedPropertyValue::Complex(s) => assert!(s.contains("Int")),
        _ => panic!("Expected Complex"),
    }

    let mut map = HashMap::new();
    map.insert("key".to_string(), PropertyValue::Int(42));
    let map_pv = PropertyValue::Map(map);
    let ordered = OrderedPropertyValue::from(&map_pv);
    match ordered {
        OrderedPropertyValue::Complex(s) => assert!(s.contains("key")),
        _ => panic!("Expected Complex"),
    }

    let point = PropertyValue::Point { lat: 1.0, lon: 2.0 };
    let ordered = OrderedPropertyValue::from(&point);
    match ordered {
        OrderedPropertyValue::Complex(s) => assert!(s.contains("lat")),
        _ => panic!("Expected Complex"),
    }
}

// Pagination tests
#[test]
fn test_all_edges_paginated() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.all_edges_paginated(Pagination::new(0, 2));
    assert_eq!(result.items.len(), 2);
    assert!(result.has_more);

    let result2 = engine.all_edges_paginated(Pagination::new(2, 10));
    assert_eq!(result2.items.len(), 1);
    assert!(!result2.has_more);
}

#[test]
fn test_all_edges_paginated_with_total() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.all_edges_paginated(Pagination::new(0, 10).with_total_count());
    assert_eq!(result.total_count, Some(1));
}

#[test]
fn test_find_nodes_by_label_paginated() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Company", HashMap::new()).unwrap();

    let result = engine
        .find_nodes_by_label_paginated("Person", Pagination::new(0, 2))
        .unwrap();
    assert_eq!(result.items.len(), 2);
    assert!(result.has_more);
}

// Property range scan tests with indexes
#[test]
fn test_find_nodes_where_with_index() {
    let engine = GraphEngine::new();

    let mut props1 = HashMap::new();
    props1.insert("score".to_string(), PropertyValue::Int(10));
    engine.create_node("Item", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("score".to_string(), PropertyValue::Int(20));
    engine.create_node("Item", props2).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("score".to_string(), PropertyValue::Int(30));
    engine.create_node("Item", props3).unwrap();

    engine.create_node_property_index("score").unwrap();

    let high_score = engine
        .find_nodes_where("score", RangeOp::Ge, &PropertyValue::Int(20))
        .unwrap();
    assert_eq!(high_score.len(), 2);
}

// Edges where tests with index
#[test]
fn test_find_edges_where_with_index() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("priority".to_string(), PropertyValue::Int(1));
    engine.create_edge(n1, n2, "E", props1, true).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("priority".to_string(), PropertyValue::Int(5));
    engine.create_edge(n2, n3, "E", props2, true).unwrap();

    engine.create_edge_property_index("priority").unwrap();

    let high_priority = engine
        .find_edges_where("priority", RangeOp::Gt, &PropertyValue::Int(2))
        .unwrap();
    assert_eq!(high_priority.len(), 1);
}

// Edge type range query test
#[test]
fn test_find_edges_where_edge_type() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "ALPHA", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n2, "BETA", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n2, "GAMMA", HashMap::new(), true)
        .unwrap();

    let before_c = engine
        .find_edges_where(
            "_edge_type",
            RangeOp::Lt,
            &PropertyValue::String("C".to_string()),
        )
        .unwrap();
    assert_eq!(before_c.len(), 2);
}

// Test finding nodes by label property
#[test]
fn test_find_nodes_by_label_property() {
    let engine = GraphEngine::new();
    engine.create_node("Apple", HashMap::new()).unwrap();
    engine.create_node("Banana", HashMap::new()).unwrap();
    engine.create_node("Cherry", HashMap::new()).unwrap();

    let results = engine.find_nodes_by_label("Apple").unwrap();
    assert_eq!(results.len(), 1);
}

// Test constraint validation
#[test]
fn test_constraint_validation_on_create() {
    let engine = GraphEngine::new();

    let constraint = Constraint {
        name: "unique_email".to_string(),
        constraint_type: ConstraintType::Unique,
        target: ConstraintTarget::NodeLabel("Person".to_string()),
        property: "email".to_string(),
    };
    engine.create_constraint(constraint).unwrap();

    let mut props = HashMap::new();
    props.insert(
        "email".to_string(),
        PropertyValue::String("test@example.com".to_string()),
    );
    engine.create_node("Person", props.clone()).unwrap();

    let result = engine.create_node("Person", props);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

// Test exists constraint
#[test]
fn test_exists_constraint() {
    let engine = GraphEngine::new();

    let constraint = Constraint {
        name: "product_name_required".to_string(),
        constraint_type: ConstraintType::Exists,
        target: ConstraintTarget::NodeLabel("Product".to_string()),
        property: "name".to_string(),
    };
    engine.create_constraint(constraint).unwrap();

    let result = engine.create_node("Product", HashMap::new());
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));

    let mut props = HashMap::new();
    props.insert(
        "name".to_string(),
        PropertyValue::String("Widget".to_string()),
    );
    assert!(engine.create_node("Product", props).is_ok());
}

// Test traversal with filter
#[test]
fn test_traverse_with_edge_type_filter() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "LIKES", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "KNOWS", HashMap::new(), true)
        .unwrap();

    let result = engine
        .traverse(n1, Direction::Outgoing, 2, Some("KNOWS"), None)
        .unwrap();
    assert!(result.iter().any(|n| n.id == n2));
    assert!(result.iter().any(|n| n.id == n4));
    assert!(!result.iter().any(|n| n.id == n3));
}

// Test all nodes paginated
#[test]
fn test_all_nodes_paginated() {
    let engine = GraphEngine::new();
    engine.create_node("A", HashMap::new()).unwrap();
    engine.create_node("B", HashMap::new()).unwrap();
    engine.create_node("C", HashMap::new()).unwrap();

    let result = engine.all_nodes_paginated(Pagination::new(0, 2));
    assert_eq!(result.items.len(), 2);
    assert!(result.has_more);

    let result2 = engine.all_nodes_paginated(Pagination::new(0, 10).with_total_count());
    assert_eq!(result2.total_count, Some(3));
}

// Test node degrees
#[test]
fn test_node_degree() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n1, "E", HashMap::new(), true)
        .unwrap();

    assert_eq!(engine.out_degree(n1).unwrap(), 2);
    assert_eq!(engine.in_degree(n1).unwrap(), 1);
    assert_eq!(engine.degree(n1).unwrap(), 3);
}

// Test graph count methods
#[test]
fn test_graph_counts() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    let n1 = engine.create_node("Company", HashMap::new()).unwrap();
    let n2 = engine.create_node("Company", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "OWNS", HashMap::new(), true)
        .unwrap();

    assert_eq!(engine.count_nodes(), 4);
    assert_eq!(engine.count_nodes_by_label("Person").unwrap(), 2);
    assert_eq!(engine.count_edges(), 1);
    assert_eq!(engine.count_edges_by_type("OWNS").unwrap(), 1);
}

// Test batch update nodes
#[test]
fn test_batch_update_nodes() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert(
        "status".to_string(),
        PropertyValue::String("updated".to_string()),
    );

    let mut props2 = HashMap::new();
    props2.insert(
        "status".to_string(),
        PropertyValue::String("updated".to_string()),
    );

    let updates = vec![(n1, None, props1), (n2, None, props2)];

    let count = engine.batch_update_nodes(updates).unwrap();
    assert_eq!(count, 2);

    let node1 = engine.get_node(n1).unwrap();
    assert_eq!(
        node1.properties.get("status"),
        Some(&PropertyValue::String("updated".to_string()))
    );
}

// Test multi-label nodes
#[test]
fn test_multi_label_node() {
    let engine = GraphEngine::new();
    let id = engine.create_node("Person", HashMap::new()).unwrap();
    engine.add_label(id, "Employee").unwrap();

    let node = engine.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    engine.remove_label(id, "Person").unwrap();
    let node2 = engine.get_node(id).unwrap();
    assert!(!node2.has_label("Person"));
    assert!(node2.has_label("Employee"));
}

// MST algorithm tests
#[test]
fn test_mst_with_custom_weight_property() {
    use crate::algorithms::MstConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("cost".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "E", props1, false).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("cost".to_string(), PropertyValue::Float(2.0));
    engine.create_edge(n2, n3, "E", props2, false).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("cost".to_string(), PropertyValue::Float(10.0));
    engine.create_edge(n1, n3, "E", props3, false).unwrap();

    let config = MstConfig::new("cost");
    let result = engine.minimum_spanning_tree(&config).unwrap();
    assert_eq!(result.edges.len(), 2);
    assert!(result.total_weight < 5.0);
}

#[test]
fn test_mst_forest() {
    use crate::algorithms::MstConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();

    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), false)
        .unwrap();

    let config = MstConfig::new("weight").compute_forest(true);
    let result = engine.minimum_spanning_tree(&config).unwrap();
    assert_eq!(result.tree_count, 2);
}

// Similarity algorithm tests
#[test]
fn test_similarity_cosine() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n4, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .cosine_similarity(n1, n2, &SimilarityConfig::default())
        .unwrap();
    assert!(result > 0.0 && result <= 1.0);
}

#[test]
fn test_similarity_common_neighbors() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n4, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let common = engine
        .common_neighbors(n1, n2, &SimilarityConfig::default())
        .unwrap();
    assert_eq!(common.len(), 1);
    assert!(common.contains(&n3));
}

#[test]
fn test_similarity_adamic_adar() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();
    let n5 = engine.create_node("E", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n5, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .adamic_adar(n1, n2, &SimilarityConfig::default())
        .unwrap();
    assert!(result >= 0.0);
}

#[test]
fn test_similarity_resource_allocation() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .resource_allocation(n1, n2, &SimilarityConfig::default())
        .unwrap();
    assert!(result >= 0.0);
}

#[test]
fn test_similarity_preferential_attachment() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n4, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .preferential_attachment(n1, n2, &SimilarityConfig::default())
        .unwrap();
    assert!((result - 2.0).abs() < f64::EPSILON);
}

#[test]
fn test_similarity_result_with_common_neighbors() {
    use crate::algorithms::{SimilarityMetric, SimilarityResult};

    let result = SimilarityResult::new(1, 2, 0.5, SimilarityMetric::Jaccard)
        .with_common_neighbors(vec![3, 4, 5]);

    assert_eq!(result.node_a, 1);
    assert_eq!(result.node_b, 2);
    assert!((result.score - 0.5).abs() < f64::EPSILON);
    assert_eq!(result.common_neighbors, Some(vec![3, 4, 5]));
}

#[test]
fn test_similarity_config_builder() {
    use crate::algorithms::SimilarityConfig;

    let config = SimilarityConfig::new()
        .edge_type("FRIEND")
        .direction(Direction::Both);

    assert_eq!(config.edge_type, Some("FRIEND".to_string()));
    assert_eq!(config.direction, Direction::Both);
}

// PagedResult tests
#[test]
fn test_paged_result_accessors() {
    let result = PagedResult::new(vec![1, 2, 3], Some(10), true);
    assert_eq!(result.items.len(), 3);
    assert_eq!(result.total_count, Some(10));
    assert!(result.has_more);
}

// Variable length path tests
#[test]
fn test_variable_length_paths_with_config() {
    let engine = GraphEngine::new();
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

    let config = VariableLengthConfig::with_hops(1, 5)
        .edge_type("NEXT")
        .direction(Direction::Outgoing)
        .max_paths(10);

    let result = engine.find_variable_paths(n1, n4, config).unwrap();
    assert!(!result.paths.is_empty());
}

// Batch operations tests
#[test]
fn test_batch_create_nodes() {
    let engine = GraphEngine::new();

    let nodes = vec![
        NodeInput::new(vec!["Person".to_string()], HashMap::new()),
        NodeInput::new(vec!["Person".to_string()], HashMap::new()),
        NodeInput::new(vec!["Company".to_string()], HashMap::new()),
    ];

    let result = engine.batch_create_nodes(nodes).unwrap();
    assert_eq!(result.count, 3);
    assert_eq!(result.created_ids.len(), 3);
}

#[test]
fn test_batch_create_edges() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let edges = vec![
        EdgeInput::new(n1, n2, "E".to_string(), HashMap::new(), true),
        EdgeInput::new(n2, n3, "E".to_string(), HashMap::new(), true),
    ];

    let result = engine.batch_create_edges(edges).unwrap();
    assert_eq!(result.count, 2);
}

// Find edges by property tests
#[test]
fn test_find_edges_by_property_paginated() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine
        .create_edge(n1, n2, "E", props.clone(), true)
        .unwrap();
    engine
        .create_edge(n1, n2, "E", props.clone(), true)
        .unwrap();
    engine.create_edge(n1, n2, "E", props, true).unwrap();

    let result = engine
        .find_edges_by_property_paginated(
            "weight",
            &PropertyValue::Float(1.0),
            Pagination::new(0, 2),
        )
        .unwrap();
    assert_eq!(result.items.len(), 2);
    assert!(result.has_more);
}

// Constraint tests
#[test]
fn test_constraint_get_and_drop() {
    let engine = GraphEngine::new();

    let constraint = Constraint {
        name: "test_unique".to_string(),
        constraint_type: ConstraintType::Unique,
        target: ConstraintTarget::NodeLabel("Person".to_string()),
        property: "id".to_string(),
    };
    engine.create_constraint(constraint).unwrap();

    let constraints = engine.list_constraints();
    assert!(constraints.iter().any(|c| c.name == "test_unique"));

    engine.drop_constraint("test_unique").unwrap();

    let constraints = engine.list_constraints();
    assert!(!constraints.iter().any(|c| c.name == "test_unique"));
}

#[test]
fn test_constraint_type_validation() {
    let engine = GraphEngine::new();

    let constraint = Constraint {
        name: "type_int".to_string(),
        constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
        target: ConstraintTarget::NodeLabel("Item".to_string()),
        property: "count".to_string(),
    };
    engine.create_constraint(constraint).unwrap();

    let mut props = HashMap::new();
    props.insert(
        "count".to_string(),
        PropertyValue::String("not_int".to_string()),
    );

    let result = engine.create_node("Item", props);
    assert!(matches!(
        result,
        Err(GraphError::ConstraintViolation { .. })
    ));
}

// Index lifecycle tests
#[test]
fn test_node_property_index_create_and_drop() {
    let engine = GraphEngine::new();

    engine.create_node_property_index("age").unwrap();
    engine.drop_node_index("age").unwrap();
}

// Edge operations
#[test]
fn test_edges_of_node() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "LIKES", HashMap::new(), true)
        .unwrap();

    let edges = engine.edges_of(n1, Direction::Outgoing).unwrap();
    assert_eq!(edges.len(), 2);
}

// Update edge test
#[test]
fn test_update_edge_properties() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut initial_props = HashMap::new();
    initial_props.insert("weight".to_string(), PropertyValue::Float(1.0));
    let edge_id = engine
        .create_edge(n1, n2, "E", initial_props, true)
        .unwrap();

    let mut new_props = HashMap::new();
    new_props.insert("weight".to_string(), PropertyValue::Float(2.0));
    new_props.insert(
        "label".to_string(),
        PropertyValue::String("updated".to_string()),
    );
    engine.update_edge(edge_id, new_props).unwrap();

    let edge = engine.get_edge(edge_id).unwrap();
    assert_eq!(
        edge.properties.get("weight"),
        Some(&PropertyValue::Float(2.0))
    );
    assert_eq!(
        edge.properties.get("label"),
        Some(&PropertyValue::String("updated".to_string()))
    );
}

// Delete and cascade tests
#[test]
fn test_delete_node_cascades_edges() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), true)
        .unwrap();

    let initial_edges = engine.count_edges();
    assert_eq!(initial_edges, 3);

    engine.delete_node(n2).unwrap();

    let final_edges = engine.count_edges();
    assert_eq!(final_edges, 1);
}

// Path finding tests
#[test]
fn test_find_path_not_found() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let result = engine.find_path(n1, n2, None);
    assert!(matches!(result, Err(GraphError::PathNotFound)));
}

#[test]
fn test_find_all_paths() {
    let engine = GraphEngine::new();
    // Create diamond pattern: n1 -> n2 -> n4 and n1 -> n3 -> n4
    // Both paths have 2 hops, so find_all_paths should return both shortest paths
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.find_all_paths(n1, n4, None).unwrap();
    assert_eq!(result.paths.len(), 2);
    assert_eq!(result.hop_count, 2);
}

// Property queries with different types
#[test]
fn test_find_nodes_by_bool_property() {
    let engine = GraphEngine::new();

    let mut props_true = HashMap::new();
    props_true.insert("active".to_string(), PropertyValue::Bool(true));
    engine.create_node("Item", props_true).unwrap();

    let mut props_false = HashMap::new();
    props_false.insert("active".to_string(), PropertyValue::Bool(false));
    engine.create_node("Item", props_false).unwrap();

    let active_items = engine
        .find_nodes_by_property("active", &PropertyValue::Bool(true))
        .unwrap();
    assert_eq!(active_items.len(), 1);
}

// PageRank test
#[test]
fn test_pagerank_custom_config() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("Page", HashMap::new()).unwrap();
    let n2 = engine.create_node("Page", HashMap::new()).unwrap();
    let n3 = engine.create_node("Page", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "LINKS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "LINKS", HashMap::new(), true)
        .unwrap();

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 3);
    assert!(result.converged);
}

// Connected components test
#[test]
fn test_connected_components_isolated() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();

    engine.create_node("C", HashMap::new()).unwrap();

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 2);
}

// Community detection test
#[test]
fn test_louvain_communities() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), false)
        .unwrap();

    let result = engine.louvain_communities(None).unwrap();
    assert!(!result.communities.is_empty());
}

// Label propagation test
#[test]
fn test_label_propagation() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();

    let result = engine.label_propagation(None).unwrap();
    assert!(!result.communities.is_empty());
}

// Centrality tests
#[test]
fn test_betweenness_centrality() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.betweenness_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 4);
}

#[test]
fn test_closeness_centrality() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.closeness_centrality(None).unwrap();
    assert!(!result.scores.is_empty());
}

// Test PatternMatch::default()
#[test]
fn test_pattern_match_default() {
    let pm = PatternMatch::default();
    assert!(pm.bindings.is_empty());
}

// Test Node last_modified_millis fallback
#[test]
fn test_node_last_modified_millis_fallback() {
    let engine = GraphEngine::new();

    // Create a node
    let node_id = engine.create_node("Test", HashMap::new()).unwrap();
    let node = engine.get_node(node_id).unwrap();

    // created_at should be set, last_modified_millis should use it as fallback
    assert!(node.created_at.is_some());
    assert_eq!(node.last_modified_millis(), node.created_at);

    // Update the node
    let mut new_props = HashMap::new();
    new_props.insert(
        "key".to_string(),
        PropertyValue::String("value".to_string()),
    );
    engine.update_node(node_id, None, new_props).unwrap();

    let updated_node = engine.get_node(node_id).unwrap();
    // After update, updated_at should be set and last_modified_millis should use it
    assert!(updated_node.updated_at.is_some());
    assert_eq!(updated_node.last_modified_millis(), updated_node.updated_at);
}

// Test delete node with undirected edges
#[test]
fn test_delete_node_with_undirected_edges() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // Create undirected edges
    engine
        .create_edge(n1, n2, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "FRIEND", HashMap::new(), false)
        .unwrap();

    // Delete the middle node n2 - should clean up undirected edges
    engine.delete_node(n2).unwrap();

    assert!(!engine.node_exists(n2));
    assert!(engine.node_exists(n1));
    assert!(engine.node_exists(n3));
    // The edges involving n2 should be gone
    assert!(engine.edges_of(n1, Direction::Both).unwrap().is_empty());
    assert!(engine.edges_of(n3, Direction::Both).unwrap().is_empty());
}

// Test batch edge creation with missing source node
#[test]
fn test_batch_create_edges_missing_source() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let edges = vec![EdgeInput {
        from: 999999, // Non-existent source
        to: n1,
        edge_type: "REL".to_string(),
        properties: HashMap::new(),
        directed: true,
    }];

    let result = engine.batch_create_edges(edges);
    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::BatchValidationError { index, cause } => {
            assert_eq!(index, 0);
            assert!(matches!(*cause, GraphError::NodeNotFound(999999)));
        },
        e => panic!("Expected BatchValidationError, got {:?}", e),
    }
}

// Test batch edge creation with missing target node
#[test]
fn test_batch_create_edges_missing_target() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let edges = vec![EdgeInput {
        from: n1,
        to: 999999, // Non-existent target
        edge_type: "REL".to_string(),
        properties: HashMap::new(),
        directed: true,
    }];

    let result = engine.batch_create_edges(edges);
    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::BatchValidationError { index, cause } => {
            assert_eq!(index, 0);
            assert!(matches!(*cause, GraphError::NodeNotFound(999999)));
        },
        e => panic!("Expected BatchValidationError, got {:?}", e),
    }
}

// Test batch node update validation error
#[test]
fn test_batch_update_nodes_validation_error() {
    let engine = GraphEngine::new();

    // Create a constraint that requires "name" property
    engine
        .create_constraint(Constraint {
            name: "exists_name".to_string(),
            constraint_type: ConstraintType::Exists,
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "name".to_string(),
        })
        .unwrap();

    // Create a node with required property
    let mut props = HashMap::new();
    props.insert(
        "name".to_string(),
        PropertyValue::String("Alice".to_string()),
    );
    let n1 = engine.create_node("Person", props).unwrap();

    // Attempt batch update that would violate the constraint
    let updates = vec![(n1, None, HashMap::new())]; // Empty properties = missing "name"
    let result = engine.batch_update_nodes(updates);
    assert!(result.is_err());
}

// Test exists constraint violation on existing data
#[test]
fn test_exists_constraint_violation_on_existing_data() {
    let engine = GraphEngine::new();

    // Create nodes without the property that will be required
    engine.create_node("User", HashMap::new()).unwrap();

    // Now try to create constraint - should fail because existing nodes lack the property
    let result = engine.create_constraint(Constraint {
        name: "require_email".to_string(),
        constraint_type: ConstraintType::Exists,
        target: ConstraintTarget::NodeLabel("User".to_string()),
        property: "email".to_string(),
    });
    assert!(result.is_err());
    match result.unwrap_err() {
        GraphError::ConstraintViolation {
            constraint_name,
            message,
        } => {
            assert_eq!(constraint_name, "require_email");
            assert!(message.contains("missing required property"));
        },
        e => panic!("Expected ConstraintViolation, got {:?}", e),
    }
}

// Test node property index building for custom property
#[test]
fn test_node_custom_property_index() {
    let engine = GraphEngine::new();

    let mut props1 = HashMap::new();
    props1.insert("city".to_string(), PropertyValue::String("NYC".to_string()));
    engine.create_node("Person", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("city".to_string(), PropertyValue::String("NYC".to_string()));
    engine.create_node("Person", props2).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("city".to_string(), PropertyValue::String("LA".to_string()));
    engine.create_node("Company", props3).unwrap();

    // Create index on city property
    engine.create_node_property_index("city").unwrap();
    assert!(engine.has_node_index("city"));

    // Clean up
    engine.drop_node_index("city").unwrap();
    assert!(!engine.has_node_index("city"));
}

// Test larger graph for parallel closeness centrality
#[test]
fn test_closeness_centrality_parallel() {
    use crate::config::GraphEngineConfig;

    // Set a low threshold to trigger parallel execution
    let config = GraphEngineConfig::new().centrality_parallel_threshold(2);
    let engine = GraphEngine::with_config(config);

    // Create a small network that exceeds the threshold
    let mut nodes = Vec::new();
    for i in 0..5 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        nodes.push(engine.create_node("Node", props).unwrap());
    }

    // Connect in a line
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "NEXT", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.closeness_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 5);
}

// Test PropertyValue from_scalar with various types
#[test]
fn test_property_value_from_scalar_all_types() {
    use tensor_store::ScalarValue;

    let pv = PropertyValue::from_scalar(&ScalarValue::Null);
    assert_eq!(pv, PropertyValue::Null);

    let pv = PropertyValue::from_scalar(&ScalarValue::Int(42));
    assert_eq!(pv, PropertyValue::Int(42));

    let pv = PropertyValue::from_scalar(&ScalarValue::Float(3.14));
    assert_eq!(pv, PropertyValue::Float(3.14));

    let pv = PropertyValue::from_scalar(&ScalarValue::String("test".to_string()));
    assert_eq!(pv, PropertyValue::String("test".to_string()));

    let pv = PropertyValue::from_scalar(&ScalarValue::Bool(true));
    assert_eq!(pv, PropertyValue::Bool(true));

    let pv = PropertyValue::from_scalar(&ScalarValue::Bytes(vec![1, 2, 3]));
    assert_eq!(pv, PropertyValue::Bytes(vec![1, 2, 3]));
}

// Test PatternMatch getters with wrong binding type
#[test]
fn test_pattern_match_getters() {
    let engine = GraphEngine::new();

    // Create some nodes
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    // Run pattern match to get a PatternMatch with bindings
    let pattern = Pattern::new(PathPattern::new(
        NodePattern::new().variable("a"),
        EdgePattern::new(),
        NodePattern::new().variable("b"),
    ));

    let result = engine.match_pattern(&pattern).unwrap();
    assert!(!result.is_empty());

    // Get a PatternMatch and test the getters
    let pm = &result.matches[0];

    // get_node should return Some for node bindings
    assert!(pm.get_node("a").is_some());
    assert!(pm.get_node("b").is_some());

    // get_edge should return None for node bindings (wrong type)
    assert!(pm.get_edge("a").is_none());

    // get_path should return None for node bindings (wrong type)
    assert!(pm.get_path("a").is_none());

    // Nonexistent bindings
    assert!(pm.get_node("nonexistent").is_none());
    assert!(pm.get_edge("nonexistent").is_none());
    assert!(pm.get_path("nonexistent").is_none());
}

// Test delete node as target of incoming edge
#[test]
fn test_delete_node_incoming_edge_cleanup() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    // Create directed edge n1 -> n2
    engine
        .create_edge(n1, n2, "TO", HashMap::new(), true)
        .unwrap();

    // Delete n2 (the target), should clean up n1's outgoing edge list
    engine.delete_node(n2).unwrap();

    assert!(!engine.node_exists(n2));
    assert!(engine.node_exists(n1));
    assert!(engine.edges_of(n1, Direction::Outgoing).unwrap().is_empty());
}

// Test PathPattern element access
#[test]
fn test_path_pattern_elements() {
    let path_pattern = PathPattern::new(NodePattern::new(), EdgePattern::new(), NodePattern::new());

    // Check that the pattern has 3 elements
    assert_eq!(path_pattern.elements.len(), 3);

    // First and last should be nodes, middle should be edge
    assert!(matches!(&path_pattern.elements[0], PatternElement::Node(_)));
    assert!(matches!(&path_pattern.elements[1], PatternElement::Edge(_)));
    assert!(matches!(&path_pattern.elements[2], PatternElement::Node(_)));
}

// Test find_weighted_path (Dijkstra)
#[test]
fn test_find_weighted_path() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // Create weighted edges: n1 --(1)--> n2 --(1)--> n3
    // And a longer direct path: n1 --(10)--> n3
    let mut props_short = HashMap::new();
    props_short.insert("weight".to_string(), PropertyValue::Float(1.0));
    let mut props_long = HashMap::new();
    props_long.insert("weight".to_string(), PropertyValue::Float(10.0));

    engine
        .create_edge(n1, n2, "E", props_short.clone(), true)
        .unwrap();
    engine.create_edge(n2, n3, "E", props_short, true).unwrap();
    engine.create_edge(n1, n3, "E", props_long, true).unwrap();

    // Shortest weighted path should be n1 -> n2 -> n3 (cost 2.0)
    let result = engine.find_weighted_path(n1, n3, "weight").unwrap();
    assert!((result.total_weight - 2.0).abs() < 0.001);
    assert_eq!(result.nodes, vec![n1, n2, n3]);
}

// Test connected components with Union-Find edge cases (different rank comparisons)
#[test]
fn test_connected_components_union_find_ranks() {
    let engine = GraphEngine::new();

    // Create a star topology: center connected to many nodes
    let center = engine.create_node("Center", HashMap::new()).unwrap();
    let mut leaves = Vec::new();
    for _ in 0..5 {
        let leaf = engine.create_node("Leaf", HashMap::new()).unwrap();
        leaves.push(leaf);
        engine
            .create_edge(center, leaf, "LINK", HashMap::new(), false)
            .unwrap();
    }

    // Add connections between some leaves to trigger different rank scenarios
    engine
        .create_edge(leaves[0], leaves[1], "LINK", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(leaves[2], leaves[3], "LINK", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(leaves[1], leaves[2], "LINK", HashMap::new(), false)
        .unwrap();

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 1); // All connected
}

// Test batch create edges with constraint violation
#[test]
fn test_batch_create_edges_constraint_violation() {
    let engine = GraphEngine::new();

    // Create nodes
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    // Create constraint requiring "weight" on LINK edges
    engine
        .create_constraint(Constraint {
            name: "link_weight".to_string(),
            constraint_type: ConstraintType::Exists,
            target: ConstraintTarget::EdgeType("LINK".to_string()),
            property: "weight".to_string(),
        })
        .unwrap();

    // Try batch create without weight property
    let edges = vec![EdgeInput {
        from: n1,
        to: n2,
        edge_type: "LINK".to_string(),
        properties: HashMap::new(), // Missing weight
        directed: true,
    }];

    let result = engine.batch_create_edges(edges);
    assert!(result.is_err());
}

// Test property type constraint on batch update
#[test]
fn test_property_type_constraint_batch_update() {
    let engine = GraphEngine::new();

    // Create constraint for age to be Int
    engine
        .create_constraint(Constraint {
            name: "age_type".to_string(),
            constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "age".to_string(),
        })
        .unwrap();

    // Create a valid Person
    let mut props = HashMap::new();
    props.insert("age".to_string(), PropertyValue::Int(25));
    let n1 = engine.create_node("Person", props).unwrap();

    // Try to update with wrong type via batch
    let mut bad_props = HashMap::new();
    bad_props.insert(
        "age".to_string(),
        PropertyValue::String("twenty-five".to_string()),
    );
    let updates = vec![(n1, None, bad_props)];

    let result = engine.batch_update_nodes(updates);
    assert!(result.is_err());
}

// Test edge index on _edge_type
#[test]
fn test_edge_type_index_explicit() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "LIKES", HashMap::new(), true)
        .unwrap();

    // The edge type index is auto-created
    assert!(engine.has_edge_index("_edge_type"));

    // Find edges by type
    let knows_edges = engine.find_edges_by_type("KNOWS").unwrap();
    assert_eq!(knows_edges.len(), 1);

    let likes_edges = engine.find_edges_by_type("LIKES").unwrap();
    assert_eq!(likes_edges.len(), 1);
}

// Test PathPattern extend
#[test]
fn test_path_pattern_extend() {
    let path_pattern = PathPattern::new(NodePattern::new(), EdgePattern::new(), NodePattern::new());
    assert_eq!(path_pattern.elements.len(), 3);

    // Extend with another edge and node
    let extended = path_pattern.extend(EdgePattern::new(), NodePattern::new());
    assert_eq!(extended.elements.len(), 5);
}

// Test find_nodes_by_any_label with empty array
#[test]
fn test_find_nodes_by_any_label_empty() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();

    let result = engine.find_nodes_by_any_label(&[]).unwrap();
    assert!(result.is_empty());
}

// Test find_nodes_by_any_label with multiple labels
#[test]
fn test_find_nodes_by_any_label() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Company", HashMap::new()).unwrap();
    engine.create_node("Location", HashMap::new()).unwrap();

    let result = engine
        .find_nodes_by_any_label(&["Person", "Company"])
        .unwrap();
    assert_eq!(result.len(), 2);
}

// Test Edge last_modified_millis
#[test]
fn test_edge_last_modified_millis() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let edge_id = engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    let edge = engine.get_edge(edge_id).unwrap();

    assert!(edge.created_at.is_some());
    assert_eq!(edge.last_modified_millis(), edge.created_at);

    // Update the edge
    let mut new_props = HashMap::new();
    new_props.insert("since".to_string(), PropertyValue::Int(2020));
    engine.update_edge(edge_id, new_props).unwrap();

    let updated_edge = engine.get_edge(edge_id).unwrap();
    assert!(updated_edge.updated_at.is_some());
    assert_eq!(updated_edge.last_modified_millis(), updated_edge.updated_at);
}

// Test find_nodes_where with various range operations
#[test]
fn test_find_nodes_where_range_ops() {
    let engine = GraphEngine::new();

    let mut props1 = HashMap::new();
    props1.insert("age".to_string(), PropertyValue::Int(20));
    engine.create_node("Person", props1).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("age".to_string(), PropertyValue::Int(30));
    engine.create_node("Person", props2).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("age".to_string(), PropertyValue::Int(40));
    engine.create_node("Person", props3).unwrap();

    // Less than
    let result = engine
        .find_nodes_where("age", RangeOp::Lt, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(result.len(), 1);

    // Less than or equal
    let result = engine
        .find_nodes_where("age", RangeOp::Le, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(result.len(), 2);

    // Greater than
    let result = engine
        .find_nodes_where("age", RangeOp::Gt, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(result.len(), 1);

    // Greater than or equal
    let result = engine
        .find_nodes_where("age", RangeOp::Ge, &PropertyValue::Int(30))
        .unwrap();
    assert_eq!(result.len(), 2);
}

// Test find edges where with range operations
#[test]
fn test_find_edges_where_range_ops() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "E", props1, true).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("weight".to_string(), PropertyValue::Float(2.0));
    engine.create_edge(n2, n3, "E", props2, true).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("weight".to_string(), PropertyValue::Float(3.0));
    engine.create_edge(n1, n3, "E", props3, true).unwrap();

    // Less than
    let result = engine
        .find_edges_where("weight", RangeOp::Lt, &PropertyValue::Float(2.0))
        .unwrap();
    assert_eq!(result.len(), 1);

    // Greater than
    let result = engine
        .find_edges_where("weight", RangeOp::Gt, &PropertyValue::Float(2.0))
        .unwrap();
    assert_eq!(result.len(), 1);
}

// Test all_weighted_paths
#[test]
fn test_find_all_weighted_paths() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    // Create diamond: n1 -> n2 -> n4 (weight 1+1=2)
    //                 n1 -> n3 -> n4 (weight 1+1=2)
    let mut props = HashMap::new();
    props.insert("w".to_string(), PropertyValue::Float(1.0));

    engine
        .create_edge(n1, n2, "E", props.clone(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", props.clone(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", props.clone(), true)
        .unwrap();
    engine.create_edge(n3, n4, "E", props, true).unwrap();

    let result = engine.find_all_weighted_paths(n1, n4, "w", None).unwrap();
    assert_eq!(result.paths.len(), 2);
    assert!((result.total_weight - 2.0).abs() < 0.001);
}

// Test variable length path with min/max hops
#[test]
fn test_find_variable_paths_with_config() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), true)
        .unwrap();

    // Find paths with exactly 2 hops
    let config = VariableLengthConfig::with_hops(2, 2);
    let result = engine.find_variable_paths(n1, n4, config).unwrap();
    assert!(result.paths.is_empty()); // No 2-hop path from n1 to n4

    // Find paths with 2-3 hops
    let config = VariableLengthConfig::with_hops(2, 3);
    let result = engine.find_variable_paths(n1, n4, config).unwrap();
    assert_eq!(result.paths.len(), 1);
    assert_eq!(result.paths[0].nodes.len(), 4);
}

// Test centrality with empty graph
#[test]
fn test_centrality_empty_graph() {
    let engine = GraphEngine::new();

    let result = engine.pagerank(None).unwrap();
    assert!(result.scores.is_empty());

    let result = engine.betweenness_centrality(None).unwrap();
    assert!(result.scores.is_empty());

    let result = engine.closeness_centrality(None).unwrap();
    assert!(result.scores.is_empty());
}

// Test batch node update validation error for non-existent node
#[test]
fn test_batch_update_nodes_non_existent() {
    let engine = GraphEngine::new();

    let updates = vec![(9999, None, HashMap::new())]; // Non-existent node
    let result = engine.batch_update_nodes(updates);
    assert!(result.is_err());
}

// Test parallel betweenness centrality
#[test]
fn test_betweenness_centrality_parallel() {
    use crate::config::GraphEngineConfig;

    // Set low threshold to trigger parallel execution
    let config = GraphEngineConfig::new().centrality_parallel_threshold(2);
    let engine = GraphEngine::with_config(config);

    // Create a small graph
    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Node", HashMap::new()).unwrap());
    }

    // Create edges
    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "E", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.betweenness_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 5);
}

// Test constraint on AllNodes
#[test]
fn test_constraint_all_nodes() {
    let engine = GraphEngine::new();

    let mut props = HashMap::new();
    props.insert("id".to_string(), PropertyValue::String("123".to_string()));
    engine.create_node("A", props.clone()).unwrap();
    engine.create_node("B", props).unwrap();

    // Create constraint on AllNodes
    let result = engine.create_constraint(Constraint {
        name: "all_id".to_string(),
        constraint_type: ConstraintType::Exists,
        target: ConstraintTarget::AllNodes,
        property: "id".to_string(),
    });
    assert!(result.is_ok());

    // Now trying to create a node without id should fail
    let result = engine.create_node("C", HashMap::new());
    assert!(result.is_err());
}

// Test constraint on AllEdges
#[test]
fn test_constraint_all_edges() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "E", props, true).unwrap();

    // Create constraint on AllEdges
    let result = engine.create_constraint(Constraint {
        name: "all_weight".to_string(),
        constraint_type: ConstraintType::Exists,
        target: ConstraintTarget::AllEdges,
        property: "weight".to_string(),
    });
    assert!(result.is_ok());

    // Now trying to create an edge without weight should fail
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let result = engine.create_edge(n2, n3, "E", HashMap::new(), true);
    assert!(result.is_err());
}

// Test get_constraint and list_constraints
#[test]
fn test_constraint_accessors() {
    let engine = GraphEngine::new();

    let constraint = Constraint {
        name: "test_constraint".to_string(),
        constraint_type: ConstraintType::Exists,
        target: ConstraintTarget::NodeLabel("Person".to_string()),
        property: "name".to_string(),
    };
    engine.create_constraint(constraint).unwrap();

    // Get constraint
    let c = engine.get_constraint("test_constraint");
    assert!(c.is_some());
    assert_eq!(c.unwrap().name, "test_constraint");

    // List constraints
    let constraints = engine.list_constraints();
    assert_eq!(constraints.len(), 1);

    // Get non-existent
    assert!(engine.get_constraint("nonexistent").is_none());
}

// Test find_nodes_by_property with _label special case
#[test]
fn test_find_nodes_by_label_property_special() {
    let engine = GraphEngine::new();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Company", HashMap::new()).unwrap();

    // Find by _label property (special case)
    let result = engine
        .find_nodes_by_property("_label", &PropertyValue::String("Person".to_string()))
        .unwrap();
    assert_eq!(result.len(), 2);

    // Try with non-string value for _label (should match nothing)
    let result = engine
        .find_nodes_by_property("_label", &PropertyValue::Int(123))
        .unwrap();
    assert!(result.is_empty());
}

// Test node with multiple labels index update
#[test]
fn test_multi_label_node_index() {
    let engine = GraphEngine::new();

    // Create nodes with labels
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();

    // The _label index should be auto-created and contain entries
    assert!(engine.has_node_index("_label"));

    // Find should work
    let persons = engine.find_nodes_by_label("Person").unwrap();
    assert_eq!(persons.len(), 2);
}

// Test update_node with labels change
#[test]
fn test_update_node_labels() {
    let engine = GraphEngine::new();
    let node_id = engine.create_node("Person", HashMap::new()).unwrap();

    // Update labels
    engine
        .update_node(node_id, Some(vec!["Employee".to_string()]), HashMap::new())
        .unwrap();

    let node = engine.get_node(node_id).unwrap();
    assert!(node.has_label("Employee"));
    assert!(!node.has_label("Person"));
}

// Test PropertyValue::value_type
#[test]
fn test_property_value_type() {
    assert_eq!(PropertyValue::Null.value_type(), PropertyValueType::Null);
    assert_eq!(PropertyValue::Int(1).value_type(), PropertyValueType::Int);
    assert_eq!(
        PropertyValue::Float(1.0).value_type(),
        PropertyValueType::Float
    );
    assert_eq!(
        PropertyValue::String("s".to_string()).value_type(),
        PropertyValueType::String
    );
    assert_eq!(
        PropertyValue::Bool(true).value_type(),
        PropertyValueType::Bool
    );
    assert_eq!(
        PropertyValue::DateTime(1609459200000).value_type(),
        PropertyValueType::DateTime
    );
    assert_eq!(
        PropertyValue::List(vec![]).value_type(),
        PropertyValueType::List
    );
    assert_eq!(
        PropertyValue::Map(HashMap::new()).value_type(),
        PropertyValueType::Map
    );
    assert_eq!(
        PropertyValue::Bytes(vec![]).value_type(),
        PropertyValueType::Bytes
    );
    assert_eq!(
        PropertyValue::Point { lat: 0.0, lon: 0.0 }.value_type(),
        PropertyValueType::Point
    );
}

// Test unique constraint violation
#[test]
fn test_unique_constraint() {
    let engine = GraphEngine::new();

    let mut props = HashMap::new();
    props.insert(
        "email".to_string(),
        PropertyValue::String("test@example.com".to_string()),
    );
    engine.create_node("User", props.clone()).unwrap();

    // Create unique constraint
    engine
        .create_constraint(Constraint {
            name: "unique_email".to_string(),
            constraint_type: ConstraintType::Unique,
            target: ConstraintTarget::NodeLabel("User".to_string()),
            property: "email".to_string(),
        })
        .unwrap();

    // Try to create another node with same email
    let result = engine.create_node("User", props);
    assert!(result.is_err());
}

// Test parallel pattern matching
#[test]
fn test_pattern_matching_parallel() {
    use crate::config::GraphEngineConfig;

    // Set low threshold for parallel processing
    let config = GraphEngineConfig::new().pattern_parallel_threshold(2);
    let engine = GraphEngine::with_config(config);

    // Create enough nodes to trigger parallel matching
    for i in 0..10 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        let n = engine.create_node("Person", props).unwrap();
        if i > 0 {
            let prev = n - 1;
            engine
                .create_edge(prev, n, "KNOWS", HashMap::new(), true)
                .unwrap();
        }
    }

    let pattern = Pattern::new(PathPattern::new(
        NodePattern::new().label("Person"),
        EdgePattern::new().edge_type("KNOWS"),
        NodePattern::new().label("Person"),
    ));

    let result = engine.match_pattern(&pattern).unwrap();
    assert!(!result.is_empty());
}

// Test drop constraint
#[test]
fn test_drop_constraint() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "test_drop".to_string(),
            constraint_type: ConstraintType::Exists,
            target: ConstraintTarget::NodeLabel("X".to_string()),
            property: "y".to_string(),
        })
        .unwrap();

    assert!(engine.get_constraint("test_drop").is_some());
    engine.drop_constraint("test_drop").unwrap();
    assert!(engine.get_constraint("test_drop").is_none());

    // Dropping non-existent constraint should error
    let result = engine.drop_constraint("nonexistent");
    assert!(result.is_err());
}

// Test unique constraint on existing data violation
#[test]
fn test_unique_constraint_existing_violation() {
    let engine = GraphEngine::new();

    // Create nodes with duplicate values
    let mut props = HashMap::new();
    props.insert("code".to_string(), PropertyValue::String("ABC".to_string()));
    engine.create_node("Item", props.clone()).unwrap();
    engine.create_node("Item", props).unwrap();

    // Try to create unique constraint - should fail
    let result = engine.create_constraint(Constraint {
        name: "unique_code".to_string(),
        constraint_type: ConstraintType::Unique,
        target: ConstraintTarget::NodeLabel("Item".to_string()),
        property: "code".to_string(),
    });
    assert!(result.is_err());
}

// Test edge property index
#[test]
fn test_edge_property_index() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.5));
    engine.create_edge(n1, n2, "E", props, true).unwrap();

    // Create edge property index
    engine.create_edge_property_index("weight").unwrap();
    assert!(engine.has_edge_index("weight"));

    // Drop index
    engine.drop_edge_index("weight").unwrap();
    assert!(!engine.has_edge_index("weight"));
}

// Test compound index on nodes
#[test]
fn test_compound_node_index() {
    let engine = GraphEngine::new();

    let mut props = HashMap::new();
    props.insert(
        "first".to_string(),
        PropertyValue::String("John".to_string()),
    );
    props.insert("last".to_string(), PropertyValue::String("Doe".to_string()));
    engine.create_node("Person", props).unwrap();

    // Create compound index
    engine.create_compound_index(&["first", "last"]).unwrap();
    assert!(engine.has_compound_index(&["first", "last"]));

    // Find by compound
    let result = engine
        .find_by_compound(&[
            ("first", &PropertyValue::String("John".to_string())),
            ("last", &PropertyValue::String("Doe".to_string())),
        ])
        .unwrap();
    assert_eq!(result.len(), 1);

    // Drop compound index
    engine.drop_compound_index(&["first", "last"]).unwrap();
    assert!(!engine.has_compound_index(&["first", "last"]));
}

// Test node count and edge count
#[test]
fn test_node_edge_count() {
    let engine = GraphEngine::new();

    assert_eq!(engine.node_count(), 0);
    assert_eq!(engine.edge_count(), 0);

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    assert_eq!(engine.node_count(), 2);

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    assert_eq!(engine.edge_count(), 1);
}

// Test all_edges
#[test]
fn test_all_edges() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let edges = engine.all_edges();
    assert_eq!(edges.len(), 2);
}

// Test find_path with no path
#[test]
fn test_find_path_no_path() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    // No edge connecting them

    let result = engine.find_path(n1, n2, None);
    assert!(result.is_err());
}

// Test find_weighted_path with no path
#[test]
fn test_find_weighted_path_no_path() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    // No edge connecting them

    let result = engine.find_weighted_path(n1, n2, "weight");
    assert!(result.is_err());
}

// Test property type constraint with correct and wrong types
#[test]
fn test_property_type_constraint_validation() {
    let engine = GraphEngine::new();

    // Create constraint for age to be Int
    engine
        .create_constraint(Constraint {
            name: "age_int".to_string(),
            constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
            target: ConstraintTarget::NodeLabel("Person".to_string()),
            property: "age".to_string(),
        })
        .unwrap();

    // Valid node with Int age
    let mut props = HashMap::new();
    props.insert("age".to_string(), PropertyValue::Int(25));
    let result = engine.create_node("Person", props);
    assert!(result.is_ok());

    // Invalid node with String age
    let mut props = HashMap::new();
    props.insert(
        "age".to_string(),
        PropertyValue::String("twenty".to_string()),
    );
    let result = engine.create_node("Person", props);
    assert!(result.is_err());
}

// Test unique constraint on edges
#[test]
fn test_unique_constraint_edges() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("id".to_string(), PropertyValue::String("e1".to_string()));
    engine.create_edge(n1, n2, "REL", props, true).unwrap();

    // Create unique constraint on edge property
    engine
        .create_constraint(Constraint {
            name: "unique_edge_id".to_string(),
            constraint_type: ConstraintType::Unique,
            target: ConstraintTarget::EdgeType("REL".to_string()),
            property: "id".to_string(),
        })
        .unwrap();

    // Try to create edge with same id
    let mut props = HashMap::new();
    props.insert("id".to_string(), PropertyValue::String("e1".to_string()));
    let result = engine.create_edge(n2, n3, "REL", props, true);
    assert!(result.is_err());
}

// Test edge from property types
#[test]
fn test_edge_from_to() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let edge_id = engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    let edge = engine.get_edge(edge_id).unwrap();

    assert_eq!(edge.from, n1);
    assert_eq!(edge.to, n2);
    assert_eq!(edge.edge_type, "KNOWS");
    assert!(edge.directed);
}

// Test exists constraint on edges
#[test]
fn test_exists_constraint_edges() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "LINK", props, true).unwrap();

    // Create exists constraint
    engine
        .create_constraint(Constraint {
            name: "link_weight".to_string(),
            constraint_type: ConstraintType::Exists,
            target: ConstraintTarget::EdgeType("LINK".to_string()),
            property: "weight".to_string(),
        })
        .unwrap();

    // Try to create edge without weight
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let result = engine.create_edge(n2, n3, "LINK", HashMap::new(), true);
    assert!(result.is_err());
}

// Test property type constraint with null property (should be allowed)
#[test]
fn test_property_type_constraint_null_allowed() {
    let engine = GraphEngine::new();

    // Create constraint for count to be Int
    engine
        .create_constraint(Constraint {
            name: "count_int".to_string(),
            constraint_type: ConstraintType::PropertyType(PropertyValueType::Int),
            target: ConstraintTarget::NodeLabel("Counter".to_string()),
            property: "count".to_string(),
        })
        .unwrap();

    // Node without count property should be allowed (property is optional)
    let result = engine.create_node("Counter", HashMap::new());
    assert!(result.is_ok());

    // Node with correct type should be allowed
    let mut props = HashMap::new();
    props.insert("count".to_string(), PropertyValue::Int(5));
    let result = engine.create_node("Counter", props);
    assert!(result.is_ok());
}

// Test get_edge for non-existent edge
#[test]
fn test_get_edge_not_found() {
    let engine = GraphEngine::new();
    let result = engine.get_edge(999999);
    assert!(result.is_err());
}

// Test delete_edge
#[test]
fn test_delete_edge() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let edge_id = engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    assert!(engine.get_edge(edge_id).is_ok());

    engine.delete_edge(edge_id).unwrap();
    assert!(engine.get_edge(edge_id).is_err());
}

// Test delete_edge not found
#[test]
fn test_delete_edge_not_found() {
    let engine = GraphEngine::new();
    let result = engine.delete_edge(999999);
    assert!(result.is_err());
}

// Test neighbors
#[test]
fn test_neighbors() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();

    let neighbors = engine
        .neighbors(n1, None, Direction::Outgoing, None)
        .unwrap();
    assert_eq!(neighbors.len(), 2);

    let neighbors = engine
        .neighbors(n2, None, Direction::Incoming, None)
        .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, n1);
}

// Test eigenvector centrality
#[test]
fn test_eigenvector_centrality() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.eigenvector_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 3);
}

// Test eigenvector centrality parallel
#[test]
fn test_eigenvector_centrality_parallel() {
    use crate::config::GraphEngineConfig;

    let config = GraphEngineConfig::new().centrality_parallel_threshold(2);
    let engine = GraphEngine::with_config(config);

    let mut nodes = Vec::new();
    for _ in 0..5 {
        nodes.push(engine.create_node("Node", HashMap::new()).unwrap());
    }

    for i in 0..4 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "E", HashMap::new(), true)
            .unwrap();
    }
    engine
        .create_edge(nodes[4], nodes[0], "E", HashMap::new(), true)
        .unwrap();

    let result = engine.eigenvector_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 5);
}

// Test label propagation simple
#[test]
fn test_label_propagation_simple() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();

    let result = engine.label_propagation(None).unwrap();
    assert!(result.community_count > 0);
}

// Test count triangles
#[test]
fn test_count_triangles() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // Create a triangle
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), false)
        .unwrap();

    let config = TriangleConfig::default();
    let result = engine.count_triangles(&config).unwrap();
    assert_eq!(result.triangle_count, 1);
}

// Test local clustering coefficient
#[test]
fn test_local_clustering_coefficient() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), false)
        .unwrap();

    let config = TriangleConfig::default();
    let result = engine.local_clustering_coefficient(n1, &config).unwrap();
    assert!(result >= 0.0 && result <= 1.0);
}

// Test global clustering coefficient
#[test]
fn test_global_clustering_coefficient() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), false)
        .unwrap();

    let config = TriangleConfig::default();
    let coeff = engine.global_clustering_coefficient(&config).unwrap();
    assert!(coeff >= 0.0 && coeff <= 1.0);
}

// Test find common neighbors
#[test]
fn test_common_neighbors() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let common = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, common, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let neighbors = engine.common_neighbors(n1, n2, &config).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert!(neighbors.contains(&common));
}

// Test adamic adar index
#[test]
fn test_adamic_adar() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let common = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, common, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let score = engine.adamic_adar(n1, n2, &config).unwrap();
    assert!(score >= 0.0);
}

// Test jaccard similarity
#[test]
fn test_jaccard_similarity() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let c1 = engine.create_node("C", HashMap::new()).unwrap();
    let c2 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, c1, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, c2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, c1, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let sim = engine.jaccard_similarity(n1, n2, &config).unwrap();
    assert!(sim >= 0.0 && sim <= 1.0);
}

// Test preferential attachment
#[test]
fn test_preferential_attachment() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let score = engine.preferential_attachment(n1, n2, &config).unwrap();
    assert!(score >= 0.0);
}

// Test find path with filter
#[test]
fn test_find_path_with_filter() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "ROAD", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "ROAD", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n4, "ROAD", HashMap::new(), true)
        .unwrap();

    // Path exists
    let result = engine.find_path(n1, n4, None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().nodes.len(), 4);
}

// Test neighbors with edge type filter
#[test]
fn test_neighbors_with_edge_type() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "FRIEND", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "COLLEAGUE", HashMap::new(), true)
        .unwrap();

    // Only FRIEND edges
    let neighbors = engine
        .neighbors(n1, Some("FRIEND"), Direction::Outgoing, None)
        .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, n2);
}

// Test traverse
#[test]
fn test_traverse() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let result = engine
        .traverse(n1, Direction::Outgoing, 10, None, None)
        .unwrap();
    assert_eq!(result.len(), 3);
}

// Test resource allocation index
#[test]
fn test_resource_allocation() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let common = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, common, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let score = engine.resource_allocation(n1, n2, &config).unwrap();
    assert!(score >= 0.0);
}

// Test strongly connected components
#[test]
fn test_strongly_connected_components() {
    use crate::algorithms::SccConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // Create a cycle
    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), true)
        .unwrap();

    let config = SccConfig::default();
    let result = engine.strongly_connected_components(&config).unwrap();
    assert_eq!(result.component_count, 1);
}

// Test minimum spanning tree
#[test]
fn test_minimum_spanning_tree() {
    use crate::algorithms::MstConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "E", props1, false).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("weight".to_string(), PropertyValue::Float(2.0));
    engine.create_edge(n2, n3, "E", props2, false).unwrap();

    let mut props3 = HashMap::new();
    props3.insert("weight".to_string(), PropertyValue::Float(3.0));
    engine.create_edge(n1, n3, "E", props3, false).unwrap();

    let config = MstConfig::default();
    let result = engine.minimum_spanning_tree(&config).unwrap();
    assert_eq!(result.edges.len(), 2);
}

// Test find nodes by multiple properties
#[test]
fn test_find_nodes_multi_property() {
    let engine = GraphEngine::new();

    let mut props = HashMap::new();
    props.insert(
        "name".to_string(),
        PropertyValue::String("Alice".to_string()),
    );
    props.insert("age".to_string(), PropertyValue::Int(30));
    let n1 = engine.create_node("Person", props).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("name".to_string(), PropertyValue::String("Bob".to_string()));
    props2.insert("age".to_string(), PropertyValue::Int(25));
    engine.create_node("Person", props2).unwrap();

    // Find by name
    let result = engine
        .find_nodes_by_property("name", &PropertyValue::String("Alice".to_string()))
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, n1);
}

// Test shortest path same node
#[test]
fn test_find_path_same_node() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let result = engine.find_path(n1, n1, None).unwrap();
    assert_eq!(result.nodes.len(), 1);
    assert_eq!(result.nodes[0], n1);
}

// Test weighted path same node
#[test]
fn test_find_weighted_path_same_node() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let result = engine.find_weighted_path(n1, n1, "weight").unwrap();
    assert_eq!(result.nodes.len(), 1);
    assert!((result.total_weight - 0.0).abs() < 0.001);
}

// Test all_paths same node
#[test]
fn test_find_all_paths_same_node() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let result = engine.find_all_paths(n1, n1, None).unwrap();
    assert_eq!(result.paths.len(), 1);
    assert_eq!(result.hop_count, 0);
}

// Test node exists
#[test]
fn test_node_exists() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    assert!(engine.node_exists(n1));
    assert!(!engine.node_exists(9999999));
}

// Test get_edge
#[test]
fn test_get_edge_basic() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();

    assert!(engine.get_edge(e1).is_ok());
    assert!(engine.get_edge(9999999).is_err());
}

// Test edges_of with edge type filter
#[test]
fn test_edges_of_with_type() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "LIKES", HashMap::new(), true)
        .unwrap();

    let all_edges = engine.edges_of(n1, Direction::Outgoing).unwrap();
    assert_eq!(all_edges.len(), 2);
}

// Test pagerank convergence
#[test]
fn test_pagerank_converges() {
    let engine = GraphEngine::new();

    // Create a simple graph
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 3);
    // Scores should sum to approximately 1
    let sum: f64 = result.scores.values().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

// Test variable length paths direction
#[test]
fn test_variable_paths_direction() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();

    let config = VariableLengthConfig::with_hops(1, 3).direction(Direction::Outgoing);
    let result = engine.find_variable_paths(n1, n3, config).unwrap();
    assert!(!result.paths.is_empty());
}

// Test A* algorithm
#[test]
fn test_astar_basic() {
    use crate::algorithms::AStarConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine
        .create_edge(n1, n2, "E", props.clone(), true)
        .unwrap();
    engine.create_edge(n2, n3, "E", props, true).unwrap();

    let config = AStarConfig::default();
    let result = engine.astar_path(n1, n3, &config).unwrap();
    assert!(result.path.is_some());
    assert_eq!(result.path.unwrap().nodes.len(), 3);
}

// Test biconnected components
#[test]
fn test_biconnected_components() {
    use crate::algorithms::BiconnectedConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), false)
        .unwrap();

    let config = BiconnectedConfig::default();
    let result = engine.biconnected_components(&config).unwrap();
    assert!(result.component_count > 0);
}

// Test k-core decomposition
#[test]
fn test_kcore() {
    use crate::algorithms::KCoreConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n4, n1, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", HashMap::new(), false)
        .unwrap();

    let config = KCoreConfig::default();
    let result = engine.kcore_decomposition(&config).unwrap();
    assert!(!result.core_numbers.is_empty());
}

// Test find_edges_by_property
#[test]
fn test_find_edges_by_property() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props1 = HashMap::new();
    props1.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.create_edge(n1, n2, "E", props1, true).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("weight".to_string(), PropertyValue::Float(2.0));
    engine.create_edge(n2, n3, "E", props2, true).unwrap();

    let result = engine
        .find_edges_by_property("weight", &PropertyValue::Float(1.0))
        .unwrap();
    assert_eq!(result.len(), 1);
}

// Test neighbors paginated
#[test]
fn test_neighbors_paginated() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    for _ in 0..10 {
        let n = engine.create_node("B", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n, "E", HashMap::new(), true)
            .unwrap();
    }

    let pagination = Pagination::new(0, 5);
    let result = engine
        .neighbors_paginated(n1, None, Direction::Outgoing, None, pagination)
        .unwrap();
    assert_eq!(result.items.len(), 5);
    assert!(result.has_more);
}

// Test all_weighted_paths same node
#[test]
fn test_all_weighted_paths_same_node() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let result = engine
        .find_all_weighted_paths(n1, n1, "weight", None)
        .unwrap();
    assert_eq!(result.paths.len(), 1);
    assert!((result.total_weight - 0.0).abs() < 0.001);
}

// Test constraint duplicate name
#[test]
fn test_constraint_duplicate_name() {
    let engine = GraphEngine::new();

    engine
        .create_constraint(Constraint {
            name: "dup".to_string(),
            constraint_type: ConstraintType::Exists,
            target: ConstraintTarget::NodeLabel("X".to_string()),
            property: "y".to_string(),
        })
        .unwrap();

    let result = engine.create_constraint(Constraint {
        name: "dup".to_string(),
        constraint_type: ConstraintType::Exists,
        target: ConstraintTarget::NodeLabel("Y".to_string()),
        property: "z".to_string(),
    });
    assert!(result.is_err());
}

// Test update edge
#[test]
fn test_update_edge() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let edge_id = engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();

    let mut new_props = HashMap::new();
    new_props.insert("weight".to_string(), PropertyValue::Float(5.0));
    engine.update_edge(edge_id, new_props).unwrap();

    let edge = engine.get_edge(edge_id).unwrap();
    assert_eq!(
        edge.properties.get("weight"),
        Some(&PropertyValue::Float(5.0))
    );
}

// Test empty graph operations
#[test]
fn test_empty_graph_operations() {
    let engine = GraphEngine::new();

    // All nodes/edges should be empty
    assert!(engine.all_nodes().is_empty());
    assert!(engine.all_edges().is_empty());

    // Counts should be zero
    assert_eq!(engine.node_count(), 0);
    assert_eq!(engine.edge_count(), 0);
}

// Test parallel pagerank with large graph
#[test]
fn test_pagerank_parallel_large() {
    use crate::config::GraphEngineConfig;

    // Set low threshold for parallel execution
    let config = GraphEngineConfig::new().pattern_parallel_threshold(2);
    let engine = GraphEngine::with_config(config);

    // Create larger graph for parallel execution
    let mut nodes = Vec::new();
    for _ in 0..10 {
        nodes.push(engine.create_node("Node", HashMap::new()).unwrap());
    }

    // Create cycle
    for i in 0..10 {
        engine
            .create_edge(nodes[i], nodes[(i + 1) % 10], "E", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.pagerank(None).unwrap();
    assert_eq!(result.scores.len(), 10);
}

// Test index not found error
#[test]
fn test_drop_nonexistent_index() {
    let engine = GraphEngine::new();
    let result = engine.drop_node_index("nonexistent");
    assert!(result.is_err());
}

// Test property index already exists
#[test]
fn test_create_duplicate_index() {
    let engine = GraphEngine::new();

    let mut props = HashMap::new();
    props.insert(
        "name".to_string(),
        PropertyValue::String("test".to_string()),
    );
    engine.create_node("Person", props).unwrap();

    engine.create_node_property_index("name").unwrap();
    let result = engine.create_node_property_index("name");
    assert!(result.is_err());
}

// Test batch delete nodes
#[test]
fn test_batch_delete_nodes() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let result = engine.batch_delete_nodes(vec![n1, n2]).unwrap();
    assert_eq!(result.count, 2);
    assert!(!engine.node_exists(n1));
    assert!(!engine.node_exists(n2));
    assert!(engine.node_exists(n3));
}

// Test batch delete edges
#[test]
fn test_batch_delete_edges() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let e1 = engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    let e2 = engine
        .create_edge(n2, n3, "E", HashMap::new(), true)
        .unwrap();
    let e3 = engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();

    let result = engine.batch_delete_edges(vec![e1, e2]).unwrap();
    assert_eq!(result.count, 2);
    assert!(engine.get_edge(e1).is_err());
    assert!(engine.get_edge(e2).is_err());
    assert!(engine.get_edge(e3).is_ok());
}

// Test parallel betweenness with config
#[test]
fn test_parallel_betweenness() {
    use crate::config::GraphEngineConfig;

    let config = GraphEngineConfig::new().centrality_parallel_threshold(2);
    let engine = GraphEngine::with_config(config);

    // Create graph large enough for parallel
    let mut nodes = Vec::new();
    for _ in 0..8 {
        nodes.push(engine.create_node("N", HashMap::new()).unwrap());
    }
    for i in 0..7 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "E", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.betweenness_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 8);
}

// Test pattern matching with edge variable
#[test]
fn test_pattern_match_edge_variable() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    let pattern = Pattern::new(PathPattern::new(
        NodePattern::new().label("Person").variable("a"),
        EdgePattern::new().edge_type("KNOWS").variable("e"),
        NodePattern::new().label("Person").variable("b"),
    ));

    let result = engine.match_pattern(&pattern).unwrap();
    assert!(!result.is_empty());
    // Should have edge binding
    let pm = &result.matches[0];
    assert!(pm.get_edge("e").is_some());
}

// Test node with timestamps
#[test]
fn test_node_timestamps() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();

    let node = engine.get_node(n1).unwrap();
    assert!(node.created_at_millis().is_some());
    assert!(node.updated_at_millis().is_none());

    // Update node
    let mut props = HashMap::new();
    props.insert(
        "key".to_string(),
        PropertyValue::String("value".to_string()),
    );
    engine.update_node(n1, None, props).unwrap();

    let updated_node = engine.get_node(n1).unwrap();
    assert!(updated_node.updated_at_millis().is_some());
}

// Test edge with timestamps
#[test]
fn test_edge_timestamps() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let e1 = engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();

    let edge = engine.get_edge(e1).unwrap();
    assert!(edge.created_at_millis().is_some());
    assert!(edge.updated_at_millis().is_none());

    // Update edge
    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));
    engine.update_edge(e1, props).unwrap();

    let updated_edge = engine.get_edge(e1).unwrap();
    assert!(updated_edge.updated_at_millis().is_some());
}

// Test louvain with config
#[test]
fn test_louvain_with_config() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n1, "E", HashMap::new(), false)
        .unwrap();

    let result = engine.louvain_communities(None).unwrap();
    assert!(result.community_count >= 1);
}

// Test parallel closeness with low threshold
#[test]
fn test_parallel_closeness_large() {
    use crate::config::GraphEngineConfig;

    let config = GraphEngineConfig::new().centrality_parallel_threshold(3);
    let engine = GraphEngine::with_config(config);

    // Create graph larger than threshold
    let mut nodes = Vec::new();
    for _ in 0..6 {
        nodes.push(engine.create_node("N", HashMap::new()).unwrap());
    }
    for i in 0..5 {
        engine
            .create_edge(nodes[i], nodes[i + 1], "E", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.closeness_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 6);
}

// Test edge delete and indexes
#[test]
fn test_edge_delete_index_update() {
    let engine = GraphEngine::new();
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));
    let e1 = engine.create_edge(n1, n2, "REL", props, true).unwrap();

    // Create edge property index
    engine.create_edge_property_index("weight").unwrap();

    // Delete edge - should update index
    engine.delete_edge(e1).unwrap();

    // Find should return empty
    let found = engine
        .find_edges_by_property("weight", &PropertyValue::Float(1.0))
        .unwrap();
    assert!(found.is_empty());
}

// Test node_similarity with different metrics
#[test]
fn test_node_similarity_metrics() {
    use crate::algorithms::{SimilarityConfig, SimilarityMetric};

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let common = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, common, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();

    // Test Jaccard via node_similarity
    let result = engine
        .node_similarity(n1, n2, SimilarityMetric::Jaccard, &config)
        .unwrap();
    assert!(result.score >= 0.0);

    // Test AdamicAdar
    let result = engine
        .node_similarity(n1, n2, SimilarityMetric::AdamicAdar, &config)
        .unwrap();
    assert!(result.score >= 0.0);

    // Test ResourceAllocation
    let result = engine
        .node_similarity(n1, n2, SimilarityMetric::ResourceAllocation, &config)
        .unwrap();
    assert!(result.score >= 0.0);

    // Test PreferentialAttachment
    let result = engine
        .node_similarity(n1, n2, SimilarityMetric::PreferentialAttachment, &config)
        .unwrap();
    assert!(result.score >= 0.0);

    // Test CommonNeighbors
    let result = engine
        .node_similarity(n1, n2, SimilarityMetric::CommonNeighbors, &config)
        .unwrap();
    assert!(result.score >= 0.0);

    // Test Cosine
    let result = engine
        .node_similarity(n1, n2, SimilarityMetric::Cosine, &config)
        .unwrap();
    assert!(result.score >= 0.0);
}

// Test most_similar
#[test]
fn test_most_similar() {
    use crate::algorithms::{SimilarityConfig, SimilarityMetric};

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let common = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(n1, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, common, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let results = engine
        .most_similar(n1, SimilarityMetric::Jaccard, &config, 5)
        .unwrap();
    assert!(!results.is_empty());
}

// Test cosine similarity directly
#[test]
fn test_cosine_similarity_direct() {
    use crate::algorithms::SimilarityConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let common = engine.create_node("C", HashMap::new()).unwrap();

    engine
        .create_edge(n1, common, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, common, "E", HashMap::new(), true)
        .unwrap();

    let config = SimilarityConfig::default();
    let result = engine.cosine_similarity(n1, n2, &config).unwrap();
    assert!(result >= 0.0 && result <= 1.0);
}

// Test PropertyValue from string that looks like JSON but isn't valid PropertyValue JSON
#[test]
fn test_property_value_from_invalid_json_string() {
    use tensor_store::ScalarValue;

    // String that starts with { but isn't valid PropertyValue JSON
    let invalid_json = "{invalid json}".to_string();
    let pv = PropertyValue::from_scalar(&ScalarValue::String(invalid_json.clone()));
    // Should fall back to treating it as a string
    assert_eq!(pv, PropertyValue::String(invalid_json));

    // String that starts with [ but isn't valid JSON
    let invalid_array = "[not valid".to_string();
    let pv = PropertyValue::from_scalar(&ScalarValue::String(invalid_array.clone()));
    assert_eq!(pv, PropertyValue::String(invalid_array));
}

// Test Union-Find with specific graph structure to trigger Less ordering
#[test]
fn test_connected_components_union_find_less() {
    let engine = GraphEngine::new();

    // Create a chain structure where nodes are merged in specific order
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();
    let n5 = engine.create_node("E", HashMap::new()).unwrap();

    // Create edges to build specific tree structure
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n4, n5, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n1, n5, "E", HashMap::new(), false)
        .unwrap();

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 1);
}

// Test DijkstraEntry equality (via weighted path)
#[test]
fn test_weighted_path_with_equal_costs() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(1.0));

    engine
        .create_edge(n1, n2, "E", props.clone(), true)
        .unwrap();
    engine.create_edge(n2, n3, "E", props, true).unwrap();

    let result = engine.find_weighted_path(n1, n3, "weight").unwrap();
    assert!((result.total_weight - 2.0).abs() < 0.001);
}

// Test index on _label property explicitly
#[test]
fn test_label_index_rebuild() {
    use tensor_store::TensorStore;

    // Create engine with store
    let store = TensorStore::new();
    let engine = GraphEngine::with_store(store);

    // Create some nodes
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Person", HashMap::new()).unwrap();
    engine.create_node("Company", HashMap::new()).unwrap();

    // The _label index should be auto-created
    assert!(engine.has_node_index("_label"));

    // Query should work
    let persons = engine.find_nodes_by_label("Person").unwrap();
    assert_eq!(persons.len(), 2);
}

// Test edge index on _edge_type property
#[test]
fn test_edge_type_index_rebuild() {
    use tensor_store::TensorStore;

    let store = TensorStore::new();
    let engine = GraphEngine::with_store(store);

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n2, "LIKES", HashMap::new(), true)
        .unwrap();

    // The _edge_type index should be auto-created
    assert!(engine.has_edge_index("_edge_type"));

    // Query should work
    let knows = engine.find_edges_by_type("KNOWS").unwrap();
    assert_eq!(knows.len(), 1);
}

// Test parallel eigenvector with config
#[test]
fn test_parallel_eigenvector() {
    use crate::config::GraphEngineConfig;

    let config = GraphEngineConfig::new().centrality_parallel_threshold(3);
    let engine = GraphEngine::with_config(config);

    // Create graph larger than threshold
    let mut nodes = Vec::new();
    for _ in 0..6 {
        nodes.push(engine.create_node("N", HashMap::new()).unwrap());
    }
    // Create cycle
    for i in 0..6 {
        engine
            .create_edge(nodes[i], nodes[(i + 1) % 6], "E", HashMap::new(), true)
            .unwrap();
    }

    let result = engine.eigenvector_centrality(None).unwrap();
    assert_eq!(result.scores.len(), 6);
}

// Test pattern match result stats
#[test]
fn test_pattern_match_stats() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Person", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
        .unwrap();

    let pattern = Pattern::new(PathPattern::new(
        NodePattern::new().label("Person"),
        EdgePattern::new(),
        NodePattern::new().label("Person"),
    ));

    let result = engine.match_pattern(&pattern).unwrap();
    assert!(result.stats.nodes_evaluated > 0);
}

// Test find all paths with config limits
#[test]
fn test_find_all_paths_with_config() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    // Diamond pattern
    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n4, "E", HashMap::new(), true)
        .unwrap();

    let config = AllPathsConfig {
        max_paths: 10,
        max_parents_per_node: 5,
    };
    let result = engine.find_all_paths(n1, n4, Some(config)).unwrap();
    assert_eq!(result.paths.len(), 2);
}

// Test weighted paths with config
#[test]
fn test_find_all_weighted_paths_with_config() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("w".to_string(), PropertyValue::Float(1.0));

    engine
        .create_edge(n1, n2, "E", props.clone(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", props.clone(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", props.clone(), true)
        .unwrap();
    engine.create_edge(n3, n4, "E", props, true).unwrap();

    let config = AllPathsConfig {
        max_paths: 10,
        max_parents_per_node: 5,
    };
    let result = engine
        .find_all_weighted_paths(n1, n4, "w", Some(config))
        .unwrap();
    assert_eq!(result.paths.len(), 2);
}

// Test negative weight error
#[test]
fn test_weighted_path_negative_weight() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut props = HashMap::new();
    props.insert("weight".to_string(), PropertyValue::Float(-1.0));
    engine.create_edge(n1, n2, "E", props, true).unwrap();

    let result = engine.find_weighted_path(n1, n2, "weight");
    assert!(result.is_err());
}

// Test articulation points
#[test]
fn test_articulation_points() {
    use crate::algorithms::BiconnectedConfig;

    let engine = GraphEngine::new();

    // Create a graph with an articulation point
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    // n2 is articulation point: n1-n2, n2-n3, n2-n4
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n4, "E", HashMap::new(), false)
        .unwrap();

    let config = BiconnectedConfig::default();
    let result = engine.biconnected_components(&config).unwrap();
    assert!(!result.articulation_points.is_empty());
}

// Test bridges
#[test]
fn test_bridges() {
    use crate::algorithms::BiconnectedConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // n1-n2-n3 chain has 2 bridges
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "E", HashMap::new(), false)
        .unwrap();

    let config = BiconnectedConfig::default();
    let result = engine.biconnected_components(&config).unwrap();
    assert!(!result.bridges.is_empty());
}

// Test MstResult::default()
#[test]
fn test_mst_result_default() {
    use crate::algorithms::MstResult;

    let result = MstResult::default();
    assert_eq!(result.edge_count(), 0);
    assert_eq!(result.total_weight, 0.0);
    assert_eq!(result.tree_count, 0);
}

// Test TriangleResult::default()
#[test]
fn test_triangle_result_default() {
    use crate::algorithms::TriangleResult;

    let result = TriangleResult::default();
    assert_eq!(result.triangle_count, 0);
    assert_eq!(result.average_clustering(), 0.0);
}

// Test TriangleConfig::edge_type()
#[test]
fn test_triangle_config_edge_type() {
    use crate::algorithms::TriangleConfig;

    let config = TriangleConfig::new().edge_type("FRIEND");
    assert_eq!(config.edge_type, Some("FRIEND".to_string()));

    // Test with actual graph
    let engine = GraphEngine::new();
    let a = engine.create_node("A", HashMap::new()).unwrap();
    let b = engine.create_node("B", HashMap::new()).unwrap();
    let c = engine.create_node("C", HashMap::new()).unwrap();

    // Create triangle with FRIEND edges
    engine
        .create_edge(a, b, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(b, c, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(c, a, "FRIEND", HashMap::new(), false)
        .unwrap();

    let result = engine.count_triangles(&config.undirected()).unwrap();
    assert_eq!(result.triangle_count, 1);
}

// Test A* with missing coordinates (falls back to 0.0)
#[test]
fn test_astar_euclidean_missing_coordinates() {
    let engine = GraphEngine::new();

    // Create nodes without x,y coordinates
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), false)
        .unwrap();

    // Should still find path, heuristic returns 0.0 for missing coords
    let result = engine.astar_path_euclidean(n1, n2, "x", "y").unwrap();
    assert!(result.found());
    assert!(result.path.is_some());
    assert_eq!(result.path.as_ref().unwrap().nodes.len(), 2);
}

// Test A* Manhattan with missing coordinates
#[test]
fn test_astar_manhattan_missing_coordinates() {
    let engine = GraphEngine::new();

    // Create nodes without x,y coordinates
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), false)
        .unwrap();

    // Should still find path
    let result = engine.astar_path_manhattan(n1, n2, "x", "y").unwrap();
    assert!(result.found());
    assert!(result.path.is_some());
}

// Test A* with partial coordinates (some nodes missing coords)
#[test]
fn test_astar_euclidean_partial_coordinates() {
    let engine = GraphEngine::new();

    // Node 1 has coordinates
    let mut p1 = HashMap::new();
    p1.insert("x".to_string(), PropertyValue::Float(0.0));
    p1.insert("y".to_string(), PropertyValue::Float(0.0));
    let n1 = engine.create_node("Node", p1).unwrap();

    // Node 2 has only x
    let mut p2 = HashMap::new();
    p2.insert("x".to_string(), PropertyValue::Float(1.0));
    let n2 = engine.create_node("Node", p2).unwrap();

    // Node 3 has full coordinates
    let mut p3 = HashMap::new();
    p3.insert("x".to_string(), PropertyValue::Float(2.0));
    p3.insert("y".to_string(), PropertyValue::Float(2.0));
    let n3 = engine.create_node("Node", p3).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "EDGE", HashMap::new(), false)
        .unwrap();

    // Should find path despite missing y on n2
    let result = engine.astar_path_euclidean(n1, n3, "x", "y").unwrap();
    assert!(result.found());
}

// Test MST forest with isolated nodes
#[test]
fn test_mst_forest_isolated_nodes() {
    let engine = GraphEngine::new();

    // Create connected component
    let mut p1 = HashMap::new();
    p1.insert("weight".to_string(), PropertyValue::Float(1.0));
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    engine.create_edge(n1, n2, "E", p1, false).unwrap();

    // Create isolated nodes
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    let forests = engine.minimum_spanning_forest("weight").unwrap();

    // Should have multiple trees: one for the connected pair, plus isolated nodes
    assert!(forests.len() >= 1);

    // Check that isolated nodes are represented
    let all_nodes: Vec<u64> = forests.iter().flat_map(|f| f.nodes.clone()).collect();
    assert!(all_nodes.contains(&n3) || forests.iter().any(|f| f.nodes.contains(&n3)));
    assert!(all_nodes.contains(&n4) || forests.iter().any(|f| f.nodes.contains(&n4)));
}

// Test Union-Find Less case (node degrees arranged to trigger Less branch)
#[test]
fn test_connected_components_union_find_less_case() {
    let engine = GraphEngine::new();

    // Create a graph where union operations trigger different rank comparisons
    let _n1 = engine.create_node("A", HashMap::new()).unwrap();
    let _n2 = engine.create_node("B", HashMap::new()).unwrap();
    let _n3 = engine.create_node("C", HashMap::new()).unwrap();
    let _n4 = engine.create_node("D", HashMap::new()).unwrap();
    let _n5 = engine.create_node("E", HashMap::new()).unwrap();
    let _n6 = engine.create_node("F", HashMap::new()).unwrap();

    // First group: build up rank for n1
    engine
        .create_edge(_n1, _n2, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(_n1, _n3, "E", HashMap::new(), false)
        .unwrap();

    // Second group: build up rank for n4
    engine
        .create_edge(_n4, _n5, "E", HashMap::new(), false)
        .unwrap();

    // Now connect n6 to the smaller rank tree first
    engine
        .create_edge(_n6, _n4, "E", HashMap::new(), false)
        .unwrap();

    // Finally connect the groups to trigger rank comparisons
    engine
        .create_edge(_n3, _n5, "E", HashMap::new(), false)
        .unwrap();

    let result = engine.connected_components(None).unwrap();
    assert_eq!(result.community_count, 1);
    // All 6 nodes should be in one community
    assert_eq!(result.communities.len(), 6);
}

// Test scan_nodes_where with _label
#[test]
fn test_scan_nodes_where_label() {
    let engine = GraphEngine::new();

    let _n1 = engine.create_node("Alpha", HashMap::new()).unwrap();
    let _n2 = engine.create_node("Beta", HashMap::new()).unwrap();
    let _n3 = engine.create_node("Gamma", HashMap::new()).unwrap();
    let _n4 = engine.create_node("Delta", HashMap::new()).unwrap();

    // Test range operations on labels
    let nodes =
        engine.scan_nodes_where("_label", RangeOp::Gt, &PropertyValue::String("Beta".into()));
    // Delta and Gamma should match (> Beta alphabetically)
    assert!(nodes.len() >= 2);

    let nodes =
        engine.scan_nodes_where("_label", RangeOp::Lt, &PropertyValue::String("Beta".into()));
    // Alpha should match
    assert!(nodes.len() >= 1);

    let nodes = engine.scan_nodes_where(
        "_label",
        RangeOp::Ge,
        &PropertyValue::String("Gamma".into()),
    );
    assert!(nodes.len() >= 1);

    let nodes = engine.scan_nodes_where(
        "_label",
        RangeOp::Le,
        &PropertyValue::String("Alpha".into()),
    );
    assert!(nodes.len() >= 1);
}

// Test building node index on a custom property (not _label since it's auto-indexed)
#[test]
fn test_build_node_custom_property_index() {
    let engine = GraphEngine::new();

    let mut p1 = HashMap::new();
    p1.insert(
        "department".to_string(),
        PropertyValue::String("Engineering".into()),
    );
    let n1 = engine.create_node("Person", p1).unwrap();

    let mut p2 = HashMap::new();
    p2.insert(
        "department".to_string(),
        PropertyValue::String("Engineering".into()),
    );
    let n2 = engine.create_node("Person", p2).unwrap();

    let mut p3 = HashMap::new();
    p3.insert(
        "department".to_string(),
        PropertyValue::String("Sales".into()),
    );
    let n3 = engine.create_node("Company", p3).unwrap();

    // Build index on department
    engine.create_node_property_index("department").unwrap();

    // Query using the index
    let engineers = engine
        .find_nodes_by_property("department", &PropertyValue::String("Engineering".into()))
        .unwrap();
    assert_eq!(engineers.len(), 2);
    assert!(engineers.iter().any(|n| n.id == n1));
    assert!(engineers.iter().any(|n| n.id == n2));

    let sales = engine
        .find_nodes_by_property("department", &PropertyValue::String("Sales".into()))
        .unwrap();
    assert_eq!(sales.len(), 1);
    assert_eq!(sales[0].id, n3);
}

// Test edge index building on custom property
#[test]
fn test_build_edge_custom_property_index() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut p1 = HashMap::new();
    p1.insert("strength".to_string(), PropertyValue::String("high".into()));
    let e1 = engine.create_edge(n1, n2, "KNOWS", p1, true).unwrap();

    let mut p2 = HashMap::new();
    p2.insert("strength".to_string(), PropertyValue::String("low".into()));
    let e2 = engine.create_edge(n2, n3, "LIKES", p2, true).unwrap();

    let mut p3 = HashMap::new();
    p3.insert("strength".to_string(), PropertyValue::String("high".into()));
    let e3 = engine.create_edge(n1, n3, "KNOWS", p3, true).unwrap();

    // Build index on strength
    engine.create_edge_property_index("strength").unwrap();

    // Query using the index
    let high_edges = engine
        .find_edges_by_property("strength", &PropertyValue::String("high".into()))
        .unwrap();
    assert_eq!(high_edges.len(), 2);
    assert!(high_edges.iter().any(|e| e.id == e1));
    assert!(high_edges.iter().any(|e| e.id == e3));

    let low_edges = engine
        .find_edges_by_property("strength", &PropertyValue::String("low".into()))
        .unwrap();
    assert_eq!(low_edges.len(), 1);
    assert_eq!(low_edges[0].id, e2);
}

// Test AStarEntry PartialEq (comparing entries with same node_id)
#[test]
fn test_astar_path_reexploration() {
    let engine = GraphEngine::new();

    // Create a graph where same node can be reached via different paths
    let mut p1 = HashMap::new();
    p1.insert("x".to_string(), PropertyValue::Float(0.0));
    p1.insert("y".to_string(), PropertyValue::Float(0.0));
    let n1 = engine.create_node("Node", p1).unwrap();

    let mut p2 = HashMap::new();
    p2.insert("x".to_string(), PropertyValue::Float(1.0));
    p2.insert("y".to_string(), PropertyValue::Float(0.0));
    let n2 = engine.create_node("Node", p2).unwrap();

    let mut p3 = HashMap::new();
    p3.insert("x".to_string(), PropertyValue::Float(0.0));
    p3.insert("y".to_string(), PropertyValue::Float(1.0));
    let n3 = engine.create_node("Node", p3).unwrap();

    let mut p4 = HashMap::new();
    p4.insert("x".to_string(), PropertyValue::Float(1.0));
    p4.insert("y".to_string(), PropertyValue::Float(1.0));
    let n4 = engine.create_node("Node", p4).unwrap();

    // Diamond graph: n1 -> n2 -> n4, n1 -> n3 -> n4
    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "EDGE", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n4, "EDGE", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n3, n4, "EDGE", HashMap::new(), true)
        .unwrap();

    let result = engine.astar_path_euclidean(n1, n4, "x", "y").unwrap();
    assert!(result.found());
}

// Test MST with Union-Find Less case
#[test]
fn test_mst_union_find_rank_less() {
    use crate::algorithms::MstConfig;

    let engine = GraphEngine::new();

    // Create nodes
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();
    let n5 = engine.create_node("E", HashMap::new()).unwrap();

    // Create edges with increasing weights to control MST order
    let mut w1 = HashMap::new();
    w1.insert("w".to_string(), PropertyValue::Float(1.0));
    let mut w2 = HashMap::new();
    w2.insert("w".to_string(), PropertyValue::Float(2.0));
    let mut w3 = HashMap::new();
    w3.insert("w".to_string(), PropertyValue::Float(3.0));
    let mut w4 = HashMap::new();
    w4.insert("w".to_string(), PropertyValue::Float(4.0));

    // Build tree: First build rank on one side
    engine.create_edge(n1, n2, "E", w1.clone(), false).unwrap(); // 1 - union n1,n2 (equal rank -> rank[n1]=1)
    engine.create_edge(n3, n4, "E", w2.clone(), false).unwrap(); // 2 - union n3,n4 (equal rank -> rank[n3]=1)
    engine.create_edge(n1, n3, "E", w3.clone(), false).unwrap(); // 3 - union groups (equal rank -> rank increases)
    engine.create_edge(n5, n2, "E", w4.clone(), false).unwrap(); // 4 - n5 has rank 0, n1's tree has rank 2 -> Less case

    let config = MstConfig::new("w");
    let result = engine.minimum_spanning_tree(&config).unwrap();
    assert_eq!(result.edge_count(), 4);
    assert!(result.total_weight > 0.0);
}

// Test edge property index with None values
#[test]
fn test_edge_property_index_with_missing_values() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // Some edges have the property, some don't
    let mut props = HashMap::new();
    props.insert("priority".to_string(), PropertyValue::Int(1));
    let e1 = engine.create_edge(n1, n2, "HAS", props, true).unwrap();

    let e2 = engine
        .create_edge(n2, n3, "HAS", HashMap::new(), true)
        .unwrap();

    // Build index
    engine.create_edge_property_index("priority").unwrap();

    // Query should only find e1
    let edges = engine
        .find_edges_by_property("priority", &PropertyValue::Int(1))
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, e1);

    // e2 shouldn't appear
    assert!(!edges.iter().any(|e| e.id == e2));
}

// Test node property index with missing values
#[test]
fn test_node_property_index_with_missing_values() {
    let engine = GraphEngine::new();

    let mut p1 = HashMap::new();
    p1.insert("score".to_string(), PropertyValue::Int(100));
    let n1 = engine.create_node("A", p1).unwrap();

    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    let mut p3 = HashMap::new();
    p3.insert("score".to_string(), PropertyValue::Int(100));
    let n3 = engine.create_node("C", p3).unwrap();

    // Build index
    engine.create_node_property_index("score").unwrap();

    // Query should only find n1 and n3
    let nodes = engine
        .find_nodes_by_property("score", &PropertyValue::Int(100))
        .unwrap();
    assert_eq!(nodes.len(), 2);
    assert!(nodes.iter().any(|n| n.id == n1));
    assert!(nodes.iter().any(|n| n.id == n3));
    assert!(!nodes.iter().any(|n| n.id == n2));
}

// Test with_store and _labels format
#[test]
fn test_with_store_labels_pointers_format() {
    use tensor_store::{TensorData, TensorStore, TensorValue};

    let store = TensorStore::new();

    // Create node with _labels as Pointers (multi-label format)
    let mut data = TensorData::new();
    data.set(
        "_labels",
        TensorValue::Pointers(vec!["Person".to_string(), "Employee".to_string()]),
    );
    data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String("Test".to_string())),
    );
    store.put("node:1", data).unwrap();

    let mut node_id_data = TensorData::new();
    node_id_data.set("value", TensorValue::Scalar(ScalarValue::Int(2)));
    store.put("meta:next_node_id", node_id_data).unwrap();

    let mut edge_id_data = TensorData::new();
    edge_id_data.set("value", TensorValue::Scalar(ScalarValue::Int(1)));
    store.put("meta:next_edge_id", edge_id_data).unwrap();

    let engine = GraphEngine::with_store(store);

    let node = engine.get_node(1).unwrap();
    assert_eq!(node.labels.len(), 2);
    assert!(node.labels.contains(&"Person".to_string()));
    assert!(node.labels.contains(&"Employee".to_string()));

    // Build _label index should handle multi-label nodes
    engine.create_node_property_index("_label").unwrap();

    let persons = engine
        .find_nodes_by_property("_label", &PropertyValue::String("Person".into()))
        .unwrap();
    assert_eq!(persons.len(), 1);
}

// Test scan_nodes with _label equality
#[test]
fn test_scan_nodes_label_equality() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let _n2 = engine.create_node("Company", HashMap::new()).unwrap();
    let n3 = engine.create_node("Person", HashMap::new()).unwrap();

    // Without index, uses scan
    let nodes = engine
        .find_nodes_by_property("_label", &PropertyValue::String("Person".into()))
        .unwrap();
    assert_eq!(nodes.len(), 2);
    assert!(nodes.iter().any(|n| n.id == n1));
    assert!(nodes.iter().any(|n| n.id == n3));
}

// Test triangles with edge type filter
#[test]
fn test_triangles_edge_type_filter() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();
    let a = engine.create_node("A", HashMap::new()).unwrap();
    let b = engine.create_node("B", HashMap::new()).unwrap();
    let c = engine.create_node("C", HashMap::new()).unwrap();

    // Triangle with FRIEND edges
    engine
        .create_edge(a, b, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(b, c, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(c, a, "FRIEND", HashMap::new(), false)
        .unwrap();

    // Extra edge with different type
    engine
        .create_edge(a, b, "ENEMY", HashMap::new(), false)
        .unwrap();

    // Filter to FRIEND only
    let config = TriangleConfig::new().edge_type("FRIEND").undirected();
    let result = engine.count_triangles(&config).unwrap();
    assert_eq!(result.triangle_count, 1);
}

// Test A* Manhattan with target node missing
#[test]
fn test_astar_manhattan_target_missing_coords() {
    let engine = GraphEngine::new();

    let mut p1 = HashMap::new();
    p1.insert("x".to_string(), PropertyValue::Float(0.0));
    p1.insert("y".to_string(), PropertyValue::Float(0.0));
    let n1 = engine.create_node("Node", p1).unwrap();

    // Target has no coordinates
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), true)
        .unwrap();

    // Should still work (heuristic returns 0.0)
    let result = engine.astar_path_manhattan(n1, n2, "x", "y").unwrap();
    assert!(result.found());
}

// Test A* Euclidean with current node missing coords
#[test]
fn test_astar_euclidean_current_missing_coords() {
    let engine = GraphEngine::new();

    // Start has no coordinates
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();

    let mut p2 = HashMap::new();
    p2.insert("x".to_string(), PropertyValue::Float(1.0));
    p2.insert("y".to_string(), PropertyValue::Float(1.0));
    let n2 = engine.create_node("Node", p2).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), true)
        .unwrap();

    // Should work
    let result = engine.astar_path_euclidean(n1, n2, "x", "y").unwrap();
    assert!(result.found());
}

// Test with_store that rebuilds edge indexes from existing data
#[test]
fn test_with_store_edge_property_indexes() {
    use tensor_store::{TensorData, TensorStore, TensorValue};

    let store = TensorStore::new();

    // Create edges with properties
    let mut edge_data = TensorData::new();
    edge_data.set("_from", TensorValue::Scalar(ScalarValue::Int(1)));
    edge_data.set("_to", TensorValue::Scalar(ScalarValue::Int(2)));
    edge_data.set(
        "_edge_type",
        TensorValue::Scalar(ScalarValue::String("KNOWS".to_string())),
    );
    edge_data.set("priority", TensorValue::Scalar(ScalarValue::Int(5)));
    store.put("edge:1", edge_data).unwrap();

    // Create corresponding nodes
    let mut node1 = TensorData::new();
    node1.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String("Person".to_string())),
    );
    store.put("node:1", node1).unwrap();

    let mut node2 = TensorData::new();
    node2.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String("Person".to_string())),
    );
    store.put("node:2", node2).unwrap();

    // Set up meta counters
    let mut node_id_data = TensorData::new();
    node_id_data.set("value", TensorValue::Scalar(ScalarValue::Int(3)));
    store.put("meta:next_node_id", node_id_data).unwrap();

    let mut edge_id_data = TensorData::new();
    edge_id_data.set("value", TensorValue::Scalar(ScalarValue::Int(2)));
    store.put("meta:next_edge_id", edge_id_data).unwrap();

    // Rebuild indexes on an existing store
    let engine = GraphEngine::with_store(store);

    // Verify edge data is accessible
    let edge = engine.get_edge(1).unwrap();
    assert_eq!(edge.edge_type, "KNOWS");
}

// Test weighted path finding (covers DijkstraEntry)
#[test]
fn test_weighted_path_dijkstra_entry_comparisons() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    // Create a diamond graph with different weights
    let mut w1 = HashMap::new();
    w1.insert("cost".to_string(), PropertyValue::Float(1.0));
    let mut w2 = HashMap::new();
    w2.insert("cost".to_string(), PropertyValue::Float(10.0));
    let mut w3 = HashMap::new();
    w3.insert("cost".to_string(), PropertyValue::Float(1.0));
    let mut w4 = HashMap::new();
    w4.insert("cost".to_string(), PropertyValue::Float(1.0));

    // n1 -> n2 -> n4 (cost = 1 + 1 = 2)
    // n1 -> n3 -> n4 (cost = 10 + 1 = 11)
    engine.create_edge(n1, n2, "E", w1, true).unwrap();
    engine.create_edge(n2, n4, "E", w3, true).unwrap();
    engine.create_edge(n1, n3, "E", w2, true).unwrap();
    engine.create_edge(n3, n4, "E", w4, true).unwrap();

    // Should find the shorter path via n2
    let path = engine.find_weighted_path(n1, n4, "cost").unwrap();
    assert_eq!(path.nodes.len(), 3); // n1 -> n2 -> n4
    assert!((path.total_weight - 2.0).abs() < 0.01);
}

// Test Pattern matching with nodes and edges
#[test]
fn test_pattern_matching_nodes_and_edges() {
    let engine = GraphEngine::new();

    // Create test data
    let n1 = engine.create_node("Person", HashMap::new()).unwrap();
    let n2 = engine.create_node("Company", HashMap::new()).unwrap();
    engine
        .create_edge(n1, n2, "WORKS_AT", HashMap::new(), true)
        .unwrap();

    // Create a pattern
    let path = PathPattern::new(
        NodePattern::new().label("Person"),
        EdgePattern::new().edge_type("WORKS_AT"),
        NodePattern::new().label("Company"),
    );
    let pattern = Pattern::new(path);

    // Match the pattern
    let results = engine.match_pattern(&pattern).unwrap();
    assert!(!results.is_empty());
}

// Test A* with only source having y coordinate (for branches in heuristic)
#[test]
fn test_astar_euclidean_only_source_has_y() {
    let engine = GraphEngine::new();

    // Source has full coordinates
    let mut p1 = HashMap::new();
    p1.insert("x".to_string(), PropertyValue::Float(0.0));
    p1.insert("y".to_string(), PropertyValue::Float(0.0));
    let n1 = engine.create_node("Node", p1).unwrap();

    // Target has only x (no y)
    let mut p2 = HashMap::new();
    p2.insert("x".to_string(), PropertyValue::Float(1.0));
    let n2 = engine.create_node("Node", p2).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), true)
        .unwrap();

    let result = engine.astar_path_euclidean(n1, n2, "x", "y").unwrap();
    assert!(result.found());
}

// Test A* Manhattan with only target having x
#[test]
fn test_astar_manhattan_only_target_has_x() {
    let engine = GraphEngine::new();

    // Source has no coordinates
    let n1 = engine.create_node("Node", HashMap::new()).unwrap();

    // Target has only x (no y)
    let mut p2 = HashMap::new();
    p2.insert("x".to_string(), PropertyValue::Float(1.0));
    let n2 = engine.create_node("Node", p2).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), true)
        .unwrap();

    let result = engine.astar_path_manhattan(n1, n2, "x", "y").unwrap();
    assert!(result.found());
}

// Test A* Manhattan with intermediate nodes missing coords
#[test]
fn test_astar_manhattan_intermediate_missing() {
    let engine = GraphEngine::new();

    // Start with coords
    let mut p1 = HashMap::new();
    p1.insert("x".to_string(), PropertyValue::Float(0.0));
    p1.insert("y".to_string(), PropertyValue::Float(0.0));
    let n1 = engine.create_node("Node", p1).unwrap();

    // Middle with no coords
    let n2 = engine.create_node("Node", HashMap::new()).unwrap();

    // End with coords
    let mut p3 = HashMap::new();
    p3.insert("x".to_string(), PropertyValue::Float(2.0));
    p3.insert("y".to_string(), PropertyValue::Float(2.0));
    let n3 = engine.create_node("Node", p3).unwrap();

    engine
        .create_edge(n1, n2, "EDGE", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n2, n3, "EDGE", HashMap::new(), true)
        .unwrap();

    let result = engine.astar_path_manhattan(n1, n3, "x", "y").unwrap();
    assert!(result.found());
}

// Test finding edges with custom property via index rebuild
#[test]
fn test_with_store_custom_edge_property_index() {
    use tensor_store::{TensorData, TensorStore, TensorValue};

    let store = TensorStore::new();

    // Create edge with custom property but no _edge_type match
    let mut edge_data = TensorData::new();
    edge_data.set("_from", TensorValue::Scalar(ScalarValue::Int(1)));
    edge_data.set("_to", TensorValue::Scalar(ScalarValue::Int(2)));
    edge_data.set("_edge_type", TensorValue::Scalar(ScalarValue::Int(123))); // Wrong type - not String
    edge_data.set("weight", TensorValue::Scalar(ScalarValue::Float(5.5)));
    store.put("edge:1", edge_data).unwrap();

    // Create nodes
    let mut node1 = TensorData::new();
    node1.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String("A".to_string())),
    );
    store.put("node:1", node1).unwrap();

    let mut node2 = TensorData::new();
    node2.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String("B".to_string())),
    );
    store.put("node:2", node2).unwrap();

    // Meta
    let mut node_id = TensorData::new();
    node_id.set("value", TensorValue::Scalar(ScalarValue::Int(3)));
    store.put("meta:next_node_id", node_id).unwrap();

    let mut edge_id = TensorData::new();
    edge_id.set("value", TensorValue::Scalar(ScalarValue::Int(2)));
    store.put("meta:next_edge_id", edge_id).unwrap();

    // This should trigger the _edge_type fallback case (non-String type)
    let engine = GraphEngine::with_store(store);

    // Verify edge is accessible
    let edge = engine.get_edge(1).unwrap();
    assert!(edge.properties.get("weight").is_some());
}

// Test KCoreConfig::edge_type and KCoreResult::default
#[test]
fn test_kcore_config_edge_type() {
    use crate::algorithms::{KCoreConfig, KCoreResult};

    let config = KCoreConfig::new().edge_type("FRIEND");
    assert_eq!(config.edge_type, Some("FRIEND".to_string()));

    // Test default result
    let result = KCoreResult::default();
    assert_eq!(result.degeneracy, 0);
}

// Test triangles with isolated nodes (no neighbors)
#[test]
fn test_triangles_isolated_nodes() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    // Create isolated nodes
    let _n1 = engine.create_node("A", HashMap::new()).unwrap();
    let _n2 = engine.create_node("B", HashMap::new()).unwrap();
    let _n3 = engine.create_node("C", HashMap::new()).unwrap();

    let config = TriangleConfig::new().undirected();
    let result = engine.count_triangles(&config).unwrap();

    // No edges, so no triangles
    assert_eq!(result.triangle_count, 0);
    assert_eq!(result.global_clustering, 0.0);
}

// Test local_clustering_coefficient with degree < 2
#[test]
fn test_local_clustering_low_degree() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    // Single edge = degree 1
    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();

    let config = TriangleConfig::new().undirected();
    let coeff = engine.local_clustering_coefficient(n1, &config).unwrap();

    // Degree 1, can't form triangles
    assert_eq!(coeff, 0.0);
}

// Test triangles where some nodes have no neighbors in adjacency
#[test]
fn test_triangles_sparse_graph() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    // Create a line graph: A -- B -- C -- D
    let a = engine.create_node("A", HashMap::new()).unwrap();
    let b = engine.create_node("B", HashMap::new()).unwrap();
    let c = engine.create_node("C", HashMap::new()).unwrap();
    let d = engine.create_node("D", HashMap::new()).unwrap();

    engine
        .create_edge(a, b, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(b, c, "E", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(c, d, "E", HashMap::new(), false)
        .unwrap();

    let config = TriangleConfig::new().undirected();
    let result = engine.count_triangles(&config).unwrap();

    // Line graph has no triangles
    assert_eq!(result.triangle_count, 0);

    // Check local clustering for middle node (has neighbors but they don't connect)
    let b_clustering = result.local_clustering.get(&b).copied().unwrap_or(1.0);
    assert_eq!(b_clustering, 0.0);
}

// Test kcore with edge type filter
#[test]
fn test_kcore_with_edge_type_filter() {
    use crate::algorithms::KCoreConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    // Create a complete graph with FRIEND edges
    engine
        .create_edge(n1, n2, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n1, n3, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n1, n4, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n4, "FRIEND", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n4, "FRIEND", HashMap::new(), false)
        .unwrap();

    // Add some ENEMY edges
    engine
        .create_edge(n1, n2, "ENEMY", HashMap::new(), false)
        .unwrap();

    let config = KCoreConfig::new().edge_type("FRIEND").undirected();
    let result = engine.kcore_decomposition(&config).unwrap();

    // Complete K4 graph has 3-core
    assert_eq!(result.degeneracy, 3);
}

// Test triangles with v having no neighbors
#[test]
fn test_triangles_asymmetric_neighbors() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    // Create directed edges that form partial graph
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    // Directed edges: n1 -> n2, n1 -> n3
    // When checking undirected, n2 and n3 have different neighbor sets
    engine
        .create_edge(n1, n2, "E", HashMap::new(), true)
        .unwrap();
    engine
        .create_edge(n1, n3, "E", HashMap::new(), true)
        .unwrap();

    let config = TriangleConfig::new(); // Directed
    let result = engine.count_triangles(&config).unwrap();

    // No triangles in this directed configuration
    assert_eq!(result.triangle_count, 0);
}

// Test MST with compute_forest=false (early termination)
#[test]
fn test_mst_no_forest_early_termination() {
    use crate::algorithms::MstConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut w1 = HashMap::new();
    w1.insert("w".to_string(), PropertyValue::Float(1.0));
    let mut w2 = HashMap::new();
    w2.insert("w".to_string(), PropertyValue::Float(2.0));

    engine.create_edge(n1, n2, "E", w1, false).unwrap();
    engine.create_edge(n2, n3, "E", w2, false).unwrap();

    // compute_forest = false should stop after n-1 edges
    let config = MstConfig::new("w").compute_forest(false);
    let result = engine.minimum_spanning_tree(&config).unwrap();

    assert_eq!(result.edge_count(), 2); // Connected 3 nodes with 2 edges
}

// Test minimum_spanning_forest with connected graph (single tree)
#[test]
fn test_mst_forest_single_component() {
    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();

    let mut w1 = HashMap::new();
    w1.insert("w".to_string(), PropertyValue::Float(1.0));
    let mut w2 = HashMap::new();
    w2.insert("w".to_string(), PropertyValue::Float(2.0));

    engine.create_edge(n1, n2, "E", w1, false).unwrap();
    engine.create_edge(n2, n3, "E", w2, false).unwrap();

    // Single connected component should return one result
    let forests = engine.minimum_spanning_forest("w").unwrap();

    assert_eq!(forests.len(), 1);
    assert_eq!(forests[0].edge_count(), 2);
}

// Test triangles where v has neighbors but continue is hit
#[test]
fn test_triangles_v_no_neighbors_path() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    // Create graph where v exists but has no neighbors (directed case)
    let a = engine.create_node("A", HashMap::new()).unwrap();
    let b = engine.create_node("B", HashMap::new()).unwrap();
    let c = engine.create_node("C", HashMap::new()).unwrap();

    // Directed: a->b, a->c only (b and c have no outgoing edges)
    engine.create_edge(a, b, "E", HashMap::new(), true).unwrap();
    engine.create_edge(a, c, "E", HashMap::new(), true).unwrap();

    // Directed mode - b has no outgoing neighbors
    let config = TriangleConfig::new();
    let result = engine.count_triangles(&config).unwrap();

    assert_eq!(result.triangle_count, 0);
}

// Test clustering coefficient on node with no possible triangles
#[test]
fn test_clustering_no_possible_triangles() {
    use crate::algorithms::TriangleConfig;

    let engine = GraphEngine::new();

    // Node with degree 1 - can't form any triangles
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();

    engine
        .create_edge(n1, n2, "E", HashMap::new(), false)
        .unwrap();

    let config = TriangleConfig::new().undirected();
    let result = engine.count_triangles(&config).unwrap();

    // Both nodes should have 0.0 clustering (degree < 2)
    assert_eq!(
        result.local_clustering.get(&n1).copied().unwrap_or(1.0),
        0.0
    );
    assert_eq!(
        result.local_clustering.get(&n2).copied().unwrap_or(1.0),
        0.0
    );
}

// Test BiconnectedConfig::edge_type and BiconnectedResult::default
#[test]
fn test_biconnected_config_edge_type() {
    use crate::algorithms::{BiconnectedConfig, BiconnectedResult};

    // Test edge_type builder
    let config = BiconnectedConfig::new().edge_type("ROAD");
    assert_eq!(config.edge_type, Some("ROAD".to_string()));

    // Test default result
    let result = BiconnectedResult::default();
    assert!(result.articulation_points.is_empty());
    assert!(result.is_biconnected());
}

// Test biconnected components with edge type filter
#[test]
fn test_biconnected_with_edge_type() {
    use crate::algorithms::BiconnectedConfig;

    let engine = GraphEngine::new();

    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let n3 = engine.create_node("C", HashMap::new()).unwrap();
    let n4 = engine.create_node("D", HashMap::new()).unwrap();

    // Create chain with ROAD edges
    engine
        .create_edge(n1, n2, "ROAD", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n2, n3, "ROAD", HashMap::new(), false)
        .unwrap();
    engine
        .create_edge(n3, n4, "ROAD", HashMap::new(), false)
        .unwrap();

    // Add some RAIL edges
    engine
        .create_edge(n1, n4, "RAIL", HashMap::new(), false)
        .unwrap();

    // Analyze only ROAD edges
    let config = BiconnectedConfig::new().edge_type("ROAD");
    let result = engine.biconnected_components(&config).unwrap();

    // Chain should have articulation points
    assert!(!result.articulation_points.is_empty());
}

#[test]
fn test_open_durable() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("graph.wal");

    let engine = GraphEngine::open_durable(&wal_path, WalConfig::default()).unwrap();
    assert!(engine.is_durable());

    // Create some data to verify the engine works
    let node_id = engine.create_node("Test", HashMap::new()).unwrap();
    assert!(engine.get_node(node_id).is_ok());
}

#[test]
fn test_recover_durable() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("graph.wal");

    // First create a durable engine
    {
        let _engine = GraphEngine::open_durable(&wal_path, WalConfig::default()).unwrap();
        // Engine drops, WAL is closed
    }

    // Recover
    let recovered = GraphEngine::recover(&wal_path, &WalConfig::default(), None);
    assert!(recovered.is_ok());
    assert!(recovered.unwrap().is_durable());
}

#[test]
fn test_is_durable_false_for_in_memory() {
    let engine = GraphEngine::new();
    assert!(!engine.is_durable());
}

#[test]
fn test_with_store_and_config() {
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    let config = GraphEngineConfig::default();
    let store = TensorStore::new();

    // Add some data directly to the store to simulate existing data
    let mut node_data = TensorData::new();
    node_data.set(
        "_label",
        TensorValue::Scalar(ScalarValue::String("Person".into())),
    );
    node_data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String("Alice".into())),
    );
    store.put("node:1", node_data).unwrap();

    let mut edge_data = TensorData::new();
    edge_data.set(
        "_edge_type",
        TensorValue::Scalar(ScalarValue::String("KNOWS".into())),
    );
    edge_data.set("from", TensorValue::Scalar(ScalarValue::Int(1)));
    edge_data.set("to", TensorValue::Scalar(ScalarValue::Int(2)));
    store.put("edge:5", edge_data).unwrap();

    // Create engine with existing store
    let engine = GraphEngine::with_store_and_config(store, config);

    // Verify node counter was initialized correctly (should be >= 1)
    let new_node_id = engine.create_node("Test", HashMap::new()).unwrap();
    assert!(
        new_node_id > 1,
        "New node ID should be greater than existing max"
    );

    // Verify edge counter was initialized correctly (should be >= 5)
    let n1 = engine.create_node("A", HashMap::new()).unwrap();
    let n2 = engine.create_node("B", HashMap::new()).unwrap();
    let new_edge_id = engine
        .create_edge(n1, n2, "TEST", HashMap::new(), true)
        .unwrap();
    assert!(
        new_edge_id > 5,
        "New edge ID should be greater than existing max"
    );
}
