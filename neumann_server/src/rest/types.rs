// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! REST API request and response types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A point with vector and optional payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointStruct {
    /// Unique point identifier.
    pub id: String,
    /// Dense vector embedding.
    pub vector: Vec<f32>,
    /// Optional payload as JSON values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

/// Request to upsert points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    /// Points to upsert.
    pub points: Vec<PointStruct>,
}

/// Response from upsert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    /// Operation status.
    pub status: String,
    /// Number of points upserted.
    pub upserted: usize,
}

/// Request to get points by IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequest {
    /// Point IDs to retrieve.
    pub ids: Vec<String>,
    /// Include payload in response.
    #[serde(default)]
    pub with_payload: bool,
    /// Include vector in response.
    #[serde(default = "default_true")]
    pub with_vector: bool,
}

fn default_true() -> bool {
    true
}

/// Response with retrieved points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetResponse {
    /// Retrieved points.
    pub points: Vec<PointStruct>,
}

/// Request to delete points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    /// Point IDs to delete.
    pub ids: Vec<String>,
}

/// Response from delete operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// Operation status.
    pub status: String,
    /// Number of points deleted.
    pub deleted: usize,
}

fn default_limit() -> usize {
    10
}

/// Request to query similar points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Query vector.
    pub vector: Vec<f32>,
    /// Maximum number of results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Number of results to skip.
    #[serde(default)]
    pub offset: usize,
    /// Minimum similarity score threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f32>,
    /// Include payload in response.
    #[serde(default)]
    pub with_payload: bool,
    /// Include vector in response.
    #[serde(default)]
    pub with_vector: bool,
}

/// A point with similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPoint {
    /// Unique point identifier.
    pub id: String,
    /// Similarity score.
    pub score: f32,
    /// Optional payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, serde_json::Value>>,
    /// Optional vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// Response with similar points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Similar points with scores.
    pub result: Vec<ScoredPoint>,
    /// Query execution time in seconds.
    pub time: f64,
}

fn default_scroll_limit() -> usize {
    100
}

/// Request to scroll through points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollRequest {
    /// Offset point ID for pagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_id: Option<String>,
    /// Maximum number of points to return.
    #[serde(default = "default_scroll_limit")]
    pub limit: usize,
    /// Include payload in response.
    #[serde(default)]
    pub with_payload: bool,
    /// Include vector in response.
    #[serde(default = "default_true")]
    pub with_vector: bool,
}

/// Response with scrolled points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollResponse {
    /// Points in this page.
    pub points: Vec<PointStruct>,
    /// Next offset ID for pagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_offset: Option<String>,
}

fn default_distance() -> String {
    "cosine".to_string()
}

/// Request to create a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    /// Vector dimension (required).
    pub dimension: usize,
    /// Distance metric: "cosine", "euclidean", "dot".
    #[serde(default = "default_distance")]
    pub distance: String,
}

/// Response from create collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionResponse {
    /// Whether the collection was created.
    pub created: bool,
}

/// Collection information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// Collection name.
    pub name: String,
    /// Number of points in collection.
    pub points_count: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Distance metric.
    pub distance: String,
}

/// Response from delete collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteCollectionResponse {
    /// Whether the collection was deleted.
    pub deleted: bool,
}

/// Response from list collections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListCollectionsResponse {
    /// Collection names.
    pub collections: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_struct_serde_roundtrip() {
        let point = PointStruct {
            id: "test-id".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            payload: Some(HashMap::from([(
                "key".to_string(),
                serde_json::json!("value"),
            )])),
        };

        let json = serde_json::to_string(&point).unwrap();
        let parsed: PointStruct = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-id");
        assert_eq!(parsed.vector, vec![0.1, 0.2, 0.3]);
        assert!(parsed.payload.is_some());
    }

    #[test]
    fn test_point_struct_optional_payload() {
        let point = PointStruct {
            id: "test-id".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            payload: None,
        };

        let json = serde_json::to_string(&point).unwrap();
        assert!(!json.contains("payload"));

        let parsed: PointStruct = serde_json::from_str(&json).unwrap();
        assert!(parsed.payload.is_none());
    }

    #[test]
    fn test_query_request_defaults() {
        let json = r#"{"vector": [0.1, 0.2, 0.3]}"#;
        let request: QueryRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.limit, 10);
        assert_eq!(request.offset, 0);
        assert!(request.score_threshold.is_none());
        assert!(!request.with_payload);
        assert!(!request.with_vector);
    }

    #[test]
    fn test_scored_point_skip_none() {
        let point = ScoredPoint {
            id: "test-id".to_string(),
            score: 0.95,
            payload: None,
            vector: None,
        };

        let json = serde_json::to_string(&point).unwrap();
        assert!(!json.contains("payload"));
        assert!(!json.contains("vector"));
    }

    #[test]
    fn test_collection_info_serde() {
        let info = CollectionInfo {
            name: "test-collection".to_string(),
            points_count: 1000,
            dimension: 128,
            distance: "cosine".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: CollectionInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "test-collection");
        assert_eq!(parsed.points_count, 1000);
        assert_eq!(parsed.dimension, 128);
        assert_eq!(parsed.distance, "cosine");
    }

    #[test]
    fn test_scroll_request_defaults() {
        let json = r"{}";
        let request: ScrollRequest = serde_json::from_str(json).unwrap();

        assert!(request.offset_id.is_none());
        assert_eq!(request.limit, 100);
        assert!(!request.with_payload);
        assert!(request.with_vector);
    }

    #[test]
    fn test_create_collection_request_defaults() {
        let json = r#"{"dimension": 128}"#;
        let request: CreateCollectionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.dimension, 128);
        assert_eq!(request.distance, "cosine");
    }

    #[test]
    fn test_get_request_defaults() {
        let json = r#"{"ids": ["id1", "id2"]}"#;
        let request: GetRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.ids, vec!["id1", "id2"]);
        assert!(!request.with_payload);
        assert!(request.with_vector);
    }
}
