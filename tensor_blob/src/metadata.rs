// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use std::collections::HashMap;

/// Options for storing a new artifact.
#[derive(Debug, Clone, Default)]
pub struct PutOptions {
    /// Content type (MIME type). Uses default if not specified.
    pub content_type: Option<String>,
    /// Creator identity (e.g., "user:alice", "agent:summarizer").
    pub created_by: Option<String>,
    /// Entities to link this artifact to.
    pub linked_to: Vec<String>,
    /// Tags to apply.
    pub tags: Vec<String>,
    /// Custom metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// Optional embedding with model name.
    pub embedding: Option<(Vec<f32>, String)>,
}

impl PutOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    #[must_use]
    pub fn with_created_by(mut self, creator: impl Into<String>) -> Self {
        self.created_by = Some(creator.into());
        self
    }

    #[must_use]
    pub fn with_link(mut self, entity: impl Into<String>) -> Self {
        self.linked_to.push(entity.into());
        self
    }

    #[must_use]
    pub fn with_links(mut self, entities: Vec<String>) -> Self {
        self.linked_to.extend(entities);
        self
    }

    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>, model: impl Into<String>) -> Self {
        self.embedding = Some((embedding, model.into()));
        self
    }
}

/// Metadata about a stored artifact.
#[derive(Debug, Clone)]
pub struct ArtifactMetadata {
    /// Unique artifact identifier.
    pub id: String,
    /// Original filename.
    pub filename: String,
    /// Content type (MIME type).
    pub content_type: String,
    /// Total size in bytes.
    pub size: usize,
    /// Full file checksum (e.g., "sha256:...").
    pub checksum: String,
    /// Number of chunks.
    pub chunk_count: usize,
    /// Size of each chunk (except possibly the last).
    pub chunk_size: usize,
    /// Creator identity.
    pub created_by: String,
    /// Creation timestamp (Unix epoch seconds).
    pub created: u64,
    /// Last modification timestamp.
    pub modified: u64,
    /// Linked entity IDs.
    pub linked_to: Vec<String>,
    /// Applied tags.
    pub tags: Vec<String>,
    /// Custom metadata.
    pub custom: HashMap<String, String>,
    /// Whether an embedding is set.
    pub has_embedding: bool,
    /// Embedding model name if set.
    pub embedding_model: Option<String>,
}

/// Updates to apply to artifact metadata.
#[derive(Debug, Clone, Default)]
pub struct MetadataUpdates {
    /// New filename.
    pub filename: Option<String>,
    /// New content type.
    pub content_type: Option<String>,
    /// Custom metadata updates. None value means delete.
    pub custom: HashMap<String, Option<String>>,
}

impl MetadataUpdates {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    #[must_use]
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    #[must_use]
    pub fn set_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), Some(value.into()));
        self
    }

    #[must_use]
    pub fn delete_meta(mut self, key: impl Into<String>) -> Self {
        self.custom.insert(key.into(), None);
        self
    }
}

/// Result from similarity search.
#[derive(Debug, Clone)]
pub struct SimilarArtifact {
    /// Artifact ID.
    pub id: String,
    /// Original filename.
    pub filename: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
}

/// Statistics about blob storage.
#[derive(Debug, Clone, Default)]
pub struct BlobStats {
    /// Total number of artifacts.
    pub artifact_count: usize,
    /// Total number of stored chunks.
    pub chunk_count: usize,
    /// Total logical bytes (sum of artifact sizes).
    pub total_bytes: usize,
    /// Actual unique bytes stored (after deduplication).
    pub unique_bytes: usize,
    /// Deduplication ratio (1.0 - unique/total).
    pub dedup_ratio: f64,
    /// Number of orphaned chunks (zero references).
    pub orphaned_chunks: usize,
}

/// Result from garbage collection.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of chunks deleted.
    pub deleted: usize,
    /// Bytes freed.
    pub freed_bytes: usize,
}

impl GcStats {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result from repair operation.
#[derive(Debug, Clone, Default)]
pub struct RepairStats {
    /// Number of artifacts checked.
    pub artifacts_checked: usize,
    /// Number of chunks verified.
    pub chunks_verified: usize,
    /// Number of references fixed.
    pub refs_fixed: usize,
    /// Number of orphans deleted.
    pub orphans_deleted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_options_builder() {
        let options = PutOptions::new()
            .with_content_type("application/pdf")
            .with_created_by("user:alice")
            .with_link("task:123")
            .with_links(vec!["project:neumann".to_string()])
            .with_tag("quarterly")
            .with_tags(vec!["finance".to_string(), "report".to_string()])
            .with_meta("author", "Alice")
            .with_embedding(vec![0.1, 0.2, 0.3], "text-embedding-3-small");

        assert_eq!(options.content_type, Some("application/pdf".to_string()));
        assert_eq!(options.created_by, Some("user:alice".to_string()));
        assert_eq!(
            options.linked_to,
            vec!["task:123".to_string(), "project:neumann".to_string()]
        );
        assert_eq!(
            options.tags,
            vec![
                "quarterly".to_string(),
                "finance".to_string(),
                "report".to_string()
            ]
        );
        assert_eq!(options.metadata.get("author"), Some(&"Alice".to_string()));
        assert!(options.embedding.is_some());
        let (emb, model) = options.embedding.unwrap();
        assert_eq!(emb, vec![0.1, 0.2, 0.3]);
        assert_eq!(model, "text-embedding-3-small");
    }

    #[test]
    fn test_metadata_updates_builder() {
        let updates = MetadataUpdates::new()
            .with_filename("new_name.pdf")
            .with_content_type("application/pdf")
            .set_meta("author", "Bob")
            .delete_meta("old_field");

        assert_eq!(updates.filename, Some("new_name.pdf".to_string()));
        assert_eq!(updates.content_type, Some("application/pdf".to_string()));
        assert_eq!(updates.custom.get("author"), Some(&Some("Bob".to_string())));
        assert_eq!(updates.custom.get("old_field"), Some(&None));
    }

    #[test]
    fn test_gc_stats_default() {
        let stats = GcStats::default();
        assert_eq!(stats.deleted, 0);
        assert_eq!(stats.freed_bytes, 0);
    }

    #[test]
    fn test_blob_stats_default() {
        let stats = BlobStats::default();
        assert_eq!(stats.artifact_count, 0);
        assert_eq!(stats.chunk_count, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.unique_bytes, 0);
        assert_eq!(stats.dedup_ratio, 0.0);
        assert_eq!(stats.orphaned_chunks, 0);
    }

    #[test]
    fn test_similar_artifact() {
        let similar = SimilarArtifact {
            id: "artifact:report.pdf".to_string(),
            filename: "report.pdf".to_string(),
            similarity: 0.92,
        };
        assert_eq!(similar.id, "artifact:report.pdf");
        assert_eq!(similar.filename, "report.pdf");
        assert!((similar.similarity - 0.92).abs() < f32::EPSILON);
    }
}
