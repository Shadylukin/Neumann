//! Conversion between Neumann types and protobuf messages.

use query_router::{
    ArtifactInfoResult, BlobStatsResult, ChainBlockInfo, ChainCodebookInfo, ChainDriftResult,
    ChainHistoryEntry, ChainResult, ChainSimilarResult, ChainTransitionAnalysis,
    CheckpointInfo as RouterCheckpointInfo, EdgeResult, NodeResult, QueryResult,
    SimilarResult as RouterSimilarResult, UnifiedResult,
};
use relational_engine::{Row as RelationalRow, Value as RelationalValue};

use crate::proto;

/// Convert a `QueryResult` to a protobuf `QueryResponse`.
#[must_use]
pub fn query_result_to_proto(result: QueryResult) -> proto::QueryResponse {
    let result_oneof = match result {
        QueryResult::Empty => proto::query_response::Result::Empty(proto::EmptyResult {}),

        QueryResult::Value(v) => {
            proto::query_response::Result::Value(proto::StringValue { value: v })
        },

        QueryResult::Count(c) => {
            proto::query_response::Result::Count(proto::CountResult { count: c as u64 })
        },

        QueryResult::Ids(ids) => proto::query_response::Result::Ids(proto::IdsResult { ids }),

        QueryResult::Rows(rows) => proto::query_response::Result::Rows(proto::RowsResult {
            rows: rows.into_iter().map(row_to_proto).collect(),
        }),

        QueryResult::Nodes(nodes) => proto::query_response::Result::Nodes(proto::NodesResult {
            nodes: nodes.into_iter().map(node_to_proto).collect(),
        }),

        QueryResult::Edges(edges) => proto::query_response::Result::Edges(proto::EdgesResult {
            edges: edges.into_iter().map(edge_to_proto).collect(),
        }),

        QueryResult::Path(path) => {
            proto::query_response::Result::Path(proto::PathResult { node_ids: path })
        },

        QueryResult::Similar(items) => {
            proto::query_response::Result::Similar(proto::SimilarResult {
                items: items.into_iter().map(similar_to_proto).collect(),
            })
        },

        QueryResult::Unified(unified) => {
            proto::query_response::Result::Unified(unified_to_proto(unified))
        },

        QueryResult::TableList(tables) => {
            proto::query_response::Result::TableList(proto::TableListResult { tables })
        },

        QueryResult::Blob(data) => proto::query_response::Result::Blob(proto::BlobResult { data }),

        QueryResult::ArtifactInfo(info) => {
            proto::query_response::Result::ArtifactInfo(artifact_info_to_proto(info))
        },

        QueryResult::ArtifactList(ids) => {
            proto::query_response::Result::ArtifactList(proto::ArtifactListResult {
                artifact_ids: ids,
            })
        },

        QueryResult::BlobStats(stats) => {
            proto::query_response::Result::BlobStats(blob_stats_to_proto(stats))
        },

        QueryResult::CheckpointList(checkpoints) => {
            proto::query_response::Result::CheckpointList(proto::CheckpointListResult {
                checkpoints: checkpoints.into_iter().map(checkpoint_to_proto).collect(),
            })
        },

        QueryResult::Chain(chain_result) => {
            proto::query_response::Result::Chain(chain_result_to_proto(chain_result))
        },

        QueryResult::PageRank(results) => {
            proto::query_response::Result::PageRank(proto::PageRankResult {
                items: results.into_iter().map(pagerank_item_to_proto).collect(),
            })
        },

        QueryResult::Centrality(results) => {
            proto::query_response::Result::Centrality(proto::CentralityResult {
                items: results.into_iter().map(centrality_item_to_proto).collect(),
            })
        },

        QueryResult::Communities(results) => {
            // Group nodes by community_id
            let mut communities: std::collections::HashMap<u64, Vec<u64>> =
                std::collections::HashMap::new();
            for r in &results {
                communities
                    .entry(r.community_id)
                    .or_default()
                    .push(r.node_id);
            }
            let mut sorted: Vec<_> = communities.into_iter().collect();
            sorted.sort_by_key(|(id, _)| *id);
            let value = sorted
                .iter()
                .map(|(community_id, nodes)| {
                    let node_list = nodes
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("Community {community_id}: [{node_list}]")
                })
                .collect::<Vec<_>>()
                .join("\n");
            proto::query_response::Result::Value(proto::StringValue { value })
        },

        QueryResult::Constraints(constraints) => {
            let value = constraints
                .iter()
                .map(|c| {
                    format!(
                        "{} on {} property '{}' ({})",
                        c.name, c.target, c.property, c.constraint_type
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            proto::query_response::Result::Value(proto::StringValue { value })
        },

        QueryResult::GraphIndexes(indexes) => {
            let value = indexes.join("\n");
            proto::query_response::Result::Value(proto::StringValue { value })
        },

        QueryResult::Aggregate(agg) => {
            let value = match agg {
                query_router::AggregateResultValue::Count(n) => format!("Count: {n}"),
                query_router::AggregateResultValue::Sum(v) => format!("Sum: {v}"),
                query_router::AggregateResultValue::Avg(v) => format!("Avg: {v}"),
                query_router::AggregateResultValue::Min(v) => format!("Min: {v}"),
                query_router::AggregateResultValue::Max(v) => format!("Max: {v}"),
            };
            proto::query_response::Result::Value(proto::StringValue { value })
        },

        QueryResult::BatchResult(batch) => {
            let value = format!("{}: {} affected", batch.operation, batch.affected_count);
            proto::query_response::Result::Value(proto::StringValue { value })
        },

        QueryResult::PatternMatch(pm) => {
            let value = format!(
                "Pattern Matches: {} found (truncated: {})",
                pm.stats.matches_found, pm.stats.truncated
            );
            proto::query_response::Result::Value(proto::StringValue { value })
        },
    };

    proto::QueryResponse {
        result: Some(result_oneof),
        error: None,
    }
}

/// Convert a relational `Row` to protobuf.
#[must_use]
pub fn row_to_proto(row: RelationalRow) -> proto::Row {
    proto::Row {
        id: row.id,
        values: row
            .values
            .into_iter()
            .map(|(name, value)| proto::ColumnValue {
                name,
                value: Some(value_to_proto(value)),
            })
            .collect(),
    }
}

/// Convert a relational `Value` to protobuf.
#[must_use]
pub fn value_to_proto(value: RelationalValue) -> proto::Value {
    let kind = match value {
        RelationalValue::Null => proto::value::Kind::Null(true),
        RelationalValue::Int(i) => proto::value::Kind::IntValue(i),
        RelationalValue::Float(f) => proto::value::Kind::FloatValue(f),
        RelationalValue::String(s) => proto::value::Kind::StringValue(s),
        RelationalValue::Bool(b) => proto::value::Kind::BoolValue(b),
    };
    proto::Value { kind: Some(kind) }
}

/// Convert a `NodeResult` to protobuf.
#[must_use]
pub fn node_to_proto(node: NodeResult) -> proto::Node {
    proto::Node {
        id: node.id,
        label: node.label,
        properties: node.properties,
    }
}

/// Convert an `EdgeResult` to protobuf.
#[must_use]
pub fn edge_to_proto(edge: EdgeResult) -> proto::Edge {
    proto::Edge {
        id: edge.id,
        from: edge.from,
        to: edge.to,
        label: edge.label,
    }
}

/// Convert a `SimilarResult` to protobuf.
#[must_use]
pub fn similar_to_proto(item: RouterSimilarResult) -> proto::SimilarItem {
    proto::SimilarItem {
        key: item.key,
        score: item.score,
    }
}

/// Convert a `PageRankResult` to protobuf.
#[must_use]
pub fn pagerank_item_to_proto(item: query_router::PageRankResult) -> proto::PageRankItem {
    proto::PageRankItem {
        node_id: item.node_id,
        score: item.score,
    }
}

/// Convert a `CentralityResult` to protobuf.
#[must_use]
pub fn centrality_item_to_proto(item: query_router::CentralityResult) -> proto::CentralityItem {
    proto::CentralityItem {
        node_id: item.node_id,
        score: item.score,
    }
}

/// Convert a `UnifiedResult` to protobuf.
#[must_use]
pub fn unified_to_proto(unified: UnifiedResult) -> proto::UnifiedQueryResult {
    proto::UnifiedQueryResult {
        description: unified.description,
        items: unified
            .items
            .into_iter()
            .map(|item| proto::UnifiedItem {
                entity_type: item.source.clone(),
                key: item.id.clone(),
                fields: item.data,
                score: item.score,
            })
            .collect(),
    }
}

/// Convert an `ArtifactInfoResult` to protobuf.
#[must_use]
pub fn artifact_info_to_proto(info: ArtifactInfoResult) -> proto::ArtifactInfo {
    proto::ArtifactInfo {
        id: info.id,
        filename: info.filename,
        content_type: info.content_type,
        size: info.size as u64,
        checksum: info.checksum,
        chunk_count: info.chunk_count as u64,
        created: info.created,
        modified: info.modified,
        created_by: info.created_by,
        tags: info.tags,
        linked_to: info.linked_to,
        custom: info.custom,
    }
}

/// Convert a `BlobStatsResult` to protobuf.
#[must_use]
pub fn blob_stats_to_proto(stats: BlobStatsResult) -> proto::BlobStatsResult {
    proto::BlobStatsResult {
        artifact_count: stats.artifact_count as u64,
        chunk_count: stats.chunk_count as u64,
        total_bytes: stats.total_bytes as u64,
        unique_bytes: stats.unique_bytes as u64,
        dedup_ratio: stats.dedup_ratio,
        orphaned_chunks: stats.orphaned_chunks as u64,
    }
}

/// Convert a `CheckpointInfo` to protobuf.
#[must_use]
pub fn checkpoint_to_proto(info: RouterCheckpointInfo) -> proto::CheckpointInfo {
    proto::CheckpointInfo {
        id: info.id,
        name: info.name,
        created_at: info.created_at,
        is_auto: info.is_auto,
    }
}

/// Convert a `ChainResult` to protobuf.
#[must_use]
pub fn chain_result_to_proto(result: ChainResult) -> proto::ChainQueryResult {
    let result_oneof = match result {
        ChainResult::TransactionBegun { tx_id } => {
            proto::chain_query_result::Result::TransactionBegun(proto::ChainTransactionBegun {
                tx_id,
            })
        },

        ChainResult::Committed { block_hash, height } => {
            proto::chain_query_result::Result::Committed(proto::ChainCommitted {
                block_hash,
                height,
            })
        },

        ChainResult::RolledBack { to_height } => {
            proto::chain_query_result::Result::RolledBack(proto::ChainRolledBack { to_height })
        },

        ChainResult::History(entries) => {
            proto::chain_query_result::Result::History(proto::ChainHistory {
                entries: entries.into_iter().map(chain_history_to_proto).collect(),
            })
        },

        ChainResult::Similar(items) => {
            proto::chain_query_result::Result::Similar(proto::ChainSimilar {
                items: items.into_iter().map(chain_similar_to_proto).collect(),
            })
        },

        ChainResult::Drift(drift) => {
            proto::chain_query_result::Result::Drift(chain_drift_to_proto(drift))
        },

        ChainResult::Height(h) => {
            proto::chain_query_result::Result::Height(proto::ChainHeight { height: h })
        },

        ChainResult::Tip { hash, height } => {
            proto::chain_query_result::Result::Tip(proto::ChainTip { hash, height })
        },

        ChainResult::Block(block) => {
            proto::chain_query_result::Result::Block(chain_block_to_proto(block))
        },

        ChainResult::Codebook(codebook) => {
            proto::chain_query_result::Result::Codebook(chain_codebook_to_proto(codebook))
        },

        ChainResult::Verified { ok, errors: _ } => {
            // Map to transition analysis with validity score
            proto::chain_query_result::Result::TransitionAnalysis(proto::ChainTransitionAnalysis {
                total_transitions: 1,
                valid_transitions: u64::from(ok),
                invalid_transitions: u64::from(!ok),
                avg_validity_score: if ok { 1.0 } else { 0.0 },
            })
        },

        ChainResult::TransitionAnalysis(analysis) => {
            proto::chain_query_result::Result::TransitionAnalysis(transition_analysis_to_proto(
                analysis,
            ))
        },
    };

    proto::ChainQueryResult {
        result: Some(result_oneof),
    }
}

/// Convert a `ChainTransitionAnalysis` to protobuf.
#[must_use]
pub fn transition_analysis_to_proto(
    analysis: ChainTransitionAnalysis,
) -> proto::ChainTransitionAnalysis {
    proto::ChainTransitionAnalysis {
        total_transitions: analysis.total_transitions as u64,
        valid_transitions: analysis.valid_transitions as u64,
        invalid_transitions: analysis.invalid_transitions as u64,
        avg_validity_score: analysis.avg_validity_score,
    }
}

/// Convert a `ChainHistoryEntry` to protobuf.
#[must_use]
pub fn chain_history_to_proto(entry: ChainHistoryEntry) -> proto::ChainHistoryEntry {
    proto::ChainHistoryEntry {
        height: entry.height,
        transaction_type: entry.transaction_type,
        data: entry.data,
    }
}

/// Convert a `ChainSimilarResult` to protobuf.
#[must_use]
pub fn chain_similar_to_proto(item: ChainSimilarResult) -> proto::ChainSimilarItem {
    proto::ChainSimilarItem {
        block_hash: item.block_hash,
        height: item.height,
        similarity: item.similarity,
    }
}

/// Convert a `ChainDriftResult` to protobuf.
#[must_use]
pub fn chain_drift_to_proto(drift: ChainDriftResult) -> proto::ChainDrift {
    proto::ChainDrift {
        from_height: drift.from_height,
        to_height: drift.to_height,
        total_drift: drift.total_drift,
        avg_drift_per_block: drift.avg_drift_per_block,
        max_drift: drift.max_drift,
    }
}

/// Convert a `ChainBlockInfo` to protobuf.
#[must_use]
pub fn chain_block_to_proto(block: ChainBlockInfo) -> proto::ChainBlockInfo {
    proto::ChainBlockInfo {
        height: block.height,
        hash: block.hash,
        prev_hash: block.prev_hash,
        timestamp: block.timestamp,
        transaction_count: block.transaction_count as u64,
        proposer: block.proposer,
    }
}

/// Convert a `ChainCodebookInfo` to protobuf.
#[must_use]
pub fn chain_codebook_to_proto(codebook: ChainCodebookInfo) -> proto::ChainCodebookInfo {
    proto::ChainCodebookInfo {
        scope: codebook.scope,
        entry_count: codebook.entry_count as u64,
        dimension: codebook.dimension as u64,
        domain: codebook.domain,
    }
}

/// Create an error response from a `RouterError`.
#[must_use]
#[allow(dead_code)]
pub fn error_to_proto(err: &query_router::RouterError) -> proto::QueryResponse {
    proto::QueryResponse {
        result: None,
        error: Some(proto::ErrorInfo {
            code: error_code_from_router_error(err).into(),
            message: err.to_string(),
            details: None,
        }),
    }
}

/// Map a `RouterError` to an error code.
fn error_code_from_router_error(err: &query_router::RouterError) -> proto::ErrorCode {
    use query_router::RouterError;

    match err {
        RouterError::ParseError(_) | RouterError::UnknownCommand(_) => {
            proto::ErrorCode::InvalidQuery
        },
        RouterError::RelationalError(_)
        | RouterError::GraphError(_)
        | RouterError::VectorError(_) => proto::ErrorCode::NotFound,
        RouterError::VaultError(_) => proto::ErrorCode::PermissionDenied,
        RouterError::AuthenticationRequired => proto::ErrorCode::Unauthenticated,
        RouterError::InvalidArgument(_) | RouterError::MissingArgument(_) => {
            proto::ErrorCode::InvalidArgument
        },
        _ => proto::ErrorCode::Internal,
    }
}

/// Convert protobuf metadata to `PutOptions`.
#[must_use]
pub fn upload_metadata_to_put_options(
    metadata: &proto::BlobUploadMetadata,
) -> tensor_blob::PutOptions {
    let mut options = tensor_blob::PutOptions::default();

    if let Some(ref ct) = metadata.content_type {
        options = options.with_content_type(ct);
    }

    if let Some(ref creator) = metadata.created_by {
        options = options.with_created_by(creator);
    }

    for tag in &metadata.tags {
        options = options.with_tag(tag);
    }

    for link in &metadata.linked_to {
        options = options.with_link(link);
    }

    for (key, value) in &metadata.custom {
        options = options.with_meta(key, value);
    }

    options
}

/// Convert `ArtifactMetadata` from blob store to protobuf.
#[must_use]
pub fn blob_metadata_to_proto(metadata: &tensor_blob::ArtifactMetadata) -> proto::ArtifactInfo {
    proto::ArtifactInfo {
        id: metadata.id.clone(),
        filename: metadata.filename.clone(),
        content_type: metadata.content_type.clone(),
        size: metadata.size as u64,
        checksum: metadata.checksum.clone(),
        chunk_count: metadata.chunk_count as u64,
        created: metadata.created,
        modified: metadata.modified,
        created_by: metadata.created_by.clone(),
        tags: metadata.tags.clone(),
        linked_to: metadata.linked_to.clone(),
        custom: metadata.custom.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_empty_result_conversion() {
        let result = QueryResult::Empty;
        let proto = query_result_to_proto(result);
        assert!(matches!(
            proto.result,
            Some(proto::query_response::Result::Empty(_))
        ));
    }

    #[test]
    fn test_value_result_conversion() {
        let result = QueryResult::Value("test value".to_string());
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Value(v)) => {
                assert_eq!(v.value, "test value");
            },
            _ => panic!("Expected Value result"),
        }
    }

    #[test]
    fn test_count_result_conversion() {
        let result = QueryResult::Count(42);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Count(c)) => {
                assert_eq!(c.count, 42);
            },
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_ids_result_conversion() {
        let result = QueryResult::Ids(vec![1, 2, 3]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Ids(ids)) => {
                assert_eq!(ids.ids, vec![1, 2, 3]);
            },
            _ => panic!("Expected Ids result"),
        }
    }

    #[test]
    fn test_row_conversion() {
        let row = RelationalRow {
            id: 1,
            values: vec![
                (
                    "name".to_string(),
                    RelationalValue::String("Alice".to_string()),
                ),
                ("age".to_string(), RelationalValue::Int(30)),
            ],
        };
        let proto = row_to_proto(row);
        assert_eq!(proto.id, 1);
        assert_eq!(proto.values.len(), 2);
    }

    #[test]
    fn test_value_conversion() {
        assert!(matches!(
            value_to_proto(RelationalValue::Null).kind,
            Some(proto::value::Kind::Null(true))
        ));

        assert!(matches!(
            value_to_proto(RelationalValue::Int(42)).kind,
            Some(proto::value::Kind::IntValue(42))
        ));

        assert!(matches!(
            value_to_proto(RelationalValue::Bool(true)).kind,
            Some(proto::value::Kind::BoolValue(true))
        ));
    }

    #[test]
    fn test_node_conversion() {
        let mut props = HashMap::new();
        props.insert("key".to_string(), "value".to_string());

        let node = NodeResult {
            id: 1,
            label: "Person".to_string(),
            properties: props,
        };

        let proto = node_to_proto(node);
        assert_eq!(proto.id, 1);
        assert_eq!(proto.label, "Person");
        assert_eq!(proto.properties.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_edge_conversion() {
        let edge = EdgeResult {
            id: 1,
            from: 2,
            to: 3,
            label: "KNOWS".to_string(),
        };

        let proto = edge_to_proto(edge);
        assert_eq!(proto.id, 1);
        assert_eq!(proto.from, 2);
        assert_eq!(proto.to, 3);
        assert_eq!(proto.label, "KNOWS");
    }

    #[test]
    fn test_similar_conversion() {
        let item = RouterSimilarResult {
            key: "item1".to_string(),
            score: 0.95,
        };

        let proto = similar_to_proto(item);
        assert_eq!(proto.key, "item1");
        assert!((proto.score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_path_result_conversion() {
        let result = QueryResult::Path(vec![1, 2, 3, 4]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Path(p)) => {
                assert_eq!(p.node_ids, vec![1, 2, 3, 4]);
            },
            _ => panic!("Expected Path result"),
        }
    }

    #[test]
    fn test_table_list_conversion() {
        let result = QueryResult::TableList(vec!["users".to_string(), "orders".to_string()]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::TableList(t)) => {
                assert_eq!(t.tables, vec!["users", "orders"]);
            },
            _ => panic!("Expected TableList result"),
        }
    }

    #[test]
    fn test_blob_result_conversion() {
        let result = QueryResult::Blob(vec![1, 2, 3, 4]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Blob(b)) => {
                assert_eq!(b.data, vec![1, 2, 3, 4]);
            },
            _ => panic!("Expected Blob result"),
        }
    }

    #[test]
    fn test_chain_transaction_begun_conversion() {
        let result = QueryResult::Chain(ChainResult::TransactionBegun {
            tx_id: "tx123".to_string(),
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::TransactionBegun(t)) => {
                    assert_eq!(t.tx_id, "tx123");
                },
                _ => panic!("Expected TransactionBegun"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_committed_conversion() {
        let result = QueryResult::Chain(ChainResult::Committed {
            block_hash: "hash123".to_string(),
            height: 42,
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Committed(t)) => {
                    assert_eq!(t.block_hash, "hash123");
                    assert_eq!(t.height, 42);
                },
                _ => panic!("Expected Committed"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_upload_metadata_conversion() {
        let metadata = proto::BlobUploadMetadata {
            filename: "test.txt".to_string(),
            content_type: Some("text/plain".to_string()),
            created_by: Some("user:alice".to_string()),
            tags: vec!["important".to_string()],
            linked_to: vec!["entity:123".to_string()],
            custom: HashMap::new(),
        };

        let options = upload_metadata_to_put_options(&metadata);
        // Options are validated through usage in the blob service
        drop(options);
    }

    #[test]
    fn test_upload_metadata_minimal() {
        let metadata = proto::BlobUploadMetadata {
            filename: "test.txt".to_string(),
            content_type: None,
            created_by: None,
            tags: vec![],
            linked_to: vec![],
            custom: HashMap::new(),
        };

        let options = upload_metadata_to_put_options(&metadata);
        drop(options);
    }

    #[test]
    fn test_upload_metadata_with_custom() {
        let mut custom = HashMap::new();
        custom.insert("priority".to_string(), "high".to_string());
        custom.insert("source".to_string(), "upload".to_string());

        let metadata = proto::BlobUploadMetadata {
            filename: "test.txt".to_string(),
            content_type: None,
            created_by: None,
            tags: vec![],
            linked_to: vec![],
            custom,
        };

        let options = upload_metadata_to_put_options(&metadata);
        drop(options);
    }

    #[test]
    fn test_nodes_result_conversion() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), "Alice".to_string());

        let result = QueryResult::Nodes(vec![
            NodeResult {
                id: 1,
                label: "Person".to_string(),
                properties: props.clone(),
            },
            NodeResult {
                id: 2,
                label: "Company".to_string(),
                properties: HashMap::new(),
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Nodes(n)) => {
                assert_eq!(n.nodes.len(), 2);
                assert_eq!(n.nodes[0].id, 1);
                assert_eq!(n.nodes[0].label, "Person");
                assert_eq!(n.nodes[1].id, 2);
            },
            _ => panic!("Expected Nodes result"),
        }
    }

    #[test]
    fn test_edges_result_conversion() {
        let result = QueryResult::Edges(vec![
            EdgeResult {
                id: 1,
                from: 10,
                to: 20,
                label: "KNOWS".to_string(),
            },
            EdgeResult {
                id: 2,
                from: 20,
                to: 30,
                label: "WORKS_AT".to_string(),
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Edges(e)) => {
                assert_eq!(e.edges.len(), 2);
                assert_eq!(e.edges[0].from, 10);
                assert_eq!(e.edges[0].to, 20);
            },
            _ => panic!("Expected Edges result"),
        }
    }

    #[test]
    fn test_similar_result_conversion() {
        let result = QueryResult::Similar(vec![
            RouterSimilarResult {
                key: "item1".to_string(),
                score: 0.95,
            },
            RouterSimilarResult {
                key: "item2".to_string(),
                score: 0.85,
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Similar(s)) => {
                assert_eq!(s.items.len(), 2);
                assert_eq!(s.items[0].key, "item1");
            },
            _ => panic!("Expected Similar result"),
        }
    }

    #[test]
    fn test_unified_result_conversion() {
        use tensor_unified::UnifiedItem;

        let result = QueryResult::Unified(UnifiedResult {
            description: "Combined results".to_string(),
            items: vec![
                UnifiedItem {
                    source: "relational".to_string(),
                    id: "user:1".to_string(),
                    data: HashMap::new(),
                    embedding: None,
                    score: Some(0.9),
                },
                UnifiedItem {
                    source: "graph".to_string(),
                    id: "node:2".to_string(),
                    data: HashMap::new(),
                    embedding: None,
                    score: None,
                },
            ],
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Unified(u)) => {
                assert_eq!(u.description, "Combined results");
                assert_eq!(u.items.len(), 2);
                assert_eq!(u.items[0].entity_type, "relational");
            },
            _ => panic!("Expected Unified result"),
        }
    }

    #[test]
    fn test_artifact_info_result_conversion() {
        let result = QueryResult::ArtifactInfo(ArtifactInfoResult {
            id: "art-123".to_string(),
            filename: "file.txt".to_string(),
            content_type: "text/plain".to_string(),
            size: 1024,
            checksum: "abc123".to_string(),
            chunk_count: 2,
            created: 1000,
            modified: 2000,
            created_by: "alice".to_string(),
            tags: vec!["tag1".to_string()],
            linked_to: vec!["entity:1".to_string()],
            custom: HashMap::new(),
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::ArtifactInfo(a)) => {
                assert_eq!(a.id, "art-123");
                assert_eq!(a.filename, "file.txt");
                assert_eq!(a.size, 1024);
            },
            _ => panic!("Expected ArtifactInfo result"),
        }
    }

    #[test]
    fn test_artifact_list_result_conversion() {
        let result = QueryResult::ArtifactList(vec![
            "art-1".to_string(),
            "art-2".to_string(),
            "art-3".to_string(),
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::ArtifactList(a)) => {
                assert_eq!(a.artifact_ids.len(), 3);
            },
            _ => panic!("Expected ArtifactList result"),
        }
    }

    #[test]
    fn test_blob_stats_result_conversion() {
        let result = QueryResult::BlobStats(BlobStatsResult {
            artifact_count: 100,
            chunk_count: 500,
            total_bytes: 1_000_000,
            unique_bytes: 800_000,
            dedup_ratio: 0.8,
            orphaned_chunks: 5,
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::BlobStats(b)) => {
                assert_eq!(b.artifact_count, 100);
                assert_eq!(b.chunk_count, 500);
                assert_eq!(b.total_bytes, 1_000_000);
                assert!((b.dedup_ratio - 0.8).abs() < 0.001);
            },
            _ => panic!("Expected BlobStats result"),
        }
    }

    #[test]
    fn test_checkpoint_list_result_conversion() {
        let result = QueryResult::CheckpointList(vec![
            RouterCheckpointInfo {
                id: "cp-1".to_string(),
                name: "checkpoint1".to_string(),
                created_at: 1000,
                is_auto: false,
            },
            RouterCheckpointInfo {
                id: "cp-2".to_string(),
                name: "checkpoint2".to_string(),
                created_at: 2000,
                is_auto: true,
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::CheckpointList(c)) => {
                assert_eq!(c.checkpoints.len(), 2);
                assert_eq!(c.checkpoints[0].id, "cp-1");
                assert!(!c.checkpoints[0].is_auto);
                assert!(c.checkpoints[1].is_auto);
            },
            _ => panic!("Expected CheckpointList result"),
        }
    }

    #[test]
    fn test_chain_rolled_back_conversion() {
        let result = QueryResult::Chain(ChainResult::RolledBack { to_height: 100 });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::RolledBack(r)) => {
                    assert_eq!(r.to_height, 100);
                },
                _ => panic!("Expected RolledBack"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_history_conversion() {
        let result = QueryResult::Chain(ChainResult::History(vec![
            ChainHistoryEntry {
                height: 1,
                transaction_type: "PUT".to_string(),
                data: Some(b"key1".to_vec()),
            },
            ChainHistoryEntry {
                height: 2,
                transaction_type: "DELETE".to_string(),
                data: Some(b"key2".to_vec()),
            },
        ]));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::History(h)) => {
                    assert_eq!(h.entries.len(), 2);
                    assert_eq!(h.entries[0].height, 1);
                },
                _ => panic!("Expected History"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_similar_conversion() {
        let result = QueryResult::Chain(ChainResult::Similar(vec![
            ChainSimilarResult {
                block_hash: "hash1".to_string(),
                height: 10,
                similarity: 0.95,
            },
            ChainSimilarResult {
                block_hash: "hash2".to_string(),
                height: 20,
                similarity: 0.85,
            },
        ]));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Similar(s)) => {
                    assert_eq!(s.items.len(), 2);
                    assert_eq!(s.items[0].block_hash, "hash1");
                },
                _ => panic!("Expected Similar"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_drift_conversion() {
        let result = QueryResult::Chain(ChainResult::Drift(ChainDriftResult {
            from_height: 1,
            to_height: 100,
            total_drift: 0.5,
            avg_drift_per_block: 0.005,
            max_drift: 0.1,
        }));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Drift(d)) => {
                    assert_eq!(d.from_height, 1);
                    assert_eq!(d.to_height, 100);
                    assert!((d.total_drift - 0.5).abs() < 0.001);
                },
                _ => panic!("Expected Drift"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_height_conversion() {
        let result = QueryResult::Chain(ChainResult::Height(42));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Height(h)) => {
                    assert_eq!(h.height, 42);
                },
                _ => panic!("Expected Height"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_tip_conversion() {
        let result = QueryResult::Chain(ChainResult::Tip {
            hash: "tiphash".to_string(),
            height: 999,
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Tip(t)) => {
                    assert_eq!(t.hash, "tiphash");
                    assert_eq!(t.height, 999);
                },
                _ => panic!("Expected Tip"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_block_conversion() {
        let result = QueryResult::Chain(ChainResult::Block(ChainBlockInfo {
            height: 50,
            hash: "blockhash".to_string(),
            prev_hash: "prevhash".to_string(),
            timestamp: 1234567890,
            transaction_count: 10,
            proposer: "node1".to_string(),
        }));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Block(b)) => {
                    assert_eq!(b.height, 50);
                    assert_eq!(b.hash, "blockhash");
                    assert_eq!(b.transaction_count, 10);
                },
                _ => panic!("Expected Block"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_codebook_conversion() {
        let result = QueryResult::Chain(ChainResult::Codebook(ChainCodebookInfo {
            scope: "global".to_string(),
            entry_count: 256,
            dimension: 64,
            domain: Some("semantic".to_string()),
        }));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::Codebook(cb)) => {
                    assert_eq!(cb.scope, "global");
                    assert_eq!(cb.entry_count, 256);
                },
                _ => panic!("Expected Codebook"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_verified_ok_conversion() {
        let result = QueryResult::Chain(ChainResult::Verified {
            ok: true,
            errors: vec![],
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::TransitionAnalysis(t)) => {
                    assert_eq!(t.valid_transitions, 1);
                    assert_eq!(t.invalid_transitions, 0);
                    assert!((t.avg_validity_score - 1.0).abs() < 0.001);
                },
                _ => panic!("Expected TransitionAnalysis"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_verified_failed_conversion() {
        let result = QueryResult::Chain(ChainResult::Verified {
            ok: false,
            errors: vec!["invalid signature".to_string()],
        });
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::TransitionAnalysis(t)) => {
                    assert_eq!(t.valid_transitions, 0);
                    assert_eq!(t.invalid_transitions, 1);
                    assert!((t.avg_validity_score - 0.0).abs() < 0.001);
                },
                _ => panic!("Expected TransitionAnalysis"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_chain_transition_analysis_conversion() {
        let result = QueryResult::Chain(ChainResult::TransitionAnalysis(ChainTransitionAnalysis {
            total_transitions: 100,
            valid_transitions: 95,
            invalid_transitions: 5,
            avg_validity_score: 0.95,
        }));
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Chain(c)) => match c.result {
                Some(proto::chain_query_result::Result::TransitionAnalysis(t)) => {
                    assert_eq!(t.total_transitions, 100);
                    assert_eq!(t.valid_transitions, 95);
                },
                _ => panic!("Expected TransitionAnalysis"),
            },
            _ => panic!("Expected Chain result"),
        }
    }

    #[test]
    fn test_error_to_proto() {
        use query_router::RouterError;

        let err = RouterError::ParseError("syntax error".to_string());
        let proto = error_to_proto(&err);
        assert!(proto.result.is_none());
        assert!(proto.error.is_some());
        let error_info = proto.error.unwrap();
        assert_eq!(error_info.code, i32::from(proto::ErrorCode::InvalidQuery));
    }

    #[test]
    fn test_error_code_from_router_error_variants() {
        use query_router::RouterError;

        assert_eq!(
            error_code_from_router_error(&RouterError::ParseError("test".to_string())),
            proto::ErrorCode::InvalidQuery
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::UnknownCommand("test".to_string())),
            proto::ErrorCode::InvalidQuery
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::RelationalError("test".to_string())),
            proto::ErrorCode::NotFound
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::GraphError("test".to_string())),
            proto::ErrorCode::NotFound
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::VectorError("test".to_string())),
            proto::ErrorCode::NotFound
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::VaultError("test".to_string())),
            proto::ErrorCode::PermissionDenied
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::AuthenticationRequired),
            proto::ErrorCode::Unauthenticated
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::InvalidArgument("test".to_string())),
            proto::ErrorCode::InvalidArgument
        );
        assert_eq!(
            error_code_from_router_error(&RouterError::MissingArgument("test".to_string())),
            proto::ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn test_blob_metadata_to_proto() {
        let metadata = tensor_blob::ArtifactMetadata {
            id: "blob-123".to_string(),
            filename: "document.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            size: 2048,
            checksum: "sha256-abc".to_string(),
            chunk_count: 4,
            chunk_size: 65536,
            created: 1000,
            modified: 2000,
            created_by: "alice".to_string(),
            tags: vec!["pdf".to_string(), "document".to_string()],
            linked_to: vec!["entity:doc1".to_string()],
            custom: HashMap::new(),
            has_embedding: false,
            embedding_model: None,
        };

        let proto = blob_metadata_to_proto(&metadata);
        assert_eq!(proto.id, "blob-123");
        assert_eq!(proto.filename, "document.pdf");
        assert_eq!(proto.content_type, "application/pdf");
        assert_eq!(proto.size, 2048);
        assert_eq!(proto.chunk_count, 4);
        assert_eq!(proto.tags.len(), 2);
        assert_eq!(proto.linked_to.len(), 1);
    }

    #[test]
    fn test_value_conversion_float() {
        let value = RelationalValue::Float(3.14159);
        let proto = value_to_proto(value);
        match proto.kind {
            Some(proto::value::Kind::FloatValue(f)) => {
                assert!((f - 3.14159).abs() < 0.0001);
            },
            _ => panic!("Expected FloatValue"),
        }
    }

    #[test]
    fn test_value_conversion_string() {
        let value = RelationalValue::String("hello world".to_string());
        let proto = value_to_proto(value);
        match proto.kind {
            Some(proto::value::Kind::StringValue(s)) => {
                assert_eq!(s, "hello world");
            },
            _ => panic!("Expected StringValue"),
        }
    }

    #[test]
    fn test_rows_result_conversion() {
        let result = QueryResult::Rows(vec![
            RelationalRow {
                id: 1,
                values: vec![
                    (
                        "name".to_string(),
                        RelationalValue::String("Alice".to_string()),
                    ),
                    ("age".to_string(), RelationalValue::Int(30)),
                ],
            },
            RelationalRow {
                id: 2,
                values: vec![
                    (
                        "name".to_string(),
                        RelationalValue::String("Bob".to_string()),
                    ),
                    ("age".to_string(), RelationalValue::Int(25)),
                ],
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Rows(r)) => {
                assert_eq!(r.rows.len(), 2);
                assert_eq!(r.rows[0].id, 1);
                assert_eq!(r.rows[0].values.len(), 2);
            },
            _ => panic!("Expected Rows result"),
        }
    }

    #[test]
    fn test_pagerank_result_conversion() {
        let result = QueryResult::PageRank(vec![
            query_router::PageRankResult {
                node_id: 1,
                score: 0.25,
            },
            query_router::PageRankResult {
                node_id: 2,
                score: 0.15,
            },
            query_router::PageRankResult {
                node_id: 3,
                score: 0.10,
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::PageRank(pr)) => {
                assert_eq!(pr.items.len(), 3);
                assert_eq!(pr.items[0].node_id, 1);
                assert!((pr.items[0].score - 0.25).abs() < 0.001);
                assert_eq!(pr.items[1].node_id, 2);
                assert!((pr.items[1].score - 0.15).abs() < 0.001);
            },
            _ => panic!("Expected PageRank result"),
        }
    }

    #[test]
    fn test_centrality_result_conversion() {
        let result = QueryResult::Centrality(vec![
            query_router::CentralityResult {
                node_id: 10,
                score: 0.85,
            },
            query_router::CentralityResult {
                node_id: 20,
                score: 0.65,
            },
        ]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Centrality(c)) => {
                assert_eq!(c.items.len(), 2);
                assert_eq!(c.items[0].node_id, 10);
                assert!((c.items[0].score - 0.85).abs() < 0.001);
            },
            _ => panic!("Expected Centrality result"),
        }
    }

    #[test]
    fn test_pagerank_empty_result() {
        let result = QueryResult::PageRank(vec![]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::PageRank(pr)) => {
                assert!(pr.items.is_empty());
            },
            _ => panic!("Expected PageRank result"),
        }
    }

    #[test]
    fn test_centrality_empty_result() {
        let result = QueryResult::Centrality(vec![]);
        let proto = query_result_to_proto(result);
        match proto.result {
            Some(proto::query_response::Result::Centrality(c)) => {
                assert!(c.items.is_empty());
            },
            _ => panic!("Expected Centrality result"),
        }
    }

    #[test]
    fn test_pagerank_item_to_proto() {
        let item = query_router::PageRankResult {
            node_id: 42,
            score: 0.123_456,
        };
        let proto = pagerank_item_to_proto(item);
        assert_eq!(proto.node_id, 42);
        assert!((proto.score - 0.123_456).abs() < 0.000_001);
    }

    #[test]
    fn test_centrality_item_to_proto() {
        let item = query_router::CentralityResult {
            node_id: 99,
            score: 0.999,
        };
        let proto = centrality_item_to_proto(item);
        assert_eq!(proto.node_id, 99);
        assert!((proto.score - 0.999).abs() < 0.001);
    }
}
