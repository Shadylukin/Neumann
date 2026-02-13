// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Native Python bindings for Neumann database via `PyO3`.
#![allow(clippy::useless_conversion)]

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use query_router::{
    ArtifactInfoResult, BlobStatsResult, ChainHistoryEntry, ChainResult, ChainSimilarResult,
    CheckpointInfo, QueryResult, QueryRouter as RustQueryRouter, UnifiedResult,
};
use tensor_compress::CompressionConfig;
use tensor_store::{TensorStore, TensorValue};
use tensor_unified::UnifiedItem;

/// Python-accessible `QueryRouter` wrapper.
#[pyclass]
struct QueryRouter {
    inner: Arc<RwLock<RustQueryRouter>>,
}

#[pymethods]
impl QueryRouter {
    /// Create a new in-memory `QueryRouter`.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RustQueryRouter::new())),
        }
    }

    /// Create a `QueryRouter` with persistent storage at the given path.
    ///
    /// If the path exists, loads the snapshot (auto-detects compressed vs legacy format).
    /// If the path does not exist, creates a fresh in-memory router.
    #[staticmethod]
    fn with_path(path: &str) -> PyResult<Self> {
        let p = Path::new(path);

        let router = if p.exists() {
            let store = TensorStore::load_snapshot_compressed(p)
                .or_else(|_| TensorStore::load_snapshot(p))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {e}")))?;
            RustQueryRouter::with_shared_store(store)
        } else {
            RustQueryRouter::new()
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(router)),
        })
    }

    /// Save state to a compressed snapshot file.
    fn save(&self, path: &str) -> PyResult<()> {
        let router = self.inner.read();
        let store = router.vector().store().clone();

        let dim = Self::detect_embedding_dimension(&store);
        let config = CompressionConfig::balanced(dim);

        store
            .save_snapshot_compressed(path, config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save: {e}")))?;
        Ok(())
    }

    /// Save state to an uncompressed snapshot file.
    ///
    /// Faster I/O but larger file size compared to `save()`.
    fn save_uncompressed(&self, path: &str) -> PyResult<()> {
        let router = self.inner.read();
        let store = router.vector().store().clone();

        store
            .save_snapshot(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save: {e}")))?;
        Ok(())
    }

    /// Execute a query and return the result as a Python dict.
    #[pyo3(signature = (query, identity=None))]
    fn execute(&self, query: &str, identity: Option<&str>) -> PyResult<PyObject> {
        let mut router = self.inner.write();

        if let Some(id) = identity {
            router.set_identity(id);
        }

        let result = router
            .execute(query)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::with_gil(|py| convert_result_to_python(py, result))
    }
}

const DEFAULT_EMBEDDING_DIM: usize = 768;

impl QueryRouter {
    fn detect_embedding_dimension(store: &TensorStore) -> usize {
        for key in store.scan("") {
            if let Ok(data) = store.get(&key) {
                for (_, value) in data.iter() {
                    if let TensorValue::Vector(v) = value {
                        if !v.is_empty() {
                            return v.len();
                        }
                    }
                }
            }
        }
        DEFAULT_EMBEDDING_DIM
    }
}

/// Convert a `QueryResult` to a Python object.
#[allow(clippy::too_many_lines)]
fn convert_result_to_python(py: Python<'_>, result: QueryResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    match result {
        QueryResult::Empty => {
            dict.set_item("type", "empty")?;
        },
        QueryResult::Value(s) => {
            dict.set_item("type", "value")?;
            dict.set_item("data", s)?;
        },
        QueryResult::Count(n) => {
            dict.set_item("type", "count")?;
            dict.set_item("data", n)?;
        },
        QueryResult::Rows(rows) => {
            dict.set_item("type", "rows")?;
            let py_rows = PyList::empty(py);
            for row in rows {
                let row_dict = PyDict::new(py);
                for (k, v) in &row.values {
                    row_dict.set_item(k, convert_value_to_python(py, v))?;
                }
                py_rows.append(row_dict)?;
            }
            dict.set_item("data", py_rows)?;
        },
        QueryResult::Nodes(nodes) => {
            dict.set_item("type", "nodes")?;
            let py_nodes = PyList::empty(py);
            for node in nodes {
                let node_dict = PyDict::new(py);
                node_dict.set_item("id", node.id)?;
                node_dict.set_item("label", &node.label)?;
                let props = PyDict::new(py);
                for (k, v) in &node.properties {
                    props.set_item(k, v)?;
                }
                node_dict.set_item("properties", props)?;
                py_nodes.append(node_dict)?;
            }
            dict.set_item("data", py_nodes)?;
        },
        QueryResult::Edges(edges) => {
            dict.set_item("type", "edges")?;
            let py_edges = PyList::empty(py);
            for edge in edges {
                let edge_dict = PyDict::new(py);
                edge_dict.set_item("id", edge.id)?;
                edge_dict.set_item("type", &edge.label)?;
                edge_dict.set_item("source", edge.from)?;
                edge_dict.set_item("target", edge.to)?;
                py_edges.append(edge_dict)?;
            }
            dict.set_item("data", py_edges)?;
        },
        QueryResult::Similar(items) => {
            dict.set_item("type", "similar")?;
            let py_items = PyList::empty(py);
            for item in items {
                let item_dict = PyDict::new(py);
                item_dict.set_item("key", &item.key)?;
                item_dict.set_item("score", item.score)?;
                py_items.append(item_dict)?;
            }
            dict.set_item("data", py_items)?;
        },
        QueryResult::Ids(ids) => {
            dict.set_item("type", "ids")?;
            let py_ids = PyList::new(py, &ids)?;
            dict.set_item("data", py_ids)?;
        },
        QueryResult::TableList(tables) => {
            dict.set_item("type", "table_list")?;
            let py_tables = PyList::new(py, &tables)?;
            dict.set_item("data", py_tables)?;
        },
        QueryResult::Blob(data) => {
            dict.set_item("type", "blob")?;
            dict.set_item("data", PyBytes::new(py, &data))?;
        },
        QueryResult::Path(node_ids) => {
            dict.set_item("type", "path")?;
            let py_ids = PyList::new(py, &node_ids)?;
            dict.set_item("data", py_ids)?;
        },
        QueryResult::ArtifactList(ids) => {
            dict.set_item("type", "artifact_list")?;
            let py_ids = PyList::new(py, &ids)?;
            dict.set_item("data", py_ids)?;
        },
        QueryResult::ArtifactInfo(info) => {
            dict.set_item("type", "artifact_info")?;
            dict.set_item("data", convert_artifact_info_to_python(py, &info)?)?;
        },
        QueryResult::BlobStats(stats) => {
            dict.set_item("type", "blob_stats")?;
            dict.set_item("data", convert_blob_stats_to_python(py, &stats)?)?;
        },
        QueryResult::CheckpointList(checkpoints) => {
            dict.set_item("type", "checkpoint_list")?;
            let py_list = PyList::empty(py);
            for cp in checkpoints {
                py_list.append(convert_checkpoint_to_python(py, &cp)?)?;
            }
            dict.set_item("data", py_list)?;
        },
        QueryResult::Unified(unified) => {
            dict.set_item("type", "unified")?;
            dict.set_item("data", convert_unified_to_python(py, &unified)?)?;
        },
        QueryResult::Chain(chain_result) => {
            dict.set_item("type", "chain")?;
            dict.set_item("data", convert_chain_result_to_python(py, &chain_result)?)?;
        },
        QueryResult::Constraints(constraints) => {
            dict.set_item("type", "constraints")?;
            let py_list = PyList::empty(py);
            for c in constraints {
                let constraint_dict = PyDict::new(py);
                constraint_dict.set_item("name", &c.name)?;
                constraint_dict.set_item("target", &c.target)?;
                constraint_dict.set_item("property", &c.property)?;
                constraint_dict.set_item("constraint_type", &c.constraint_type)?;
                py_list.append(constraint_dict)?;
            }
            dict.set_item("data", py_list)?;
        },
        QueryResult::GraphIndexes(indexes) => {
            dict.set_item("type", "graph_indexes")?;
            dict.set_item("data", PyList::new(py, &indexes)?)?;
        },
        QueryResult::PageRank(result) => {
            dict.set_item("type", "pagerank")?;
            let py_list = PyList::empty(py);
            for item in &result.items {
                let item_dict = PyDict::new(py);
                item_dict.set_item("node_id", item.node_id)?;
                item_dict.set_item("score", item.score)?;
                py_list.append(item_dict)?;
            }
            dict.set_item("data", py_list)?;
            dict.set_item("iterations", result.iterations)?;
            dict.set_item("convergence", result.convergence)?;
            dict.set_item("converged", result.converged)?;
        },
        QueryResult::Centrality(result) => {
            dict.set_item("type", "centrality")?;
            let py_list = PyList::empty(py);
            for item in &result.items {
                let item_dict = PyDict::new(py);
                item_dict.set_item("node_id", item.node_id)?;
                item_dict.set_item("score", item.score)?;
                py_list.append(item_dict)?;
            }
            dict.set_item("data", py_list)?;
        },
        QueryResult::Communities(result) => {
            dict.set_item("type", "communities")?;
            let py_list = PyList::empty(py);
            for item in &result.items {
                let item_dict = PyDict::new(py);
                item_dict.set_item("node_id", item.node_id)?;
                item_dict.set_item("community_id", item.community_id)?;
                py_list.append(item_dict)?;
            }
            dict.set_item("data", py_list)?;
            dict.set_item("community_count", result.community_count)?;
        },
        QueryResult::Aggregate(result) => {
            dict.set_item("type", "aggregate")?;
            use query_router::AggregateResultValue;
            match result {
                AggregateResultValue::Count(v) => dict.set_item("data", v)?,
                AggregateResultValue::Sum(v) => dict.set_item("data", v)?,
                AggregateResultValue::Avg(v) => dict.set_item("data", v)?,
                AggregateResultValue::Min(v) => dict.set_item("data", v)?,
                AggregateResultValue::Max(v) => dict.set_item("data", v)?,
            }
        },
        QueryResult::BatchResult(result) => {
            dict.set_item("type", "batch")?;
            let result_dict = PyDict::new(py);
            result_dict.set_item("operation", &result.operation)?;
            result_dict.set_item("affected_count", result.affected_count)?;
            if let Some(ref ids) = result.created_ids {
                result_dict.set_item("created_ids", PyList::new(py, ids)?)?;
            }
            dict.set_item("data", result_dict)?;
        },
        QueryResult::PatternMatch(result) => {
            dict.set_item("type", "pattern_match")?;
            let py_list = PyList::empty(py);
            for matched in &result.matches {
                let bindings_dict = PyDict::new(py);
                for name in matched.bindings.keys() {
                    bindings_dict.set_item(name, name)?;
                }
                py_list.append(bindings_dict)?;
            }
            dict.set_item("data", py_list)?;
        },
    }

    Ok(dict.into())
}

/// Convert a `relational_engine::Value` to a Python object.
fn convert_value_to_python(py: Python<'_>, value: &relational_engine::Value) -> PyObject {
    use pyo3::IntoPyObject;
    use relational_engine::Value;

    match value {
        Value::Null => py.None(),
        Value::Int(i) => i
            .into_pyobject(py)
            .map_or_else(|_| py.None(), |v| v.into_any().unbind()),
        Value::Float(f) => f
            .into_pyobject(py)
            .map_or_else(|_| py.None(), |v| v.into_any().unbind()),
        Value::String(s) => s
            .into_pyobject(py)
            .map_or_else(|_| py.None(), |v| v.into_any().unbind()),
        Value::Bool(b) => b
            .into_pyobject(py)
            .map_or_else(|_| py.None(), |v| v.to_owned().into_any().unbind()),
        Value::Bytes(b) => b
            .as_slice()
            .into_pyobject(py)
            .map_or_else(|_| py.None(), |v| v.into_any().unbind()),
        Value::Json(j) => j
            .to_string()
            .into_pyobject(py)
            .map_or_else(|_| py.None(), |v| v.into_any().unbind()),
        _ => py.None(),
    }
}

fn convert_artifact_info_to_python<'py>(
    py: Python<'py>,
    info: &ArtifactInfoResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &info.id)?;
    dict.set_item("filename", &info.filename)?;
    dict.set_item("content_type", &info.content_type)?;
    dict.set_item("size", info.size)?;
    dict.set_item("checksum", &info.checksum)?;
    dict.set_item("chunk_count", info.chunk_count)?;
    dict.set_item("created", info.created)?;
    dict.set_item("modified", info.modified)?;
    dict.set_item("created_by", &info.created_by)?;
    dict.set_item("tags", PyList::new(py, &info.tags)?)?;
    dict.set_item("linked_to", PyList::new(py, &info.linked_to)?)?;
    let custom = PyDict::new(py);
    for (k, v) in &info.custom {
        custom.set_item(k, v)?;
    }
    dict.set_item("custom", custom)?;
    Ok(dict)
}

fn convert_blob_stats_to_python<'py>(
    py: Python<'py>,
    stats: &BlobStatsResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("artifact_count", stats.artifact_count)?;
    dict.set_item("chunk_count", stats.chunk_count)?;
    dict.set_item("total_bytes", stats.total_bytes)?;
    dict.set_item("unique_bytes", stats.unique_bytes)?;
    dict.set_item("dedup_ratio", stats.dedup_ratio)?;
    dict.set_item("orphaned_chunks", stats.orphaned_chunks)?;
    Ok(dict)
}

fn convert_checkpoint_to_python<'py>(
    py: Python<'py>,
    cp: &CheckpointInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &cp.id)?;
    dict.set_item("name", &cp.name)?;
    dict.set_item("created_at", cp.created_at)?;
    dict.set_item("is_auto", cp.is_auto)?;
    Ok(dict)
}

fn convert_unified_item_to_python<'py>(
    py: Python<'py>,
    item: &UnifiedItem,
) -> PyResult<Bound<'py, PyDict>> {
    let item_dict = PyDict::new(py);
    item_dict.set_item("source", &item.source)?;
    item_dict.set_item("id", &item.id)?;
    let data = PyDict::new(py);
    for (k, v) in &item.data {
        data.set_item(k, v)?;
    }
    item_dict.set_item("data", data)?;
    if let Some(ref emb) = item.embedding {
        item_dict.set_item("embedding", PyList::new(py, emb)?)?;
    }
    if let Some(score) = item.score {
        item_dict.set_item("score", score)?;
    }
    Ok(item_dict)
}

fn convert_unified_to_python<'py>(
    py: Python<'py>,
    unified: &UnifiedResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("description", &unified.description)?;
    let items = PyList::empty(py);
    for item in &unified.items {
        items.append(convert_unified_item_to_python(py, item)?)?;
    }
    dict.set_item("items", items)?;
    Ok(dict)
}

#[allow(clippy::too_many_lines)]
fn convert_chain_result_to_python<'py>(
    py: Python<'py>,
    result: &ChainResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match result {
        ChainResult::TransactionBegun { tx_id } => {
            dict.set_item("chain_type", "transaction_begun")?;
            dict.set_item("tx_id", tx_id)?;
        },
        ChainResult::Committed { block_hash, height } => {
            dict.set_item("chain_type", "committed")?;
            dict.set_item("block_hash", block_hash)?;
            dict.set_item("height", *height)?;
        },
        ChainResult::RolledBack { to_height } => {
            dict.set_item("chain_type", "rolled_back")?;
            dict.set_item("to_height", *to_height)?;
        },
        ChainResult::History(entries) => {
            dict.set_item("chain_type", "history")?;
            let py_entries = PyList::empty(py);
            for entry in entries {
                py_entries.append(convert_chain_history_entry_to_python(py, entry)?)?;
            }
            dict.set_item("entries", py_entries)?;
        },
        ChainResult::Similar(items) => {
            dict.set_item("chain_type", "similar")?;
            let py_items = PyList::empty(py);
            for item in items {
                py_items.append(convert_chain_similar_to_python(py, item)?)?;
            }
            dict.set_item("items", py_items)?;
        },
        ChainResult::Drift(drift) => {
            dict.set_item("chain_type", "drift")?;
            dict.set_item("from_height", drift.from_height)?;
            dict.set_item("to_height", drift.to_height)?;
            dict.set_item("total_drift", drift.total_drift)?;
            dict.set_item("avg_drift_per_block", drift.avg_drift_per_block)?;
            dict.set_item("max_drift", drift.max_drift)?;
        },
        ChainResult::Height(h) => {
            dict.set_item("chain_type", "height")?;
            dict.set_item("height", *h)?;
        },
        ChainResult::Tip { hash, height } => {
            dict.set_item("chain_type", "tip")?;
            dict.set_item("hash", hash)?;
            dict.set_item("height", *height)?;
        },
        ChainResult::Block(block) => {
            dict.set_item("chain_type", "block")?;
            dict.set_item("height", block.height)?;
            dict.set_item("hash", &block.hash)?;
            dict.set_item("prev_hash", &block.prev_hash)?;
            dict.set_item("timestamp", block.timestamp)?;
            dict.set_item("transaction_count", block.transaction_count)?;
            dict.set_item("proposer", &block.proposer)?;
        },
        ChainResult::Codebook(cb) => {
            dict.set_item("chain_type", "codebook")?;
            dict.set_item("scope", &cb.scope)?;
            dict.set_item("entry_count", cb.entry_count)?;
            dict.set_item("dimension", cb.dimension)?;
            if let Some(ref domain) = cb.domain {
                dict.set_item("domain", domain)?;
            }
        },
        ChainResult::Verified { ok, errors } => {
            dict.set_item("chain_type", "verified")?;
            dict.set_item("ok", *ok)?;
            dict.set_item("errors", PyList::new(py, errors)?)?;
        },
        ChainResult::TransitionAnalysis(analysis) => {
            dict.set_item("chain_type", "transition_analysis")?;
            dict.set_item("total_transitions", analysis.total_transitions)?;
            dict.set_item("valid_transitions", analysis.valid_transitions)?;
            dict.set_item("invalid_transitions", analysis.invalid_transitions)?;
            dict.set_item("avg_validity_score", analysis.avg_validity_score)?;
        },
    }
    Ok(dict)
}

fn convert_chain_history_entry_to_python<'py>(
    py: Python<'py>,
    entry: &ChainHistoryEntry,
) -> PyResult<Bound<'py, PyDict>> {
    let e = PyDict::new(py);
    e.set_item("height", entry.height)?;
    e.set_item("transaction_type", &entry.transaction_type)?;
    if let Some(ref data) = entry.data {
        e.set_item("data", PyBytes::new(py, data))?;
    }
    Ok(e)
}

fn convert_chain_similar_to_python<'py>(
    py: Python<'py>,
    item: &ChainSimilarResult,
) -> PyResult<Bound<'py, PyDict>> {
    let i = PyDict::new(py);
    i.set_item("block_hash", &item.block_hash)?;
    i.set_item("height", item.height)?;
    i.set_item("similarity", item.similarity)?;
    Ok(i)
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QueryRouter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use query_router::{
        ChainBlockInfo, ChainCodebookInfo, ChainDriftResult, ChainTransitionAnalysis,
    };
    use std::fs;
    use tempfile::tempdir;
    use tensor_store::TensorData;

    #[test]
    fn test_new_creates_in_memory_router() {
        let router = QueryRouter::new();
        let inner = router.inner.read();
        assert!(inner.vector().store().is_empty());
    }

    #[test]
    fn test_with_path_nonexistent_creates_fresh() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.db");

        let router = QueryRouter::with_path(path.to_str().unwrap()).unwrap();
        let inner = router.inner.read();
        assert!(inner.vector().store().is_empty());
    }

    #[test]
    fn test_save_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let router = QueryRouter::new();
        router.save(path.to_str().unwrap()).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_save_uncompressed_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_uncompressed.db");

        let router = QueryRouter::new();
        router.save_uncompressed(path.to_str().unwrap()).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_roundtrip_compressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip.db");

        let router = QueryRouter::new();
        {
            let inner = router.inner.read();
            let store = inner.vector().store();
            let mut data = TensorData::new();
            data.set(
                "name",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                    "test".to_string(),
                )),
            );
            store.put("entity:1", data).unwrap();
        }

        router.save(path.to_str().unwrap()).unwrap();

        let loaded = QueryRouter::with_path(path.to_str().unwrap()).unwrap();
        let inner = loaded.inner.read();
        let store = inner.vector().store();
        let data = store.get("entity:1").unwrap();
        assert!(data.get("name").is_some());
    }

    #[test]
    fn test_roundtrip_uncompressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip_uncompressed.db");

        let router = QueryRouter::new();
        {
            let inner = router.inner.read();
            let store = inner.vector().store();
            let mut data = TensorData::new();
            data.set(
                "value",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(42)),
            );
            store.put("entity:2", data).unwrap();
        }

        router.save_uncompressed(path.to_str().unwrap()).unwrap();

        let loaded = QueryRouter::with_path(path.to_str().unwrap()).unwrap();
        let inner = loaded.inner.read();
        let store = inner.vector().store();
        let data = store.get("entity:2").unwrap();
        assert!(data.get("value").is_some());
    }

    #[test]
    fn test_with_path_loads_compressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("compressed.db");

        let router = QueryRouter::new();
        {
            let inner = router.inner.read();
            let store = inner.vector().store();
            let mut data = TensorData::new();
            data.set(
                "field",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                    "compressed".to_string(),
                )),
            );
            store.put("test:1", data).unwrap();
        }
        router.save(path.to_str().unwrap()).unwrap();

        let loaded = QueryRouter::with_path(path.to_str().unwrap()).unwrap();
        let inner = loaded.inner.read();
        assert!(!inner.vector().store().is_empty());
    }

    #[test]
    fn test_with_path_loads_uncompressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("uncompressed.db");

        let router = QueryRouter::new();
        {
            let inner = router.inner.read();
            let store = inner.vector().store();
            let mut data = TensorData::new();
            data.set(
                "field",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                    "uncompressed".to_string(),
                )),
            );
            store.put("test:2", data).unwrap();
        }
        router.save_uncompressed(path.to_str().unwrap()).unwrap();

        let loaded = QueryRouter::with_path(path.to_str().unwrap()).unwrap();
        let inner = loaded.inner.read();
        assert!(!inner.vector().store().is_empty());
    }

    #[test]
    fn test_save_error_invalid_path() {
        let router = QueryRouter::new();
        let result = router.save("/nonexistent/deeply/nested/path/test.db");
        assert!(result.is_err());
    }

    #[test]
    fn test_with_path_error_corrupt_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt.db");

        fs::write(&path, b"not a valid snapshot").unwrap();

        let result = QueryRouter::with_path(path.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_embedding_dimension_default() {
        let store = TensorStore::new();
        let dim = QueryRouter::detect_embedding_dimension(&store);
        assert_eq!(dim, DEFAULT_EMBEDDING_DIM);
    }

    #[test]
    fn test_detect_embedding_dimension_from_vector() {
        let store = TensorStore::new();
        let mut data = TensorData::new();
        data.set("embedding", TensorValue::Vector(vec![0.1; 1536]));
        store.put("test:1", data).unwrap();

        let dim = QueryRouter::detect_embedding_dimension(&store);
        assert_eq!(dim, 1536);
    }

    #[test]
    fn test_convert_artifact_info() {
        use std::collections::HashMap;
        let info = ArtifactInfoResult {
            id: "art-123".to_string(),
            filename: "test.txt".to_string(),
            content_type: "text/plain".to_string(),
            size: 1024,
            checksum: "abc123".to_string(),
            chunk_count: 2,
            created: 1000,
            modified: 2000,
            created_by: "user".to_string(),
            tags: vec!["tag1".to_string(), "tag2".to_string()],
            linked_to: vec!["other".to_string()],
            custom: HashMap::from([("key".to_string(), "value".to_string())]),
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_artifact_info_to_python(py, &info).unwrap();
            assert_eq!(
                dict.get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "art-123"
            );
            assert_eq!(
                dict.get_item("size")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1024
            );
            assert_eq!(
                dict.get_item("chunk_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
        });
    }

    #[test]
    fn test_convert_blob_stats() {
        let stats = BlobStatsResult {
            artifact_count: 10,
            chunk_count: 50,
            total_bytes: 10000,
            unique_bytes: 8000,
            dedup_ratio: 0.8,
            orphaned_chunks: 2,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_blob_stats_to_python(py, &stats).unwrap();
            assert_eq!(
                dict.get_item("artifact_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                10
            );
            assert_eq!(
                dict.get_item("chunk_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                50
            );
            assert_eq!(
                dict.get_item("dedup_ratio")
                    .unwrap()
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                0.8
            );
        });
    }

    #[test]
    fn test_convert_checkpoint() {
        let cp = CheckpointInfo {
            id: "cp-1".to_string(),
            name: "checkpoint1".to_string(),
            created_at: 12345,
            is_auto: false,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_checkpoint_to_python(py, &cp).unwrap();
            assert_eq!(
                dict.get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "cp-1"
            );
            assert_eq!(
                dict.get_item("name")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "checkpoint1"
            );
            assert_eq!(
                dict.get_item("is_auto")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap(),
                false
            );
        });
    }

    #[test]
    fn test_convert_unified_item() {
        use std::collections::HashMap;
        let item = UnifiedItem {
            source: "vector".to_string(),
            id: "entity:1".to_string(),
            data: HashMap::from([("field".to_string(), "value".to_string())]),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            score: Some(0.95),
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_unified_item_to_python(py, &item).unwrap();
            assert_eq!(
                dict.get_item("source")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "vector"
            );
            assert_eq!(
                dict.get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "entity:1"
            );
            let score: f32 = dict.get_item("score").unwrap().unwrap().extract().unwrap();
            assert!((score - 0.95).abs() < 0.001);
        });
    }

    #[test]
    fn test_convert_unified_result() {
        use std::collections::HashMap;
        let unified = UnifiedResult {
            description: "test results".to_string(),
            items: vec![UnifiedItem {
                source: "graph".to_string(),
                id: "node:1".to_string(),
                data: HashMap::new(),
                embedding: None,
                score: None,
            }],
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_unified_to_python(py, &unified).unwrap();
            assert_eq!(
                dict.get_item("description")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "test results"
            );
        });
    }

    #[test]
    fn test_convert_chain_transaction_begun() {
        let result = ChainResult::TransactionBegun {
            tx_id: "tx-123".to_string(),
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "transaction_begun"
            );
            assert_eq!(
                dict.get_item("tx_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "tx-123"
            );
        });
    }

    #[test]
    fn test_convert_chain_committed() {
        let result = ChainResult::Committed {
            block_hash: "hash123".to_string(),
            height: 100,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "committed"
            );
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                100
            );
        });
    }

    #[test]
    fn test_convert_chain_rolled_back() {
        let result = ChainResult::RolledBack { to_height: 50 };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "rolled_back"
            );
            assert_eq!(
                dict.get_item("to_height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                50
            );
        });
    }

    #[test]
    fn test_convert_chain_history() {
        let result = ChainResult::History(vec![
            ChainHistoryEntry {
                height: 10,
                transaction_type: "put".to_string(),
                data: Some(vec![1, 2, 3]),
            },
            ChainHistoryEntry {
                height: 20,
                transaction_type: "delete".to_string(),
                data: None,
            },
        ]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "history"
            );
        });
    }

    #[test]
    fn test_convert_chain_similar() {
        let result = ChainResult::Similar(vec![ChainSimilarResult {
            block_hash: "hash1".to_string(),
            height: 5,
            similarity: 0.9,
        }]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "similar"
            );
        });
    }

    #[test]
    fn test_convert_chain_drift() {
        let result = ChainResult::Drift(ChainDriftResult {
            from_height: 0,
            to_height: 100,
            total_drift: 5.0,
            avg_drift_per_block: 0.05,
            max_drift: 1.2,
        });
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "drift"
            );
            assert_eq!(
                dict.get_item("from_height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                0
            );
            assert_eq!(
                dict.get_item("to_height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                100
            );
        });
    }

    #[test]
    fn test_convert_chain_height() {
        let result = ChainResult::Height(42);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "height"
            );
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                42
            );
        });
    }

    #[test]
    fn test_convert_chain_tip() {
        let result = ChainResult::Tip {
            hash: "tiphash".to_string(),
            height: 999,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "tip"
            );
            assert_eq!(
                dict.get_item("hash")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "tiphash"
            );
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                999
            );
        });
    }

    #[test]
    fn test_convert_chain_block() {
        let result = ChainResult::Block(ChainBlockInfo {
            height: 10,
            hash: "blockhash".to_string(),
            prev_hash: "prevhash".to_string(),
            timestamp: 1234567890,
            transaction_count: 5,
            proposer: "node1".to_string(),
        });
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "block"
            );
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                10
            );
            assert_eq!(
                dict.get_item("proposer")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "node1"
            );
        });
    }

    #[test]
    fn test_convert_chain_codebook() {
        let result = ChainResult::Codebook(ChainCodebookInfo {
            scope: "global".to_string(),
            entry_count: 256,
            dimension: 64,
            domain: Some("embeddings".to_string()),
        });
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "codebook"
            );
            assert_eq!(
                dict.get_item("entry_count")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                256
            );
            assert_eq!(
                dict.get_item("domain")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "embeddings"
            );
        });
    }

    #[test]
    fn test_convert_chain_codebook_no_domain() {
        let result = ChainResult::Codebook(ChainCodebookInfo {
            scope: "local".to_string(),
            entry_count: 128,
            dimension: 32,
            domain: None,
        });
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "codebook"
            );
            assert!(dict.get_item("domain").unwrap().is_none());
        });
    }

    #[test]
    fn test_convert_chain_verified() {
        let result = ChainResult::Verified {
            ok: true,
            errors: vec![],
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "verified"
            );
            assert_eq!(
                dict.get_item("ok")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap(),
                true
            );
        });
    }

    #[test]
    fn test_convert_chain_verified_with_errors() {
        let result = ChainResult::Verified {
            ok: false,
            errors: vec!["error1".to_string(), "error2".to_string()],
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("ok")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap(),
                false
            );
        });
    }

    #[test]
    fn test_convert_chain_transition_analysis() {
        let result = ChainResult::TransitionAnalysis(ChainTransitionAnalysis {
            total_transitions: 100,
            valid_transitions: 95,
            invalid_transitions: 5,
            avg_validity_score: 0.95,
        });
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_result_to_python(py, &result).unwrap();
            assert_eq!(
                dict.get_item("chain_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "transition_analysis"
            );
            assert_eq!(
                dict.get_item("total_transitions")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                100
            );
            assert_eq!(
                dict.get_item("valid_transitions")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                95
            );
        });
    }

    #[test]
    fn test_convert_chain_history_entry() {
        let entry = ChainHistoryEntry {
            height: 15,
            transaction_type: "update".to_string(),
            data: Some(vec![4, 5, 6]),
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_history_entry_to_python(py, &entry).unwrap();
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                15
            );
            assert_eq!(
                dict.get_item("transaction_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "update"
            );
        });
    }

    #[test]
    fn test_convert_chain_history_entry_no_data() {
        let entry = ChainHistoryEntry {
            height: 20,
            transaction_type: "delete".to_string(),
            data: None,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_history_entry_to_python(py, &entry).unwrap();
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                20
            );
            assert!(dict.get_item("data").unwrap().is_none());
        });
    }

    #[test]
    fn test_convert_chain_similar_item() {
        let item = ChainSimilarResult {
            block_hash: "similar_hash".to_string(),
            height: 25,
            similarity: 0.85,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_chain_similar_to_python(py, &item).unwrap();
            assert_eq!(
                dict.get_item("block_hash")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "similar_hash"
            );
            assert_eq!(
                dict.get_item("height")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                25
            );
            let sim: f32 = dict
                .get_item("similarity")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((sim - 0.85).abs() < 0.001);
        });
    }

    #[test]
    fn test_convert_path_result() {
        let result = QueryResult::Path(vec![1, 2, 3, 4, 5]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_result_to_python(py, result).unwrap();
            let bound = dict.downcast_bound::<PyDict>(py).unwrap();
            assert_eq!(
                bound
                    .get_item("type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "path"
            );
        });
    }

    #[test]
    fn test_convert_artifact_list_result() {
        let result = QueryResult::ArtifactList(vec![
            "art-1".to_string(),
            "art-2".to_string(),
            "art-3".to_string(),
        ]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_result_to_python(py, result).unwrap();
            let bound = dict.downcast_bound::<PyDict>(py).unwrap();
            assert_eq!(
                bound
                    .get_item("type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "artifact_list"
            );
        });
    }

    #[test]
    fn test_convert_unified_item_no_embedding_no_score() {
        use std::collections::HashMap;
        let item = UnifiedItem {
            source: "relational".to_string(),
            id: "row:1".to_string(),
            data: HashMap::new(),
            embedding: None,
            score: None,
        };
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = convert_unified_item_to_python(py, &item).unwrap();
            assert_eq!(
                dict.get_item("source")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "relational"
            );
            assert!(dict.get_item("embedding").unwrap().is_none());
            assert!(dict.get_item("score").unwrap().is_none());
        });
    }
}
