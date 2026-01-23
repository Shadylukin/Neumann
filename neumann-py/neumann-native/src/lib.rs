//! Native Python bindings for Neumann database via PyO3.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};

use query_router::{QueryResult, QueryRouter as RustQueryRouter};

/// Python-accessible QueryRouter wrapper.
#[pyclass]
struct QueryRouter {
    inner: Arc<RwLock<RustQueryRouter>>,
}

#[pymethods]
impl QueryRouter {
    /// Create a new in-memory QueryRouter.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RustQueryRouter::new())),
        }
    }

    /// Create a QueryRouter with persistent storage at the given path.
    #[staticmethod]
    fn with_path(_path: &str) -> PyResult<Self> {
        // For now, just create an in-memory router
        // Persistence support can be added later
        Ok(Self {
            inner: Arc::new(RwLock::new(RustQueryRouter::new())),
        })
    }

    /// Execute a query and return the result as a Python dict.
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

/// Convert a QueryResult to a Python object.
fn convert_result_to_python(py: Python<'_>, result: QueryResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    match result {
        QueryResult::Empty => {
            dict.set_item("type", "empty")?;
        }
        QueryResult::Value(s) => {
            dict.set_item("type", "value")?;
            dict.set_item("data", s)?;
        }
        QueryResult::Count(n) => {
            dict.set_item("type", "count")?;
            dict.set_item("data", n)?;
        }
        QueryResult::Rows(rows) => {
            dict.set_item("type", "rows")?;
            let py_rows = PyList::empty(py);
            for row in rows {
                let row_dict = PyDict::new(py);
                for (k, v) in row.values() {
                    row_dict.set_item(k, convert_value_to_python(py, v)?)?;
                }
                py_rows.append(row_dict)?;
            }
            dict.set_item("data", py_rows)?;
        }
        QueryResult::Nodes(nodes) => {
            dict.set_item("type", "nodes")?;
            let py_nodes = PyList::empty(py);
            for node in nodes {
                let node_dict = PyDict::new(py);
                node_dict.set_item("id", node.id())?;
                node_dict.set_item("label", node.label())?;
                let props = PyDict::new(py);
                for (k, v) in node.properties() {
                    props.set_item(k, convert_value_to_python(py, v)?)?;
                }
                node_dict.set_item("properties", props)?;
                py_nodes.append(node_dict)?;
            }
            dict.set_item("data", py_nodes)?;
        }
        QueryResult::Edges(edges) => {
            dict.set_item("type", "edges")?;
            let py_edges = PyList::empty(py);
            for edge in edges {
                let edge_dict = PyDict::new(py);
                edge_dict.set_item("id", edge.id())?;
                edge_dict.set_item("type", edge.edge_type())?;
                edge_dict.set_item("source", edge.source())?;
                edge_dict.set_item("target", edge.target())?;
                let props = PyDict::new(py);
                for (k, v) in edge.properties() {
                    props.set_item(k, convert_value_to_python(py, v)?)?;
                }
                edge_dict.set_item("properties", props)?;
                py_edges.append(edge_dict)?;
            }
            dict.set_item("data", py_edges)?;
        }
        QueryResult::Similar(items) => {
            dict.set_item("type", "similar")?;
            let py_items = PyList::empty(py);
            for item in items {
                let item_dict = PyDict::new(py);
                item_dict.set_item("key", item.key)?;
                item_dict.set_item("score", item.score)?;
                py_items.append(item_dict)?;
            }
            dict.set_item("data", py_items)?;
        }
        QueryResult::Ids(ids) => {
            dict.set_item("type", "ids")?;
            let py_ids = PyList::new(py, &ids)?;
            dict.set_item("data", py_ids)?;
        }
        QueryResult::TableList(tables) => {
            dict.set_item("type", "table_list")?;
            let py_tables = PyList::new(py, &tables)?;
            dict.set_item("data", py_tables)?;
        }
        QueryResult::Blob(data) => {
            dict.set_item("type", "blob")?;
            dict.set_item("data", PyBytes::new(py, &data))?;
        }
        _ => {
            dict.set_item("type", "unknown")?;
        }
    }

    Ok(dict.into())
}

/// Convert a relational_engine Value to a Python object.
fn convert_value_to_python(py: Python<'_>, value: &relational_engine::Value) -> PyResult<PyObject> {
    use relational_engine::Value;

    match value {
        Value::Null => Ok(py.None()),
        Value::Int(i) => Ok(i.into_pyobject(py)?.into_any().unbind()),
        Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.into_any().unbind()),
    }
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QueryRouter>()?;
    Ok(())
}
