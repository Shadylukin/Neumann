// SPDX-License-Identifier: MIT OR Apache-2.0
//! `DESCRIBE` statement execution.

use neumann_parser::{DescribeStmt, DescribeTarget};

use crate::{QueryResult, QueryRouter, Result};

/// Execute a `DESCRIBE TABLE|NODE|EDGE ...` statement.
pub fn exec_describe(router: &QueryRouter, desc: &DescribeStmt) -> Result<QueryResult> {
    match &desc.target {
        DescribeTarget::Table(name) => {
            use std::fmt::Write;
            let schema = router.relational.get_schema(&name.name)?;
            let mut info = format!("Table: {}\n", name.name);
            info.push_str("Columns:\n");
            for col in &schema.columns {
                let _ = writeln!(
                    info,
                    "  {} {:?}{}",
                    col.name,
                    col.column_type,
                    if col.nullable { "" } else { " NOT NULL" }
                );
            }
            Ok(QueryResult::Value(info))
        },
        DescribeTarget::Node(label) => {
            let total_nodes = router.graph.node_count();
            Ok(QueryResult::Value(format!(
                "Node label '{}': Use NODE LIST {} to see nodes. Total nodes in graph: {}",
                label.name, label.name, total_nodes
            )))
        },
        DescribeTarget::Edge(edge_type) => {
            let total_edges = router.graph.edge_count();
            Ok(QueryResult::Value(format!(
                "Edge type '{}': Use EDGE LIST {} to see edges. Total edges in graph: {}",
                edge_type.name, edge_type.name, total_edges
            )))
        },
    }
}
