// SPDX-License-Identifier: MIT OR Apache-2.0
//! SQL statement execution: `SELECT` (with WHERE/ORDER BY/LIMIT/GROUP BY/HAVING/JOIN),
//! `INSERT`, `UPDATE`, `DELETE`, `CREATE TABLE`. Aggregate helpers live here too.

#![allow(
    clippy::too_many_lines,
    reason = "SQL execution covers many sub-clauses"
)]

use std::collections::HashMap;

use neumann_parser::{
    self as parser, DeleteStmt, Expr, ExprKind, InsertSource, InsertStmt, JoinCondition, JoinKind,
    SelectStmt, TableRefKind, UpdateStmt,
};
use relational_engine::{ColumnarScanOptions, Condition, Row, Value};
use tensor_checkpoint::DestructiveOp;

use crate::policy::ProtectedOpResult;
use crate::{exec, protection, QueryResult, QueryRouter, Result, RouterError};

/// Aggregate function types for SELECT queries.
#[derive(Debug, Clone)]
pub enum AggregateFunc {
    Count(Option<String>),
    Sum(String),
    Avg(String),
    Min(String),
    Max(String),
}

impl QueryRouter {
    pub(crate) fn exec_select(&self, select: &SelectStmt) -> Result<QueryResult> {
        let from = select
            .from
            .as_ref()
            .ok_or_else(|| RouterError::MissingArgument("FROM clause".to_string()))?;

        let table_name = match &from.table.kind {
            TableRefKind::Table(ident) => &ident.name,
            TableRefKind::Subquery(_) => {
                return Err(RouterError::ParseError(
                    "Subqueries not yet supported".to_string(),
                ))
            },
        };

        // Handle JOINs if present
        if !from.joins.is_empty() {
            return self.exec_select_with_joins(select, table_name, from);
        }

        let condition = if let Some(ref where_expr) = select.where_clause {
            self.expr_to_condition(where_expr)?
        } else {
            Condition::True
        };

        // Check for aggregate functions in SELECT
        if let Some(agg_result) = self.try_exec_aggregates(select, table_name, &condition)? {
            return Ok(agg_result);
        }

        // Extract column projection from SELECT clause
        let projection = self.extract_projection(&select.columns)?;

        let options = ColumnarScanOptions {
            projection,
            prefer_columnar: true,
        };

        let mut rows = self
            .relational
            .select_columnar(table_name, condition, options)?;

        // Apply ORDER BY clause if present
        if !select.order_by.is_empty() {
            self.sort_rows(&mut rows, &select.order_by);
        }

        // Apply OFFSET clause if present
        if let Some(ref offset_expr) = select.offset {
            if let ExprKind::Literal(neumann_parser::Literal::Integer(n)) = &offset_expr.kind {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let offset = *n as usize; // OFFSET values are small positive integers
                if offset < rows.len() {
                    rows = rows.into_iter().skip(offset).collect();
                } else {
                    rows.clear();
                }
            }
        }

        // Apply LIMIT clause if present
        if let Some(ref limit_expr) = select.limit {
            if let ExprKind::Literal(neumann_parser::Literal::Integer(n)) = &limit_expr.kind {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let limit = *n as usize; // LIMIT values are small positive integers
                rows.truncate(limit);
            }
        }

        Ok(QueryResult::Rows(rows))
    }

    #[allow(clippy::too_many_lines)] // JOIN execution requires handling multiple join types and conditions
    pub(crate) fn exec_select_with_joins(
        &self,
        select: &SelectStmt,
        left_table: &str,
        from: &neumann_parser::FromClause,
    ) -> Result<QueryResult> {
        // For now, support only single JOIN (A JOIN B)
        // Multi-join (A JOIN B JOIN C) would require iterative approach
        if from.joins.len() > 1 {
            return Err(RouterError::ParseError(
                "Multiple JOINs not yet supported; use single JOIN".to_string(),
            ));
        }

        let join = &from.joins[0];
        let right_table = match &join.table.kind {
            TableRefKind::Table(ident) => &ident.name,
            TableRefKind::Subquery(_) => {
                return Err(RouterError::ParseError(
                    "Subquery JOINs not yet supported".to_string(),
                ))
            },
        };

        // Get table aliases or use table names
        let left_alias: &str = match &from.table.alias {
            Some(a) => &a.name,
            None => left_table,
        };
        let right_alias: &str = join.table.alias.as_ref().map_or(right_table, |a| &a.name);

        // Execute the appropriate join type
        let mut rows: Vec<Row> = match join.kind {
            JoinKind::Inner => {
                let (on_a, on_b) =
                    self.get_join_columns(join.condition.as_ref(), left_table, right_table)?;
                let pairs = self
                    .relational
                    .join(left_table, right_table, &on_a, &on_b)?;
                pairs
                    .into_iter()
                    .map(|(a, b)| self.merge_rows(Some(&a), Some(&b), left_alias, right_alias))
                    .collect()
            },
            JoinKind::Left => {
                let (on_a, on_b) =
                    self.get_join_columns(join.condition.as_ref(), left_table, right_table)?;
                let pairs = self
                    .relational
                    .left_join(left_table, right_table, &on_a, &on_b)?;
                pairs
                    .into_iter()
                    .map(|(a, b)| self.merge_rows(Some(&a), b.as_ref(), left_alias, right_alias))
                    .collect()
            },
            JoinKind::Right => {
                let (on_a, on_b) =
                    self.get_join_columns(join.condition.as_ref(), left_table, right_table)?;
                let pairs = self
                    .relational
                    .right_join(left_table, right_table, &on_a, &on_b)?;
                pairs
                    .into_iter()
                    .map(|(a, b)| self.merge_rows(a.as_ref(), Some(&b), left_alias, right_alias))
                    .collect()
            },
            JoinKind::Full => {
                let (on_a, on_b) =
                    self.get_join_columns(join.condition.as_ref(), left_table, right_table)?;
                let pairs = self
                    .relational
                    .full_join(left_table, right_table, &on_a, &on_b)?;
                pairs
                    .into_iter()
                    .map(|(a, b)| self.merge_rows(a.as_ref(), b.as_ref(), left_alias, right_alias))
                    .collect()
            },
            JoinKind::Cross => {
                let pairs = self.relational.cross_join(left_table, right_table)?;
                pairs
                    .into_iter()
                    .map(|(a, b)| self.merge_rows(Some(&a), Some(&b), left_alias, right_alias))
                    .collect()
            },
            JoinKind::Natural => {
                let pairs = self.relational.natural_join(left_table, right_table)?;
                pairs
                    .into_iter()
                    .map(|(a, b)| self.merge_rows(Some(&a), Some(&b), left_alias, right_alias))
                    .collect()
            },
        };

        // Apply WHERE clause if present
        if let Some(ref where_expr) = select.where_clause {
            rows.retain(|row| self.evaluate_join_condition(where_expr, row));
        }

        // Apply ORDER BY clause if present
        if !select.order_by.is_empty() {
            self.sort_rows(&mut rows, &select.order_by);
        }

        // Apply OFFSET clause if present
        if let Some(ref offset_expr) = select.offset {
            if let ExprKind::Literal(neumann_parser::Literal::Integer(n)) = &offset_expr.kind {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let offset = *n as usize; // OFFSET values are small positive integers
                if offset < rows.len() {
                    rows = rows.into_iter().skip(offset).collect();
                } else {
                    rows.clear();
                }
            }
        }

        // Apply LIMIT clause if present
        if let Some(ref limit_expr) = select.limit {
            if let ExprKind::Literal(neumann_parser::Literal::Integer(n)) = &limit_expr.kind {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let limit = *n as usize; // LIMIT values are small positive integers
                rows.truncate(limit);
            }
        }

        Ok(QueryResult::Rows(rows))
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn get_join_columns(
        &self,
        condition: Option<&JoinCondition>,
        left_table: &str,
        right_table: &str,
    ) -> Result<(String, String)> {
        exec::expr::get_join_columns(condition, left_table, right_table)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn evaluate_join_condition(&self, expr: &Expr, row: &Row) -> bool {
        exec::expr::evaluate_join_condition(expr, row)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    fn sort_rows(&self, rows: &mut [Row], order_by: &[neumann_parser::OrderByItem]) {
        exec::expr::sort_rows(rows, order_by);
    }

    // ========== Aggregate Function Handling ==========

    pub(crate) fn try_exec_aggregates(
        &self,
        select: &SelectStmt,
        table_name: &str,
        condition: &Condition,
    ) -> Result<Option<QueryResult>> {
        // Check if any column is an aggregate function
        let mut aggregates: Vec<(String, AggregateFunc)> = Vec::new();
        let mut non_agg_columns: Vec<(String, Expr)> = Vec::new();

        for item in &select.columns {
            if let Some(agg) = self.parse_aggregate(&item.expr) {
                let alias = item.alias.as_ref().map_or_else(
                    || self.aggregate_default_name(&item.expr),
                    |a| a.name.clone(),
                );
                aggregates.push((alias, agg));
            } else {
                let alias = item.alias.as_ref().map_or_else(
                    || {
                        self.expr_to_column_name(&item.expr)
                            .unwrap_or_else(|_| "?".to_string())
                    },
                    |a| a.name.clone(),
                );
                non_agg_columns.push((alias, item.expr.clone()));
            }
        }

        // If GROUP BY is present, handle grouped aggregation
        if !select.group_by.is_empty() {
            return self.exec_grouped_aggregates(
                select,
                table_name,
                condition,
                &aggregates,
                &non_agg_columns,
            );
        }

        if aggregates.is_empty() {
            return Ok(None);
        }

        // Compute aggregate values for the whole table
        let mut values: Vec<(String, Value)> = Vec::new();

        for (alias, agg) in aggregates {
            let val = match agg {
                AggregateFunc::Count(col) => {
                    let count = if let Some(ref column_name) = col {
                        self.relational
                            .count_column(table_name, column_name, condition.clone())?
                    } else {
                        self.relational.count(table_name, condition.clone())?
                    };
                    #[allow(clippy::cast_possible_wrap)]
                    Value::Int(count as i64)
                },
                AggregateFunc::Sum(col) => {
                    let sum = self.relational.sum(table_name, &col, condition.clone())?;
                    Value::Float(sum)
                },
                AggregateFunc::Avg(col) => self
                    .relational
                    .avg(table_name, &col, condition.clone())?
                    .map_or(Value::Null, Value::Float),
                AggregateFunc::Min(col) => self
                    .relational
                    .min(table_name, &col, condition.clone())?
                    .unwrap_or(Value::Null),
                AggregateFunc::Max(col) => self
                    .relational
                    .max(table_name, &col, condition.clone())?
                    .unwrap_or(Value::Null),
            };
            values.push((alias, val));
        }

        // Return single row with aggregate results
        let row = Row { id: 0, values };
        Ok(Some(QueryResult::Rows(vec![row])))
    }

    pub(crate) fn exec_grouped_aggregates(
        &self,
        select: &SelectStmt,
        table_name: &str,
        condition: &Condition,
        aggregates: &[(String, AggregateFunc)],
        non_agg_columns: &[(String, Expr)],
    ) -> Result<Option<QueryResult>> {
        use std::collections::HashMap;

        // Get all rows matching the WHERE condition
        let rows = self.relational.select_columnar(
            table_name,
            condition.clone(),
            ColumnarScanOptions {
                projection: None,
                prefer_columnar: true,
            },
        )?;

        // Extract group key column names from GROUP BY expressions
        let group_key_names: Vec<String> = select
            .group_by
            .iter()
            .filter_map(|expr| self.expr_to_column_name(expr).ok())
            .collect();

        // Group rows by GROUP BY column values (use string key since Value doesn't impl Hash)
        let mut groups: HashMap<String, (Vec<Value>, Vec<&Row>)> = HashMap::new();
        for row in &rows {
            let group_key: Vec<Value> = group_key_names
                .iter()
                .map(|col| {
                    row.values
                        .iter()
                        .find(|(c, _)| c == col)
                        .map_or(Value::Null, |(_, v)| v.clone())
                })
                .collect();
            let key_str = self.values_to_group_key(&group_key);
            groups
                .entry(key_str)
                .or_insert_with(|| (group_key, Vec::new()))
                .1
                .push(row);
        }

        // Compute aggregates for each group
        let mut result_rows: Vec<Row> = Vec::new();

        for (_, (_group_key, group_rows)) in groups {
            let mut values: Vec<(String, Value)> = Vec::new();

            // Add non-aggregate columns (group key columns)
            for (alias, expr) in non_agg_columns {
                let val = group_rows.first().map_or(Value::Null, |first_row| {
                    self.get_row_value(expr, first_row).unwrap_or(Value::Null)
                });
                values.push((alias.clone(), val));
            }

            // Compute aggregates for this group
            for (alias, agg) in aggregates {
                let val = self.compute_aggregate_for_group(agg, &group_rows);
                values.push((alias.clone(), val));
            }

            let row = Row { id: 0, values };

            // Apply HAVING filter if present
            if let Some(ref having_expr) = select.having {
                if !self.evaluate_join_condition(having_expr, &row) {
                    continue;
                }
            }

            result_rows.push(row);
        }

        Ok(Some(QueryResult::Rows(result_rows)))
    }

    #[allow(clippy::too_many_lines)] // Aggregate computation requires handling all SQL aggregate functions
    #[allow(clippy::unused_self)] // Method signature for API consistency
    pub(crate) fn compute_aggregate_for_group(&self, agg: &AggregateFunc, rows: &[&Row]) -> Value {
        match agg {
            AggregateFunc::Count(col) => {
                let count = if col.is_none() {
                    rows.len() as u64
                } else {
                    rows.iter()
                        .filter(|r| {
                            r.values.iter().any(|(c, v)| {
                                c == col.as_ref().unwrap() && !matches!(v, Value::Null)
                            })
                        })
                        .count() as u64
                };
                #[allow(clippy::cast_possible_wrap)]
                Value::Int(count as i64)
            },
            AggregateFunc::Sum(col) => {
                let mut sum = 0.0;
                for row in rows {
                    if let Some((_, val)) = row.values.iter().find(|(c, _)| c == col) {
                        match val {
                            #[allow(clippy::cast_precision_loss)]
                            Value::Int(i) => sum += *i as f64,
                            Value::Float(f) => sum += *f,
                            _ => {},
                        }
                    }
                }
                Value::Float(sum)
            },
            AggregateFunc::Avg(col) => {
                let mut sum = 0.0;
                let mut count = 0;
                for row in rows {
                    if let Some((_, val)) = row.values.iter().find(|(c, _)| c == col) {
                        match val {
                            Value::Int(i) => {
                                #[allow(clippy::cast_precision_loss)]
                                {
                                    sum += *i as f64;
                                }
                                count += 1;
                            },
                            Value::Float(f) => {
                                sum += *f;
                                count += 1;
                            },
                            _ => {},
                        }
                    }
                }
                if count == 0 {
                    Value::Null
                } else {
                    Value::Float(sum / f64::from(count))
                }
            },
            AggregateFunc::Min(col) => {
                let mut min_val: Option<Value> = None;
                for row in rows {
                    if let Some((_, val)) = row.values.iter().find(|(c, _)| c == col) {
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        min_val = Some(min_val.as_ref().map_or_else(
                            || val.clone(),
                            |current| match (current, val) {
                                (Value::Int(a), Value::Int(b)) if b < a => val.clone(),
                                (Value::Float(a), Value::Float(b)) if b < a => val.clone(),
                                (Value::String(a), Value::String(b)) if b < a => val.clone(),
                                _ => current.clone(),
                            },
                        ));
                    }
                }
                min_val.unwrap_or(Value::Null)
            },
            AggregateFunc::Max(col) => {
                let mut max_val: Option<Value> = None;
                for row in rows {
                    if let Some((_, val)) = row.values.iter().find(|(c, _)| c == col) {
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        max_val = Some(max_val.as_ref().map_or_else(
                            || val.clone(),
                            |current| match (current, val) {
                                (Value::Int(a), Value::Int(b)) if b > a => val.clone(),
                                (Value::Float(a), Value::Float(b)) if b > a => val.clone(),
                                (Value::String(a), Value::String(b)) if b > a => val.clone(),
                                _ => current.clone(),
                            },
                        ));
                    }
                }
                max_val.unwrap_or(Value::Null)
            },
        }
    }

    #[allow(clippy::unused_self)] // Method signature for API consistency
    pub(crate) fn values_to_group_key(&self, values: &[Value]) -> String {
        values
            .iter()
            .map(|v| match v {
                Value::Null => "NULL".to_string(),
                Value::Int(i) => format!("I:{i}"),
                Value::Float(f) => format!("F:{f}"),
                Value::String(s) => format!("S:{s}"),
                Value::Bool(b) => format!("B:{b}"),
                Value::Bytes(b) => format!("BY:{}", hex::encode(b)),
                Value::Json(j) => format!("J:{j}"),
                _ => "UNKNOWN".to_string(),
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    pub(crate) fn parse_aggregate(&self, expr: &Expr) -> Option<AggregateFunc> {
        if let ExprKind::Call(call) = &expr.kind {
            let name = call.name.name.to_uppercase();
            match name.as_str() {
                "COUNT" => {
                    if call.args.is_empty() || matches!(&call.args[0].kind, ExprKind::Wildcard) {
                        Some(AggregateFunc::Count(None))
                    } else {
                        let col = self.expr_to_column_name(&call.args[0]).ok()?;
                        Some(AggregateFunc::Count(Some(col)))
                    }
                },
                "SUM" => {
                    if call.args.is_empty() {
                        return None;
                    }
                    let col = self.expr_to_column_name(&call.args[0]).ok()?;
                    Some(AggregateFunc::Sum(col))
                },
                "AVG" => {
                    if call.args.is_empty() {
                        return None;
                    }
                    let col = self.expr_to_column_name(&call.args[0]).ok()?;
                    Some(AggregateFunc::Avg(col))
                },
                "MIN" => {
                    if call.args.is_empty() {
                        return None;
                    }
                    let col = self.expr_to_column_name(&call.args[0]).ok()?;
                    Some(AggregateFunc::Min(col))
                },
                "MAX" => {
                    if call.args.is_empty() {
                        return None;
                    }
                    let col = self.expr_to_column_name(&call.args[0]).ok()?;
                    Some(AggregateFunc::Max(col))
                },
                _ => None,
            }
        } else {
            None
        }
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn aggregate_default_name(&self, expr: &Expr) -> String {
        exec::expr::aggregate_default_name(expr)
    }

    #[allow(
        clippy::unused_self,
        reason = "Phase 3a wrapper; removed after Phase 3l"
    )]
    pub(crate) fn get_row_value(&self, expr: &Expr, row: &Row) -> Option<Value> {
        exec::expr::get_row_value(expr, row)
    }

    #[allow(clippy::unused_self)] // Method signature for API consistency
    #[allow(clippy::unnecessary_wraps)] // Returns Result for API consistency with other extract methods
    pub(crate) fn extract_projection(
        &self,
        items: &[neumann_parser::SelectItem],
    ) -> Result<Option<Vec<String>>> {
        // Check for SELECT *
        if items.len() == 1 && matches!(&items[0].expr.kind, ExprKind::Wildcard) {
            return Ok(None);
        }

        // Check if any item is a wildcard
        for item in items {
            if matches!(
                &item.expr.kind,
                ExprKind::Wildcard | ExprKind::QualifiedWildcard(_)
            ) {
                return Ok(None);
            }
        }

        let mut columns = Vec::with_capacity(items.len());
        for item in items {
            match &item.expr.kind {
                ExprKind::Ident(ident) => {
                    columns.push(ident.name.clone());
                },
                ExprKind::Qualified(_, name) => {
                    columns.push(name.name.clone());
                },
                _ => {
                    // For expressions (COUNT(*), a+b, etc.), fall back to all columns
                    return Ok(None);
                },
            }
        }

        Ok(Some(columns))
    }

    pub(crate) fn exec_insert(&self, insert: &InsertStmt) -> Result<QueryResult> {
        match &insert.source {
            InsertSource::Values(rows) => {
                let mut ids = Vec::new();
                for row_values in rows {
                    let mut values = HashMap::new();
                    // Match columns to values
                    if let Some(ref cols) = insert.columns {
                        // Explicit columns specified
                        for (col, val) in cols.iter().zip(row_values.iter()) {
                            values.insert(col.name.clone(), self.expr_to_value(val)?);
                        }
                    } else {
                        // No columns specified - use table schema order
                        let schema = self.relational.get_schema(&insert.table.name)?;
                        for (col, val) in schema.columns.iter().zip(row_values.iter()) {
                            values.insert(col.name.clone(), self.expr_to_value(val)?);
                        }
                    }
                    let id = self.relational.insert(&insert.table.name, values)?;
                    ids.push(id);
                }
                Ok(QueryResult::Ids(ids))
            },
            InsertSource::Query(select) => {
                // Execute the SELECT query first
                let select_result = self.exec_select(select)?;

                // Extract rows from the result
                let QueryResult::Rows(rows) = select_result else {
                    return Err(RouterError::ParseError(
                        "INSERT ... SELECT query did not return rows".to_string(),
                    ));
                };

                if rows.is_empty() {
                    return Ok(QueryResult::Ids(vec![]));
                }

                // Get the target table schema
                let schema = self.relational.get_schema(&insert.table.name)?;

                // Determine column mapping
                let columns: Vec<String> = if let Some(ref cols) = insert.columns {
                    cols.iter().map(|c| c.name.clone()).collect()
                } else {
                    schema.columns.iter().map(|c| c.name.clone()).collect()
                };

                // Insert each row from the SELECT result
                let mut ids = Vec::new();
                for row in rows {
                    let mut values: HashMap<String, Value> = HashMap::new();

                    for col in &columns {
                        if let Some(val) = row.get(col) {
                            values.insert(col.clone(), val.clone());
                        }
                    }

                    let id = self.relational.insert(&insert.table.name, values)?;
                    ids.push(id);
                }

                Ok(QueryResult::Ids(ids))
            },
        }
    }

    // ========== Auto-Checkpoint Protection ==========

    /// Check and optionally create checkpoint before destructive operation.
    /// Delegates to [`protection`].
    pub(crate) fn protect_destructive_op(
        &self,
        command: &str,
        op: DestructiveOp,
        sample_data: Vec<String>,
    ) -> ProtectedOpResult {
        protection::protect_destructive_op(self, command, op, sample_data)
    }

    /// Collect sample data for a relational delete preview.
    /// Delegates to [`protection`].
    pub(crate) fn collect_delete_sample(
        &self,
        table: &str,
        condition: &Condition,
        limit: usize,
    ) -> (usize, Vec<String>) {
        protection::collect_delete_sample(self, table, condition, limit)
    }

    /// Collect sample data for a DROP TABLE preview.
    /// Delegates to [`protection`].
    pub(crate) fn collect_table_sample(&self, table: &str, limit: usize) -> (usize, Vec<String>) {
        protection::collect_table_sample(self, table, limit)
    }

    /// Collect info about a node for deletion preview.
    /// Delegates to [`protection`].
    pub(crate) fn collect_node_info(&self, node_id: u64) -> (usize, Vec<String>) {
        protection::collect_node_info(self, node_id)
    }

    /// Collect info about an edge for deletion preview.
    /// Delegates to [`protection`].
    pub(crate) fn collect_edge_info(&self, edge_id: u64) -> Vec<String> {
        protection::collect_edge_info(self, edge_id)
    }

    // ========== Query Execution Methods ==========

    pub(crate) fn exec_update(&self, update: &UpdateStmt) -> Result<QueryResult> {
        let condition = if let Some(ref where_expr) = update.where_clause {
            self.expr_to_condition(where_expr)?
        } else {
            Condition::True
        };

        let mut values = HashMap::new();
        for assign in &update.assignments {
            values.insert(
                assign.column.name.clone(),
                self.expr_to_value(&assign.value)?,
            );
        }

        let count = self
            .relational
            .update(&update.table.name, condition, values)?;
        Ok(QueryResult::Count(count))
    }

    pub(crate) fn exec_delete(&self, delete: &DeleteStmt) -> Result<QueryResult> {
        let table = &delete.table.name;
        let condition = if let Some(ref where_expr) = delete.where_clause {
            self.expr_to_condition(where_expr)?
        } else {
            Condition::True
        };

        // Collect sample data for preview
        let (row_count, sample_data) = self.collect_delete_sample(table, &condition, 5);

        // Check for auto-checkpoint protection
        if row_count > 0 {
            let op = DestructiveOp::Delete {
                table: table.clone(),
                row_count,
            };

            let command = format!(
                "DELETE FROM {}{}",
                table,
                if delete.where_clause.is_some() {
                    " WHERE ..."
                } else {
                    ""
                }
            );

            match self.protect_destructive_op(&command, op, sample_data) {
                ProtectedOpResult::Proceed => {},
                ProtectedOpResult::Cancelled => {
                    return Err(RouterError::CheckpointError(
                        "Operation cancelled by user".to_string(),
                    ));
                },
            }
        }

        let count = self.relational.delete_rows(table, condition)?;
        Ok(QueryResult::Count(count))
    }

    pub(crate) fn exec_create_table(
        &self,
        create: &parser::CreateTableStmt,
    ) -> Result<QueryResult> {
        if create.if_not_exists && self.relational.table_exists(&create.table.name) {
            return Ok(QueryResult::Empty);
        }

        let mut columns = Vec::new();
        for col in &create.columns {
            let col_type = self.data_type_to_column_type(&col.data_type)?;
            let mut column = relational_engine::Column::new(&col.name.name, col_type);

            // Check for nullable
            let is_nullable = !col
                .constraints
                .iter()
                .any(|c| matches!(c, parser::ColumnConstraint::NotNull));
            if is_nullable {
                column = column.nullable();
            }
            columns.push(column);
        }

        let schema = relational_engine::Schema::new(columns);
        self.relational.create_table(&create.table.name, schema)?;
        Ok(QueryResult::Empty)
    }

    /// Execute `DROP TABLE [IF EXISTS] <name>`.
    pub(crate) fn exec_drop_table(&self, drop: &parser::DropTableStmt) -> Result<QueryResult> {
        let table = &drop.table.name;
        let (row_count, sample_data) = self.collect_table_sample(table, 5);
        let op = DestructiveOp::DropTable {
            table: table.clone(),
            row_count,
        };
        match self.protect_destructive_op(&format!("DROP TABLE {table}"), op, sample_data) {
            ProtectedOpResult::Proceed => {},
            ProtectedOpResult::Cancelled => {
                return Err(RouterError::CheckpointError(
                    "Operation cancelled by user".to_string(),
                ));
            },
        }
        self.relational.drop_table(table)?;
        Ok(QueryResult::Empty)
    }

    /// Execute `CREATE INDEX ON <table>(<columns>)`. Only the first column is used.
    pub(crate) fn exec_create_index(
        &self,
        create: &parser::CreateIndexStmt,
    ) -> Result<QueryResult> {
        if let Some(col) = create.columns.first() {
            self.relational
                .create_index(&create.table.name, &col.name)?;
        }
        Ok(QueryResult::Empty)
    }

    /// Execute `DROP INDEX ON <table>(<column>)` or `DROP INDEX <name>`.
    pub(crate) fn exec_drop_index(&self, drop: &parser::DropIndexStmt) -> Result<QueryResult> {
        if let (Some(table), Some(column)) = (&drop.table, &drop.column) {
            if drop.if_exists && !self.relational.has_index(&table.name, &column.name) {
                return Ok(QueryResult::Empty);
            }
            let op = DestructiveOp::DropIndex {
                table: table.name.clone(),
                column: column.name.clone(),
            };
            match self.protect_destructive_op(
                &format!("DROP INDEX ON {}({})", table.name, column.name),
                op,
                vec![format!("index on {}.{}", table.name, column.name)],
            ) {
                ProtectedOpResult::Proceed => {},
                ProtectedOpResult::Cancelled => {
                    return Err(RouterError::CheckpointError(
                        "Operation cancelled by user".to_string(),
                    ));
                },
            }
            self.relational.drop_index(&table.name, &column.name)?;
            Ok(QueryResult::Empty)
        } else if let Some(name) = &drop.name {
            Err(RouterError::ParseError(format!(
                "Named index '{}' not supported. Use: DROP INDEX ON table(column)",
                name.name
            )))
        } else {
            Err(RouterError::ParseError(
                "Invalid DROP INDEX syntax".to_string(),
            ))
        }
    }

    /// Execute `SHOW TABLES`.
    pub(crate) fn exec_show_tables(&self) -> QueryResult {
        QueryResult::TableList(self.relational.list_tables())
    }
}
