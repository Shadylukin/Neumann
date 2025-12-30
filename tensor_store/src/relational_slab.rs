//! Columnar storage for relational data.
//!
//! RelationalSlab stores table rows in a column-oriented format, optimizing
//! for scan operations and batch inserts. Each column is stored as a separate
//! vector, providing cache-friendly access patterns for analytical queries.
//!
//! # Design Philosophy
//!
//! - Column-oriented storage for efficient scans
//! - Deletion via bitmap (no data movement)
//! - B-tree indexes that split nodes (no resize stalls)
//! - Row IDs are stable (append-only)

use bitvec::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

/// Range operation for B-tree index queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// Row identifier within a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RowId(pub u64);

impl RowId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for RowId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

/// Column data type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    Int,
    Float,
    String,
    Bool,
}

/// Column definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: ColumnType,
    pub nullable: bool,
}

impl ColumnDef {
    pub fn new(name: &str, col_type: ColumnType, nullable: bool) -> Self {
        Self {
            name: name.to_string(),
            col_type,
            nullable,
        }
    }
}

/// Table schema definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<String>,
}

impl TableSchema {
    pub fn new(columns: Vec<ColumnDef>) -> Self {
        Self {
            columns,
            primary_key: None,
        }
    }

    pub fn with_primary_key(mut self, column: &str) -> Self {
        self.primary_key = Some(column.to_string());
        self
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }
}

/// Column value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl ColumnValue {
    pub fn is_null(&self) -> bool {
        matches!(self, ColumnValue::Null)
    }
}

/// A row of column values.
pub type Row = Vec<ColumnValue>;

/// Columnar storage for a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ColumnStorage {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<Option<String>>),
    Bool(BitVec),
}

impl ColumnStorage {
    fn new(col_type: &ColumnType, capacity: usize) -> Self {
        match col_type {
            ColumnType::Int => Self::Int(Vec::with_capacity(capacity)),
            ColumnType::Float => Self::Float(Vec::with_capacity(capacity)),
            ColumnType::String => Self::String(Vec::with_capacity(capacity)),
            ColumnType::Bool => Self::Bool(BitVec::with_capacity(capacity)),
        }
    }

    fn push(&mut self, value: &ColumnValue) {
        match (self, value) {
            (Self::Int(v), ColumnValue::Int(i)) => v.push(*i),
            (Self::Int(v), ColumnValue::Null) => v.push(0),
            (Self::Float(v), ColumnValue::Float(f)) => v.push(*f),
            (Self::Float(v), ColumnValue::Null) => v.push(0.0),
            (Self::String(v), ColumnValue::String(s)) => v.push(Some(s.clone())),
            (Self::String(v), ColumnValue::Null) => v.push(None),
            (Self::Bool(v), ColumnValue::Bool(b)) => v.push(*b),
            (Self::Bool(v), ColumnValue::Null) => v.push(false),
            _ => {}, // Type mismatch - should be validated before
        }
    }

    fn get(&self, idx: usize) -> ColumnValue {
        match self {
            Self::Int(v) => v
                .get(idx)
                .map(|i| ColumnValue::Int(*i))
                .unwrap_or(ColumnValue::Null),
            Self::Float(v) => v
                .get(idx)
                .map(|f| ColumnValue::Float(*f))
                .unwrap_or(ColumnValue::Null),
            Self::String(v) => v
                .get(idx)
                .and_then(|s| s.clone())
                .map(ColumnValue::String)
                .unwrap_or(ColumnValue::Null),
            Self::Bool(v) => v
                .get(idx)
                .map(|b| ColumnValue::Bool(*b))
                .unwrap_or(ColumnValue::Null),
        }
    }
}

/// Per-table storage with columnar layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableStorage {
    schema: TableSchema,
    columns: Vec<ColumnStorage>,
    alive: BitVec,
    null_bitmaps: Vec<BitVec>,
    total_rows: usize,
    live_rows: usize,
    hash_indexes: HashMap<String, BTreeMap<i64, Vec<RowId>>>,
}

impl TableStorage {
    fn new(schema: TableSchema) -> Self {
        let num_cols = schema.columns.len();
        let columns: Vec<ColumnStorage> = schema
            .columns
            .iter()
            .map(|c| ColumnStorage::new(&c.col_type, 100))
            .collect();
        let null_bitmaps = vec![BitVec::new(); num_cols];

        Self {
            schema,
            columns,
            alive: BitVec::new(),
            null_bitmaps,
            total_rows: 0,
            live_rows: 0,
            hash_indexes: HashMap::new(),
        }
    }

    fn insert(&mut self, row: &Row) -> RowId {
        let row_id = RowId::new(self.total_rows as u64);

        for (col_idx, value) in row.iter().enumerate() {
            self.columns[col_idx].push(value);
            self.null_bitmaps[col_idx].push(value.is_null());
        }

        self.alive.push(true);
        self.total_rows += 1;
        self.live_rows += 1;

        // Update indexes
        for (col_name, index) in &mut self.hash_indexes {
            if let Some(col_idx) = self.schema.column_index(col_name) {
                if let ColumnValue::Int(key) = &row[col_idx] {
                    index.entry(*key).or_default().push(row_id);
                }
            }
        }

        row_id
    }

    fn get(&self, row_id: RowId) -> Option<Row> {
        let idx = row_id.as_u64() as usize;
        if idx >= self.total_rows || !self.alive[idx] {
            return None;
        }

        let row: Row = self
            .columns
            .iter()
            .enumerate()
            .map(|(col_idx, col)| {
                if self.null_bitmaps[col_idx][idx] {
                    ColumnValue::Null
                } else {
                    col.get(idx)
                }
            })
            .collect();

        Some(row)
    }

    fn delete(&mut self, row_id: RowId) -> bool {
        let idx = row_id.as_u64() as usize;
        if idx >= self.total_rows || !self.alive[idx] {
            return false;
        }

        self.alive.set(idx, false);
        self.live_rows -= 1;
        true
    }

    fn scan<F>(&self, predicate: F) -> Vec<(RowId, Row)>
    where
        F: Fn(&Row) -> bool,
    {
        let mut results = Vec::new();

        for idx in 0..self.total_rows {
            if !self.alive[idx] {
                continue;
            }

            let row: Row = self
                .columns
                .iter()
                .enumerate()
                .map(|(col_idx, col)| {
                    if self.null_bitmaps[col_idx][idx] {
                        ColumnValue::Null
                    } else {
                        col.get(idx)
                    }
                })
                .collect();

            if predicate(&row) {
                results.push((RowId::new(idx as u64), row));
            }
        }

        results
    }

    fn create_index(&mut self, column: &str) -> bool {
        if self.hash_indexes.contains_key(column) {
            return false;
        }

        let col_idx = match self.schema.column_index(column) {
            Some(idx) => idx,
            None => return false,
        };

        // Only support int indexes for now
        if self.schema.columns[col_idx].col_type != ColumnType::Int {
            return false;
        }

        let mut index = BTreeMap::new();

        if let ColumnStorage::Int(values) = &self.columns[col_idx] {
            for (idx, value) in values.iter().enumerate() {
                if self.alive[idx] {
                    index
                        .entry(*value)
                        .or_insert_with(Vec::new)
                        .push(RowId::new(idx as u64));
                }
            }
        }

        self.hash_indexes.insert(column.to_string(), index);
        true
    }

    fn index_lookup(&self, column: &str, key: i64) -> Vec<RowId> {
        self.hash_indexes
            .get(column)
            .and_then(|idx| idx.get(&key))
            .map(|rows| {
                rows.iter()
                    .filter(|r| self.alive[r.as_u64() as usize])
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn index_range(&self, column: &str, op: RangeOp, key: i64) -> Vec<RowId> {
        let Some(index) = self.hash_indexes.get(column) else {
            return Vec::new();
        };

        let alive = &self.alive;
        let filter_alive = |rows: &Vec<RowId>| -> Vec<RowId> {
            rows.iter()
                .filter(|r| alive[r.as_u64() as usize])
                .copied()
                .collect()
        };

        let mut result = Vec::new();

        match op {
            RangeOp::Lt => {
                for (_, rows) in index.range(..key) {
                    result.extend(filter_alive(rows));
                }
            },
            RangeOp::Le => {
                for (_, rows) in index.range(..=key) {
                    result.extend(filter_alive(rows));
                }
            },
            RangeOp::Gt => {
                for (_, rows) in
                    index.range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
                {
                    result.extend(filter_alive(rows));
                }
            },
            RangeOp::Ge => {
                for (_, rows) in index.range(key..) {
                    result.extend(filter_alive(rows));
                }
            },
        }

        result
    }

    fn index_between(&self, column: &str, min: i64, max: i64) -> Vec<RowId> {
        let Some(index) = self.hash_indexes.get(column) else {
            return Vec::new();
        };

        let mut result = Vec::new();
        for (_, rows) in index.range(min..=max) {
            for row_id in rows {
                if self.alive[row_id.as_u64() as usize] {
                    result.push(*row_id);
                }
            }
        }
        result
    }
}

/// Columnar storage for relational tables.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` for concurrent access.
pub struct RelationalSlab {
    tables: RwLock<BTreeMap<String, TableStorage>>,
    next_table_id: AtomicU64,
}

impl RelationalSlab {
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            next_table_id: AtomicU64::new(0),
        }
    }

    /// Create a new table.
    pub fn create_table(&self, name: &str, schema: TableSchema) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        if tables.contains_key(name) {
            return Err(RelationalError::TableExists(name.to_string()));
        }
        tables.insert(name.to_string(), TableStorage::new(schema));
        self.next_table_id.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Drop a table.
    pub fn drop_table(&self, name: &str) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        if tables.remove(name).is_none() {
            return Err(RelationalError::TableNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Check if a table exists.
    pub fn table_exists(&self, name: &str) -> bool {
        self.tables.read().contains_key(name)
    }

    /// Get table schema.
    pub fn get_schema(&self, name: &str) -> Option<TableSchema> {
        self.tables.read().get(name).map(|t| t.schema.clone())
    }

    /// Insert a row.
    pub fn insert(&self, table: &str, row: Row) -> Result<RowId, RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if row.len() != storage.schema.columns.len() {
            return Err(RelationalError::ColumnMismatch {
                expected: storage.schema.columns.len(),
                actual: row.len(),
            });
        }

        Ok(storage.insert(&row))
    }

    /// Batch insert rows.
    pub fn batch_insert(&self, table: &str, rows: Vec<Row>) -> Result<Vec<RowId>, RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let mut row_ids = Vec::with_capacity(rows.len());
        for row in rows {
            if row.len() != storage.schema.columns.len() {
                return Err(RelationalError::ColumnMismatch {
                    expected: storage.schema.columns.len(),
                    actual: row.len(),
                });
            }
            row_ids.push(storage.insert(&row));
        }

        Ok(row_ids)
    }

    /// Get a row by ID.
    pub fn get(&self, table: &str, row_id: RowId) -> Result<Option<Row>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.get(row_id))
    }

    /// Delete a row.
    pub fn delete(&self, table: &str, row_id: RowId) -> Result<bool, RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.delete(row_id))
    }

    /// Scan a table with a predicate.
    pub fn scan<F>(&self, table: &str, predicate: F) -> Result<Vec<(RowId, Row)>, RelationalError>
    where
        F: Fn(&Row) -> bool,
    {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.scan(predicate))
    }

    /// Get all rows from a table.
    pub fn scan_all(&self, table: &str) -> Result<Vec<(RowId, Row)>, RelationalError> {
        self.scan(table, |_| true)
    }

    /// Create an index on a column.
    pub fn create_index(&self, table: &str, column: &str) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if !storage.create_index(column) {
            return Err(RelationalError::IndexCreationFailed(column.to_string()));
        }

        Ok(())
    }

    /// Look up rows by indexed column.
    pub fn index_lookup(
        &self,
        table: &str,
        column: &str,
        key: i64,
    ) -> Result<Vec<RowId>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.index_lookup(column, key))
    }

    /// Range query on indexed column using B-tree's O(log n) range operations.
    pub fn index_range(
        &self,
        table: &str,
        column: &str,
        op: RangeOp,
        key: i64,
    ) -> Result<Vec<RowId>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.index_range(column, op, key))
    }

    /// Range query for values between min and max (inclusive).
    pub fn index_between(
        &self,
        table: &str,
        column: &str,
        min: i64,
        max: i64,
    ) -> Result<Vec<RowId>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.index_between(column, min, max))
    }

    /// Get the number of live rows in a table.
    pub fn row_count(&self, table: &str) -> Result<usize, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        Ok(storage.live_rows)
    }

    /// Get the number of tables.
    pub fn table_count(&self) -> usize {
        self.tables.read().len()
    }

    /// Get all table names.
    pub fn table_names(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    /// Clear all tables.
    pub fn clear(&self) {
        self.tables.write().clear();
    }

    /// Get serializable state for snapshots.
    pub fn snapshot(&self) -> RelationalSlabSnapshot {
        RelationalSlabSnapshot {
            tables: self.tables.read().clone(),
        }
    }

    /// Restore from a snapshot.
    pub fn restore(snapshot: RelationalSlabSnapshot) -> Self {
        Self {
            tables: RwLock::new(snapshot.tables),
            next_table_id: AtomicU64::new(0),
        }
    }
}

impl Default for RelationalSlab {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from RelationalSlab operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationalError {
    TableNotFound(String),
    TableExists(String),
    ColumnMismatch { expected: usize, actual: usize },
    IndexCreationFailed(String),
}

impl std::fmt::Display for RelationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableNotFound(name) => write!(f, "table not found: {}", name),
            Self::TableExists(name) => write!(f, "table already exists: {}", name),
            Self::ColumnMismatch { expected, actual } => {
                write!(f, "column mismatch: expected {}, got {}", expected, actual)
            },
            Self::IndexCreationFailed(col) => {
                write!(f, "index creation failed for column: {}", col)
            },
        }
    }
}

impl std::error::Error for RelationalError {}

/// Serializable snapshot of RelationalSlab state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalSlabSnapshot {
    tables: BTreeMap<String, TableStorage>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    fn create_test_schema() -> TableSchema {
        TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("name", ColumnType::String, true),
            ColumnDef::new("age", ColumnType::Int, true),
        ])
    }

    fn create_test_row(id: i64, name: &str, age: i64) -> Row {
        vec![
            ColumnValue::Int(id),
            ColumnValue::String(name.to_string()),
            ColumnValue::Int(age),
        ]
    }

    #[test]
    fn test_new() {
        let slab = RelationalSlab::new();
        assert_eq!(slab.table_count(), 0);
    }

    #[test]
    fn test_create_table() {
        let slab = RelationalSlab::new();
        let schema = create_test_schema();

        slab.create_table("users", schema).unwrap();

        assert!(slab.table_exists("users"));
        assert_eq!(slab.table_count(), 1);
    }

    #[test]
    fn test_create_duplicate_table() {
        let slab = RelationalSlab::new();
        let schema = create_test_schema();

        slab.create_table("users", schema.clone()).unwrap();
        let result = slab.create_table("users", schema);

        assert!(matches!(result, Err(RelationalError::TableExists(_))));
    }

    #[test]
    fn test_drop_table() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.drop_table("users").unwrap();

        assert!(!slab.table_exists("users"));
    }

    #[test]
    fn test_drop_nonexistent_table() {
        let slab = RelationalSlab::new();
        let result = slab.drop_table("users");

        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn test_insert_get() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row = create_test_row(1, "Alice", 30);
        let row_id = slab.insert("users", row.clone()).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved, row);
    }

    #[test]
    fn test_insert_column_mismatch() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let bad_row = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
        ];
        let result = slab.insert("users", bad_row);

        assert!(matches!(
            result,
            Err(RelationalError::ColumnMismatch { .. })
        ));
    }

    #[test]
    fn test_batch_insert() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let rows = vec![
            create_test_row(1, "Alice", 30),
            create_test_row(2, "Bob", 25),
            create_test_row(3, "Carol", 35),
        ];

        let row_ids = slab.batch_insert("users", rows).unwrap();
        assert_eq!(row_ids.len(), 3);
        assert_eq!(slab.row_count("users").unwrap(), 3);
    }

    #[test]
    fn test_delete() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 1);

        slab.delete("users", row_id).unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 0);

        let retrieved = slab.get("users", row_id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_scan() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20 + i))
                .unwrap();
        }

        // Scan for age > 25
        let results = slab
            .scan(
                "users",
                |row| matches!(&row[2], ColumnValue::Int(age) if *age > 25),
            )
            .unwrap();

        assert_eq!(results.len(), 4); // ages 26, 27, 28, 29
    }

    #[test]
    fn test_scan_all() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..5 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20 + i))
                .unwrap();
        }

        let results = slab.scan_all("users").unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_create_index() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_lookup("users", "id", 5).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_index_lookup_after_delete() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.create_index("users", "id").unwrap();

        // Before delete
        let results = slab.index_lookup("users", "id", 1).unwrap();
        assert_eq!(results.len(), 1);

        // After delete
        slab.delete("users", row_id).unwrap();
        let results = slab.index_lookup("users", "id", 1).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_get_schema() {
        let slab = RelationalSlab::new();
        let schema = create_test_schema();
        slab.create_table("users", schema.clone()).unwrap();

        let retrieved = slab.get_schema("users").unwrap();
        assert_eq!(retrieved.columns.len(), 3);
    }

    #[test]
    fn test_table_names() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();
        slab.create_table("posts", create_test_schema()).unwrap();

        let names = slab.table_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"users".to_string()));
        assert!(names.contains(&"posts".to_string()));
    }

    #[test]
    fn test_clear() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();
        slab.create_table("posts", create_test_schema()).unwrap();

        slab.clear();

        assert_eq!(slab.table_count(), 0);
    }

    #[test]
    fn test_snapshot_restore() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();
        slab.insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        let snapshot = slab.snapshot();
        let restored = RelationalSlab::restore(snapshot);

        assert!(restored.table_exists("users"));
        assert_eq!(restored.row_count("users").unwrap(), 1);
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let slab = Arc::new(RelationalSlab::new());
        slab.create_table("users", create_test_schema()).unwrap();

        let mut handles = vec![];

        // Writer threads
        for t in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let _ = s.insert(
                        "users",
                        create_test_row(t * 1000 + i, &format!("User{}", i), 20),
                    );
                }
            }));
        }

        // Reader threads
        for _ in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = s.row_count("users");
                    let _ = s.scan_all("users");
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(slab.row_count("users").unwrap(), 400);
    }

    #[test]
    fn test_no_resize_stall() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let count = 10_000;
        let start = Instant::now();
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count as i64 {
            let op_start = Instant::now();
            let _ = slab.insert("users", create_test_row(i, &format!("User{}", i), 20));
            let op_time = op_start.elapsed();
            if op_time > max_op_time {
                max_op_time = op_time;
            }
        }

        let total_time = start.elapsed();

        assert!(
            max_op_time.as_millis() < 50,
            "Max operation time {:?} exceeded 10ms threshold",
            max_op_time
        );

        let ops_per_sec = count as f64 / total_time.as_secs_f64();
        assert!(
            ops_per_sec > 5_000.0, // Lower threshold accounts for coverage overhead
            "Throughput {:.0} ops/sec too low",
            ops_per_sec
        );
    }

    #[test]
    fn test_null_values() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row = vec![ColumnValue::Int(1), ColumnValue::Null, ColumnValue::Null];
        let row_id = slab.insert("users", row.clone()).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved[1], ColumnValue::Null);
        assert_eq!(retrieved[2], ColumnValue::Null);
    }

    #[test]
    fn test_error_display() {
        let err = RelationalError::TableNotFound("users".to_string());
        assert!(err.to_string().contains("users"));

        let err = RelationalError::ColumnMismatch {
            expected: 3,
            actual: 2,
        };
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn test_default() {
        let slab = RelationalSlab::default();
        assert_eq!(slab.table_count(), 0);
    }

    #[test]
    fn test_column_value_is_null() {
        assert!(ColumnValue::Null.is_null());
        assert!(!ColumnValue::Int(1).is_null());
    }

    #[test]
    fn test_column_def_new() {
        let def = ColumnDef::new("name", ColumnType::String, true);
        assert_eq!(def.name, "name");
        assert_eq!(def.col_type, ColumnType::String);
        assert!(def.nullable);
    }

    #[test]
    fn test_table_schema_column_index() {
        let schema = create_test_schema();
        assert_eq!(schema.column_index("id"), Some(0));
        assert_eq!(schema.column_index("name"), Some(1));
        assert_eq!(schema.column_index("nonexistent"), None);
    }

    #[test]
    fn test_row_id_from_u64() {
        let row_id: RowId = 42u64.into();
        assert_eq!(row_id.as_u64(), 42);
    }

    #[test]
    fn test_with_primary_key() {
        let schema = TableSchema::new(vec![ColumnDef::new("id", ColumnType::Int, false)])
            .with_primary_key("id");

        assert_eq!(schema.primary_key, Some("id".to_string()));
    }

    #[test]
    fn test_float_and_bool_columns() {
        let schema = TableSchema::new(vec![
            ColumnDef::new("score", ColumnType::Float, false),
            ColumnDef::new("active", ColumnType::Bool, false),
        ]);

        let slab = RelationalSlab::new();
        slab.create_table("data", schema).unwrap();

        // Insert row with float and bool
        let row = vec![ColumnValue::Float(3.14), ColumnValue::Bool(true)];
        let row_id = slab.insert("data", row).unwrap();

        // Retrieve and verify
        let retrieved = slab.get("data", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Float(3.14));
        assert_eq!(retrieved[1], ColumnValue::Bool(true));
    }

    #[test]
    fn test_float_and_bool_null_values() {
        let schema = TableSchema::new(vec![
            ColumnDef::new("score", ColumnType::Float, true),
            ColumnDef::new("active", ColumnType::Bool, true),
        ]);

        let slab = RelationalSlab::new();
        slab.create_table("data", schema).unwrap();

        // Insert row with null float and bool
        let row = vec![ColumnValue::Null, ColumnValue::Null];
        let row_id = slab.insert("data", row).unwrap();

        // Retrieve and verify nulls
        let retrieved = slab.get("data", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Null);
        assert_eq!(retrieved[1], ColumnValue::Null);
    }

    #[test]
    fn test_index_update_on_insert() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Create index BEFORE inserting
        slab.create_index("users", "id").unwrap();

        // Now insert - this triggers the index update code path
        let row_id = slab
            .insert("users", create_test_row(100, "Alice", 30))
            .unwrap();

        // Verify index works
        let results = slab.index_lookup("users", "id", 100).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], row_id);
    }

    #[test]
    fn test_batch_insert_column_mismatch() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let bad_rows = vec![
            create_test_row(1, "Alice", 30),
            vec![ColumnValue::Int(2)], // Wrong number of columns
        ];

        let result = slab.batch_insert("users", bad_rows);
        assert!(matches!(
            result,
            Err(RelationalError::ColumnMismatch { .. })
        ));
    }

    #[test]
    fn test_create_index_on_non_int_column() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Try to create index on string column (not supported)
        let result = slab.create_index("users", "name");
        assert!(matches!(
            result,
            Err(RelationalError::IndexCreationFailed(_))
        ));
    }

    #[test]
    fn test_create_index_on_nonexistent_column() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let result = slab.create_index("users", "nonexistent");
        assert!(matches!(
            result,
            Err(RelationalError::IndexCreationFailed(_))
        ));
    }

    #[test]
    fn test_create_duplicate_index() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.create_index("users", "id").unwrap();
        let result = slab.create_index("users", "id");
        assert!(matches!(
            result,
            Err(RelationalError::IndexCreationFailed(_))
        ));
    }

    #[test]
    fn test_scan_with_deleted_rows() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Insert multiple rows
        let id1 = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        let _id2 = slab.insert("users", create_test_row(2, "Bob", 25)).unwrap();
        let id3 = slab
            .insert("users", create_test_row(3, "Carol", 35))
            .unwrap();

        // Delete middle row
        slab.delete("users", id1).unwrap();

        // Scan should skip deleted row
        let results = slab.scan_all("users").unwrap();
        assert_eq!(results.len(), 2);

        // Verify we got the right rows (ids 2 and 3)
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        assert!(ids.contains(&1)); // row_id 1 is the second insert
        assert!(ids.contains(&2)); // row_id 2 is the third insert

        // Verify row_id 0 (first insert) is not in results
        assert!(!ids.contains(&(id1.as_u64()))); // id1 is 0
        assert!(ids.contains(&(id3.as_u64() - 1))); // id2 is 1
    }

    #[test]
    fn test_delete_already_deleted() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Delete once
        assert!(slab.delete("users", row_id).unwrap());

        // Delete again - should return false
        assert!(!slab.delete("users", row_id).unwrap());
    }

    #[test]
    fn test_delete_out_of_bounds() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Try to delete row that was never inserted
        assert!(!slab.delete("users", RowId::new(999)).unwrap());
    }

    #[test]
    fn test_get_deleted_row() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.delete("users", row_id).unwrap();

        // Get deleted row should return None
        assert!(slab.get("users", row_id).unwrap().is_none());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Get row that was never inserted
        assert!(slab.get("users", RowId::new(999)).unwrap().is_none());
    }

    #[test]
    fn test_error_display_all_variants() {
        let e1 = RelationalError::TableNotFound("t".to_string());
        assert!(e1.to_string().contains("not found"));

        let e2 = RelationalError::TableExists("t".to_string());
        assert!(e2.to_string().contains("already exists"));

        let e3 = RelationalError::ColumnMismatch {
            expected: 3,
            actual: 2,
        };
        assert!(e3.to_string().contains("mismatch"));

        let e4 = RelationalError::IndexCreationFailed("c".to_string());
        assert!(e4.to_string().contains("index creation failed"));
    }

    #[test]
    fn test_scan_with_null_columns() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Insert row with null values
        let row = vec![ColumnValue::Int(1), ColumnValue::Null, ColumnValue::Null];
        slab.insert("users", row).unwrap();

        // Scan and verify nulls are preserved
        let results = slab.scan_all("users").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1[1], ColumnValue::Null);
        assert_eq!(results[0].1[2], ColumnValue::Null);
    }

    #[test]
    fn test_column_value_variants() {
        assert!(!ColumnValue::Float(1.0).is_null());
        assert!(!ColumnValue::Bool(true).is_null());
        assert!(!ColumnValue::String("test".to_string()).is_null());
    }

    #[test]
    fn test_index_range_lt() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Lt, 5).unwrap();
        assert_eq!(results.len(), 5); // ids 0, 1, 2, 3, 4
    }

    #[test]
    fn test_index_range_le() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Le, 5).unwrap();
        assert_eq!(results.len(), 6); // ids 0, 1, 2, 3, 4, 5
    }

    #[test]
    fn test_index_range_gt() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Gt, 5).unwrap();
        assert_eq!(results.len(), 4); // ids 6, 7, 8, 9
    }

    #[test]
    fn test_index_range_ge() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Ge, 5).unwrap();
        assert_eq!(results.len(), 5); // ids 5, 6, 7, 8, 9
    }

    #[test]
    fn test_index_between() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_between("users", "id", 3, 7).unwrap();
        assert_eq!(results.len(), 5); // ids 3, 4, 5, 6, 7
    }

    #[test]
    fn test_index_range_with_deleted_rows() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        // Delete some rows
        slab.delete("users", RowId::new(3)).unwrap();
        slab.delete("users", RowId::new(5)).unwrap();

        let results = slab.index_range("users", "id", RangeOp::Lt, 7).unwrap();
        assert_eq!(results.len(), 5); // ids 0, 1, 2, 4, 6 (3, 5 deleted)
    }

    #[test]
    fn test_index_range_no_index() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..5 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        // No index created
        let results = slab.index_range("users", "id", RangeOp::Lt, 3).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_index_range_performance() {
        let slab = RelationalSlab::new();
        slab.create_table("perf", create_test_schema()).unwrap();

        let count = 10_000;
        for i in 0..count {
            slab.insert("perf", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("perf", "id").unwrap();

        let start = Instant::now();
        let results = slab.index_range("perf", "id", RangeOp::Lt, 100).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 100);
        assert!(
            elapsed.as_micros() < 5000,
            "Range query took {:?}, expected < 5ms",
            elapsed
        );
    }
}
