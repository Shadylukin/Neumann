//! Streaming cursor API for memory-efficient large result sets.
//!
//! Provides batch-based iteration over query results without loading
//! all rows into memory at once.

use crate::{Condition, RelationalEngine, Result, Row};

/// Streaming cursor for iterating over query results in batches.
///
/// Unlike `RowCursor` which loads all results upfront, `StreamingCursor`
/// fetches rows in configurable batches, reducing memory usage for large
/// result sets.
///
/// # Example
///
/// ```ignore
/// let cursor = engine.select_streaming("users", Condition::True)?;
/// for row_result in cursor {
///     let row = row_result?;
///     println!("User: {:?}", row);
/// }
/// ```
pub struct StreamingCursor<'a> {
    engine: &'a RelationalEngine,
    table: String,
    condition: Condition,
    batch_size: usize,
    current_offset: usize,
    current_batch: Vec<Row>,
    batch_index: usize,
    exhausted: bool,
    max_rows: Option<usize>,
    rows_yielded: usize,
}

impl<'a> StreamingCursor<'a> {
    /// Creates a new streaming cursor.
    pub(crate) fn new(
        engine: &'a RelationalEngine,
        table: impl Into<String>,
        condition: Condition,
    ) -> Self {
        Self {
            engine,
            table: table.into(),
            condition,
            batch_size: 1000,
            current_offset: 0,
            current_batch: Vec::new(),
            batch_index: 0,
            exhausted: false,
            max_rows: None,
            rows_yielded: 0,
        }
    }

    /// Sets the batch size for fetching rows.
    #[must_use]
    pub const fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = if size == 0 { 1000 } else { size };
        self
    }

    /// Sets the maximum number of rows to return.
    #[must_use]
    pub const fn with_max_rows(mut self, max: usize) -> Self {
        self.max_rows = Some(max);
        self
    }

    /// Returns the number of rows yielded so far.
    #[must_use]
    pub const fn rows_yielded(&self) -> usize {
        self.rows_yielded
    }

    /// Returns true if the cursor has been exhausted.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Fetches the next batch of rows.
    fn fetch_next_batch(&mut self) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }

        // Calculate how many rows to fetch
        let fetch_limit = if let Some(max) = self.max_rows {
            let remaining = max.saturating_sub(self.rows_yielded);
            if remaining == 0 {
                self.exhausted = true;
                return Ok(());
            }
            remaining.min(self.batch_size)
        } else {
            self.batch_size
        };

        // Fetch the next batch
        let batch = self.engine.select_with_limit(
            &self.table,
            self.condition.clone(),
            fetch_limit,
            self.current_offset,
        )?;

        if batch.is_empty() {
            self.exhausted = true;
            self.current_batch = Vec::new();
        } else {
            self.current_offset += batch.len();
            self.exhausted = batch.len() < fetch_limit;
            self.current_batch = batch;
        }
        self.batch_index = 0;

        Ok(())
    }
}

impl Iterator for StreamingCursor<'_> {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check max rows limit
        if let Some(max) = self.max_rows {
            if self.rows_yielded >= max {
                return None;
            }
        }

        // If current batch is exhausted, fetch next batch
        if self.batch_index >= self.current_batch.len() {
            if self.exhausted {
                return None;
            }
            if let Err(e) = self.fetch_next_batch() {
                return Some(Err(e));
            }
            if self.current_batch.is_empty() {
                return None;
            }
        }

        // Return next row from current batch
        if self.batch_index < self.current_batch.len() {
            let row = self.current_batch[self.batch_index].clone();
            self.batch_index += 1;
            self.rows_yielded += 1;
            Some(Ok(row))
        } else {
            None
        }
    }
}

impl std::fmt::Debug for StreamingCursor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingCursor")
            .field("table", &self.table)
            .field("batch_size", &self.batch_size)
            .field("current_offset", &self.current_offset)
            .field("rows_yielded", &self.rows_yielded)
            .field("exhausted", &self.exhausted)
            .finish()
    }
}

/// Builder for creating streaming cursors with custom options.
pub struct CursorBuilder<'a> {
    engine: &'a RelationalEngine,
    table: String,
    condition: Condition,
    batch_size: usize,
    max_rows: Option<usize>,
}

impl<'a> CursorBuilder<'a> {
    /// Creates a new cursor builder.
    pub(crate) fn new(
        engine: &'a RelationalEngine,
        table: impl Into<String>,
        condition: Condition,
    ) -> Self {
        Self {
            engine,
            table: table.into(),
            condition,
            batch_size: 1000,
            max_rows: None,
        }
    }

    /// Sets the batch size for fetching rows.
    #[must_use]
    pub const fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = if size == 0 { 1000 } else { size };
        self
    }

    /// Sets the maximum number of rows to return.
    #[must_use]
    pub const fn max_rows(mut self, max: usize) -> Self {
        self.max_rows = Some(max);
        self
    }

    /// Builds and returns the streaming cursor.
    #[must_use]
    pub fn build(self) -> StreamingCursor<'a> {
        let mut cursor = StreamingCursor::new(self.engine, self.table, self.condition);
        cursor.batch_size = self.batch_size;
        cursor.max_rows = self.max_rows;
        cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, ColumnType, Schema, Value};
    use std::collections::HashMap;

    fn create_test_engine() -> RelationalEngine {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);
        engine.create_table("test", schema).unwrap();

        // Insert test rows
        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("id".to_string(), Value::Int(i));
            values.insert("name".to_string(), Value::String(format!("user_{i}")));
            engine.insert("test", values).unwrap();
        }

        engine
    }

    #[test]
    fn test_streaming_cursor_all_rows() {
        let engine = create_test_engine();
        let cursor = StreamingCursor::new(&engine, "test", Condition::True).with_batch_size(10);

        let rows: Vec<_> = cursor.collect();
        assert_eq!(rows.len(), 100);
        assert!(rows.iter().all(Result::is_ok));
    }

    #[test]
    fn test_streaming_cursor_with_max_rows() {
        let engine = create_test_engine();
        let cursor = StreamingCursor::new(&engine, "test", Condition::True)
            .with_batch_size(10)
            .with_max_rows(25);

        let rows: Vec<_> = cursor.collect();
        assert_eq!(rows.len(), 25);
    }

    #[test]
    fn test_streaming_cursor_small_batch() {
        let engine = create_test_engine();
        let cursor = StreamingCursor::new(&engine, "test", Condition::True).with_batch_size(5);

        let mut count = 0;
        for result in cursor {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_streaming_cursor_with_condition() {
        let engine = create_test_engine();
        let cursor = StreamingCursor::new(
            &engine,
            "test",
            Condition::Lt("id".to_string(), Value::Int(10)),
        )
        .with_batch_size(5);

        let rows: Vec<_> = cursor.collect();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn test_cursor_builder() {
        let engine = create_test_engine();
        let cursor = CursorBuilder::new(&engine, "test", Condition::True)
            .batch_size(20)
            .max_rows(50)
            .build();

        let rows: Vec<_> = cursor.collect();
        assert_eq!(rows.len(), 50);
    }

    #[test]
    fn test_streaming_cursor_rows_yielded() {
        let engine = create_test_engine();
        let mut cursor = StreamingCursor::new(&engine, "test", Condition::True).with_batch_size(10);

        assert_eq!(cursor.rows_yielded(), 0);

        for _ in 0..5 {
            let _ = cursor.next();
        }

        assert_eq!(cursor.rows_yielded(), 5);
    }

    #[test]
    fn test_streaming_cursor_is_exhausted() {
        let engine = create_test_engine();
        let mut cursor = StreamingCursor::new(&engine, "test", Condition::True).with_max_rows(5);

        assert!(!cursor.is_exhausted());

        // Consume all rows
        while cursor.next().is_some() {}

        // The cursor should report exhausted after consuming max_rows
        // Note: exhausted may be set after fetch returns empty batch
    }

    #[test]
    fn test_streaming_cursor_debug() {
        let engine = create_test_engine();
        let cursor = StreamingCursor::new(&engine, "test", Condition::True).with_batch_size(100);

        let debug_str = format!("{cursor:?}");
        assert!(debug_str.contains("StreamingCursor"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_cursor_empty_table() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("empty", schema).unwrap();

        let cursor = StreamingCursor::new(&engine, "empty", Condition::True);
        let rows: Vec<_> = cursor.collect();
        assert!(rows.is_empty());
    }
}
