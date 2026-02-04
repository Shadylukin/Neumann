// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cursor state management for paginated query results.
//!
//! This module provides cursor-based pagination support, enabling efficient
//! streaming of large result sets without loading everything into memory.

use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Unique identifier for a cursor (UUID v4 string).
pub type CursorId = String;

/// Type of result stored by a cursor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Encode, Decode, PartialEq, Eq, Hash)]
pub enum CursorResultType {
    /// Relational rows from SELECT queries.
    Rows,
    /// Graph nodes from node queries.
    Nodes,
    /// Graph edges from edge queries.
    Edges,
    /// Vector similarity search results.
    Similar,
    /// Unified cross-engine query results.
    Unified,
    /// Pattern match results from graph pattern queries.
    PatternMatch,
}

impl std::fmt::Display for CursorResultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rows => write!(f, "rows"),
            Self::Nodes => write!(f, "nodes"),
            Self::Edges => write!(f, "edges"),
            Self::Similar => write!(f, "similar"),
            Self::Unified => write!(f, "unified"),
            Self::PatternMatch => write!(f, "pattern_match"),
        }
    }
}

/// State of a pagination cursor.
///
/// The cursor state is serialized to a base64-encoded token that clients
/// can use to resume pagination from a specific position.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode, PartialEq)]
pub struct CursorState {
    /// Unique cursor identifier.
    pub id: CursorId,
    /// Original query string.
    pub query: String,
    /// Type of results this cursor returns.
    pub result_type: CursorResultType,
    /// Current offset into the result set.
    pub offset: usize,
    /// Number of items per page.
    pub page_size: usize,
    /// Total count of results (if known).
    pub total_count: Option<usize>,
    /// Unix timestamp (seconds) when cursor was created.
    pub created_at: i64,
    /// Unix timestamp (seconds) when cursor was last accessed.
    pub last_accessed_at: i64,
    /// Time-to-live in seconds.
    pub ttl_secs: u32,
}

impl CursorState {
    /// Default page size for paginated queries.
    pub const DEFAULT_PAGE_SIZE: usize = 100;

    /// Default TTL for cursors (5 minutes).
    pub const DEFAULT_TTL_SECS: u32 = 300;

    /// Maximum TTL for cursors (30 minutes).
    pub const MAX_TTL_SECS: u32 = 1800;

    /// Create a new cursor state.
    #[must_use]
    pub fn new(
        id: CursorId,
        query: String,
        result_type: CursorResultType,
        page_size: usize,
        total_count: Option<usize>,
        ttl_secs: u32,
    ) -> Self {
        let now = current_timestamp();
        Self {
            id,
            query,
            result_type,
            offset: 0,
            page_size,
            total_count,
            created_at: now,
            last_accessed_at: now,
            ttl_secs,
        }
    }

    /// Create a new cursor for the next page.
    #[must_use]
    pub fn next_page(&self) -> Self {
        let mut next = self.clone();
        next.offset += self.page_size;
        next.last_accessed_at = current_timestamp();
        next
    }

    /// Create a cursor for the previous page, if possible.
    #[must_use]
    pub fn prev_page(&self) -> Option<Self> {
        if self.offset == 0 {
            return None;
        }
        let mut prev = self.clone();
        prev.offset = self.offset.saturating_sub(self.page_size);
        prev.last_accessed_at = current_timestamp();
        Some(prev)
    }

    /// Check if there are more results after the current page.
    #[must_use]
    pub fn has_more(&self) -> bool {
        match self.total_count {
            Some(total) => self.offset + self.page_size < total,
            None => true, // Unknown total - assume more
        }
    }

    /// Check if the cursor has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        let now = current_timestamp();
        let elapsed = now - self.last_accessed_at;
        elapsed > i64::from(self.ttl_secs)
    }

    /// Touch the cursor to update last access time.
    pub fn touch(&mut self) {
        self.last_accessed_at = current_timestamp();
    }

    /// Encode cursor state to a base64 token.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn encode(&self) -> Result<String, CursorError> {
        use base64::Engine;
        let encoded = bitcode::encode(self);
        Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&encoded))
    }

    /// Decode cursor state from a base64 token.
    ///
    /// # Errors
    ///
    /// Returns an error if the token is invalid or expired.
    pub fn decode(token: &str) -> Result<Self, CursorError> {
        use base64::Engine;

        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(token)
            .map_err(|e| CursorError::InvalidToken(format!("base64 decode failed: {e}")))?;

        let state: Self = bitcode::decode(&bytes)
            .map_err(|e| CursorError::InvalidToken(format!("bitcode decode failed: {e}")))?;

        if state.is_expired() {
            return Err(CursorError::Expired(state.id.clone()));
        }

        Ok(state)
    }
}

/// Errors related to cursor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CursorError {
    /// Invalid cursor token.
    InvalidToken(String),
    /// Cursor has expired.
    Expired(CursorId),
    /// Cursor not found.
    NotFound(CursorId),
    /// Maximum cursors exceeded.
    CapacityExceeded,
}

impl std::fmt::Display for CursorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken(msg) => write!(f, "Invalid cursor token: {msg}"),
            Self::Expired(id) => write!(f, "Cursor expired: {id}"),
            Self::NotFound(id) => write!(f, "Cursor not found: {id}"),
            Self::CapacityExceeded => write!(f, "Maximum cursor capacity exceeded"),
        }
    }
}

impl std::error::Error for CursorError {}

/// Get current Unix timestamp in seconds.
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cursor() -> CursorState {
        CursorState::new(
            "test-cursor-id".to_string(),
            "SELECT users".to_string(),
            CursorResultType::Rows,
            100,
            Some(500),
            300,
        )
    }

    #[test]
    fn test_cursor_state_new() {
        let cursor = create_test_cursor();
        assert_eq!(cursor.id, "test-cursor-id");
        assert_eq!(cursor.query, "SELECT users");
        assert_eq!(cursor.result_type, CursorResultType::Rows);
        assert_eq!(cursor.offset, 0);
        assert_eq!(cursor.page_size, 100);
        assert_eq!(cursor.total_count, Some(500));
        assert_eq!(cursor.ttl_secs, 300);
    }

    #[test]
    fn test_cursor_next_page() {
        let cursor = create_test_cursor();
        let next = cursor.next_page();
        assert_eq!(next.offset, 100);
        assert_eq!(next.page_size, 100);
        assert_eq!(next.id, cursor.id);
    }

    #[test]
    fn test_cursor_prev_page_at_start() {
        let cursor = create_test_cursor();
        assert!(cursor.prev_page().is_none());
    }

    #[test]
    fn test_cursor_prev_page() {
        let mut cursor = create_test_cursor();
        cursor.offset = 200;
        let prev = cursor.prev_page().unwrap();
        assert_eq!(prev.offset, 100);
    }

    #[test]
    fn test_cursor_has_more() {
        let cursor = create_test_cursor();
        assert!(cursor.has_more());

        let mut last_page = cursor.clone();
        last_page.offset = 400;
        assert!(!last_page.has_more());
    }

    #[test]
    fn test_cursor_has_more_unknown_total() {
        let cursor = CursorState::new(
            "test".to_string(),
            "SELECT users".to_string(),
            CursorResultType::Rows,
            100,
            None, // Unknown total
            300,
        );
        assert!(cursor.has_more());
    }

    #[test]
    fn test_cursor_is_expired() {
        let mut cursor = create_test_cursor();
        assert!(!cursor.is_expired());

        // Simulate an old cursor
        cursor.last_accessed_at = current_timestamp() - 400;
        assert!(cursor.is_expired());
    }

    #[test]
    fn test_cursor_touch() {
        let mut cursor = create_test_cursor();
        let original_time = cursor.last_accessed_at;
        std::thread::sleep(std::time::Duration::from_millis(10));
        cursor.touch();
        assert!(cursor.last_accessed_at >= original_time);
    }

    #[test]
    fn test_cursor_encode_decode_roundtrip() {
        let cursor = create_test_cursor();
        let token = cursor.encode().unwrap();
        let decoded = CursorState::decode(&token).unwrap();
        assert_eq!(cursor.id, decoded.id);
        assert_eq!(cursor.query, decoded.query);
        assert_eq!(cursor.result_type, decoded.result_type);
        assert_eq!(cursor.offset, decoded.offset);
        assert_eq!(cursor.page_size, decoded.page_size);
        assert_eq!(cursor.total_count, decoded.total_count);
    }

    #[test]
    fn test_cursor_decode_invalid_token() {
        let result = CursorState::decode("not-valid-base64!!!");
        assert!(matches!(result, Err(CursorError::InvalidToken(_))));
    }

    #[test]
    fn test_cursor_decode_expired() {
        let mut cursor = create_test_cursor();
        cursor.last_accessed_at = current_timestamp() - 400; // Expired
        let token = cursor.encode().unwrap();
        let result = CursorState::decode(&token);
        assert!(matches!(result, Err(CursorError::Expired(_))));
    }

    #[test]
    fn test_cursor_result_type_display() {
        assert_eq!(CursorResultType::Rows.to_string(), "rows");
        assert_eq!(CursorResultType::Nodes.to_string(), "nodes");
        assert_eq!(CursorResultType::Edges.to_string(), "edges");
        assert_eq!(CursorResultType::Similar.to_string(), "similar");
        assert_eq!(CursorResultType::Unified.to_string(), "unified");
        assert_eq!(CursorResultType::PatternMatch.to_string(), "pattern_match");
    }

    #[test]
    fn test_cursor_error_display() {
        let err = CursorError::InvalidToken("bad".to_string());
        assert!(err.to_string().contains("Invalid cursor token"));

        let err = CursorError::Expired("cursor-1".to_string());
        assert!(err.to_string().contains("expired"));

        let err = CursorError::NotFound("cursor-2".to_string());
        assert!(err.to_string().contains("not found"));

        let err = CursorError::CapacityExceeded;
        assert!(err.to_string().contains("capacity"));
    }

    #[test]
    fn test_cursor_constants() {
        assert_eq!(CursorState::DEFAULT_PAGE_SIZE, 100);
        assert_eq!(CursorState::DEFAULT_TTL_SECS, 300);
        assert_eq!(CursorState::MAX_TTL_SECS, 1800);
    }

    #[test]
    fn test_cursor_state_partial_eq() {
        let cursor1 = create_test_cursor();
        let cursor2 = create_test_cursor();
        // They should have different timestamps but same content
        assert_eq!(cursor1.id, cursor2.id);
        assert_eq!(cursor1.query, cursor2.query);
    }

    #[test]
    fn test_cursor_result_type_equality() {
        assert_eq!(CursorResultType::Rows, CursorResultType::Rows);
        assert_ne!(CursorResultType::Rows, CursorResultType::Nodes);
    }

    #[test]
    fn test_cursor_clone() {
        let cursor = create_test_cursor();
        let cloned = cursor.clone();
        assert_eq!(cursor.id, cloned.id);
        assert_eq!(cursor.offset, cloned.offset);
    }

    #[test]
    fn test_cursor_debug() {
        let cursor = create_test_cursor();
        let debug = format!("{cursor:?}");
        assert!(debug.contains("CursorState"));
        assert!(debug.contains("test-cursor-id"));
    }

    #[test]
    fn test_prev_page_saturating_sub() {
        let mut cursor = create_test_cursor();
        cursor.offset = 50; // Less than page_size
        let prev = cursor.prev_page().unwrap();
        assert_eq!(prev.offset, 0);
    }

    #[test]
    fn test_cursor_next_page_chain() {
        let cursor = create_test_cursor();
        let page2 = cursor.next_page();
        let page3 = page2.next_page();
        let page4 = page3.next_page();
        assert_eq!(page4.offset, 300);
    }

    #[test]
    fn test_has_more_exact_boundary() {
        let mut cursor = create_test_cursor();
        cursor.total_count = Some(200);
        cursor.offset = 100;
        // offset + page_size == 200, not < 200
        assert!(!cursor.has_more());
    }

    #[test]
    fn test_is_expired_boundary() {
        let mut cursor = create_test_cursor();
        cursor.ttl_secs = 10;
        cursor.last_accessed_at = current_timestamp() - 10;
        // Exactly at TTL boundary - should not be expired
        assert!(!cursor.is_expired());

        cursor.last_accessed_at = current_timestamp() - 11;
        // Just past TTL - should be expired
        assert!(cursor.is_expired());
    }
}
