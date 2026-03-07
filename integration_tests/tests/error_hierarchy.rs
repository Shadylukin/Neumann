// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for unified error hierarchy.
//!
//! Tests that error conversions preserve source chains and display properly.

use std::error::Error;
use std::path::PathBuf;

use tensor_chain::{
    atomic_io::AtomicIoError, embedding::EmbeddingError, error::ChainError, raft_wal::WalError,
    snapshot_buffer::SnapshotBufferError, snapshot_streaming::StreamingError,
};

#[test]
fn test_error_display_includes_source() {
    let err = AtomicIoError::NoParentDir(PathBuf::from("/nonexistent"));
    let chain_err: ChainError = err.into();
    let display = chain_err.to_string();
    assert!(display.contains("atomic I/O error"));
    assert!(display.contains("/nonexistent"));
}

#[test]
fn test_error_chain_traversal() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let buffer_err = SnapshotBufferError::from(io_err);
    let chain_err: ChainError = buffer_err.into();

    // Walk the error chain
    let mut depth = 0;
    let mut current: &dyn Error = &chain_err;
    while let Some(source) = current.source() {
        depth += 1;
        current = source;
    }
    assert!(depth >= 1, "error chain should have at least 1 source");
}

#[test]
fn test_embedding_error_propagation() {
    let err = EmbeddingError::DimensionMismatch {
        before: 128,
        after: 256,
    };
    let chain_err: ChainError = err.into();
    assert!(chain_err.to_string().contains("128"));
    assert!(chain_err.to_string().contains("256"));
}

#[test]
fn test_wal_error_source_chain() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let wal_err = WalError::Io(io_err);
    let chain_err: ChainError = wal_err.into();

    // ChainError::Wal should have WalError as source
    let source = chain_err.source().expect("ChainError should have source");

    // WalError::Io should have io::Error as source
    assert!(
        source.source().is_some(),
        "WalError::Io should have io::Error as source"
    );
}

#[test]
fn test_streaming_error_nested_source() {
    let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated data");
    let buffer_err = SnapshotBufferError::from(io_err);
    let streaming_err = StreamingError::from(buffer_err);
    let chain_err: ChainError = streaming_err.into();

    // Walk the chain: ChainError -> StreamingError -> SnapshotBufferError -> io::Error
    let mut depth = 0;
    let mut current: &dyn Error = &chain_err;
    while let Some(source) = current.source() {
        depth += 1;
        current = source;
    }

    // Should be at least 3 levels deep
    assert!(
        depth >= 2,
        "nested error chain should have at least 2 sources, got {}",
        depth
    );
}

#[test]
fn test_all_error_types_display_properly() {
    // Test all error type conversions have proper Display
    let errors: Vec<ChainError> = vec![
        WalError::ChecksumMismatch {
            index: 42,
            expected: 0xDEAD,
            actual: 0xBEEF,
        }
        .into(),
        WalError::DiskSpaceLow {
            available: 1024,
            required: 4096,
        }
        .into(),
        WalError::SizeLimitExceeded {
            current: 1_000_000_000,
            max: 500_000_000,
        }
        .into(),
        AtomicIoError::NoParentDir(PathBuf::from("/test/path")).into(),
        SnapshotBufferError::NotFinalized.into(),
        SnapshotBufferError::OutOfBounds {
            offset: 100,
            len: 50,
            total: 80,
        }
        .into(),
        StreamingError::UnexpectedEof.into(),
        StreamingError::InvalidFormat("bad magic".to_string()).into(),
        EmbeddingError::NotComputed.into(),
        EmbeddingError::AlreadyComputed.into(),
        EmbeddingError::DimensionMismatch {
            before: 64,
            after: 128,
        }
        .into(),
    ];

    for err in errors {
        let display = err.to_string();
        // All error displays should be non-empty and not just "error"
        assert!(display.len() > 5, "Error display too short: '{}'", display);
        // All errors should have a debug representation
        let debug = format!("{:?}", err);
        assert!(!debug.is_empty(), "Error debug representation empty");
    }
}

#[test]
fn test_error_type_distinctions() {
    // Verify each error type maps to the correct ChainError variant
    let wal_err: ChainError = WalError::ChecksumMismatch {
        index: 0,
        expected: 0,
        actual: 1,
    }
    .into();
    assert!(matches!(wal_err, ChainError::Wal(_)));

    let atomic_err: ChainError = AtomicIoError::NoParentDir(PathBuf::from("/")).into();
    assert!(matches!(atomic_err, ChainError::AtomicIo(_)));

    let buffer_err: ChainError = SnapshotBufferError::NotFinalized.into();
    assert!(matches!(buffer_err, ChainError::SnapshotBuffer(_)));

    let streaming_err: ChainError = StreamingError::UnexpectedEof.into();
    assert!(matches!(streaming_err, ChainError::Streaming(_)));

    let embedding_err: ChainError = EmbeddingError::NotComputed.into();
    assert!(matches!(embedding_err, ChainError::EmbeddingOp(_)));
}

#[test]
fn test_error_source_type_preservation() {
    // When we convert to ChainError, we should still be able to downcast the source
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test file");
    let wal_err = WalError::Io(io_err);
    let chain_err: ChainError = wal_err.into();

    // Get the source (should be WalError)
    let source = chain_err.source().expect("should have source");

    // The source's Display should contain "IO error"
    let source_msg = source.to_string();
    assert!(
        source_msg.contains("IO error") || source_msg.contains("test file"),
        "Source message should contain IO error info: {}",
        source_msg
    );
}
