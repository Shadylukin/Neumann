#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::error::Error;
use std::path::PathBuf;

#[derive(Arbitrary, Debug)]
enum FuzzErrorKind {
    WalChecksum {
        index: u64,
        expected: u32,
        actual: u32,
    },
    WalDiskSpace {
        available: u64,
        required: u64,
    },
    WalSizeLimit {
        current: u64,
        max: u64,
    },
    AtomicNoParent {
        path: String,
    },
    SnapshotOutOfBounds {
        offset: u64,
        len: usize,
        total: u64,
    },
    SnapshotNotFinalized,
    EmbeddingNotComputed,
    EmbeddingAlreadyComputed,
    EmbeddingDimMismatch {
        before: usize,
        after: usize,
    },
    StreamingUnexpectedEof,
    StreamingInvalidFormat {
        msg: String,
    },
}

fuzz_target!(|kind: FuzzErrorKind| {
    use tensor_chain::{
        atomic_io::AtomicIoError, embedding::EmbeddingError, error::ChainError, raft_wal::WalError,
        snapshot_buffer::SnapshotBufferError, snapshot_streaming::StreamingError,
    };

    let chain_err: ChainError = match kind {
        FuzzErrorKind::WalChecksum {
            index,
            expected,
            actual,
        } => WalError::ChecksumMismatch {
            index,
            expected,
            actual,
        }
        .into(),
        FuzzErrorKind::WalDiskSpace {
            available,
            required,
        } => WalError::DiskSpaceLow {
            available,
            required,
        }
        .into(),
        FuzzErrorKind::WalSizeLimit { current, max } => {
            WalError::SizeLimitExceeded { current, max }.into()
        },
        FuzzErrorKind::AtomicNoParent { path } => {
            AtomicIoError::NoParentDir(PathBuf::from(path)).into()
        },
        FuzzErrorKind::SnapshotOutOfBounds { offset, len, total } => {
            SnapshotBufferError::OutOfBounds { offset, len, total }.into()
        },
        FuzzErrorKind::SnapshotNotFinalized => SnapshotBufferError::NotFinalized.into(),
        FuzzErrorKind::EmbeddingNotComputed => EmbeddingError::NotComputed.into(),
        FuzzErrorKind::EmbeddingAlreadyComputed => EmbeddingError::AlreadyComputed.into(),
        FuzzErrorKind::EmbeddingDimMismatch { before, after } => {
            EmbeddingError::DimensionMismatch { before, after }.into()
        },
        FuzzErrorKind::StreamingUnexpectedEof => StreamingError::UnexpectedEof.into(),
        FuzzErrorKind::StreamingInvalidFormat { msg } => StreamingError::InvalidFormat(msg).into(),
    };

    // Property 1: Display works without panicking
    let display = chain_err.to_string();
    assert!(!display.is_empty(), "Display should not be empty");

    // Property 2: Debug works without panicking
    let debug = format!("{:?}", chain_err);
    assert!(!debug.is_empty(), "Debug should not be empty");

    // Property 3: source() doesn't panic
    let _ = chain_err.source();

    // Property 4: Error chain is traversable
    let mut current: &dyn Error = &chain_err;
    let mut depth = 0;
    while let Some(source) = current.source() {
        depth += 1;
        current = source;
        // Guard against infinite loops (shouldn't happen, but safety)
        if depth > 100 {
            break;
        }
    }

    // Property 5: All nested sources have valid Display
    let mut current: &dyn Error = &chain_err;
    while let Some(source) = current.source() {
        let source_display = source.to_string();
        assert!(
            !source_display.is_empty(),
            "Source display should not be empty"
        );
        current = source;
    }
});
