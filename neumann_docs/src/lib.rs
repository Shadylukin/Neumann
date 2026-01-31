// SPDX-License-Identifier: MIT OR Apache-2.0
//! Neumann Documentation Storage System
//!
//! Index and search Neumann documentation using all three engines:
//! - Relational engine for metadata
//! - Graph engine for document relationships
//! - Vector engine for semantic search

#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

pub mod commands;
pub mod embeddings;
pub mod indexer;
pub mod markdown;

pub use commands::{execute_command, Command};
pub use embeddings::TfIdfEmbedder;
pub use indexer::{DocIndexer, IndexStats};
pub use markdown::{extract_links, parse_markdown, ParsedDoc};
