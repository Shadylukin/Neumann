// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-domain execution helpers for [`crate::QueryRouter`].
//!
//! Each submodule owns a single statement family (SQL, graph, vector, blob,
//! etc.). All helpers are `pub(crate) fn` free functions taking
//! `&crate::QueryRouter` as their first parameter; the router is reduced to a
//! thin dispatcher that matches on `StatementKind` and forwards to the right
//! module.

pub mod blob;
pub mod cache;
pub mod chain;
pub mod checkpoint;
pub mod cluster;
pub mod describe;
pub mod expr;
pub mod graph;
pub mod sql;
pub mod unified;
pub mod vault;
pub mod vector;
