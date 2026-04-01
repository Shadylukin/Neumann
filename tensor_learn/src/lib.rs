// SPDX-License-Identifier: MIT OR Apache-2.0
//! Geometric intelligence model with hyperbolic codebook learning.
//!
//! This crate provides a Poincare disk-based codebook where entries are
//! graph nodes with hierarchical relationships and hyperbolic embeddings.
//! The hierarchy is naturally captured by the hyperbolic geometry: nodes
//! closer to the origin represent more general concepts, while nodes
//! near the boundary represent more specific ones.
//!
//! # Architecture
//!
//! - [`matrix::Matrix8x8`] -- 8x8 transformation matrices
//! - [`poincare::PoincarePoint`] -- points in the Poincare disk model
//! - [`codebook::Codebook`] -- graph + vector backed codebook
//! - [`training::TrainingSession`] -- iterative training with loss tracking
//! - [`viz::VizData`] -- visualization export for the dashboard

pub mod codebook;
pub mod error;
pub mod grokking;
pub mod matrix;
pub mod poincare;
pub mod riemannian;
pub mod task;
pub mod training;
pub mod viz;

pub use codebook::Codebook;
pub use error::LearnError;
pub use grokking::{GrokConfig, GrokSession, GrokStats};
pub use matrix::Matrix8x8;
pub use poincare::PoincarePoint;
pub use riemannian::{euclidean_grad_distance, riemannian_grad, riemannian_update};
pub use task::HierarchyTask;
pub use training::{TrainingConfig, TrainingSession, TrainingStats, TrainingStatus};
pub use viz::VizData;
