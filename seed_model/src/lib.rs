//! Seed Model: Pure Geometric Intelligence
//!
//! Implementation of the Tensor Scaling Theory - pure geometric, no RNN.
//!
//! ## Core Principles
//!
//! 1. **No forced zeros**: Shape comes from data, not pre-allocated containers
//! 2. **No arbitrary constants**: Depth, dimensions, sizes emerge from learning
//! 3. **Adaptive compute**: Terminate on convergence, not after N steps
//! 4. **Response IS the shape**: No translation layer needed
//!
//! ## Architecture
//!
//! ```text
//! tokens → SparseEmbed → GeometricVQ → logits
//!              |              |
//!         SparseVector   values ARE vocab distributions
//! ```
//!
//! ## What's Emergent
//!
//! - **Depth**: Stops when residual is 90% sparse
//! - **Temperature**: Learned per level
//! - **Effective dimensions**: Sparse storage = only active dims matter
//!
//! ## Usage
//!
//! ```rust,ignore
//! use seed_model::{GeometricConfig, PureGeometricModel};
//!
//! let config = GeometricConfig::from_vocab(vocab_size);
//! let mut model = PureGeometricModel::new(&config);
//!
//! let (logits, depths, sparsities) = model.forward(&tokens);
//! let loss = model.train_step(&tokens, &targets, 0.01);
//! ```

pub mod corpus;
pub mod sparse_model;

pub use sparse_model::{
    GeometricConfig, GeometricContext, GeometricVQLevel, PureGeometricModel, SparseEmbeddingTable,
};
