// SPDX-License-Identifier: MIT OR Apache-2.0
//! Style system for terminal output.
//!
//! Provides themes (color palettes) and icons for consistent, visually appealing output.

mod icons;
mod theme;

pub use icons::Icons;
pub use theme::{styled, Theme};
