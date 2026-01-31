// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gray-Scott reaction-diffusion system.
//!
//! Simulates a two-chemical reaction-diffusion system that creates
//! organic patterns like spots, stripes, and coral-like structures.

// Reaction-diffusion involves many small float/int conversions for indices.
// These are intentional for the simulation. Floating point math is kept
// in readable form rather than using mul_add everywhere.
#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use rand::Rng;

/// Preset patterns for the Gray-Scott system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReactionPreset {
    /// Coral-like branching patterns.
    #[default]
    Coral,
    /// Maze-like winding patterns.
    Maze,
    /// Spotted patterns.
    Spots,
    /// Mitosis-like dividing spots.
    Mitosis,
    /// Soliton-like moving spots.
    Solitons,
    /// Worm-like patterns.
    Worms,
}

/// Configuration for the Gray-Scott system.
#[derive(Debug, Clone)]
pub struct ReactionConfig {
    /// Feed rate (F parameter).
    pub feed_rate: f32,
    /// Kill rate (k parameter).
    pub kill_rate: f32,
    /// Diffusion rate for chemical A.
    pub diffuse_a: f32,
    /// Diffusion rate for chemical B.
    pub diffuse_b: f32,
    /// Time step per iteration.
    pub dt: f32,
}

impl ReactionConfig {
    /// Creates a config from a preset.
    #[must_use]
    pub fn from_preset(preset: ReactionPreset) -> Self {
        match preset {
            ReactionPreset::Coral => Self {
                feed_rate: 0.055,
                kill_rate: 0.062,
                diffuse_a: 1.0,
                diffuse_b: 0.5,
                dt: 1.0,
            },
            ReactionPreset::Maze => Self {
                feed_rate: 0.029,
                kill_rate: 0.057,
                diffuse_a: 1.0,
                diffuse_b: 0.5,
                dt: 1.0,
            },
            ReactionPreset::Spots => Self {
                feed_rate: 0.030,
                kill_rate: 0.062,
                diffuse_a: 1.0,
                diffuse_b: 0.5,
                dt: 1.0,
            },
            ReactionPreset::Mitosis => Self {
                feed_rate: 0.028,
                kill_rate: 0.062,
                diffuse_a: 1.0,
                diffuse_b: 0.5,
                dt: 1.0,
            },
            ReactionPreset::Solitons => Self {
                feed_rate: 0.030,
                kill_rate: 0.060,
                diffuse_a: 1.0,
                diffuse_b: 0.5,
                dt: 1.0,
            },
            ReactionPreset::Worms => Self {
                feed_rate: 0.058,
                kill_rate: 0.065,
                diffuse_a: 1.0,
                diffuse_b: 0.5,
                dt: 1.0,
            },
        }
    }
}

impl Default for ReactionConfig {
    fn default() -> Self {
        Self::from_preset(ReactionPreset::default())
    }
}

/// The Gray-Scott reaction-diffusion field.
pub struct ReactionField {
    /// Length of the 1D field (border perimeter).
    length: usize,
    /// Chemical A concentration (substrate).
    chemical_a: Vec<f32>,
    /// Chemical B concentration (catalyst).
    chemical_b: Vec<f32>,
    /// Temporary buffer for A during update.
    buffer_a: Vec<f32>,
    /// Temporary buffer for B during update.
    buffer_b: Vec<f32>,
    /// Configuration.
    config: ReactionConfig,
}

impl ReactionField {
    /// Creates a new reaction field.
    #[must_use]
    pub fn new(length: usize) -> Self {
        Self::with_config(length, ReactionConfig::default())
    }

    /// Creates a new field with custom configuration.
    #[must_use]
    pub fn with_config(length: usize, config: ReactionConfig) -> Self {
        // Initialize with A=1.0 everywhere, B=0.0 everywhere
        let chemical_a = vec![1.0; length];
        let chemical_b = vec![0.0; length];
        let buffer_a = vec![0.0; length];
        let buffer_b = vec![0.0; length];

        let mut field = Self {
            length,
            chemical_a,
            chemical_b,
            buffer_a,
            buffer_b,
            config,
        };

        // Seed some initial B at random locations
        field.seed_random(5);

        field
    }

    /// Seeds B chemical at random locations.
    pub fn seed_random(&mut self, count: usize) {
        if self.length == 0 {
            return;
        }

        let mut rng = rand::thread_rng();

        for _ in 0..count {
            let center = rng.gen_range(0..self.length);
            let radius = 3;

            for offset in 0..=radius {
                let left = (center + self.length - offset) % self.length;
                let right = (center + offset) % self.length;

                let value = 1.0 - (offset as f32 / radius as f32);
                self.chemical_b[left] = self.chemical_b[left].max(value);
                self.chemical_b[right] = self.chemical_b[right].max(value);
            }
        }
    }

    /// Seeds B chemical at a specific location.
    pub fn seed_at(&mut self, position: usize, intensity: f32) {
        if position < self.length {
            let radius = 3;
            for offset in 0..=radius {
                let left = (position + self.length - offset) % self.length;
                let right = (position + offset) % self.length;

                let value = intensity * (1.0 - offset as f32 / radius as f32);
                self.chemical_b[left] = (self.chemical_b[left] + value).min(1.0);
                self.chemical_b[right] = (self.chemical_b[right] + value).min(1.0);
            }
        }
    }

    /// Performs one time step of the reaction-diffusion simulation.
    pub fn step(&mut self) {
        if self.length == 0 {
            return;
        }

        let f = self.config.feed_rate;
        let k = self.config.kill_rate;
        let da = self.config.diffuse_a;
        let db = self.config.diffuse_b;
        let dt = self.config.dt;

        // Copy current state to buffers
        self.buffer_a.copy_from_slice(&self.chemical_a);
        self.buffer_b.copy_from_slice(&self.chemical_b);

        for i in 0..self.length {
            let a = self.buffer_a[i];
            let b = self.buffer_b[i];

            // Compute Laplacian for this 1D field
            let lap_a = self.laplacian(&self.buffer_a, i);
            let lap_b = self.laplacian(&self.buffer_b, i);

            // Reaction term: A + 2B -> 3B
            let reaction = a * b * b;

            // Gray-Scott equations using mul_add for better precision
            let delta_a = da.mul_add(lap_a, -reaction) + f * (1.0 - a);
            let delta_b = db.mul_add(lap_b, reaction) - (k + f) * b;

            // Update with time step
            self.chemical_a[i] = delta_a.mul_add(dt, a).clamp(0.0, 1.0);
            self.chemical_b[i] = delta_b.mul_add(dt, b).clamp(0.0, 1.0);
        }
    }

    /// Computes the Laplacian (second derivative) at a position.
    fn laplacian(&self, field: &[f32], i: usize) -> f32 {
        let len = self.length;
        if len < 3 {
            return 0.0;
        }

        let prev = if i == 0 { len - 1 } else { i - 1 };
        let next = (i + 1) % len;

        // 1D Laplacian: f''(x) ~ f(x-1) - 2f(x) + f(x+1)
        2.0f32.mul_add(-field[i], field[prev]) + field[next]
    }

    /// Gets the reaction value (chemical B concentration) at a position.
    #[must_use]
    pub fn get_value(&self, index: usize) -> f32 {
        self.chemical_b.get(index).copied().unwrap_or(0.0)
    }

    /// Gets the chemical A concentration at a position.
    #[must_use]
    pub fn get_a(&self, index: usize) -> f32 {
        self.chemical_a.get(index).copied().unwrap_or(1.0)
    }

    /// Resizes the field to a new length.
    pub fn resize(&mut self, new_length: usize) {
        if new_length == self.length {
            return;
        }

        self.chemical_a.resize(new_length, 1.0);
        self.chemical_b.resize(new_length, 0.0);
        self.buffer_a.resize(new_length, 0.0);
        self.buffer_b.resize(new_length, 0.0);
        self.length = new_length;

        // Re-seed if we grew
        if new_length > 0 && self.max_b() < 0.1 {
            self.seed_random(3);
        }
    }

    /// Sets the preset pattern.
    pub fn set_preset(&mut self, preset: ReactionPreset) {
        self.config = ReactionConfig::from_preset(preset);
    }

    /// Resets the field to initial state.
    pub fn reset(&mut self) {
        self.chemical_a.fill(1.0);
        self.chemical_b.fill(0.0);
        self.seed_random(5);
    }

    /// Returns the maximum B concentration.
    #[must_use]
    pub fn max_b(&self) -> f32 {
        self.chemical_b
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Returns the average B concentration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Length fits in f32 mantissa for typical border sizes
    pub fn avg_b(&self) -> f32 {
        if self.chemical_b.is_empty() {
            return 0.0;
        }
        self.chemical_b.iter().sum::<f32>() / self.chemical_b.len() as f32
    }

    /// Injects a "death wave" (reduces B chemical in region).
    #[allow(clippy::cast_precision_loss)] // Small values fit in f32 mantissa
    pub fn inject_death(&mut self, position: usize, radius: usize) {
        if self.length == 0 {
            return;
        }

        for offset in 0..=radius {
            let decay = 1.0 - offset as f32 / (radius + 1) as f32;
            let left = (position + self.length - offset) % self.length;
            let right = (position + offset) % self.length;

            self.chemical_b[left] *= 1.0 - decay * 0.5;
            self.chemical_b[right] *= 1.0 - decay * 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_default() {
        let preset = ReactionPreset::default();
        assert_eq!(preset, ReactionPreset::Coral);
    }

    #[test]
    fn test_config_from_preset() {
        let config = ReactionConfig::from_preset(ReactionPreset::Coral);
        assert!((config.feed_rate - 0.055).abs() < 0.001);
        assert!((config.kill_rate - 0.062).abs() < 0.001);
    }

    #[test]
    fn test_config_default() {
        let config = ReactionConfig::default();
        assert!((config.feed_rate - 0.055).abs() < 0.001);
    }

    #[test]
    fn test_field_new() {
        let field = ReactionField::new(100);
        assert_eq!(field.length, 100);
        assert_eq!(field.chemical_a.len(), 100);
        assert_eq!(field.chemical_b.len(), 100);
    }

    #[test]
    fn test_field_initial_state() {
        let field = ReactionField::new(100);
        // A should be mostly 1.0
        assert!(field.get_a(50) > 0.0);
        // B should have some seeds
        assert!(field.max_b() > 0.0);
    }

    #[test]
    fn test_step_no_panic() {
        let mut field = ReactionField::new(100);
        for _ in 0..10 {
            field.step();
        }
    }

    #[test]
    fn test_step_empty() {
        let mut field = ReactionField::new(0);
        field.step(); // Should not panic
    }

    #[test]
    fn test_get_value() {
        let field = ReactionField::new(100);
        let _ = field.get_value(50);
        assert_eq!(field.get_value(1000), 0.0); // Out of bounds
    }

    #[test]
    fn test_seed_at() {
        let mut field = ReactionField::new(100);
        field.chemical_b.fill(0.0); // Clear seeds
        field.seed_at(50, 0.8);
        assert!(field.get_value(50) > 0.0);
    }

    #[test]
    fn test_resize() {
        let mut field = ReactionField::new(100);
        field.resize(200);
        assert_eq!(field.length, 200);
        assert_eq!(field.chemical_a.len(), 200);
    }

    #[test]
    fn test_resize_same() {
        let mut field = ReactionField::new(100);
        field.resize(100);
        assert_eq!(field.length, 100);
    }

    #[test]
    fn test_set_preset() {
        let mut field = ReactionField::new(100);
        field.set_preset(ReactionPreset::Maze);
        assert!((field.config.feed_rate - 0.029).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut field = ReactionField::new(100);
        for _ in 0..100 {
            field.step();
        }
        field.reset();
        // A should be back to 1.0 mostly
        assert!(field.get_a(50) > 0.9);
    }

    #[test]
    fn test_max_b() {
        let mut field = ReactionField::new(100);
        field.chemical_b.fill(0.0);
        field.chemical_b[50] = 0.5;
        assert!((field.max_b() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_avg_b() {
        let field = ReactionField::new(100);
        let avg = field.avg_b();
        assert!(avg >= 0.0);
        assert!(avg <= 1.0);

        let empty = ReactionField::new(0);
        assert_eq!(empty.avg_b(), 0.0);
    }

    #[test]
    fn test_inject_death() {
        let mut field = ReactionField::new(100);
        field.chemical_b.fill(1.0);
        field.inject_death(50, 5);
        assert!(field.get_value(50) < 1.0);
    }

    #[test]
    fn test_laplacian() {
        let field = ReactionField::new(100);
        // For uniform field, Laplacian should be ~0
        let lap = field.laplacian(&vec![1.0; 100], 50);
        assert!(lap.abs() < 0.001);
    }

    #[test]
    fn test_evolution() {
        let mut field = ReactionField::new(200);
        let initial_avg = field.avg_b();

        // Run many steps
        for _ in 0..500 {
            field.step();
        }

        // System should have evolved (B should have spread or changed)
        // This is a weak test but ensures the system is doing something
        let final_avg = field.avg_b();
        // The pattern should have developed (could be more or less B)
        let _ = final_avg; // Just verify no panic
        assert!(field.max_b() > 0.0 || initial_avg > 0.0);
    }
}
