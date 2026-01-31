// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tensor Rust Organism (TRO) - Web Living Border
//!
//! Web-based implementation of the TRO living border system, providing
//! Canvas-rendered Physarum simulation with glowing particle effects
//! and real-time activity pulses via WebSocket.

mod activity;

pub use activity::{ActivityPulse, OpType, TroMessage};

use serde::{Deserialize, Serialize};

/// Configuration for the web TRO system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroConfig {
    /// Enable TRO border animation.
    pub enabled: bool,
    /// Target frames per second (default: 30 for smooth web animation).
    pub fps: u32,
    /// Number of Physarum agents (default: 500 for web performance).
    pub agent_count: usize,
    /// Color palette to use.
    pub palette: Palette,
    /// Enable CRT effects (scanlines, glow).
    pub crt_effects: bool,
    /// Particle trail length (0-20).
    pub trail_length: u8,
    /// Glow intensity (0.0-1.0).
    pub glow_intensity: f32,
}

impl Default for TroConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fps: 30,
            agent_count: 500,
            palette: Palette::PhosphorGreen,
            crt_effects: true,
            trail_length: 10,
            glow_intensity: 0.7,
        }
    }
}

impl TroConfig {
    /// Creates a minimal config for low-resource environments.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            fps: 15,
            agent_count: 200,
            palette: Palette::PhosphorGreen,
            crt_effects: false,
            trail_length: 5,
            glow_intensity: 0.4,
        }
    }

    /// Creates a high-performance config for powerful devices.
    #[must_use]
    pub fn high_performance() -> Self {
        Self {
            enabled: true,
            fps: 60,
            agent_count: 1000,
            palette: Palette::PhosphorGreen,
            crt_effects: true,
            trail_length: 15,
            glow_intensity: 0.9,
        }
    }
}

/// Color palettes for the TRO border.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Palette {
    /// Classic phosphor green (#00ee00).
    #[default]
    PhosphorGreen,
    /// Warm amber (#ffb641).
    Amber,
    /// Blood rust red (#942222).
    Rust,
    /// Cool cyan ghost (#00bbcc).
    Ghost,
    /// Glitch purple (#aa44aa).
    Glitch,
}

impl Palette {
    /// Returns the CSS color string for this palette's primary color.
    #[must_use]
    pub const fn primary_css(&self) -> &'static str {
        match self {
            Self::PhosphorGreen => "#00ee00",
            Self::Amber => "#ffb641",
            Self::Rust => "#942222",
            Self::Ghost => "#00bbcc",
            Self::Glitch => "#aa44aa",
        }
    }

    /// Returns the CSS color string for this palette's dim color.
    #[must_use]
    pub const fn dim_css(&self) -> &'static str {
        match self {
            Self::PhosphorGreen => "#008e00",
            Self::Amber => "#cc8800",
            Self::Rust => "#621f22",
            Self::Ghost => "#007788",
            Self::Glitch => "#660066",
        }
    }

    /// Returns the CSS color string for this palette's dark color.
    #[must_use]
    pub const fn dark_css(&self) -> &'static str {
        match self {
            Self::PhosphorGreen => "#005f00",
            Self::Amber => "#663300",
            Self::Rust => "#4a2125",
            Self::Ghost => "#003344",
            Self::Glitch => "#440044",
        }
    }

    /// Returns RGBA components (0-255) for the primary color.
    #[must_use]
    pub const fn primary_rgba(&self) -> (u8, u8, u8, u8) {
        match self {
            Self::PhosphorGreen => (0, 238, 0, 255),
            Self::Amber => (255, 182, 65, 255),
            Self::Rust => (148, 34, 34, 255),
            Self::Ghost => (0, 187, 204, 255),
            Self::Glitch => (170, 68, 170, 255),
        }
    }

    /// Returns the name of this palette.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::PhosphorGreen => "phosphor_green",
            Self::Amber => "amber",
            Self::Rust => "rust",
            Self::Ghost => "ghost",
            Self::Glitch => "glitch",
        }
    }
}

/// Physarum configuration for the web simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysarumConfig {
    /// Sensor angle offset (radians).
    pub sensor_angle: f32,
    /// Distance ahead to sense pheromone.
    pub sensor_distance: f32,
    /// Maximum rotation per step (radians).
    pub rotation_angle: f32,
    /// Movement speed (pixels per step).
    pub speed: f32,
    /// Amount of pheromone deposited per step.
    pub deposit_amount: f32,
    /// Pheromone decay rate per step (0.0-1.0).
    pub decay_rate: f32,
    /// Diffusion rate (spread to neighbors).
    pub diffusion_rate: f32,
}

impl Default for PhysarumConfig {
    fn default() -> Self {
        Self {
            sensor_angle: std::f32::consts::PI / 4.0,
            sensor_distance: 9.0,
            rotation_angle: std::f32::consts::PI / 6.0,
            speed: 2.0,
            deposit_amount: 0.5,
            decay_rate: 0.95,
            diffusion_rate: 0.2,
        }
    }
}

/// State snapshot for syncing with client-side simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroState {
    /// Current pheromone field (intensity 0.0-1.0 per border cell).
    pub pheromone: Vec<f32>,
    /// Activity heat from recent operations.
    pub activity_heat: Vec<f32>,
    /// Current palette.
    pub palette: Palette,
    /// Whether in glitch mode.
    pub in_glitch: bool,
    /// Frame number (for animation sync).
    pub frame: u64,
}

impl TroState {
    /// Creates a new empty state.
    #[must_use]
    pub fn new(border_length: usize) -> Self {
        Self {
            pheromone: vec![0.0; border_length],
            activity_heat: vec![0.0; border_length],
            palette: Palette::PhosphorGreen,
            in_glitch: false,
            frame: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tro_config_default() {
        let config = TroConfig::default();
        assert!(config.enabled);
        assert_eq!(config.fps, 30);
        assert_eq!(config.agent_count, 500);
        assert!(config.crt_effects);
    }

    #[test]
    fn test_tro_config_minimal() {
        let config = TroConfig::minimal();
        assert!(config.enabled);
        assert_eq!(config.fps, 15);
        assert_eq!(config.agent_count, 200);
        assert!(!config.crt_effects);
    }

    #[test]
    fn test_tro_config_high_performance() {
        let config = TroConfig::high_performance();
        assert!(config.enabled);
        assert_eq!(config.fps, 60);
        assert_eq!(config.agent_count, 1000);
        assert!(config.crt_effects);
    }

    #[test]
    fn test_palette_css_colors() {
        let green = Palette::PhosphorGreen;
        assert_eq!(green.primary_css(), "#00ee00");
        assert_eq!(green.dim_css(), "#008e00");
        assert_eq!(green.dark_css(), "#005f00");

        let amber = Palette::Amber;
        assert_eq!(amber.primary_css(), "#ffb641");
    }

    #[test]
    fn test_palette_rgba() {
        let green = Palette::PhosphorGreen;
        assert_eq!(green.primary_rgba(), (0, 238, 0, 255));

        let rust = Palette::Rust;
        assert_eq!(rust.primary_rgba(), (148, 34, 34, 255));
    }

    #[test]
    fn test_palette_name() {
        assert_eq!(Palette::PhosphorGreen.name(), "phosphor_green");
        assert_eq!(Palette::Amber.name(), "amber");
        assert_eq!(Palette::Rust.name(), "rust");
        assert_eq!(Palette::Ghost.name(), "ghost");
        assert_eq!(Palette::Glitch.name(), "glitch");
    }

    #[test]
    fn test_physarum_config_default() {
        let config = PhysarumConfig::default();
        assert!(config.sensor_angle > 0.0);
        assert!(config.decay_rate > 0.0);
        assert!(config.decay_rate <= 1.0);
    }

    #[test]
    fn test_tro_state_new() {
        let state = TroState::new(100);
        assert_eq!(state.pheromone.len(), 100);
        assert_eq!(state.activity_heat.len(), 100);
        assert!(!state.in_glitch);
        assert_eq!(state.frame, 0);
    }

    #[test]
    fn test_palette_default() {
        let p = Palette::default();
        assert_eq!(p, Palette::PhosphorGreen);
    }

    #[test]
    fn test_config_serialization() {
        let config = TroConfig::default();
        let json = serde_json::to_string(&config).expect("serialization failed");
        assert!(json.contains("enabled"));
        assert!(json.contains("fps"));

        let decoded: TroConfig = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.fps, config.fps);
    }

    #[test]
    fn test_palette_serialization() {
        let palette = Palette::Amber;
        let json = serde_json::to_string(&palette).expect("serialization failed");
        assert!(json.contains("amber"));

        let decoded: Palette = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded, Palette::Amber);
    }
}
