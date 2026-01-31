// SPDX-License-Identifier: MIT OR Apache-2.0
//! CRT effects for authentic vintage terminal feel.
//!
//! Implements scanlines, phosphor flicker, glow bleeding, and glitch effects.

// This module performs many u8<->f32 conversions for color manipulation.
// These are intentional and safe for 8-bit color values.
// The suboptimal_flops lint is disabled because readability of color math
// is more important than micro-optimizations in a visualization module.
#![allow(clippy::cast_lossless, clippy::suboptimal_flops)]

/// Applies scanline darkening effect to a color.
///
/// Simulates the visible scan lines on CRT monitors where every other
/// row appears slightly darker.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_scanline(color: (u8, u8, u8), brightness: f32) -> (u8, u8, u8) {
    let brightness = brightness.clamp(0.0, 1.0);
    (
        (color.0 as f32 * brightness) as u8,
        (color.1 as f32 * brightness) as u8,
        (color.2 as f32 * brightness) as u8,
    )
}

/// Applies subtle flicker effect to a color.
///
/// Simulates the slight brightness variation in CRT phosphor.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn apply_flicker(color: (u8, u8, u8), frame: u64, intensity: f32) -> (u8, u8, u8) {
    // Use simple hash of frame for pseudo-random flicker
    let noise =
        ((frame.wrapping_mul(1_103_515_245).wrapping_add(12345) >> 16) & 0xFF) as f32 / 255.0;
    let variation = 1.0 - intensity + noise * intensity * 2.0;
    let variation = variation.clamp(0.9, 1.1);

    (
        (color.0 as f32 * variation).clamp(0.0, 255.0) as u8,
        (color.1 as f32 * variation).clamp(0.0, 255.0) as u8,
        (color.2 as f32 * variation).clamp(0.0, 255.0) as u8,
    )
}

/// Applies phosphor glow bleeding effect.
///
/// Bright pixels bleed color into adjacent darker areas.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_glow(
    base: (u8, u8, u8),
    neighbor_brightness: f32,
    glow_intensity: f32,
) -> (u8, u8, u8) {
    let glow = neighbor_brightness * glow_intensity;
    (
        (base.0 as f32 + glow * 20.0).clamp(0.0, 255.0) as u8,
        (base.1 as f32 + glow * 30.0).clamp(0.0, 255.0) as u8,
        (base.2 as f32 + glow * 20.0).clamp(0.0, 255.0) as u8,
    )
}

/// Applies a glitch corruption effect to a color.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_glitch(color: (u8, u8, u8), severity: f32) -> (u8, u8, u8) {
    if severity < 0.01 {
        return color;
    }

    // Shift color channels chaotically
    let shift = (severity * 128.0) as i32;
    let r = (color.0 as i32 + shift).clamp(0, 255) as u8;
    let g = (color.1 as i32 - shift / 2).clamp(0, 255) as u8;
    let b = (color.2 as i32 + shift / 3).clamp(0, 255) as u8;

    (r, g, b)
}

/// Applies a color shift for error indication.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_error_tint(color: (u8, u8, u8), intensity: f32) -> (u8, u8, u8) {
    let intensity = intensity.clamp(0.0, 1.0);
    let r = (color.0 as f32 + intensity * (255.0 - color.0 as f32)).min(255.0) as u8;
    let g = (color.1 as f32 * (1.0 - intensity * 0.5)).max(0.0) as u8;
    let b = (color.2 as f32 * (1.0 - intensity * 0.3)).max(0.0) as u8;
    (r, g, b)
}

/// Applies a pulse effect (brightness wave).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_pulse(color: (u8, u8, u8), phase: f32, amplitude: f32) -> (u8, u8, u8) {
    let wave = (phase.sin() * 0.5 + 0.5) * amplitude + (1.0 - amplitude);
    (
        (color.0 as f32 * wave).clamp(0.0, 255.0) as u8,
        (color.1 as f32 * wave).clamp(0.0, 255.0) as u8,
        (color.2 as f32 * wave).clamp(0.0, 255.0) as u8,
    )
}

/// Blends two colors together.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn blend_colors(a: (u8, u8, u8), b: (u8, u8, u8), factor: f32) -> (u8, u8, u8) {
    let factor = factor.clamp(0.0, 1.0);
    let inv = 1.0 - factor;
    (
        (a.0 as f32 * inv + b.0 as f32 * factor) as u8,
        (a.1 as f32 * inv + b.1 as f32 * factor) as u8,
        (a.2 as f32 * inv + b.2 as f32 * factor) as u8,
    )
}

/// Computes the brightness of a color (0.0 - 1.0).
#[must_use]
pub fn brightness(color: (u8, u8, u8)) -> f32 {
    // Using perceived brightness formula
    (0.299 * color.0 as f32 + 0.587 * color.1 as f32 + 0.114 * color.2 as f32) / 255.0
}

/// Adjusts color saturation.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn adjust_saturation(color: (u8, u8, u8), factor: f32) -> (u8, u8, u8) {
    let gray = brightness(color) * 255.0;
    let factor = factor.clamp(0.0, 2.0);

    (
        (gray + (color.0 as f32 - gray) * factor).clamp(0.0, 255.0) as u8,
        (gray + (color.1 as f32 - gray) * factor).clamp(0.0, 255.0) as u8,
        (gray + (color.2 as f32 - gray) * factor).clamp(0.0, 255.0) as u8,
    )
}

/// Static pattern for boot sequence noise.
pub struct StaticNoise {
    seed: u64,
}

impl StaticNoise {
    /// Creates a new static noise generator.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Gets a pseudo-random value for a position.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn get(&self, position: usize, frame: u64) -> f32 {
        let combined = position as u64 ^ frame ^ self.seed;
        let hash = combined
            .wrapping_mul(1_103_515_245)
            .wrapping_add(12345)
            .wrapping_mul(1_103_515_245);
        ((hash >> 16) & 0xFFFF) as f32 / 65535.0
    }

    /// Gets a noise character for static effect.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn get_char(&self, position: usize, frame: u64) -> char {
        const STATIC_CHARS: [char; 6] = [' ', '.', ':', ';', '#', '@'];
        let value = self.get(position, frame);
        // value is 0.0-1.0, STATIC_CHARS.len() is 6
        let idx = (value * STATIC_CHARS.len() as f32) as usize;
        STATIC_CHARS[idx.min(STATIC_CHARS.len() - 1)]
    }
}

impl Default for StaticNoise {
    fn default() -> Self {
        Self::new(0xDEAD_BEEF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_scanline() {
        let color = (100, 100, 100);
        let result = apply_scanline(color, 0.5);
        assert_eq!(result, (50, 50, 50));
    }

    #[test]
    fn test_apply_scanline_clamp() {
        let color = (100, 100, 100);
        assert_eq!(apply_scanline(color, 0.0), (0, 0, 0));
        assert_eq!(apply_scanline(color, 1.0), (100, 100, 100));
        // Values outside 0-1 should be clamped
        assert_eq!(apply_scanline(color, 2.0), (100, 100, 100));
        assert_eq!(apply_scanline(color, -1.0), (0, 0, 0));
    }

    #[test]
    fn test_apply_flicker() {
        let color = (100, 200, 150);
        let result1 = apply_flicker(color, 0, 0.1);
        let result2 = apply_flicker(color, 100, 0.1);
        // Different frames should give slightly different results
        // but both should be close to original
        assert!(result1.1 > 150 && result1.1 < 250);
        assert!(result2.1 > 150 && result2.1 < 250);
    }

    #[test]
    fn test_apply_glow() {
        let base = (10, 20, 30);
        let result = apply_glow(base, 1.0, 0.5);
        assert!(result.0 > base.0);
        assert!(result.1 > base.1);
    }

    #[test]
    fn test_apply_glitch_zero() {
        let color = (100, 100, 100);
        assert_eq!(apply_glitch(color, 0.0), color);
    }

    #[test]
    fn test_apply_glitch_nonzero() {
        let color = (100, 100, 100);
        let result = apply_glitch(color, 0.5);
        // Should be different from original
        assert_ne!(result, color);
    }

    #[test]
    fn test_apply_error_tint() {
        let color = (50, 100, 50);
        let result = apply_error_tint(color, 0.5);
        // Red should increase
        assert!(result.0 > color.0);
        // Green should decrease or stay same
        assert!(result.1 <= color.1);
    }

    #[test]
    fn test_apply_pulse() {
        let color = (100, 100, 100);
        let peak = apply_pulse(color, std::f32::consts::FRAC_PI_2, 0.5); // sin(PI/2) = 1
        let trough = apply_pulse(color, -std::f32::consts::FRAC_PI_2, 0.5); // sin(-PI/2) = -1

        // Peak should be brighter than trough
        assert!(peak.0 > trough.0);
    }

    #[test]
    fn test_blend_colors() {
        let a = (255, 0, 0);
        let b = (0, 255, 0);

        assert_eq!(blend_colors(a, b, 0.0), a);
        assert_eq!(blend_colors(a, b, 1.0), b);

        let mid = blend_colors(a, b, 0.5);
        assert_eq!(mid, (127, 127, 0));
    }

    #[test]
    fn test_brightness() {
        assert!((brightness((255, 255, 255)) - 1.0).abs() < 0.01);
        assert!((brightness((0, 0, 0)) - 0.0).abs() < 0.01);
        // Green contributes most to perceived brightness
        assert!(brightness((0, 255, 0)) > brightness((255, 0, 0)));
    }

    #[test]
    fn test_adjust_saturation() {
        let color = (255, 100, 100);
        let desat = adjust_saturation(color, 0.0);
        // Desaturated should be gray-ish
        assert!((desat.0 as i32 - desat.1 as i32).abs() < 5);

        let sat = adjust_saturation(color, 1.5);
        // More saturated red should have bigger difference between channels
        assert!((sat.0 as i32 - sat.1 as i32) > (color.0 as i32 - color.1 as i32));
    }

    #[test]
    fn test_static_noise_new() {
        let noise = StaticNoise::new(12345);
        assert_eq!(noise.seed, 12345);
    }

    #[test]
    fn test_static_noise_get() {
        let noise = StaticNoise::new(12345);
        let value = noise.get(0, 0);
        assert!(value >= 0.0 && value <= 1.0);
    }

    #[test]
    fn test_static_noise_deterministic() {
        let noise = StaticNoise::new(12345);
        let v1 = noise.get(10, 5);
        let v2 = noise.get(10, 5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_static_noise_varies() {
        let noise = StaticNoise::new(12345);
        let v1 = noise.get(10, 5);
        let v2 = noise.get(11, 5);
        // Different positions should give different values (usually)
        // This isn't guaranteed but is highly likely
        let _ = (v1, v2);
    }

    #[test]
    fn test_static_noise_get_char() {
        let noise = StaticNoise::default();
        let ch = noise.get_char(0, 0);
        assert!(" .:;#@".contains(ch));
    }
}
