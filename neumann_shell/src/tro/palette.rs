// SPDX-License-Identifier: MIT OR Apache-2.0
//! CRT phosphor color palettes for TRO rendering.

// Palette interpolation involves many small float/int conversions.
// These are all intentional for color manipulation.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::suboptimal_flops
)]

/// Available color palettes for the TRO display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Palette {
    /// Classic green phosphor (VT100, early terminals).
    #[default]
    PhosphorGreen,
    /// Amber phosphor (Hercules monitors).
    PhosphorAmber,
    /// Blue phosphor (rare vintage monitors).
    PhosphorBlue,
    /// Organic decay (green with brown undertones).
    OrganicDecay,
    /// Heat damage (orange/red highlights).
    HeatDamage,
    /// Glitch entropy (purple/magenta corruption).
    GlitchEntropy,
    /// Memory ghost (cyan traces).
    MemoryGhost,
    /// Web UI phosphor green (#00ee00).
    WebPhosphor,
    /// Web UI amber (#ffb641).
    WebAmber,
    /// Web UI blood rust (#942222).
    WebRust,
}

impl Palette {
    /// Returns the color gradient for this palette (5 colors, low to high intensity).
    #[must_use]
    pub fn get_colors(&self) -> &'static [(u8, u8, u8)] {
        match self {
            Self::PhosphorGreen => &PHOSPHOR_GREEN,
            Self::PhosphorAmber => &PHOSPHOR_AMBER,
            Self::PhosphorBlue => &PHOSPHOR_BLUE,
            Self::OrganicDecay => &ORGANIC_DECAY,
            Self::HeatDamage => &HEAT_DAMAGE,
            Self::GlitchEntropy => &GLITCH_ENTROPY,
            Self::MemoryGhost => &MEMORY_GHOST,
            Self::WebPhosphor => &WEB_PHOSPHOR,
            Self::WebAmber => &WEB_AMBER,
            Self::WebRust => &WEB_RUST,
        }
    }

    /// Returns the base (darkest) color for the palette.
    #[must_use]
    pub fn base_color(&self) -> (u8, u8, u8) {
        self.get_colors()[0]
    }

    /// Returns the peak (brightest) color for the palette.
    #[must_use]
    pub fn peak_color(&self) -> (u8, u8, u8) {
        let colors = self.get_colors();
        colors[colors.len() - 1]
    }

    /// Parses a palette from a string name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "green" | "phosphorgreen" | "phosphor-green" => Some(Self::PhosphorGreen),
            "amber" | "phosphoramber" | "phosphor-amber" => Some(Self::PhosphorAmber),
            "blue" | "phosphorblue" | "phosphor-blue" => Some(Self::PhosphorBlue),
            "organic" | "organicdecay" | "organic-decay" => Some(Self::OrganicDecay),
            "heat" | "heatdamage" | "heat-damage" => Some(Self::HeatDamage),
            "glitch" | "glitchentropy" | "glitch-entropy" => Some(Self::GlitchEntropy),
            "ghost" | "memoryghost" | "memory-ghost" => Some(Self::MemoryGhost),
            "webgreen" | "web-green" | "webphosphor" | "web-phosphor" => Some(Self::WebPhosphor),
            "webamber" | "web-amber" => Some(Self::WebAmber),
            "webrust" | "web-rust" | "rust" => Some(Self::WebRust),
            _ => None,
        }
    }

    /// Returns the name of this palette.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::PhosphorGreen => "green",
            Self::PhosphorAmber => "amber",
            Self::PhosphorBlue => "blue",
            Self::OrganicDecay => "organic",
            Self::HeatDamage => "heat",
            Self::GlitchEntropy => "glitch",
            Self::MemoryGhost => "ghost",
            Self::WebPhosphor => "webgreen",
            Self::WebAmber => "webamber",
            Self::WebRust => "webrust",
        }
    }

    /// Returns all available palette names.
    #[must_use]
    pub const fn all_names() -> &'static [&'static str] {
        &[
            "green", "amber", "blue", "organic", "heat", "glitch", "ghost", "webgreen", "webamber",
            "webrust",
        ]
    }

    /// Interpolates between two colors in the palette by intensity.
    #[must_use]
    pub fn interpolate(&self, intensity: f32) -> (u8, u8, u8) {
        let colors = self.get_colors();
        if colors.is_empty() {
            return (0, 51, 0);
        }

        let intensity = intensity.clamp(0.0, 1.0);
        let scaled = intensity * (colors.len() - 1) as f32;
        let idx = scaled as usize;
        let frac = scaled - idx as f32;

        if idx >= colors.len() - 1 {
            return colors[colors.len() - 1];
        }

        let c1 = colors[idx];
        let c2 = colors[idx + 1];

        (
            lerp_u8(c1.0, c2.0, frac),
            lerp_u8(c1.1, c2.1, frac),
            lerp_u8(c1.2, c2.2, frac),
        )
    }
}

/// Linear interpolation for u8 values.
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

// Base decay:     #001100 -> #003300 -> #006600 -> #00AA00 -> #33FF00
const PHOSPHOR_GREEN: [(u8, u8, u8); 6] = [
    (0x00, 0x11, 0x00), // Deep dark green
    (0x00, 0x33, 0x00), // Dark green
    (0x00, 0x66, 0x00), // Medium green
    (0x00, 0xAA, 0x00), // Bright green
    (0x33, 0xFF, 0x00), // Vivid green
    (0x77, 0xFF, 0x77), // Peak green with glow
];

// Amber phosphor: warm orange tones
const PHOSPHOR_AMBER: [(u8, u8, u8); 6] = [
    (0x11, 0x08, 0x00), // Deep dark amber
    (0x33, 0x1A, 0x00), // Dark amber
    (0x66, 0x33, 0x00), // Medium amber
    (0xAA, 0x55, 0x00), // Bright amber
    (0xFF, 0x99, 0x00), // Vivid amber
    (0xFF, 0xCC, 0x77), // Peak amber with glow
];

// Blue phosphor: cool blue tones
const PHOSPHOR_BLUE: [(u8, u8, u8); 6] = [
    (0x00, 0x08, 0x11), // Deep dark blue
    (0x00, 0x1A, 0x33), // Dark blue
    (0x00, 0x33, 0x66), // Medium blue
    (0x00, 0x55, 0xAA), // Bright blue
    (0x00, 0x99, 0xFF), // Vivid blue
    (0x77, 0xCC, 0xFF), // Peak blue with glow
];

// Organic growth: #445500 -> #667700 -> #558822 -> #449933 -> #33AA44
const ORGANIC_DECAY: [(u8, u8, u8); 6] = [
    (0x22, 0x11, 0x00), // Dark brown base
    (0x44, 0x55, 0x00), // Olive brown
    (0x66, 0x77, 0x00), // Yellow-green
    (0x55, 0x88, 0x22), // Forest green
    (0x44, 0x99, 0x33), // Bright organic
    (0x33, 0xAA, 0x44), // Peak organic
];

// Heat damage: #882200 -> #AA3300 -> #CC5500 -> #EE7700 -> #FFCC88
const HEAT_DAMAGE: [(u8, u8, u8); 6] = [
    (0x44, 0x11, 0x00), // Deep heat
    (0x88, 0x22, 0x00), // Dark heat
    (0xAA, 0x33, 0x00), // Medium heat
    (0xCC, 0x55, 0x00), // Bright heat
    (0xEE, 0x77, 0x00), // Hot orange
    (0xFF, 0xCC, 0x88), // White-hot
];

// Glitch/entropy: #440044 -> #660066 -> #882288 -> #AA44AA -> #CC66CC
const GLITCH_ENTROPY: [(u8, u8, u8); 6] = [
    (0x22, 0x00, 0x22), // Deep glitch
    (0x44, 0x00, 0x44), // Dark glitch
    (0x66, 0x00, 0x66), // Medium glitch
    (0x88, 0x22, 0x88), // Bright glitch
    (0xAA, 0x44, 0xAA), // Vivid glitch
    (0xCC, 0x66, 0xCC), // Peak glitch
];

// Memory ghost: #003344 -> #005566 -> #007788 -> #0099AA -> #00BBCC
const MEMORY_GHOST: [(u8, u8, u8); 6] = [
    (0x00, 0x11, 0x22), // Deep ghost
    (0x00, 0x33, 0x44), // Dark ghost
    (0x00, 0x55, 0x66), // Medium ghost
    (0x00, 0x77, 0x88), // Bright ghost
    (0x00, 0x99, 0xAA), // Vivid ghost
    (0x00, 0xBB, 0xCC), // Peak ghost
];

// Web UI phosphor green (synced with --phosphor-green: #00ee00)
const WEB_PHOSPHOR: [(u8, u8, u8); 6] = [
    (0x00, 0x11, 0x00), // Deep dark
    (0x00, 0x5F, 0x00), // --phosphor-green-dark
    (0x00, 0x8E, 0x00), // --phosphor-green-dim
    (0x00, 0xCC, 0x00), // Medium
    (0x00, 0xEE, 0x00), // --phosphor-green
    (0x77, 0xFF, 0x77), // Glow
];

// Web UI amber (synced with --amber-glow: #ffb641)
const WEB_AMBER: [(u8, u8, u8); 6] = [
    (0x11, 0x08, 0x00), // Deep dark
    (0x33, 0x1A, 0x00), // Dark amber
    (0x66, 0x33, 0x00), // Medium amber
    (0xCC, 0x88, 0x00), // Bright
    (0xFF, 0xB6, 0x41), // --amber-glow
    (0xFF, 0xDD, 0x88), // Peak with glow
];

// Web UI blood rust (synced with --blood-rust: #942222)
const WEB_RUST: [(u8, u8, u8); 6] = [
    (0x1A, 0x0A, 0x0B), // Deep dark
    (0x4A, 0x21, 0x25), // --dark-rust
    (0x62, 0x2F, 0x22), // --corroded-brown
    (0x94, 0x22, 0x22), // --blood-rust
    (0xCC, 0x44, 0x44), // Bright rust
    (0xFF, 0x66, 0x66), // Peak rust
];

/// Active pulse colors (for database activity feedback).
pub const ACTIVE_PULSE: [(u8, u8, u8); 5] = [
    (0x11, 0xDD, 0x11), // Base pulse
    (0x22, 0xEE, 0x22), // Building
    (0x44, 0xFF, 0x44), // Active
    (0x77, 0xFF, 0x77), // Peak
    (0xAA, 0xFF, 0xAA), // Flash
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_default() {
        let p = Palette::default();
        assert_eq!(p, Palette::PhosphorGreen);
    }

    #[test]
    fn test_get_colors() {
        let p = Palette::PhosphorGreen;
        let colors = p.get_colors();
        assert_eq!(colors.len(), 6);
    }

    #[test]
    fn test_base_color() {
        let p = Palette::PhosphorGreen;
        let base = p.base_color();
        assert_eq!(base, (0x00, 0x11, 0x00));
    }

    #[test]
    fn test_peak_color() {
        let p = Palette::PhosphorGreen;
        let peak = p.peak_color();
        assert_eq!(peak, (0x77, 0xFF, 0x77));
    }

    #[test]
    fn test_interpolate_zero() {
        let p = Palette::PhosphorGreen;
        let color = p.interpolate(0.0);
        assert_eq!(color, p.base_color());
    }

    #[test]
    fn test_interpolate_one() {
        let p = Palette::PhosphorGreen;
        let color = p.interpolate(1.0);
        assert_eq!(color, p.peak_color());
    }

    #[test]
    fn test_interpolate_mid() {
        let p = Palette::PhosphorGreen;
        let color = p.interpolate(0.5);
        // Should be somewhere in the middle
        assert!(color.1 > p.base_color().1);
        assert!(color.1 < p.peak_color().1);
    }

    #[test]
    fn test_interpolate_clamp() {
        let p = Palette::PhosphorGreen;
        let color_neg = p.interpolate(-1.0);
        let color_over = p.interpolate(2.0);
        assert_eq!(color_neg, p.base_color());
        assert_eq!(color_over, p.peak_color());
    }

    #[test]
    fn test_all_palettes_have_colors() {
        let palettes = [
            Palette::PhosphorGreen,
            Palette::PhosphorAmber,
            Palette::PhosphorBlue,
            Palette::OrganicDecay,
            Palette::HeatDamage,
            Palette::GlitchEntropy,
            Palette::MemoryGhost,
            Palette::WebPhosphor,
            Palette::WebAmber,
            Palette::WebRust,
        ];

        for p in palettes {
            let colors = p.get_colors();
            assert!(!colors.is_empty(), "Palette {p:?} has no colors");
            assert!(colors.len() >= 5, "Palette {p:?} has too few colors");
        }
    }

    #[test]
    fn test_lerp_u8() {
        assert_eq!(lerp_u8(0, 100, 0.0), 0);
        assert_eq!(lerp_u8(0, 100, 1.0), 100);
        assert_eq!(lerp_u8(0, 100, 0.5), 50);
    }

    #[test]
    fn test_web_palettes_from_name() {
        assert_eq!(Palette::from_name("webgreen"), Some(Palette::WebPhosphor));
        assert_eq!(Palette::from_name("webamber"), Some(Palette::WebAmber));
        assert_eq!(Palette::from_name("webrust"), Some(Palette::WebRust));
        assert_eq!(Palette::from_name("web-green"), Some(Palette::WebPhosphor));
    }

    #[test]
    fn test_web_palette_names() {
        assert_eq!(Palette::WebPhosphor.name(), "webgreen");
        assert_eq!(Palette::WebAmber.name(), "webamber");
        assert_eq!(Palette::WebRust.name(), "webrust");
    }

    #[test]
    fn test_web_palette_colors() {
        // Web phosphor peak should be #00EE00
        let peak = Palette::WebPhosphor.peak_color();
        assert_eq!(peak, (0x77, 0xFF, 0x77));

        // Web amber peak should contain the amber glow
        let amber_peak = Palette::WebAmber.peak_color();
        assert_eq!(amber_peak, (0xFF, 0xDD, 0x88));

        // Web rust base should be dark
        let rust_base = Palette::WebRust.base_color();
        assert_eq!(rust_base, (0x1A, 0x0A, 0x0B));
    }
}
