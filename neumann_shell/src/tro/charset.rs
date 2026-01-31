// SPDX-License-Identifier: MIT OR Apache-2.0
//! Character gradients for TRO visual intensity mapping.

/// Rust/corrosion characters (low to high intensity).
/// Used for base decay patterns - sparse, industrial feel.
pub const RUST_CHARS: [char; 8] = ['.', ':', ';', '*', '#', '%', '@', '&'];

/// Organic growth characters (pulsing, alive).
/// Used for reaction-diffusion patterns - rounded, biological.
pub const ORGANIC_CHARS: [char; 8] = ['~', 'o', 'O', '0', 'Q', '@', '8', 'B'];

/// Physarum trail characters (network-like).
/// Used for slime mold paths - connected, flowing.
pub const TRAIL_CHARS: [char; 8] = ['.', '+', 'x', 'X', '#', '@', '%', '&'];

/// Heat/activity characters (sharp, bright).
/// Used for database activity feedback - energetic, urgent.
pub const HEAT_CHARS: [char; 8] = ['^', 'n', 'm', 'M', 'W', '#', '@', '%'];

/// Glitch characters (chaotic).
/// Used for error states - broken, corrupted.
pub const GLITCH_CHARS: [char; 8] = ['?', '!', '/', '\\', 'X', '#', '@', '%'];

/// Memory trace characters (fading).
/// Used for ghost patterns from past activity.
pub const GHOST_CHARS: [char; 8] = [' ', '.', '-', '=', '+', '*', '#', '@'];

/// Border frame characters for structural elements.
pub const BORDER_CHARS: BorderChars = BorderChars {
    top_left: '+',
    top_right: '+',
    bottom_left: '+',
    bottom_right: '+',
    horizontal: '-',
    vertical: '|',
};

/// Unicode border characters for enhanced display.
pub const UNICODE_BORDER: BorderChars = BorderChars {
    top_left: '\u{250C}',     // Box drawings light down and right
    top_right: '\u{2510}',    // Box drawings light down and left
    bottom_left: '\u{2514}',  // Box drawings light up and right
    bottom_right: '\u{2518}', // Box drawings light up and left
    horizontal: '\u{2500}',   // Box drawings light horizontal
    vertical: '\u{2502}',     // Box drawings light vertical
};

/// Heavy unicode border characters.
pub const HEAVY_BORDER: BorderChars = BorderChars {
    top_left: '\u{250F}',     // Box drawings heavy down and right
    top_right: '\u{2513}',    // Box drawings heavy down and left
    bottom_left: '\u{2517}',  // Box drawings heavy up and right
    bottom_right: '\u{251B}', // Box drawings heavy up and left
    horizontal: '\u{2501}',   // Box drawings heavy horizontal
    vertical: '\u{2503}',     // Box drawings heavy vertical
};

/// Double border characters.
pub const DOUBLE_BORDER: BorderChars = BorderChars {
    top_left: '\u{2554}',     // Box drawings double down and right
    top_right: '\u{2557}',    // Box drawings double down and left
    bottom_left: '\u{255A}',  // Box drawings double up and right
    bottom_right: '\u{255D}', // Box drawings double up and left
    horizontal: '\u{2550}',   // Box drawings double horizontal
    vertical: '\u{2551}',     // Box drawings double vertical
};

/// Border character set.
#[derive(Debug, Clone, Copy)]
pub struct BorderChars {
    pub top_left: char,
    pub top_right: char,
    pub bottom_left: char,
    pub bottom_right: char,
    pub horizontal: char,
    pub vertical: char,
}

/// Selects a character from a gradient based on intensity.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn select_char(chars: &[char], intensity: f32) -> char {
    if chars.is_empty() {
        return ' ';
    }

    let intensity = intensity.clamp(0.0, 1.0);
    // chars.len() is small (8-10), so cast to f32 is safe; intensity is clamped to 0.0-1.0
    let idx = (intensity * (chars.len() - 1) as f32).round() as usize;
    chars[idx.min(chars.len() - 1)]
}

/// Blends between two character sets based on a blend factor.
#[must_use]
pub fn blend_chars(primary: &[char], secondary: &[char], intensity: f32, blend: f32) -> char {
    // Use random-ish selection based on intensity bits
    let primary_char = select_char(primary, intensity);
    let secondary_char = select_char(secondary, intensity);

    // Simple threshold blend
    if blend > 0.5 {
        secondary_char
    } else {
        primary_char
    }
}

/// Block characters for smooth intensity gradients (Unicode).
pub const BLOCK_CHARS: [char; 9] = [
    ' ',        // Empty
    '\u{2591}', // Light shade
    '\u{2592}', // Medium shade
    '\u{2593}', // Dark shade
    '\u{2588}', // Full block
    '\u{2584}', // Lower half block
    '\u{2580}', // Upper half block
    '\u{258C}', // Left half block
    '\u{2590}', // Right half block
];

/// ASCII fallback for block characters.
pub const BLOCK_CHARS_ASCII: [char; 9] = [
    ' ', // Empty
    '.', // Light
    ':', // Light-medium
    '+', // Medium
    '#', // Medium-dark
    '%', // Dark
    '@', // Very dark
    '&', // Near full
    'M', // Full
];

/// Braille patterns for fine-grained density (Unicode).
pub const BRAILLE_DENSITY: [char; 8] = [
    '\u{2800}', // Empty
    '\u{2801}', // 1 dot
    '\u{2803}', // 2 dots
    '\u{2807}', // 3 dots
    '\u{280F}', // 4 dots
    '\u{281F}', // 5 dots
    '\u{283F}', // 6 dots
    '\u{28FF}', // Full 8 dots
];

/// ASCII fallback for braille patterns.
pub const BRAILLE_DENSITY_ASCII: [char; 8] = [
    ' ', // Empty
    '.', // 1 dot equivalent
    ':', // 2 dots
    ';', // 3 dots
    '+', // 4 dots
    '*', // 5 dots
    '#', // 6 dots
    '@', // Full
];

/// Braille base character (U+2800).
pub const BRAILLE_BASE: char = '\u{2800}';

/// Half-block characters for 1x2 sub-cell resolution.
/// Index = upper * 2 + lower (bit 1 = upper, bit 0 = lower).
pub const HALF_BLOCKS: [char; 4] = [
    ' ',        // 00 - empty
    '\u{2584}', // 01 - lower half only
    '\u{2580}', // 10 - upper half only
    '\u{2588}', // 11 - full block
];

/// Quadrant block characters for 2x2 sub-cell resolution.
/// Index is bitmask: bit 3=TL, bit 2=TR, bit 1=BL, bit 0=BR.
pub const QUADRANT_CHARS: [char; 16] = [
    ' ',        // 0000
    '\u{2597}', // 0001 - quadrant lower right
    '\u{2596}', // 0010 - quadrant lower left
    '\u{2584}', // 0011 - lower half
    '\u{259D}', // 0100 - quadrant upper right
    '\u{2590}', // 0101 - right half
    '\u{259E}', // 0110 - quadrant diagonal
    '\u{259F}', // 0111 - missing upper left
    '\u{2598}', // 1000 - quadrant upper left
    '\u{259A}', // 1001 - quadrant anti-diagonal
    '\u{258C}', // 1010 - left half
    '\u{2599}', // 1011 - missing upper right
    '\u{2580}', // 1100 - upper half
    '\u{259C}', // 1101 - missing lower left
    '\u{259B}', // 1110 - missing lower right
    '\u{2588}', // 1111 - full block
];

/// Converts 8 intensity values (2x4 grid) to a braille character.
///
/// Braille dot layout:
/// ```text
/// [0] [3]
/// [1] [4]
/// [2] [5]
/// [6] [7]
/// ```
#[must_use]
pub fn braille_from_grid(intensities: &[f32; 8], threshold: f32) -> char {
    let mut pattern: u8 = 0;

    // Left column: dots 1,2,3,7 (bits 0,1,2,6)
    if intensities[0] >= threshold {
        pattern |= 0x01;
    }
    if intensities[1] >= threshold {
        pattern |= 0x02;
    }
    if intensities[2] >= threshold {
        pattern |= 0x04;
    }
    if intensities[6] >= threshold {
        pattern |= 0x40;
    }

    // Right column: dots 4,5,6,8 (bits 3,4,5,7)
    if intensities[3] >= threshold {
        pattern |= 0x08;
    }
    if intensities[4] >= threshold {
        pattern |= 0x10;
    }
    if intensities[5] >= threshold {
        pattern |= 0x20;
    }
    if intensities[7] >= threshold {
        pattern |= 0x80;
    }

    char::from_u32(0x2800 + u32::from(pattern)).unwrap_or(' ')
}

/// Converts 4 intensity values (2x2 grid) to a quadrant block character.
///
/// Grid layout:
/// ```text
/// [0] [1]
/// [2] [3]
/// ```
#[must_use]
pub fn quadrant_from_grid(intensities: [f32; 4], threshold: f32) -> char {
    let mask = usize::from(intensities[0] >= threshold) * 8
        + usize::from(intensities[1] >= threshold) * 4
        + usize::from(intensities[2] >= threshold) * 2
        + usize::from(intensities[3] >= threshold);
    QUADRANT_CHARS[mask]
}

/// Converts 2 intensity values (1x2 vertical) to a half-block character.
///
/// Grid layout:
/// ```text
/// [0] - upper
/// [1] - lower
/// ```
#[must_use]
pub fn half_block_from_grid(upper: f32, lower: f32, threshold: f32) -> char {
    let mask = usize::from(upper >= threshold) * 2 + usize::from(lower >= threshold);
    HALF_BLOCKS[mask]
}

/// Character set mode for terminal compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CharsetMode {
    /// Full Unicode support (default).
    #[default]
    Unicode,
    /// ASCII-only mode for limited terminals.
    Ascii,
}

impl CharsetMode {
    /// Returns the block character set for this mode.
    #[must_use]
    pub const fn block_chars(&self) -> &'static [char; 9] {
        match self {
            Self::Unicode => &BLOCK_CHARS,
            Self::Ascii => &BLOCK_CHARS_ASCII,
        }
    }

    /// Returns the braille/density character set for this mode.
    #[must_use]
    pub const fn density_chars(&self) -> &'static [char; 8] {
        match self {
            Self::Unicode => &BRAILLE_DENSITY,
            Self::Ascii => &BRAILLE_DENSITY_ASCII,
        }
    }

    /// Returns the border character set for this mode.
    #[must_use]
    pub const fn border_chars(&self) -> &'static BorderChars {
        match self {
            Self::Unicode => &UNICODE_BORDER,
            Self::Ascii => &BORDER_CHARS,
        }
    }

    /// Detects the appropriate charset mode for the current terminal.
    #[must_use]
    pub fn detect() -> Self {
        // Check LANG/LC_ALL for UTF-8 support
        if let Ok(lang) = std::env::var("LANG") {
            if lang.to_lowercase().contains("utf") {
                return Self::Unicode;
            }
        }
        if let Ok(lc) = std::env::var("LC_ALL") {
            if lc.to_lowercase().contains("utf") {
                return Self::Unicode;
            }
        }

        // Check TERM for known limited terminals
        if let Ok(term) = std::env::var("TERM") {
            let term_lower = term.to_lowercase();
            // These terminals typically don't support Unicode well
            if term_lower.contains("linux")
                || term_lower.contains("vt100")
                || term_lower.contains("dumb")
                || term_lower.contains("ansi")
            {
                return Self::Ascii;
            }
        }

        // Default to Unicode on modern systems
        Self::Unicode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_chars_length() {
        assert_eq!(RUST_CHARS.len(), 8);
    }

    #[test]
    fn test_organic_chars_length() {
        assert_eq!(ORGANIC_CHARS.len(), 8);
    }

    #[test]
    fn test_trail_chars_length() {
        assert_eq!(TRAIL_CHARS.len(), 8);
    }

    #[test]
    fn test_heat_chars_length() {
        assert_eq!(HEAT_CHARS.len(), 8);
    }

    #[test]
    fn test_glitch_chars_length() {
        assert_eq!(GLITCH_CHARS.len(), 8);
    }

    #[test]
    fn test_select_char_zero() {
        let ch = select_char(&RUST_CHARS, 0.0);
        assert_eq!(ch, '.');
    }

    #[test]
    fn test_select_char_one() {
        let ch = select_char(&RUST_CHARS, 1.0);
        assert_eq!(ch, '&');
    }

    #[test]
    fn test_select_char_mid() {
        let ch = select_char(&RUST_CHARS, 0.5);
        // Should be around index 3-4
        assert!(RUST_CHARS.contains(&ch));
    }

    #[test]
    fn test_select_char_empty() {
        let ch = select_char(&[], 0.5);
        assert_eq!(ch, ' ');
    }

    #[test]
    fn test_select_char_clamp_negative() {
        let ch = select_char(&RUST_CHARS, -1.0);
        assert_eq!(ch, '.');
    }

    #[test]
    fn test_select_char_clamp_over() {
        let ch = select_char(&RUST_CHARS, 2.0);
        assert_eq!(ch, '&');
    }

    #[test]
    fn test_blend_chars_low() {
        let ch = blend_chars(&RUST_CHARS, &ORGANIC_CHARS, 0.5, 0.2);
        assert!(RUST_CHARS.contains(&ch));
    }

    #[test]
    fn test_blend_chars_high() {
        let ch = blend_chars(&RUST_CHARS, &ORGANIC_CHARS, 0.5, 0.8);
        assert!(ORGANIC_CHARS.contains(&ch));
    }

    #[test]
    fn test_border_chars() {
        assert_eq!(BORDER_CHARS.top_left, '+');
        assert_eq!(BORDER_CHARS.horizontal, '-');
        assert_eq!(BORDER_CHARS.vertical, '|');
    }

    #[test]
    fn test_unicode_border_chars() {
        assert_eq!(UNICODE_BORDER.top_left, '\u{250C}');
        assert_eq!(UNICODE_BORDER.horizontal, '\u{2500}');
    }

    #[test]
    fn test_block_chars_length() {
        assert_eq!(BLOCK_CHARS.len(), 9);
    }

    #[test]
    fn test_braille_density_length() {
        assert_eq!(BRAILLE_DENSITY.len(), 8);
    }

    #[test]
    fn test_block_chars_ascii_length() {
        assert_eq!(BLOCK_CHARS_ASCII.len(), 9);
    }

    #[test]
    fn test_braille_density_ascii_length() {
        assert_eq!(BRAILLE_DENSITY_ASCII.len(), 8);
    }

    #[test]
    fn test_charset_mode_default() {
        let mode = CharsetMode::default();
        assert_eq!(mode, CharsetMode::Unicode);
    }

    #[test]
    fn test_charset_mode_block_chars() {
        assert_eq!(CharsetMode::Unicode.block_chars(), &BLOCK_CHARS);
        assert_eq!(CharsetMode::Ascii.block_chars(), &BLOCK_CHARS_ASCII);
    }

    #[test]
    fn test_charset_mode_density_chars() {
        assert_eq!(CharsetMode::Unicode.density_chars(), &BRAILLE_DENSITY);
        assert_eq!(CharsetMode::Ascii.density_chars(), &BRAILLE_DENSITY_ASCII);
    }

    #[test]
    fn test_charset_mode_border_chars() {
        assert_eq!(CharsetMode::Unicode.border_chars().top_left, '\u{250C}');
        assert_eq!(CharsetMode::Ascii.border_chars().top_left, '+');
    }

    #[test]
    fn test_ascii_chars_are_ascii() {
        // Verify all ASCII fallback chars are actually ASCII
        for &ch in &BLOCK_CHARS_ASCII {
            assert!(ch.is_ascii(), "Block char {ch:?} is not ASCII");
        }
        for &ch in &BRAILLE_DENSITY_ASCII {
            assert!(ch.is_ascii(), "Density char {ch:?} is not ASCII");
        }
        assert!(BORDER_CHARS.top_left.is_ascii());
        assert!(BORDER_CHARS.horizontal.is_ascii());
        assert!(BORDER_CHARS.vertical.is_ascii());
    }

    #[test]
    fn test_braille_base_constant() {
        assert_eq!(BRAILLE_BASE, '\u{2800}');
    }

    #[test]
    fn test_half_blocks_length() {
        assert_eq!(HALF_BLOCKS.len(), 4);
    }

    #[test]
    fn test_quadrant_chars_length() {
        assert_eq!(QUADRANT_CHARS.len(), 16);
    }

    #[test]
    fn test_braille_from_grid_empty() {
        let intensities = [0.0; 8];
        let ch = braille_from_grid(&intensities, 0.5);
        assert_eq!(ch, '\u{2800}'); // Empty braille
    }

    #[test]
    fn test_braille_from_grid_full() {
        let intensities = [1.0; 8];
        let ch = braille_from_grid(&intensities, 0.5);
        assert_eq!(ch, '\u{28FF}'); // Full braille
    }

    #[test]
    fn test_braille_from_grid_single_dot() {
        let mut intensities = [0.0; 8];
        intensities[0] = 1.0;
        let ch = braille_from_grid(&intensities, 0.5);
        assert_eq!(ch, '\u{2801}'); // Single top-left dot
    }

    #[test]
    fn test_quadrant_from_grid_empty() {
        let intensities = [0.0; 4];
        let ch = quadrant_from_grid(intensities, 0.5);
        assert_eq!(ch, ' ');
    }

    #[test]
    fn test_quadrant_from_grid_full() {
        let intensities = [1.0; 4];
        let ch = quadrant_from_grid(intensities, 0.5);
        assert_eq!(ch, '\u{2588}'); // Full block
    }

    #[test]
    fn test_quadrant_from_grid_lower_half() {
        let intensities = [0.0, 0.0, 1.0, 1.0];
        let ch = quadrant_from_grid(intensities, 0.5);
        assert_eq!(ch, '\u{2584}'); // Lower half
    }

    #[test]
    fn test_half_block_from_grid_empty() {
        let ch = half_block_from_grid(0.0, 0.0, 0.5);
        assert_eq!(ch, ' ');
    }

    #[test]
    fn test_half_block_from_grid_upper() {
        let ch = half_block_from_grid(1.0, 0.0, 0.5);
        assert_eq!(ch, '\u{2580}'); // Upper half
    }

    #[test]
    fn test_half_block_from_grid_lower() {
        let ch = half_block_from_grid(0.0, 1.0, 0.5);
        assert_eq!(ch, '\u{2584}'); // Lower half
    }

    #[test]
    fn test_half_block_from_grid_full() {
        let ch = half_block_from_grid(1.0, 1.0, 0.5);
        assert_eq!(ch, '\u{2588}'); // Full block
    }
}
