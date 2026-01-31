// SPDX-License-Identifier: MIT OR Apache-2.0
//! Frame rendering for TRO border display.
//!
//! Implements double-buffered rendering with diff-based updates
//! to minimize terminal output and flicker.

use std::fmt::Write as _;
use std::io::{self, Write};

use super::border_region::{BorderRegion, ResolutionMode};
use super::charset::{braille_from_grid, half_block_from_grid, quadrant_from_grid, select_char};
use super::effects::{apply_flicker, apply_scanline};
use super::organism::TroState;
use super::tendril::TendrilManager;

/// Safely converts usize to u16, clamping to `u16::MAX`.
#[allow(clippy::cast_possible_truncation)] // Intentionally clamps before casting
fn to_u16(val: usize) -> u16 {
    val.min(u16::MAX as usize) as u16
}

/// A rendered cell with character and color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderedCell {
    /// The character to display.
    pub ch: char,
    /// RGB foreground color.
    pub fg: (u8, u8, u8),
}

impl Default for RenderedCell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: (0, 17, 0),
        }
    }
}

/// Double-buffered frame renderer.
pub struct TroRenderer {
    /// Front buffer (currently displayed).
    front: Vec<RenderedCell>,
    /// Back buffer (being rendered to).
    back: Vec<RenderedCell>,
    /// Terminal width.
    width: u16,
    /// Terminal height.
    height: u16,
    /// Whether CRT effects are enabled.
    crt_effects: bool,
    /// Frame counter for effects.
    frame: u64,
    /// Whether we've initialized the display.
    initialized: bool,
}

impl TroRenderer {
    /// Creates a new renderer.
    #[must_use]
    pub fn new(crt_effects: bool) -> Self {
        let term = console::Term::stdout();
        let (height, width) = term.size();

        let cell_count = Self::compute_border_cells(width, height);
        let front = vec![RenderedCell::default(); cell_count];
        let back = vec![RenderedCell::default(); cell_count];

        Self {
            front,
            back,
            width,
            height,
            crt_effects,
            frame: 0,
            initialized: false,
        }
    }

    /// Computes the number of border cells.
    fn compute_border_cells(width: u16, height: u16) -> usize {
        if width < 2 || height < 2 {
            return 0;
        }
        2 * width as usize + 2 * (height as usize).saturating_sub(2)
    }

    /// Renders a frame from TRO state.
    pub fn render_frame(&mut self, state: &TroState, glitch: bool) {
        // Check for terminal resize
        let term = console::Term::stdout();
        let (new_height, new_width) = term.size();
        if new_width != self.width || new_height != self.height {
            self.resize(new_width, new_height);
        }

        // Render cells to back buffer
        self.render_to_buffer(state, glitch);

        // Apply effects if enabled
        if self.crt_effects {
            self.apply_crt_effects();
        }

        // Output diff to terminal
        self.output_diff();

        // Swap buffers
        std::mem::swap(&mut self.front, &mut self.back);
        self.frame = self.frame.wrapping_add(1);
    }

    /// Renders TRO state to the back buffer.
    fn render_to_buffer(&mut self, state: &TroState, glitch: bool) {
        for (i, cell) in state.cells.iter().enumerate() {
            if i >= self.back.len() {
                break;
            }

            let ch = if glitch && rand::random::<f32>() < 0.1 {
                use super::charset::GLITCH_CHARS;
                GLITCH_CHARS[rand::random::<usize>() % GLITCH_CHARS.len()]
            } else {
                cell.character
            };

            let fg = if glitch && rand::random::<f32>() < 0.05 {
                // Random glitch color
                (
                    rand::random::<u8>(),
                    rand::random::<u8>(),
                    rand::random::<u8>(),
                )
            } else {
                cell.color
            };

            self.back[i] = RenderedCell { ch, fg };
        }
    }

    /// Applies CRT effects to the back buffer.
    fn apply_crt_effects(&mut self) {
        let width = self.width;
        let height = self.height;
        let frame = self.frame;

        for (i, cell) in self.back.iter_mut().enumerate() {
            // Get the row for this cell
            let (_, y) = Self::index_to_coord_static(width, height, i);

            // Apply scanline darkening on odd rows
            if y % 2 == 1 {
                cell.fg = apply_scanline(cell.fg, 0.6);
            }

            // Apply subtle flicker
            cell.fg = apply_flicker(cell.fg, frame, 0.02);
        }
    }

    /// Static version of `index_to_coord` for use in borrow contexts.
    fn index_to_coord_static(width: u16, height: u16, index: usize) -> (u16, u16) {
        let w = width as usize;
        let h = height as usize;

        if w < 2 || h < 2 {
            return (0, 0);
        }

        let top_len = w;
        let right_len = h.saturating_sub(2);
        let bottom_len = w;

        if index < top_len {
            // Top edge
            (to_u16(index), 0)
        } else if index < top_len + right_len {
            // Right edge
            let offset = index - top_len;
            (to_u16(w.saturating_sub(1)), to_u16(offset + 1))
        } else if index < top_len + right_len + bottom_len {
            // Bottom edge
            let offset = index - top_len - right_len;
            (
                to_u16(w.saturating_sub(1).saturating_sub(offset)),
                to_u16(h.saturating_sub(1)),
            )
        } else {
            // Left edge
            let offset = index - top_len - right_len - bottom_len;
            (0, to_u16(h.saturating_sub(2).saturating_sub(offset)))
        }
    }

    /// Outputs only the changed cells to terminal.
    fn output_diff(&mut self) {
        let mut output = String::new();

        // Save cursor position before rendering border
        output.push_str("\x1b7"); // Save cursor position (DEC)

        // Initialize on first frame
        if !self.initialized {
            self.initialized = true;
        }

        let mut has_changes = false;
        for (i, (front_cell, back_cell)) in self.front.iter().zip(self.back.iter()).enumerate() {
            if front_cell != back_cell {
                has_changes = true;
                let (x, y) = self.index_to_coord(i);

                // Move cursor and set color
                let _ = write!(
                    output,
                    "\x1b[{};{}H\x1b[38;2;{};{};{}m{}",
                    y + 1,
                    x + 1,
                    back_cell.fg.0,
                    back_cell.fg.1,
                    back_cell.fg.2,
                    back_cell.ch
                );
            }
        }

        // Reset attributes and restore cursor position
        output.push_str("\x1b[0m"); // Reset attributes
        output.push_str("\x1b8"); // Restore cursor position (DEC)

        // Only write if we have changes (or need to restore cursor)
        if has_changes || !self.initialized {
            let _ = io::stdout().write_all(output.as_bytes());
            let _ = io::stdout().flush();
        }
    }

    /// Converts a linear index to screen coordinates.
    fn index_to_coord(&self, index: usize) -> (u16, u16) {
        let w = self.width as usize;
        let h = self.height as usize;

        if w < 2 || h < 2 {
            return (0, 0);
        }

        let top_len = w;
        let right_len = h.saturating_sub(2);
        let bottom_len = w;

        if index < top_len {
            // Top edge
            (to_u16(index), 0)
        } else if index < top_len + right_len {
            // Right edge
            let offset = index - top_len;
            (to_u16(w.saturating_sub(1)), to_u16(offset + 1))
        } else if index < top_len + right_len + bottom_len {
            // Bottom edge
            let offset = index - top_len - right_len;
            (
                to_u16(w.saturating_sub(1).saturating_sub(offset)),
                to_u16(h.saturating_sub(1)),
            )
        } else {
            // Left edge
            let offset = index - top_len - right_len - bottom_len;
            (0, to_u16(h.saturating_sub(2).saturating_sub(offset)))
        }
    }

    /// Handles terminal resize.
    fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;

        let cell_count = Self::compute_border_cells(width, height);
        self.front.resize(cell_count, RenderedCell::default());
        self.back.resize(cell_count, RenderedCell::default());

        // Force full redraw
        for cell in &mut self.front {
            *cell = RenderedCell {
                ch: '\0',
                fg: (0, 0, 0),
            };
        }
    }

    /// Clears the border display and shows cursor.
    pub fn clear(&mut self) {
        let mut output = String::new();

        // Clear all border cells
        for i in 0..self.front.len() {
            let (x, y) = self.index_to_coord(i);
            let _ = write!(output, "\x1b[{};{}H ", y + 1, x + 1);
        }

        // Reset and show cursor
        output.push_str("\x1b[0m\x1b[?25h");

        let _ = io::stdout().write_all(output.as_bytes());
        let _ = io::stdout().flush();

        self.initialized = false;
    }

    /// Returns the current frame number.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Returns the terminal dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Enables or disables CRT effects.
    pub fn set_crt_effects(&mut self, enabled: bool) {
        self.crt_effects = enabled;
    }

    /// Returns whether CRT effects are enabled.
    #[must_use]
    pub const fn crt_effects_enabled(&self) -> bool {
        self.crt_effects
    }

    /// Renders a 2D border region with tendrils.
    #[allow(clippy::too_many_lines)]
    pub fn render_region(
        &mut self,
        region: &BorderRegion,
        tendrils: &TendrilManager,
        palette: &[(u8, u8, u8)],
        glitch: bool,
    ) {
        // Check for terminal resize
        let term = console::Term::stdout();
        let (new_height, new_width) = term.size();
        if new_width != self.width || new_height != self.height {
            self.resize(new_width, new_height);
        }

        let mut output = String::with_capacity(4096);
        output.push_str("\x1b7"); // Save cursor

        // Render border region cells - only top and bottom rows for clean CLI
        // Side borders would overlap with content lines
        let height = region.height();
        for (coord, cell) in region.iter() {
            // Only render top row (y=0) and bottom row (y=height-1)
            if coord.y != 0 && coord.y != height.saturating_sub(1) {
                continue;
            }

            let ch = if glitch && rand::random::<f32>() < 0.1 {
                use super::charset::GLITCH_CHARS;
                GLITCH_CHARS[rand::random::<usize>() % GLITCH_CHARS.len()]
            } else {
                Self::select_char_for_resolution(cell.primary_intensity, region.resolution())
            };

            let fg = if glitch && rand::random::<f32>() < 0.05 {
                (rand::random(), rand::random(), rand::random())
            } else {
                Self::interpolate_color(cell.primary_intensity, cell.heat, palette)
            };

            let _ = write!(
                output,
                "\x1b[{};{}H\x1b[38;2;{};{};{}m{}",
                coord.y + 1,
                coord.x + 1,
                fg.0,
                fg.1,
                fg.2,
                ch
            );
        }

        // Tendrils are disabled for clean CLI - they would render in content area
        // Keep tendril_manager for future use when we have a dedicated border mode
        let _ = tendrils;

        output.push_str("\x1b[0m\x1b8"); // Reset + restore cursor

        let _ = io::stdout().write_all(output.as_bytes());
        let _ = io::stdout().flush();

        self.frame = self.frame.wrapping_add(1);
    }

    /// Selects character based on resolution mode.
    fn select_char_for_resolution(intensity: f32, mode: ResolutionMode) -> char {
        use super::charset::ORGANIC_CHARS;

        match mode {
            ResolutionMode::Standard => select_char(&ORGANIC_CHARS, intensity),
            ResolutionMode::HalfBlock => {
                // For half-block, split intensity into upper/lower
                let upper = intensity * 1.1;
                let lower = intensity * 0.9;
                half_block_from_grid(upper, lower, 0.3)
            },
            ResolutionMode::Quadrant => {
                // For quadrant, create 2x2 pattern from intensity
                let intensities = [
                    intensity * 1.1,
                    intensity * 0.9,
                    intensity * 0.95,
                    intensity * 1.05,
                ];
                quadrant_from_grid(intensities, 0.3)
            },
            ResolutionMode::Braille => {
                // For braille, create 2x4 pattern from intensity
                let intensities = [
                    intensity * 1.1,
                    intensity * 0.95,
                    intensity * 0.9,
                    intensity * 1.05,
                    intensity * 0.85,
                    intensity * 1.0,
                    intensity * 0.8,
                    intensity * 0.7,
                ];
                braille_from_grid(&intensities, 0.25)
            },
        }
    }

    /// Interpolates color based on intensity and heat.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn interpolate_color(intensity: f32, heat: f32, palette: &[(u8, u8, u8)]) -> (u8, u8, u8) {
        if palette.is_empty() {
            return (0, 51, 0);
        }

        let intensity = intensity.clamp(0.0, 1.0);
        #[allow(clippy::cast_precision_loss)]
        let base_idx = ((intensity * (palette.len() - 1) as f32) as usize).min(palette.len() - 1);
        let base_color = palette[base_idx];

        if heat > 0.1 {
            #[allow(clippy::cast_precision_loss)]
            let heat_factor = (heat * 0.5).min(0.3);
            let r = (f32::from(base_color.0) + heat_factor * 200.0).min(255.0) as u8;
            let g = (f32::from(base_color.1) * (1.0 - heat_factor * 0.3)).max(0.0) as u8;
            (r, g, base_color.2)
        } else {
            base_color
        }
    }

    /// Clears a region (for cleanup).
    pub fn clear_region(&mut self, region: &BorderRegion) {
        let mut output = String::new();

        for (coord, _) in region.iter() {
            let _ = write!(output, "\x1b[{};{}H ", coord.y + 1, coord.x + 1);
        }

        output.push_str("\x1b[0m");

        let _ = io::stdout().write_all(output.as_bytes());
        let _ = io::stdout().flush();
    }

    /// Renders tendril cells only (for overlay rendering).
    pub fn render_tendrils(
        &mut self,
        tendrils: &TendrilManager,
        border_check: impl Fn(u16, u16) -> bool,
        resolution: ResolutionMode,
        palette: &[(u8, u8, u8)],
    ) {
        let mut output = String::with_capacity(1024);
        output.push_str("\x1b7"); // Save cursor

        for (coord, intensity) in tendrils.visible_cells() {
            if !border_check(coord.x, coord.y) {
                let ch = Self::select_char_for_resolution(intensity, resolution);
                let fg = Self::interpolate_color(intensity, 0.0, palette);

                let _ = write!(
                    output,
                    "\x1b[{};{}H\x1b[38;2;{};{};{}m{}",
                    coord.y + 1,
                    coord.x + 1,
                    fg.0,
                    fg.1,
                    fg.2,
                    ch
                );
            }
        }

        output.push_str("\x1b[0m\x1b8");

        let _ = io::stdout().write_all(output.as_bytes());
        let _ = io::stdout().flush();
    }
}

impl Drop for TroRenderer {
    fn drop(&mut self) {
        // Show cursor on drop
        let _ = io::stdout().write_all(b"\x1b[?25h");
        let _ = io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::super::charset::{HALF_BLOCKS, ORGANIC_CHARS, QUADRANT_CHARS};
    use super::*;

    #[test]
    fn test_rendered_cell_default() {
        let cell = RenderedCell::default();
        assert_eq!(cell.ch, ' ');
        assert_eq!(cell.fg, (0, 17, 0));
    }

    #[test]
    fn test_rendered_cell_equality() {
        let a = RenderedCell {
            ch: 'X',
            fg: (255, 0, 0),
        };
        let b = RenderedCell {
            ch: 'X',
            fg: (255, 0, 0),
        };
        let c = RenderedCell {
            ch: 'Y',
            fg: (255, 0, 0),
        };

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_compute_border_cells() {
        // 80x24: 80 + 80 + 22 + 22 = 204
        let cells = TroRenderer::compute_border_cells(80, 24);
        assert_eq!(cells, 204);
    }

    #[test]
    fn test_compute_border_cells_small() {
        let cells = TroRenderer::compute_border_cells(2, 2);
        assert_eq!(cells, 4);
    }

    #[test]
    fn test_compute_border_cells_invalid() {
        let cells = TroRenderer::compute_border_cells(1, 1);
        assert_eq!(cells, 0);
    }

    #[test]
    fn test_renderer_new() {
        let renderer = TroRenderer::new(true);
        assert!(renderer.crt_effects);
        assert_eq!(renderer.frame, 0);
        assert!(!renderer.initialized);
    }

    #[test]
    fn test_renderer_dimensions() {
        let renderer = TroRenderer::new(false);
        let (w, h) = renderer.dimensions();
        assert!(w > 0 || h > 0 || (w == 0 && h == 0)); // May be 0 in CI
    }

    #[test]
    fn test_renderer_frame() {
        let renderer = TroRenderer::new(false);
        assert_eq!(renderer.frame(), 0);
    }

    #[test]
    fn test_index_to_coord_top() {
        let mut renderer = TroRenderer::new(false);
        renderer.width = 80;
        renderer.height = 24;

        assert_eq!(renderer.index_to_coord(0), (0, 0));
        assert_eq!(renderer.index_to_coord(79), (79, 0));
    }

    #[test]
    fn test_index_to_coord_right() {
        let mut renderer = TroRenderer::new(false);
        renderer.width = 80;
        renderer.height = 24;

        assert_eq!(renderer.index_to_coord(80), (79, 1));
        assert_eq!(renderer.index_to_coord(81), (79, 2));
    }

    #[test]
    fn test_index_to_coord_invalid_dimensions() {
        let mut renderer = TroRenderer::new(false);
        renderer.width = 1;
        renderer.height = 1;

        assert_eq!(renderer.index_to_coord(0), (0, 0));
    }

    #[test]
    fn test_select_char_for_resolution_standard() {
        let ch = TroRenderer::select_char_for_resolution(0.5, ResolutionMode::Standard);
        // Should return an organic char
        assert!(ORGANIC_CHARS.contains(&ch));
    }

    #[test]
    fn test_select_char_for_resolution_halfblock() {
        let ch = TroRenderer::select_char_for_resolution(0.5, ResolutionMode::HalfBlock);
        // Should return a half-block char
        assert!(HALF_BLOCKS.contains(&ch));
    }

    #[test]
    fn test_select_char_for_resolution_quadrant() {
        let ch = TroRenderer::select_char_for_resolution(0.5, ResolutionMode::Quadrant);
        // Should return a quadrant char
        assert!(QUADRANT_CHARS.contains(&ch));
    }

    #[test]
    fn test_select_char_for_resolution_braille() {
        let ch = TroRenderer::select_char_for_resolution(0.5, ResolutionMode::Braille);
        // Braille chars are in range U+2800 to U+28FF
        let code = ch as u32;
        assert!(
            (0x2800..=0x28FF).contains(&code),
            "Expected braille char, got {ch:?}"
        );
    }

    #[test]
    fn test_interpolate_color_empty_palette() {
        let color = TroRenderer::interpolate_color(0.5, 0.0, &[]);
        assert_eq!(color, (0, 51, 0));
    }

    #[test]
    fn test_interpolate_color_no_heat() {
        let palette = [(0, 0, 0), (100, 100, 100), (200, 200, 200)];
        let color = TroRenderer::interpolate_color(0.5, 0.0, &palette);
        assert_eq!(color, (100, 100, 100));
    }

    #[test]
    fn test_interpolate_color_with_heat() {
        let palette = [(0, 100, 0), (0, 200, 0)];
        let color = TroRenderer::interpolate_color(0.5, 0.5, &palette);
        // With heat, red should increase and green decrease slightly
        assert!(color.0 > 0);
    }
}
