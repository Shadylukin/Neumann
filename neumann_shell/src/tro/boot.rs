// SPDX-License-Identifier: MIT OR Apache-2.0
//! Boot sequence animation for TRO shell startup.
//!
//! Implements a Fallout-style POST (Power-On Self-Test) sequence
//! that types out system initialization messages with variable delays.

use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use super::effects::StaticNoise;
use super::palette::Palette;

/// Boot sequence animation style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BootStyle {
    /// Full POST sequence with delays and effects.
    #[default]
    Full,
    /// Compact sequence (faster, fewer effects).
    Compact,
    /// No boot sequence.
    None,
}

/// Boot sequence animator.
pub struct BootSequence {
    style: BootStyle,
    palette: Palette,
    noise: StaticNoise,
}

impl BootSequence {
    /// Creates a new boot sequence.
    #[must_use]
    pub fn new(style: BootStyle, palette: Palette) -> Self {
        Self {
            style,
            palette,
            noise: StaticNoise::default(),
        }
    }

    /// Runs the boot sequence animation.
    pub fn run(&self, version: &str) {
        match self.style {
            BootStyle::Full => self.run_full(version),
            BootStyle::Compact => self.run_compact(version),
            BootStyle::None => {},
        }
    }

    /// Runs the full POST sequence.
    fn run_full(&self, version: &str) {
        let base_color = self.palette.get_colors()[2];
        let bright_color = self.palette.peak_color();
        let warn_color = (0xFF, 0x66, 0x00); // Orange/red for warnings

        // Clear screen and set color
        print!("\x1b[2J\x1b[H");
        print!(
            "\x1b[38;2;{};{};{}m",
            base_color.0, base_color.1, base_color.2
        );
        let _ = io::stdout().flush();

        // Static noise burst
        self.static_burst(3, 50);

        // Header
        let header = format!(
            r"
  ################################################################

  SCRUNCHEE SYSTEMS (TM) TERMLINK PROTOCOL
  NEUMANN TENSOR DATABASE SYSTEM v{version}

"
        );

        self.type_text(&header, 8, 15);
        thread::sleep(Duration::from_millis(200));

        // Initialization sequence
        let init_lines = [
            ("INITIALIZING TENSOR SLAB ROUTER", 400, true),
            ("LOADING ARCHETYPE REGISTRY", 350, true),
            ("CALIBRATING HYPERBOLIC MANIFOLD", 500, true),
            ("SPAWNING RUST ORGANISM", 600, true),
        ];

        for (msg, delay, success) in init_lines {
            self.type_progress_line(msg, delay, success, bright_color);
        }

        thread::sleep(Duration::from_millis(300));

        // Warning messages
        print!(
            "\x1b[38;2;{};{};{}m",
            warn_color.0, warn_color.1, warn_color.2
        );
        let _ = io::stdout().flush();

        self.type_text_flash("\n  WARNING: ORGANIC CONTAMINATION DETECTED\n", 30, 3);
        thread::sleep(Duration::from_millis(150));

        self.type_text_flash("  WARNING: SYSTEM INTEGRITY AT 73%\n", 30, 3);
        thread::sleep(Duration::from_millis(300));

        // Reset to base color
        print!(
            "\x1b[38;2;{};{};{}m",
            base_color.0, base_color.1, base_color.2
        );
        let _ = io::stdout().flush();

        self.type_text("\n  ENTERING INTERACTIVE MODE\n", 20, 40);

        // Footer
        self.type_text(
            "\n  ################################################################\n\n",
            5,
            10,
        );

        // Final static burst as organism spawns
        self.static_burst(2, 30);

        // Reset attributes
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }

    /// Runs the compact boot sequence.
    fn run_compact(&self, version: &str) {
        let base_color = self.palette.get_colors()[2];

        print!(
            "\x1b[38;2;{};{};{}m",
            base_color.0, base_color.1, base_color.2
        );

        let text = format!(
            "NEUMANN v{version} - Tensor Database Engine\n\
             Initializing... [OK]\n\n"
        );

        self.type_text(&text, 5, 20);

        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }

    /// Types text character by character with variable delays.
    #[allow(clippy::unused_self)] // Method pattern for consistency with other type_* methods
    fn type_text(&self, text: &str, base_delay_ms: u64, variance_ms: u64) {
        for ch in text.chars() {
            print!("{ch}");
            let _ = io::stdout().flush();

            let delay = if ch == '\n' {
                base_delay_ms * 3
            } else if ch == '.' || ch == ',' {
                base_delay_ms * 2
            } else {
                let variance = if variance_ms > 0 {
                    rand::random::<u64>() % variance_ms
                } else {
                    0
                };
                base_delay_ms + variance
            };

            thread::sleep(Duration::from_millis(delay));
        }
    }

    /// Types text with periodic flash effect.
    fn type_text_flash(&self, text: &str, delay_ms: u64, flash_count: u32) {
        self.type_text(text, delay_ms, 10);

        // Flash the line
        for _ in 0..flash_count {
            thread::sleep(Duration::from_millis(100));
            print!("\x1b[?5h"); // Reverse video
            let _ = io::stdout().flush();
            thread::sleep(Duration::from_millis(50));
            print!("\x1b[?5l"); // Normal video
            let _ = io::stdout().flush();
        }
    }

    /// Types a progress line with [OK] or [FAIL] indicator.
    fn type_progress_line(
        &self,
        message: &str,
        process_delay_ms: u64,
        success: bool,
        color: (u8, u8, u8),
    ) {
        // Type the message
        print!("  {message}");
        let _ = io::stdout().flush();

        // Add dots with delay
        let dot_count = 40 - message.len().min(35);
        for _ in 0..dot_count {
            print!(".");
            let _ = io::stdout().flush();
            thread::sleep(Duration::from_millis(process_delay_ms / dot_count as u64));
        }

        // Show result
        thread::sleep(Duration::from_millis(50));

        if success {
            print!(" \x1b[38;2;{};{};{}m[OK]\x1b[0m", color.0, color.1, color.2);
        } else {
            print!(" \x1b[38;2;255;0;0m[FAIL]\x1b[0m");
        }
        println!();
        let _ = io::stdout().flush();

        // Reset color for next line
        let base = self.palette.get_colors()[2];
        print!("\x1b[38;2;{};{};{}m", base.0, base.1, base.2);
        let _ = io::stdout().flush();
    }

    /// Displays a burst of static noise.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn static_burst(&self, lines: usize, duration_ms: u64) {
        let term = console::Term::stdout();
        let (_, width) = term.size();
        let width = width as usize;

        let frames = duration_ms / 33; // ~30fps

        for frame in 0..frames {
            // Move to start
            print!("\x1b[{lines}A");
            let _ = io::stdout().flush();

            for line in 0..lines {
                for col in 0..width {
                    let pos = line * width + col;
                    let ch = self.noise.get_char(pos, frame);
                    let intensity = self.noise.get(pos, frame);
                    let brightness = ((intensity * 100.0) as u8).saturating_add(20);
                    print!("\x1b[38;2;0;{brightness};0m{ch}");
                }
                println!();
            }

            let _ = io::stdout().flush();
            thread::sleep(Duration::from_millis(33));
        }

        // Clear static lines
        for _ in 0..lines {
            println!("{:width$}", "", width = width);
        }
        print!("\x1b[{lines}A");
        let _ = io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_style_default() {
        assert_eq!(BootStyle::default(), BootStyle::Full);
    }

    #[test]
    fn test_boot_sequence_new() {
        let seq = BootSequence::new(BootStyle::Full, Palette::PhosphorGreen);
        assert_eq!(seq.style, BootStyle::Full);
    }

    #[test]
    fn test_boot_sequence_none_is_noop() {
        let seq = BootSequence::new(BootStyle::None, Palette::PhosphorGreen);
        seq.run("0.1.0"); // Should complete instantly
    }

    // Note: Full and compact sequences write to stdout and include delays,
    // so we don't test them in unit tests. They're tested manually.
}
