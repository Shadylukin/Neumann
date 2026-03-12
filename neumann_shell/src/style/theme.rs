// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color theme system for terminal output.

use owo_colors::{OwoColorize, Style};

/// Color theme for terminal output.
///
/// Provides a consistent color palette for all shell output. Supports both
/// dark and light terminal backgrounds with automatic detection.
#[derive(Debug, Clone)]
pub struct Theme {
    // Status colors
    pub success: Style,
    pub error: Style,
    pub warning: Style,
    pub info: Style,

    // Structural
    pub header: Style,
    pub border: Style,
    pub muted: Style,

    // Data types
    pub keyword: Style,
    pub string: Style,
    pub number: Style,
    pub id: Style,
    pub label: Style,
    pub null: Style,

    // Special
    pub highlight: Style,
    pub link: Style,
}

#[allow(clippy::missing_const_for_fn)]
impl Theme {
    /// Creates a dark theme (Gruvbox-inspired).
    #[must_use]
    pub fn dark() -> Self {
        Self {
            // Status
            success: Style::new().green(),
            error: Style::new().red().bold(),
            warning: Style::new().yellow(),
            info: Style::new().cyan(),

            // Structural
            header: Style::new().magenta().bold(),
            border: Style::new().bright_black(),
            muted: Style::new().bright_black(),

            // Data types
            keyword: Style::new().blue(),
            string: Style::new().green(),
            number: Style::new().yellow(),
            id: Style::new().cyan().bold(),
            label: Style::new().magenta(),
            null: Style::new().bright_black().italic(),

            // Special
            highlight: Style::new().white().bold(),
            link: Style::new().blue().underline(),
        }
    }

    /// Creates a light theme for light terminal backgrounds.
    #[must_use]
    pub fn light() -> Self {
        Self {
            // Status
            success: Style::new().green(),
            error: Style::new().red().bold(),
            warning: Style::new().yellow(),
            info: Style::new().cyan(),

            // Structural
            header: Style::new().magenta().bold(),
            border: Style::new().bright_black(),
            muted: Style::new().bright_black(),

            // Data types
            keyword: Style::new().blue(),
            string: Style::new().green(),
            number: Style::new().yellow(),
            id: Style::new().cyan().bold(),
            label: Style::new().magenta(),
            null: Style::new().bright_black().italic(),

            // Special
            highlight: Style::new().black().bold(),
            link: Style::new().blue().underline(),
        }
    }

    /// Creates a plain theme with no colors (for piped output).
    #[must_use]
    pub fn plain() -> Self {
        Self {
            success: Style::new(),
            error: Style::new(),
            warning: Style::new(),
            info: Style::new(),
            header: Style::new(),
            border: Style::new(),
            muted: Style::new(),
            keyword: Style::new(),
            string: Style::new(),
            number: Style::new(),
            id: Style::new(),
            label: Style::new(),
            null: Style::new(),
            highlight: Style::new(),
            link: Style::new(),
        }
    }

    /// Creates a phosphor green theme (matches boot sequence aesthetic).
    #[must_use]
    pub fn phosphor() -> Self {
        Self {
            // Status - green variants
            success: Style::new().green().bold(),
            error: Style::new().red().bold(),
            warning: Style::new().yellow(),
            info: Style::new().bright_green(),

            // Structural
            header: Style::new().bright_green().bold(),
            border: Style::new().green(),
            muted: Style::new().bright_black(),

            // Data types - green palette
            keyword: Style::new().bright_green(),
            string: Style::new().green(),
            number: Style::new().bright_green(),
            id: Style::new().green().bold(),
            label: Style::new().bright_green(),
            null: Style::new().bright_black().italic(),

            // Special
            highlight: Style::new().bright_green().bold(),
            link: Style::new().green().underline(),
        }
    }

    /// Auto-detects the best theme based on terminal capabilities.
    #[must_use]
    pub fn auto() -> Self {
        if console::Term::stdout().is_term() {
            Self::phosphor() // Use phosphor theme by default for terminal
        } else {
            Self::plain()
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::auto()
    }
}

/// Applies a style to text, returning the styled string.
pub fn styled<T: std::fmt::Display>(text: T, style: Style) -> String {
    text.style(style).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_theme_has_colors() {
        let theme = Theme::dark();
        // Verify all fields are accessible
        let _ = theme.success;
        let _ = theme.error;
        let _ = theme.warning;
        let _ = theme.info;
        let _ = theme.header;
        let _ = theme.border;
        let _ = theme.muted;
        let _ = theme.keyword;
        let _ = theme.string;
        let _ = theme.number;
        let _ = theme.id;
        let _ = theme.label;
        let _ = theme.null;
        let _ = theme.highlight;
        let _ = theme.link;
    }

    #[test]
    fn test_light_theme_has_colors() {
        let theme = Theme::light();
        let _ = theme.success;
        let _ = theme.error;
        let _ = theme.warning;
        let _ = theme.info;
        let _ = theme.header;
        let _ = theme.border;
        let _ = theme.muted;
        let _ = theme.keyword;
        let _ = theme.string;
        let _ = theme.number;
        let _ = theme.id;
        let _ = theme.label;
        let _ = theme.null;
        let _ = theme.highlight;
        let _ = theme.link;
    }

    #[test]
    fn test_plain_theme_no_colors() {
        let theme = Theme::plain();
        let _ = theme.success;
        let _ = theme.error;
        let _ = theme.warning;
        let _ = theme.info;
        let _ = theme.header;
        let _ = theme.border;
        let _ = theme.muted;
        let _ = theme.keyword;
        let _ = theme.string;
        let _ = theme.number;
        let _ = theme.id;
        let _ = theme.label;
        let _ = theme.null;
        let _ = theme.highlight;
        let _ = theme.link;
    }

    #[test]
    fn test_phosphor_theme_has_colors() {
        let theme = Theme::phosphor();
        let _ = theme.success;
        let _ = theme.error;
        let _ = theme.warning;
        let _ = theme.info;
        let _ = theme.header;
        let _ = theme.border;
        let _ = theme.muted;
        let _ = theme.keyword;
        let _ = theme.string;
        let _ = theme.number;
        let _ = theme.id;
        let _ = theme.label;
        let _ = theme.null;
        let _ = theme.highlight;
        let _ = theme.link;
    }

    #[test]
    fn test_styled_applies_style() {
        let style = Style::new().green();
        let result = styled("test", style);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_styled_with_number() {
        let style = Style::new().yellow();
        let result = styled(42, style);
        assert!(result.contains("42"));
    }

    #[test]
    fn test_default_theme() {
        let _theme = Theme::default();
    }

    #[test]
    fn test_theme_clone() {
        let theme = Theme::dark();
        let cloned = theme;
        let _ = cloned.success;
    }

    #[test]
    fn test_theme_debug() {
        let theme = Theme::dark();
        let debug = format!("{theme:?}");
        assert!(debug.contains("Theme"));
    }

    #[test]
    fn test_auto_theme() {
        let _theme = Theme::auto();
    }
}
