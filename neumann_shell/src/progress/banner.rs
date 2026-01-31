// SPDX-License-Identifier: MIT OR Apache-2.0
//! Welcome banner for shell startup.

#![allow(clippy::format_push_string)]

use crate::style::{styled, Theme};

/// ASCII art banner for Neumann shell.
const BANNER: &str = r"
    _   _
   | \ | | ___ _   _ _ __ ___   __ _ _ __  _ __
   |  \| |/ _ \ | | | '_ ` _ \ / _` | '_ \| '_ \
   | |\  |  __/ |_| | | | | | | (_| | | | | | | |
   |_| \_|\___|\__,_|_| |_| |_|\__,_|_| |_|_| |_|
";

/// Generates the welcome banner with version and hints.
#[must_use]
pub fn welcome_banner(version: &str, theme: &Theme) -> String {
    let mut output = String::new();

    // Banner art
    output.push_str(&styled(BANNER, theme.header));
    output.push_str(&format!(
        "                                          {}\n",
        styled(format!("v{version}"), theme.muted)
    ));

    // Tagline
    output.push_str(&format!(
        "        {}\n\n",
        styled("Unified Tensor Database Engine", theme.info)
    ));

    // Feature highlights
    output.push_str(&format!(
        "   {} + {} + {} = {}\n\n",
        styled("Relational", theme.keyword),
        styled("Graph", theme.keyword),
        styled("Vector", theme.keyword),
        styled("Unified Queries", theme.highlight)
    ));

    // Hints
    output.push_str(&format!(
        "   Type '{}' for commands, {} for completion\n",
        styled("help", theme.string),
        styled("Tab", theme.muted)
    ));

    output
}

/// Generates a compact banner (for non-TTY or small terminals).
#[must_use]
pub fn compact_banner(version: &str, theme: &Theme) -> String {
    format!(
        "{} {} - {}\n",
        styled("Neumann", theme.header),
        styled(format!("v{version}"), theme.muted),
        styled("Type 'help' for commands", theme.info)
    )
}

/// Generates the goodbye message.
#[must_use]
pub fn goodbye_message(theme: &Theme) -> String {
    format!(
        "{} {}",
        styled("Session terminated.", theme.muted),
        styled("Goodbye!", theme.info)
    )
}

/// Checks if the terminal supports the full banner.
#[must_use]
pub fn supports_full_banner() -> bool {
    if !console::Term::stdout().is_term() {
        return false;
    }

    // Check terminal width (banner needs ~60 columns)
    let term = console::Term::stdout();
    term.size().1 >= 60
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welcome_banner_contains_version() {
        let theme = Theme::plain();
        let banner = welcome_banner("0.1.0", &theme);
        assert!(banner.contains("0.1.0"));
    }

    #[test]
    fn test_welcome_banner_contains_features() {
        let theme = Theme::plain();
        let banner = welcome_banner("0.1.0", &theme);
        assert!(banner.contains("Relational"));
        assert!(banner.contains("Graph"));
        assert!(banner.contains("Vector"));
    }

    #[test]
    fn test_welcome_banner_contains_help_hint() {
        let theme = Theme::plain();
        let banner = welcome_banner("0.1.0", &theme);
        assert!(banner.contains("help"));
        assert!(banner.contains("Tab"));
    }

    #[test]
    fn test_compact_banner() {
        let theme = Theme::plain();
        let banner = compact_banner("0.1.0", &theme);
        assert!(banner.contains("Neumann"));
        assert!(banner.contains("0.1.0"));
        assert!(banner.contains("help"));
    }

    #[test]
    fn test_goodbye_message() {
        let theme = Theme::plain();
        let msg = goodbye_message(&theme);
        assert!(msg.contains("Goodbye"));
    }

    #[test]
    fn test_banner_ascii_art() {
        assert!(BANNER.contains("Neumann") || BANNER.contains("|"));
    }
}
