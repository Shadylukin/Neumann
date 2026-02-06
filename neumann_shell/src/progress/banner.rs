// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
        assert!(BANNER.contains("Neumann") || BANNER.contains('|'));
    }

    #[test]
    fn test_supports_full_banner() {
        // Just verify it returns a boolean without panicking
        let _ = supports_full_banner();
    }

    #[test]
    fn test_welcome_banner_not_empty() {
        let theme = Theme::plain();
        let banner = welcome_banner("1.0.0", &theme);
        assert!(!banner.is_empty());
    }

    #[test]
    fn test_compact_banner_short() {
        let theme = Theme::plain();
        let banner = compact_banner("0.1.0", &theme);
        // Compact banner should be shorter than welcome banner
        let welcome = welcome_banner("0.1.0", &theme);
        assert!(banner.len() < welcome.len());
    }

    #[test]
    fn test_goodbye_message_contains_terminated() {
        let theme = Theme::plain();
        let msg = goodbye_message(&theme);
        assert!(msg.contains("terminated"));
    }

    #[test]
    fn test_banner_with_different_versions() {
        let theme = Theme::plain();

        let banner1 = welcome_banner("0.0.1", &theme);
        assert!(banner1.contains("0.0.1"));

        let banner2 = welcome_banner("10.20.30", &theme);
        assert!(banner2.contains("10.20.30"));

        let banner3 = welcome_banner("1.0.0-beta", &theme);
        assert!(banner3.contains("1.0.0-beta"));
    }

    #[test]
    fn test_banner_contains_unified_queries() {
        let theme = Theme::plain();
        let banner = welcome_banner("0.1.0", &theme);
        assert!(banner.contains("Unified"));
    }
}
