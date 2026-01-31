// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unicode and ASCII icons for terminal output.

/// Icon set for terminal display.
///
/// Provides both Unicode and ASCII variants for terminal compatibility.
#[derive(Debug, Clone, Copy)]
pub struct Icons {
    pub success: &'static str,
    pub error: &'static str,
    pub warning: &'static str,
    pub info: &'static str,
    pub table: &'static str,
    pub node: &'static str,
    pub edge: &'static str,
    pub vector: &'static str,
    pub blob: &'static str,
    pub key: &'static str,
    pub chain: &'static str,
    pub bullet: &'static str,
    pub arrow: &'static str,
    pub check: &'static str,
    pub cross: &'static str,
    pub spinner: &'static str,
}

impl Icons {
    /// Unicode icons for terminals with good Unicode support.
    pub const UNICODE: Self = Self {
        success: "\u{2714}", // checkmark
        error: "\u{2718}",   // X mark
        warning: "\u{26A0}", // warning triangle
        info: "\u{2139}",    // info circle
        table: "\u{2630}",   // trigram
        node: "\u{25CF}",    // filled circle
        edge: "\u{2192}",    // right arrow
        vector: "\u{2248}",  // approximately equal
        blob: "\u{25A0}",    // filled square
        key: "\u{1F511}",    // key emoji (fallback to ASCII in most contexts)
        chain: "\u{26D3}",   // chains
        bullet: "\u{2022}",  // bullet
        arrow: "\u{2192}",   // right arrow
        check: "\u{2714}",   // checkmark
        cross: "\u{2718}",   // X mark
        spinner: "\u{25CF}", // filled circle
    };

    /// ASCII-only icons for maximum compatibility.
    pub const ASCII: Self = Self {
        success: "[ok]",
        error: "[!!]",
        warning: "[!]",
        info: "[i]",
        table: "[#]",
        node: "(o)",
        edge: "->",
        vector: "[~]",
        blob: "[@]",
        key: "[*]",
        chain: "[=]",
        bullet: "*",
        arrow: "->",
        check: "[ok]",
        cross: "[x]",
        spinner: "*",
    };

    /// Auto-detect which icon set to use based on terminal capabilities.
    #[must_use]
    pub const fn auto() -> &'static Self {
        // For now, prefer ASCII for maximum compatibility
        // In the future, could detect Unicode support via console crate
        &Self::ASCII
    }
}

impl Default for &Icons {
    fn default() -> Self {
        Icons::auto()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_icons_not_empty() {
        assert!(!Icons::UNICODE.success.is_empty());
        assert!(!Icons::UNICODE.error.is_empty());
        assert!(!Icons::UNICODE.node.is_empty());
    }

    #[test]
    fn test_ascii_icons_not_empty() {
        assert!(!Icons::ASCII.success.is_empty());
        assert!(!Icons::ASCII.error.is_empty());
        assert!(!Icons::ASCII.node.is_empty());
    }

    #[test]
    fn test_auto_returns_valid_icons() {
        let icons = Icons::auto();
        assert!(!icons.success.is_empty());
    }
}
