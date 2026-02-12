// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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

    /// Returns ASCII icons for plain/no-color mode.
    #[must_use]
    pub const fn plain() -> &'static Self {
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

    #[test]
    fn test_plain_returns_ascii_icons() {
        let icons = Icons::plain();
        assert!(!icons.success.is_empty());
        // Plain should return ASCII icons
        assert_eq!(icons.success, Icons::ASCII.success);
    }

    #[test]
    fn test_default_impl() {
        let icons: &Icons = Default::default();
        assert!(!icons.success.is_empty());
    }

    #[test]
    fn test_all_unicode_icons_fields() {
        let icons = &Icons::UNICODE;
        assert!(!icons.warning.is_empty());
        assert!(!icons.info.is_empty());
        assert!(!icons.table.is_empty());
        assert!(!icons.edge.is_empty());
        assert!(!icons.vector.is_empty());
        assert!(!icons.blob.is_empty());
        assert!(!icons.key.is_empty());
        assert!(!icons.chain.is_empty());
        assert!(!icons.bullet.is_empty());
        assert!(!icons.arrow.is_empty());
        assert!(!icons.check.is_empty());
        assert!(!icons.cross.is_empty());
        assert!(!icons.spinner.is_empty());
    }

    #[test]
    fn test_all_ascii_icons_fields() {
        let icons = &Icons::ASCII;
        assert!(!icons.warning.is_empty());
        assert!(!icons.info.is_empty());
        assert!(!icons.table.is_empty());
        assert!(!icons.edge.is_empty());
        assert!(!icons.vector.is_empty());
        assert!(!icons.blob.is_empty());
        assert!(!icons.key.is_empty());
        assert!(!icons.chain.is_empty());
        assert!(!icons.bullet.is_empty());
        assert!(!icons.arrow.is_empty());
        assert!(!icons.check.is_empty());
        assert!(!icons.cross.is_empty());
        assert!(!icons.spinner.is_empty());
    }

    #[test]
    fn test_icons_debug() {
        let icons = Icons::ASCII;
        let debug_str = format!("{icons:?}");
        assert!(debug_str.contains("Icons"));
    }

    #[test]
    fn test_icons_clone() {
        let icons = Icons::ASCII;
        let cloned = icons;
        assert_eq!(icons.success, cloned.success);
    }
}
