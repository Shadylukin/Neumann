// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Inline SVG icon system for the Memoria design system.
//!
//! Each function returns a `Markup` fragment containing a single `<svg>` element
//! with a 24x24 viewBox, `stroke="currentColor"`, stroke-width 1.5, and
//! round line caps / joins. Icons inherit color from their parent element.

use maud::{html, Markup, PreEscaped};

/// Wrap raw SVG path content into a standard 24x24 icon element.
fn icon_svg(inner: &str) -> Markup {
    html! {
        (PreEscaped(format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>"#
        )))
    }
}

/// Database / storage icon.
#[must_use]
pub fn icon_database() -> Markup {
    icon_svg(
        r#"<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>"#,
    )
}

/// Vector / embedding icon.
#[must_use]
pub fn icon_vector() -> Markup {
    icon_svg(
        r#"<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>"#,
    )
}

/// Graph / network icon.
#[must_use]
pub fn icon_graph() -> Markup {
    icon_svg(
        r#"<circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><path d="M8.5 7.5L15.5 16.5"/><path d="M15.5 7.5L8.5 16.5"/>"#,
    )
}

/// Contraction / tensor icon.
#[must_use]
pub fn icon_contraction() -> Markup {
    icon_svg(
        r#"<path d="M4 4h6v6H4z"/><path d="M14 4h6v6h-6z"/><path d="M4 14h6v6H4z"/><path d="M14 14h6v6h-6z"/><path d="M10 7h4"/><path d="M7 10v4"/><path d="M17 10v4"/><path d="M10 17h4"/>"#,
    )
}

/// Search / magnifying glass icon.
#[must_use]
pub fn icon_search() -> Markup {
    icon_svg(r#"<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>"#)
}

/// Table / grid icon.
#[must_use]
pub fn icon_table() -> Markup {
    icon_svg(
        r#"<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/>"#,
    )
}

/// Chart / bar chart icon.
#[must_use]
pub fn icon_chart() -> Markup {
    icon_svg(r#"<path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/>"#)
}

/// Settings / gear icon.
#[must_use]
pub fn icon_settings() -> Markup {
    icon_svg(
        r#"<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1.08 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.26.604.852.997 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1.08z"/>"#,
    )
}

/// Chevron right icon.
#[must_use]
pub fn icon_chevron_right() -> Markup {
    icon_svg(r#"<path d="M9 18l6-6-6-6"/>"#)
}

/// Chevron down icon.
#[must_use]
pub fn icon_chevron_down() -> Markup {
    icon_svg(r#"<path d="M6 9l6 6 6-6"/>"#)
}

/// Plus / add icon.
#[must_use]
pub fn icon_plus() -> Markup {
    icon_svg(r#"<path d="M12 5v14"/><path d="M5 12h14"/>"#)
}

/// Trash / delete icon.
#[must_use]
pub fn icon_trash() -> Markup {
    icon_svg(
        r#"<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>"#,
    )
}

/// Edit / pencil icon.
#[must_use]
pub fn icon_edit() -> Markup {
    icon_svg(r#"<path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>"#)
}

/// Copy / clipboard icon.
#[must_use]
pub fn icon_copy() -> Markup {
    icon_svg(
        r#"<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>"#,
    )
}

/// Check / success icon.
#[must_use]
pub fn icon_check() -> Markup {
    icon_svg(r#"<path d="M20 6L9 17l-5-5"/>"#)
}

/// X / close icon.
#[must_use]
pub fn icon_x() -> Markup {
    icon_svg(r#"<path d="M18 6L6 18"/><path d="M6 6l12 12"/>"#)
}

/// Arrow up icon.
#[must_use]
pub fn icon_arrow_up() -> Markup {
    icon_svg(r#"<path d="M12 19V5"/><path d="M5 12l7-7 7 7"/>"#)
}

/// Arrow down icon.
#[must_use]
pub fn icon_arrow_down() -> Markup {
    icon_svg(r#"<path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/>"#)
}

/// Filter / funnel icon.
#[must_use]
pub fn icon_filter() -> Markup {
    icon_svg(r#"<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z"/>"#)
}

/// Sort icon.
#[must_use]
pub fn icon_sort() -> Markup {
    icon_svg(
        r#"<path d="M11 5h10"/><path d="M11 9h7"/><path d="M11 13h4"/><path d="M3 17l3 3 3-3"/><path d="M6 18V4"/>"#,
    )
}

/// Eye / view icon.
#[must_use]
pub fn icon_eye() -> Markup {
    icon_svg(
        r#"<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>"#,
    )
}

/// Layers / stack icon.
#[must_use]
pub fn icon_layers() -> Markup {
    icon_svg(
        r#"<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>"#,
    )
}

/// Grid / layout icon.
#[must_use]
pub fn icon_grid() -> Markup {
    icon_svg(
        r#"<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>"#,
    )
}

/// Activity / pulse icon.
#[must_use]
pub fn icon_activity() -> Markup {
    icon_svg(r#"<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>"#)
}

/// Zap / lightning icon.
#[must_use]
pub fn icon_zap() -> Markup {
    icon_svg(r#"<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>"#)
}

/// Info / information icon.
#[must_use]
pub fn icon_info() -> Markup {
    icon_svg(r#"<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>"#)
}

/// Lock / padlock icon (vault).
#[must_use]
pub fn icon_lock() -> Markup {
    icon_svg(
        r#"<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>"#,
    )
}

/// Key icon (secrets).
#[must_use]
pub fn icon_key() -> Markup {
    icon_svg(
        r#"<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.78 7.78 5.5 5.5 0 0 1 7.78-7.78zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>"#,
    )
}

/// Shield icon (security / audit).
#[must_use]
pub fn icon_shield() -> Markup {
    icon_svg(r#"<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>"#)
}

/// Blob / archive-box icon.
#[must_use]
pub fn icon_blob() -> Markup {
    icon_svg(r#"<path d="M21 8v13H3V8"/><path d="M1 3h22v5H1z"/><path d="M10 12h4"/>"#)
}

/// Checkpoint / clock-rewind icon.
#[must_use]
pub fn icon_checkpoint() -> Markup {
    icon_svg(
        r#"<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/><path d="M4 4l2 2"/><path d="M20 4l-2 2"/>"#,
    )
}

/// Storage / hard-drive icon.
#[must_use]
pub fn icon_storage() -> Markup {
    icon_svg(
        r#"<path d="M22 12H2"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><path d="M6 16h.01"/><path d="M10 16h.01"/>"#,
    )
}

/// Chain / consensus icon (linked nodes).
#[must_use]
pub fn icon_chain() -> Markup {
    icon_svg(
        r#"<circle cx="6" cy="12" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><path d="M9 12h3"/><path d="M15 8l-3 4"/><path d="M15 16l-3-4"/>"#,
    )
}

/// Cache / layers-with-clock icon.
#[must_use]
pub fn icon_cache() -> Markup {
    icon_svg(
        r#"<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/><circle cx="19" cy="19" r="3" fill="none"/><path d="M19 18v1h1"/>"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all icon functions return valid SVG markup.
    macro_rules! test_icon {
        ($name:ident) => {
            #[test]
            fn $name() {
                let markup = super::$name();
                let html = markup.into_string();
                assert!(html.contains("<svg"), "icon must contain <svg");
                assert!(
                    html.contains("viewBox=\"0 0 24 24\""),
                    "icon must have 24x24 viewBox"
                );
                assert!(
                    html.contains("stroke=\"currentColor\""),
                    "icon must use currentColor"
                );
                assert!(
                    html.contains("stroke-width=\"1.5\""),
                    "icon must use stroke-width 1.5"
                );
            }
        };
    }

    test_icon!(icon_database);
    test_icon!(icon_vector);
    test_icon!(icon_graph);
    test_icon!(icon_contraction);
    test_icon!(icon_search);
    test_icon!(icon_table);
    test_icon!(icon_chart);
    test_icon!(icon_settings);
    test_icon!(icon_chevron_right);
    test_icon!(icon_chevron_down);
    test_icon!(icon_plus);
    test_icon!(icon_trash);
    test_icon!(icon_edit);
    test_icon!(icon_copy);
    test_icon!(icon_check);
    test_icon!(icon_x);
    test_icon!(icon_arrow_up);
    test_icon!(icon_arrow_down);
    test_icon!(icon_filter);
    test_icon!(icon_sort);
    test_icon!(icon_eye);
    test_icon!(icon_layers);
    test_icon!(icon_grid);
    test_icon!(icon_activity);
    test_icon!(icon_zap);
    test_icon!(icon_info);
    test_icon!(icon_lock);
    test_icon!(icon_key);
    test_icon!(icon_shield);
    test_icon!(icon_blob);
    test_icon!(icon_checkpoint);
    test_icon!(icon_storage);
    test_icon!(icon_chain);
    test_icon!(icon_cache);

    #[test]
    fn test_icon_svg_helper_wraps_content() {
        let result = icon_svg(r#"<path d="M0 0"/>"#);
        let html = result.into_string();
        assert!(html.contains(r#"<path d="M0 0"/>"#));
        assert!(html.contains("xmlns=\"http://www.w3.org/2000/svg\""));
    }
}
