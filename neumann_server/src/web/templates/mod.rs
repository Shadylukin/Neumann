// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maud HTML templates for the dystopian terminal admin UI.

pub mod layout;

pub use layout::{
    breadcrumb, btn_terminal, empty_state, engine_section, expandable_json,
    expandable_payload_preview, expandable_string, expandable_text, expandable_vector,
    format_number, layout, loading_indicator, page_header, progress_bar, stat_card, table_header,
    terminal_panel, terminal_panel_rust,
};
