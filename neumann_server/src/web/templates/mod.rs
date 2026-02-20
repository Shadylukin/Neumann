// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Maud HTML templates for the Memoria design system admin UI.

pub mod layout;

pub use layout::{
    format_bytes, format_number, layout, m_badge, m_breadcrumb, m_btn, m_card, m_card_interactive,
    m_empty, m_expandable_json, m_expandable_string, m_expandable_text, m_expandable_vector,
    m_header, m_loading, m_pagination, m_payload_preview, m_progress, m_section, m_stat,
    m_table_header, m_tabs,
};
