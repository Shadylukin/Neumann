// SPDX-License-Identifier: MIT OR Apache-2.0
//! Base layout template for the Memoria design system admin UI.
//!
//! Features a monochromatic dark-themed interface with opacity-based hierarchy,
//! blur-to-clear animations, and neutral color palette.

use maud::{html, Markup, PreEscaped, DOCTYPE};

use crate::web::assets::{ADMIN_CSS, MEMORIA_SCRIPT, TAILWIND_CONFIG};
use crate::web::icons::{
    icon_blob, icon_cache, icon_chain, icon_checkpoint, icon_contraction, icon_database,
    icon_graph, icon_key, icon_storage, icon_vector, icon_zap,
};
use crate::web::NavItem;

/// Render the base HTML layout with Memoria design system styling.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn layout(title: &str, active: NavItem, content: Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" class="dark" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (title) " | NEUMANN" }

                // Tailwind CSS via CDN
                script src="https://cdn.tailwindcss.com" {}
                script { (PreEscaped(TAILWIND_CONFIG)) }

                // HTMX for interactivity
                script src="https://unpkg.com/htmx.org@1.9.12" defer {}

                // Memoria Fonts from Google Fonts
                link rel="preconnect" href="https://fonts.googleapis.com";
                link rel="preconnect" href="https://fonts.gstatic.com" crossorigin;
                link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400&family=JetBrains+Mono:wght@400&display=swap";

                // Memoria design system styles
                style { (PreEscaped(ADMIN_CSS)) }
            }
            body class="bg-neutral-950 text-white font-sans" {
                div class="min-h-screen flex flex-col" {
                    // Sidebar (desktop)
                    (sidebar(active))

                    // Main content area
                    div id="main-content" class="lg:ml-64 flex-1 flex flex-col" {
                        // Main content
                        main class="flex-1 p-4 lg:p-6" {
                            div class="max-w-7xl mx-auto" {
                                (content)
                            }
                        }
                    }
                }

                // Bottom nav (mobile)
                (mobile_nav(active))

                // Keyboard help overlay
                (keyboard_help())

                // Keyboard navigation script
                script { (PreEscaped(KEYBOARD_NAV_SCRIPT)) }

                // Memoria interactions script
                script { (PreEscaped(MEMORIA_SCRIPT)) }
            }
        }
    }
}

/// Keyboard navigation JavaScript with vim-style enhancements.
const KEYBOARD_NAV_SCRIPT: &str = r#"
(function() {
    'use strict';

    // State for g-prefix mode
    let gPrefixMode = false;
    let gPrefixTimeout = null;
    let selectedRowIndex = -1;

    // Clear g-prefix mode after timeout
    function clearGPrefix() {
        gPrefixMode = false;
        if (gPrefixTimeout) {
            clearTimeout(gPrefixTimeout);
            gPrefixTimeout = null;
        }
        updateGPrefixIndicator(false);
    }

    // Set g-prefix mode with timeout
    function setGPrefix() {
        gPrefixMode = true;
        updateGPrefixIndicator(true);
        gPrefixTimeout = setTimeout(clearGPrefix, 1500);
    }

    // Update visual indicator for g-prefix mode
    function updateGPrefixIndicator(active) {
        const indicator = document.getElementById('g-prefix-indicator');
        if (indicator) {
            indicator.classList.toggle('hidden', !active);
        }
    }

    // Get all navigable rows in the current table
    function getTableRows() {
        const table = document.querySelector('.data-table tbody, .m-table tbody');
        return table ? Array.from(table.querySelectorAll('tr')) : [];
    }

    // Select a row by index
    function selectRow(index) {
        const rows = getTableRows();
        if (rows.length === 0) return;

        // Clamp index
        index = Math.max(0, Math.min(rows.length - 1, index));

        // Deselect previous
        rows.forEach(r => r.classList.remove('row-selected'));

        // Select new
        rows[index].classList.add('row-selected');
        rows[index].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        selectedRowIndex = index;
    }

    // Navigate to next row
    function selectNextRow() {
        const rows = getTableRows();
        if (rows.length === 0) return;
        selectRow(selectedRowIndex < 0 ? 0 : selectedRowIndex + 1);
    }

    // Navigate to previous row
    function selectPrevRow() {
        const rows = getTableRows();
        if (rows.length === 0) return;
        selectRow(selectedRowIndex < 0 ? 0 : selectedRowIndex - 1);
    }

    // Navigate to first row
    function selectFirstRow() {
        selectRow(0);
    }

    // Navigate to last row
    function selectLastRow() {
        const rows = getTableRows();
        if (rows.length > 0) {
            selectRow(rows.length - 1);
        }
    }

    // Expand/click selected row
    function expandSelectedRow() {
        const rows = getTableRows();
        if (selectedRowIndex >= 0 && selectedRowIndex < rows.length) {
            const row = rows[selectedRowIndex];
            // Click the row link if it exists
            const link = row.querySelector('a');
            if (link) {
                link.click();
            } else {
                // Or dispatch click event on row
                row.click();
            }
        }
    }

    // Focus search input
    function focusSearch() {
        const search = document.querySelector(
            'input[type="search"], input[type="text"][placeholder*="earch"], ' +
            '#terminal-input, input.m-terminal-input-field'
        );
        if (search) {
            search.focus();
            search.select();
            return true;
        }
        return false;
    }

    // Navigate to previous/next page
    function navigatePage(direction) {
        const links = document.querySelectorAll('a[href*="page="], a[href*="offset="]');
        for (const link of links) {
            const text = link.textContent.toLowerCase();
            if (direction === 'prev' && (text.includes('prev') || text.includes('<'))) {
                link.click();
                return;
            }
            if (direction === 'next' && (text.includes('next') || text.includes('>'))) {
                link.click();
                return;
            }
        }
    }

    // Main keydown handler
    document.addEventListener('keydown', function(e) {
        // Skip if user is typing in a form field (unless Escape)
        const tag = e.target.tagName;
        const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable;

        if (isInput && e.key !== 'Escape') return;

        const key = e.key;
        const keyLower = key.toLowerCase();

        // Handle Escape - close any open modal or clear g-prefix
        if (key === 'Escape') {
            if (gPrefixMode) {
                clearGPrefix();
                e.preventDefault();
                return;
            }

            const nodeModal = document.getElementById('node-modal');
            const helpModal = document.getElementById('keyboard-help');
            if (nodeModal && !nodeModal.classList.contains('hidden')) {
                if (window.Memoria) {
                    window.Memoria.closeModal(nodeModal);
                } else {
                    nodeModal.classList.add('hidden');
                    nodeModal.classList.remove('flex');
                }
                e.preventDefault();
                return;
            }
            if (helpModal && !helpModal.classList.contains('hidden')) {
                helpModal.classList.add('hidden');
                e.preventDefault();
                return;
            }

            // Blur any focused input
            if (isInput) {
                e.target.blur();
                e.preventDefault();
                return;
            }
            return;
        }

        // Skip if modal is open
        const nodeModal = document.getElementById('node-modal');
        const helpModal = document.getElementById('keyboard-help');
        if ((nodeModal && !nodeModal.classList.contains('hidden')) ||
            (helpModal && !helpModal.classList.contains('hidden'))) {
            return;
        }

        // Handle g-prefix mode
        if (gPrefixMode) {
            clearGPrefix();
            e.preventDefault();

            switch(keyLower) {
                case 'g': selectFirstRow(); break;
                case 'a': window.location.href = '/graph/algorithms'; break;
                case 'm': window.location.href = '/metrics'; break;
                case 'd': window.location.href = '/'; break;
                case 'r': window.location.href = '/relational'; break;
                case 'v': window.location.href = '/vector'; break;
                case 'h': window.location.href = '/graph'; break;
                case 't': window.location.href = '/contraction'; break;
                case 's': window.location.href = '/vault'; break;
                case 'c': window.location.href = '/cache'; break;
                case 'b': window.location.href = '/blob'; break;
                case 'p': window.location.href = '/checkpoint'; break;
                case 'i': window.location.href = '/storage'; break;
                case 'n': window.location.href = '/chain'; break;
            }
            return;
        }

        // Navigation shortcuts
        switch(key) {
            // Vim-style row navigation
            case 'j':
                selectNextRow();
                e.preventDefault();
                break;

            case 'k':
                selectPrevRow();
                e.preventDefault();
                break;

            // Go to end of list
            case 'G':
                selectLastRow();
                e.preventDefault();
                break;

            // Search focus
            case '/':
                if (focusSearch()) {
                    e.preventDefault();
                }
                break;

            // Enter to expand selected row
            case 'Enter':
                expandSelectedRow();
                break;

            // Page navigation
            case '[':
                navigatePage('prev');
                e.preventDefault();
                break;

            case ']':
                navigatePage('next');
                e.preventDefault();
                break;

            default:
                // Lowercase navigation
                switch(keyLower) {
                    case 'g':
                        // Enter g-prefix mode
                        setGPrefix();
                        e.preventDefault();
                        break;

                    case 'v':
                        window.location.href = '/vector';
                        break;

                    case 'r':
                        window.location.href = '/relational';
                        break;

                    case 'd':
                        window.location.href = '/';
                        break;

                    case 's':
                        window.location.href = '/vault';
                        break;

                    case 'c':
                        window.location.href = '/cache';
                        break;

                    case 't':
                        window.location.href = '/contraction';
                        break;

                    case 'b':
                        window.location.href = '/blob';
                        break;

                    case 'p':
                        window.location.href = '/checkpoint';
                        break;

                    case 'i':
                        window.location.href = '/storage';
                        break;

                    case 'n':
                        window.location.href = '/chain';
                        break;

                    case '?':
                        // Show keyboard help
                        const help = document.getElementById('keyboard-help');
                        if (help) {
                            help.classList.toggle('hidden');
                            e.preventDefault();
                        }
                        break;
                }
        }
    }, true);

    // Inject CSS for selected row
    const style = document.createElement('style');
    style.textContent = `
        .row-selected {
            background-color: rgba(255, 255, 255, 0.08) !important;
            border-left: 2px solid var(--border-emphasis) !important;
        }
        .row-selected td {
            color: var(--text-primary) !important;
        }
        #g-prefix-indicator {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-emphasis);
            padding: 0.5rem 1rem;
            font-family: var(--font-mono);
            color: var(--text-primary);
            border-radius: 8px;
            z-index: 100;
        }
    `;
    document.head.appendChild(style);

    // Create g-prefix indicator
    const indicator = document.createElement('div');
    indicator.id = 'g-prefix-indicator';
    indicator.className = 'hidden';
    indicator.textContent = 'g-';
    document.body.appendChild(indicator);
})();
"#;

/// Keyboard help overlay.
fn keyboard_help() -> Markup {
    html! {
        div id="keyboard-help" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-black/90" onclick="this.classList.add('hidden')" {
            div class="m-card w-[32rem] max-w-[90vw]" onclick="event.stopPropagation()" {
                div class="m-card-header" { "KEYBOARD SHORTCUTS" }
                div class="m-card-content" {
                    div class="space-y-4 text-sm" {
                        // Navigation
                        div {
                            div class="text-neutral-400 mb-2 uppercase tracking-wider text-xs" { "NAVIGATION" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="m-kbd mr-2" { "D" } span class="text-neutral-400" { "Dashboard" } }
                                div { span class="m-kbd mr-2" { "V" } span class="text-neutral-400" { "Vector" } }
                                div { span class="m-kbd mr-2" { "R" } span class="text-neutral-400" { "Relational" } }
                                div { span class="m-kbd mr-2" { "S" } span class="text-neutral-400" { "Vault" } }
                                div { span class="m-kbd mr-2" { "C" } span class="text-neutral-400" { "Cache" } }
                                div { span class="m-kbd mr-2" { "B" } span class="text-neutral-400" { "Blob" } }
                                div { span class="m-kbd mr-2" { "P" } span class="text-neutral-400" { "Checkpoint" } }
                                div { span class="m-kbd mr-2" { "I" } span class="text-neutral-400" { "Storage" } }
                                div { span class="m-kbd mr-2" { "T" } span class="text-neutral-400" { "Contraction" } }
                                div { span class="m-kbd mr-2" { "N" } span class="text-neutral-400" { "Chain" } }
                            }
                        }
                        // G-prefix navigation
                        div {
                            div class="text-neutral-400 mb-2 uppercase tracking-wider text-xs" { "G-PREFIX" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="m-kbd mr-2" { "g+g" } span class="text-neutral-400" { "Top of list" } }
                                div { span class="m-kbd mr-2" { "g+h" } span class="text-neutral-400" { "Graph engine" } }
                                div { span class="m-kbd mr-2" { "g+a" } span class="text-neutral-400" { "Algorithms" } }
                                div { span class="m-kbd mr-2" { "g+t" } span class="text-neutral-400" { "Contraction" } }
                                div { span class="m-kbd mr-2" { "g+s" } span class="text-neutral-400" { "Vault" } }
                                div { span class="m-kbd mr-2" { "g+c" } span class="text-neutral-400" { "Cache" } }
                                div { span class="m-kbd mr-2" { "g+m" } span class="text-neutral-400" { "Metrics" } }
                            }
                        }
                        // Table navigation
                        div {
                            div class="text-neutral-400 mb-2 uppercase tracking-wider text-xs" { "TABLE NAVIGATION" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="m-kbd mr-2" { "j" } span class="text-neutral-400" { "Next row" } }
                                div { span class="m-kbd mr-2" { "k" } span class="text-neutral-400" { "Previous row" } }
                                div { span class="m-kbd mr-2" { "G" } span class="text-neutral-400" { "Last row" } }
                                div { span class="m-kbd mr-2" { "Enter" } span class="text-neutral-400" { "Expand row" } }
                                div { span class="m-kbd mr-2" { "[" } span class="text-neutral-400" { "Previous page" } }
                                div { span class="m-kbd mr-2" { "]" } span class="text-neutral-400" { "Next page" } }
                            }
                        }
                        // General
                        div {
                            div class="text-neutral-400 mb-2 uppercase tracking-wider text-xs" { "GENERAL" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="m-kbd mr-2" { "/" } span class="text-neutral-400" { "Focus search" } }
                                div { span class="m-kbd mr-2" { "?" } span class="text-neutral-400" { "This help" } }
                                div { span class="m-kbd mr-2" { "Esc" } span class="text-neutral-400" { "Close/unfocus" } }
                            }
                        }
                    }
                }
                div class="m-card-footer text-center text-neutral-500" {
                    "Press ? or Esc to close"
                }
            }
        }
    }
}

/// Render a page header with title and optional description.
#[must_use]
pub fn m_header(title: &str, description: Option<&str>) -> Markup {
    html! {
        div class="mb-6" {
            h1 class="text-2xl text-white tracking-wider uppercase" style="font-weight: 300;" {
                (title)
            }
            @if let Some(desc) = description {
                p class="text-neutral-400 mt-1" { (desc) }
            }
        }
    }
}

/// Render the desktop sidebar with navigation.
fn sidebar(active: NavItem) -> Markup {
    html! {
        aside class="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col bg-neutral-900 border-r border-neutral-800" {
            // Wordmark
            div class="px-4 pt-4 pb-3 border-b border-neutral-800" {
                div class="text-white text-2xl tracking-[0.2em] uppercase" style="font-weight: 100; opacity: 0.8;" {
                    "NEUMANN"
                }
                div class="text-neutral-500 text-xs tracking-[0.15em] mt-1" {
                    "TENSOR DATABASE"
                }
            }

            // Navigation
            nav class="flex-1 px-2 py-4 m-nav stagger-container" {
                div class="mb-2 px-2" {
                    span class="text-xs text-neutral-500 tracking-widest" {
                        "NAVIGATION"
                    }
                }

                (nav_item("/", "D", "DASHBOARD", &icon_zap(), active == NavItem::Dashboard))

                div class="mt-4 mb-2 px-2" {
                    span class="text-xs text-neutral-500 tracking-widest" {
                        "ENGINES"
                    }
                }

                (nav_item("/graph", "G", "GRAPH", &icon_graph(), active == NavItem::Graph))
                (nav_item("/vector", "V", "VECTOR", &icon_vector(), active == NavItem::Vector))
                (nav_item("/relational", "R", "RELATIONAL", &icon_database(), active == NavItem::Relational))

                div class="mt-4 mb-2 px-2" {
                    span class="text-xs text-neutral-500 tracking-widest" {
                        "STORAGE"
                    }
                }

                (nav_item("/vault", "S", "VAULT", &icon_key(), active == NavItem::Vault))
                (nav_item("/cache", "C", "CACHE", &icon_cache(), active == NavItem::Cache))
                (nav_item("/blob", "B", "BLOB", &icon_blob(), active == NavItem::Blob))
                (nav_item("/checkpoint", "P", "CHECKPOINT", &icon_checkpoint(), active == NavItem::Checkpoint))

                div class="mt-4 mb-2 px-2" {
                    span class="text-xs text-neutral-500 tracking-widest" {
                        "INTERNALS"
                    }
                }

                (nav_item("/storage", "I", "STORAGE", &icon_storage(), active == NavItem::Storage))
                (nav_item("/chain", "N", "CHAIN", &icon_chain(), active == NavItem::Chain))

                div class="mt-4 mb-2 px-2" {
                    span class="text-xs text-neutral-500 tracking-widest" {
                        "ANALYSIS"
                    }
                }

                (nav_item("/contraction", "T", "CONTRACTION", &icon_contraction(), active == NavItem::Contraction))
            }

            // Keyboard hints — clickable to open full help overlay
            div class="px-4 py-3 border-t border-neutral-800" {
                div class="flex items-center justify-between cursor-pointer group"
                    onclick="document.getElementById('keyboard-help').classList.toggle('hidden')" {
                    span class="text-xs text-neutral-500 tracking-widest uppercase group-hover:text-neutral-300 transition-colors" {
                        "KEYBOARD SHORTCUTS"
                    }
                    span class="m-kbd text-neutral-600 group-hover:text-neutral-400 transition-colors" { "?" }
                }
                div class="mt-2 grid grid-cols-2 gap-1 text-xs" {
                    span { span class="m-kbd" { "D" } span class="text-neutral-400" { "Dash" } }
                    span { span class="m-kbd" { "G" } span class="text-neutral-400" { "Graph" } }
                    span { span class="m-kbd" { "V" } span class="text-neutral-400" { "Vector" } }
                    span { span class="m-kbd" { "R" } span class="text-neutral-400" { "Rel" } }
                    span { span class="m-kbd" { "S" } span class="text-neutral-400" { "Vault" } }
                    span { span class="m-kbd" { "C" } span class="text-neutral-400" { "Cache" } }
                    span { span class="m-kbd" { "B" } span class="text-neutral-400" { "Blob" } }
                    span { span class="m-kbd" { "P" } span class="text-neutral-400" { "Chkpt" } }
                    span { span class="m-kbd" { "I" } span class="text-neutral-400" { "Store" } }
                    span { span class="m-kbd" { "N" } span class="text-neutral-400" { "Chain" } }
                    span { span class="m-kbd" { "T" } span class="text-neutral-400" { "Contr" } }
                }
            }
        }
    }
}

/// Render the mobile bottom navigation.
fn mobile_nav(active: NavItem) -> Markup {
    html! {
        nav class="lg:hidden fixed bottom-0 left-0 right-0 bg-neutral-900 border-t border-neutral-800 px-2 py-1" {
            div class="flex justify-around" {
                (mobile_nav_item("/", "DASH", &icon_zap(), active == NavItem::Dashboard))
                (mobile_nav_item("/graph", "GRAPH", &icon_graph(), active == NavItem::Graph))
                (mobile_nav_item("/vector", "VEC", &icon_vector(), active == NavItem::Vector))
                (mobile_nav_item("/relational", "REL", &icon_database(), active == NavItem::Relational))
                (mobile_nav_item("/vault", "VAULT", &icon_key(), active == NavItem::Vault))
                (mobile_nav_item("/cache", "CACHE", &icon_cache(), active == NavItem::Cache))
                (mobile_nav_item("/blob", "BLOB", &icon_blob(), active == NavItem::Blob))
                (mobile_nav_item("/chain", "CHAIN", &icon_chain(), active == NavItem::Chain))
                (mobile_nav_item("/contraction", "TENS", &icon_contraction(), active == NavItem::Contraction))
            }
        }
    }
}

/// Render a navigation item with icon and keyboard hint.
fn nav_item(href: &str, key: &str, label: &str, icon: &Markup, is_active: bool) -> Markup {
    let active_class = if is_active { "active" } else { "" };

    html! {
        a href=(href) class=(format!("flex items-center gap-2 px-3 py-2 my-1 stagger-item {active_class}")) {
            span class="m-icon-sm inline-flex" { (icon) }
            span { (label) }
            span class="m-kbd ml-auto" { (key) }
        }
    }
}

/// Render a mobile navigation item with SVG icon.
fn mobile_nav_item(href: &str, label: &str, icon: &Markup, is_active: bool) -> Markup {
    let active_class = if is_active {
        "text-white"
    } else {
        "text-neutral-500"
    };

    html! {
        a href=(href) class=(format!("flex flex-col items-center gap-1 px-3 py-2 {active_class}")) {
            span class="m-icon-sm inline-flex" { (icon) }
            span class="text-[10px] tracking-wider" { (label) }
        }
    }
}

/// Render a stat card with opacity-based hierarchy and animated counter.
#[must_use]
pub fn m_stat(label: &str, value: &str, subtitle: &str, _engine: &str) -> Markup {
    html! {
        div class="m-stat" {
            div class="m-stat-label" { (label) }
            div class="m-stat-value" data-counter=(value) { (value) }
            div class="m-stat-subtitle" { (subtitle) }
        }
    }
}

/// Render a Memoria card panel.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn m_card(title: &str, content: Markup) -> Markup {
    html! {
        div class="m-card" {
            div class="m-card-header" { (title) }
            div class="m-card-content" {
                (content)
            }
        }
    }
}

/// Render an interactive card with hover effects.
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn m_card_interactive(title: &str, content: Markup) -> Markup {
    html! {
        div class="m-card-interactive" {
            div class="m-card-header" { (title) }
            div class="m-card-content" {
                (content)
            }
        }
    }
}

/// Render a Memoria badge.
#[must_use]
pub fn m_badge(text: &str) -> Markup {
    html! {
        span class="m-badge" { (text) }
    }
}

/// Render Memoria tabs.
///
/// Each item is `(label, href, is_active)`.
#[must_use]
pub fn m_tabs(items: &[(&str, &str, bool)]) -> Markup {
    html! {
        div class="m-tabs" {
            @for (label, href, active) in items {
                a href=(*href) class=(if *active { "m-tab active" } else { "m-tab" }) {
                    (*label)
                }
            }
        }
    }
}

/// Render a section card for the dashboard.
#[must_use]
pub fn m_section(title: &str, _engine: &str, items: &[(String, String)]) -> Markup {
    html! {
        div class="m-card" {
            div class="m-card-header" { (title) }
            div class="m-card-content" {
                @if items.is_empty() {
                    div class="text-neutral-500 text-sm" {
                        "No entries"
                    }
                } @else {
                    ul class="space-y-2" {
                        @for (name, count) in items {
                            li class="flex items-center justify-between" {
                                span class="text-white" { (name) }
                                span class="text-neutral-400 font-mono" { (count) }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Render an empty state message with lamp container.
#[must_use]
pub fn m_empty(title: &str, description: &str) -> Markup {
    html! {
        div class="text-center py-12 m-card m-lamp" {
            div class="text-5xl text-neutral-600 mb-4" style="font-weight: 100;" { "---" }
            h3 class="text-lg text-white" style="font-weight: 300;" { (title) }
            p class="text-sm text-neutral-400 mt-1" { (description) }
        }
    }
}

/// Format a byte count as a human-readable string (B/KB/MB/GB).
#[must_use]
#[allow(clippy::cast_precision_loss)] // Precision loss acceptable for display formatting
pub fn format_bytes(n: usize) -> String {
    if n < 1024 {
        return format!("{n} B");
    }
    if n < 1_048_576 {
        let kb = n as f64 / 1024.0;
        return format!("{kb:.1} KB");
    }
    if n < 1_073_741_824 {
        let mb = n as f64 / 1_048_576.0;
        return format!("{mb:.1} MB");
    }
    let gb = n as f64 / 1_073_741_824.0;
    format!("{gb:.2} GB")
}

/// Format a number with thousands separators.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Precision loss acceptable for display formatting
pub fn format_number(n: usize) -> String {
    if n < 1000 {
        return n.to_string();
    }
    if n < 1_000_000 {
        let k = n as f64 / 1000.0;
        return format!("{k:.1}K");
    }
    let m = n as f64 / 1_000_000.0;
    format!("{m:.1}M")
}

/// Render a breadcrumb trail for navigation.
#[must_use]
pub fn m_breadcrumb(items: &[(&str, &str)]) -> Markup {
    html! {
        nav class="m-breadcrumb mb-4" {
            @for (i, (href, label)) in items.iter().enumerate() {
                @if i > 0 {
                    span class="separator" { "/" }
                }
                @if i == items.len() - 1 {
                    span class="text-white" { (label) }
                } @else {
                    a href=(href) { (label) }
                }
            }
        }
    }
}

/// Render a Memoria-styled button.
#[must_use]
pub fn m_btn(label: &str, href: Option<&str>) -> Markup {
    href.map_or_else(
        || {
            html! {
                button type="submit" class="m-btn" { (label) }
            }
        },
        |h| {
            html! {
                a href=(h) class="m-btn inline-block" { (label) }
            }
        },
    )
}

/// Render a table header row.
#[must_use]
pub fn m_table_header(columns: &[&str]) -> Markup {
    html! {
        thead {
            tr {
                @for col in columns {
                    th { (col) }
                }
            }
        }
    }
}

/// Render expandable text content.
#[must_use]
pub fn m_expandable_text(content: &str, max_chars: usize, color_class: &str) -> Markup {
    if content.len() <= max_chars {
        return html! {
            span class=(color_class) { (content) }
        };
    }

    let preview = &content[..max_chars.min(content.len())];

    html! {
        span class="m-expandable" {
            span class="preview" {
                span class=(color_class) { (preview) "..." }
                span class="m-expand-btn ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.nextElementSibling.style.display='inline'" {
                    "more"
                }
            }
            span class="full" style="display:none" {
                span class=(format!("{color_class} whitespace-pre-wrap")) { (content) }
                span class="m-expand-btn ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.previousElementSibling.style.display='inline'" {
                    "less"
                }
            }
        }
    }
}

/// Render expandable text with quoted string formatting.
#[must_use]
pub fn m_expandable_string(content: &str, max_chars: usize) -> Markup {
    if content.len() <= max_chars {
        return html! {
            span class="text-neutral-300" { "\"" (content) "\"" }
        };
    }

    let preview = &content[..max_chars.min(content.len())];

    html! {
        span class="m-expandable" {
            span class="preview" {
                span class="text-neutral-300" { "\"" (preview) "...\"" }
                span class="m-expand-btn ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.nextElementSibling.style.display='inline'" {
                    "more"
                }
            }
            span class="full" style="display:none" {
                span class="text-neutral-300 whitespace-pre-wrap" { "\"" (content) "\"" }
                span class="m-expand-btn ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.previousElementSibling.style.display='inline'" {
                    "less"
                }
            }
        }
    }
}

/// Render expandable JSON content.
#[must_use]
pub fn m_expandable_json(content: &str, max_chars: usize) -> Markup {
    if content.len() <= max_chars {
        return html! {
            span class="text-white font-mono text-sm" { (content) }
        };
    }

    let preview = &content[..max_chars.min(content.len())];

    html! {
        span class="m-expandable" {
            span class="preview" {
                span class="text-white font-mono text-sm" { (preview) "..." }
                span class="m-expand-btn ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.nextElementSibling.style.display='inline'" {
                    "more"
                }
            }
            span class="full" style="display:none" {
                span class="text-white font-mono text-sm whitespace-pre-wrap" { (content) }
                span class="m-expand-btn ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.previousElementSibling.style.display='inline'" {
                    "less"
                }
            }
        }
    }
}

/// Render expandable vector content with dimension info.
#[must_use]
pub fn m_expandable_vector(vec: &[f32], max_preview: usize) -> Markup {
    let total = vec.len();

    if total <= max_preview {
        let formatted = vec
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ");
        return html! {
            span class="font-mono text-xs text-neutral-400" { "[" (formatted) "]" }
        };
    }

    // Show first few and last few
    let first: Vec<String> = vec
        .iter()
        .take(max_preview / 2)
        .map(|v| format!("{v:.6}"))
        .collect();
    let last: Vec<String> = vec
        .iter()
        .rev()
        .take(max_preview / 4)
        .rev()
        .map(|v| format!("{v:.6}"))
        .collect();
    let hidden = total - first.len() - last.len();
    let preview = format!(
        "[{}, ... ({hidden} more) ..., {}]",
        first.join(", "),
        last.join(", ")
    );

    // Full formatted vector for expansion
    let full_formatted = vec
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i > 0 && i % 10 == 0 {
                format!("\n  {v:.6}")
            } else {
                format!("{v:.6}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    html! {
        details class="m-expandable-details" {
            summary class="cursor-pointer list-none" {
                span class="font-mono text-xs text-neutral-400" { (preview) }
                span class="m-expand-btn ml-2" {
                    "show all " (total)
                }
            }
            div class="mt-2 p-3 bg-neutral-950 border border-neutral-800 rounded-lg max-h-96 overflow-auto" {
                div class="flex items-center justify-between mb-2 pb-2 border-b border-neutral-800" {
                    span class="text-xs text-neutral-500 font-mono" { (total) " dimensions" }
                    span
                        class="m-expand-btn cursor-pointer"
                        onclick="navigator.clipboard.writeText(this.closest('details').querySelector('pre').textContent)"
                    {
                        "copy"
                    }
                }
                pre class="font-mono text-xs text-neutral-400 whitespace-pre-wrap" {
                    "[" (full_formatted) "]"
                }
            }
        }
    }
}

/// Render expandable payload preview.
#[must_use]
pub fn m_payload_preview(items: &[(String, String)], max_items: usize) -> Markup {
    if items.len() <= max_items {
        let preview = items
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join(", ");
        return html! {
            span class="font-mono text-xs" { "{ " (preview) " }" }
        };
    }

    let shown: Vec<String> = items
        .iter()
        .take(max_items)
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    let hidden_count = items.len() - max_items;

    html! {
        details class="m-expandable-details" {
            summary class="cursor-pointer list-none" {
                span class="font-mono text-xs" { "{ " (shown.join(", ")) " " }
                span class="m-expand-btn" {
                    "+" (hidden_count) " more"
                }
                span class="font-mono text-xs" { " }" }
            }
            div class="mt-2 p-3 bg-neutral-950 border border-neutral-800 rounded-lg" {
                dl class="space-y-1" {
                    @for (key, value) in items {
                        div class="flex gap-2" {
                            dt class="font-mono text-xs text-neutral-400 min-w-[100px]" { (key) ":" }
                            dd class="font-mono text-xs text-white break-all" { (value) }
                        }
                    }
                }
            }
        }
    }
}

/// Render a loading indicator with blur-to-clear animation.
#[must_use]
pub fn m_loading(text: &str) -> Markup {
    html! {
        div class="text-center py-4 animate-blur-reveal" {
            span class="text-neutral-400" { (text) }
        }
    }
}

/// Render a pagination control.
///
/// Generates Previous / page numbers / Next buttons with `m-btn` styling.
/// Handles edge cases: page 1 has no previous, last page has no next.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn m_pagination(current_page: usize, total_pages: usize, base_url: &str) -> Markup {
    if total_pages <= 1 {
        return html! {};
    }

    let separator = if base_url.contains('?') { "&" } else { "?" };

    html! {
        nav class="flex items-center justify-center gap-2 mt-6" {
            // Previous
            @if current_page > 1 {
                a href=(format!("{base_url}{separator}page={}", current_page - 1))
                    class="m-btn text-sm" { "PREV" }
            } @else {
                span class="m-btn text-sm opacity-30 pointer-events-none" { "PREV" }
            }

            // Page numbers
            @for p in 1..=total_pages {
                @if p == current_page {
                    span class="m-btn text-sm" style="background: var(--bg-active);" {
                        (p)
                    }
                } @else if total_pages <= 7
                    || p <= 2
                    || p > total_pages - 2
                    || (p >= current_page.saturating_sub(1) && p <= current_page + 1) {
                    a href=(format!("{base_url}{separator}page={p}"))
                        class="m-btn text-sm" { (p) }
                } @else if p == 3 || p == total_pages - 2 {
                    span class="text-neutral-500 px-1" { "..." }
                }
            }

            // Next
            @if current_page < total_pages {
                a href=(format!("{base_url}{separator}page={}", current_page + 1))
                    class="m-btn text-sm" { "NEXT" }
            } @else {
                span class="m-btn text-sm opacity-30 pointer-events-none" { "NEXT" }
            }
        }
    }
}

/// Render a thin progress bar.
#[must_use]
pub fn m_progress(percent: u8) -> Markup {
    html! {
        div class="w-full bg-neutral-700 rounded-sm overflow-hidden" style="height: 2px;" {
            div class="bg-white h-full transition-all" style=(format!("width: {}%", percent)) {}
        }
        span class="text-xs text-neutral-400 font-mono ml-2" { (percent) "%" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_uses_memoria_styling() {
        let content = html! { p { "Test content" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("bg-neutral-950"));
        assert!(html_str.contains("text-white"));
        assert!(html_str.contains("font-sans"));
    }

    #[test]
    fn test_layout_no_crt_effects() {
        let content = html! { p { "Test content" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(!html_str.contains("crt-scanlines"));
        assert!(!html_str.contains("crt-flicker"));
        assert!(!html_str.contains("crt-vignette"));
    }

    #[test]
    fn test_layout_contains_memoria_fonts() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("Inter"));
        assert!(html_str.contains("JetBrains+Mono"));
        assert!(!html_str.contains("VT323"));
        assert!(!html_str.contains("Orbitron"));
        assert!(!html_str.contains("Rajdhani"));
    }

    #[test]
    fn test_layout_contains_wordmark() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("NEUMANN"));
    }

    #[test]
    fn test_layout_has_memoria_script() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("Memoria"));
    }

    #[test]
    fn test_layout_no_tro_or_audio() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(!html_str.contains("TRO"));
        assert!(!html_str.contains("NeumannAudio"));
    }

    #[test]
    fn test_m_stat_renders() {
        let card = m_stat("Test", "100", "subtitle", "relational");
        assert!(card.0.contains("m-stat"));
        assert!(card.0.contains("m-stat-label"));
        assert!(card.0.contains("m-stat-value"));
        assert!(card.0.contains("Test"));
        assert!(card.0.contains("100"));
        assert!(card.0.contains("subtitle"));
        assert!(card.0.contains("data-counter"));
    }

    #[test]
    fn test_m_stat_no_engine_border() {
        // All engines should produce the same m-stat (no colored borders)
        let r = m_stat("Test", "100", "sub", "relational");
        let v = m_stat("Test", "100", "sub", "vector");
        let g = m_stat("Test", "100", "sub", "graph");
        let u = m_stat("Test", "100", "sub", "unknown");
        // All should contain m-stat but not engine-specific classes
        for card in [&r, &v, &g, &u] {
            assert!(card.0.contains("m-stat"));
            assert!(!card.0.contains("border-amber"));
            assert!(!card.0.contains("border-rust"));
            assert!(!card.0.contains("border-phosphor"));
        }
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1.0K");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1_000_000), "1.0M");
    }

    #[test]
    fn test_m_breadcrumb_rendering() {
        let bc = m_breadcrumb(&[("/", "ROOT"), ("/tables", "TABLES")]);
        let html_str = bc.0;
        assert!(html_str.contains("ROOT"));
        assert!(html_str.contains("TABLES"));
        assert!(html_str.contains("m-breadcrumb"));
    }

    #[test]
    fn test_m_progress() {
        let bar = m_progress(50);
        assert!(bar.0.contains("50%"));
        assert!(bar.0.contains("bg-white"));
        assert!(bar.0.contains("bg-neutral-700"));
    }

    #[test]
    fn test_m_empty() {
        let state = m_empty("No Data", "Nothing to display");
        assert!(state.0.contains("No Data"));
        assert!(state.0.contains("Nothing to display"));
        assert!(state.0.contains("m-lamp"));
    }

    #[test]
    fn test_m_header_without_description() {
        let header = m_header("TEST TITLE", None);
        assert!(header.0.contains("TEST TITLE"));
        assert!(!header.0.contains("text-neutral-400 mt-1"));
    }

    #[test]
    fn test_m_header_with_description() {
        let header = m_header("MAIN", Some("This is a description"));
        assert!(header.0.contains("MAIN"));
        assert!(header.0.contains("This is a description"));
    }

    #[test]
    fn test_m_card() {
        let content = html! { p { "Inner content" } };
        let panel = m_card("PANEL TITLE", content);
        assert!(panel.0.contains("m-card"));
        assert!(panel.0.contains("PANEL TITLE"));
        assert!(panel.0.contains("Inner content"));
    }

    #[test]
    fn test_m_section_empty() {
        let items: Vec<(String, String)> = vec![];
        let section = m_section("EMPTY SECTION", "relational", &items);
        assert!(section.0.contains("No entries"));
        assert!(section.0.contains("m-card"));
    }

    #[test]
    fn test_m_section_with_items() {
        let items = vec![
            ("Item1".to_string(), "10".to_string()),
            ("Item2".to_string(), "20".to_string()),
        ];
        let section = m_section("ITEMS SECTION", "vector", &items);
        assert!(section.0.contains("Item1"));
        assert!(section.0.contains("10"));
        assert!(section.0.contains("Item2"));
        assert!(section.0.contains("20"));
        assert!(section.0.contains("m-card"));
    }

    #[test]
    fn test_m_section_graph() {
        let items = vec![("Nodes".to_string(), "100".to_string())];
        let section = m_section("GRAPH DATA", "graph", &items);
        assert!(section.0.contains("m-card"));
    }

    #[test]
    fn test_m_section_unknown_engine() {
        let items = vec![("Data".to_string(), "42".to_string())];
        let section = m_section("UNKNOWN", "unknown", &items);
        assert!(!section.0.contains("border-amber"));
        assert!(!section.0.contains("border-rust"));
    }

    #[test]
    fn test_m_btn_with_href() {
        let btn = m_btn("CLICK ME", Some("/test"));
        assert!(btn.0.contains("href=\"/test\""));
        assert!(btn.0.contains("CLICK ME"));
        assert!(btn.0.contains("<a "));
        assert!(btn.0.contains("m-btn"));
    }

    #[test]
    fn test_m_btn_without_href() {
        let btn = m_btn("SUBMIT", None);
        assert!(btn.0.contains("<button"));
        assert!(btn.0.contains("SUBMIT"));
        assert!(btn.0.contains("type=\"submit\""));
        assert!(btn.0.contains("m-btn"));
    }

    #[test]
    fn test_m_table_header_single_column() {
        let header = m_table_header(&["ID"]);
        assert!(header.0.contains("<th>"));
        assert!(header.0.contains("ID"));
    }

    #[test]
    fn test_m_table_header_multiple_columns() {
        let header = m_table_header(&["ID", "NAME", "VALUE"]);
        assert!(header.0.contains("ID"));
        assert!(header.0.contains("NAME"));
        assert!(header.0.contains("VALUE"));
    }

    #[test]
    fn test_m_expandable_text_short() {
        let text = m_expandable_text("Short text", 50, "text-white");
        assert!(text.0.contains("Short text"));
        assert!(!text.0.contains("more"));
    }

    #[test]
    fn test_m_expandable_text_long() {
        let long_text = "a".repeat(100);
        let text = m_expandable_text(&long_text, 20, "text-white");
        assert!(text.0.contains("more"));
        assert!(text.0.contains("less"));
        assert!(text.0.contains("..."));
    }

    #[test]
    fn test_m_expandable_string_short() {
        let text = m_expandable_string("Hello", 50);
        assert!(text.0.contains("Hello"));
        assert!(!text.0.contains("more"));
    }

    #[test]
    fn test_m_expandable_string_long() {
        let long_text = "a".repeat(100);
        let text = m_expandable_string(&long_text, 20);
        assert!(text.0.contains("more"));
        assert!(text.0.contains("..."));
    }

    #[test]
    fn test_m_expandable_json_short() {
        let json = r#"{"key": "value"}"#;
        let text = m_expandable_json(json, 50);
        assert!(text.0.contains("key"));
        assert!(!text.0.contains("more"));
    }

    #[test]
    fn test_m_expandable_json_long() {
        let long_json = format!("{{\"data\": \"{}\"}}", "x".repeat(100));
        let text = m_expandable_json(&long_json, 20);
        assert!(text.0.contains("more"));
        assert!(text.0.contains("less"));
    }

    #[test]
    fn test_m_expandable_vector_short() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = m_expandable_vector(&vec, 10);
        assert!(result.0.contains("1.000000"));
        assert!(result.0.contains("2.000000"));
        assert!(result.0.contains("3.000000"));
        assert!(!result.0.contains("show all"));
    }

    #[test]
    fn test_m_expandable_vector_long() {
        let vec: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = m_expandable_vector(&vec, 10);
        assert!(result.0.contains("show all 100"));
        assert!(result.0.contains("dimensions"));
        assert!(result.0.contains("copy"));
    }

    #[test]
    fn test_m_payload_preview_few_items() {
        let items = vec![
            ("name".to_string(), "Alice".to_string()),
            ("age".to_string(), "30".to_string()),
        ];
        let result = m_payload_preview(&items, 5);
        assert!(result.0.contains("name"));
        assert!(result.0.contains("Alice"));
    }

    #[test]
    fn test_m_payload_preview_many_items() {
        let items: Vec<_> = (0..10)
            .map(|i| (format!("key{i}"), format!("value{i}")))
            .collect();
        let result = m_payload_preview(&items, 3);
        assert!(result.0.contains("+7 more"));
    }

    #[test]
    fn test_m_loading() {
        let loading = m_loading("Loading...");
        assert!(loading.0.contains("Loading..."));
        assert!(loading.0.contains("animate-blur-reveal"));
    }

    #[test]
    fn test_m_progress_zero() {
        let bar = m_progress(0);
        assert!(bar.0.contains("0%"));
    }

    #[test]
    fn test_m_progress_full() {
        let bar = m_progress(100);
        assert!(bar.0.contains("100%"));
    }

    #[test]
    fn test_m_progress_quarter() {
        let bar = m_progress(25);
        assert!(bar.0.contains("25%"));
    }

    #[test]
    fn test_format_number_edge_cases() {
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1001), "1.0K");
        assert_eq!(format_number(999_999), "1000.0K");
        assert_eq!(format_number(10_000_000), "10.0M");
    }

    #[test]
    fn test_m_breadcrumb_single_item() {
        let bc = m_breadcrumb(&[("", "CURRENT")]);
        assert!(bc.0.contains("CURRENT"));
    }

    #[test]
    fn test_m_breadcrumb_three_items() {
        let bc = m_breadcrumb(&[("/", "HOME"), ("/level1", "LEVEL1"), ("", "CURRENT")]);
        assert!(bc.0.contains("HOME"));
        assert!(bc.0.contains("LEVEL1"));
        assert!(bc.0.contains("CURRENT"));
    }

    #[test]
    fn test_nav_item_active() {
        let icon = icon_zap();
        let item = nav_item("/test", "T", "TEST", &icon, true);
        assert!(item.0.contains("active"));
        assert!(item.0.contains("TEST"));
        assert!(item.0.contains("/test"));
        assert!(item.0.contains("svg"));
    }

    #[test]
    fn test_nav_item_inactive() {
        let icon = icon_zap();
        let item = nav_item("/test", "T", "TEST", &icon, false);
        let html = item.0;
        assert!(html.contains("TEST"));
        assert!(html.contains("svg"));
    }

    #[test]
    fn test_mobile_nav_item_active() {
        let icon = icon_zap();
        let item = mobile_nav_item("/test", "T", &icon, true);
        assert!(item.0.contains("text-white"));
        assert!(item.0.contains("svg"));
    }

    #[test]
    fn test_mobile_nav_item_inactive() {
        let icon = icon_zap();
        let item = mobile_nav_item("/test", "T", &icon, false);
        assert!(item.0.contains("text-neutral-500"));
        assert!(item.0.contains("svg"));
    }

    #[test]
    fn test_layout_different_nav_items() {
        let content = html! { p { "Test" } };

        let result = layout("Test", NavItem::Graph, content.clone());
        assert!(result.0.contains("GRAPH"));

        let result = layout("Test", NavItem::Vector, content.clone());
        assert!(result.0.contains("VECTOR"));

        let result = layout("Test", NavItem::Relational, content);
        assert!(result.0.contains("RELATIONAL"));
    }

    #[test]
    fn test_layout_has_main_content_id() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        assert!(result.0.contains("id=\"main-content\""));
    }

    #[test]
    fn test_keyboard_nav_no_tro_toggle() {
        assert!(!KEYBOARD_NAV_SCRIPT.contains("TRO"));
        assert!(!KEYBOARD_NAV_SCRIPT.contains("NeumannAudio"));
    }

    #[test]
    fn test_keyboard_nav_neutral_row_selected() {
        assert!(KEYBOARD_NAV_SCRIPT.contains("rgba(255, 255, 255, 0.08)"));
        assert!(KEYBOARD_NAV_SCRIPT.contains("--border-emphasis"));
    }

    #[test]
    fn test_m_card_interactive() {
        let content = html! { p { "Interactive content" } };
        let card = m_card_interactive("INTERACTIVE", content);
        assert!(card.0.contains("m-card-interactive"));
        assert!(card.0.contains("INTERACTIVE"));
        assert!(card.0.contains("Interactive content"));
    }

    #[test]
    fn test_m_badge() {
        let badge = m_badge("v1.0");
        assert!(badge.0.contains("m-badge"));
        assert!(badge.0.contains("v1.0"));
    }

    #[test]
    fn test_m_tabs_active() {
        let tabs = m_tabs(&[("Tab1", "/a", true), ("Tab2", "/b", false)]);
        assert!(tabs.0.contains("m-tabs"));
        assert!(tabs.0.contains("m-tab active"));
        assert!(tabs.0.contains("Tab1"));
        assert!(tabs.0.contains("Tab2"));
        assert!(tabs.0.contains("href=\"/a\""));
    }

    #[test]
    fn test_m_tabs_none_active() {
        let tabs = m_tabs(&[("A", "/a", false), ("B", "/b", false)]);
        assert!(!tabs.0.contains("active"));
    }

    #[test]
    fn test_keyboard_help_has_vault_cache_shortcuts() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("g+s"));
        assert!(html_str.contains("g+c"));
        assert!(html_str.contains("Vault"));
        assert!(html_str.contains("Cache"));
    }

    #[test]
    fn test_m_pagination_single_page() {
        let result = m_pagination(1, 1, "/test");
        let html_str = result.into_string();
        // Single page renders nothing
        assert!(html_str.is_empty());
    }

    #[test]
    fn test_m_pagination_first_page() {
        let result = m_pagination(1, 5, "/test");
        let html_str = result.into_string();
        assert!(html_str.contains("NEXT"));
        assert!(html_str.contains("PREV"));
        // PREV should be disabled
        assert!(html_str.contains("pointer-events-none"));
    }

    #[test]
    fn test_m_pagination_last_page() {
        let result = m_pagination(5, 5, "/test");
        let html_str = result.into_string();
        assert!(html_str.contains("PREV"));
        assert!(html_str.contains("NEXT"));
    }

    #[test]
    fn test_m_pagination_middle_page() {
        let result = m_pagination(3, 5, "/test");
        let html_str = result.into_string();
        assert!(html_str.contains("page=2"));
        assert!(html_str.contains("page=4"));
    }

    #[test]
    fn test_m_pagination_with_query_params() {
        let result = m_pagination(1, 3, "/test?filter=abc");
        let html_str = result.into_string();
        assert!(html_str.contains("&amp;page=2"));
    }

    #[test]
    fn test_sidebar_has_vault_and_cache() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("VAULT"));
        assert!(html_str.contains("CACHE"));
        assert!(html_str.contains("STORAGE"));
    }

    #[test]
    fn test_mobile_nav_has_vault_and_cache() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("VAULT"));
        assert!(html_str.contains("CACHE"));
    }

    #[test]
    fn test_nav_item_vault_active() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Vault, content);
        let html_str = result.0;
        assert!(html_str.contains("VAULT"));
    }

    #[test]
    fn test_nav_item_cache_active() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Cache, content);
        let html_str = result.0;
        assert!(html_str.contains("CACHE"));
    }
}
