// SPDX-License-Identifier: MIT OR Apache-2.0
//! Base layout template for the dystopian terminal admin UI.
//!
//! Features ASCII art logo, CRT effects, phosphor-styled navigation,
//! and a status bar with system metrics.

use maud::{html, Markup, PreEscaped, DOCTYPE};

use crate::web::assets::{ADMIN_CSS, AUDIO_SCRIPT, TAILWIND_CONFIG, TRO_CSS, TRO_SCRIPT};
use crate::web::NavItem;

/// ASCII art NEUMANN logo for the terminal header (Fallout block style).
const ASCII_LOGO: &str = "
███╗   ██╗███████╗██╗   ██╗███╗   ███╗ █████╗ ███╗   ██╗███╗   ██╗
████╗  ██║██╔════╝██║   ██║████╗ ████║██╔══██╗████╗  ██║████╗  ██║
██╔██╗ ██║█████╗  ██║   ██║██╔████╔██║███████║██╔██╗ ██║██╔██╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══██║██║╚██╗██║██║╚██╗██║
██║ ╚████║███████╗╚██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║██║ ╚████║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝";

/// Render the base HTML layout with dystopian terminal styling.
#[must_use]
pub fn layout(title: &str, active: NavItem, content: Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" class="dark" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (title) " | NEUMANN TERMINAL" }

                // Tailwind CSS via CDN
                script src="https://cdn.tailwindcss.com" {}
                script { (PreEscaped(TAILWIND_CONFIG)) }

                // HTMX for interactivity
                script src="https://unpkg.com/htmx.org@1.9.12" defer {}

                // Terminal Fonts from Google Fonts
                link rel="preconnect" href="https://fonts.googleapis.com";
                link rel="preconnect" href="https://fonts.gstatic.com" crossorigin;
                link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=VT323&family=Orbitron:wght@400;500;700&family=Rajdhani:wght@400;500;600&display=swap";

                // Custom terminal styles
                style { (PreEscaped(ADMIN_CSS)) }

                // TRO Living Border styles
                style { (PreEscaped(TRO_CSS)) }
            }
            body class="crt-scanlines crt-flicker" {
                div class="min-h-screen flex flex-col" {
                    // Sidebar (desktop)
                    (sidebar(active))

                    // Main content area
                    div class="lg:ml-64 flex-1 flex flex-col" {
                        // Status bar
                        (status_bar())

                        // Main content
                        main class="flex-1 p-4 lg:p-6 crt-vignette" {
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

                // TRO Living Border script
                script { (PreEscaped(TRO_SCRIPT)) }

                // Audio feedback script
                script { (PreEscaped(AUDIO_SCRIPT)) }
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
        const table = document.querySelector('.data-table tbody, .table-rust tbody');
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

        // Play navigation sound if available
        if (window.NeumannAudio) {
            window.NeumannAudio.playClick();
        }
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
            '#terminal-input, input.terminal-input-field'
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
                nodeModal.classList.add('hidden');
                nodeModal.classList.remove('flex');
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
                case 'c': window.location.href = '/achievements'; break;
                case 'd': window.location.href = '/'; break;
                case 'r': window.location.href = '/relational'; break;
                case 'v': window.location.href = '/vector'; break;
                case 'h': window.location.href = '/graph'; break;
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

                    case '?':
                        // Show keyboard help
                        const help = document.getElementById('keyboard-help');
                        if (help) {
                            help.classList.toggle('hidden');
                            e.preventDefault();
                        }
                        break;

                    case 't':
                        // Toggle TRO living border
                        if (window.TRO) {
                            window.TRO.config.enabled = !window.TRO.config.enabled;
                            if (window.TRO.config.enabled) {
                                window.TRO.init();
                            }
                            e.preventDefault();
                        }
                        break;

                    case 's':
                        // Toggle sound
                        if (window.NeumannAudio) {
                            const enabled = !window.NeumannAudio.isEnabled();
                            window.NeumannAudio.setEnabled(enabled);
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
            background-color: rgba(0, 238, 0, 0.15) !important;
            outline: 1px solid var(--phosphor-green-dim) !important;
        }
        .row-selected td {
            color: var(--phosphor-green) !important;
        }
        #g-prefix-indicator {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            background: var(--soot-gray);
            border: 1px solid var(--phosphor-green-dim);
            padding: 0.5rem 1rem;
            font-family: 'VT323', monospace;
            color: var(--phosphor-green);
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

/// Render the terminal status bar with system info.
fn status_bar() -> Markup {
    html! {
        div class="status-bar border-b" {
            span class="status-bar-item" {
                span class="text-phosphor font-display text-xs tracking-wider" { "NEUMANN" }
                span class="text-phosphor-dark" { "v0.1" }
            }
            span class="status-bar-divider" { "|" }
            span class="status-bar-item" {
                span class="text-phosphor-dim" { "SYS:" }
                span class="text-phosphor" { "ONLINE" }
            }
            span class="status-bar-divider" { "|" }
            span class="status-bar-item" {
                span class="text-phosphor-dim cursor-pointer hover:text-phosphor" onclick="document.getElementById('keyboard-help').classList.toggle('hidden')" { "[?] HELP" }
            }
            span class="status-bar-divider" { "|" }
            span class="status-bar-item ml-auto" {
                span class="status-indicator status-indicator-connected" {}
                span class="text-phosphor-dim" { "CONNECTED" }
            }
        }
    }
}

/// Keyboard help overlay.
fn keyboard_help() -> Markup {
    html! {
        div id="keyboard-help" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-black/90" onclick="this.classList.add('hidden')" {
            div class="terminal-panel w-[32rem] max-w-[90vw]" onclick="event.stopPropagation()" {
                div class="panel-header" { "KEYBOARD SHORTCUTS" }
                div class="panel-content" {
                    div class="space-y-4 font-terminal text-sm" {
                        // Navigation
                        div {
                            div class="text-amber-glow mb-2" { "[ NAVIGATION ]" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="kbd-hint mr-2" { "D" } span class="text-phosphor-dim" { "Dashboard" } }
                                div { span class="kbd-hint mr-2" { "V" } span class="text-phosphor-dim" { "Vector" } }
                                div { span class="kbd-hint mr-2" { "R" } span class="text-phosphor-dim" { "Relational" } }
                            }
                        }
                        // G-prefix navigation
                        div {
                            div class="text-amber-glow mb-2" { "[ G-PREFIX ]" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="kbd-hint mr-2" { "g+g" } span class="text-phosphor-dim" { "Top of list" } }
                                div { span class="kbd-hint mr-2" { "g+h" } span class="text-phosphor-dim" { "Graph engine" } }
                                div { span class="kbd-hint mr-2" { "g+a" } span class="text-phosphor-dim" { "Algorithms" } }
                                div { span class="kbd-hint mr-2" { "g+m" } span class="text-phosphor-dim" { "Metrics" } }
                                div { span class="kbd-hint mr-2" { "g+c" } span class="text-phosphor-dim" { "Achievements" } }
                            }
                        }
                        // Table navigation
                        div {
                            div class="text-amber-glow mb-2" { "[ TABLE NAVIGATION ]" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="kbd-hint mr-2" { "j" } span class="text-phosphor-dim" { "Next row" } }
                                div { span class="kbd-hint mr-2" { "k" } span class="text-phosphor-dim" { "Previous row" } }
                                div { span class="kbd-hint mr-2" { "G" } span class="text-phosphor-dim" { "Last row" } }
                                div { span class="kbd-hint mr-2" { "Enter" } span class="text-phosphor-dim" { "Expand row" } }
                                div { span class="kbd-hint mr-2" { "[" } span class="text-phosphor-dim" { "Previous page" } }
                                div { span class="kbd-hint mr-2" { "]" } span class="text-phosphor-dim" { "Next page" } }
                            }
                        }
                        // General
                        div {
                            div class="text-amber-glow mb-2" { "[ GENERAL ]" }
                            div class="grid grid-cols-2 gap-2" {
                                div { span class="kbd-hint mr-2" { "/" } span class="text-phosphor-dim" { "Focus search" } }
                                div { span class="kbd-hint mr-2" { "?" } span class="text-phosphor-dim" { "This help" } }
                                div { span class="kbd-hint mr-2" { "Esc" } span class="text-phosphor-dim" { "Close/unfocus" } }
                                div { span class="kbd-hint mr-2" { "T" } span class="text-phosphor-dim" { "Toggle TRO border" } }
                                div { span class="kbd-hint mr-2" { "S" } span class="text-phosphor-dim" { "Toggle sound" } }
                            }
                        }
                    }
                }
                div class="panel-footer text-center text-phosphor-dark" {
                    "Press ? or Esc to close"
                }
            }
        }
    }
}

/// Render a page header with title and optional description.
#[must_use]
pub fn page_header(title: &str, description: Option<&str>) -> Markup {
    html! {
        div class="mb-6" {
            h1 class="text-2xl font-display text-phosphor phosphor-glow tracking-wider uppercase" {
                (title)
            }
            @if let Some(desc) = description {
                p class="text-phosphor-dim font-terminal mt-1" { (desc) }
            }
        }
    }
}

/// Render the desktop sidebar with ASCII art and navigation.
fn sidebar(active: NavItem) -> Markup {
    html! {
        aside class="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col bg-terminal-soot border-r border-phosphor-dim" {
            // ASCII Logo - Fallout block style
            div class="px-1 pt-3 pb-2 border-b border-phosphor-dark overflow-hidden" {
                pre class="text-phosphor text-[4px] leading-[1.15] font-mono phosphor-glow-subtle whitespace-pre" style="transform: scale(0.95); transform-origin: center top;" {
                    (ASCII_LOGO)
                }
                div class="text-center text-amber-glow text-[9px] font-terminal mt-2 tracking-[0.25em]" {
                    "TENSOR DATABASE"
                }
            }

            // Navigation
            nav class="flex-1 px-2 py-4 nav-terminal" {
                div class="mb-2 px-2" {
                    span class="text-xs text-phosphor-dark font-terminal tracking-widest" {
                        "[ NAVIGATION ]"
                    }
                }

                (nav_item("/", "D", "DASHBOARD", active == NavItem::Dashboard))

                div class="mt-4 mb-2 px-2" {
                    span class="text-xs text-phosphor-dark font-terminal tracking-widest" {
                        "[ ENGINES ]"
                    }
                }

                (nav_item("/graph", "G", "GRAPH", active == NavItem::Graph))
                (nav_item("/vector", "V", "VECTOR", active == NavItem::Vector))
                (nav_item("/relational", "R", "RELATIONAL", active == NavItem::Relational))
            }

            // Keyboard hints
            div class="px-4 py-3 border-t border-phosphor-dark" {
                div class="text-xs text-phosphor-dark font-terminal" {
                    "KEYBOARD SHORTCUTS"
                }
                div class="mt-2 grid grid-cols-2 gap-1 text-xs font-terminal" {
                    span { span class="kbd-hint" { "D" } span class="text-phosphor-dim" { "Dash" } }
                    span { span class="kbd-hint" { "G" } span class="text-phosphor-dim" { "Graph" } }
                    span { span class="kbd-hint" { "V" } span class="text-phosphor-dim" { "Vector" } }
                    span { span class="kbd-hint" { "R" } span class="text-phosphor-dim" { "Rel" } }
                }
            }

            // Status footer
            div class="px-4 py-3 border-t border-phosphor-dark" {
                div class="flex items-center gap-2" {
                    span class="status-indicator status-indicator-connected" {}
                    span class="text-sm text-phosphor-dim font-terminal" { "SYSTEM READY" }
                }
            }
        }
    }
}

/// Render the mobile bottom navigation.
fn mobile_nav(active: NavItem) -> Markup {
    html! {
        nav class="lg:hidden fixed bottom-0 left-0 right-0 bg-terminal-soot border-t border-phosphor-dim px-2 py-1" {
            div class="flex justify-around" {
                (mobile_nav_item("/", "D", active == NavItem::Dashboard))
                (mobile_nav_item("/graph", "G", active == NavItem::Graph))
                (mobile_nav_item("/vector", "V", active == NavItem::Vector))
                (mobile_nav_item("/relational", "R", active == NavItem::Relational))
            }
        }
    }
}

/// Render a navigation item with keyboard hint.
fn nav_item(href: &str, key: &str, label: &str, is_active: bool) -> Markup {
    let active_class = if is_active { "active" } else { "" };

    html! {
        a href=(href) class=(format!("flex items-center gap-2 px-3 py-2 my-1 {active_class}")) {
            span class="kbd-hint" { (key) }
            span { (label) }
        }
    }
}

/// Render a mobile navigation item.
fn mobile_nav_item(href: &str, key: &str, is_active: bool) -> Markup {
    let active_class = if is_active {
        "text-phosphor phosphor-glow-subtle"
    } else {
        "text-phosphor-dim"
    };

    html! {
        a href=(href) class=(format!("flex flex-col items-center gap-1 px-4 py-2 {active_class}")) {
            span class="font-terminal text-lg" { "[" (key) "]" }
        }
    }
}

/// Render a terminal-styled stat card.
#[must_use]
pub fn stat_card(label: &str, value: &str, subtitle: &str, engine: &str) -> Markup {
    let border_class = match engine {
        "relational" => "stat-card-relational",
        "vector" => "stat-card-vector",
        "graph" => "stat-card-graph",
        _ => "",
    };

    html! {
        div class=(format!("stat-card {border_class}")) {
            div class="stat-card-label" { (label) }
            div class="stat-card-value" { (value) }
            div class="stat-card-subtitle" { (subtitle) }
        }
    }
}

/// Render a terminal panel with ASCII box decorations.
#[must_use]
pub fn terminal_panel(title: &str, content: Markup) -> Markup {
    html! {
        div class="terminal-panel" {
            div class="panel-header" { (title) }
            div class="panel-content" {
                (content)
            }
        }
    }
}

/// Render a terminal panel with rust accent.
#[must_use]
pub fn terminal_panel_rust(title: &str, content: Markup) -> Markup {
    html! {
        div class="terminal-panel terminal-panel-rust" {
            div class="panel-header" { (title) }
            div class="panel-content" {
                (content)
            }
        }
    }
}

/// Render an engine section card for the dashboard.
#[must_use]
pub fn engine_section(title: &str, engine: &str, items: &[(String, String)]) -> Markup {
    let border_class = match engine {
        "relational" => "border-l-2 border-amber-glow",
        "vector" => "border-l-2 border-rust-blood",
        "graph" => "border-l-2 border-phosphor",
        _ => "",
    };

    html! {
        div class=(format!("terminal-panel {border_class}")) {
            div class="panel-header" { (title) }
            div class="panel-content" {
                @if items.is_empty() {
                    div class="text-phosphor-dark font-terminal italic" {
                        "< NO DATA >"
                    }
                } @else {
                    ul class="space-y-2" {
                        @for (name, count) in items {
                            li class="flex items-center justify-between font-terminal" {
                                span class="text-phosphor" { (name) }
                                span class="text-amber font-data" { (count) }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Render an empty state message.
#[must_use]
pub fn empty_state(title: &str, description: &str) -> Markup {
    html! {
        div class="text-center py-12 terminal-panel" {
            div class="text-6xl text-phosphor-dark mb-4 font-terminal" { "[ ]" }
            h3 class="text-lg font-display text-phosphor phosphor-glow-subtle" { (title) }
            p class="text-sm text-phosphor-dim font-terminal mt-1" { (description) }
        }
    }
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
pub fn breadcrumb(items: &[(&str, &str)]) -> Markup {
    html! {
        nav class="breadcrumb-terminal mb-4" {
            span class="text-phosphor-dark" { "> " }
            @for (i, (href, label)) in items.iter().enumerate() {
                @if i > 0 {
                    span class="separator" { ">" }
                }
                @if i == items.len() - 1 {
                    span class="text-phosphor" { (label) }
                } @else {
                    a href=(href) { (label) }
                }
            }
        }
    }
}

/// Render a terminal-styled button.
#[must_use]
pub fn btn_terminal(label: &str, href: Option<&str>) -> Markup {
    if let Some(h) = href {
        html! {
            a href=(h) class="btn-terminal inline-block" { (label) }
        }
    } else {
        html! {
            button type="submit" class="btn-terminal" { (label) }
        }
    }
}

/// Render a terminal-styled table.
#[must_use]
pub fn table_header(columns: &[&str]) -> Markup {
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

/// Render expandable text content with terminal styling.
#[must_use]
pub fn expandable_text(content: &str, max_chars: usize, color_class: &str) -> Markup {
    if content.len() <= max_chars {
        return html! {
            span class=(color_class) { (content) }
        };
    }

    let preview = &content[..max_chars.min(content.len())];

    html! {
        span class="expandable-terminal" {
            span class="preview" {
                span class=(color_class) { (preview) "..." }
                span class="expand-btn-terminal ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.nextElementSibling.style.display='inline'" {
                    "[MORE]"
                }
            }
            span class="full" style="display:none" {
                span class=(format!("{color_class} whitespace-pre-wrap")) { (content) }
                span class="expand-btn-terminal ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.previousElementSibling.style.display='inline'" {
                    "[LESS]"
                }
            }
        }
    }
}

/// Render expandable text with quoted string formatting.
#[must_use]
pub fn expandable_string(content: &str, max_chars: usize) -> Markup {
    if content.len() <= max_chars {
        return html! {
            span class="text-amber" { "\"" (content) "\"" }
        };
    }

    let preview = &content[..max_chars.min(content.len())];

    html! {
        span class="expandable-terminal" {
            span class="preview" {
                span class="text-amber" { "\"" (preview) "...\"" }
                span class="expand-btn-terminal ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.nextElementSibling.style.display='inline'" {
                    "[MORE]"
                }
            }
            span class="full" style="display:none" {
                span class="text-amber whitespace-pre-wrap" { "\"" (content) "\"" }
                span class="expand-btn-terminal ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.previousElementSibling.style.display='inline'" {
                    "[LESS]"
                }
            }
        }
    }
}

/// Render expandable JSON content with terminal styling.
#[must_use]
pub fn expandable_json(content: &str, max_chars: usize) -> Markup {
    if content.len() <= max_chars {
        return html! {
            span class="text-phosphor font-terminal text-sm" { (content) }
        };
    }

    let preview = &content[..max_chars.min(content.len())];

    html! {
        span class="expandable-terminal" {
            span class="preview" {
                span class="text-phosphor font-terminal text-sm" { (preview) "..." }
                span class="expand-btn-terminal ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.nextElementSibling.style.display='inline'" {
                    "[MORE]"
                }
            }
            span class="full" style="display:none" {
                span class="text-phosphor font-terminal text-sm whitespace-pre-wrap" { (content) }
                span class="expand-btn-terminal ml-1"
                    onclick="this.parentElement.style.display='none';this.parentElement.previousElementSibling.style.display='inline'" {
                    "[LESS]"
                }
            }
        }
    }
}

/// Render expandable vector content with dimension info.
#[must_use]
pub fn expandable_vector(vec: &[f32], max_preview: usize) -> Markup {
    let total = vec.len();

    if total <= max_preview {
        let formatted = vec
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ");
        return html! {
            span class="font-terminal text-xs text-phosphor-dim" { "[" (formatted) "]" }
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
        details class="expandable-details-terminal" {
            summary class="cursor-pointer list-none" {
                span class="font-terminal text-xs text-phosphor-dim" { (preview) }
                span class="expand-btn-terminal ml-2" {
                    "[SHOW ALL " (total) "]"
                }
            }
            div class="mt-2 p-3 bg-terminal-deep border border-phosphor-dark max-h-96 overflow-auto" {
                div class="flex items-center justify-between mb-2 pb-2 border-b border-phosphor-dark" {
                    span class="text-xs text-phosphor-dark font-terminal" { (total) " DIMENSIONS" }
                    span
                        class="expand-btn-terminal cursor-pointer"
                        onclick="navigator.clipboard.writeText(this.closest('details').querySelector('pre').textContent)"
                    {
                        "[COPY]"
                    }
                }
                pre class="font-terminal text-xs text-phosphor-dim whitespace-pre-wrap" {
                    "[" (full_formatted) "]"
                }
            }
        }
    }
}

/// Render expandable payload preview with terminal styling.
#[must_use]
pub fn expandable_payload_preview(items: &[(String, String)], max_items: usize) -> Markup {
    if items.len() <= max_items {
        let preview = items
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join(", ");
        return html! {
            span class="font-terminal text-xs" { "{ " (preview) " }" }
        };
    }

    let shown: Vec<String> = items
        .iter()
        .take(max_items)
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    let hidden_count = items.len() - max_items;

    html! {
        details class="expandable-details-terminal" {
            summary class="cursor-pointer list-none" {
                span class="font-terminal text-xs" { "{ " (shown.join(", ")) " " }
                span class="expand-btn-terminal" {
                    "[+" (hidden_count) " MORE]"
                }
                span class="font-terminal text-xs" { " }" }
            }
            div class="mt-2 p-3 bg-terminal-deep border border-phosphor-dark" {
                dl class="space-y-1" {
                    @for (key, value) in items {
                        div class="flex gap-2" {
                            dt class="font-terminal text-xs text-phosphor-dim min-w-[100px]" { (key) ":" }
                            dd class="font-terminal text-xs text-phosphor break-all" { (value) }
                        }
                    }
                }
            }
        }
    }
}

/// Render a loading indicator with terminal ASCII style.
#[must_use]
pub fn loading_indicator(text: &str) -> Markup {
    html! {
        div class="loading-terminal text-center py-4" {
            span class="loading-bar" { "[ " }
            span class="loading-text" { (text) }
            span class="loading-bar" { " ]" }
        }
    }
}

/// Render an ASCII progress bar.
#[must_use]
pub fn progress_bar(percent: u8) -> Markup {
    let filled = (percent as usize * 20) / 100;
    let empty = 20 - filled;
    let filled_chars = "#".repeat(filled);
    let empty_chars = "-".repeat(empty);

    html! {
        span class="progress-terminal font-terminal" {
            "[" (filled_chars) (empty_chars) "] " (percent) "%"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_contains_crt_effects() {
        let content = html! { p { "Test content" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("crt-scanlines"));
        assert!(html_str.contains("crt-flicker"));
        assert!(html_str.contains("crt-vignette"));
    }

    #[test]
    fn test_layout_contains_terminal_fonts() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("VT323"));
        assert!(html_str.contains("Orbitron"));
        assert!(html_str.contains("Rajdhani"));
    }

    #[test]
    fn test_layout_contains_ascii_logo() {
        let content = html! { p { "Test" } };
        let result = layout("Test", NavItem::Dashboard, content);
        let html_str = result.0;
        assert!(html_str.contains("NEUMANN"));
    }

    #[test]
    fn test_stat_card_engine_classes() {
        let card = stat_card("Test", "100", "subtitle", "relational");
        assert!(card.0.contains("stat-card-relational"));

        let card = stat_card("Test", "100", "subtitle", "vector");
        assert!(card.0.contains("stat-card-vector"));

        let card = stat_card("Test", "100", "subtitle", "graph");
        assert!(card.0.contains("stat-card-graph"));
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
    fn test_breadcrumb_rendering() {
        let bc = breadcrumb(&[("/", "ROOT"), ("/tables", "TABLES")]);
        let html_str = bc.0;
        assert!(html_str.contains("ROOT"));
        assert!(html_str.contains("TABLES"));
        assert!(html_str.contains("&gt;")); // HTML encoded >
    }

    #[test]
    fn test_progress_bar() {
        let bar = progress_bar(50);
        assert!(bar.0.contains("##########"));
        assert!(bar.0.contains("50%"));
    }

    #[test]
    fn test_empty_state() {
        let state = empty_state("No Data", "Nothing to display");
        assert!(state.0.contains("No Data"));
        assert!(state.0.contains("Nothing to display"));
        assert!(state.0.contains("[ ]"));
    }
}
