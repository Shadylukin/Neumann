// SPDX-License-Identifier: MIT OR Apache-2.0
//! Embedded CSS and JavaScript assets for the dystopian terminal admin UI.
//!
//! Inspired by Fallout 1/2 terminals, Rupture Farms industrial aesthetic,
//! and classic CRT phosphor displays. Includes TRO (Tensor Rust Organism)
//! living border simulation with Physarum-based particle effects.

/// Custom CSS for the dystopian terminal admin interface.
///
/// Features:
/// - Phosphor green primary with amber accents
/// - CRT scan lines, flicker, and vignette effects
/// - Industrial rust-laden accent colors
/// - Terminal-style typography
pub const ADMIN_CSS: &str = r"
/* ============================================
   DYSTOPIAN TERMINAL DESIGN SYSTEM
   ============================================ */

/* Color Palette Variables */
:root {
    /* Primary Terminal Colors (Fallout Pip-Boy inspired) */
    --phosphor-green: #00ee00;
    --phosphor-green-dim: #008e00;
    --phosphor-green-dark: #005f00;
    --amber-glow: #ffb641;
    --amber-dim: #ffb000;

    /* Rupture Farms Industrial */
    --blood-rust: #942222;
    --dark-rust: #4a2125;
    --corroded-brown: #622f22;
    --deep-black: #0c0c0c;
    --soot-gray: #1a1a1a;
    --industrial-gray: #383838;

    /* Derived Colors */
    --text-primary: var(--phosphor-green);
    --text-secondary: var(--phosphor-green-dim);
    --text-tertiary: var(--phosphor-green-dark);
    --border-primary: var(--phosphor-green-dim);
    --border-secondary: var(--dark-rust);
    --bg-primary: var(--deep-black);
    --bg-secondary: var(--soot-gray);
    --bg-tertiary: var(--industrial-gray);

    /* Typography Scale */
    --text-xs: 0.75rem;
    --text-sm: 0.875rem;
    --text-base: 1rem;
    --text-lg: 1.25rem;
    --text-xl: 1.5rem;
    --text-2xl: 2rem;

    /* Font Families */
    --font-terminal: 'VT323', 'Courier New', monospace;
    --font-display: 'Orbitron', 'VT323', monospace;
    --font-data: 'Rajdhani', 'VT323', monospace;
}

/* ============================================
   CRT EFFECTS
   ============================================ */

/* Scan Lines Overlay */
.crt-scanlines::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 0, 0, 0.15),
        rgba(0, 0, 0, 0.15) 1px,
        transparent 1px,
        transparent 2px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Phosphor Text Glow */
.phosphor-glow {
    text-shadow:
        0 0 5px currentColor,
        0 0 10px currentColor,
        0 0 20px currentColor;
}

.phosphor-glow-subtle {
    text-shadow:
        0 0 3px currentColor,
        0 0 6px currentColor;
}

/* CRT Screen Flicker (subtle) */
@keyframes flicker {
    0%, 100% { opacity: 1; }
    92% { opacity: 0.97; }
    94% { opacity: 1; }
    97% { opacity: 0.98; }
}

.crt-flicker {
    animation: flicker 0.15s infinite;
}

/* Vignette Effect */
.crt-vignette {
    box-shadow: inset 0 0 100px rgba(0, 0, 0, 0.5);
}

/* CRT Curvature (subtle) */
.crt-curve {
    border-radius: 20px;
    box-shadow:
        inset 0 0 100px rgba(0, 0, 0, 0.5),
        0 0 20px rgba(0, 238, 0, 0.1);
}

/* ============================================
   BASE STYLES
   ============================================ */

body {
    background-color: var(--deep-black);
    color: var(--phosphor-green);
    font-family: var(--font-terminal);
    font-size: var(--text-base);
    line-height: 1.5;
}

/* Terminal Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--soot-gray);
    border: 1px solid var(--phosphor-green-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--phosphor-green-dim);
    border-radius: 0;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--phosphor-green);
    box-shadow: 0 0 5px var(--phosphor-green);
}

/* Selection */
::selection {
    background: var(--phosphor-green);
    color: var(--deep-black);
}

/* ============================================
   TERMINAL BUTTON COMPONENTS
   ============================================ */

.btn-terminal {
    background: transparent;
    border: 1px solid var(--phosphor-green);
    color: var(--phosphor-green);
    font-family: var(--font-terminal);
    padding: 0.5rem 1rem;
    text-shadow: 0 0 5px var(--phosphor-green);
    transition: all 0.1s;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.btn-terminal:hover {
    background: var(--phosphor-green);
    color: var(--deep-black);
    box-shadow: 0 0 10px var(--phosphor-green);
}

.btn-terminal:active {
    transform: scale(0.98);
}

.btn-terminal-amber {
    border-color: var(--amber-glow);
    color: var(--amber-glow);
    text-shadow: 0 0 5px var(--amber-glow);
}

.btn-terminal-amber:hover {
    background: var(--amber-glow);
    color: var(--deep-black);
    box-shadow: 0 0 10px var(--amber-glow);
}

.btn-terminal-rust {
    border-color: var(--blood-rust);
    color: var(--blood-rust);
    text-shadow: 0 0 5px var(--blood-rust);
}

.btn-terminal-rust:hover {
    background: var(--blood-rust);
    color: var(--deep-black);
    box-shadow: 0 0 10px var(--blood-rust);
}

/* Keyboard shortcut badge */
.kbd-hint {
    display: inline-block;
    padding: 0.125rem 0.375rem;
    border: 1px solid var(--phosphor-green-dim);
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
    color: var(--phosphor-green-dim);
    margin-right: 0.25rem;
}

/* ============================================
   DATA TABLE (RUST-LADEN)
   ============================================ */

.table-rust {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-data);
}

.table-rust th {
    background: var(--corroded-brown);
    color: var(--amber-glow);
    border-bottom: 2px solid var(--blood-rust);
    padding: 0.75rem 1rem;
    text-align: left;
    font-family: var(--font-terminal);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: var(--text-sm);
}

.table-rust td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--dark-rust);
    color: var(--phosphor-green);
}

.table-rust tr:nth-child(even) {
    background: rgba(74, 33, 37, 0.2);
}

.table-rust tr:hover {
    background: rgba(0, 238, 0, 0.1);
}

.table-rust tr:hover td {
    text-shadow: 0 0 3px var(--phosphor-green);
}

/* ============================================
   INPUT FIELDS
   ============================================ */

.input-terminal {
    background: var(--deep-black);
    border: 1px solid var(--phosphor-green-dim);
    color: var(--phosphor-green);
    font-family: var(--font-terminal);
    padding: 0.5rem 0.75rem;
    caret-color: var(--phosphor-green);
    transition: all 0.1s;
}

.input-terminal::placeholder {
    color: var(--phosphor-green-dark);
}

.input-terminal:focus {
    outline: none;
    border-color: var(--phosphor-green);
    box-shadow: 0 0 5px var(--phosphor-green);
}

.input-terminal:disabled {
    border-color: var(--phosphor-green-dark);
    color: var(--phosphor-green-dark);
    cursor: not-allowed;
}

/* ============================================
   TERMINAL PANELS (ASCII BOX DRAWING)
   ============================================ */

.terminal-panel {
    background: var(--soot-gray);
    border: 1px solid var(--phosphor-green-dim);
    position: relative;
}

.panel-header {
    background: var(--deep-black);
    border-bottom: 1px solid var(--phosphor-green-dim);
    padding: 0.5rem 1rem;
    font-family: var(--font-terminal);
    color: var(--phosphor-green);
    text-shadow: 0 0 5px var(--phosphor-green);
}

.panel-header::before {
    content: '\2554\2550\2550[ ';
    color: var(--phosphor-green-dim);
}

.panel-header::after {
    content: ' ]\2550\2550\2557';
    color: var(--phosphor-green-dim);
}

.panel-content {
    padding: 1rem;
}

.panel-footer {
    background: var(--deep-black);
    border-top: 1px solid var(--phosphor-green-dim);
    padding: 0.5rem 1rem;
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
    color: var(--phosphor-green-dim);
}

/* Double-line panel variant */
.terminal-panel-double {
    border: 2px double var(--phosphor-green-dim);
}

/* Rust-accented panel */
.terminal-panel-rust {
    border-color: var(--blood-rust);
    background: var(--dark-rust);
}

.terminal-panel-rust .panel-header {
    background: var(--corroded-brown);
    border-bottom-color: var(--blood-rust);
    color: var(--amber-glow);
}

/* ============================================
   STAT CARDS
   ============================================ */

.stat-card {
    background: var(--soot-gray);
    border: 1px solid var(--phosphor-green-dim);
    padding: 1rem;
    transition: all 0.1s;
}

.stat-card:hover {
    border-color: var(--phosphor-green);
    box-shadow: 0 0 10px rgba(0, 238, 0, 0.2);
}

.stat-card-label {
    font-family: var(--font-terminal);
    font-size: var(--text-sm);
    color: var(--phosphor-green-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.stat-card-value {
    font-family: var(--font-data);
    font-size: var(--text-2xl);
    font-weight: 600;
    color: var(--phosphor-green);
    text-shadow: 0 0 10px var(--phosphor-green);
    margin: 0.25rem 0;
}

.stat-card-subtitle {
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
    color: var(--phosphor-green-dark);
}

/* Engine-specific stat cards */
.stat-card-relational { border-left: 3px solid var(--amber-glow); }
.stat-card-vector { border-left: 3px solid var(--blood-rust); }
.stat-card-graph { border-left: 3px solid var(--phosphor-green); }

/* ============================================
   LOADING INDICATOR
   ============================================ */

.loading-terminal {
    font-family: var(--font-terminal);
    color: var(--phosphor-green);
}

.loading-bar {
    display: inline-block;
    color: var(--phosphor-green);
    text-shadow: 0 0 5px var(--phosphor-green);
}

@keyframes loading-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-terminal .loading-text {
    animation: loading-pulse 0.8s ease-in-out infinite;
}

/* Progress bar ASCII style */
.progress-terminal {
    font-family: var(--font-terminal);
    color: var(--phosphor-green);
}

/* ============================================
   GRAPH VISUALIZATION
   ============================================ */

.graph-node {
    fill: var(--soot-gray);
    stroke: var(--phosphor-green);
    stroke-width: 2;
    transition: all 0.1s;
}

.graph-node:hover {
    fill: var(--industrial-gray);
    filter: drop-shadow(0 0 8px var(--phosphor-green));
}

.graph-node-selected {
    fill: var(--phosphor-green-dark);
    stroke: var(--phosphor-green);
    stroke-width: 3;
    filter: drop-shadow(0 0 12px var(--phosphor-green));
}

.graph-edge {
    stroke: var(--phosphor-green-dim);
    stroke-opacity: 0.6;
    stroke-width: 1;
}

.graph-edge:hover {
    stroke: var(--phosphor-green);
    stroke-opacity: 1;
    stroke-width: 2;
}

.graph-label {
    fill: var(--phosphor-green);
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
    text-shadow: 0 0 3px var(--phosphor-green);
}

#graph-container {
    background: var(--deep-black);
    border: 1px solid var(--phosphor-green-dim);
    position: relative;
}

#graph-container::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 40%, rgba(0, 0, 0, 0.3) 100%);
    pointer-events: none;
    z-index: 10;
}

#graph-container canvas {
    position: relative;
    z-index: 1;
}

/* ============================================
   STATUS BAR
   ============================================ */

.status-bar {
    background: var(--deep-black);
    border: 1px solid var(--phosphor-green-dim);
    font-family: var(--font-terminal);
    font-size: var(--text-sm);
    color: var(--phosphor-green-dim);
    padding: 0.25rem 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.status-bar-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-bar-divider {
    color: var(--phosphor-green-dark);
}

.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse-glow 2s ease-in-out infinite;
}

.status-indicator-connected {
    background: var(--phosphor-green);
    box-shadow: 0 0 5px var(--phosphor-green);
}

.status-indicator-warning {
    background: var(--amber-glow);
    box-shadow: 0 0 5px var(--amber-glow);
}

.status-indicator-error {
    background: var(--blood-rust);
    box-shadow: 0 0 5px var(--blood-rust);
}

@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ============================================
   NAVIGATION
   ============================================ */

.nav-terminal {
    font-family: var(--font-terminal);
}

.nav-terminal a {
    color: var(--phosphor-green-dim);
    text-decoration: none;
    padding: 0.5rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: all 0.1s;
    border-left: 2px solid transparent;
}

.nav-terminal a:hover {
    color: var(--phosphor-green);
    text-shadow: 0 0 5px var(--phosphor-green);
    background: rgba(0, 238, 0, 0.05);
}

.nav-terminal a.active {
    color: var(--phosphor-green);
    text-shadow: 0 0 5px var(--phosphor-green);
    border-left-color: var(--phosphor-green);
    background: rgba(0, 238, 0, 0.1);
}

/* Breadcrumb Trail */
.breadcrumb-terminal {
    font-family: var(--font-terminal);
    font-size: var(--text-sm);
    color: var(--phosphor-green-dim);
}

.breadcrumb-terminal a {
    color: var(--phosphor-green-dim);
    text-decoration: none;
}

.breadcrumb-terminal a:hover {
    color: var(--phosphor-green);
    text-shadow: 0 0 3px var(--phosphor-green);
}

.breadcrumb-terminal .separator {
    margin: 0 0.5rem;
    color: var(--phosphor-green-dark);
}

/* ============================================
   HTMX LOADING STATES
   ============================================ */

.htmx-request {
    opacity: 0.7;
    pointer-events: none;
}

.htmx-request::after {
    content: '_';
    animation: blink-cursor 0.5s step-end infinite;
    margin-left: 0.25rem;
}

@keyframes blink-cursor {
    50% { opacity: 0; }
}

/* ============================================
   EXPANDABLE CONTENT
   ============================================ */

.expandable-terminal {
    position: relative;
}

.expand-btn-terminal {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.5rem;
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
    color: var(--phosphor-green-dim);
    background: transparent;
    border: 1px solid var(--phosphor-green-dark);
    cursor: pointer;
    transition: all 0.1s;
}

.expand-btn-terminal:hover {
    color: var(--phosphor-green);
    border-color: var(--phosphor-green-dim);
    text-shadow: 0 0 3px var(--phosphor-green);
}

details.expandable-details-terminal summary {
    cursor: pointer;
    list-style: none;
}

details.expandable-details-terminal summary::-webkit-details-marker {
    display: none;
}

details.expandable-details-terminal[open] .expand-btn-terminal {
    background: rgba(0, 238, 0, 0.1);
    border-color: var(--phosphor-green);
}

/* ============================================
   MODAL DIALOGS
   ============================================ */

.modal-terminal {
    position: fixed;
    inset: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.9);
    opacity: 0;
    visibility: hidden;
    transition: opacity 200ms ease, visibility 200ms ease;
}

.modal-terminal.active {
    opacity: 1;
    visibility: visible;
}

.modal-terminal-body {
    max-width: 90vw;
    max-height: 85vh;
    overflow: auto;
    background: var(--soot-gray);
    border: 2px solid var(--phosphor-green-dim);
    padding: 1.5rem;
    box-shadow:
        0 0 20px rgba(0, 238, 0, 0.2),
        inset 0 0 50px rgba(0, 0, 0, 0.3);
}

/* ============================================
   TOOLTIPS
   ============================================ */

.tooltip-terminal {
    position: relative;
}

.tooltip-terminal::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.375rem 0.5rem;
    background: var(--soot-gray);
    border: 1px solid var(--phosphor-green-dim);
    color: var(--phosphor-green);
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: opacity 150ms ease, visibility 150ms ease;
    z-index: 50;
}

.tooltip-terminal:hover::after {
    opacity: 1;
    visibility: visible;
}

/* ============================================
   UTILITY CLASSES
   ============================================ */

.text-phosphor { color: var(--phosphor-green); }
.text-phosphor-dim { color: var(--phosphor-green-dim); }
.text-phosphor-dark { color: var(--phosphor-green-dark); }
.text-amber { color: var(--amber-glow); }
.text-rust { color: var(--blood-rust); }

.bg-deep { background: var(--deep-black); }
.bg-soot { background: var(--soot-gray); }
.bg-industrial { background: var(--industrial-gray); }
.bg-rust { background: var(--dark-rust); }

.border-phosphor { border-color: var(--phosphor-green); }
.border-phosphor-dim { border-color: var(--phosphor-green-dim); }
.border-rust { border-color: var(--blood-rust); }

.glow-phosphor { text-shadow: 0 0 10px var(--phosphor-green); }
.glow-amber { text-shadow: 0 0 10px var(--amber-glow); }
.glow-rust { text-shadow: 0 0 10px var(--blood-rust); }

.font-terminal { font-family: var(--font-terminal); }
.font-display { font-family: var(--font-display); }
.font-data { font-family: var(--font-data); }

/* ============================================
   REDUCED MOTION
   ============================================ */

@media (prefers-reduced-motion: reduce) {
    .crt-flicker,
    .loading-terminal .loading-text,
    .status-indicator,
    .htmx-request::after {
        animation: none;
    }

    * {
        transition-duration: 0s !important;
    }
}

/* ============================================
   FOCUS STYLES (ACCESSIBILITY)
   ============================================ */

button:focus-visible,
a:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--phosphor-green);
    outline-offset: 2px;
    box-shadow: 0 0 10px var(--phosphor-green);
}

/* ============================================
   ANIMATIONS
   ============================================ */

@keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slide-in {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-fade-in {
    animation: fade-in 200ms ease;
}

.animate-slide-in {
    animation: slide-in 200ms ease;
}

/* ============================================
   PRINT STYLES
   ============================================ */

/* ============================================
   GAME-LIKE INTERACTIVE EFFECTS
   ============================================ */

/* Graph container - properly contained */
#graph-container {
    position: relative;
    z-index: 1;
    overflow: hidden;
    background: var(--deep-black);
}

/* Force graph to respect container bounds */
.terminal-panel:has(#graph-container) {
    overflow: hidden;
    position: relative;
    z-index: 1;
}

.terminal-panel:has(#graph-container) .panel-content {
    overflow: hidden;
    padding: 0;
}

/* Typewriter cursor blink */
@keyframes cursor-blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

.terminal-cursor {
    display: inline-block;
    width: 0.6em;
    height: 1.2em;
    background: var(--phosphor-green);
    animation: cursor-blink 1s step-end infinite;
    vertical-align: text-bottom;
    box-shadow: 0 0 5px var(--phosphor-green);
}

/* Click flash effect (Fallout-style selection) */
@keyframes click-flash {
    0% { background: var(--phosphor-green); color: var(--deep-black); }
    100% { background: transparent; color: var(--phosphor-green); }
}

.click-feedback:active {
    animation: click-flash 0.15s ease-out;
}

/* Hover scan effect (like selecting menu items) */
@keyframes hover-scan {
    0% { background-position: 0% 0%; }
    100% { background-position: 0% 100%; }
}

.nav-item:hover {
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(0, 238, 0, 0.1) 50%,
        transparent 100%
    );
    background-size: 100% 200%;
    animation: hover-scan 0.3s ease-out forwards;
}

/* Boot-up text reveal animation */
@keyframes text-reveal {
    from { clip-path: inset(0 100% 0 0); }
    to { clip-path: inset(0 0 0 0); }
}

.text-reveal {
    animation: text-reveal 0.5s ease-out forwards;
}

/* Glitch effect for errors */
@keyframes glitch {
    0% { transform: translate(0); }
    20% { transform: translate(-2px, 2px); }
    40% { transform: translate(-2px, -2px); }
    60% { transform: translate(2px, 2px); }
    80% { transform: translate(2px, -2px); }
    100% { transform: translate(0); }
}

.glitch-text {
    animation: glitch 0.3s ease-in-out;
}

/* Power-up glow pulse */
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 5px var(--phosphor-green); }
    50% { box-shadow: 0 0 20px var(--phosphor-green), 0 0 30px var(--phosphor-green); }
}

.glow-pulse {
    animation: glow-pulse 2s ease-in-out infinite;
}

/* Terminal typing indicator */
@keyframes typing-dots {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
}

.typing-indicator::after {
    content: '.';
    animation: typing-dots 1.5s steps(1) infinite;
}

/* Selection highlight (like Fallout inventory) */
.selectable-item {
    transition: all 0.1s ease;
    border-left: 3px solid transparent;
}

.selectable-item:hover {
    border-left-color: var(--phosphor-green);
    background: rgba(0, 238, 0, 0.05);
    padding-left: 0.5rem;
}

.selectable-item.selected {
    border-left-color: var(--amber-glow);
    background: rgba(255, 182, 65, 0.1);
}

/* Interactive terminal input */
.terminal-input-line {
    display: block;
    font-family: var(--font-terminal);
}

.terminal-input-line::before {
    display: none;
}

.terminal-input-field {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--phosphor-green);
    font-family: var(--font-terminal);
    font-size: inherit;
    outline: none;
    caret-color: var(--phosphor-green);
}

.terminal-textarea {
    width: 100%;
    min-height: 60px;
    max-height: 200px;
    resize: none;
    line-height: 1.4;
    padding: 0.5rem;
    background: rgba(0, 50, 0, 0.3);
    border: 1px solid var(--phosphor-green-dark);
    border-radius: 4px;
}

.terminal-textarea:focus {
    border-color: var(--phosphor-green);
    box-shadow: 0 0 8px rgba(0, 255, 65, 0.3);
}

.terminal-textarea::placeholder {
    color: var(--phosphor-green-dark);
    opacity: 0.6;
}

/* Live terminal output */
.terminal-output {
    font-family: var(--font-terminal);
    font-size: var(--text-sm);
    line-height: 1.4;
    max-height: 200px;
    overflow-y: auto;
}

.terminal-output-line {
    padding: 0.125rem 0;
    color: var(--phosphor-green-dim);
}

.terminal-output-line.success {
    color: var(--phosphor-green);
}

.terminal-output-line.error {
    color: var(--blood-rust);
}

.terminal-output-line.warning {
    color: var(--amber-glow);
}

/* Radar ping effect for notifications */
@keyframes radar-ping {
    0% {
        transform: scale(0.8);
        opacity: 1;
    }
    100% {
        transform: scale(2);
        opacity: 0;
    }
}

.radar-ping {
    position: relative;
}

.radar-ping::after {
    content: '';
    position: absolute;
    inset: 0;
    border: 1px solid var(--phosphor-green);
    border-radius: 50%;
    animation: radar-ping 1.5s ease-out infinite;
}

/* ============================================
   PRINT STYLES
   ============================================ */

@media print {
    .crt-scanlines::before,
    .crt-vignette,
    .crt-flicker {
        display: none !important;
    }

    body {
        background: white;
        color: black;
    }
}
";

/// Tailwind CSS configuration for dystopian terminal theme.
///
/// Extends default Tailwind with terminal-specific colors and fonts.
pub const TAILWIND_CONFIG: &str = r"
tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            fontFamily: {
                terminal: ['VT323', 'Courier New', 'monospace'],
                display: ['Orbitron', 'VT323', 'monospace'],
                data: ['Rajdhani', 'VT323', 'monospace'],
            },
            colors: {
                phosphor: {
                    DEFAULT: '#00ee00',
                    dim: '#008e00',
                    dark: '#005f00',
                },
                amber: {
                    glow: '#ffb641',
                    dim: '#ffb000',
                },
                rust: {
                    blood: '#942222',
                    dark: '#4a2125',
                    corroded: '#622f22',
                },
                terminal: {
                    deep: '#0c0c0c',
                    soot: '#1a1a1a',
                    industrial: '#383838',
                },
                engine: {
                    relational: '#ffb641',
                    vector: '#942222',
                    graph: '#00ee00',
                }
            }
        }
    }
}
";

/// CSS styles for TRO living border animation.
///
/// Includes:
/// - Border canvas container styling
/// - Particle glow effects
/// - Activity pulse animations
/// - CRT overlay effects for the border
pub const TRO_CSS: &str = r"
/* ============================================
   TRO LIVING BORDER STYLES
   ============================================ */

/* Main TRO container - wraps entire page */
.tro-container {
    position: relative;
    min-height: 100vh;
}

/* Canvas border wrapper */
.tro-border {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9000;
}

/* Each side canvas */
.tro-border-top,
.tro-border-bottom,
.tro-border-left,
.tro-border-right {
    position: absolute;
    background: transparent;
}

.tro-border-top {
    top: 0;
    left: 0;
    right: 0;
    height: 20px;
}

.tro-border-bottom {
    bottom: 0;
    left: 0;
    right: 0;
    height: 20px;
}

.tro-border-left {
    top: 20px;
    left: 0;
    bottom: 20px;
    width: 20px;
}

.tro-border-right {
    top: 20px;
    right: 0;
    bottom: 20px;
    width: 20px;
}

/* Phosphor glow overlay */
.tro-glow-overlay {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 8999;
    box-shadow:
        inset 0 0 30px rgba(0, 238, 0, 0.1),
        inset 0 0 60px rgba(0, 238, 0, 0.05);
}

/* Activity pulse burst effect */
@keyframes tro-pulse-burst {
    0% {
        transform: scale(0.5);
        opacity: 1;
    }
    100% {
        transform: scale(2);
        opacity: 0;
    }
}

.tro-pulse {
    position: absolute;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--phosphor-green) 0%, transparent 70%);
    animation: tro-pulse-burst 0.5s ease-out forwards;
    pointer-events: none;
}

.tro-pulse-put {
    background: radial-gradient(circle, var(--phosphor-green) 0%, transparent 70%);
}

.tro-pulse-get {
    background: radial-gradient(circle, rgba(0, 187, 204, 0.8) 0%, transparent 70%);
}

.tro-pulse-delete {
    background: radial-gradient(circle, var(--blood-rust) 0%, transparent 70%);
}

.tro-pulse-error {
    background: radial-gradient(circle, #ff4444 0%, transparent 70%);
    animation-duration: 0.8s;
}

.tro-pulse-scan {
    background: radial-gradient(circle, var(--amber-glow) 0%, transparent 70%);
}

/* Glitch effect for border */
@keyframes tro-glitch {
    0%, 100% { filter: none; }
    10% { filter: hue-rotate(90deg) saturate(2); }
    20% { filter: hue-rotate(-90deg) brightness(1.2); }
    30% { filter: hue-rotate(180deg); }
    40% { filter: none; }
    50% { filter: invert(0.1) hue-rotate(45deg); }
    60% { filter: none; }
}

.tro-glitch-active {
    animation: tro-glitch 0.3s ease-in-out;
}

/* Scanline overlay for border */
.tro-scanlines {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9001;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 0, 0, 0.03) 2px,
        rgba(0, 0, 0, 0.03) 4px
    );
}

/* Settings panel for TRO */
.tro-settings {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    background: var(--soot-gray);
    border: 1px solid var(--phosphor-green-dim);
    padding: 0.5rem;
    z-index: 9100;
    font-family: var(--font-terminal);
    font-size: var(--text-xs);
}

.tro-settings-toggle {
    cursor: pointer;
    color: var(--phosphor-green-dim);
}

.tro-settings-toggle:hover {
    color: var(--phosphor-green);
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    .tro-border canvas {
        animation: none !important;
    }
    .tro-pulse {
        animation: none;
        opacity: 0;
    }
}
";

/// JavaScript for TRO Physarum simulation with Canvas rendering.
///
/// Features:
/// - Agent-based Physarum slime mold simulation
/// - Pheromone trail visualization with glowing particles
/// - Real-time activity pulses from database operations
/// - Smooth 60fps animation with requestAnimationFrame
pub const TRO_SCRIPT: &str = r"
// TRO Living Border - Physarum Simulation
const TRO = {
    // Configuration
    config: {
        enabled: false,  // Disabled by default for performance - press 'T' to toggle
        fps: 15,
        agentCount: 100,
        palette: 'phosphor_green',
        crtEffects: true,
        trailLength: 10,
        glowIntensity: 0.7
    },

    // Palette definitions
    palettes: {
        phosphor_green: {
            primary: [0, 238, 0],
            dim: [0, 142, 0],
            dark: [0, 95, 0],
            glow: 'rgba(0, 238, 0, 0.6)'
        },
        amber: {
            primary: [255, 182, 65],
            dim: [204, 136, 0],
            dark: [102, 51, 0],
            glow: 'rgba(255, 182, 65, 0.6)'
        },
        rust: {
            primary: [148, 34, 34],
            dim: [98, 31, 34],
            dark: [74, 33, 37],
            glow: 'rgba(148, 34, 34, 0.6)'
        },
        ghost: {
            primary: [0, 187, 204],
            dim: [0, 119, 136],
            dark: [0, 51, 68],
            glow: 'rgba(0, 187, 204, 0.6)'
        },
        glitch: {
            primary: [170, 68, 170],
            dim: [102, 0, 102],
            dark: [68, 0, 68],
            glow: 'rgba(170, 68, 170, 0.6)'
        }
    },

    // Physarum configuration
    physarum: {
        sensorAngle: Math.PI / 4,
        sensorDistance: 9,
        rotationAngle: Math.PI / 6,
        speed: 2,
        depositAmount: 0.5,
        decayRate: 0.95,
        diffusionRate: 0.2
    },

    // State
    state: {
        agents: [],
        pheromone: [],
        pheromoneBuffer: [],
        activityHeat: [],
        borderLength: 0,
        frame: 0,
        paused: false,
        inGlitch: false,
        glitchUntil: 0
    },

    // Canvas references
    canvases: {
        top: null,
        bottom: null,
        left: null,
        right: null
    },
    contexts: {
        top: null,
        bottom: null,
        left: null,
        right: null
    },

    // Initialize TRO system
    init() {
        if (!this.config.enabled) return;

        this.createCanvases();
        this.calculateBorderLength();
        this.initPhysarum();
        this.startRenderLoop();

        // Listen for window resize
        window.addEventListener('resize', () => this.handleResize());

        console.log('[TRO] Physarum border initialized');
    },

    // Create canvas elements
    createCanvases() {
        const container = document.createElement('div');
        container.className = 'tro-border';
        container.id = 'tro-border';

        const sides = ['top', 'bottom', 'left', 'right'];
        sides.forEach(side => {
            const canvas = document.createElement('canvas');
            canvas.className = `tro-border-${side}`;
            canvas.id = `tro-canvas-${side}`;
            container.appendChild(canvas);
            this.canvases[side] = canvas;
            this.contexts[side] = canvas.getContext('2d');
        });

        // Add glow overlay
        const glow = document.createElement('div');
        glow.className = 'tro-glow-overlay';
        container.appendChild(glow);

        // Add scanlines if CRT effects enabled
        if (this.config.crtEffects) {
            const scanlines = document.createElement('div');
            scanlines.className = 'tro-scanlines';
            container.appendChild(scanlines);
        }

        document.body.appendChild(container);
        this.resizeCanvases();
    },

    // Resize canvases to match window
    resizeCanvases() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        const borderWidth = 20;

        this.canvases.top.width = w;
        this.canvases.top.height = borderWidth;
        this.canvases.bottom.width = w;
        this.canvases.bottom.height = borderWidth;
        this.canvases.left.width = borderWidth;
        this.canvases.left.height = h - borderWidth * 2;
        this.canvases.right.width = borderWidth;
        this.canvases.right.height = h - borderWidth * 2;
    },

    // Calculate total border length
    calculateBorderLength() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        this.state.borderLength = 2 * w + 2 * (h - 40);
    },

    // Initialize Physarum agents and pheromone field
    initPhysarum() {
        const len = this.state.borderLength;
        this.state.pheromone = new Float32Array(len);
        this.state.pheromoneBuffer = new Float32Array(len);
        this.state.activityHeat = new Float32Array(len);
        this.state.agents = [];

        for (let i = 0; i < this.config.agentCount; i++) {
            this.state.agents.push({
                position: Math.random() * len,
                direction: Math.random() < 0.5 ? 1 : -1,
                speed: 0.5 + Math.random() * 0.5,
                active: true
            });
        }
    },

    // Update Physarum simulation
    updatePhysarum() {
        if (this.state.paused) return;

        const len = this.state.borderLength;
        if (len === 0) return;

        const wrap = (pos) => ((pos % len) + len) % len;

        // Update agents
        for (const agent of this.state.agents) {
            if (!agent.active) continue;

            const sd = this.physarum.sensorDistance;
            const leftPos = Math.floor(wrap(agent.position + agent.direction * sd * 0.7)) % len;
            const centerPos = Math.floor(wrap(agent.position + agent.direction * sd)) % len;
            const rightPos = Math.floor(wrap(agent.position + agent.direction * sd * 1.3)) % len;

            const left = this.state.pheromone[leftPos] || 0;
            const center = this.state.pheromone[centerPos] || 0;
            const right = this.state.pheromone[rightPos] || 0;

            // Decide direction
            if (center >= left && center >= right) {
                if (Math.random() < 0.1) {
                    agent.direction *= Math.random() < 0.5 ? 1 : -1;
                }
            } else if (left > right) {
                agent.direction = -Math.abs(agent.direction);
            } else if (right > left) {
                agent.direction = Math.abs(agent.direction);
            } else {
                agent.direction = Math.random() < 0.5 ? 1 : -1;
            }

            // Move
            agent.position = wrap(agent.position + agent.direction * agent.speed * this.physarum.speed);

            // Deposit pheromone
            const idx = Math.floor(agent.position) % len;
            this.state.pheromone[idx] = Math.min(1, this.state.pheromone[idx] + this.physarum.depositAmount);
        }

        // Diffuse and decay
        this.state.pheromoneBuffer.set(this.state.pheromone);
        for (let i = 0; i < len; i++) {
            const prev = (i === 0) ? len - 1 : i - 1;
            const next = (i + 1) % len;

            const diffused = this.state.pheromoneBuffer[i] * (1 - this.physarum.diffusionRate) +
                (this.state.pheromoneBuffer[prev] + this.state.pheromoneBuffer[next]) *
                (this.physarum.diffusionRate / 2);

            this.state.pheromone[i] = diffused * this.physarum.decayRate;
        }

        // Apply activity heat
        for (let i = 0; i < len; i++) {
            if (this.state.activityHeat[i] > 0) {
                this.state.pheromone[i] = Math.min(1, this.state.pheromone[i] + this.state.activityHeat[i]);
                this.state.activityHeat[i] *= 0.9;
            }
        }
    },

    // Convert border index to canvas coordinates
    indexToCoord(index) {
        const w = window.innerWidth;
        const h = window.innerHeight - 40;
        const len = this.state.borderLength;

        index = ((index % len) + len) % len;

        // Top edge
        if (index < w) {
            return { side: 'top', x: index, y: 10 };
        }
        // Right edge
        if (index < w + h) {
            return { side: 'right', x: 10, y: index - w };
        }
        // Bottom edge
        if (index < 2 * w + h) {
            return { side: 'bottom', x: 2 * w + h - index, y: 10 };
        }
        // Left edge
        return { side: 'left', x: 10, y: 2 * w + 2 * h - index };
    },

    // Render frame
    render() {
        const palette = this.palettes[this.config.palette] || this.palettes.phosphor_green;

        // Clear all canvases
        for (const side of ['top', 'bottom', 'left', 'right']) {
            const ctx = this.contexts[side];
            const canvas = this.canvases[side];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        // Check glitch state
        if (this.state.inGlitch && Date.now() > this.state.glitchUntil) {
            this.state.inGlitch = false;
            document.getElementById('tro-border')?.classList.remove('tro-glitch-active');
        }

        // Render pheromone field
        const len = this.state.borderLength;
        for (let i = 0; i < len; i += 2) {
            const intensity = this.state.pheromone[i];
            if (intensity < 0.05) continue;

            const coord = this.indexToCoord(i);
            const ctx = this.contexts[coord.side];
            if (!ctx) continue;

            // Draw glowing particle
            const size = 3 + intensity * 5;
            const alpha = intensity * this.config.glowIntensity;

            // Glow
            const gradient = ctx.createRadialGradient(coord.x, coord.y, 0, coord.x, coord.y, size * 2);
            gradient.addColorStop(0, `rgba(${palette.primary.join(',')}, ${alpha})`);
            gradient.addColorStop(0.5, `rgba(${palette.dim.join(',')}, ${alpha * 0.5})`);
            gradient.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.arc(coord.x, coord.y, size * 2, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(coord.x, coord.y, size * 0.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${palette.primary.join(',')}, ${alpha})`;
            ctx.fill();
        }

        this.state.frame++;
    },

    // Start render loop
    startRenderLoop() {
        const frameTime = 1000 / this.config.fps;
        let lastFrame = 0;

        const loop = (timestamp) => {
            if (!this.config.enabled) return;

            if (timestamp - lastFrame >= frameTime) {
                this.updatePhysarum();
                this.render();
                lastFrame = timestamp;
            }

            requestAnimationFrame(loop);
        };

        requestAnimationFrame(loop);
    },

    // Handle window resize
    handleResize() {
        this.resizeCanvases();
        const oldLen = this.state.borderLength;
        this.calculateBorderLength();
        const newLen = this.state.borderLength;

        if (oldLen !== newLen) {
            // Resize arrays
            const newPheromone = new Float32Array(newLen);
            const newBuffer = new Float32Array(newLen);
            const newHeat = new Float32Array(newLen);

            for (let i = 0; i < Math.min(oldLen, newLen); i++) {
                newPheromone[i] = this.state.pheromone[i];
                newHeat[i] = this.state.activityHeat[i];
            }

            this.state.pheromone = newPheromone;
            this.state.pheromoneBuffer = newBuffer;
            this.state.activityHeat = newHeat;

            // Rescale agent positions
            for (const agent of this.state.agents) {
                agent.position = (agent.position / oldLen) * newLen;
            }
        }
    },

    // Inject activity pulse
    pulse(position, intensity, radius) {
        const len = this.state.borderLength;
        const center = Math.floor(position * len);

        for (let offset = 0; offset <= radius; offset++) {
            const decay = 1 - offset / (radius + 1);
            const value = intensity * decay;

            const left = ((center - offset) % len + len) % len;
            const right = (center + offset) % len;

            this.state.activityHeat[left] = Math.min(1, this.state.activityHeat[left] + value);
            if (offset > 0) {
                this.state.activityHeat[right] = Math.min(1, this.state.activityHeat[right] + value);
            }
        }

        // Spawn extra agents at pulse location
        const spawnCount = Math.ceil(intensity * 5);
        for (let i = 0; i < spawnCount; i++) {
            this.state.agents.push({
                position: center + (Math.random() - 0.5) * radius,
                direction: Math.random() < 0.5 ? 1 : -1,
                speed: 0.5 + intensity * 0.5,
                active: true
            });
        }

        // Limit agent count
        if (this.state.agents.length > this.config.agentCount * 2) {
            this.state.agents = this.state.agents.slice(-this.config.agentCount);
        }
    },

    // Trigger glitch effect
    glitch(durationMs) {
        this.state.inGlitch = true;
        this.state.glitchUntil = Date.now() + durationMs;
        document.getElementById('tro-border')?.classList.add('tro-glitch-active');
    },

    // Set palette
    setPalette(name) {
        if (this.palettes[name]) {
            this.config.palette = name;
        }
    },

    // Pause/resume
    pause() { this.state.paused = true; },
    resume() { this.state.paused = false; },

    // Enable/disable
    enable() { this.config.enabled = true; this.startRenderLoop(); },
    disable() { this.config.enabled = false; }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => TRO.init());
} else {
    TRO.init();
}

// Expose to window for external control
window.TRO = TRO;
";

/// Audio feedback system using Web Audio API.
///
/// Provides subtle audio cues for UI interactions with configurable settings.
/// All sounds are synthesized (no external files required).
pub const AUDIO_SCRIPT: &str = r#"
// ============================================
// NEUMANN AUDIO FEEDBACK SYSTEM
// ============================================

const NeumannAudio = (function() {
    'use strict';

    let ctx = null;
    let masterGain = null;
    let enabled = true;
    let volume = 0.3;

    // Settings per category
    const settings = {
        navigation: true,
        feedback: true,
        achievements: true
    };

    // Initialize audio context (must be called after user interaction)
    function init() {
        if (ctx) return true;

        try {
            ctx = new (window.AudioContext || window.webkitAudioContext)();
            masterGain = ctx.createGain();
            masterGain.gain.value = volume;
            masterGain.connect(ctx.destination);

            // Load saved settings
            loadSettings();
            return true;
        } catch (e) {
            console.warn('Web Audio API not supported:', e);
            return false;
        }
    }

    // Resume context if suspended (required by browsers after page load)
    function resume() {
        if (ctx && ctx.state === 'suspended') {
            ctx.resume();
        }
    }

    // Load settings from localStorage
    function loadSettings() {
        try {
            const saved = localStorage.getItem('neumann_audio_settings');
            if (saved) {
                const parsed = JSON.parse(saved);
                enabled = parsed.enabled ?? true;
                volume = parsed.volume ?? 0.3;
                Object.assign(settings, parsed.settings ?? {});
                if (masterGain) {
                    masterGain.gain.value = volume;
                }
            }
        } catch (e) {
            // Ignore localStorage errors
        }
    }

    // Save settings to localStorage
    function saveSettings() {
        try {
            localStorage.setItem('neumann_audio_settings', JSON.stringify({
                enabled,
                volume,
                settings
            }));
        } catch (e) {
            // Ignore localStorage errors
        }
    }

    // Create oscillator with envelope
    function createTone(frequency, duration, type = 'sine', attackTime = 0.01, decayTime = 0.1) {
        if (!ctx || !enabled) return;

        const osc = ctx.createOscillator();
        const env = ctx.createGain();

        osc.type = type;
        osc.frequency.setValueAtTime(frequency, ctx.currentTime);

        // ADSR-ish envelope
        env.gain.setValueAtTime(0, ctx.currentTime);
        env.gain.linearRampToValueAtTime(0.8, ctx.currentTime + attackTime);
        env.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

        osc.connect(env);
        env.connect(masterGain);

        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + duration);
    }

    // Play a click sound for navigation
    function playClick() {
        if (!enabled || !settings.navigation) return;
        init();
        resume();

        // Soft click with slight pitch variation
        const freq = 440 * (0.95 + Math.random() * 0.1);
        createTone(freq, 0.05, 'sine', 0.005, 0.04);
    }

    // Play success sound
    function playSuccess() {
        if (!enabled || !settings.feedback) return;
        init();
        resume();

        // Ascending two-tone
        createTone(523.25, 0.1, 'sine', 0.01, 0.08);  // C5
        setTimeout(() => {
            createTone(659.25, 0.15, 'sine', 0.01, 0.12);  // E5
        }, 80);
    }

    // Play error sound
    function playError() {
        if (!enabled || !settings.feedback) return;
        init();
        resume();

        // Descending dissonant tone
        createTone(220, 0.15, 'sawtooth', 0.01, 0.12);  // A3
        setTimeout(() => {
            createTone(165, 0.2, 'sawtooth', 0.01, 0.15);  // E3
        }, 100);
    }

    // Play typing sound
    function playType() {
        if (!enabled || !settings.navigation) return;
        init();
        resume();

        // Very soft, quick click
        const freq = 800 * (0.9 + Math.random() * 0.2);
        createTone(freq, 0.02, 'sine', 0.002, 0.015);
    }

    // Play query execute sound (whoosh)
    function playQueryStart() {
        if (!enabled || !settings.feedback) return;
        init();
        resume();

        // Rising whoosh using filtered noise simulation
        const osc = ctx.createOscillator();
        const env = ctx.createGain();
        const filter = ctx.createBiquadFilter();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(200, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(600, ctx.currentTime + 0.15);

        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(400, ctx.currentTime);
        filter.frequency.exponentialRampToValueAtTime(2000, ctx.currentTime + 0.15);

        env.gain.setValueAtTime(0, ctx.currentTime);
        env.gain.linearRampToValueAtTime(0.3, ctx.currentTime + 0.02);
        env.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);

        osc.connect(filter);
        filter.connect(env);
        env.connect(masterGain);

        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.15);
    }

    // Play query complete sound (satisfying click/lock)
    function playQueryComplete() {
        if (!enabled || !settings.feedback) return;
        init();
        resume();

        // Satisfying "lock" sound
        createTone(880, 0.03, 'square', 0.002, 0.025);
        setTimeout(() => {
            createTone(1100, 0.06, 'sine', 0.005, 0.05);
        }, 25);
    }

    // Play achievement unlock sound
    function playAchievement() {
        if (!enabled || !settings.achievements) return;
        init();
        resume();

        // Ascending arpeggio
        const notes = [523.25, 659.25, 783.99, 1046.5];  // C5, E5, G5, C6
        notes.forEach((freq, i) => {
            setTimeout(() => {
                createTone(freq, 0.2, 'sine', 0.01, 0.15);
            }, i * 80);
        });

        // Final sparkle
        setTimeout(() => {
            createTone(1568, 0.3, 'triangle', 0.02, 0.25);  // G6
        }, 350);
    }

    // Play level up sound
    function playLevelUp() {
        if (!enabled || !settings.achievements) return;
        init();
        resume();

        // Triumphant fanfare
        const notes = [392, 523.25, 659.25, 783.99];  // G4, C5, E5, G5
        notes.forEach((freq, i) => {
            setTimeout(() => {
                createTone(freq, 0.25, 'sine', 0.02, 0.2);
            }, i * 120);
        });

        // Chord at the end
        setTimeout(() => {
            createTone(523.25, 0.4, 'sine', 0.02, 0.35);
            createTone(659.25, 0.4, 'sine', 0.02, 0.35);
            createTone(783.99, 0.4, 'sine', 0.02, 0.35);
        }, 500);
    }

    // Play navigation page change sound
    function playNavigate() {
        if (!enabled || !settings.navigation) return;
        init();
        resume();

        // Soft slide sound
        const osc = ctx.createOscillator();
        const env = ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(600, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.08);

        env.gain.setValueAtTime(0, ctx.currentTime);
        env.gain.linearRampToValueAtTime(0.2, ctx.currentTime + 0.01);
        env.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);

        osc.connect(env);
        env.connect(masterGain);

        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.08);
    }

    // Play hover sound
    function playHover() {
        if (!enabled || !settings.navigation) return;
        init();
        resume();

        // Very subtle high-frequency blip
        createTone(1200, 0.015, 'sine', 0.002, 0.01);
    }

    // Set master volume (0.0 - 1.0)
    function setVolume(v) {
        volume = Math.max(0, Math.min(1, v));
        if (masterGain) {
            masterGain.gain.value = volume;
        }
        saveSettings();
    }

    // Get current volume
    function getVolume() {
        return volume;
    }

    // Enable/disable all audio
    function setEnabled(e) {
        enabled = !!e;
        saveSettings();
    }

    // Check if enabled
    function isEnabled() {
        return enabled;
    }

    // Set category setting
    function setCategorySetting(category, value) {
        if (category in settings) {
            settings[category] = !!value;
            saveSettings();
        }
    }

    // Get category setting
    function getCategorySetting(category) {
        return settings[category] ?? false;
    }

    // Get all settings
    function getSettings() {
        return {
            enabled,
            volume,
            navigation: settings.navigation,
            feedback: settings.feedback,
            achievements: settings.achievements
        };
    }

    // Apply settings
    function applySettings(s) {
        if (typeof s.enabled === 'boolean') enabled = s.enabled;
        if (typeof s.volume === 'number') setVolume(s.volume);
        if (typeof s.navigation === 'boolean') settings.navigation = s.navigation;
        if (typeof s.feedback === 'boolean') settings.feedback = s.feedback;
        if (typeof s.achievements === 'boolean') settings.achievements = s.achievements;
        saveSettings();
    }

    // Public API
    return {
        init,
        resume,

        // Sound effects
        playClick,
        playSuccess,
        playError,
        playType,
        playQueryStart,
        playQueryComplete,
        playAchievement,
        playLevelUp,
        playNavigate,
        playHover,

        // Settings
        setVolume,
        getVolume,
        setEnabled,
        isEnabled,
        setCategorySetting,
        getCategorySetting,
        getSettings,
        applySettings
    };
})();

// Initialize on first user interaction
document.addEventListener('click', function initAudio() {
    NeumannAudio.init();
    document.removeEventListener('click', initAudio);
}, { once: true });

// Also initialize on keypress
document.addEventListener('keydown', function initAudioKey() {
    NeumannAudio.init();
    document.removeEventListener('keydown', initAudioKey);
}, { once: true });

// Expose to window
window.NeumannAudio = NeumannAudio;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_css_variables_defined() {
        assert!(ADMIN_CSS.contains("--phosphor-green"));
        assert!(ADMIN_CSS.contains("--blood-rust"));
        assert!(ADMIN_CSS.contains("--amber-glow"));
        assert!(ADMIN_CSS.contains("--deep-black"));
    }

    #[test]
    fn test_crt_effects_defined() {
        assert!(ADMIN_CSS.contains("crt-scanlines"));
        assert!(ADMIN_CSS.contains("crt-flicker"));
        assert!(ADMIN_CSS.contains("crt-vignette"));
        assert!(ADMIN_CSS.contains("phosphor-glow"));
    }

    #[test]
    fn test_terminal_components_defined() {
        assert!(ADMIN_CSS.contains("btn-terminal"));
        assert!(ADMIN_CSS.contains("table-rust"));
        assert!(ADMIN_CSS.contains("input-terminal"));
        assert!(ADMIN_CSS.contains("terminal-panel"));
        assert!(ADMIN_CSS.contains("stat-card"));
    }

    #[test]
    fn test_tailwind_config_has_terminal_theme() {
        assert!(TAILWIND_CONFIG.contains("phosphor"));
        assert!(TAILWIND_CONFIG.contains("terminal"));
        assert!(TAILWIND_CONFIG.contains("VT323"));
        assert!(TAILWIND_CONFIG.contains("Orbitron"));
    }

    #[test]
    fn test_tro_css_border_classes() {
        assert!(TRO_CSS.contains("tro-border"));
        assert!(TRO_CSS.contains("tro-border-top"));
        assert!(TRO_CSS.contains("tro-border-bottom"));
        assert!(TRO_CSS.contains("tro-border-left"));
        assert!(TRO_CSS.contains("tro-border-right"));
    }

    #[test]
    fn test_tro_css_pulse_classes() {
        assert!(TRO_CSS.contains("tro-pulse"));
        assert!(TRO_CSS.contains("tro-pulse-put"));
        assert!(TRO_CSS.contains("tro-pulse-get"));
        assert!(TRO_CSS.contains("tro-pulse-delete"));
        assert!(TRO_CSS.contains("tro-pulse-error"));
    }

    #[test]
    fn test_tro_css_effects() {
        assert!(TRO_CSS.contains("tro-glitch-active"));
        assert!(TRO_CSS.contains("tro-scanlines"));
        assert!(TRO_CSS.contains("tro-glow-overlay"));
    }

    #[test]
    fn test_tro_script_physarum_simulation() {
        assert!(TRO_SCRIPT.contains("TRO"));
        assert!(TRO_SCRIPT.contains("Physarum"));
        assert!(TRO_SCRIPT.contains("pheromone"));
        assert!(TRO_SCRIPT.contains("agents"));
    }

    #[test]
    fn test_tro_script_palettes() {
        assert!(TRO_SCRIPT.contains("phosphor_green"));
        assert!(TRO_SCRIPT.contains("amber"));
        assert!(TRO_SCRIPT.contains("rust"));
        assert!(TRO_SCRIPT.contains("ghost"));
        assert!(TRO_SCRIPT.contains("glitch"));
    }

    #[test]
    fn test_tro_script_api() {
        assert!(TRO_SCRIPT.contains("init()"));
        assert!(TRO_SCRIPT.contains("pulse("));
        assert!(TRO_SCRIPT.contains("glitch("));
        assert!(TRO_SCRIPT.contains("setPalette("));
    }

    #[test]
    fn test_tro_script_canvas_rendering() {
        assert!(TRO_SCRIPT.contains("createCanvases"));
        assert!(TRO_SCRIPT.contains("getContext('2d')"));
        assert!(TRO_SCRIPT.contains("requestAnimationFrame"));
    }

    #[test]
    fn test_audio_script_initialization() {
        assert!(AUDIO_SCRIPT.contains("NeumannAudio"));
        assert!(AUDIO_SCRIPT.contains("AudioContext"));
        assert!(AUDIO_SCRIPT.contains("init()"));
        assert!(AUDIO_SCRIPT.contains("resume()"));
    }

    #[test]
    fn test_audio_script_sound_effects() {
        assert!(AUDIO_SCRIPT.contains("playClick"));
        assert!(AUDIO_SCRIPT.contains("playSuccess"));
        assert!(AUDIO_SCRIPT.contains("playError"));
        assert!(AUDIO_SCRIPT.contains("playType"));
        assert!(AUDIO_SCRIPT.contains("playQueryStart"));
        assert!(AUDIO_SCRIPT.contains("playQueryComplete"));
        assert!(AUDIO_SCRIPT.contains("playAchievement"));
        assert!(AUDIO_SCRIPT.contains("playLevelUp"));
        assert!(AUDIO_SCRIPT.contains("playNavigate"));
        assert!(AUDIO_SCRIPT.contains("playHover"));
    }

    #[test]
    fn test_audio_script_settings() {
        assert!(AUDIO_SCRIPT.contains("setVolume"));
        assert!(AUDIO_SCRIPT.contains("getVolume"));
        assert!(AUDIO_SCRIPT.contains("setEnabled"));
        assert!(AUDIO_SCRIPT.contains("isEnabled"));
        assert!(AUDIO_SCRIPT.contains("setCategorySetting"));
        assert!(AUDIO_SCRIPT.contains("getCategorySetting"));
    }

    #[test]
    fn test_audio_script_persistence() {
        assert!(AUDIO_SCRIPT.contains("localStorage"));
        assert!(AUDIO_SCRIPT.contains("neumann_audio_settings"));
        assert!(AUDIO_SCRIPT.contains("loadSettings"));
        assert!(AUDIO_SCRIPT.contains("saveSettings"));
    }

    #[test]
    fn test_audio_script_web_audio_api() {
        assert!(AUDIO_SCRIPT.contains("createOscillator"));
        assert!(AUDIO_SCRIPT.contains("createGain"));
        assert!(AUDIO_SCRIPT.contains("createBiquadFilter"));
        assert!(AUDIO_SCRIPT.contains("masterGain"));
    }

    #[test]
    fn test_audio_script_exposed_to_window() {
        assert!(AUDIO_SCRIPT.contains("window.NeumannAudio"));
    }
}
