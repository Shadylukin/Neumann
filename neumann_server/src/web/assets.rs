// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Embedded CSS and JavaScript assets for the Memoria design system.
//!
//! Monochromatic, dark-themed, data-forward interface using opacity-based
//! hierarchy, blur-to-clear animations, and strictly neutral colors.

/// Custom CSS for the Memoria admin interface.
///
/// Features:
/// - Neutral monochromatic palette (white on near-black)
/// - Opacity-based visual hierarchy (100/80/60/40/20%)
/// - Blur-to-clear entry animations
/// - `Inter` + `JetBrains Mono` typography
pub const ADMIN_CSS: &str = r"
/* ============================================
   MEMORIA DESIGN SYSTEM
   ============================================ */

/* Design Tokens */
:root {
    /* Background scale (neutral, dark-to-light) */
    --bg-primary: #0a0a0a;
    --bg-elevated: #171717;
    --bg-surface: #1f1f1f;
    --bg-hover: #262626;
    --bg-active: #2e2e2e;

    /* Text scale (white at varying opacity) */
    --text-primary: #ffffff;
    --text-secondary: #a3a3a3;
    --text-tertiary: #737373;
    --text-disabled: #525252;
    --text-ghost: #404040;

    /* Border scale */
    --border-default: #262626;
    --border-subtle: #1f1f1f;
    --border-emphasis: #404040;

    /* Typography Scale */
    --text-xs: 0.75rem;
    --text-sm: 0.875rem;
    --text-base: 1rem;
    --text-lg: 1.25rem;
    --text-xl: 1.5rem;
    --text-2xl: 2rem;
    --text-3xl: 3rem;

    /* Font Families */
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'Courier New', monospace;
}

/* ============================================
   BASE STYLES
   ============================================ */

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: var(--text-base);
    font-weight: 400;
    line-height: 1.5;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-elevated);
}

::-webkit-scrollbar-thumb {
    background: var(--border-emphasis);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-tertiary);
}

/* Selection */
::selection {
    background: rgba(255, 255, 255, 0.2);
    color: var(--text-primary);
}

/* ============================================
   CARD COMPONENT (.m-card)
   ============================================ */

.m-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    position: relative;
}

.m-card-header {
    border-bottom: 1px solid var(--border-default);
    padding: 0.75rem 1rem;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.m-card-content {
    padding: 1rem;
}

.m-card-footer {
    border-top: 1px solid var(--border-default);
    padding: 0.5rem 1rem;
    font-size: var(--text-xs);
    color: var(--text-tertiary);
}

/* ============================================
   BUTTON COMPONENT (.m-btn)
   ============================================ */

.m-btn {
    background: transparent;
    border: 1px solid var(--border-emphasis);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-weight: 400;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: background 150ms ease, transform 100ms ease;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: var(--text-sm);
}

.m-btn:hover {
    background: var(--bg-hover);
}

.m-btn:active {
    transform: scale(0.98);
}

.m-btn-active {
    background: var(--bg-active);
    border-color: var(--border-emphasis);
}

.m-btn:disabled {
    color: var(--text-disabled);
    border-color: var(--border-default);
    cursor: not-allowed;
}

/* ============================================
   STATUS DOT (.m-dot)
   ============================================ */

.m-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 9999px;
    background-color: white;
}

/* ============================================
   KEYBOARD HINT (.m-kbd)
   ============================================ */

.m-kbd {
    display: inline-block;
    padding: 0.125rem 0.375rem;
    border: 1px solid var(--border-emphasis);
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-right: 0.25rem;
}

/* ============================================
   DATA TABLE (.m-table)
   ============================================ */

.m-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-sans);
}

.m-table th {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-default);
    padding: 0.75rem 1rem;
    text-align: left;
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: var(--text-xs);
    font-weight: 400;
}

.m-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}

.m-table tr:hover {
    background: var(--bg-hover);
}

/* ============================================
   INPUT FIELDS (.m-input)
   ============================================ */

.m-input {
    background: var(--bg-primary);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: var(--text-primary);
    font-family: var(--font-sans);
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    caret-color: var(--text-primary);
    transition: border-color 200ms ease;
}

.m-input::placeholder {
    color: var(--text-ghost);
}

.m-input:focus {
    outline: none;
    border-color: rgba(255, 255, 255, 0.6);
}

.m-input:disabled {
    border-color: var(--border-subtle);
    color: var(--text-disabled);
    cursor: not-allowed;
}

/* ============================================
   STAT COMPONENT (.m-stat)
   ============================================ */

.m-stat {
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    padding: 1rem;
    transition: background 150ms ease;
}

.m-stat:hover {
    background: var(--bg-surface);
}

.m-stat-label {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.m-stat-value {
    font-size: var(--text-3xl);
    font-weight: 300;
    color: var(--text-primary);
    margin: 0.25rem 0;
    font-family: var(--font-sans);
}

.m-stat-subtitle {
    font-size: var(--text-xs);
    color: var(--text-ghost);
}

/* ============================================
   MODAL (.m-modal)
   ============================================ */

.m-modal {
    position: fixed;
    inset: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    visibility: hidden;
}

.m-modal.active {
    opacity: 1;
    visibility: visible;
}

.m-modal-body {
    max-width: 90vw;
    max-height: 85vh;
    overflow: auto;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 16px;
    padding: 1.5rem;
}

/* ============================================
   NAVIGATION (.m-nav)
   ============================================ */

.m-nav {
    font-family: var(--font-sans);
}

.m-nav a {
    color: var(--text-tertiary);
    text-decoration: none;
    padding: 0.5rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: color 150ms ease, background 150ms ease;
    border-left: 2px solid transparent;
    border-radius: 0 8px 8px 0;
}

.m-nav a:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
}

.m-nav a.active {
    color: var(--text-primary);
    border-left-color: var(--text-primary);
    background: var(--bg-surface);
}

/* Breadcrumb */
.m-breadcrumb {
    font-size: var(--text-sm);
    color: var(--text-tertiary);
}

.m-breadcrumb a {
    color: var(--text-tertiary);
    text-decoration: none;
}

.m-breadcrumb a:hover {
    color: var(--text-primary);
}

.m-breadcrumb .separator {
    margin: 0 0.5rem;
    color: var(--text-ghost);
}

/* ============================================
   LOADING STATES
   ============================================ */

.htmx-request {
    opacity: 0.7;
    pointer-events: none;
}

/* ============================================
   EXPANDABLE CONTENT (.m-expandable)
   ============================================ */

.m-expandable {
    position: relative;
}

.m-expand-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.5rem;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    background: transparent;
    border: 1px solid var(--border-default);
    border-radius: 4px;
    cursor: pointer;
    transition: color 150ms ease, border-color 150ms ease;
}

.m-expand-btn:hover {
    color: var(--text-primary);
    border-color: var(--border-emphasis);
}

details.m-expandable-details summary {
    cursor: pointer;
    list-style: none;
}

details.m-expandable-details summary::-webkit-details-marker {
    display: none;
}

details.m-expandable-details[open] .m-expand-btn {
    background: var(--bg-hover);
    border-color: var(--border-emphasis);
}

/* ============================================
   TOOLTIP
   ============================================ */

.m-tooltip {
    position: relative;
}

.m-tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.375rem 0.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: var(--text-xs);
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: opacity 150ms ease, visibility 150ms ease;
    z-index: 50;
}

.m-tooltip:hover::after {
    opacity: 1;
    visibility: visible;
}

/* ============================================
   GRAPH VISUALIZATION
   ============================================ */

#graph-container {
    background: var(--bg-primary);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    position: relative;
    z-index: 1;
    overflow: hidden;
}

#graph-container canvas {
    position: relative;
    z-index: 1;
}

.m-card:has(#graph-container) {
    overflow: hidden;
    position: relative;
    z-index: 1;
}

.m-card:has(#graph-container) .m-card-content {
    overflow: hidden;
    padding: 0;
}

/* ============================================
   TERMINAL OUTPUT
   ============================================ */

.m-terminal-output {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.4;
    max-height: 200px;
    overflow-y: auto;
}

.m-terminal-output-line {
    padding: 0.125rem 0;
    color: var(--text-tertiary);
}

.m-terminal-output-line.success {
    color: var(--text-primary);
}

.m-terminal-output-line.error {
    color: var(--text-tertiary);
    opacity: 0.6;
}

.m-terminal-output-line.warning {
    color: var(--text-secondary);
}

.m-terminal-input-field {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: inherit;
    outline: none;
    caret-color: var(--text-primary);
}

.m-terminal-textarea {
    width: 100%;
    min-height: 60px;
    max-height: 200px;
    resize: none;
    line-height: 1.4;
    padding: 0.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 8px;
    color: var(--text-primary);
    font-family: var(--font-mono);
}

.m-terminal-textarea:focus {
    border-color: var(--border-emphasis);
}

.m-terminal-textarea::placeholder {
    color: var(--text-ghost);
}

.m-terminal-input-line {
    display: block;
    font-family: var(--font-mono);
}

/* ============================================
   OPACITY UTILITIES
   ============================================ */

.opacity-hero { opacity: 1; }
.opacity-primary { opacity: 0.8; }
.opacity-secondary { opacity: 0.6; }
.opacity-tertiary { opacity: 0.4; }
.opacity-ghost { opacity: 0.2; }

/* ============================================
   ANIMATIONS
   ============================================ */

@keyframes blur-reveal {
    from {
        filter: blur(8px);
        opacity: 0;
    }
    to {
        filter: blur(0);
        opacity: 1;
    }
}

.animate-blur-reveal {
    animation: blur-reveal 400ms ease-out forwards;
}

@keyframes stagger-entry {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.stagger-item {
    animation: stagger-entry 300ms ease-out forwards;
    opacity: 0;
}

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
   LAMP CONTAINER (empty state)
   ============================================ */

.m-lamp {
    background: radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.05) 0%, transparent 60%);
}

/* ============================================
   BORDER RADIUS TOKENS
   ============================================ */

:root {
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* ============================================
   INTERACTIVE CARD (.m-card-interactive)
   ============================================ */

.m-card-interactive {
    background: rgba(23, 23, 23, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    position: relative;
    transition: transform 200ms ease, background 200ms ease;
}

.m-card-interactive:hover {
    transform: translateY(-1px);
    background: rgba(23, 23, 23, 0.8);
}

/* ============================================
   TYPOGRAPHY EXTENSIONS
   ============================================ */

.text-hero {
    font-weight: 100;
    font-size: 48px;
    line-height: 1.1;
}

.text-display {
    font-weight: 300;
    font-size: 32px;
    line-height: 1.2;
}

.text-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-tertiary);
    font-size: 12px;
}

/* ============================================
   BADGE (.m-badge)
   ============================================ */

.m-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.125rem 0.5rem;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    color: var(--text-secondary);
    font-family: var(--font-mono);
}

/* ============================================
   TABS (.m-tabs)
   ============================================ */

.m-tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border-default);
    margin-bottom: 1rem;
}

.m-tab {
    padding: 0.5rem 1rem;
    color: var(--text-tertiary);
    text-decoration: none;
    font-size: var(--text-sm);
    border-bottom: 2px solid transparent;
    transition: color 150ms ease, border-color 150ms ease;
}

.m-tab:hover {
    color: var(--text-secondary);
}

.m-tab.active {
    color: var(--text-primary);
    border-bottom-color: var(--text-primary);
}

/* ============================================
   SCROLL FADE (.m-scroll-fade)
   ============================================ */

.m-scroll-fade {
    mask-image: linear-gradient(to bottom, black 80%, transparent 100%);
    -webkit-mask-image: linear-gradient(to bottom, black 80%, transparent 100%);
}

/* ============================================
   FOCUS BLUR (.focus-blur)
   ============================================ */

.focus-blur {
    filter: blur(4px);
    opacity: 0.6;
    transition: filter 300ms ease, opacity 300ms ease;
}

/* ============================================
   DOT PULSE (.m-dot-pulse)
   ============================================ */

.m-dot-pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-primary);
    animation: m-dot-pulse-anim 1.5s ease-in-out infinite;
}

@keyframes m-dot-pulse-anim {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
}

/* ============================================
   ICON UTILITIES
   ============================================ */

.m-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.25rem;
    height: 1.25rem;
    flex-shrink: 0;
}

.m-icon svg {
    width: 100%;
    height: 100%;
}

.m-icon-sm {
    width: 1rem;
    height: 1rem;
}

.m-icon-lg {
    width: 1.5rem;
    height: 1.5rem;
}

/* ============================================
   REDUCED MOTION
   ============================================ */

@media (prefers-reduced-motion: reduce) {
    .stagger-item,
    .animate-blur-reveal,
    .m-dot-pulse {
        animation: none;
        opacity: 1;
        filter: none;
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
    outline: 2px solid var(--text-primary);
    outline-offset: 2px;
}

/* ============================================
   PRINT STYLES
   ============================================ */

@media print {
    body {
        background: white;
        color: black;
    }
}

/* ============================================
   TOAST NOTIFICATIONS (.m-toast)
   ============================================ */

.m-toast-container {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    z-index: 200;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    pointer-events: none;
}

.m-toast {
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: var(--text-sm);
    color: var(--text-secondary);
    max-width: 24rem;
    pointer-events: auto;
    animation: m-toast-in 200ms ease forwards;
}

.m-toast.m-toast-out {
    animation: m-toast-out 200ms ease forwards;
}

.m-toast-success {
    opacity: 1;
}

.m-toast-error {
    opacity: 0.8;
}

@keyframes m-toast-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes m-toast-out {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(8px); }
}

/* ============================================
   PAGINATION
   ============================================ */

.m-pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
";

/// Tailwind CSS configuration for the Memoria design system.
///
/// Extends default Tailwind with neutral colors and Inter/JetBrains Mono fonts.
pub const TAILWIND_CONFIG: &str = r"
tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                mono: ['JetBrains Mono', 'Courier New', 'monospace'],
            },
            colors: {
                neutral: {
                    750: '#333333',
                    850: '#1f1f1f',
                }
            },
            borderRadius: {
                'sm': '8px',
                'DEFAULT': '12px',
                'lg': '16px',
            }
        }
    }
}
";

/// JavaScript for the Memoria design system interactions.
///
/// Features:
/// - Animated counters with 600ms tween
/// - Card glare effect on mousemove
/// - Blur-to-clear `IntersectionObserver`
/// - Stagger-entry observer
/// - Spring physics engine for modals/drawers
/// - Focus-blur backdrop manager
pub const MEMORIA_SCRIPT: &str = r"
// ============================================
// MEMORIA DESIGN SYSTEM - INTERACTIONS
// ============================================

(function() {
    'use strict';

    // ---- Animated Counters ----
    function animateCounter(el, target, duration) {
        duration = duration || 600;
        var start = parseInt(el.textContent.replace(/[^0-9]/g, ''), 10) || 0;
        var delta = target - start;
        if (delta === 0) return;
        var accel = Math.abs(delta) > 1000;
        var startTime = null;

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var progress = Math.min((timestamp - startTime) / duration, 1);
            var eased = accel
                ? progress * progress * (3 - 2 * progress)
                : progress;
            var current = Math.round(start + delta * eased);
            el.textContent = current.toLocaleString();
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    // ---- Card Glare ----
    function initCardGlare() {
        document.querySelectorAll('.m-card, .m-stat').forEach(function(card) {
            if (card.dataset.glareInit) return;
            card.dataset.glareInit = '1';
            var glare = null;
            var targetX = 50, targetY = 50;
            var currentX = 50, currentY = 50;
            var raf = null;

            card.addEventListener('mouseenter', function() {
                glare = document.createElement('div');
                glare.style.cssText = 'position:absolute;inset:0;pointer-events:none;border-radius:inherit;z-index:1;';
                card.style.position = 'relative';
                card.appendChild(glare);
            });

            card.addEventListener('mousemove', function(e) {
                if (!glare) return;
                var rect = card.getBoundingClientRect();
                targetX = ((e.clientX - rect.left) / rect.width) * 100;
                targetY = ((e.clientY - rect.top) / rect.height) * 100;
                if (!raf) {
                    raf = requestAnimationFrame(function lerpGlare() {
                        currentX += (targetX - currentX) * 0.1;
                        currentY += (targetY - currentY) * 0.1;
                        if (glare) {
                            glare.style.background = 'radial-gradient(60% 60% at ' + currentX + '% ' + currentY + '%, rgba(255,255,255,0.03), transparent)';
                        }
                        if (Math.abs(targetX - currentX) > 0.5 || Math.abs(targetY - currentY) > 0.5) {
                            raf = requestAnimationFrame(lerpGlare);
                        } else {
                            raf = null;
                        }
                    });
                }
            });

            card.addEventListener('mouseleave', function() {
                if (glare) { glare.remove(); glare = null; }
                raf = null;
            });
        });
    }

    // ---- Blur-to-Clear Observer ----
    function initBlurReveal() {
        if (!('IntersectionObserver' in window)) return;
        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-blur-reveal');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.blur-reveal').forEach(function(el) {
            observer.observe(el);
        });
    }

    // ---- Stagger Entry Observer ----
    function initStaggerEntry() {
        if (!('IntersectionObserver' in window)) return;
        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    var items = entry.target.querySelectorAll('.stagger-item');
                    items.forEach(function(item, i) {
                        var delay = Math.min(i, 7) * 50;
                        item.style.animationDelay = delay + 'ms';
                    });
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.stagger-container').forEach(function(el) {
            observer.observe(el);
        });
    }

    // ---- Spring Physics Engine ----
    function springAnimate(from, to, config, onUpdate, onComplete) {
        config = config || {};
        var stiffness = config.stiffness || 300;
        var damping = config.damping || 30;
        var mass = config.mass || 1;

        var position = from;
        var velocity = 0;
        var lastTime = null;
        var settled = false;

        function tick(timestamp) {
            if (!lastTime) { lastTime = timestamp; }
            var dt = Math.min((timestamp - lastTime) / 1000, 0.064);
            lastTime = timestamp;

            var springForce = -stiffness * (position - to);
            var dampingForce = -damping * velocity;
            var acceleration = (springForce + dampingForce) / mass;

            velocity += acceleration * dt;
            position += velocity * dt;

            if (Math.abs(position - to) < 0.01 && Math.abs(velocity) < 0.01) {
                position = to;
                settled = true;
            }

            onUpdate(position);

            if (!settled) {
                requestAnimationFrame(tick);
            } else if (onComplete) {
                onComplete();
            }
        }
        requestAnimationFrame(tick);
    }

    // ---- Focus-Blur Backdrop Manager ----
    var backdropActive = false;

    function enableBackdrop() {
        var main = document.getElementById('main-content');
        if (main && !backdropActive) {
            main.style.transition = 'filter 300ms ease, opacity 300ms ease';
            main.style.filter = 'blur(4px)';
            main.style.opacity = '0.6';
            backdropActive = true;
        }
    }

    function disableBackdrop() {
        var main = document.getElementById('main-content');
        if (main && backdropActive) {
            main.style.filter = '';
            main.style.opacity = '';
            backdropActive = false;
        }
    }

    // ---- Modal Spring Open/Close ----
    window.Memoria = {
        animateCounter: animateCounter,
        springAnimate: springAnimate,

        openModal: function(modalEl) {
            if (!modalEl) return;
            var body = modalEl.querySelector('.m-modal-body');
            enableBackdrop();
            modalEl.classList.remove('hidden');
            modalEl.classList.add('flex');
            modalEl.style.opacity = '1';
            modalEl.style.visibility = 'visible';
            if (body) {
                body.style.opacity = '0';
                springAnimate(30, 0, { stiffness: 300, damping: 30, mass: 1 },
                    function(val) {
                        body.style.transform = 'translateY(' + val + 'px)';
                        body.style.opacity = Math.max(0, 1 - Math.abs(val) / 30);
                    },
                    function() {
                        body.style.transform = '';
                        body.style.opacity = '1';
                    }
                );
            }
        },

        closeModal: function(modalEl) {
            if (!modalEl) return;
            var body = modalEl.querySelector('.m-modal-body');
            disableBackdrop();
            if (body) {
                springAnimate(0, 30, { stiffness: 300, damping: 30, mass: 1 },
                    function(val) {
                        body.style.transform = 'translateY(' + val + 'px)';
                        body.style.opacity = Math.max(0, 1 - Math.abs(val) / 30);
                    },
                    function() {
                        modalEl.classList.add('hidden');
                        modalEl.classList.remove('flex');
                        modalEl.style.opacity = '';
                        modalEl.style.visibility = '';
                        body.style.transform = '';
                        body.style.opacity = '';
                    }
                );
            } else {
                modalEl.classList.add('hidden');
                modalEl.classList.remove('flex');
            }
        }
    };

    // ---- Initialize on DOM ready ----
    function init() {
        initCardGlare();
        initBlurReveal();
        initStaggerEntry();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_css_memoria_tokens_defined() {
        assert!(ADMIN_CSS.contains("--bg-primary: #0a0a0a"));
        assert!(ADMIN_CSS.contains("--bg-elevated: #171717"));
        assert!(ADMIN_CSS.contains("--text-primary: #ffffff"));
        assert!(ADMIN_CSS.contains("--text-secondary: #a3a3a3"));
    }

    #[test]
    fn test_css_memoria_components_defined() {
        assert!(ADMIN_CSS.contains(".m-card"));
        assert!(ADMIN_CSS.contains(".m-btn"));
        assert!(ADMIN_CSS.contains(".m-table"));
        assert!(ADMIN_CSS.contains(".m-input"));
        assert!(ADMIN_CSS.contains(".m-stat"));
    }

    #[test]
    fn test_css_no_phosphor_references() {
        assert!(!ADMIN_CSS.contains("phosphor-green"));
        assert!(!ADMIN_CSS.contains("#00ee00"));
        assert!(!ADMIN_CSS.contains("VT323"));
    }

    #[test]
    fn test_css_no_bold_weights() {
        assert!(!ADMIN_CSS.contains("font-weight: 700"));
        assert!(!ADMIN_CSS.contains("font-weight: bold"));
    }

    #[test]
    fn test_css_no_shadows() {
        // CSS uses no box-shadow or drop-shadow
        assert!(!ADMIN_CSS.contains("box-shadow"));
        assert!(!ADMIN_CSS.contains("drop-shadow"));
        assert!(!ADMIN_CSS.contains("text-shadow"));
    }

    #[test]
    fn test_css_no_crt_effects() {
        assert!(!ADMIN_CSS.contains("crt-scanlines"));
        assert!(!ADMIN_CSS.contains("crt-flicker"));
        assert!(!ADMIN_CSS.contains("crt-vignette"));
    }

    #[test]
    fn test_css_blur_reveal_animation() {
        assert!(ADMIN_CSS.contains("blur-reveal"));
        assert!(ADMIN_CSS.contains("stagger-entry"));
    }

    #[test]
    fn test_css_opacity_utilities() {
        assert!(ADMIN_CSS.contains(".opacity-hero"));
        assert!(ADMIN_CSS.contains(".opacity-primary"));
        assert!(ADMIN_CSS.contains(".opacity-secondary"));
        assert!(ADMIN_CSS.contains(".opacity-tertiary"));
        assert!(ADMIN_CSS.contains(".opacity-ghost"));
    }

    #[test]
    fn test_tailwind_config_has_memoria_theme() {
        assert!(TAILWIND_CONFIG.contains("Inter"));
        assert!(TAILWIND_CONFIG.contains("JetBrains Mono"));
        assert!(!TAILWIND_CONFIG.contains("VT323"));
        assert!(!TAILWIND_CONFIG.contains("Orbitron"));
    }

    #[test]
    fn test_memoria_script_spring_physics() {
        assert!(MEMORIA_SCRIPT.contains("springAnimate"));
        assert!(MEMORIA_SCRIPT.contains("stiffness"));
        assert!(MEMORIA_SCRIPT.contains("damping"));
        assert!(MEMORIA_SCRIPT.contains("velocity"));
    }

    #[test]
    fn test_memoria_script_card_glare() {
        assert!(MEMORIA_SCRIPT.contains("initCardGlare"));
        assert!(MEMORIA_SCRIPT.contains("radial-gradient"));
    }

    #[test]
    fn test_memoria_script_blur_reveal() {
        assert!(MEMORIA_SCRIPT.contains("initBlurReveal"));
        assert!(MEMORIA_SCRIPT.contains("IntersectionObserver"));
    }

    #[test]
    fn test_memoria_script_stagger_entry() {
        assert!(MEMORIA_SCRIPT.contains("initStaggerEntry"));
        assert!(MEMORIA_SCRIPT.contains("animationDelay"));
    }

    #[test]
    fn test_memoria_script_modal_api() {
        assert!(MEMORIA_SCRIPT.contains("openModal"));
        assert!(MEMORIA_SCRIPT.contains("closeModal"));
        assert!(MEMORIA_SCRIPT.contains("Memoria"));
    }

    #[test]
    fn test_memoria_script_backdrop_manager() {
        assert!(MEMORIA_SCRIPT.contains("enableBackdrop"));
        assert!(MEMORIA_SCRIPT.contains("disableBackdrop"));
        assert!(MEMORIA_SCRIPT.contains("blur(4px)"));
    }

    #[test]
    fn test_memoria_script_counter_animation() {
        assert!(MEMORIA_SCRIPT.contains("animateCounter"));
        assert!(MEMORIA_SCRIPT.contains("requestAnimationFrame"));
    }

    #[test]
    fn test_memoria_script_no_shadow_usage() {
        assert!(!MEMORIA_SCRIPT.contains("shadowBlur"));
        assert!(!MEMORIA_SCRIPT.contains("shadowColor"));
    }

    #[test]
    fn test_css_interactive_card() {
        assert!(ADMIN_CSS.contains(".m-card-interactive"));
        assert!(ADMIN_CSS.contains("backdrop-filter"));
        assert!(ADMIN_CSS.contains("translateY(-1px)"));
    }

    #[test]
    fn test_css_badge() {
        assert!(ADMIN_CSS.contains(".m-badge"));
        assert!(ADMIN_CSS.contains("rgba(255, 255, 255, 0.08)"));
    }

    #[test]
    fn test_css_tabs() {
        assert!(ADMIN_CSS.contains(".m-tabs"));
        assert!(ADMIN_CSS.contains(".m-tab"));
        assert!(ADMIN_CSS.contains(".m-tab.active"));
    }

    #[test]
    fn test_css_typography_extensions() {
        assert!(ADMIN_CSS.contains(".text-hero"));
        assert!(ADMIN_CSS.contains(".text-display"));
        assert!(ADMIN_CSS.contains(".text-label"));
    }

    #[test]
    fn test_css_radius_tokens() {
        assert!(ADMIN_CSS.contains("--radius-sm: 8px"));
        assert!(ADMIN_CSS.contains("--radius-md: 12px"));
        assert!(ADMIN_CSS.contains("--radius-lg: 16px"));
    }

    #[test]
    fn test_css_dot_pulse() {
        assert!(ADMIN_CSS.contains(".m-dot-pulse"));
        assert!(ADMIN_CSS.contains("m-dot-pulse-anim"));
    }

    #[test]
    fn test_css_icon_utilities() {
        assert!(ADMIN_CSS.contains(".m-icon"));
        assert!(ADMIN_CSS.contains(".m-icon-sm"));
        assert!(ADMIN_CSS.contains(".m-icon-lg"));
    }

    #[test]
    fn test_css_toast() {
        assert!(ADMIN_CSS.contains(".m-toast"));
        assert!(ADMIN_CSS.contains(".m-toast-container"));
        assert!(ADMIN_CSS.contains(".m-toast-success"));
        assert!(ADMIN_CSS.contains(".m-toast-error"));
        assert!(ADMIN_CSS.contains("m-toast-in"));
        assert!(ADMIN_CSS.contains("m-toast-out"));
    }

    #[test]
    fn test_css_pagination() {
        assert!(ADMIN_CSS.contains(".m-pagination"));
    }

    #[test]
    fn test_css_scroll_fade() {
        assert!(ADMIN_CSS.contains(".m-scroll-fade"));
        assert!(ADMIN_CSS.contains("mask-image"));
    }

    #[test]
    fn test_css_focus_blur() {
        assert!(ADMIN_CSS.contains(".focus-blur"));
    }
}
