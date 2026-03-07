// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Vault view handlers for the admin UI.
//!
//! Provides five views for browsing and managing secrets stored in
//! `tensor_vault`: secrets list, secret detail, click-to-reveal, audit log,
//! and vault status dashboard.

use std::fmt::Write;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use maud::{html, Markup};
use tensor_vault::Vault;

use crate::web::icons::{icon_chevron_right, icon_key};
use crate::web::templates::{
    layout, m_badge, m_breadcrumb, m_card, m_empty, m_header, m_pagination, m_stat, m_table_header,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Admin requester identity for vault operations.
const VAULT_REQUESTER: &str = "node:root";

/// Items per page in paginated lists.
const PAGE_SIZE: usize = 20;

/// Query parameters for the secrets list page.
#[derive(Debug, serde::Deserialize)]
pub struct SecretsListParams {
    /// Current page number (1-based).
    page: Option<usize>,
    /// Key pattern filter.
    pattern: Option<String>,
}

/// Query parameters for the audit log page.
#[derive(Debug, serde::Deserialize)]
pub struct AuditParams {
    /// Filter by entity.
    entity: Option<String>,
    /// Filter by key.
    key: Option<String>,
    /// Current page number (1-based).
    page: Option<usize>,
}

/// Format a Unix timestamp (milliseconds) into a human-readable string.
fn format_timestamp(ts: i64) -> String {
    if ts == 0 {
        return "-".to_string();
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let delta_secs = (now_ms - ts) / 1000;
    if delta_secs < 0 {
        return "just now".to_string();
    }
    let mins = delta_secs / 60;
    let hours = mins / 60;
    let days = hours / 24;
    if days > 0 {
        format!("{days}d ago")
    } else if hours > 0 {
        format!("{hours}h ago")
    } else if mins > 0 {
        format!("{mins}m ago")
    } else {
        "just now".to_string()
    }
}

/// Render secrets table content.
fn render_secrets_table(vault: &Vault, pattern: &str, page: usize) -> Markup {
    let offset = (page - 1) * PAGE_SIZE;

    match vault.list_paginated(VAULT_REQUESTER, pattern, offset, PAGE_SIZE) {
        Ok(paged) => {
            let total_pages = if paged.total == 0 {
                1
            } else {
                paged.total.div_ceil(PAGE_SIZE)
            };

            let summaries = vault
                .list_with_metadata(VAULT_REQUESTER, pattern)
                .unwrap_or_default();

            let base_url = if pattern == "*" {
                "/vault".to_string()
            } else {
                format!("/vault?pattern={pattern}")
            };

            html! {
                (m_header("VAULT", Some("Secret management and access control")))

                form method="get" action="/vault" class="mb-6" {
                    div class="flex gap-2" {
                        input
                            type="text"
                            name="pattern"
                            placeholder="Search secrets (glob pattern)..."
                            value=(if pattern == "*" { "" } else { pattern })
                            class="flex-1 bg-neutral-800 border border-neutral-700 rounded px-3 py-2 text-sm text-white placeholder-neutral-500 focus:border-neutral-500 focus:outline-none";
                        button type="submit" class="m-btn" { "SEARCH" }
                    }
                }

                div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6" {
                    (m_stat("TOTAL SECRETS", &paged.total.to_string(), "in vault", "vault"))
                    (m_stat(
                        "CURRENT PAGE",
                        &format!("{page} / {total_pages}"),
                        &format!("{} shown", paged.secrets.len()),
                        "vault",
                    ))
                }

                @if summaries.is_empty() {
                    (m_empty("No secrets found", "Store a secret to get started"))
                } @else {
                    (m_card("SECRETS", html! {
                        table class="m-table w-full" {
                            (m_table_header(&["KEY", "VERSIONS", "CREATED", "LAST ACCESSED", "ENTITIES", ""]))
                            tbody {
                                @for summary in &summaries {
                                    @let in_page = paged.secrets.contains(&summary.key);
                                    @if in_page {
                                        tr {
                                            td {
                                                a href=(format!("/vault/{}", summary.key))
                                                    class="text-white hover:text-neutral-300 flex items-center gap-2" {
                                                    span class="w-4 h-4 opacity-60" { (icon_key()) }
                                                    (summary.key)
                                                }
                                            }
                                            td { (m_badge(&format!("v{}", summary.version_count))) }
                                            td class="text-neutral-400 text-sm" {
                                                (format_timestamp(summary.created_at))
                                            }
                                            td class="text-neutral-400 text-sm" {
                                                (summary.last_accessed.map_or_else(
                                                    || "never".to_string(),
                                                    format_timestamp,
                                                ))
                                            }
                                            td class="text-neutral-400 text-sm" {
                                                (summary.entity_count)
                                            }
                                            td {
                                                a href=(format!("/vault/{}", summary.key))
                                                    class="opacity-40 hover:opacity-100" {
                                                    span class="w-4 h-4" { (icon_chevron_right()) }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }))

                    (m_pagination(page, total_pages, &base_url))
                }
            }
        },
        Err(e) => html! {
            (m_header("VAULT", Some("Secret management and access control")))
            (m_empty("Error loading secrets", &e.to_string()))
        },
    }
}

/// Secrets list view -- browse all secrets with pagination.
pub async fn secrets_list(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<SecretsListParams>,
) -> Markup {
    let page = params.page.unwrap_or(1).max(1);
    let pattern = params.pattern.as_deref().unwrap_or("*");

    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_header("VAULT", Some("Secret management and access control")))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| render_secrets_table(vault, pattern, page),
    );

    layout("Vault", NavItem::Vault, content)
}

/// Render the detail content for a single secret.
fn render_secret_detail(vault: &Vault, key: &str) -> Markup {
    let versions = vault.list_versions(VAULT_REQUESTER, key);
    let current_ver = vault.current_version(VAULT_REQUESTER, key);
    let expiration = vault.get_expiration(VAULT_REQUESTER, key).ok().flatten();
    let rotation_policy = vault.get_rotation_policy(VAULT_REQUESTER, key);
    let changelog = vault.changelog(VAULT_REQUESTER, key).unwrap_or_default();

    match versions {
        Ok(versions) => {
            let ver_num = current_ver.unwrap_or(0);
            let exp_str = expiration.map_or_else(|| "none".to_string(), format_timestamp);
            let rot_str = if rotation_policy.is_some() {
                "active"
            } else {
                "none"
            };

            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", key)]))
                (m_header(key, Some("Secret detail and version history")))

                div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6" {
                    (m_stat("CURRENT VERSION", &format!("v{ver_num}"), "active", "vault"))
                    (m_stat("TOTAL VERSIONS", &versions.len().to_string(), "stored", "vault"))
                    (m_stat("EXPIRATION", &exp_str, "ttl status", "vault"))
                    (m_stat("ROTATION", rot_str, "policy", "vault"))
                }

                (m_card("SECRET VALUE", html! {
                    div class="flex items-center gap-4" {
                        code id="secret-value" class="flex-1 bg-neutral-900 px-4 py-3 rounded text-sm font-mono text-neutral-500" {
                            "**********"
                        }
                        button
                            class="m-btn"
                            hx-post=(format!("/vault/reveal?key={key}"))
                            hx-target="#secret-value"
                            hx-swap="innerHTML" { "REVEAL" }
                    }
                }))

                (render_version_table(&versions))

                @if !changelog.is_empty() {
                    (render_changelog(&changelog))
                }
            }
        },
        Err(e) => html! {
            (m_breadcrumb(&[("/vault", "VAULT"), ("", key)]))
            (m_header(key, Some("Secret detail")))
            (m_empty("Secret not found", &e.to_string()))
        },
    }
}

/// Render version history table.
fn render_version_table(versions: &[tensor_vault::VersionInfo]) -> Markup {
    m_card(
        "VERSION HISTORY",
        if versions.is_empty() {
            m_empty("No versions", "Secret has no version history")
        } else {
            html! {
                table class="m-table w-full" {
                    (m_table_header(&["VERSION", "CREATED"]))
                    tbody {
                        @for v in versions {
                            tr {
                                td { (m_badge(&format!("v{}", v.version))) }
                                td class="text-neutral-400 text-sm" {
                                    (format_timestamp(v.created_at))
                                }
                            }
                        }
                    }
                }
            }
        },
    )
}

/// Render changelog table.
fn render_changelog(changelog: &[tensor_vault::ChangelogEntry]) -> Markup {
    m_card(
        "CHANGELOG",
        html! {
            table class="m-table w-full" {
                (m_table_header(&["OPERATION", "ENTITY", "VERSION", "TIMESTAMP"]))
                tbody {
                    @for entry in changelog {
                        tr {
                            td { (m_badge(&entry.operation.to_uppercase())) }
                            td class="text-neutral-400 text-sm" { (entry.entity) }
                            td class="text-neutral-400 text-sm" {
                                @if let Some(v) = entry.version {
                                    (format!("v{v}"))
                                } @else {
                                    "-"
                                }
                            }
                            td class="text-neutral-400 text-sm" {
                                (format_timestamp(entry.timestamp))
                            }
                        }
                    }
                }
            }
        },
    )
}

/// Secret detail view -- show metadata, versions, and access info.
pub async fn secret_detail(
    State(ctx): State<Arc<AdminContext>>,
    Path(key): Path<String>,
) -> Markup {
    // Wildcard paths include a leading '/' -- strip it.
    let key = key.strip_prefix('/').unwrap_or(&key);
    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", key)]))
                (m_header(key, Some("Secret detail")))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| render_secret_detail(vault, key),
    );

    layout("Secret Detail", NavItem::Vault, content)
}

/// Query parameters for reveal endpoint.
#[derive(Debug, serde::Deserialize)]
pub struct RevealParams {
    /// Secret key to reveal.
    key: String,
}

/// Reveal secret value -- returns HTML fragment for HTMX swap.
pub async fn reveal_value(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<RevealParams>,
) -> Markup {
    ctx.vault.as_ref().map_or_else(
        || {
            html! {
                span class="text-neutral-500 text-sm" { "Vault not configured" }
            }
        },
        |vault| match vault.get(VAULT_REQUESTER, &params.key) {
            Ok(value) => html! {
                span class="text-white font-mono" { (value) }
                script { "setTimeout(function(){document.getElementById('secret-value').innerHTML='**********';},30000);" }
            },
            Err(e) => html! {
                span class="text-red-400 text-sm" { "Error: " (e) }
            },
        },
    )
}

/// Build the base URL for audit log pagination.
fn build_audit_base_url(entity: Option<&str>, key: Option<&str>) -> String {
    let mut url = "/vault/audit".to_string();
    let has_param = entity.is_some_and(|entity| {
        let _ = write!(url, "?entity={entity}");
        true
    });
    if let Some(key) = key {
        let sep = if has_param { '&' } else { '?' };
        let _ = write!(url, "{sep}key={key}");
    }
    url
}

/// Render audit log content.
fn render_audit_content(vault: &Vault, params: &AuditParams, page: usize) -> Markup {
    let entries = params.key.as_ref().map_or_else(
        || {
            params.entity.as_ref().map_or_else(
                || vault.audit_recent(500).unwrap_or_default(),
                |entity| vault.audit_by_entity(entity).unwrap_or_default(),
            )
        },
        |key| vault.audit_log(key).unwrap_or_default(),
    );

    let total_entries = entries.len();
    let total_pages = if total_entries == 0 {
        1
    } else {
        total_entries.div_ceil(PAGE_SIZE)
    };
    let start = (page - 1) * PAGE_SIZE;
    let page_entries: Vec<_> = entries.iter().skip(start).take(PAGE_SIZE).collect();
    let base_url = build_audit_base_url(params.entity.as_deref(), params.key.as_deref());

    html! {
        (m_breadcrumb(&[("/vault", "VAULT"), ("", "AUDIT LOG")]))
        (m_header("AUDIT LOG", Some("Recent vault operations and access history")))

        form method="get" action="/vault/audit" class="mb-6" {
            div class="flex gap-2" {
                input
                    type="text"
                    name="entity"
                    placeholder="Filter by entity..."
                    value=(params.entity.as_deref().unwrap_or(""))
                    class="flex-1 bg-neutral-800 border border-neutral-700 rounded px-3 py-2 text-sm text-white placeholder-neutral-500 focus:border-neutral-500 focus:outline-none";
                input
                    type="text"
                    name="key"
                    placeholder="Filter by key..."
                    value=(params.key.as_deref().unwrap_or(""))
                    class="flex-1 bg-neutral-800 border border-neutral-700 rounded px-3 py-2 text-sm text-white placeholder-neutral-500 focus:border-neutral-500 focus:outline-none";
                button type="submit" class="m-btn" { "FILTER" }
            }
        }

        (m_stat("TOTAL ENTRIES", &total_entries.to_string(), "audit records", "vault"))

        @if page_entries.is_empty() {
            (m_empty("No audit entries", "No operations have been recorded yet"))
        } @else {
            (m_card("OPERATIONS", html! {
                table class="m-table w-full" {
                    (m_table_header(&["TIMESTAMP", "ENTITY", "OPERATION", "KEY"]))
                    tbody {
                        @for entry in &page_entries {
                            tr {
                                td class="text-neutral-400 text-sm font-mono" {
                                    (format_timestamp(entry.timestamp))
                                }
                                td class="text-neutral-300 text-sm" { (&entry.entity) }
                                td { (m_badge(&format!("{:?}", entry.operation))) }
                                td class="text-neutral-300 text-sm font-mono" { (&entry.secret_key) }
                            }
                        }
                    }
                }
            }))

            (m_pagination(page, total_pages, &base_url))
        }
    }
}

/// Audit log view -- recent vault operations.
pub async fn audit_log(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<AuditParams>,
) -> Markup {
    let page = params.page.unwrap_or(1).max(1);

    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "AUDIT LOG")]))
                (m_header("AUDIT LOG", Some("Recent vault operations")))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| render_audit_content(vault, &params, page),
    );

    layout("Audit Log", NavItem::Vault, content)
}

/// Render vault status content.
fn render_vault_status(vault: &Vault) -> Markup {
    let status = vault.vault_status();
    let policies = vault.list_rotation_policies();
    let snapshots = vault.list_snapshots();

    html! {
        (m_breadcrumb(&[("/vault", "VAULT"), ("", "STATUS")]))
        (m_header("VAULT STATUS", Some("System health, seal status, and configuration")))

        div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6" {
            (m_stat(
                "SEAL STATUS",
                if status.sealed { "SEALED" } else { "UNSEALED" },
                if status.sealed { "vault locked" } else { "operational" },
                "vault",
            ))
            (m_stat("TOTAL SECRETS", &status.total_secrets.to_string(), "stored", "vault"))
            (m_stat("PENDING ROTATIONS", &status.pending_rotations.to_string(), "awaiting action", "vault"))
            (m_stat("SNAPSHOTS", &status.snapshot_count.to_string(), "backups available", "vault"))
        }

        @if !status.sync_health.is_empty() {
            (m_card("SYNC HEALTH", html! {
                table class="m-table w-full" {
                    (m_table_header(&["TARGET", "STATUS"]))
                    tbody {
                        @for (name, reachable) in &status.sync_health {
                            tr {
                                td class="text-neutral-300" { (name) }
                                td {
                                    @let (dot_cls, label) = if *reachable {
                                        ("bg-green-500", "REACHABLE")
                                    } else {
                                        ("bg-red-500", "UNREACHABLE")
                                    };
                                    span class="inline-flex items-center gap-1" {
                                        span class=(format!("w-2 h-2 rounded-full {dot_cls}")) {}
                                        span class="text-neutral-400 text-sm" { (label) }
                                    }
                                }
                            }
                        }
                    }
                }
            }))
        }

        @if !policies.is_empty() {
            (m_card("ROTATION POLICIES", html! {
                table class="m-table w-full" {
                    (m_table_header(&["SECRET KEY", "INTERVAL", "GENERATOR"]))
                    tbody {
                        @for policy in &policies {
                            tr {
                                td class="text-neutral-300 font-mono text-sm" { (&policy.secret_key) }
                                td class="text-neutral-400 text-sm" {
                                    (format!("{}s", policy.interval_ms / 1000))
                                }
                                td { (m_badge(&format!("{:?}", policy.generator))) }
                            }
                        }
                    }
                }
            }))
        }

        @if !snapshots.is_empty() {
            (m_card("SNAPSHOTS", html! {
                table class="m-table w-full" {
                    (m_table_header(&["LABEL", "ID", "CREATED"]))
                    tbody {
                        @for snap in &snapshots {
                            tr {
                                td class="text-neutral-300" { (&snap.label) }
                                td class="text-neutral-400 font-mono text-sm" {
                                    (&snap.id[..8.min(snap.id.len())])
                                }
                                td class="text-neutral-400 text-sm" {
                                    (format_timestamp(snap.created_at_ms))
                                }
                            }
                        }
                    }
                }
            }))
        }
    }
}

/// Vault status view -- seal status, health, quotas, rotation policies.
pub async fn vault_status(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "STATUS")]))
                (m_header("VAULT STATUS", Some("System health and configuration")))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| render_vault_status(vault),
    );

    layout("Vault Status", NavItem::Vault, content)
}

/// Security audit dashboard showing graph topology analysis.
pub async fn security_dashboard(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "SECURITY")]))
                (m_header("SECURITY AUDIT", None))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| {
            let report = vault.security_audit();
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "SECURITY")]))
                (m_header("SECURITY AUDIT", Some("Graph topology analysis")))

                div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6 stagger-container" {
                    (m_stat("ENTITIES", &report.total_entities.to_string(), "in graph", "vault"))
                    (m_stat("SECRETS", &report.total_secrets.to_string(), "stored", "vault"))
                    (m_stat("EDGES", &report.total_edges.to_string(), "access paths", "vault"))
                    (m_stat("CYCLES", &report.cycles.len().to_string(), "detected", "vault"))
                    (m_stat("SPOFs", &report.single_points_of_failure.len().to_string(), "single points of failure", "vault"))
                    (m_stat("OVER-PRIVILEGED", &report.over_privileged.len().to_string(), "entities", "vault"))
                }

                @if !report.single_points_of_failure.is_empty() {
                    (m_card("SINGLE POINTS OF FAILURE", html! {
                        ul class="space-y-2" {
                            @for spof in &report.single_points_of_failure {
                                li class="flex justify-between" {
                                    span class="text-white font-mono" { (&spof.entity) }
                                    span class="text-neutral-400 text-xs" {
                                        (spof.secrets_affected) " secrets at risk"
                                    }
                                }
                            }
                        }
                    }))
                }

                div class="mt-4 flex gap-4" {
                    a href="/vault/critical" class="m-btn inline-block" { "CRITICAL ENTITIES" }
                    a href="/vault/blast-radius" class="m-btn inline-block" { "BLAST RADIUS" }
                }
            }
        },
    );

    layout("Security Audit", NavItem::Vault, content)
}

/// Query parameters for blast radius lookup.
#[derive(Debug, serde::Deserialize)]
pub struct BlastRadiusParams {
    /// Entity key to analyze.
    entity: Option<String>,
}

/// Blast radius analysis for a given entity.
pub async fn blast_radius_view(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<BlastRadiusParams>,
) -> Markup {
    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "BLAST RADIUS")]))
                (m_header("BLAST RADIUS", None))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "BLAST RADIUS")]))
                (m_header("BLAST RADIUS ANALYSIS", Some("Reachable secrets from an entity")))

                // Input form
                div class="m-card mb-6" {
                    div class="m-card-header" { "ANALYZE ENTITY" }
                    div class="m-card-content" {
                        form action="/vault/blast-radius" method="get" class="flex gap-4" {
                            input
                                type="text"
                                name="entity"
                                placeholder="Enter entity key..."
                                value=(params.entity.as_deref().unwrap_or(""))
                                class="flex-1 bg-neutral-800 text-white px-3 py-2 rounded font-mono text-sm border border-neutral-700 focus:border-white focus:outline-none";
                            button type="submit" class="m-btn" { "ANALYZE" }
                        }
                    }
                }

                @if let Some(ref entity) = params.entity {
                    @let result = vault.blast_radius(entity);
                    div class="grid grid-cols-2 gap-4 mb-6" {
                        (m_stat("ENTITY", entity, "analyzed", "vault"))
                        (m_stat("REACHABLE", &result.total_secrets.to_string(), "secrets", "vault"))
                    }

                    @if result.secrets.is_empty() {
                        (m_empty("No Reachable Secrets", "This entity cannot reach any secrets"))
                    } @else {
                        (m_card("REACHABLE SECRETS", html! {
                            ul class="space-y-2" {
                                @for secret in &result.secrets {
                                    li class="flex justify-between" {
                                        span class="text-white font-mono" { (&secret.secret_name) }
                                        span class="text-neutral-400 text-xs" {
                                            (format!("{:?}", secret.permission))
                                            " (" (secret.hop_count) " hops)"
                                        }
                                    }
                                }
                            }
                        }))
                    }
                }
            }
        },
    );

    layout("Blast Radius", NavItem::Vault, content)
}

/// Critical entities list.
pub async fn critical_entities_view(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let content = ctx.vault.as_ref().map_or_else(
        || {
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "CRITICAL")]))
                (m_header("CRITICAL ENTITIES", None))
                (m_empty("Vault not configured", "No vault engine is attached to this server"))
            }
        },
        |vault| {
            let entities = vault.find_critical_entities();
            html! {
                (m_breadcrumb(&[("/vault", "VAULT"), ("", "CRITICAL")]))
                (m_header("CRITICAL ENTITIES", Some(&format!("{} identified", entities.len()))))

                @if entities.is_empty() {
                    (m_empty("No Critical Entities", "No entities with disproportionate access detected"))
                } @else {
                    div class="m-card" {
                        div class="m-card-content overflow-x-auto" {
                            table class="m-table w-full" {
                                (m_table_header(&["ENTITY", "SPOF", "DEPENDENT", "REACHABLE", "PAGERANK"]))
                                tbody {
                                    @for e in &entities {
                                        tr {
                                            td class="text-white font-mono" { (&e.entity) }
                                            td class="text-neutral-400" {
                                                @if e.is_single_point_of_failure { "YES" } @else { "NO" }
                                            }
                                            td class="text-neutral-400 font-mono" {
                                                (e.secrets_solely_dependent)
                                            }
                                            td class="text-neutral-400 font-mono" {
                                                (e.total_reachable_secrets)
                                            }
                                            td class="text-neutral-400 font-mono" {
                                                (format!("{:.3}", e.pagerank_score))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    );

    layout("Critical Entities", NavItem::Vault, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use axum::extract::{Path, Query, State};
    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;
    use tensor_store::TensorStore;
    use tensor_vault::{Vault, VaultConfig};
    use vector_engine::VectorEngine;

    fn create_test_context() -> Arc<AdminContext> {
        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph: Arc::new(GraphEngine::new()),
            unified: None,
            vault: None,
            cache: None,
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
        })
    }

    fn create_vault_context() -> Arc<AdminContext> {
        let graph = Arc::new(GraphEngine::new());
        let store = TensorStore::new();
        let config = VaultConfig {
            argon2_memory_cost: 256,
            argon2_time_cost: 1,
            argon2_parallelism: 1,
            ..VaultConfig::default()
        };
        let vault = Vault::new(b"test-master-key-1234567890", graph.clone(), store, config)
            .expect("vault creation");

        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph,
            unified: None,
            vault: Some(Arc::new(vault)),
            cache: None,
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
        })
    }

    fn create_seeded_vault_context() -> Arc<AdminContext> {
        let graph = Arc::new(GraphEngine::new());
        let store = TensorStore::new();
        let config = VaultConfig {
            argon2_memory_cost: 256,
            argon2_time_cost: 1,
            argon2_parallelism: 1,
            ..VaultConfig::default()
        };
        let vault = Vault::new(b"test-master-key-1234567890", graph.clone(), store, config)
            .expect("vault creation");

        vault.set(VAULT_REQUESTER, "db_password", "s3cret").unwrap();
        vault.set(VAULT_REQUESTER, "api_key", "ak-12345").unwrap();
        vault
            .set(VAULT_REQUESTER, "jwt_secret", "jwt-abc-xyz")
            .unwrap();

        Arc::new(AdminContext {
            relational: Arc::new(RelationalEngine::new()),
            vector: Arc::new(VectorEngine::new()),
            graph,
            unified: None,
            vault: Some(Arc::new(vault)),
            cache: None,
            blob: None,
            checkpoint: None,
            store: None,
            chain: None,
            auth_config: None,
            metrics: None,
            query_router: None,
        })
    }

    #[tokio::test]
    async fn test_secrets_list_no_vault() {
        let ctx = create_test_context();
        let params = SecretsListParams {
            page: None,
            pattern: None,
        };
        let html = secrets_list(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_secrets_list_empty_vault() {
        let ctx = create_vault_context();
        let params = SecretsListParams {
            page: None,
            pattern: None,
        };
        let html = secrets_list(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("VAULT"));
        assert!(html.contains("TOTAL SECRETS"));
    }

    #[tokio::test]
    async fn test_secrets_list_with_secrets() {
        let ctx = create_seeded_vault_context();
        let params = SecretsListParams {
            page: None,
            pattern: None,
        };
        let html = secrets_list(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("db_password"));
        assert!(html.contains("api_key"));
        assert!(html.contains("jwt_secret"));
    }

    #[tokio::test]
    async fn test_secrets_list_with_pattern() {
        let ctx = create_seeded_vault_context();
        let params = SecretsListParams {
            page: None,
            pattern: Some("db_*".to_string()),
        };
        let html = secrets_list(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("VAULT"));
        assert!(html.contains("SEARCH"));
    }

    #[tokio::test]
    async fn test_secrets_list_page_2() {
        let ctx = create_seeded_vault_context();
        let params = SecretsListParams {
            page: Some(2),
            pattern: None,
        };
        let html = secrets_list(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("VAULT"));
    }

    #[tokio::test]
    async fn test_secret_detail_no_vault() {
        let ctx = create_test_context();
        let html = secret_detail(State(ctx), Path("test_key".to_string()))
            .await
            .into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_secret_detail_existing_key() {
        let ctx = create_seeded_vault_context();
        let html = secret_detail(State(ctx), Path("db_password".to_string()))
            .await
            .into_string();
        assert!(html.contains("db_password"));
        assert!(html.contains("CURRENT VERSION"));
        assert!(html.contains("VERSION HISTORY"));
        assert!(html.contains("REVEAL"));
    }

    #[tokio::test]
    async fn test_secret_detail_missing_key() {
        let ctx = create_vault_context();
        let html = secret_detail(State(ctx), Path("nonexistent".to_string()))
            .await
            .into_string();
        assert!(html.contains("not found") || html.contains("Not") || html.contains("error"));
    }

    #[tokio::test]
    async fn test_secret_detail_shows_breadcrumb() {
        let ctx = create_seeded_vault_context();
        let html = secret_detail(State(ctx), Path("api_key".to_string()))
            .await
            .into_string();
        assert!(html.contains("VAULT"));
        assert!(html.contains("api_key"));
    }

    #[tokio::test]
    async fn test_secret_detail_slash_key() {
        // Wildcard paths include a leading '/' -- verify stripping works.
        let ctx = create_seeded_vault_context();
        let html = secret_detail(State(ctx), Path("/db_password".to_string()))
            .await
            .into_string();
        assert!(html.contains("db_password"));
        assert!(html.contains("REVEAL"));
    }

    #[tokio::test]
    async fn test_reveal_value_no_vault() {
        let ctx = create_test_context();
        let params = RevealParams {
            key: "test".to_string(),
        };
        let html = reveal_value(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("not configured"));
    }

    #[tokio::test]
    async fn test_reveal_value_existing() {
        let ctx = create_seeded_vault_context();
        let params = RevealParams {
            key: "db_password".to_string(),
        };
        let html = reveal_value(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("s3cret"));
        assert!(html.contains("setTimeout"));
    }

    #[tokio::test]
    async fn test_reveal_value_missing_key() {
        let ctx = create_vault_context();
        let params = RevealParams {
            key: "nonexistent".to_string(),
        };
        let html = reveal_value(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("Error") || html.contains("error"));
    }

    #[tokio::test]
    async fn test_audit_log_no_vault() {
        let ctx = create_test_context();
        let params = AuditParams {
            entity: None,
            key: None,
            page: None,
        };
        let html = audit_log(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_audit_log_empty() {
        let ctx = create_vault_context();
        let params = AuditParams {
            entity: None,
            key: None,
            page: None,
        };
        let html = audit_log(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("AUDIT LOG"));
    }

    #[tokio::test]
    async fn test_audit_log_with_entries() {
        let ctx = create_seeded_vault_context();
        let params = AuditParams {
            entity: None,
            key: None,
            page: None,
        };
        let html = audit_log(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("AUDIT LOG"));
        assert!(html.contains("OPERATIONS") || html.contains("TOTAL ENTRIES"));
    }

    #[tokio::test]
    async fn test_audit_log_filtered_by_key() {
        let ctx = create_seeded_vault_context();
        let params = AuditParams {
            entity: None,
            key: Some("db_password".to_string()),
            page: None,
        };
        let html = audit_log(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("AUDIT LOG"));
    }

    #[tokio::test]
    async fn test_audit_log_filtered_by_entity() {
        let ctx = create_seeded_vault_context();
        let params = AuditParams {
            entity: Some(VAULT_REQUESTER.to_string()),
            key: None,
            page: None,
        };
        let html = audit_log(State(ctx), Query(params)).await.into_string();
        assert!(html.contains("AUDIT LOG"));
    }

    #[tokio::test]
    async fn test_vault_status_no_vault() {
        let ctx = create_test_context();
        let html = vault_status(State(ctx)).await.into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_vault_status_empty() {
        let ctx = create_vault_context();
        let html = vault_status(State(ctx)).await.into_string();
        assert!(html.contains("VAULT STATUS"));
        assert!(html.contains("SEAL STATUS"));
        assert!(html.contains("UNSEALED"));
        assert!(html.contains("TOTAL SECRETS"));
    }

    #[tokio::test]
    async fn test_vault_status_with_secrets() {
        let ctx = create_seeded_vault_context();
        let html = vault_status(State(ctx)).await.into_string();
        assert!(html.contains("VAULT STATUS"));
        assert!(html.contains("TOTAL SECRETS"));
    }

    #[tokio::test]
    async fn test_vault_status_shows_breadcrumb() {
        let ctx = create_vault_context();
        let html = vault_status(State(ctx)).await.into_string();
        assert!(html.contains("VAULT"));
        assert!(html.contains("STATUS"));
    }

    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(format_timestamp(0), "-");
    }

    #[test]
    fn test_format_timestamp_recent() {
        #[allow(clippy::cast_possible_wrap)]
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let result = format_timestamp(now_ms);
        assert!(result == "just now" || result.contains("m ago"));
    }

    #[test]
    fn test_format_timestamp_old() {
        #[allow(clippy::cast_possible_wrap)]
        let old_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            - 100 * 86400 * 1000;
        let result = format_timestamp(old_ms);
        assert!(result.contains("d ago"));
    }

    #[test]
    fn test_build_audit_base_url_no_params() {
        assert_eq!(build_audit_base_url(None, None), "/vault/audit");
    }

    #[test]
    fn test_build_audit_base_url_entity_only() {
        let url = build_audit_base_url(Some("admin"), None);
        assert!(url.contains("?entity=admin"));
    }

    #[test]
    fn test_build_audit_base_url_both_params() {
        let url = build_audit_base_url(Some("admin"), Some("db_password"));
        assert!(url.contains("?entity=admin"));
        assert!(url.contains("&key=db_password"));
    }

    // Security views tests

    #[tokio::test]
    async fn test_security_dashboard_no_vault() {
        let ctx = create_test_context();
        let result = security_dashboard(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_security_dashboard_with_vault() {
        let ctx = create_vault_context();
        let result = security_dashboard(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("SECURITY AUDIT"));
        assert!(html.contains("ENTITIES"));
        assert!(html.contains("SECRETS"));
        assert!(html.contains("EDGES"));
    }

    #[tokio::test]
    async fn test_blast_radius_no_vault() {
        let ctx = create_test_context();
        let params = BlastRadiusParams { entity: None };
        let result = blast_radius_view(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_blast_radius_form() {
        let ctx = create_vault_context();
        let params = BlastRadiusParams { entity: None };
        let result = blast_radius_view(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("BLAST RADIUS"));
        assert!(html.contains("ANALYZE"));
    }

    #[tokio::test]
    async fn test_blast_radius_with_entity() {
        let ctx = create_vault_context();
        let params = BlastRadiusParams {
            entity: Some("admin".to_string()),
        };
        let result = blast_radius_view(State(ctx), Query(params)).await;
        let html = result.into_string();
        assert!(html.contains("BLAST RADIUS"));
        assert!(html.contains("admin"));
        assert!(html.contains("REACHABLE"));
    }

    #[tokio::test]
    async fn test_critical_entities_no_vault() {
        let ctx = create_test_context();
        let result = critical_entities_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("Vault not configured"));
    }

    #[tokio::test]
    async fn test_critical_entities_with_vault() {
        let ctx = create_vault_context();
        let result = critical_entities_view(State(ctx)).await;
        let html = result.into_string();
        assert!(html.contains("CRITICAL ENTITIES"));
    }
}
