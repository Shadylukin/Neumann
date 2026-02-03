// SPDX-License-Identifier: MIT OR Apache-2.0
//! Handlers for graph engine browsing with dystopian terminal styling.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::Json;
use axum::Form;
use maud::{html, Markup, PreEscaped};
use serde::{Deserialize, Serialize};

use graph_engine::{Node, PageRankConfig, PropertyValue};

use crate::web::templates::layout;
use crate::web::templates::layout::{
    breadcrumb, empty_state, format_number, page_header, stat_card,
};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Query parameters for pagination.
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    /// Page number (zero-indexed).
    #[serde(default)]
    pub page: usize,
    /// Number of items per page.
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    /// Filter by node label.
    #[serde(default)]
    pub label: Option<String>,
    /// Filter by edge type.
    #[serde(default)]
    pub edge_type: Option<String>,
}

const fn default_page_size() -> usize {
    50
}

/// Graph overview page.
pub async fn overview(State(ctx): State<Arc<AdminContext>>) -> Markup {
    let node_count = ctx.graph.node_count();
    let edge_count = ctx.graph.edge_count();

    let content = html! {
        (page_header("GRAPH ENGINE", Some("Navigate nodes and edge relationships")))

        // Stats
        div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6" {
            (stat_card("NODES", &format_number(node_count), "graph entities", "graph"))
            (stat_card("EDGES", &format_number(edge_count), "relationships", "graph"))
        }

        // Quick actions
        div class="terminal-panel mb-6" {
            div class="panel-header" { "OPERATIONS" }
            div class="panel-content" {
                div class="flex flex-wrap gap-2" {
                    a href="/graph/nodes" class="btn-terminal" { "[ BROWSE NODES ]" }
                    a href="/graph/edges" class="btn-terminal" { "[ BROWSE EDGES ]" }
                    a href="/graph/path" class="btn-terminal btn-terminal-amber" { "[ FIND PATH ]" }
                    a href="/graph/algorithms" class="btn-terminal btn-terminal-amber" { "[ ALGORITHMS ]" }
                }
            }
        }

        // Graph Visualization
        @if node_count > 0 {
            div class="mb-6" {
                div class="terminal-panel" {
                    div class="panel-header" { "GRAPH VISUALIZATION" }
                    div class="panel-content p-0" {
                        div id="graph-container" class="w-full h-[400px]" {}
                    }
                    div class="panel-footer" {
                        "Drag to pan, scroll to zoom, click nodes for details"
                    }
                }
            }

            // force-graph script with terminal phosphor theme
            script src="https://unpkg.com/force-graph" {}
            script { (PreEscaped(r#"
                (function initGraph() {
                    // Wait for ForceGraph to load
                    if (typeof ForceGraph === 'undefined') {
                        setTimeout(initGraph, 50);
                        return;
                    }

                    var container = document.getElementById('graph-container');
                    if (!container) {
                        console.error('Graph container not found');
                        return;
                    }

                    fetch('/api/graph/subgraph?limit=100')
                        .then(function(res) { return res.json(); })
                        .then(function(data) {
                            // Terminal phosphor color palette
                            var nodeColors = {
                                'Person': '#00ee00',
                                'Company': '#ffb641',
                                'Location': '#008e00',
                                'Document': '#00ee00',
                                'Section': '#ffb641',
                                'default': '#00ee00'
                            };
                            var linkColors = {
                                'KNOWS': '#005f00',
                                'WORKS_AT': '#ffb000',
                                'IN_SECTION': '#008e00',
                                'LINKS_TO': '#005f00',
                                'RELATED_TO': '#4a2125',
                                'default': '#005f00'
                            };

                            var Graph = ForceGraph()
                                (container)
                                .width(container.clientWidth)
                                .height(container.clientHeight || 400)
                                .graphData(data)
                                .backgroundColor('#0c0c0c')
                                .nodeColor(function(node) { return nodeColors[node.label] || nodeColors.default; })
                                .nodeRelSize(6)
                                .nodeCanvasObject(function(node, ctx, globalScale) {
                                    // Draw node with glow effect
                                    var color = nodeColors[node.label] || nodeColors.default;
                                    var size = 5;

                                    // Outer glow
                                    ctx.shadowColor = color;
                                    ctx.shadowBlur = 15;
                                    ctx.beginPath();
                                    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
                                    ctx.fillStyle = color;
                                    ctx.fill();

                                    // Inner bright core
                                    ctx.shadowBlur = 0;
                                    ctx.beginPath();
                                    ctx.arc(node.x, node.y, size * 0.6, 0, 2 * Math.PI);
                                    ctx.fillStyle = '#ffffff';
                                    ctx.globalAlpha = 0.3;
                                    ctx.fill();
                                    ctx.globalAlpha = 1;

                                    // Label
                                    var label = node.name || node.label;
                                    var fontSize = 10 / globalScale;
                                    ctx.font = fontSize + 'px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillStyle = color;
                                    ctx.shadowColor = color;
                                    ctx.shadowBlur = 5;
                                    ctx.fillText(label, node.x, node.y + size + fontSize);
                                })
                                .linkColor(function(link) { return linkColors[link.type] || linkColors.default; })
                                .linkWidth(1.5)
                                .linkDirectionalArrowLength(4)
                                .linkDirectionalArrowRelPos(1)
                                .linkDirectionalParticles(1)
                                .linkDirectionalParticleWidth(2)
                                .linkDirectionalParticleColor(function(link) { return linkColors[link.type] || linkColors.default; })
                                .onNodeClick(function(node) {
                                    showNodeModal(node);
                                })
                                .cooldownTicks(100)
                                .onEngineStop(function() { Graph.zoomToFit(400, 50); });

                            // Handle window resize
                            window.addEventListener('resize', function() {
                                Graph.width(container.clientWidth).height(container.clientHeight || 400);
                            });
                        })
                        .catch(function(err) {
                            console.error('Failed to load graph data:', err);
                            container.innerHTML = '<div style="color:#942222;padding:20px;text-align:center">ERROR: Failed to load graph data</div>';
                        });
                })();

                function showNodeModal(node) {
                    var modal = document.getElementById('node-modal');
                    var content = document.getElementById('node-modal-content');
                    var name = node.name || '< unnamed >';

                    content.innerHTML =
                        '<div class=\"text-amber-glow text-lg mb-3 font-display tracking-wider\">' +
                            '══[ NODE INSPECTOR ]══' +
                        '</div>' +
                        '<div class=\"space-y-2 font-terminal text-sm\">' +
                            '<div class=\"flex\"><span class=\"text-phosphor-dim w-24\">ID:</span>' +
                            '<span class=\"text-phosphor\">' + node.id + '</span></div>' +
                            '<div class=\"flex\"><span class=\"text-phosphor-dim w-24\">TYPE:</span>' +
                            '<span class=\"text-amber-glow\">' + node.label + '</span></div>' +
                            '<div class=\"flex\"><span class=\"text-phosphor-dim w-24\">NAME:</span>' +
                            '<span class=\"text-phosphor\">' + name + '</span></div>' +
                            '<div class=\"border-t border-phosphor-dark my-3 pt-3\">' +
                                '<div class=\"text-phosphor-dim mb-2\">[ ACTIONS ]</div>' +
                                '<div class=\"flex gap-2 flex-wrap\">' +
                                    '<a href=\"/graph/nodes?label=' + node.label + '\" class=\"btn-terminal text-xs\">[ FILTER BY TYPE ]</a>' +
                                '</div>' +
                            '</div>' +
                        '</div>' +
                        '<div class=\"text-phosphor-dark text-xs mt-4 text-center\">Click outside or press ESC to close</div>';

                    modal.classList.remove('hidden');
                    modal.classList.add('flex');
                }

                function closeNodeModal() {
                    var modal = document.getElementById('node-modal');
                    modal.classList.add('hidden');
                    modal.classList.remove('flex');
                }
            "#)) }

            // Node detail modal
            div id="node-modal" class="hidden fixed inset-0 z-50 items-center justify-center bg-black/80" onclick="if(event.target === this) closeNodeModal()" {
                div class="terminal-panel w-96 max-w-[90vw] animate-slide-in" onclick="event.stopPropagation()" {
                    div class="panel-content" id="node-modal-content" {}
                }
            }
        }

        // Info panels
        div class="grid grid-cols-1 lg:grid-cols-2 gap-6" {
            div class="terminal-panel" {
                div class="panel-header" { "NODES" }
                div class="panel-content" {
                    @if node_count == 0 {
                        p class="text-phosphor-dim italic" { "< NO NODES IN GRAPH >" }
                    } @else {
                        p class="text-phosphor-dim" {
                            "Browse all " span class="text-phosphor" { (format_number(node_count)) } " nodes or filter by label."
                        }
                        a href="/graph/nodes" class="btn-terminal mt-3 inline-block" { "[ VIEW ALL ]" }
                    }
                }
            }

            div class="terminal-panel" {
                div class="panel-header" { "EDGES" }
                div class="panel-content" {
                    @if edge_count == 0 {
                        p class="text-phosphor-dim italic" { "< NO EDGES IN GRAPH >" }
                    } @else {
                        p class="text-phosphor-dim" {
                            "Browse all " span class="text-phosphor" { (format_number(edge_count)) } " edges or filter by type."
                        }
                        a href="/graph/edges" class="btn-terminal mt-3 inline-block" { "[ VIEW ALL ]" }
                    }
                }
            }
        }
    };

    layout("Graph Overview", NavItem::Graph, content)
}

/// Browse nodes with optional label filter.
pub async fn nodes_list(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<PaginationParams>,
) -> Markup {
    let page = params.page;
    let page_size = params.page_size.min(100);
    let offset = page * page_size;

    let (nodes, total): (Vec<Node>, usize) = if let Some(ref label) = params.label {
        match ctx.graph.find_nodes_by_label(label) {
            Ok(nodes) => {
                let total = nodes.len();
                let paginated = nodes.into_iter().skip(offset).take(page_size).collect();
                (paginated, total)
            },
            Err(_) => (Vec::new(), 0),
        }
    } else {
        let nodes = ctx.graph.all_nodes();
        let total = nodes.len();
        let paginated = nodes.into_iter().skip(offset).take(page_size).collect();
        (paginated, total)
    };

    let content = html! {
        (breadcrumb(&[("/graph", "GRAPH"), ("", if params.label.is_some() { "FILTERED NODES" } else { "NODES" })]))

        (page_header(
            if params.label.is_some() { "NODES BY LABEL" } else { "ALL NODES" },
            Some(&format!("{} entities", format_number(total)))
        ))

        // Filter
        div class="terminal-panel mb-4" {
            div class="panel-header" { "FILTER" }
            div class="panel-content" {
                form method="get" action="/graph/nodes" class="flex items-center gap-4" {
                    input
                        type="text"
                        name="label"
                        placeholder="Filter by label..."
                        value=(params.label.clone().unwrap_or_default())
                        class="input-terminal flex-1";
                    button type="submit" class="btn-terminal" { "[ APPLY ]" }
                    @if params.label.is_some() {
                        a href="/graph/nodes" class="btn-terminal btn-terminal-rust" { "[ CLEAR ]" }
                    }
                }
            }
        }

        @if nodes.is_empty() {
            (empty_state("NO NODES", "Create nodes to populate the graph"))
        } @else {
            (render_nodes_table(&nodes))
            (pagination(page, page_size, total, "/graph/nodes", params.label.as_deref()))
        }
    };

    layout("Nodes", NavItem::Graph, content)
}

/// Browse edges with optional type filter.
pub async fn edges_list(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<PaginationParams>,
) -> Markup {
    use graph_engine::Edge;

    let page = params.page;
    let page_size = params.page_size.min(100);
    let offset = page * page_size;

    let (edges, total): (Vec<Edge>, usize) = if let Some(ref edge_type) = params.edge_type {
        match ctx.graph.find_edges_by_type(edge_type) {
            Ok(edges) => {
                let total = edges.len();
                let paginated = edges.into_iter().skip(offset).take(page_size).collect();
                (paginated, total)
            },
            Err(_) => (Vec::new(), 0),
        }
    } else {
        let edges = ctx.graph.all_edges();
        let total = edges.len();
        let paginated = edges.into_iter().skip(offset).take(page_size).collect();
        (paginated, total)
    };

    let content = html! {
        (breadcrumb(&[("/graph", "GRAPH"), ("", if params.edge_type.is_some() { "FILTERED EDGES" } else { "EDGES" })]))

        (page_header(
            if params.edge_type.is_some() { "EDGES BY TYPE" } else { "ALL EDGES" },
            Some(&format!("{} relationships", format_number(total)))
        ))

        // Filter
        div class="terminal-panel mb-4" {
            div class="panel-header" { "FILTER" }
            div class="panel-content" {
                form method="get" action="/graph/edges" class="flex items-center gap-4" {
                    input
                        type="text"
                        name="edge_type"
                        placeholder="Filter by type..."
                        value=(params.edge_type.clone().unwrap_or_default())
                        class="input-terminal flex-1";
                    button type="submit" class="btn-terminal" { "[ APPLY ]" }
                    @if params.edge_type.is_some() {
                        a href="/graph/edges" class="btn-terminal btn-terminal-rust" { "[ CLEAR ]" }
                    }
                }
            }
        }

        @if edges.is_empty() {
            (empty_state("NO EDGES", "Create edges to connect nodes"))
        } @else {
            div class="terminal-panel" {
                div class="panel-header" { "EDGE REGISTRY" }
                div class="panel-content p-0 overflow-x-auto" {
                    table class="table-rust min-w-max" {
                        thead {
                            tr {
                                th { "ID" }
                                th { "FROM" }
                                th { "TYPE" }
                                th { "TO" }
                                th { "DIRECTED" }
                                th { "PROPERTIES" }
                            }
                        }
                        tbody {
                            @for edge in &edges {
                                tr {
                                    td class="text-phosphor-dim font-data" { (edge.id) }
                                    td class="text-phosphor" { (edge.from) }
                                    td {
                                        span class="text-amber" { "[:" (edge.edge_type.clone()) "]" }
                                    }
                                    td class="text-phosphor" { (edge.to) }
                                    td class="text-phosphor-dim" {
                                        @if edge.directed { "YES" } @else { "NO" }
                                    }
                                    td class="text-phosphor-dim" {
                                        (render_properties_summary(&edge.properties))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (pagination(page, page_size, total, "/graph/edges", params.edge_type.as_deref()))
        }
    };

    layout("Edges", NavItem::Graph, content)
}

/// Path finder form parameters.
#[derive(Debug, Deserialize)]
pub struct PathFinderParams {
    /// Source node ID.
    #[serde(default)]
    pub from: String,
    /// Target node ID.
    #[serde(default)]
    pub to: String,
}

/// Path finder form.
pub async fn path_finder(State(_ctx): State<Arc<AdminContext>>) -> Markup {
    render_path_finder_page(None, None)
}

/// Path finder submission.
pub async fn path_finder_submit(
    State(ctx): State<Arc<AdminContext>>,
    Form(params): Form<PathFinderParams>,
) -> Markup {
    let from_id: Result<u64, _> = params.from.trim().parse();
    let to_id: Result<u64, _> = params.to.trim().parse();

    match (from_id, to_id) {
        (Ok(from), Ok(to)) => {
            let path = ctx.graph.find_path(from, to, None);
            render_path_finder_page(Some(&params), Some(path))
        },
        _ => render_path_finder_page(
            Some(&params),
            Some(Err(graph_engine::GraphError::StorageError(
                "Invalid node ID format".to_string(),
            ))),
        ),
    }
}

fn render_path_finder_page(
    params: Option<&PathFinderParams>,
    result: Option<Result<graph_engine::Path, graph_engine::GraphError>>,
) -> Markup {
    let content = html! {
        (breadcrumb(&[("/graph", "GRAPH"), ("", "PATH FINDER")]))

        (page_header("PATH FINDER", Some("Compute shortest path between nodes")))

        // Form
        div class="terminal-panel mb-6" {
            div class="panel-header" { "PATH PARAMETERS" }
            div class="panel-content" {
                form method="post" action="/graph/path" class="space-y-4" {
                    div class="grid grid-cols-1 md:grid-cols-2 gap-4" {
                        div {
                            label for="from" class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                "SOURCE NODE ID"
                            }
                            input
                                type="text"
                                id="from"
                                name="from"
                                value=(params.map_or(String::new(), |p| p.from.clone()))
                                class="input-terminal w-full"
                                placeholder="e.g., 123";
                        }
                        div {
                            label for="to" class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                "TARGET NODE ID"
                            }
                            input
                                type="text"
                                id="to"
                                name="to"
                                value=(params.map_or(String::new(), |p| p.to.clone()))
                                class="input-terminal w-full"
                                placeholder="e.g., 456";
                        }
                    }
                    button type="submit" class="btn-terminal" { "[ COMPUTE PATH ]" }
                }
            }
        }

        // Results
        @if let Some(result) = result {
            @match result {
                Ok(path) => {
                    div class="terminal-panel" {
                        div class="panel-header" { "PATH FOUND" }
                        div class="panel-content" {
                            div class="mb-4 text-phosphor font-terminal" {
                                "[ " (path.nodes.len()) " NODES IN PATH ]"
                            }

                            // Visual path representation
                            div class="overflow-x-auto mb-4" {
                                div class="flex items-center gap-2 min-w-max" {
                                    @for (idx, node_id) in path.nodes.iter().enumerate() {
                                        div class="flex flex-col items-center" {
                                            div class="w-12 h-12 border-2 border-phosphor bg-terminal-soot flex items-center justify-center" {
                                                span class="text-sm font-data text-phosphor" { (node_id) }
                                            }
                                        }
                                        @if idx < path.nodes.len() - 1 {
                                            div class="flex items-center text-phosphor-dim" {
                                                "--->"
                                            }
                                        }
                                    }
                                }
                            }

                            // Path sequence
                            div class="text-sm text-phosphor-dim font-terminal" {
                                "SEQUENCE: "
                                @for (idx, node_id) in path.nodes.iter().enumerate() {
                                    span class="text-phosphor" { (node_id) }
                                    @if idx < path.nodes.len() - 1 {
                                        span { " -> " }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(ref e) if matches!(e, graph_engine::GraphError::PathNotFound) => {
                    (empty_state("NO PATH FOUND", "The nodes are not connected"))
                }
                Err(e) => {
                    div class="terminal-panel terminal-panel-rust" {
                        div class="panel-header" { "ERROR" }
                        div class="panel-content text-amber" {
                            (e.to_string())
                        }
                    }
                }
            }
        }
    };

    layout("Path Finder", NavItem::Graph, content)
}

/// Algorithm form parameters.
#[derive(Debug, Deserialize)]
pub struct AlgorithmParams {
    /// Algorithm name (pagerank, components).
    #[serde(default)]
    pub algorithm: String,
    /// Number of top results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

const fn default_top_k() -> usize {
    10
}

/// Algorithms page.
pub async fn algorithms(State(_ctx): State<Arc<AdminContext>>) -> Markup {
    render_algorithms_page(None, None)
}

/// Algorithm execution.
pub async fn algorithms_submit(
    State(ctx): State<Arc<AdminContext>>,
    Form(params): Form<AlgorithmParams>,
) -> Markup {
    let result = match params.algorithm.as_str() {
        "pagerank" => match ctx.graph.pagerank(Some(PageRankConfig::default())) {
            Ok(pr) => {
                let top = pr.top_k(params.top_k);
                Some(("PageRank", top))
            },
            Err(_) => Some(("PageRank", vec![])),
        },
        "components" => {
            match ctx
                .graph
                .connected_components(Some(graph_engine::CommunityConfig::default()))
            {
                Ok(result) => {
                    let by_size = result.communities_by_size();
                    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for display
                    let top: Vec<(u64, f64)> = by_size
                        .into_iter()
                        .take(params.top_k)
                        .map(|(id, size)| (id, size as f64))
                        .collect();
                    Some(("Connected Components", top))
                },
                Err(_) => Some(("Connected Components", vec![])),
            }
        },
        _ => None,
    };

    render_algorithms_page(Some(&params), result)
}

fn render_algorithms_page(
    params: Option<&AlgorithmParams>,
    result: Option<(&'static str, Vec<(u64, f64)>)>,
) -> Markup {
    let content = html! {
        (breadcrumb(&[("/graph", "GRAPH"), ("", "ALGORITHMS")]))

        (page_header("GRAPH ALGORITHMS", Some("Execute analysis algorithms")))

        // Form
        div class="terminal-panel mb-6" {
            div class="panel-header" { "ALGORITHM PARAMETERS" }
            div class="panel-content" {
                form method="post" action="/graph/algorithms" class="space-y-4" {
                    div class="grid grid-cols-1 md:grid-cols-2 gap-4" {
                        div {
                            label for="algorithm" class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                "ALGORITHM"
                            }
                            select
                                id="algorithm"
                                name="algorithm"
                                class="input-terminal w-full"
                            {
                                option value="pagerank" selected[params.map_or(true, |p| p.algorithm == "pagerank")] { "PageRank" }
                                option value="components" selected[params.map_or(false, |p| p.algorithm == "components")] { "Connected Components" }
                            }
                        }
                        div {
                            label for="top_k" class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                "TOP K RESULTS"
                            }
                            input
                                type="number"
                                id="top_k"
                                name="top_k"
                                min="1"
                                max="100"
                                value=(params.map_or(10, |p| p.top_k))
                                class="input-terminal w-full";
                        }
                    }
                    button type="submit" class="btn-terminal" { "[ EXECUTE ]" }
                }
            }
        }

        // Results
        @if let Some((name, results)) = result {
            div class="terminal-panel" {
                div class="panel-header" { (name.to_uppercase()) " RESULTS" }
                div class="panel-content p-0" {
                    @if results.is_empty() {
                        div class="p-4 text-phosphor-dim italic" { "< NO RESULTS >" }
                    } @else {
                        table class="table-rust" {
                            thead {
                                tr {
                                    th class="w-16" { "#" }
                                    th { "NODE ID" }
                                    th class="text-right w-32" { "SCORE" }
                                }
                            }
                            tbody {
                                @for (idx, (node_id, score)) in results.iter().enumerate() {
                                    tr {
                                        td class="text-phosphor-dim font-data" { (idx + 1) }
                                        td class="text-phosphor font-data" { (node_id) }
                                        td class="text-right text-amber font-data" { (format!("{score:.6}")) }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    layout("Algorithms", NavItem::Graph, content)
}

// API endpoint for graph visualization

/// Subgraph query parameters.
#[derive(Debug, Deserialize)]
pub struct SubgraphParams {
    /// Center node ID for BFS traversal.
    #[serde(default)]
    pub center: Option<u64>,
    /// Traversal depth from center.
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Maximum number of nodes to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

const fn default_depth() -> usize {
    2
}

const fn default_limit() -> usize {
    100
}

/// Graph data for visualization.
#[derive(Debug, Serialize)]
pub struct GraphData {
    /// Nodes in the subgraph.
    pub nodes: Vec<GraphNode>,
    /// Edges in the subgraph.
    pub links: Vec<GraphLink>,
}

/// Node for visualization.
#[derive(Debug, Serialize)]
pub struct GraphNode {
    /// Node identifier.
    pub id: String,
    /// Node label/type.
    pub label: String,
    /// Optional display name.
    pub name: Option<String>,
}

/// Link for visualization.
#[derive(Debug, Serialize)]
pub struct GraphLink {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Edge type/relationship.
    #[serde(rename = "type")]
    pub edge_type: String,
}

/// API endpoint for fetching subgraph data.
pub async fn api_subgraph(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<SubgraphParams>,
) -> Json<GraphData> {
    let mut nodes_map: HashMap<u64, GraphNode> = HashMap::new();
    let mut links = Vec::new();

    if let Some(center_id) = params.center {
        if let Ok(center) = ctx.graph.get_node(center_id) {
            nodes_map.insert(
                center.id,
                GraphNode {
                    id: center.id.to_string(),
                    label: center.labels.first().cloned().unwrap_or_default(),
                    name: get_node_display_name(&center.properties),
                },
            );

            collect_neighbors(
                &ctx.graph,
                center_id,
                params.depth,
                params.limit,
                &mut nodes_map,
                &mut links,
            );
        }
    } else {
        let all_nodes = ctx.graph.all_nodes();
        for node in all_nodes.into_iter().take(params.limit) {
            nodes_map.insert(
                node.id,
                GraphNode {
                    id: node.id.to_string(),
                    label: node.labels.first().cloned().unwrap_or_default(),
                    name: get_node_display_name(&node.properties),
                },
            );
        }

        let node_ids: std::collections::HashSet<u64> = nodes_map.keys().copied().collect();
        let all_edges = ctx.graph.all_edges();
        for edge in all_edges {
            if node_ids.contains(&edge.from) && node_ids.contains(&edge.to) {
                links.push(GraphLink {
                    source: edge.from.to_string(),
                    target: edge.to.to_string(),
                    edge_type: edge.edge_type,
                });
            }
        }
    }

    Json(GraphData {
        nodes: nodes_map.into_values().collect(),
        links,
    })
}

fn collect_neighbors(
    graph: &graph_engine::GraphEngine,
    center: u64,
    max_depth: usize,
    max_nodes: usize,
    nodes_map: &mut HashMap<u64, GraphNode>,
    links: &mut Vec<GraphLink>,
) {
    use std::collections::VecDeque;

    let mut queue = VecDeque::new();
    queue.push_back((center, 0));

    while let Some((node_id, depth)) = queue.pop_front() {
        if depth >= max_depth || nodes_map.len() >= max_nodes {
            continue;
        }

        let edges = graph
            .edges_of(node_id, graph_engine::Direction::Outgoing)
            .unwrap_or_default();
        for edge in edges {
            let neighbor_id = edge.to;

            links.push(GraphLink {
                source: edge.from.to_string(),
                target: edge.to.to_string(),
                edge_type: edge.edge_type,
            });

            if !nodes_map.contains_key(&neighbor_id) {
                if let Ok(neighbor) = graph.get_node(neighbor_id) {
                    nodes_map.insert(
                        neighbor.id,
                        GraphNode {
                            id: neighbor.id.to_string(),
                            label: neighbor.labels.first().cloned().unwrap_or_default(),
                            name: get_node_display_name(&neighbor.properties),
                        },
                    );
                    queue.push_back((neighbor_id, depth + 1));
                }
            }
        }
    }
}

fn get_node_display_name(props: &HashMap<String, PropertyValue>) -> Option<String> {
    for key in ["name", "title", "label", "id"] {
        if let Some(PropertyValue::String(s)) = props.get(key) {
            return Some(s.clone());
        }
    }
    None
}

// Helper functions

fn render_nodes_table(nodes: &[Node]) -> Markup {
    html! {
        div class="terminal-panel" {
            div class="panel-header" { "NODE REGISTRY" }
            div class="panel-content p-0 overflow-x-auto" {
                table class="table-rust min-w-max" {
                    thead {
                        tr {
                            th { "ID" }
                            th { "LABELS" }
                            th { "PROPERTIES" }
                        }
                    }
                    tbody {
                        @for node in nodes {
                            tr {
                                td class="text-phosphor font-data" { (node.id) }
                                td {
                                    div class="flex flex-wrap gap-1" {
                                        @for label in &node.labels {
                                            span class="text-amber" { "[:" (label) "]" }
                                        }
                                        @if node.labels.is_empty() {
                                            span class="text-phosphor-dark italic" { "none" }
                                        }
                                    }
                                }
                                td class="text-phosphor-dim" {
                                    (render_properties_summary(&node.properties))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn render_properties_summary(props: &HashMap<String, PropertyValue>) -> Markup {
    if props.is_empty() {
        return html! { span class="italic" { "none" } };
    }

    let summary: Vec<String> = props
        .iter()
        .take(3)
        .map(|(k, v)| {
            let value_str = match v {
                PropertyValue::Null => "null".to_string(),
                PropertyValue::Int(i) => i.to_string(),
                PropertyValue::Float(f) => format!("{f:.2}"),
                PropertyValue::String(s) => {
                    if s.len() > 20 {
                        format!("\"{}...\"", &s[..20])
                    } else {
                        format!("\"{s}\"")
                    }
                },
                PropertyValue::Bool(b) => b.to_string(),
                _ => "[...]".to_string(),
            };
            format!("{k}: {value_str}")
        })
        .collect();

    let text = summary.join(", ");
    let has_more = props.len() > 3;

    html! {
        span class="font-terminal text-sm" {
            "{" (text) @if has_more { ", ..." } "}"
        }
    }
}

fn pagination(
    page: usize,
    page_size: usize,
    total: usize,
    base_url: &str,
    filter: Option<&str>,
) -> Markup {
    let total_pages = total.div_ceil(page_size);
    let has_prev = page > 0;
    let has_next = page + 1 < total_pages;

    let filter_param = filter.map_or(String::new(), |f| format!("&label={f}"));

    html! {
        div class="mt-4 flex items-center justify-between font-terminal" {
            div class="text-sm text-phosphor-dim" {
                "SHOWING " (page * page_size + 1) " - " (((page + 1) * page_size).min(total)) " OF " (format_number(total))
            }
            div class="flex items-center gap-2" {
                @if has_prev {
                    a href=(format!("{base_url}?page={}&page_size={page_size}{filter_param}", page - 1))
                      class="btn-terminal text-sm" {
                        "[ PREV ]"
                    }
                }
                span class="px-3 py-1 text-sm text-phosphor-dim" {
                    "PAGE " (page + 1) " / " (total_pages.max(1))
                }
                @if has_next {
                    a href=(format!("{base_url}?page={}&page_size={page_size}{filter_param}", page + 1))
                      class="btn-terminal text-sm" {
                        "[ NEXT ]"
                    }
                }
            }
        }
    }
}
