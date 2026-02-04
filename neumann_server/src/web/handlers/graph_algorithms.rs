// SPDX-License-Identifier: MIT OR Apache-2.0
//! Handlers for graph algorithm execution with dystopian terminal styling.
//!
//! Exposes all 17 graph algorithms with configurable parameters and result visualization.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::Form;
use maud::{html, Markup};
use serde::{Deserialize, Serialize};

use graph_engine::{
    AStarConfig, BiconnectedConfig, CentralityConfig, CommunityConfig, Direction, KCoreConfig,
    MstConfig, PageRankConfig, SccConfig, SimilarityConfig, SimilarityMetric, TriangleConfig,
};

use crate::web::templates::layout;
use crate::web::templates::layout::{breadcrumb, empty_state, format_number, page_header};
use crate::web::AdminContext;
use crate::web::NavItem;

/// Algorithm categories for UI organization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmCategory {
    /// Node importance and influence metrics.
    Centrality,
    /// Community and cluster detection.
    Community,
    /// Path finding and traversal algorithms.
    Pathfinding,
    /// Structural analysis algorithms.
    Structure,
    /// Node similarity metrics.
    Similarity,
}

impl AlgorithmCategory {
    const fn label(self) -> &'static str {
        match self {
            Self::Centrality => "CENTRALITY",
            Self::Community => "COMMUNITY",
            Self::Pathfinding => "PATHFINDING",
            Self::Structure => "STRUCTURE",
            Self::Similarity => "SIMILARITY",
        }
    }
}

/// Algorithm definition for UI rendering.
struct AlgorithmDef {
    id: &'static str,
    name: &'static str,
    category: AlgorithmCategory,
    description: &'static str,
    params: &'static [ParamDef],
}

struct ParamDef {
    name: &'static str,
    label: &'static str,
    param_type: ParamType,
    default: &'static str,
    description: &'static str,
}

enum ParamType {
    Float,
    Int,
    NodeId,
    Direction,
    SimilarityMetric,
}

/// All available algorithms with their configurations.
const ALGORITHMS: &[AlgorithmDef] = &[
    // Centrality
    AlgorithmDef {
        id: "pagerank",
        name: "PageRank",
        category: AlgorithmCategory::Centrality,
        description: "Compute node importance based on link structure (iterative random walk)",
        params: &[
            ParamDef {
                name: "damping",
                label: "Damping Factor",
                param_type: ParamType::Float,
                default: "0.85",
                description: "Probability of following links vs random jump (0.0-1.0)",
            },
            ParamDef {
                name: "tolerance",
                label: "Tolerance",
                param_type: ParamType::Float,
                default: "0.000001",
                description: "Convergence threshold",
            },
            ParamDef {
                name: "max_iterations",
                label: "Max Iterations",
                param_type: ParamType::Int,
                default: "100",
                description: "Maximum iteration count",
            },
            ParamDef {
                name: "top_k",
                label: "Top K",
                param_type: ParamType::Int,
                default: "20",
                description: "Number of top results to show",
            },
        ],
    },
    AlgorithmDef {
        id: "betweenness",
        name: "Betweenness Centrality",
        category: AlgorithmCategory::Centrality,
        description: "Measure how often a node lies on shortest paths between other nodes",
        params: &[
            ParamDef {
                name: "top_k",
                label: "Top K",
                param_type: ParamType::Int,
                default: "20",
                description: "Number of top results to show",
            },
            ParamDef {
                name: "direction",
                label: "Direction",
                param_type: ParamType::Direction,
                default: "both",
                description: "Edge direction to follow",
            },
        ],
    },
    AlgorithmDef {
        id: "closeness",
        name: "Closeness Centrality",
        category: AlgorithmCategory::Centrality,
        description: "Measure average shortest path distance to all other nodes",
        params: &[
            ParamDef {
                name: "top_k",
                label: "Top K",
                param_type: ParamType::Int,
                default: "20",
                description: "Number of top results to show",
            },
            ParamDef {
                name: "direction",
                label: "Direction",
                param_type: ParamType::Direction,
                default: "both",
                description: "Edge direction to follow",
            },
        ],
    },
    AlgorithmDef {
        id: "eigenvector",
        name: "Eigenvector Centrality",
        category: AlgorithmCategory::Centrality,
        description: "Measure influence based on connections to other influential nodes",
        params: &[
            ParamDef {
                name: "top_k",
                label: "Top K",
                param_type: ParamType::Int,
                default: "20",
                description: "Number of top results to show",
            },
            ParamDef {
                name: "max_iterations",
                label: "Max Iterations",
                param_type: ParamType::Int,
                default: "100",
                description: "Maximum iteration count",
            },
            ParamDef {
                name: "tolerance",
                label: "Tolerance",
                param_type: ParamType::Float,
                default: "0.000001",
                description: "Convergence threshold",
            },
        ],
    },
    // Community
    AlgorithmDef {
        id: "louvain",
        name: "Louvain Communities",
        category: AlgorithmCategory::Community,
        description: "Detect communities by modularity optimization (hierarchical)",
        params: &[
            ParamDef {
                name: "resolution",
                label: "Resolution",
                param_type: ParamType::Float,
                default: "1.0",
                description: "Controls community size (higher = smaller communities)",
            },
            ParamDef {
                name: "max_passes",
                label: "Max Passes",
                param_type: ParamType::Int,
                default: "10",
                description: "Maximum number of Louvain passes",
            },
        ],
    },
    AlgorithmDef {
        id: "label_propagation",
        name: "Label Propagation",
        category: AlgorithmCategory::Community,
        description: "Fast community detection via label spreading",
        params: &[ParamDef {
            name: "max_iterations",
            label: "Max Iterations",
            param_type: ParamType::Int,
            default: "100",
            description: "Maximum iteration count",
        }],
    },
    AlgorithmDef {
        id: "connected_components",
        name: "Connected Components",
        category: AlgorithmCategory::Community,
        description: "Find groups of interconnected nodes (Union-Find)",
        params: &[ParamDef {
            name: "direction",
            label: "Direction",
            param_type: ParamType::Direction,
            default: "both",
            description: "Edge direction (both = undirected)",
        }],
    },
    // Pathfinding
    AlgorithmDef {
        id: "astar",
        name: "A* Pathfinding",
        category: AlgorithmCategory::Pathfinding,
        description: "Find optimal path using heuristic-guided search",
        params: &[
            ParamDef {
                name: "from",
                label: "Source Node",
                param_type: ParamType::NodeId,
                default: "",
                description: "Starting node ID",
            },
            ParamDef {
                name: "to",
                label: "Target Node",
                param_type: ParamType::NodeId,
                default: "",
                description: "Destination node ID",
            },
            ParamDef {
                name: "max_depth",
                label: "Max Depth",
                param_type: ParamType::Int,
                default: "100",
                description: "Maximum search depth",
            },
        ],
    },
    AlgorithmDef {
        id: "dijkstra",
        name: "Dijkstra Shortest Path",
        category: AlgorithmCategory::Pathfinding,
        description: "Find shortest weighted path between nodes",
        params: &[
            ParamDef {
                name: "from",
                label: "Source Node",
                param_type: ParamType::NodeId,
                default: "",
                description: "Starting node ID",
            },
            ParamDef {
                name: "to",
                label: "Target Node",
                param_type: ParamType::NodeId,
                default: "",
                description: "Destination node ID",
            },
            ParamDef {
                name: "weight_property",
                label: "Weight Property",
                param_type: ParamType::Int,
                default: "",
                description: "Edge property name for weights (empty = uniform)",
            },
        ],
    },
    AlgorithmDef {
        id: "variable_paths",
        name: "Variable-Length Paths",
        category: AlgorithmCategory::Pathfinding,
        description: "Find all paths within hop range between two nodes",
        params: &[
            ParamDef {
                name: "from",
                label: "Source Node",
                param_type: ParamType::NodeId,
                default: "",
                description: "Starting node ID",
            },
            ParamDef {
                name: "to",
                label: "Target Node",
                param_type: ParamType::NodeId,
                default: "",
                description: "Destination node ID",
            },
            ParamDef {
                name: "min_hops",
                label: "Min Hops",
                param_type: ParamType::Int,
                default: "1",
                description: "Minimum path length",
            },
            ParamDef {
                name: "max_hops",
                label: "Max Hops",
                param_type: ParamType::Int,
                default: "3",
                description: "Maximum path length",
            },
            ParamDef {
                name: "max_paths",
                label: "Max Paths",
                param_type: ParamType::Int,
                default: "100",
                description: "Maximum paths to return",
            },
        ],
    },
    // Structure
    AlgorithmDef {
        id: "kcore",
        name: "K-Core Decomposition",
        category: AlgorithmCategory::Structure,
        description: "Find densely connected subgraphs by core number",
        params: &[ParamDef {
            name: "min_k",
            label: "Minimum K",
            param_type: ParamType::Int,
            default: "1",
            description: "Minimum core number to report",
        }],
    },
    AlgorithmDef {
        id: "scc",
        name: "Strongly Connected Components",
        category: AlgorithmCategory::Structure,
        description: "Find maximal strongly connected subgraphs (Tarjan)",
        params: &[],
    },
    AlgorithmDef {
        id: "mst",
        name: "Minimum Spanning Tree",
        category: AlgorithmCategory::Structure,
        description: "Find minimum weight tree connecting all nodes (Kruskal)",
        params: &[ParamDef {
            name: "weight_property",
            label: "Weight Property",
            param_type: ParamType::Int,
            default: "",
            description: "Edge property for weights (empty = uniform)",
        }],
    },
    AlgorithmDef {
        id: "biconnected",
        name: "Biconnected Components",
        category: AlgorithmCategory::Structure,
        description: "Find articulation points and bridges",
        params: &[],
    },
    AlgorithmDef {
        id: "triangles",
        name: "Triangle Counting",
        category: AlgorithmCategory::Structure,
        description: "Count triangles and compute clustering coefficients",
        params: &[ParamDef {
            name: "top_k",
            label: "Top K",
            param_type: ParamType::Int,
            default: "20",
            description: "Number of top nodes by triangle count",
        }],
    },
    // Similarity
    AlgorithmDef {
        id: "similarity",
        name: "Node Similarity",
        category: AlgorithmCategory::Similarity,
        description: "Compare neighborhoods using various metrics",
        params: &[
            ParamDef {
                name: "node_a",
                label: "Node A",
                param_type: ParamType::NodeId,
                default: "",
                description: "First node ID",
            },
            ParamDef {
                name: "node_b",
                label: "Node B",
                param_type: ParamType::NodeId,
                default: "",
                description: "Second node ID",
            },
            ParamDef {
                name: "metric",
                label: "Similarity Metric",
                param_type: ParamType::SimilarityMetric,
                default: "jaccard",
                description: "Metric to compute",
            },
        ],
    },
];

/// Query parameters for algorithm dashboard.
#[derive(Debug, Deserialize)]
pub struct DashboardParams {
    /// Filter by algorithm category.
    #[serde(default)]
    pub category: Option<String>,
}

/// Algorithm execution form parameters.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Deserialize)]
pub struct ExecuteParams {
    /// Algorithm identifier.
    pub algorithm: String,
    /// Number of top results to return.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Maximum iterations for iterative algorithms.
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Convergence tolerance.
    #[serde(default)]
    pub tolerance: Option<f64>,
    /// Edge direction for traversal.
    #[serde(default)]
    pub direction: Option<String>,
    /// PageRank damping factor.
    #[serde(default)]
    pub damping: Option<f64>,
    /// Louvain resolution parameter.
    #[serde(default)]
    pub resolution: Option<f64>,
    /// Maximum Louvain passes.
    #[serde(default)]
    pub max_passes: Option<usize>,
    /// Source node ID for pathfinding.
    #[serde(default)]
    pub from: Option<String>,
    /// Target node ID for pathfinding.
    #[serde(default)]
    pub to: Option<String>,
    /// Maximum search depth.
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// Edge property to use as weight.
    #[serde(default)]
    pub weight_property: Option<String>,
    /// Minimum hops for variable-length paths.
    #[serde(default)]
    pub min_hops: Option<usize>,
    /// Maximum hops for variable-length paths.
    #[serde(default)]
    pub max_hops: Option<usize>,
    /// Maximum paths to return.
    #[serde(default)]
    pub max_paths: Option<usize>,
    /// Minimum k-core value.
    #[serde(default)]
    pub min_k: Option<usize>,
    /// First node for similarity comparison.
    #[serde(default)]
    pub node_a: Option<String>,
    /// Second node for similarity comparison.
    #[serde(default)]
    pub node_b: Option<String>,
    /// Similarity metric to use.
    #[serde(default)]
    pub metric: Option<String>,
}

/// Algorithm execution result for display.
#[derive(Debug, Serialize)]
pub struct AlgorithmResult {
    /// Algorithm identifier.
    pub algorithm: String,
    /// Execution status.
    pub status: ResultStatus,
    /// Execution time in milliseconds.
    pub elapsed_ms: u64,
    /// Result data variant.
    pub data: ResultData,
}

/// Execution result status.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultStatus {
    /// Algorithm completed successfully.
    Success,
    /// Algorithm execution failed.
    Error,
    /// No data returned from algorithm.
    NoData,
}

/// Result data variants for different algorithm types.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResultData {
    /// Node scores (centrality, triangle counts).
    Scores(Vec<(u64, f64)>),
    /// Community detection results.
    Communities(CommunityData),
    /// Path finding results.
    Path(PathData),
    /// Structural analysis results.
    Structure(StructureData),
    /// Similarity score.
    Similarity(f64),
    /// Error message.
    Error(String),
    /// Empty result.
    Empty,
}

/// Community detection result data.
#[derive(Debug, Serialize)]
pub struct CommunityData {
    /// Number of communities found.
    pub count: usize,
    /// Modularity score (if applicable).
    pub modularity: Option<f64>,
    /// Community IDs with their sizes.
    pub communities: Vec<(u64, usize)>,
}

/// Pathfinding result data.
#[derive(Debug, Serialize)]
pub struct PathData {
    /// Nodes in the path.
    pub nodes: Vec<u64>,
    /// Total path weight (if weighted).
    pub weight: Option<f64>,
    /// Whether a path was found.
    pub found: bool,
}

/// Structural analysis result data.
#[derive(Debug, Serialize)]
pub struct StructureData {
    /// Summary statistics.
    pub summary: HashMap<String, String>,
    /// Detailed items.
    pub items: Vec<(String, String)>,
}

/// Main algorithms dashboard page.
pub async fn dashboard(
    State(ctx): State<Arc<AdminContext>>,
    Query(params): Query<DashboardParams>,
) -> Markup {
    let node_count = ctx.graph.node_count();
    let edge_count = ctx.graph.edge_count();

    let selected_category = params.category.as_deref();

    let content = html! {
        (breadcrumb(&[("/graph", "GRAPH"), ("", "ALGORITHMS")]))

        (page_header("ALGORITHM DASHBOARD", Some("Execute graph analysis algorithms")))

        // Stats
        div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6" {
            div class="terminal-panel" {
                div class="panel-content text-center" {
                    div class="text-3xl font-data text-phosphor" { (format_number(node_count)) }
                    div class="text-xs text-phosphor-dim font-terminal" { "NODES" }
                }
            }
            div class="terminal-panel" {
                div class="panel-content text-center" {
                    div class="text-3xl font-data text-phosphor" { (format_number(edge_count)) }
                    div class="text-xs text-phosphor-dim font-terminal" { "EDGES" }
                }
            }
            div class="terminal-panel" {
                div class="panel-content text-center" {
                    div class="text-3xl font-data text-amber" { (ALGORITHMS.len()) }
                    div class="text-xs text-phosphor-dim font-terminal" { "ALGORITHMS" }
                }
            }
        }

        // Category tabs
        div class="terminal-panel mb-6" {
            div class="panel-header" { "ALGORITHM CATEGORIES" }
            div class="panel-content" {
                div class="flex flex-wrap gap-2" {
                    a href="/graph/algorithms/dashboard"
                      class=(if selected_category.is_none() { "btn-terminal btn-terminal-amber" } else { "btn-terminal" }) {
                        "[ ALL ]"
                    }
                    @for cat in [AlgorithmCategory::Centrality, AlgorithmCategory::Community, AlgorithmCategory::Pathfinding, AlgorithmCategory::Structure, AlgorithmCategory::Similarity] {
                        a href=(format!("/graph/algorithms/dashboard?category={}", cat.label().to_lowercase()))
                          class=(if selected_category.is_some_and(|c| c.eq_ignore_ascii_case(cat.label())) { "btn-terminal btn-terminal-amber" } else { "btn-terminal" }) {
                            "[ " (cat.label()) " ]"
                        }
                    }
                }
            }
        }

        // Algorithm grid
        @if node_count == 0 {
            (empty_state("NO GRAPH DATA", "Create nodes and edges to run algorithms"))
        } @else {
            div class="grid grid-cols-1 lg:grid-cols-2 gap-4" {
                @for algo in ALGORITHMS {
                    @if selected_category.is_none() || selected_category.is_some_and(|c| c.eq_ignore_ascii_case(algo.category.label())) {
                        (render_algorithm_card(algo))
                    }
                }
            }
        }
    };

    layout::layout("Algorithm Dashboard", NavItem::Graph, content)
}

/// Execute algorithm page with form.
pub async fn execute_form(
    State(_ctx): State<Arc<AdminContext>>,
    Query(params): Query<ExecuteParams>,
) -> Markup {
    let algo = ALGORITHMS.iter().find(|a| a.id == params.algorithm);

    let content = if let Some(algo) = algo {
        html! {
            (breadcrumb(&[("/graph", "GRAPH"), ("/graph/algorithms/dashboard", "ALGORITHMS"), ("", algo.name)]))

            (page_header(algo.name, Some(algo.description)))

            // Parameter form
            div class="terminal-panel mb-6" {
                div class="panel-header" { "PARAMETERS" }
                div class="panel-content" {
                    form method="post" action="/graph/algorithms/execute" class="space-y-4" {
                        input type="hidden" name="algorithm" value=(algo.id);

                        div class="grid grid-cols-1 md:grid-cols-2 gap-4" {
                            @for param in algo.params {
                                div {
                                    label for=(param.name) class="block text-sm text-phosphor-dim mb-2 font-terminal" {
                                        (param.label)
                                    }
                                    @match param.param_type {
                                        ParamType::Float | ParamType::Int | ParamType::NodeId => {
                                            input
                                                type="text"
                                                id=(param.name)
                                                name=(param.name)
                                                value=(param.default)
                                                placeholder=(param.description)
                                                class="input-terminal w-full";
                                        }
                                        ParamType::Direction => {
                                            select id=(param.name) name=(param.name) class="input-terminal w-full" {
                                                option value="both" selected[param.default == "both"] { "Both (Undirected)" }
                                                option value="outgoing" selected[param.default == "outgoing"] { "Outgoing" }
                                                option value="incoming" selected[param.default == "incoming"] { "Incoming" }
                                            }
                                        }
                                        ParamType::SimilarityMetric => {
                                            select id=(param.name) name=(param.name) class="input-terminal w-full" {
                                                option value="jaccard" selected[param.default == "jaccard"] { "Jaccard" }
                                                option value="cosine" { "Cosine" }
                                                option value="adamic_adar" { "Adamic-Adar" }
                                                option value="resource_allocation" { "Resource Allocation" }
                                                option value="preferential_attachment" { "Preferential Attachment" }
                                                option value="common_neighbors" { "Common Neighbors" }
                                            }
                                        }
                                    }
                                    p class="text-xs text-phosphor-dark mt-1" { (param.description) }
                                }
                            }
                        }

                        button type="submit" class="btn-terminal btn-terminal-amber" { "[ EXECUTE ALGORITHM ]" }
                    }
                }
            }

            // Info panel
            div class="terminal-panel" {
                div class="panel-header" { "ALGORITHM INFO" }
                div class="panel-content text-sm font-terminal" {
                    div class="flex gap-2 mb-2" {
                        span class="text-phosphor-dim" { "CATEGORY:" }
                        span class="text-amber" { (algo.category.label()) }
                    }
                    p class="text-phosphor-dim" { (algo.description) }
                }
            }
        }
    } else {
        html! {
            (breadcrumb(&[("/graph", "GRAPH"), ("/graph/algorithms/dashboard", "ALGORITHMS"), ("", "NOT FOUND")]))
            (empty_state("ALGORITHM NOT FOUND", "The requested algorithm does not exist"))
        }
    };

    layout::layout("Execute Algorithm", NavItem::Graph, content)
}

/// Execute algorithm and show results.
pub async fn execute_submit(
    State(ctx): State<Arc<AdminContext>>,
    Form(params): Form<ExecuteParams>,
) -> Markup {
    let start = std::time::Instant::now();
    let result = execute_algorithm(&ctx, &params);
    #[allow(clippy::cast_possible_truncation)] // Execution time won't exceed u64::MAX milliseconds
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let algo = ALGORITHMS.iter().find(|a| a.id == params.algorithm);
    let algo_name = algo.map_or("Unknown", |a| a.name);

    let status_text = match result.status {
        ResultStatus::Success => ("text-phosphor", "SUCCESS"),
        ResultStatus::Error => ("text-rust-blood", "ERROR"),
        ResultStatus::NoData => ("text-amber", "NO DATA"),
    };

    let content = html! {
        (breadcrumb(&[("/graph", "GRAPH"), ("/graph/algorithms/dashboard", "ALGORITHMS"), ("", algo_name)]))

        (page_header(&format!("{} RESULTS", algo_name.to_uppercase()), None))

        // Execution stats
        div class="terminal-panel mb-6" {
            div class="panel-header" { "EXECUTION STATS" }
            div class="panel-content" {
                div class="grid grid-cols-2 md:grid-cols-4 gap-4 font-terminal text-sm" {
                    div {
                        span class="text-phosphor-dim" { "ALGORITHM: " }
                        span class="text-phosphor" { (algo_name) }
                    }
                    div {
                        span class="text-phosphor-dim" { "STATUS: " }
                        span class=(status_text.0) { (status_text.1) }
                    }
                    div {
                        span class="text-phosphor-dim" { "TIME: " }
                        span class="text-amber font-data" { (elapsed_ms) "ms" }
                    }
                    div {
                        a href=(format!("/graph/algorithms/execute?algorithm={}", params.algorithm)) class="btn-terminal text-xs" {
                            "[ RUN AGAIN ]"
                        }
                    }
                }
            }
        }

        // Results
        (render_result(&result))

        // Back link
        div class="mt-6" {
            a href="/graph/algorithms/dashboard" class="btn-terminal" { "[ BACK TO DASHBOARD ]" }
        }
    };

    layout::layout("Algorithm Results", NavItem::Graph, content)
}

fn parse_direction(s: &str) -> Direction {
    match s.to_lowercase().as_str() {
        "outgoing" | "out" => Direction::Outgoing,
        "incoming" | "in" => Direction::Incoming,
        _ => Direction::Both,
    }
}

fn parse_similarity_metric(s: &str) -> SimilarityMetric {
    match s.to_lowercase().as_str() {
        "cosine" => SimilarityMetric::Cosine,
        "adamic_adar" | "adamicadar" => SimilarityMetric::AdamicAdar,
        "resource_allocation" | "resourceallocation" => SimilarityMetric::ResourceAllocation,
        "preferential_attachment" | "preferentialattachment" => {
            SimilarityMetric::PreferentialAttachment
        },
        "common_neighbors" | "commonneighbors" => SimilarityMetric::CommonNeighbors,
        _ => SimilarityMetric::Jaccard,
    }
}

fn execute_algorithm(ctx: &AdminContext, params: &ExecuteParams) -> AlgorithmResult {
    let algorithm = params.algorithm.clone();

    match params.algorithm.as_str() {
        "pagerank" => {
            let config = PageRankConfig::default()
                .damping(params.damping.unwrap_or(0.85))
                .tolerance(params.tolerance.unwrap_or(1e-6))
                .max_iterations(params.max_iterations.unwrap_or(100));

            match ctx.graph.pagerank(Some(config)) {
                Ok(pr) => {
                    let top_k = params.top_k.unwrap_or(20);
                    let scores = pr.top_k(top_k);
                    AlgorithmResult {
                        algorithm,
                        status: if scores.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Scores(scores),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "betweenness" => {
            let direction = params
                .direction
                .as_deref()
                .map_or(Direction::Both, parse_direction);
            let config = CentralityConfig::default().direction(direction);

            match ctx.graph.betweenness_centrality(Some(config)) {
                Ok(result) => {
                    let top_k = params.top_k.unwrap_or(20);
                    let mut scores: Vec<_> = result.scores.into_iter().collect();
                    scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scores.truncate(top_k);
                    AlgorithmResult {
                        algorithm,
                        status: if scores.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Scores(scores),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "closeness" => {
            let direction = params
                .direction
                .as_deref()
                .map_or(Direction::Both, parse_direction);
            let config = CentralityConfig::default().direction(direction);

            match ctx.graph.closeness_centrality(Some(config)) {
                Ok(result) => {
                    let top_k = params.top_k.unwrap_or(20);
                    let mut scores: Vec<_> = result.scores.into_iter().collect();
                    scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scores.truncate(top_k);
                    AlgorithmResult {
                        algorithm,
                        status: if scores.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Scores(scores),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "eigenvector" => {
            let config = CentralityConfig::default()
                .max_iterations(params.max_iterations.unwrap_or(100))
                .tolerance(params.tolerance.unwrap_or(1e-6));

            match ctx.graph.eigenvector_centrality(Some(config)) {
                Ok(result) => {
                    let top_k = params.top_k.unwrap_or(20);
                    let mut scores: Vec<_> = result.scores.into_iter().collect();
                    scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scores.truncate(top_k);
                    AlgorithmResult {
                        algorithm,
                        status: if scores.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Scores(scores),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "louvain" => {
            let config = CommunityConfig::default()
                .resolution(params.resolution.unwrap_or(1.0))
                .max_passes(params.max_passes.unwrap_or(10));

            match ctx.graph.louvain_communities(Some(config)) {
                Ok(result) => {
                    let communities: Vec<_> = result.communities_by_size();
                    AlgorithmResult {
                        algorithm,
                        status: if communities.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Communities(CommunityData {
                            count: result.community_count,
                            modularity: result.modularity,
                            communities,
                        }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "label_propagation" => {
            let config =
                CommunityConfig::default().max_iterations(params.max_iterations.unwrap_or(100));

            match ctx.graph.label_propagation(Some(config)) {
                Ok(result) => {
                    let communities: Vec<_> = result.communities_by_size();
                    AlgorithmResult {
                        algorithm,
                        status: if communities.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Communities(CommunityData {
                            count: result.community_count,
                            modularity: None,
                            communities,
                        }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "connected_components" => {
            let direction = params
                .direction
                .as_deref()
                .map_or(Direction::Both, parse_direction);
            let config = CommunityConfig::default().direction(direction);

            match ctx.graph.connected_components(Some(config)) {
                Ok(result) => {
                    let communities: Vec<_> = result.communities_by_size();
                    AlgorithmResult {
                        algorithm,
                        status: if communities.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Communities(CommunityData {
                            count: result.community_count,
                            modularity: None,
                            communities,
                        }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "astar" => {
            let from: Option<u64> = params.from.as_ref().and_then(|s| s.parse().ok());
            let to: Option<u64> = params.to.as_ref().and_then(|s| s.parse().ok());

            match (from, to) {
                (Some(from_id), Some(to_id)) => {
                    let config = AStarConfig::new();

                    match ctx.graph.astar_path(from_id, to_id, &config) {
                        Ok(result) => {
                            if let Some(weighted_path) = result.path {
                                AlgorithmResult {
                                    algorithm,
                                    status: ResultStatus::Success,
                                    elapsed_ms: 0,
                                    data: ResultData::Path(PathData {
                                        nodes: weighted_path.nodes,
                                        weight: Some(weighted_path.total_weight),
                                        found: true,
                                    }),
                                }
                            } else {
                                AlgorithmResult {
                                    algorithm,
                                    status: ResultStatus::NoData,
                                    elapsed_ms: 0,
                                    data: ResultData::Path(PathData {
                                        nodes: vec![],
                                        weight: None,
                                        found: false,
                                    }),
                                }
                            }
                        },
                        Err(e) => AlgorithmResult {
                            algorithm,
                            status: ResultStatus::Error,
                            elapsed_ms: 0,
                            data: ResultData::Error(e.to_string()),
                        },
                    }
                },
                _ => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error("Invalid source or target node ID".to_string()),
                },
            }
        },

        "dijkstra" => {
            let from: Option<u64> = params.from.as_ref().and_then(|s| s.parse().ok());
            let to: Option<u64> = params.to.as_ref().and_then(|s| s.parse().ok());
            let weight_prop = params
                .weight_property
                .as_ref()
                .filter(|s| !s.is_empty())
                .map_or("weight", String::as_str);

            match (from, to) {
                (Some(from_id), Some(to_id)) => {
                    match ctx.graph.find_weighted_path(from_id, to_id, weight_prop) {
                        Ok(path) => AlgorithmResult {
                            algorithm,
                            status: ResultStatus::Success,
                            elapsed_ms: 0,
                            data: ResultData::Path(PathData {
                                nodes: path.nodes,
                                weight: Some(path.total_weight),
                                found: true,
                            }),
                        },
                        Err(e) => {
                            if matches!(e, graph_engine::GraphError::PathNotFound) {
                                AlgorithmResult {
                                    algorithm,
                                    status: ResultStatus::NoData,
                                    elapsed_ms: 0,
                                    data: ResultData::Path(PathData {
                                        nodes: vec![],
                                        weight: None,
                                        found: false,
                                    }),
                                }
                            } else {
                                AlgorithmResult {
                                    algorithm,
                                    status: ResultStatus::Error,
                                    elapsed_ms: 0,
                                    data: ResultData::Error(e.to_string()),
                                }
                            }
                        },
                    }
                },
                _ => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error("Invalid source or target node ID".to_string()),
                },
            }
        },

        "variable_paths" => {
            let from: Option<u64> = params.from.as_ref().and_then(|s| s.parse().ok());
            let to: Option<u64> = params.to.as_ref().and_then(|s| s.parse().ok());

            match (from, to) {
                (Some(from_id), Some(to_id)) => {
                    let config = graph_engine::VariableLengthConfig::with_hops(
                        params.min_hops.unwrap_or(1),
                        params.max_hops.unwrap_or(3),
                    )
                    .max_paths(params.max_paths.unwrap_or(100));

                    match ctx.graph.find_variable_paths(from_id, to_id, config) {
                        Ok(result) => {
                            let paths = &result.paths;
                            let summary = HashMap::from([
                                ("paths_found".to_string(), paths.len().to_string()),
                                (
                                    "min_length".to_string(),
                                    paths
                                        .iter()
                                        .map(|p| p.nodes.len())
                                        .min()
                                        .unwrap_or(0)
                                        .to_string(),
                                ),
                                (
                                    "max_length".to_string(),
                                    paths
                                        .iter()
                                        .map(|p| p.nodes.len())
                                        .max()
                                        .unwrap_or(0)
                                        .to_string(),
                                ),
                            ]);
                            let items: Vec<_> = paths
                                .iter()
                                .take(20)
                                .map(|p| {
                                    let path_str = p
                                        .nodes
                                        .iter()
                                        .map(ToString::to_string)
                                        .collect::<Vec<_>>()
                                        .join(" -> ");
                                    (format!("{} nodes", p.nodes.len()), path_str)
                                })
                                .collect();
                            AlgorithmResult {
                                algorithm,
                                status: if paths.is_empty() {
                                    ResultStatus::NoData
                                } else {
                                    ResultStatus::Success
                                },
                                elapsed_ms: 0,
                                data: ResultData::Structure(StructureData { summary, items }),
                            }
                        },
                        Err(e) => AlgorithmResult {
                            algorithm,
                            status: ResultStatus::Error,
                            elapsed_ms: 0,
                            data: ResultData::Error(e.to_string()),
                        },
                    }
                },
                _ => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error("Invalid source or target node ID".to_string()),
                },
            }
        },

        "kcore" => {
            let config = KCoreConfig::default();

            match ctx.graph.kcore_decomposition(&config) {
                Ok(result) => {
                    let min_k = params.min_k.unwrap_or(1);
                    let mut summary = HashMap::new();
                    summary.insert("degeneracy".to_string(), result.degeneracy.to_string());
                    summary.insert("num_cores".to_string(), result.cores.len().to_string());

                    let items: Vec<_> = result
                        .core_numbers
                        .iter()
                        .filter(|(_, &k)| k >= min_k)
                        .take(50)
                        .map(|(node_id, k)| (format!("Node {node_id}"), format!("k={k}")))
                        .collect();

                    AlgorithmResult {
                        algorithm,
                        status: if items.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Structure(StructureData { summary, items }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "scc" => {
            let config = SccConfig::default();

            match ctx.graph.strongly_connected_components(&config) {
                Ok(result) => {
                    let mut summary = HashMap::new();
                    summary.insert(
                        "component_count".to_string(),
                        result.component_count.to_string(),
                    );

                    let items: Vec<_> = result
                        .members
                        .iter()
                        .enumerate()
                        .take(20)
                        .map(|(i, comp): (usize, &Vec<u64>)| {
                            let preview: String = comp
                                .iter()
                                .take(5)
                                .map(|n: &u64| n.to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            let suffix = if comp.len() > 5 {
                                format!("... ({} total)", comp.len())
                            } else {
                                String::new()
                            };
                            (format!("SCC {}", i + 1), format!("[{preview}{suffix}]"))
                        })
                        .collect();

                    AlgorithmResult {
                        algorithm,
                        status: if items.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Structure(StructureData { summary, items }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "mst" => {
            let weight_prop = params
                .weight_property
                .as_ref()
                .filter(|s| !s.is_empty())
                .map_or("weight", String::as_str);
            let config = MstConfig::new(weight_prop);

            match ctx.graph.minimum_spanning_tree(&config) {
                Ok(result) => {
                    let mut summary = HashMap::new();
                    summary.insert("edge_count".to_string(), result.edges.len().to_string());
                    summary.insert(
                        "total_weight".to_string(),
                        format!("{:.4}", result.total_weight),
                    );

                    let items: Vec<_> = result
                        .edges
                        .iter()
                        .take(30)
                        .map(|e| {
                            (
                                format!("{} -> {}", e.from, e.to),
                                format!("weight: {:.4}", e.weight),
                            )
                        })
                        .collect();

                    AlgorithmResult {
                        algorithm,
                        status: if items.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Structure(StructureData { summary, items }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "biconnected" => {
            let config = BiconnectedConfig::default();

            match ctx.graph.biconnected_components(&config) {
                Ok(result) => {
                    let mut summary = HashMap::new();
                    summary.insert(
                        "articulation_points".to_string(),
                        result.articulation_points.len().to_string(),
                    );
                    summary.insert("bridges".to_string(), result.bridges.len().to_string());
                    summary.insert("components".to_string(), result.component_count.to_string());

                    let mut items = Vec::new();
                    for ap in result.articulation_points.iter().take(10) {
                        items.push((
                            format!("Articulation Point {ap}"),
                            "critical node".to_string(),
                        ));
                    }
                    for (from, to) in result.bridges.iter().take(10) {
                        items.push((format!("Bridge {from} - {to}"), "critical edge".to_string()));
                    }

                    AlgorithmResult {
                        algorithm,
                        status: ResultStatus::Success,
                        elapsed_ms: 0,
                        data: ResultData::Structure(StructureData { summary, items }),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "triangles" => {
            let config = TriangleConfig::default();

            match ctx.graph.count_triangles(&config) {
                Ok(result) => {
                    let top_k = params.top_k.unwrap_or(20);
                    #[allow(clippy::cast_precision_loss)]
                    let mut scores: Vec<_> = result
                        .node_triangles
                        .iter()
                        .map(|(k, v)| (*k, *v as f64))
                        .collect();
                    scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scores.truncate(top_k);

                    AlgorithmResult {
                        algorithm,
                        status: if scores.is_empty() {
                            ResultStatus::NoData
                        } else {
                            ResultStatus::Success
                        },
                        elapsed_ms: 0,
                        data: ResultData::Scores(scores),
                    }
                },
                Err(e) => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error(e.to_string()),
                },
            }
        },

        "similarity" => {
            let node_a: Option<u64> = params.node_a.as_ref().and_then(|s| s.parse().ok());
            let node_b: Option<u64> = params.node_b.as_ref().and_then(|s| s.parse().ok());
            let metric = params
                .metric
                .as_deref()
                .map_or(SimilarityMetric::Jaccard, parse_similarity_metric);

            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    let config = SimilarityConfig::new();

                    match ctx.graph.node_similarity(a, b, metric, &config) {
                        Ok(result) => AlgorithmResult {
                            algorithm,
                            status: ResultStatus::Success,
                            elapsed_ms: 0,
                            data: ResultData::Similarity(result.score),
                        },
                        Err(e) => AlgorithmResult {
                            algorithm,
                            status: ResultStatus::Error,
                            elapsed_ms: 0,
                            data: ResultData::Error(e.to_string()),
                        },
                    }
                },
                _ => AlgorithmResult {
                    algorithm,
                    status: ResultStatus::Error,
                    elapsed_ms: 0,
                    data: ResultData::Error("Invalid node IDs".to_string()),
                },
            }
        },

        _ => AlgorithmResult {
            algorithm,
            status: ResultStatus::Error,
            elapsed_ms: 0,
            data: ResultData::Error("Unknown algorithm".to_string()),
        },
    }
}

fn render_algorithm_card(algo: &AlgorithmDef) -> Markup {
    let category_color = match algo.category {
        AlgorithmCategory::Centrality => "text-phosphor",
        AlgorithmCategory::Community => "text-amber",
        AlgorithmCategory::Pathfinding => "text-rust-blood",
        AlgorithmCategory::Structure => "text-phosphor-dim",
        AlgorithmCategory::Similarity => "text-amber-glow",
    };

    html! {
        div class="terminal-panel hover:border-phosphor transition-colors" {
            div class="panel-header flex justify-between items-center" {
                span { (algo.name) }
                span class=(format!("text-xs {category_color}")) { (algo.category.label()) }
            }
            div class="panel-content" {
                p class="text-sm text-phosphor-dim mb-4" { (algo.description) }

                div class="flex justify-between items-center" {
                    span class="text-xs text-phosphor-dark" {
                        (algo.params.len()) " parameters"
                    }
                    a href=(format!("/graph/algorithms/execute?algorithm={}", algo.id))
                      class="btn-terminal text-xs" {
                        "[ CONFIGURE ]"
                    }
                }
            }
        }
    }
}

fn render_result(result: &AlgorithmResult) -> Markup {
    match &result.data {
        ResultData::Scores(scores) => html! {
            div class="terminal-panel" {
                div class="panel-header" { "NODE SCORES" }
                div class="panel-content p-0" {
                    @if scores.is_empty() {
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
                                @for (idx, (node_id, score)) in scores.iter().enumerate() {
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
        },

        ResultData::Communities(data) => html! {
            div class="terminal-panel" {
                div class="panel-header" { "COMMUNITY DETECTION" }
                div class="panel-content" {
                    // Stats
                    div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4 font-terminal text-sm" {
                        div {
                            span class="text-phosphor-dim" { "COMMUNITIES: " }
                            span class="text-amber font-data" { (data.count) }
                        }
                        @if let Some(mod_val) = data.modularity {
                            div {
                                span class="text-phosphor-dim" { "MODULARITY: " }
                                span class="text-amber font-data" { (format!("{mod_val:.4}")) }
                            }
                        }
                    }

                    // Community list
                    @if data.communities.is_empty() {
                        div class="text-phosphor-dim italic" { "< NO COMMUNITIES FOUND >" }
                    } @else {
                        table class="table-rust" {
                            thead {
                                tr {
                                    th { "COMMUNITY ID" }
                                    th class="text-right" { "SIZE" }
                                }
                            }
                            tbody {
                                @for (comm_id, size) in data.communities.iter().take(20) {
                                    tr {
                                        td class="text-phosphor font-data" { (comm_id) }
                                        td class="text-right text-amber font-data" { (size) }
                                    }
                                }
                            }
                        }
                        @if data.communities.len() > 20 {
                            div class="mt-2 text-xs text-phosphor-dark" {
                                "Showing 20 of " (data.communities.len()) " communities"
                            }
                        }
                    }
                }
            }
        },

        ResultData::Path(data) => html! {
            div class="terminal-panel" {
                div class="panel-header" { "PATH RESULT" }
                div class="panel-content" {
                    @if data.found {
                        div class="mb-4 font-terminal text-sm" {
                            span class="text-phosphor-dim" { "PATH LENGTH: " }
                            span class="text-phosphor" { (data.nodes.len()) " nodes" }
                            @if let Some(weight) = data.weight {
                                span class="ml-4 text-phosphor-dim" { "TOTAL WEIGHT: " }
                                span class="text-amber font-data" { (format!("{weight:.4}")) }
                            }
                        }

                        // Visual path
                        div class="overflow-x-auto mb-4" {
                            div class="flex items-center gap-2 min-w-max" {
                                @for (idx, node_id) in data.nodes.iter().enumerate() {
                                    div class="flex flex-col items-center" {
                                        div class="w-12 h-12 border-2 border-phosphor bg-terminal-soot flex items-center justify-center" {
                                            span class="text-sm font-data text-phosphor" { (node_id) }
                                        }
                                    }
                                    @if idx < data.nodes.len() - 1 {
                                        div class="flex items-center text-phosphor-dim" { "--->" }
                                    }
                                }
                            }
                        }

                        // Path sequence
                        div class="text-sm text-phosphor-dim font-terminal" {
                            "SEQUENCE: "
                            @for (idx, node_id) in data.nodes.iter().enumerate() {
                                span class="text-phosphor" { (node_id) }
                                @if idx < data.nodes.len() - 1 { " -> " }
                            }
                        }
                    } @else {
                        (empty_state("NO PATH FOUND", "The nodes are not connected"))
                    }
                }
            }
        },

        ResultData::Structure(data) => html! {
            div class="terminal-panel" {
                div class="panel-header" { "STRUCTURE ANALYSIS" }
                div class="panel-content" {
                    // Summary stats
                    @if !data.summary.is_empty() {
                        div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4 font-terminal text-sm" {
                            @for (key, value) in &data.summary {
                                div {
                                    span class="text-phosphor-dim" { (key.to_uppercase()) ": " }
                                    span class="text-amber font-data" { (value) }
                                }
                            }
                        }
                    }

                    // Items list
                    @if data.items.is_empty() {
                        div class="text-phosphor-dim italic" { "< NO ITEMS >" }
                    } @else {
                        div class="space-y-1 max-h-96 overflow-y-auto" {
                            @for (label, value) in &data.items {
                                div class="flex justify-between font-terminal text-sm border-b border-phosphor-dark pb-1" {
                                    span class="text-phosphor" { (label) }
                                    span class="text-phosphor-dim" { (value) }
                                }
                            }
                        }
                    }
                }
            }
        },

        ResultData::Similarity(score) => html! {
            div class="terminal-panel" {
                div class="panel-header" { "SIMILARITY SCORE" }
                div class="panel-content text-center py-8" {
                    div class="text-5xl font-data text-amber mb-4" { (format!("{score:.6}")) }
                    div class="text-phosphor-dim font-terminal" { "SIMILARITY COEFFICIENT" }
                }
            }
        },

        ResultData::Error(msg) => html! {
            div class="terminal-panel terminal-panel-rust" {
                div class="panel-header" { "ERROR" }
                div class="panel-content text-amber" { (msg) }
            }
        },

        ResultData::Empty => html! {
            (empty_state("NO DATA", "No results to display"))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_category_label() {
        assert_eq!(AlgorithmCategory::Centrality.label(), "CENTRALITY");
        assert_eq!(AlgorithmCategory::Community.label(), "COMMUNITY");
        assert_eq!(AlgorithmCategory::Pathfinding.label(), "PATHFINDING");
        assert_eq!(AlgorithmCategory::Structure.label(), "STRUCTURE");
        assert_eq!(AlgorithmCategory::Similarity.label(), "SIMILARITY");
    }

    #[test]
    fn test_parse_direction() {
        assert!(matches!(parse_direction("outgoing"), Direction::Outgoing));
        assert!(matches!(parse_direction("incoming"), Direction::Incoming));
        assert!(matches!(parse_direction("both"), Direction::Both));
        assert!(matches!(parse_direction("invalid"), Direction::Both));
    }

    #[test]
    fn test_parse_similarity_metric() {
        assert!(matches!(
            parse_similarity_metric("jaccard"),
            SimilarityMetric::Jaccard
        ));
        assert!(matches!(
            parse_similarity_metric("cosine"),
            SimilarityMetric::Cosine
        ));
        assert!(matches!(
            parse_similarity_metric("adamic_adar"),
            SimilarityMetric::AdamicAdar
        ));
    }

    #[test]
    fn test_algorithms_defined() {
        assert_eq!(ALGORITHMS.len(), 16);

        let centrality: Vec<_> = ALGORITHMS
            .iter()
            .filter(|a| a.category == AlgorithmCategory::Centrality)
            .collect();
        assert_eq!(centrality.len(), 4);

        let community: Vec<_> = ALGORITHMS
            .iter()
            .filter(|a| a.category == AlgorithmCategory::Community)
            .collect();
        assert_eq!(community.len(), 3);
    }

    #[test]
    fn test_algorithm_card_rendering() {
        let algo = &ALGORITHMS[0];
        let card = render_algorithm_card(algo);
        let html = card.0;
        assert!(html.contains(algo.name));
        assert!(html.contains("CONFIGURE"));
    }

    #[test]
    fn test_result_rendering_scores() {
        let result = AlgorithmResult {
            algorithm: "pagerank".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 100,
            data: ResultData::Scores(vec![(1, 0.5), (2, 0.3)]),
        };
        let html = render_result(&result).0;
        assert!(html.contains("NODE SCORES"));
        assert!(html.contains("0.500000"));
    }

    #[test]
    fn test_result_rendering_error() {
        let result = AlgorithmResult {
            algorithm: "test".to_string(),
            status: ResultStatus::Error,
            elapsed_ms: 0,
            data: ResultData::Error("Test error".to_string()),
        };
        let html = render_result(&result).0;
        assert!(html.contains("ERROR"));
        assert!(html.contains("Test error"));
    }

    #[test]
    fn test_result_rendering_similarity() {
        let result = AlgorithmResult {
            algorithm: "similarity".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 50,
            data: ResultData::Similarity(0.75),
        };
        let html = render_result(&result).0;
        assert!(html.contains("SIMILARITY SCORE"));
        assert!(html.contains("0.750000"));
    }

    #[test]
    fn test_result_rendering_communities() {
        let result = AlgorithmResult {
            algorithm: "louvain".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 200,
            data: ResultData::Communities(CommunityData {
                count: 3,
                modularity: Some(0.45),
                communities: vec![(1, 10), (2, 5)],
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("COMMUNITY DETECTION"));
        assert!(html.contains("COMMUNITIES:"));
        assert!(html.contains("MODULARITY:"));
    }

    #[test]
    fn test_result_rendering_path() {
        let result = AlgorithmResult {
            algorithm: "astar".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 10,
            data: ResultData::Path(PathData {
                nodes: vec![1, 2, 3],
                weight: Some(2.5),
                found: true,
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("PATH RESULT"));
        assert!(html.contains("3 nodes"));
        assert!(html.contains("TOTAL WEIGHT"));
    }

    #[test]
    fn test_result_rendering_structure() {
        let result = AlgorithmResult {
            algorithm: "kcore".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 50,
            data: ResultData::Structure(StructureData {
                summary: HashMap::from([("degeneracy".to_string(), "5".to_string())]),
                items: vec![("Node 1".to_string(), "k=5".to_string())],
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("STRUCTURE ANALYSIS"));
        assert!(html.contains("DEGENERACY"));
    }

    // ========== Additional tests for improved coverage ==========

    #[test]
    fn test_dashboard_params_defaults() {
        let params: DashboardParams = serde_json::from_str("{}").unwrap();
        assert!(params.category.is_none());
    }

    #[test]
    fn test_dashboard_params_with_category() {
        let params: DashboardParams =
            serde_json::from_str(r#"{"category": "centrality"}"#).unwrap();
        assert_eq!(params.category.as_deref(), Some("centrality"));
    }

    #[test]
    fn test_execute_params_defaults() {
        let params: ExecuteParams = serde_json::from_str(r#"{"algorithm": "pagerank"}"#).unwrap();
        assert_eq!(params.algorithm, "pagerank");
        assert!(params.top_k.is_none());
        assert!(params.max_iterations.is_none());
        assert!(params.tolerance.is_none());
        assert!(params.direction.is_none());
        assert!(params.damping.is_none());
    }

    #[test]
    fn test_execute_params_full() {
        let params: ExecuteParams = serde_json::from_str(
            r#"{
            "algorithm": "pagerank",
            "top_k": 10,
            "max_iterations": 50,
            "tolerance": 0.001,
            "damping": 0.9
        }"#,
        )
        .unwrap();
        assert_eq!(params.algorithm, "pagerank");
        assert_eq!(params.top_k, Some(10));
        assert_eq!(params.max_iterations, Some(50));
        assert_eq!(params.tolerance, Some(0.001));
        assert_eq!(params.damping, Some(0.9));
    }

    #[test]
    fn test_execute_params_pathfinding() {
        let params: ExecuteParams = serde_json::from_str(
            r#"{
            "algorithm": "astar",
            "from": "123",
            "to": "456",
            "max_depth": 10
        }"#,
        )
        .unwrap();
        assert_eq!(params.from.as_deref(), Some("123"));
        assert_eq!(params.to.as_deref(), Some("456"));
        assert_eq!(params.max_depth, Some(10));
    }

    #[test]
    fn test_execute_params_variable_paths() {
        let params: ExecuteParams = serde_json::from_str(
            r#"{
            "algorithm": "variable_paths",
            "from": "1",
            "to": "2",
            "min_hops": 2,
            "max_hops": 5,
            "max_paths": 50
        }"#,
        )
        .unwrap();
        assert_eq!(params.min_hops, Some(2));
        assert_eq!(params.max_hops, Some(5));
        assert_eq!(params.max_paths, Some(50));
    }

    #[test]
    fn test_execute_params_kcore() {
        let params: ExecuteParams = serde_json::from_str(
            r#"{
            "algorithm": "kcore",
            "min_k": 3
        }"#,
        )
        .unwrap();
        assert_eq!(params.min_k, Some(3));
    }

    #[test]
    fn test_execute_params_similarity() {
        let params: ExecuteParams = serde_json::from_str(
            r#"{
            "algorithm": "similarity",
            "node_a": "10",
            "node_b": "20",
            "metric": "cosine"
        }"#,
        )
        .unwrap();
        assert_eq!(params.node_a.as_deref(), Some("10"));
        assert_eq!(params.node_b.as_deref(), Some("20"));
        assert_eq!(params.metric.as_deref(), Some("cosine"));
    }

    #[test]
    fn test_execute_params_louvain() {
        let params: ExecuteParams = serde_json::from_str(
            r#"{
            "algorithm": "louvain",
            "resolution": 1.5,
            "max_passes": 20
        }"#,
        )
        .unwrap();
        assert_eq!(params.resolution, Some(1.5));
        assert_eq!(params.max_passes, Some(20));
    }

    #[test]
    fn test_result_status_serialization() {
        let success = serde_json::to_string(&ResultStatus::Success).unwrap();
        assert_eq!(success, "\"success\"");

        let error = serde_json::to_string(&ResultStatus::Error).unwrap();
        assert_eq!(error, "\"error\"");

        let no_data = serde_json::to_string(&ResultStatus::NoData).unwrap();
        assert_eq!(no_data, "\"no_data\"");
    }

    #[test]
    fn test_community_data_serialization() {
        let data = CommunityData {
            count: 5,
            modularity: Some(0.65),
            communities: vec![(1, 100), (2, 50)],
        };
        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"count\":5"));
        assert!(json.contains("\"modularity\":0.65"));
        assert!(json.contains("\"communities\""));
    }

    #[test]
    fn test_path_data_serialization() {
        let data = PathData {
            nodes: vec![1, 2, 3, 4],
            weight: Some(10.5),
            found: true,
        };
        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"nodes\":[1,2,3,4]"));
        assert!(json.contains("\"weight\":10.5"));
        assert!(json.contains("\"found\":true"));
    }

    #[test]
    fn test_path_data_not_found() {
        let data = PathData {
            nodes: vec![],
            weight: None,
            found: false,
        };
        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"nodes\":[]"));
        assert!(json.contains("\"weight\":null"));
        assert!(json.contains("\"found\":false"));
    }

    #[test]
    fn test_structure_data_serialization() {
        let data = StructureData {
            summary: HashMap::from([("key1".to_string(), "value1".to_string())]),
            items: vec![("item1".to_string(), "desc1".to_string())],
        };
        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"summary\""));
        assert!(json.contains("\"items\""));
    }

    #[test]
    fn test_algorithm_result_serialization() {
        let result = AlgorithmResult {
            algorithm: "test".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 42,
            data: ResultData::Empty,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"algorithm\":\"test\""));
        assert!(json.contains("\"status\":\"success\""));
        assert!(json.contains("\"elapsed_ms\":42"));
    }

    #[test]
    fn test_parse_direction_variations() {
        assert!(matches!(parse_direction("out"), Direction::Outgoing));
        assert!(matches!(parse_direction("in"), Direction::Incoming));
        assert!(matches!(parse_direction("OUTGOING"), Direction::Outgoing));
        assert!(matches!(parse_direction("INCOMING"), Direction::Incoming));
        assert!(matches!(parse_direction("BOTH"), Direction::Both));
        assert!(matches!(parse_direction("unknown"), Direction::Both));
    }

    #[test]
    fn test_parse_similarity_metric_variations() {
        assert!(matches!(
            parse_similarity_metric("adamicadar"),
            SimilarityMetric::AdamicAdar
        ));
        assert!(matches!(
            parse_similarity_metric("resourceallocation"),
            SimilarityMetric::ResourceAllocation
        ));
        assert!(matches!(
            parse_similarity_metric("resource_allocation"),
            SimilarityMetric::ResourceAllocation
        ));
        assert!(matches!(
            parse_similarity_metric("preferentialattachment"),
            SimilarityMetric::PreferentialAttachment
        ));
        assert!(matches!(
            parse_similarity_metric("preferential_attachment"),
            SimilarityMetric::PreferentialAttachment
        ));
        assert!(matches!(
            parse_similarity_metric("commonneighbors"),
            SimilarityMetric::CommonNeighbors
        ));
        assert!(matches!(
            parse_similarity_metric("common_neighbors"),
            SimilarityMetric::CommonNeighbors
        ));
        assert!(matches!(
            parse_similarity_metric("unknown"),
            SimilarityMetric::Jaccard
        ));
    }

    #[test]
    fn test_algorithm_category_serialization() {
        let centrality = serde_json::to_string(&AlgorithmCategory::Centrality).unwrap();
        assert_eq!(centrality, "\"centrality\"");

        let community = serde_json::to_string(&AlgorithmCategory::Community).unwrap();
        assert_eq!(community, "\"community\"");

        let pathfinding = serde_json::to_string(&AlgorithmCategory::Pathfinding).unwrap();
        assert_eq!(pathfinding, "\"pathfinding\"");

        let structure = serde_json::to_string(&AlgorithmCategory::Structure).unwrap();
        assert_eq!(structure, "\"structure\"");

        let similarity = serde_json::to_string(&AlgorithmCategory::Similarity).unwrap();
        assert_eq!(similarity, "\"similarity\"");
    }

    #[test]
    fn test_result_rendering_empty() {
        let result = AlgorithmResult {
            algorithm: "empty".to_string(),
            status: ResultStatus::NoData,
            elapsed_ms: 0,
            data: ResultData::Empty,
        };
        let html = render_result(&result).0;
        assert!(html.contains("NO DATA"));
    }

    #[test]
    fn test_result_rendering_path_not_found() {
        let result = AlgorithmResult {
            algorithm: "astar".to_string(),
            status: ResultStatus::NoData,
            elapsed_ms: 10,
            data: ResultData::Path(PathData {
                nodes: vec![],
                weight: None,
                found: false,
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("NO PATH FOUND"));
    }

    #[test]
    fn test_result_rendering_scores_empty() {
        let result = AlgorithmResult {
            algorithm: "pagerank".to_string(),
            status: ResultStatus::NoData,
            elapsed_ms: 0,
            data: ResultData::Scores(vec![]),
        };
        let html = render_result(&result).0;
        assert!(html.contains("NO RESULTS"));
    }

    #[test]
    fn test_result_rendering_communities_empty() {
        let result = AlgorithmResult {
            algorithm: "louvain".to_string(),
            status: ResultStatus::NoData,
            elapsed_ms: 0,
            data: ResultData::Communities(CommunityData {
                count: 0,
                modularity: None,
                communities: vec![],
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("NO COMMUNITIES FOUND"));
    }

    #[test]
    fn test_result_rendering_structure_empty() {
        let result = AlgorithmResult {
            algorithm: "kcore".to_string(),
            status: ResultStatus::NoData,
            elapsed_ms: 0,
            data: ResultData::Structure(StructureData {
                summary: HashMap::new(),
                items: vec![],
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("NO ITEMS"));
    }

    #[test]
    fn test_algorithm_card_community_category() {
        let algo = ALGORITHMS
            .iter()
            .find(|a| a.category == AlgorithmCategory::Community)
            .unwrap();
        let card = render_algorithm_card(algo);
        let html = card.0;
        assert!(html.contains("text-amber"));
        assert!(html.contains("COMMUNITY"));
    }

    #[test]
    fn test_algorithm_card_pathfinding_category() {
        let algo = ALGORITHMS
            .iter()
            .find(|a| a.category == AlgorithmCategory::Pathfinding)
            .unwrap();
        let card = render_algorithm_card(algo);
        let html = card.0;
        assert!(html.contains("text-rust-blood"));
        assert!(html.contains("PATHFINDING"));
    }

    #[test]
    fn test_algorithm_card_structure_category() {
        let algo = ALGORITHMS
            .iter()
            .find(|a| a.category == AlgorithmCategory::Structure)
            .unwrap();
        let card = render_algorithm_card(algo);
        let html = card.0;
        assert!(html.contains("text-phosphor-dim"));
        assert!(html.contains("STRUCTURE"));
    }

    #[test]
    fn test_algorithm_card_similarity_category() {
        let algo = ALGORITHMS
            .iter()
            .find(|a| a.category == AlgorithmCategory::Similarity)
            .unwrap();
        let card = render_algorithm_card(algo);
        let html = card.0;
        assert!(html.contains("text-amber-glow"));
        assert!(html.contains("SIMILARITY"));
    }

    #[test]
    fn test_algorithms_have_unique_ids() {
        let mut ids: Vec<_> = ALGORITHMS.iter().map(|a| a.id).collect();
        ids.sort();
        let original_len = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "Algorithm IDs must be unique");
    }

    #[test]
    fn test_all_algorithms_have_descriptions() {
        for algo in ALGORITHMS {
            assert!(
                !algo.description.is_empty(),
                "Algorithm {} has empty description",
                algo.id
            );
            assert!(
                !algo.name.is_empty(),
                "Algorithm {} has empty name",
                algo.id
            );
        }
    }

    #[test]
    fn test_result_rendering_communities_with_many() {
        let communities: Vec<_> = (0..25).map(|i| (i as u64, i * 10)).collect();
        let result = AlgorithmResult {
            algorithm: "louvain".to_string(),
            status: ResultStatus::Success,
            elapsed_ms: 100,
            data: ResultData::Communities(CommunityData {
                count: 25,
                modularity: Some(0.5),
                communities,
            }),
        };
        let html = render_result(&result).0;
        assert!(html.contains("Showing 20 of 25 communities"));
    }
}
