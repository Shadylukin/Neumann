use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use parking_lot::RwLock;
use tracing_subscriber::EnvFilter;

use graph_engine::GraphEngine;
use neumann_server::{NeumannServer, RestConfig, ServerConfig};
use query_router::QueryRouter;
use relational_engine::RelationalEngine;
use tensor_spatial::SpatialIndex3D;
use tensor_store::TensorStore;
use vector_engine::VectorEngine;

/// Galaxy server for the Knowledge Galaxy 3D visualization demo.
#[derive(Parser, Debug)]
#[command(name = "galaxy-server")]
struct Args {
    /// Path to the Neumann snapshot database file.
    #[arg(long, default_value = "galaxy.db")]
    db: PathBuf,

    /// Web admin port (serves /api/galaxy).
    #[arg(long, default_value = "9000")]
    web_port: u16,

    /// REST API port (serves /collections/galaxy/spatial3d/*).
    #[arg(long, default_value = "8080")]
    rest_port: u16,

    /// gRPC port (standard Neumann gRPC).
    #[arg(long, default_value = "9200")]
    grpc_port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Load snapshot if it exists
    let store = if args.db.exists() {
        tracing::info!("Loading snapshot from {}", args.db.display());
        TensorStore::load_snapshot_compressed(&args.db)
            .or_else(|_| TensorStore::load_snapshot(&args.db))
            .map_err(|e| format!("Failed to load snapshot {}: {e}", args.db.display()))?
    } else {
        tracing::warn!(
            "No snapshot found at {}, starting with empty store",
            args.db.display()
        );
        TensorStore::new()
    };

    // Create shared engine instances so all contexts see the same state
    let relational = Arc::new(RelationalEngine::with_store(store.clone()));
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let vector = Arc::new(VectorEngine::with_store(store.clone()));

    // Create QueryRouter with the shared engines
    let router = Arc::new(RwLock::new(QueryRouter::with_engines(
        Arc::clone(&relational),
        Arc::clone(&graph),
        Arc::clone(&vector),
    )));

    // 3D spatial index (ephemeral, loaded via REST on each boot)
    let spatial_3d = Arc::new(RwLock::new(SpatialIndex3D::<String>::new()));

    // Build addresses
    let web_addr: SocketAddr = format!("0.0.0.0:{}", args.web_port).parse()?;
    let rest_addr: SocketAddr = format!("0.0.0.0:{}", args.rest_port).parse()?;
    let grpc_addr: SocketAddr = format!("0.0.0.0:{}", args.grpc_port).parse()?;

    // Build server config with CORS enabled for the frontend
    let config = ServerConfig::default()
        .with_bind_addr(grpc_addr)
        .with_web_addr(web_addr)
        .with_rest_addr(rest_addr)
        .with_rest_config(RestConfig {
            cors_enabled: true,
            cors_origins: vec!["http://localhost:5173".into()],
            ..RestConfig::default()
        });

    // Assemble server
    let server = NeumannServer::new(Arc::clone(&router), config)
        .with_relational_engine(relational)
        .with_vector_engine(vector)
        .with_graph_engine(graph)
        .with_spatial_3d(spatial_3d);

    tracing::info!("Galaxy server starting:");
    tracing::info!("  Web (galaxy API): http://localhost:{}", args.web_port);
    tracing::info!("  REST (spatial 3D): http://localhost:{}", args.rest_port);
    tracing::info!("  gRPC:              http://localhost:{}", args.grpc_port);
    tracing::info!(
        "  Frontend:          http://localhost:5173 (start with `cd frontend && npm run dev`)"
    );

    server.serve().await.map_err(|e| e.to_string())?;
    Ok(())
}
