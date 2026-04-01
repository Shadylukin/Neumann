// SPDX-License-Identifier: MIT OR Apache-2.0
//! Neumann Server binary entry point.

use std::sync::Arc;

use neumann_server::{NeumannServer, ServerConfig};
use parking_lot::RwLock;
use query_router::QueryRouter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("neumann_server=info".parse()?)
                .add_directive("query_router=info".parse()?)
                .add_directive("tower_http=debug".parse()?),
        )
        .init();

    // Load configuration from environment or defaults
    let mut config = ServerConfig::from_env()?;

    tracing::info!("Starting Neumann server on {}", config.bind_addr);

    // Create the query router
    let router = Arc::new(RwLock::new(QueryRouter::new()));

    // Initialize cluster mode if configured
    if let Some(ref cluster_config) = config.cluster {
        router.write().init_cluster_with_wal(
            &cluster_config.node_id,
            cluster_config.raft_bind_addr,
            &cluster_config.peers,
            &cluster_config.data_dir,
        )?;
        tracing::info!(
            "Cluster mode: {} listening on {}",
            cluster_config.node_id,
            cluster_config.raft_bind_addr,
        );
    }

    // Create and run the server
    let mut server = NeumannServer::new(router, config.clone());

    // Optionally attach a grokking experiment (NEUMANN_LEARN=1)
    if std::env::var("NEUMANN_LEARN").is_ok() {
        if config.web_addr.is_none() {
            config.web_addr = Some("127.0.0.1:9000".parse()?);
        }

        server = NeumannServer::new(Arc::new(RwLock::new(QueryRouter::new())), config)
            .with_relational_engine(Arc::new(relational_engine::RelationalEngine::new()))
            .with_vector_engine(Arc::new(vector_engine::VectorEngine::new()))
            .with_graph_engine(Arc::new(graph_engine::GraphEngine::new()));

        let grok = tensor_learn::GrokSession::new(tensor_learn::GrokConfig::default());
        server = server.with_grok(Arc::new(RwLock::new(grok)));

        tracing::info!("Grokking dashboard at http://127.0.0.1:9000/learn");
    }

    server.serve().await?;

    Ok(())
}
