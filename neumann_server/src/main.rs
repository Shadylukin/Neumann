// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
    let config = ServerConfig::from_env()?;

    tracing::info!("Starting Neumann server on {}", config.bind_addr);

    // Create the query router
    let router = Arc::new(RwLock::new(QueryRouter::new()));

    // Create and run the server
    let server = NeumannServer::new(router, config);
    server.serve().await?;

    Ok(())
}
