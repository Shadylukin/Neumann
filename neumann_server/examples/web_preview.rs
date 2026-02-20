//! Full-featured preview runner for the Memoria web admin UI.
//!
//! Seeds every subsystem with demo data so all dashboard pages display
//! meaningful content during design review.
#![allow(missing_docs)]

use std::sync::Arc;

use graph_engine::GraphEngine;
use opentelemetry::metrics::MeterProvider;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use relational_engine::{Column, ColumnType, RelationalEngine, Schema, Value};
use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_cache::{Cache, CacheConfig};
use tensor_checkpoint::{CheckpointConfig, CheckpointManager};
use tensor_store::TensorStore;
use tensor_unified::UnifiedEngine;
use tensor_vault::{Vault, VaultConfig};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use vector_engine::VectorEngine;

use neumann_server::metrics::ServerMetrics;
use neumann_server::web::{self, AdminContext, ChainStatus};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter("neumann_server=info,tower_http=debug")
        .init();

    // Instrumented store so the Storage Internals page has data
    let store = TensorStore::with_instrumentation(16);

    let relational = Arc::new(RelationalEngine::with_store(store.clone()));
    let vector = Arc::new(VectorEngine::with_store(store.clone()));
    let graph = Arc::new(GraphEngine::with_store(store.clone()));

    // ── Relational ──────────────────────────────────────────────
    seed_relational(&relational);

    // ── Vector ──────────────────────────────────────────────────
    seed_vectors(&vector);

    // ── Graph ───────────────────────────────────────────────────
    seed_graph(&graph);

    // ── Unified / Contraction ───────────────────────────────────
    let unified = Arc::new(UnifiedEngine::with_engines(
        store.clone(),
        Arc::clone(&relational),
        Arc::clone(&graph),
        Arc::clone(&vector),
    ));

    // ── Vault ───────────────────────────────────────────────────
    let vault = seed_vault(&graph, &store);

    // ── Cache ───────────────────────────────────────────────────
    let cache = seed_cache();

    // ── Blob Storage ────────────────────────────────────────────
    let blob = seed_blob(&store).await;
    let blob_arc = Arc::new(Mutex::new(blob));

    // ── Checkpoint ──────────────────────────────────────────────
    let checkpoint = seed_checkpoint(Arc::clone(&blob_arc), &store).await;

    // ── Chain / Consensus ───────────────────────────────────────
    let chain = demo_chain_status();

    // ── Metrics ─────────────────────────────────────────────────
    let metrics = seed_metrics();

    // ── Build AdminContext directly ─────────────────────────────
    let ctx = Arc::new(
        AdminContext::new(
            Arc::clone(&relational),
            Arc::clone(&vector),
            Arc::clone(&graph),
        )
        .with_unified(Some(unified))
        .with_vault(Some(vault))
        .with_cache(Some(cache))
        .with_blob(Some(blob_arc))
        .with_checkpoint(Some(Arc::new(checkpoint)))
        .with_store(Some(store))
        .with_chain(Some(chain))
        .with_metrics(Some(metrics)),
    );

    let web_router = web::router(ctx);
    let addr = "127.0.0.1:9000";
    let listener = TcpListener::bind(addr).await?;

    eprintln!("\n  Memoria Web Admin UI: http://{addr}\n");
    tracing::info!("Web admin UI enabled on {addr}");

    axum::serve(listener, web_router).await?;

    Ok(())
}

// ── Seed helpers ────────────────────────────────────────────────

fn seed_relational(engine: &RelationalEngine) {
    // Users table
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("email", ColumnType::String),
        Column::new("role", ColumnType::String),
    ]);
    engine.create_table("users", schema).ok();
    let users = [
        (0, "Alice", "alice@example.com", "admin"),
        (1, "Bob", "bob@example.com", "engineer"),
        (2, "Carol", "carol@example.com", "designer"),
        (3, "Dave", "dave@example.com", "engineer"),
        (4, "Eve", "eve@example.com", "analyst"),
        (5, "Frank", "frank@example.com", "engineer"),
        (6, "Grace", "grace@example.com", "admin"),
        (7, "Heidi", "heidi@example.com", "designer"),
    ];
    for (id, name, email, role) in &users {
        engine
            .insert(
                "users",
                vec![
                    ("id".to_string(), Value::Int(*id)),
                    ("name".to_string(), Value::String(name.to_string())),
                    ("email".to_string(), Value::String(email.to_string())),
                    ("role".to_string(), Value::String(role.to_string())),
                ]
                .into_iter()
                .collect(),
            )
            .ok();
    }
    // Add hash index on email
    engine.create_index("users", "email").ok();

    // Purchases table
    let purchase_schema = Schema::new(vec![
        Column::new("buyer", ColumnType::String),
        Column::new("item", ColumnType::String),
        Column::new("amount", ColumnType::Float),
    ]);
    engine.create_table("purchases", purchase_schema).ok();
    let purchases = [
        ("alice", "laptop", 1299.99),
        ("alice", "mouse", 49.99),
        ("bob", "laptop", 1299.99),
        ("bob", "keyboard", 149.99),
        ("carol", "mouse", 49.99),
        ("carol", "monitor", 499.99),
        ("dave", "laptop", 1499.99),
        ("eve", "keyboard", 129.99),
        ("frank", "monitor", 599.99),
        ("grace", "mouse", 39.99),
    ];
    for (buyer, item, amount) in &purchases {
        engine
            .insert(
                "purchases",
                vec![
                    ("buyer".to_string(), Value::String(buyer.to_string())),
                    ("item".to_string(), Value::String(item.to_string())),
                    ("amount".to_string(), Value::Float(*amount)),
                ]
                .into_iter()
                .collect(),
            )
            .ok();
    }

    // Products table
    let product_schema = Schema::new(vec![
        Column::new("sku", ColumnType::String),
        Column::new("name", ColumnType::String),
        Column::new("price", ColumnType::Float),
        Column::new("category", ColumnType::String),
    ]);
    engine.create_table("products", product_schema).ok();
    let products = [
        ("SKU-001", "Laptop Pro", 1299.99, "electronics"),
        ("SKU-002", "Wireless Mouse", 49.99, "peripherals"),
        ("SKU-003", "Mechanical Keyboard", 149.99, "peripherals"),
        ("SKU-004", "4K Monitor", 499.99, "displays"),
        ("SKU-005", "USB-C Hub", 79.99, "accessories"),
    ];
    for (sku, name, price, category) in &products {
        engine
            .insert(
                "products",
                vec![
                    ("sku".to_string(), Value::String(sku.to_string())),
                    ("name".to_string(), Value::String(name.to_string())),
                    ("price".to_string(), Value::Float(*price)),
                    ("category".to_string(), Value::String(category.to_string())),
                ]
                .into_iter()
                .collect(),
            )
            .ok();
    }
    engine.create_index("products", "sku").ok();
}

fn seed_vectors(engine: &VectorEngine) {
    // Default collection embeddings
    let embeddings: &[(&str, &[f32])] = &[
        ("alice", &[1.0, 0.5, 0.3, 0.8]),
        ("bob", &[0.9, 0.6, 0.2, 0.7]),
        ("carol", &[0.1, 0.2, 0.9, 0.4]),
        ("dave", &[0.8, 0.3, 0.5, 0.6]),
        ("laptop", &[0.8, 0.4, 0.3, 0.9]),
        ("mouse", &[0.7, 0.5, 0.4, 0.6]),
        ("keyboard", &[0.6, 0.7, 0.2, 0.5]),
        ("monitor", &[0.5, 0.3, 0.8, 0.7]),
    ];
    for (key, emb) in embeddings {
        engine.store_embedding(key, emb.to_vec()).ok();
    }

    // Named collection: "product_embeddings"
    engine
        .create_collection("product_embeddings", Default::default())
        .ok();
    let product_embs: &[(&str, &[f32])] = &[
        ("SKU-001", &[0.95, 0.8, 0.1, 0.3]),
        ("SKU-002", &[0.2, 0.9, 0.7, 0.1]),
        ("SKU-003", &[0.3, 0.85, 0.6, 0.2]),
        ("SKU-004", &[0.7, 0.3, 0.9, 0.5]),
        ("SKU-005", &[0.4, 0.6, 0.5, 0.8]),
    ];
    for (key, emb) in product_embs {
        engine
            .store_in_collection("product_embeddings", key, emb.to_vec())
            .ok();
    }

    // Named collection: "document_embeddings"
    engine
        .create_collection("document_embeddings", Default::default())
        .ok();
    let doc_embs: &[(&str, &[f32])] = &[
        ("doc-arch", &[0.9, 0.1, 0.4, 0.7]),
        ("doc-api", &[0.8, 0.3, 0.5, 0.6]),
        ("doc-deploy", &[0.2, 0.7, 0.8, 0.3]),
    ];
    for (key, emb) in doc_embs {
        engine
            .store_in_collection("document_embeddings", key, emb.to_vec())
            .ok();
    }
}

fn seed_graph(engine: &GraphEngine) {
    use graph_engine::PropertyValue;
    use std::collections::HashMap;

    fn named(name: &str) -> HashMap<String, PropertyValue> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), PropertyValue::String(name.to_string()));
        m
    }

    // Users
    let alice = engine.create_node("User", named("Alice")).unwrap();
    let bob = engine.create_node("User", named("Bob")).unwrap();
    let carol = engine.create_node("User", named("Carol")).unwrap();
    let dave = engine.create_node("User", named("Dave")).unwrap();
    let eve = engine.create_node("User", named("Eve")).unwrap();

    // Items
    let laptop = engine.create_node("Item", named("Laptop")).unwrap();
    let mouse = engine.create_node("Item", named("Mouse")).unwrap();
    let keyboard = engine.create_node("Item", named("Keyboard")).unwrap();
    let monitor = engine.create_node("Item", named("Monitor")).unwrap();
    let hub = engine.create_node("Item", named("USB-C Hub")).unwrap();

    // Teams
    let eng_team = engine.create_node("Team", named("Engineering")).unwrap();
    let design_team = engine.create_node("Team", named("Design")).unwrap();

    let d = HashMap::new();

    // Purchase edges
    engine
        .create_edge(alice, laptop, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(alice, mouse, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(bob, laptop, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(bob, keyboard, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(carol, mouse, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(carol, monitor, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(dave, laptop, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(dave, hub, "BOUGHT", d.clone(), true)
        .ok();
    engine
        .create_edge(eve, keyboard, "BOUGHT", d.clone(), true)
        .ok();

    // Social edges
    engine
        .create_edge(alice, bob, "KNOWS", d.clone(), true)
        .ok();
    engine
        .create_edge(bob, carol, "KNOWS", d.clone(), true)
        .ok();
    engine
        .create_edge(carol, dave, "KNOWS", d.clone(), true)
        .ok();
    engine.create_edge(dave, eve, "KNOWS", d.clone(), true).ok();
    engine
        .create_edge(alice, carol, "KNOWS", d.clone(), true)
        .ok();

    // Team membership
    engine
        .create_edge(alice, eng_team, "MEMBER_OF", d.clone(), true)
        .ok();
    engine
        .create_edge(bob, eng_team, "MEMBER_OF", d.clone(), true)
        .ok();
    engine
        .create_edge(dave, eng_team, "MEMBER_OF", d.clone(), true)
        .ok();
    engine
        .create_edge(carol, design_team, "MEMBER_OF", d.clone(), true)
        .ok();
    engine
        .create_edge(eve, design_team, "MEMBER_OF", d, true)
        .ok();
}

fn seed_vault(graph: &Arc<GraphEngine>, store: &TensorStore) -> Arc<Vault> {
    let config = VaultConfig {
        argon2_memory_cost: 256,
        argon2_time_cost: 1,
        argon2_parallelism: 1,
        ..VaultConfig::default()
    };
    let vault = Arc::new(
        Vault::new(
            b"preview_master_key",
            Arc::clone(graph),
            store.clone(),
            config,
        )
        .expect("vault init"),
    );

    // Database secrets
    vault
        .set(Vault::ROOT, "db/password", "s3cret_p0stgres!")
        .ok();
    vault
        .set(Vault::ROOT, "db/replica_password", "r3pl1ca_pass")
        .ok();

    // API keys
    vault
        .set(Vault::ROOT, "api/stripe_key", "sk_live_abc123def456")
        .ok();
    vault
        .set(Vault::ROOT, "api/sendgrid_key", "SG.xyz789abc")
        .ok();

    // Auth secrets
    vault
        .set(Vault::ROOT, "jwt/signing_key", "hmac-sha256-key-value")
        .ok();
    vault
        .set(Vault::ROOT, "jwt/refresh_secret", "refresh-tok-secret")
        .ok();

    // Cloud credentials
    vault
        .set(Vault::ROOT, "aws/access_key_id", "AKIAIOSFODNN7EXAMPLE")
        .ok();
    vault
        .set(
            Vault::ROOT,
            "aws/secret_access_key",
            "wJalrXUtnFEMI/K7MDENG",
        )
        .ok();

    // Grant access
    let keys = [
        "db/password",
        "db/replica_password",
        "api/stripe_key",
        "api/sendgrid_key",
        "jwt/signing_key",
        "jwt/refresh_secret",
        "aws/access_key_id",
        "aws/secret_access_key",
    ];
    for key in &keys {
        vault.grant(Vault::ROOT, "admin", key).ok();
    }
    // Partial access for "deploy" entity
    vault.grant(Vault::ROOT, "deploy", "db/password").ok();
    vault.grant(Vault::ROOT, "deploy", "aws/access_key_id").ok();
    vault
        .grant(Vault::ROOT, "deploy", "aws/secret_access_key")
        .ok();

    vault
}

fn seed_cache() -> Arc<Cache> {
    let config = CacheConfig::development();
    let dim = config.embedding_dim;
    let cache = Arc::new(Cache::with_config(config).expect("cache init"));

    let emb = vec![0.1_f32; dim];
    let entries: &[(&str, &str)] = &[
        (
            "What is Neumann?",
            "Neumann is a unified tensor-based runtime for relational, graph, and vector data.",
        ),
        (
            "How do tensors work?",
            "Tensors are multi-dimensional arrays that generalize scalars, vectors, and matrices.",
        ),
        (
            "Explain HNSW indexing",
            "HNSW builds a navigable small-world graph for approximate nearest neighbor search.",
        ),
        (
            "What is Raft consensus?",
            "Raft is a consensus algorithm that ensures all nodes in a cluster agree on state.",
        ),
        (
            "How does 2PC work?",
            "Two-phase commit coordinates distributed transactions with prepare and commit phases.",
        ),
        (
            "What are sparse vectors?",
            "Sparse vectors store only non-zero elements, saving memory for high-dimensional data.",
        ),
    ];
    for (question, answer) in entries {
        cache.put(question, &emb, answer, "gpt-4", None).ok();
    }
    // Generate some cache hits for stats
    for question in ["What is Neumann?", "How do tensors work?"] {
        let _ = cache.get(question, None);
    }

    cache
}

async fn seed_blob(store: &TensorStore) -> BlobStore {
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .expect("blob init");

    // Upload demo artifacts
    blob.put(
        "architecture.md",
        b"# Architecture\n\nNeumann uses a unified tensor store as the foundation...",
        PutOptions::new()
            .with_content_type("text/markdown")
            .with_created_by("user:alice")
            .with_tag("documentation")
            .with_tag("architecture")
            .with_meta("version", "2.1"),
    )
    .await
    .ok();

    blob.put(
        "quarterly_report.pdf",
        &vec![0u8; 2048], // placeholder bytes
        PutOptions::new()
            .with_content_type("application/pdf")
            .with_created_by("user:eve")
            .with_tag("report")
            .with_tag("Q4-2025")
            .with_meta("pages", "24"),
    )
    .await
    .ok();

    blob.put(
        "model_weights.bin",
        &vec![0u8; 4096], // placeholder bytes
        PutOptions::new()
            .with_content_type("application/octet-stream")
            .with_created_by("user:dave")
            .with_tag("ml-model")
            .with_meta("model", "embedding-v3")
            .with_meta("dimensions", "768"),
    )
    .await
    .ok();

    blob.put(
        "schema_backup.json",
        b"{\"tables\":[\"users\",\"purchases\",\"products\"],\"version\":3}",
        PutOptions::new()
            .with_content_type("application/json")
            .with_created_by("system:backup")
            .with_tag("backup")
            .with_tag("schema"),
    )
    .await
    .ok();

    blob.put(
        "logo.svg",
        b"<svg viewBox=\"0 0 100 100\"><circle cx=\"50\" cy=\"50\" r=\"40\"/></svg>",
        PutOptions::new()
            .with_content_type("image/svg+xml")
            .with_created_by("user:carol")
            .with_tag("branding"),
    )
    .await
    .ok();

    blob
}

async fn seed_checkpoint(blob: Arc<Mutex<BlobStore>>, store: &TensorStore) -> CheckpointManager {
    let config = CheckpointConfig::default()
        .with_max_checkpoints(10)
        .with_auto_checkpoint(true);

    let mgr = CheckpointManager::new(blob, config);

    mgr.create(Some("initial_seed"), store).await.ok();
    mgr.create(Some("post_migration_v2"), store).await.ok();
    mgr.create(Some("pre_deploy_release"), store).await.ok();

    mgr
}

fn demo_chain_status() -> Arc<ChainStatus> {
    Arc::new(ChainStatus {
        raft_state: "Leader".to_string(),
        current_term: 47,
        commit_index: 12_834,
        log_length: 12_901,
        leader_id: Some("node-alpha".to_string()),
        fast_path_rate: 0.942,
        heartbeat_success_rate: 0.997,
        heartbeat_successes: 48_250,
        heartbeat_failures: 145,
        quorum_checks: 16_100,
        quorum_lost_events: 3,
        leader_step_downs: 5,

        tx_started: 85_400,
        tx_committed: 81_130,
        tx_aborted: 2_560,
        tx_timed_out: 870,
        tx_conflicts: 840,
        tx_commit_rate: 0.95,
        tx_conflict_rate: 0.0098,
        tx_pending: 12,

        deadlocks_detected: 23,
        victims_aborted: 23,
        detection_cycles: 4_200,
        max_cycle_length: 5,
        deadlock_enabled: true,
    })
}

fn seed_metrics() -> Arc<ServerMetrics> {
    let provider = SdkMeterProvider::builder().build();
    let meter = provider.meter("neumann-preview");
    let metrics = Arc::new(ServerMetrics::new(meter));

    // Simulate some request history
    for _ in 0..150 {
        metrics.record_request("query", "execute", true, 45.0);
    }
    for _ in 0..12 {
        metrics.record_request("query", "execute", false, 120.0);
    }
    for _ in 0..3 {
        metrics.record_auth_failure("invalid_api_key");
    }
    for _ in 0..2 {
        metrics.record_rate_limited("user:bot", "search");
    }

    metrics
}
