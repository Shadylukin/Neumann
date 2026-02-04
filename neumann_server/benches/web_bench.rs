// SPDX-License-Identifier: MIT OR Apache-2.0
//! Benchmarks for the web admin UI performance.
// criterion_group! and criterion_main! macros generate items without documentation.
// This allow is required because we cannot add doc comments to macro-generated code.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::collections::HashMap;
use std::sync::Arc;

use graph_engine::GraphEngine;
use maud::Markup;
use relational_engine::{Column, ColumnType, RelationalEngine, Schema, Value};
use tensor_store::{ScalarValue, TensorValue};
use vector_engine::VectorEngine;

use neumann_server::web::templates::layout::{
    empty_state, engine_section, expandable_string, expandable_text, expandable_vector,
    format_number, layout, page_header, stat_card,
};
use neumann_server::web::NavItem;

fn setup_engines() -> (Arc<RelationalEngine>, Arc<VectorEngine>, Arc<GraphEngine>) {
    let relational = Arc::new(RelationalEngine::new());
    let vector = Arc::new(VectorEngine::new());
    let graph = Arc::new(GraphEngine::new());

    // Create test table with data
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("value", ColumnType::Float),
    ]);
    relational.create_table("test", schema).unwrap();

    for i in 0..1000 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("name".to_string(), Value::String(format!("item_{i}")));
        row.insert("value".to_string(), Value::Float(i as f64 * 1.5));
        relational.insert("test", row).unwrap();
    }

    // Create vector collection with data
    vector
        .create_collection(
            "docs",
            vector_engine::VectorCollectionConfig::default()
                .with_dimension(128)
                .with_metric(vector_engine::DistanceMetric::Cosine),
        )
        .unwrap();

    for i in 0..500 {
        let embedding: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect();
        let mut metadata = HashMap::new();
        metadata.insert(
            "title".to_string(),
            TensorValue::Scalar(ScalarValue::String(format!("Document {i}"))),
        );
        metadata.insert(
            "content".to_string(),
            TensorValue::Scalar(ScalarValue::String(
                "This is a sample document content that is reasonably long to simulate real data. "
                    .repeat(10),
            )),
        );
        vector
            .store_in_collection_with_metadata("docs", &format!("doc_{i}"), embedding, metadata)
            .unwrap();
    }

    // Create graph with nodes and edges
    for i in 0..200 {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            graph_engine::PropertyValue::String(format!("Node {i}")),
        );
        graph.create_node("Entity", props).unwrap();
    }

    for i in 0..150 {
        let _ = graph.create_edge(
            (i % 200) as u64,
            ((i + 1) % 200) as u64,
            "CONNECTS",
            HashMap::new(),
            true,
        );
    }

    (relational, vector, graph)
}

fn bench_template_rendering(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_rendering");

    // Benchmark layout rendering
    group.bench_function("layout_empty", |b| {
        b.iter(|| {
            let content = maud::html! { div { "Hello" } };
            let _: Markup = layout("Test", NavItem::Dashboard, content);
        });
    });

    // Benchmark stat card
    group.bench_function("stat_card", |b| {
        b.iter(|| {
            let _: Markup = stat_card("Tables", "24", "relational", "orange");
        });
    });

    // Benchmark page header
    group.bench_function("page_header", |b| {
        b.iter(|| {
            let _: Markup = page_header("Test Page", Some("Description here"));
        });
    });

    // Benchmark empty state
    group.bench_function("empty_state", |b| {
        b.iter(|| {
            let _: Markup = empty_state("No Data", "Nothing to show");
        });
    });

    // Benchmark engine section
    group.bench_function("engine_section", |b| {
        let items = vec![
            ("users".to_string(), "12,450".to_string()),
            ("orders".to_string(), "89,234".to_string()),
            ("products".to_string(), "3,456".to_string()),
        ];
        b.iter(|| {
            let _: Markup = engine_section("Relational", "orange", black_box(&items));
        });
    });

    // Benchmark format_number
    group.bench_function("format_number", |b| {
        b.iter(|| {
            let _ = format_number(black_box(1_234_567));
        });
    });

    group.finish();
}

fn bench_expandable_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("expandable_components");

    let short_text = "Short text content";
    let long_text = "This is a much longer text that will require expansion. ".repeat(20);

    // Benchmark expandable_text short (no expansion)
    group.bench_function("expandable_text_short", |b| {
        b.iter(|| {
            let _: Markup = expandable_text(black_box(short_text), 100, "text-white");
        });
    });

    // Benchmark expandable_text long (with expansion)
    group.bench_function("expandable_text_long", |b| {
        b.iter(|| {
            let _: Markup = expandable_text(black_box(&long_text), 100, "text-white");
        });
    });

    // Benchmark expandable_string short
    group.bench_function("expandable_string_short", |b| {
        b.iter(|| {
            let _: Markup = expandable_string(black_box(short_text), 100);
        });
    });

    // Benchmark expandable_string long
    group.bench_function("expandable_string_long", |b| {
        b.iter(|| {
            let _: Markup = expandable_string(black_box(&long_text), 100);
        });
    });

    // Benchmark expandable_vector
    let small_vec: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
    let large_vec: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

    group.bench_function("expandable_vector_small", |b| {
        b.iter(|| {
            let _: Markup = expandable_vector(black_box(&small_vec), 15);
        });
    });

    group.bench_function("expandable_vector_large", |b| {
        b.iter(|| {
            let _: Markup = expandable_vector(black_box(&large_vec), 15);
        });
    });

    group.finish();
}

fn bench_data_operations(c: &mut Criterion) {
    let (relational, vector, graph) = setup_engines();

    let mut group = c.benchmark_group("data_operations");
    group.throughput(Throughput::Elements(1));

    // Relational operations
    group.bench_function("relational_list_tables", |b| {
        b.iter(|| {
            let _tables = relational.list_tables();
        });
    });

    group.bench_function("relational_row_count", |b| {
        b.iter(|| {
            let _count = relational.row_count("test");
        });
    });

    group.bench_function("relational_get_schema", |b| {
        b.iter(|| {
            let _schema = relational.get_schema("test");
        });
    });

    group.bench_function("relational_select_50", |b| {
        b.iter(|| {
            let _rows =
                relational.select_with_limit("test", relational_engine::Condition::True, 50, 0);
        });
    });

    // Vector operations
    group.bench_function("vector_list_collections", |b| {
        b.iter(|| {
            let _collections = vector.list_collections();
        });
    });

    group.bench_function("vector_collection_count", |b| {
        b.iter(|| {
            let _count = vector.collection_count("docs");
        });
    });

    group.bench_function("vector_list_keys", |b| {
        b.iter(|| {
            let _keys = vector.list_collection_keys("docs");
        });
    });

    group.bench_function("vector_get_embedding", |b| {
        b.iter(|| {
            let _embedding = vector.get_from_collection("docs", "doc_0");
        });
    });

    group.bench_function("vector_search_k10", |b| {
        let query: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        b.iter(|| {
            let _results = vector.search_in_collection("docs", black_box(&query), 10);
        });
    });

    // Graph operations
    group.bench_function("graph_node_count", |b| {
        b.iter(|| {
            let _count = graph.node_count();
        });
    });

    group.bench_function("graph_edge_count", |b| {
        b.iter(|| {
            let _count = graph.edge_count();
        });
    });

    group.bench_function("graph_all_nodes_paginated", |b| {
        b.iter(|| {
            let _nodes = graph.all_nodes_paginated(graph_engine::Pagination::new(0, 50));
        });
    });

    group.bench_function("graph_find_path", |b| {
        b.iter(|| {
            let _path = graph.find_path(0, 10, None);
        });
    });

    group.bench_function("graph_pagerank", |b| {
        b.iter(|| {
            let _results = graph.pagerank(Some(graph_engine::PageRankConfig {
                damping: 0.85,
                max_iterations: 10,
                tolerance: 0.0001,
                direction: graph_engine::Direction::Outgoing,
                edge_type: None,
            }));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_template_rendering,
    bench_expandable_components,
    bench_data_operations
);
criterion_main!(benches);
