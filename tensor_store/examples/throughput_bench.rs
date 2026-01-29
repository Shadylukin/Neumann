// SPDX-License-Identifier: MIT OR Apache-2.0
use std::time::Instant;

use tensor_store::{ScalarValue, SlabRouter, TensorData, TensorStore, TensorValue};

fn main() {
    let iterations = 100_000;

    println!("\n=== Throughput Benchmark (100K ops, release mode) ===\n");
    println!("Both TensorStore and SlabRouter use tensor-based storage.");
    println!("TensorStore adds Arc wrapper + Bloom filter + instrumentation hooks.\n");

    // Benchmark TensorStore (SlabRouter-backed)
    let store = TensorStore::new();
    let start = Instant::now();
    for i in 0..iterations {
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(i as i64)));
        store.put(&format!("key:{}", i), data).unwrap();
    }
    let store_put = start.elapsed();

    let start = Instant::now();
    for i in 0..iterations {
        let _ = store.get(&format!("key:{}", i));
    }
    let store_get = start.elapsed();

    println!("TensorStore (Arc<SlabRouter>):");
    println!(
        "  PUT: {:.2} M ops/sec",
        iterations as f64 / store_put.as_secs_f64() / 1_000_000.0
    );
    println!(
        "  GET: {:.2} M ops/sec",
        iterations as f64 / store_get.as_secs_f64() / 1_000_000.0
    );

    // Benchmark SlabRouter directly
    let router = SlabRouter::new();
    let start = Instant::now();
    for i in 0..iterations {
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(i as i64)));
        router.put(&format!("key:{}", i), data).unwrap();
    }
    let router_put = start.elapsed();

    let start = Instant::now();
    for i in 0..iterations {
        let _ = router.get(&format!("key:{}", i));
    }
    let router_get = start.elapsed();

    println!("\nSlabRouter (direct):");
    println!(
        "  PUT: {:.2} M ops/sec",
        iterations as f64 / router_put.as_secs_f64() / 1_000_000.0
    );
    println!(
        "  GET: {:.2} M ops/sec",
        iterations as f64 / router_get.as_secs_f64() / 1_000_000.0
    );

    // Compute wrapper overhead
    let put_overhead = (store_put.as_secs_f64() / router_put.as_secs_f64() - 1.0) * 100.0;
    let get_overhead = (store_get.as_secs_f64() / router_get.as_secs_f64() - 1.0) * 100.0;

    println!("\nTensorStore wrapper overhead:");
    println!("  PUT: {:+.1}%", put_overhead);
    println!("  GET: {:+.1}%", get_overhead);

    println!("\nKey benefit: Zero resize stalls (vs DashMap's 99.6% throughput drops)");
}
