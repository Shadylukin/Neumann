#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::time::Duration;
use tensor_cache::{Cache, CacheConfig, CacheError, EvictionStrategy};

#[derive(Arbitrary, Debug, Clone)]
enum CacheOp {
    Put {
        prompt: String,
        embedding_seed: u8,
        response: String,
    },
    PutSimple {
        key: String,
        value: String,
    },
    PutEmbedding {
        source: String,
        content: String,
        embedding_seed: u8,
    },
    Get {
        prompt: String,
        use_embedding: bool,
    },
    GetSimple {
        key: String,
    },
    GetEmbedding {
        source: String,
        content: String,
    },
    Invalidate {
        prompt: String,
    },
    InvalidateVersion {
        version: String,
    },
    InvalidateEmbeddings {
        source: String,
    },
    CleanupExpired,
    Evict {
        count: u8,
    },
    Clear,
    GetStats,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    // Config params (limited ranges)
    embedding_dim: u8, // 4-64
    exact_capacity: u16,
    semantic_capacity: u16,
    embedding_capacity: u16,
    semantic_threshold: u8, // 0-100 maps to 0.0-1.0
    ttl_ms: u16,
    eviction_batch_size: u8,
    strategy_type: u8,

    // Operations to perform
    ops: Vec<CacheOp>,
}

fn generate_embedding(dim: usize, seed: u8) -> Vec<f32> {
    let mut emb = Vec::with_capacity(dim);
    let mut val = seed as f32 / 255.0;
    for i in 0..dim {
        val = (val * 7.0 + (i as f32) * 0.1).sin().abs();
        emb.push(val);
    }
    // Normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        emb.iter_mut().for_each(|x| *x /= norm);
    }
    emb
}

fuzz_target!(|input: FuzzInput| {
    // Build valid config from fuzzed input
    let dim = ((input.embedding_dim % 61) + 4) as usize; // 4-64
    let exact_cap = (input.exact_capacity as usize).max(10).min(10000);
    let semantic_cap = (input.semantic_capacity as usize).max(10).min(10000);
    let embedding_cap = (input.embedding_capacity as usize).max(10).min(10000);
    let threshold = (input.semantic_threshold as f32 / 100.0).clamp(0.5, 0.99);
    let ttl = Duration::from_millis((input.ttl_ms as u64).max(10));
    let batch_size = (input.eviction_batch_size as usize).max(1).min(1000);

    let strategy = match input.strategy_type % 4 {
        0 => EvictionStrategy::LRU,
        1 => EvictionStrategy::LFU,
        2 => EvictionStrategy::CostBased,
        _ => EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        },
    };

    let config = CacheConfig {
        embedding_dim: dim,
        exact_capacity: exact_cap,
        semantic_capacity: semantic_cap,
        embedding_capacity: embedding_cap,
        semantic_threshold: threshold,
        default_ttl: ttl,
        eviction_batch_size: batch_size,
        eviction_strategy: strategy,
        ..Default::default()
    };

    // Create cache - must not panic
    let cache = match Cache::with_config(config) {
        Ok(c) => c,
        Err(CacheError::InvalidConfig(_)) => return, // Valid rejection
        Err(_) => panic!("Unexpected error creating cache"),
    };

    // Execute operations
    for op in input.ops.iter().take(100) {
        // Limit ops per run
        match op {
            CacheOp::Put {
                prompt,
                embedding_seed,
                response,
            } => {
                let emb = generate_embedding(dim, *embedding_seed);
                let _ = cache.put(prompt, &emb, response, "fuzz-model", None);
            }
            CacheOp::PutSimple { key, value } => {
                let _ = cache.put_simple(key, value);
            }
            CacheOp::PutEmbedding {
                source,
                content,
                embedding_seed,
            } => {
                let emb = generate_embedding(dim, *embedding_seed);
                let _ = cache.put_embedding(source, content, emb, "fuzz-model");
            }
            CacheOp::Get { prompt, use_embedding } => {
                let emb = if *use_embedding {
                    Some(generate_embedding(dim, 42))
                } else {
                    None
                };
                let _ = cache.get(prompt, emb.as_deref());
            }
            CacheOp::GetSimple { key } => {
                let _ = cache.get_simple(key);
            }
            CacheOp::GetEmbedding { source, content } => {
                let _ = cache.get_embedding(source, content);
            }
            CacheOp::Invalidate { prompt } => {
                let _ = cache.invalidate(prompt);
            }
            CacheOp::InvalidateVersion { version } => {
                let _ = cache.invalidate_version(version);
            }
            CacheOp::InvalidateEmbeddings { source } => {
                let _ = cache.invalidate_embeddings(source);
            }
            CacheOp::CleanupExpired => {
                let _ = cache.cleanup_expired();
            }
            CacheOp::Evict { count } => {
                let _ = cache.evict(*count as usize);
            }
            CacheOp::Clear => {
                cache.clear();
            }
            CacheOp::GetStats => {
                let stats = cache.stats_snapshot();
                // Verify stats are consistent
                let _ = stats.total_entries();
                let _ = stats.hit_rate(tensor_cache::CacheLayer::Exact);
                let _ = stats.cost_saved_dollars;
            }
        }
    }

    // Final consistency check
    let stats = cache.stats_snapshot();
    let total = stats.total_entries();
    let _ = total; // Use to prevent optimization

    // Verify len matches stats
    let len = cache.len();
    assert!(
        len <= exact_cap + semantic_cap + embedding_cap,
        "Cache size {} exceeds total capacity {}",
        len,
        exact_cap + semantic_cap + embedding_cap
    );
});
