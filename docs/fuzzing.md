# Fuzzing

The project uses cargo-fuzz (libFuzzer-based) for coverage-guided fuzzing.

## Fuzz Targets

| Target | Module | What it tests |
| ------ | ------ | ------------- |
| `parser_parse` | neumann_parser | Statement parsing |
| `parser_parse_all` | neumann_parser | Multi-statement parsing |
| `parser_parse_expr` | neumann_parser | Expression parsing |
| `parser_tokenize` | neumann_parser | Lexer/tokenization |
| `compress_ids` | tensor_compress | Varint ID compression |
| `compress_rle` | tensor_compress | RLE encode/decode |
| `compress_snapshot` | tensor_compress | Snapshot serialization |
| `vault_cipher` | tensor_vault | AES-256-GCM roundtrip |
| `checkpoint_state` | tensor_checkpoint | Checkpoint bincode |
| `storage_sparse_vector` | tensor_store | Sparse vector roundtrip |
| `slab_entity_index` | tensor_store | EntityIndex operations |
| `consistent_hash` | tensor_store | Consistent hash partitioner |
| `tcp_framing` | tensor_chain | TCP wire protocol codec |
| `membership` | tensor_chain | Cluster config serialization |
| `relational_condition` | relational_engine | Condition evaluation |
| `relational_engine_ops` | relational_engine | Engine CRUD operations |
| `cache_eviction_scorer` | tensor_cache | Eviction strategy scoring |
| `cache_semantic_search` | tensor_cache | Semantic search with metrics |
| `cache_metric_roundtrip` | tensor_cache | Metric consistency |
| `archetype_registry` | tensor_store | ArchetypeRegistry bincode |
| `dtx_state_cleanup` | tensor_chain | Distributed tx cleanup |
| `error_hierarchy` | integration_tests | Error type hierarchy |

## Running Locally

```bash
# Install cargo-fuzz (requires nightly)
cargo install cargo-fuzz

# List available targets
cd fuzz && cargo +nightly fuzz list

# Run a specific target for 60 seconds
cargo +nightly fuzz run parser_parse -- -max_total_time=60

# Run without sanitizer (2x faster for safe Rust)
cargo +nightly fuzz run parser_parse --sanitizer none

# Reproduce a crash
cargo +nightly fuzz run parser_parse artifacts/parser_parse/crash-xxx
```

## Adding New Fuzz Targets

1. Create target file in `fuzz/fuzz_targets/<name>.rs`
2. Add `[[bin]]` entry to `fuzz/Cargo.toml`
3. Add seed corpus files to `fuzz/corpus/<name>/`
4. Update CI matrix in `.github/workflows/fuzz.yml`
