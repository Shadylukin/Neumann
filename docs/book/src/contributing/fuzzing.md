# Fuzzing

Neumann uses cargo-fuzz (libFuzzer-based) for coverage-guided fuzzing.

## Setup

```bash
# Install cargo-fuzz (requires nightly)
cargo install cargo-fuzz

# List available targets
cd fuzz && cargo +nightly fuzz list
```

## Running Fuzz Targets

```bash
# Run a specific target for 60 seconds
cargo +nightly fuzz run parser_parse -- -max_total_time=60

# Run without sanitizer (2x faster for safe Rust)
cargo +nightly fuzz run parser_parse --sanitizer none

# Reproduce a crash
cargo +nightly fuzz run parser_parse artifacts/parser_parse/crash-xxx
```

## Available Targets

| Target | Module | What it tests |
| --- | --- | --- |
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

## Adding a New Fuzz Target

1. Create target file in `fuzz/fuzz_targets/<name>.rs`:

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Your fuzzing code here
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = my_crate::parse(input);
    }
});
```

1. Add entry to `fuzz/Cargo.toml`:

```toml
[[bin]]
name = "my_target"
path = "fuzz_targets/my_target.rs"
test = false
doc = false
bench = false
```

1. Add seed corpus files to `fuzz/corpus/<name>/`:

```bash
mkdir -p fuzz/corpus/my_target
echo "valid input 1" > fuzz/corpus/my_target/seed1
echo "valid input 2" > fuzz/corpus/my_target/seed2
```

1. Update CI matrix in `.github/workflows/fuzz.yml`

## Structured Fuzzing

For complex input types, use `arbitrary`:

```rust
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct MyInput {
    field1: u32,
    field2: String,
}

fuzz_target!(|input: MyInput| {
    let _ = my_crate::process(&input);
});
```

## Investigating Crashes

```bash
# View crash input
xxd artifacts/my_target/crash-xxx

# Minimize crash
cargo +nightly fuzz tmin my_target artifacts/my_target/crash-xxx

# Debug
cargo +nightly fuzz run my_target artifacts/my_target/crash-xxx -- -verbosity=2
```

## CI Integration

Fuzz tests run in CI for 60 seconds per target. See
`.github/workflows/fuzz.yml`.

## Best Practices

1. **Add corpus seeds**: Real-world inputs help fuzzer find paths
2. **Use structured fuzzing**: For complex inputs
3. **Run locally**: Before pushing changes to fuzzed code
4. **Minimize crashes**: Smaller inputs are easier to debug
5. **Keep targets focused**: One functionality per target
