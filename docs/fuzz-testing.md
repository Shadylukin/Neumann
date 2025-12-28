# Fuzz Testing

Neumann uses [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) for coverage-guided fuzzing with libFuzzer.

## Overview

Fuzzing complements unit tests by automatically discovering edge cases that humans miss. The fuzzer generates random inputs, mutates them based on code coverage feedback, and reports any crashes or assertion failures.

## Fuzz Targets

| Target | Module | Description |
|--------|--------|-------------|
| `parser_parse` | neumann_parser | Main statement parser entry point |
| `parser_parse_all` | neumann_parser | Multi-statement parsing (semicolon-separated) |
| `parser_parse_expr` | neumann_parser | Expression parser with Pratt parsing |
| `parser_tokenize` | neumann_parser | Lexer/tokenization |
| `compress_ids` | tensor_compress | Varint ID compression roundtrip |
| `compress_rle` | tensor_compress | Run-length encoding roundtrip |
| `compress_snapshot` | tensor_compress | CompressedSnapshot bincode serialization |
| `vault_cipher` | tensor_vault | AES-256-GCM encrypt/decrypt roundtrip |
| `checkpoint_state` | tensor_checkpoint | CheckpointState bincode serialization |
| `storage_sparse_vector` | tensor_store | SparseVector dense/sparse conversion |

## Quick Start

```bash
# Install cargo-fuzz (requires Rust nightly)
cargo install cargo-fuzz

# List available targets
cd fuzz && cargo +nightly fuzz list

# Run a target for 60 seconds
cargo +nightly fuzz run parser_parse -- -max_total_time=60

# Run without sanitizer (2x faster for safe Rust code)
cargo +nightly fuzz run parser_parse --sanitizer none

# Run indefinitely (Ctrl+C to stop)
cargo +nightly fuzz run parser_parse --sanitizer none
```

## Reproducing Crashes

When the fuzzer finds a crash, it saves the input to `fuzz/artifacts/<target>/`:

```bash
# Reproduce a specific crash
cargo +nightly fuzz run parser_parse artifacts/parser_parse/crash-abc123

# Minimize the crashing input
cargo +nightly fuzz tmin parser_parse artifacts/parser_parse/crash-abc123
```

## Corpus Management

The fuzzer maintains a corpus of interesting inputs in `fuzz/corpus/<target>/`. These are inputs that triggered new code coverage.

```bash
# Merge and deduplicate corpus
cargo +nightly fuzz cmin parser_parse

# Show corpus statistics
ls -la fuzz/corpus/parser_parse/ | wc -l
```

The corpus is checked into git so coverage improvements persist across runs.

## Adding New Fuzz Targets

1. **Create the target file** in `fuzz/fuzz_targets/<name>.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Your fuzzing logic here
    // Should never panic on valid OR invalid input
});
```

2. **Add to Cargo.toml**:

```toml
[[bin]]
name = "my_new_target"
path = "fuzz_targets/my_new_target.rs"
test = false
doc = false
bench = false
```

3. **Create seed corpus** in `fuzz/corpus/<name>/`:
   - Add representative valid inputs
   - Include edge cases (empty, minimal, maximal)

4. **Update CI** in `.github/workflows/fuzz.yml`:
   - Add target to the matrix

## Structured Fuzzing with Arbitrary

For complex input types, use the `arbitrary` crate:

```rust
#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MyInput {
    values: Vec<i64>,
    limit: u16,
}

fuzz_target!(|input: MyInput| {
    // input is a structured type, not raw bytes
});
```

## CI Integration

Fuzzing runs in CI via `.github/workflows/fuzz.yml`:

- **PR checks**: 30 seconds per target (quick sanity check)
- **Weekly scheduled**: 10 minutes per target (deep exploration)
- **Manual dispatch**: Configurable duration

Crash artifacts are uploaded as GitHub Actions artifacts.

## Performance Tips

1. **Disable sanitizer** for safe Rust code:
   ```bash
   cargo +nightly fuzz run target --sanitizer none
   ```
   This provides ~2x speedup since ASAN overhead is unnecessary for memory-safe code.

2. **Use multiple cores**:
   ```bash
   cargo +nightly fuzz run target --sanitizer none -j$(nproc)
   ```

3. **Limit input size** to avoid OOM:
   ```bash
   cargo +nightly fuzz run target -- -max_len=1024
   ```

4. **Run overnight** for deep coverage:
   ```bash
   cargo +nightly fuzz run target --sanitizer none -- -max_total_time=28800
   ```

## Interpreting Results

The fuzzer outputs statistics like:

```
#12345 NEW cov: 850 ft: 1200 corp: 150/5000b exec/s: 50000
```

- `#12345`: Total executions
- `NEW`: Found new coverage
- `cov: 850`: Unique code edges covered
- `ft: 1200`: Feature count (more granular coverage)
- `corp: 150/5000b`: Corpus size (150 inputs, 5KB total)
- `exec/s: 50000`: Executions per second

Good fuzzing shows:
- Steady growth in `cov` and `ft` early on
- High `exec/s` (50k+ for small targets)
- Corpus stabilizing after initial exploration

## Security Considerations

Fuzzing is particularly valuable for:

1. **Parsers** - Complex state machines with many edge cases
2. **Serialization** - Malformed input handling
3. **Cryptography** - Roundtrip correctness
4. **Compression** - Decoder robustness

All fuzz targets should handle malformed input gracefully (return errors, not panic).

## Bugs Found

The fuzzing infrastructure has discovered real bugs:

| Date | Target | Bug | Fix |
|------|--------|-----|-----|
| 2025-12-28 | compress_ids | varint_decode shift overflow (>10 continuation bytes) | Added shift bounds check |

## Resources

- [Rust Fuzz Book](https://rust-fuzz.github.io/book/)
- [cargo-fuzz Documentation](https://github.com/rust-fuzz/cargo-fuzz)
- [LibFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
