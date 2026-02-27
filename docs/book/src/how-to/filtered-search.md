# Run Filtered Similarity Searches

Step-by-step guide for combining metadata filters with vector similarity search
to narrow results without post-processing.

> **See Also:**
> [Vector Engine API Reference](../reference/api/vector-engine.md) |
> [Filtered Search Design](../explanation/filtered-search-design.md) |
> [Embeddings Search How-To](embeddings-search.md)

## Prerequisites

Store embeddings with metadata. Filters operate on metadata fields attached
during storage:

```rust
use vector_engine::{VectorEngine, EmbeddingInput};
use tensor_store::TensorValue;
use std::collections::HashMap;

let engine = VectorEngine::new();

// Store embeddings with metadata
let mut meta = HashMap::new();
meta.insert("category".to_string(), TensorValue::from("science"));
meta.insert("year".to_string(), TensorValue::from(2024i64));
meta.insert("author".to_string(), TensorValue::from("Alice"));

engine.store_embedding_with_metadata("doc1", vec![0.1, 0.2, 0.3], meta)?;
```

## Build Filter Conditions

Import the filter types:

```rust
use vector_engine::{FilterCondition, FilterValue};
```

### Simple Equality Filter

```rust
let filter = FilterCondition::Eq(
    "category".to_string(),
    FilterValue::String("science".to_string()),
);
```

### Comparison Filters

```rust
// Documents published after 2020
let filter = FilterCondition::Gt("year".to_string(), FilterValue::Int(2020));

// Price at most 100.0
let filter = FilterCondition::Le("price".to_string(), FilterValue::Float(100.0));
```

### Combine Conditions with AND / OR

```rust
// Science documents published after 2020
let filter = FilterCondition::Eq(
    "category".to_string(),
    FilterValue::String("science".to_string()),
).and(
    FilterCondition::Gt("year".to_string(), FilterValue::Int(2020)),
);

// Documents by Alice or Bob
let filter = FilterCondition::Eq(
    "author".to_string(),
    FilterValue::String("Alice".to_string()),
).or(
    FilterCondition::Eq(
        "author".to_string(),
        FilterValue::String("Bob".to_string()),
    ),
);
```

### Set Membership (IN)

```rust
let filter = FilterCondition::In(
    "status".to_string(),
    vec![
        FilterValue::String("active".to_string()),
        FilterValue::String("pending".to_string()),
    ],
);
```

### String Operations

```rust
// Title contains "rust"
let filter = FilterCondition::Contains(
    "title".to_string(),
    "rust".to_string(),
);

// Key starts with "doc:"
let filter = FilterCondition::StartsWith(
    "name".to_string(),
    "doc:".to_string(),
);
```

### Existence Check

```rust
// Only embeddings that have a "summary" metadata field
let filter = FilterCondition::Exists("summary".to_string());
```

### Not Equal / Negation

```rust
// Exclude deleted documents
let filter = FilterCondition::Ne(
    "status".to_string(),
    FilterValue::String("deleted".to_string()),
);
```

## Run Filtered Searches

### Auto Strategy (Recommended Default)

Pass `None` for the config to let the engine choose the best strategy:

```rust
let query = vec![0.1, 0.2, 0.3];
let filter = FilterCondition::Eq(
    "category".to_string(),
    FilterValue::String("science".to_string()),
);

let results = engine.search_similar_filtered(&query, 10, &filter, None)?;

for result in &results {
    println!("Key: {}, Score: {}", result.key, result.score);
}
```

### Explicit Pre-Filter (Best for Selective Filters)

Use when the filter matches a small fraction of embeddings (< 10%):

```rust
use vector_engine::FilteredSearchConfig;

let config = FilteredSearchConfig::pre_filter();
let results = engine.search_similar_filtered(&query, 10, &filter, Some(config))?;
```

### Explicit Post-Filter with Custom Oversample

Use when the filter matches most embeddings. Increase the oversample factor for
more selective filters:

```rust
// Oversample 5x: retrieve 50 candidates, filter down to top 10
let config = FilteredSearchConfig::post_filter().with_oversample(5);
let results = engine.search_similar_filtered(&query, 10, &filter, Some(config))?;
```

## Inspect Filter Behavior

Before running expensive searches, check how selective your filter is:

```rust
// Estimate selectivity (0.0 = matches nothing, 1.0 = matches all)
let selectivity = engine.estimate_filter_selectivity(&filter);
println!("Estimated selectivity: {:.1}%", selectivity * 100.0);

// Get exact count of matching embeddings
let matching = engine.count_matching(&filter);
println!("{} embeddings match the filter", matching);

// List all matching keys
let keys = engine.list_keys_matching(&filter);
```

## Filtered Search in Collections

Filters also work within named collections:

```rust
let filter = FilterCondition::Eq(
    "author".to_string(),
    FilterValue::String("Alice".to_string()),
);

let results = engine.search_filtered_in_collection(
    "documents",
    &query,
    10,
    &filter,
    None,  // auto strategy
)?;
```

See [Vector Collections How-To](vector-collections.md) for collection setup.

## Filter Condition Reference

| Condition | Description | Example |
| --- | --- | --- |
| `Eq(field, value)` | Equality | `category = "science"` |
| `Ne(field, value)` | Not equal | `status != "deleted"` |
| `Lt(field, value)` | Less than | `price < 100` |
| `Le(field, value)` | Less than or equal | `price <= 100` |
| `Gt(field, value)` | Greater than | `year > 2020` |
| `Ge(field, value)` | Greater than or equal | `year >= 2020` |
| `And(a, b)` | Logical AND | Combined conditions |
| `Or(a, b)` | Logical OR | Alternative conditions |
| `In(field, values)` | Value in list | `status IN ["active", "pending"]` |
| `Contains(field, substr)` | String contains | `title CONTAINS "rust"` |
| `StartsWith(field, prefix)` | String prefix | `name STARTS WITH "doc:"` |
| `Exists(field)` | Field exists | `HAS embedding` |
| `True` | Always matches | No filter |

## Related Pages

- [Filtered Search Design](../explanation/filtered-search-design.md) -- understand pre-filter vs post-filter tradeoffs
- [Embeddings Search How-To](embeddings-search.md) -- basic storage and search operations
- [Vector Collections How-To](vector-collections.md) -- organize embeddings into namespaces
- [Vector Engine API Reference](../reference/api/vector-engine.md) -- complete type tables
