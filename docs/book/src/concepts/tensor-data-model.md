# Tensor Data Model

Neumann uses a unified tensor-based data model that represents all data types as mathematical tensors.

## Core Types

### TensorValue

The fundamental value type:

| Variant | Description | Example |
|---------|-------------|---------|
| `Scalar(ScalarValue)` | Single value | `42`, `"hello"`, `true` |
| `Vector(Vec<f32>)` | Dense embedding | `[0.1, 0.2, 0.3]` |
| `Pointer(String)` | Reference to entity | `"user_123"` |
| `Pointers(Vec<String>)` | Multiple references | `["a", "b", "c"]` |

### ScalarValue

Primitive values:

| Variant | Rust Type | Example |
|---------|-----------|---------|
| `Int(i64)` | 64-bit integer | `42` |
| `Float(f64)` | 64-bit float | `3.14` |
| `String(String)` | UTF-8 string | `"hello"` |
| `Bool(bool)` | Boolean | `true` |
| `Bytes(Vec<u8>)` | Binary data | `[0x01, 0x02]` |
| `Null` | Null value | `NULL` |

### TensorData

A map of field names to TensorValues:

```rust
// Conceptually: HashMap<String, TensorValue>
let user = TensorData::new()
    .with("id", TensorValue::Scalar(ScalarValue::Int(1)))
    .with("name", TensorValue::Scalar(ScalarValue::String("Alice".into())))
    .with("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3]));
```

## Sparse Vectors

For high-dimensional sparse data:

```rust
// Only stores non-zero values
let sparse = SparseVector::new(1000)  // 1000 dimensions
    .with_value(42, 0.5)
    .with_value(100, 0.3)
    .with_value(500, 0.8);
```

### Operations

| Operation | Description |
|-----------|-------------|
| `cosine_similarity` | Cosine distance between vectors |
| `euclidean_distance` | L2 distance |
| `dot_product` | Inner product |
| `weighted_average` | Blend multiple vectors |
| `project_orthogonal` | Remove component |

## Type Mapping

### Relational Engine

| SQL Type | TensorValue |
|----------|-------------|
| `INT` | `Scalar(Int)` |
| `FLOAT` | `Scalar(Float)` |
| `STRING` | `Scalar(String)` |
| `BOOL` | `Scalar(Bool)` |
| `VECTOR(n)` | `Vector` |

### Graph Engine

| Graph Element | TensorValue |
|---------------|-------------|
| Node ID | `Scalar(String)` |
| Edge target | `Pointer` |
| Properties | `TensorData` |

### Vector Engine

| Vector Type | TensorValue |
|-------------|-------------|
| Dense | `Vector` |
| Sparse | `SparseVector` (internal) |

## Storage Layout

Data is stored in TensorStore as key-value pairs:

```
Key: "users/1"
Value: TensorData {
    "id": Scalar(Int(1)),
    "name": Scalar(String("Alice")),
    "embedding": Vector([0.1, 0.2, ...])
}
```
