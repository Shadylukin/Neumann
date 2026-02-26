# gRPC API Reference

## Endpoint

Default port: **9200**. Protocol: gRPC over HTTP/2.

## Services

### QueryService (`neumann.v1.QueryService`)

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `Execute` | `QueryRequest` | `QueryResponse` | Single query |
| `ExecuteStream` | `QueryRequest` | `stream QueryResponseChunk` | Streaming results |
| `ExecuteBatch` | `BatchQueryRequest` | `BatchQueryResponse` | Multiple queries |
| `ExecutePaginated` | `PaginatedQueryRequest` | `PaginatedQueryResponse` | Cursor-based pagination |
| `CloseCursor` | `CloseCursorRequest` | `CloseCursorResponse` | Free cursor resources |

### BlobService (`neumann.v1.BlobService`)

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `Upload` | `stream BlobUploadRequest` | `BlobUploadResponse` | Upload blob chunks |
| `Download` | `BlobDownloadRequest` | `stream BlobDownloadChunk` | Download blob |
| `Delete` | `BlobDeleteRequest` | `BlobDeleteResponse` | Delete blob |
| `GetMetadata` | `BlobMetadataRequest` | `ArtifactInfo` | Get artifact info |

### Health (`neumann.v1.Health`)

| RPC | Request | Response |
|-----|---------|----------|
| `Check` | `HealthCheckRequest` | `HealthCheckResponse` |

## Authentication

Pass API key via gRPC metadata header: `x-api-key: <key>`.

## QueryResponse Structure

`QueryResponse.result` is a `oneof` with these variants:

`empty`, `value`, `count`, `ids`, `rows`, `nodes`, `edges`, `path`,
`similar`, `unified`, `table_list`, `blob`, `artifact_info`, `artifact_list`,
`blob_stats`, `checkpoint_list`, `chain`, `page_rank`, `centrality`,
`communities`, `constraints`, `aggregate`, `batch_operation`,
`graph_indexes`, `pattern_match`, `spatial`.

Optional `error` field carries `ErrorInfo` with `ErrorCode` enum.

## grpcurl Examples

```bash
# Health check
grpcurl -plaintext localhost:9200 neumann.v1.Health/Check

# Execute query
grpcurl -plaintext -d '{"query": "SELECT users"}' \
  localhost:9200 neumann.v1.QueryService/Execute

# With API key
grpcurl -plaintext \
  -H 'x-api-key: your-api-key' \
  -d '{"query": "SELECT users", "identity": "admin"}' \
  localhost:9200 neumann.v1.QueryService/Execute

# Batch queries
grpcurl -plaintext -d '{
  "queries": [
    {"query": "SELECT users"},
    {"query": "SHOW TABLES"}
  ]
}' localhost:9200 neumann.v1.QueryService/ExecuteBatch

# Paginated query
grpcurl -plaintext -d '{
  "query": "SELECT users",
  "page_size": 50,
  "count_total": true
}' localhost:9200 neumann.v1.QueryService/ExecutePaginated

# List services
grpcurl -plaintext localhost:9200 list
```

## TLS

For TLS-enabled servers, remove `-plaintext` and add certificate flags:

```bash
grpcurl -cacert ca.crt -cert client.crt -key client.key \
  server:9200 neumann.v1.Health/Check
```
