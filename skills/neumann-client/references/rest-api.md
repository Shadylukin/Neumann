# REST/HTTP API Reference

The primary Neumann API is gRPC on port 9200. REST access is available
through the gRPC-Web gateway when enabled in the server configuration.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/neumann.v1.QueryService/Execute` | Execute a query |
| `POST` | `/neumann.v1.QueryService/ExecuteBatch` | Batch queries |
| `POST` | `/neumann.v1.QueryService/ExecutePaginated` | Paginated query |
| `POST` | `/neumann.v1.BlobService/Upload` | Upload blob |
| `POST` | `/neumann.v1.BlobService/Download` | Download blob |
| `POST` | `/neumann.v1.Health/Check` | Health check |

## Request Format

Content-Type: `application/grpc-web+proto` or `application/grpc-web-text+proto`
(base64-encoded for text mode).

```bash
# Using xh (httpie-compatible)
xh POST https://api.example.com/neumann.v1.QueryService/Execute \
  Content-Type:application/json \
  x-api-key:your-api-key \
  query="SELECT users"
```

## Authentication

Pass the API key in the `x-api-key` HTTP header.

## Response Format

Responses follow the protobuf `QueryResponse` message structure, serialized
as JSON when using `application/json` content type via the gRPC-Web gateway.

## Browser Usage

The TypeScript SDK's `connectWeb()` method handles gRPC-Web framing
automatically. For direct REST calls, use the gRPC-Web text protocol
with base64 encoding.

```typescript
// Preferred: use the SDK
const client = await NeumannClient.connectWeb('https://api.example.com:9200');
```

## Notes

- Streaming RPCs (`ExecuteStream`, `BlobService/Download`) use server-sent
  events or chunked transfer encoding in gRPC-Web mode.
- Not all gRPC features are available via the REST gateway. For full
  functionality, use the native gRPC client SDKs.
