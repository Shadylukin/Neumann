# TypeScript SDK Quickstart

The Neumann TypeScript SDK provides a fully typed client for Node.js (gRPC) and
browser (gRPC-Web) environments.

## Installation

```bash
npm install @neumann/client
# or
yarn add @neumann/client
```

## Connect

### Node.js (gRPC)

```typescript
import { NeumannClient } from '@neumann/client';

const client = await NeumannClient.connect('localhost:9200', {
  apiKey: 'your-api-key',
  tls: false,
});
```

### Browser (gRPC-Web)

```typescript
const client = await NeumannClient.connectWeb('http://localhost:9200');
```

## Execute Queries

```typescript
// Single query
const result = await client.query('SELECT * FROM users WHERE age > 25');

// Batch queries
const results = await client.executeBatch([
  "INSERT INTO users VALUES (1, 'Alice', 30)",
  "INSERT INTO users VALUES (2, 'Bob', 25)",
]);

// Streaming results
for await (const chunk of client.executeStream('SELECT * FROM large_table')) {
  process(chunk);
}

// Paginated results
const page = await client.executePaginated('SELECT * FROM users', {
  pageSize: 100,
  countTotal: true,
});
console.log(`Total: ${page.totalCount}, Has more: ${page.hasMore}`);
```

## Handle Results

Results use discriminated unions with type guard functions:

```typescript
import {
  isRowsResult,
  isNodesResult,
  isSimilarResult,
  isCountResult,
  rowToObject,
} from '@neumann/client';

const result = await client.query('SELECT * FROM users');

if (isRowsResult(result)) {
  for (const row of result.rows) {
    const obj = rowToObject(row);
    console.log(obj.name, obj.age);
  }
}

if (isNodesResult(result)) {
  for (const node of result.nodes) {
    console.log(node.id, node.label, node.properties);
  }
}

if (isSimilarResult(result)) {
  for (const item of result.items) {
    console.log(`${item.key}: ${item.score.toFixed(4)}`);
  }
}

if (isCountResult(result)) {
  console.log(`Count: ${result.count}`);
}
```

### Result type guards

| Guard | Result Field | Description |
|-------|-------------|-------------|
| `isRowsResult` | `result.rows` | Relational query results |
| `isNodesResult` | `result.nodes` | Graph nodes |
| `isEdgesResult` | `result.edges` | Graph edges |
| `isSimilarResult` | `result.items` | Vector similarity results |
| `isCountResult` | `result.count` | Integer count |
| `isValueResult` | `result.value` | Single scalar value |
| `isErrorResult` | `result.error` | Error message |

## Transactions

```typescript
// Automatic commit/rollback
const result = await client.withTransaction(async (tx) => {
  await tx.execute("INSERT INTO users VALUES (1, 'Alice', 30)");
  await tx.execute("INSERT INTO users VALUES (2, 'Bob', 25)");
  return 'inserted';
});
// Transaction is committed on success, rolled back on error

// Manual control
const tx = client.beginTransaction();
await tx.begin();
await tx.execute("INSERT INTO users VALUES (3, 'Carol', 28)");
await tx.commit(); // or tx.rollback()
```

## Vector Operations

For dedicated vector operations, use `VectorClient`:

```typescript
import { VectorClient } from '@neumann/client';

const vectors = await VectorClient.connect('localhost:9200');

// Create a collection
await vectors.createCollection('documents', 384, 'cosine');

// Upsert points
await vectors.upsertPoints('documents', [
  { id: 'doc1', vector: [0.1, 0.2, ...], payload: { title: 'Hello' } },
  { id: 'doc2', vector: [0.3, 0.4, ...], payload: { title: 'World' } },
]);

// Query similar points
const results = await vectors.queryPoints('documents', [0.15, 0.25, ...], {
  limit: 10,
  scoreThreshold: 0.8,
  withPayload: true,
});

for (const point of results) {
  console.log(`${point.id}: ${point.score.toFixed(4)} - ${JSON.stringify(point.payload)}`);
}

// Scroll through all points
for await (const point of vectors.scrollAllPoints('documents')) {
  console.log(point.id);
}

// Manage collections
const names = await vectors.listCollections();
const info = await vectors.getCollection('documents');
const count = await vectors.countPoints('documents');

vectors.close();
```

## Blob Operations

Upload and download binary artifacts:

```typescript
import { BlobClient } from '@neumann/client';

// Upload from buffer
const result = await blob.uploadBlob('document.pdf', Buffer.from(data), {
  contentType: 'application/pdf',
  tags: ['quarterly', 'report'],
  linkedTo: ['entity-id'],
});
console.log(`Uploaded: ${result.artifactId}`);

// Download as buffer
const data = await blob.downloadBlobFull(result.artifactId);

// Stream download
for await (const chunk of blob.downloadBlob(result.artifactId)) {
  process(chunk);
}

// Metadata
const metadata = await blob.getBlobMetadata(result.artifactId);
console.log(`Size: ${metadata.size}, Type: ${metadata.contentType}`);
```

## Error Handling

```typescript
import {
  NeumannError,
  ConnectionError,
  AuthenticationError,
  NotFoundError,
  ParseError,
  ErrorCode,
} from '@neumann/client';

try {
  const result = await client.query('SELECT * FROM nonexistent');
} catch (err) {
  if (err instanceof NotFoundError) {
    console.log(`Not found: ${err.message}`);
  } else if (err instanceof ParseError) {
    console.log(`Syntax error: ${err.message}`);
  } else if (err instanceof ConnectionError) {
    console.log(`Connection failed: ${err.message}`);
  } else if (err instanceof NeumannError) {
    console.log(`Error [${ErrorCode[err.code]}]: ${err.message}`);
  }
}
```

## Configuration

```typescript
import {
  ClientConfig,
  mergeClientConfig,
  noRetryConfig,
  fastFailConfig,
  highLatencyConfig,
} from '@neumann/client';

const config: ClientConfig = {
  timeout: {
    defaultTimeoutS: 30,
    connectTimeoutS: 10,
  },
  retry: {
    maxAttempts: 3,
    initialBackoffMs: 100,
    maxBackoffMs: 10000,
    backoffMultiplier: 2.0,
  },
};

const client = await NeumannClient.connect('localhost:9200', { config });

// Preset configurations
const fast = fastFailConfig();        // 5s timeout, 1 attempt
const noRetry = noRetryConfig();      // Default timeout, 1 attempt
const highLat = highLatencyConfig();   // 120s timeout, 5 attempts
```

## Pagination

Iterate through large result sets:

```typescript
// Single page
const page = await client.executePaginated('SELECT * FROM users', {
  pageSize: 100,
  countTotal: true,
  cursorTtlSecs: 300,
});

// Iterate all pages
for await (const result of client.executeAllPages('SELECT * FROM users')) {
  if (isRowsResult(result)) {
    for (const row of result.rows) {
      process(rowToObject(row));
    }
  }
}

// Clean up cursor
if (page.nextCursor) {
  await client.closeCursor(page.nextCursor);
}
```

## Next Steps

- [Query Language Reference](../reference/query-language.md) -- All commands
- [Python SDK](python-quickstart.md) -- Python alternative
- [Architecture](../architecture/neumann-ts.md) -- SDK internals
