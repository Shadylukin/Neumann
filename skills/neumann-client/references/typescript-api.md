# TypeScript SDK Reference

## Installation

```bash
npm install @neumann/client
```

## NeumannClient

```typescript
import { NeumannClient, ConnectOptions } from '@neumann/client';

// Node.js (gRPC)
const client = await NeumannClient.connect('localhost:9200', {
  apiKey: 'your-api-key',
});

// Browser (gRPC-Web)
const client = await NeumannClient.connectWeb('https://api.example.com:9200');

// Execute
const result = await client.query('SELECT users');
const results = await client.executeBatch(['SELECT users', 'SELECT orders']);
const stream = await client.executeStream('SELECT users');
const page = await client.executePaginated('SELECT users', { pageSize: 50 });

client.close();
```

## Type Guards

Always use type guards to narrow results before accessing data.

```typescript
import {
  isRowsResult, isNodesResult, isEdgesResult, isSimilarResult,
  isCountResult, isValueResult, isErrorResult, isPathsResult,
  isTableListResult, isBlobResult, isBlobInfoResult, isPageRankResult,
  isCentralityResult, isCommunitiesResult, isPatternMatchResult,
  isAggregateResult, isBatchOperationResult, isChainQueryResult,
  isUnifiedResult, rowToObject, nodeToObject, edgeToObject,
} from '@neumann/client';

if (isRowsResult(result)) {
  for (const row of result.rows) {
    const obj = rowToObject(row);  // Record<string, unknown>
  }
} else if (isNodesResult(result)) {
  for (const node of result.nodes) {
    console.log(node.id, node.label, node.properties);
  }
} else if (isErrorResult(result)) {
  console.error(result.error.message, result.error.code);
}
```

## VectorClient

```typescript
import { VectorClient } from '@neumann/client';

const vc = await VectorClient.connect('localhost:9200');
await vc.createCollection('docs', 384, 'cosine');
await vc.upsertPoints('docs', [
  { id: 'doc1', vector: [0.1, 0.2, ...], payload: { title: 'Hello' } },
]);
const results = await vc.queryPoints('docs', queryVector, { limit: 10 });
const scroll = await vc.scrollAllPoints('docs', { limit: 100 });
vc.close();
```

## BlobClient

```typescript
import { BlobClient } from '@neumann/client';

const blobs = new BlobClient(channel);
const uploaded = await blobs.uploadBlob(buffer, { filename: 'report.pdf' });
const stream = await blobs.downloadBlob(artifactId);
const full = await blobs.downloadBlobFull(artifactId);
const meta = await blobs.getBlobMetadata(artifactId);
```

## Transactions

```typescript
// Auto-commit (recommended)
const result = await client.withTransaction(async (tx) => {
  await tx.execute("INSERT INTO users (name) VALUES ('Alice')");
  await tx.execute("INSERT INTO orders (user) VALUES ('Alice')");
  return 'done';
});

// Manual
const tx = await client.beginTransaction();
try {
  await tx.execute("INSERT INTO users (name) VALUES ('Bob')");
  await tx.commit();
} catch (e) {
  await tx.rollback();
}
```

## Pagination

```typescript
const page = await client.executePaginated('SELECT users', {
  pageSize: 50,
  countTotal: true,
});
// page.result, page.nextCursor, page.hasMore, page.totalCount

const allPages = await client.executeAllPages('SELECT users', { pageSize: 100 });
await client.closeCursor(cursorToken);
```

## Error Types

```typescript
import {
  NeumannError,         // Base: code, message
  ConnectionError,      // Server unreachable
  AuthenticationError,  // Bad API key
  PermissionDeniedError,// Insufficient privileges
  NotFoundError,        // Missing resource
  InvalidArgumentError, // Bad parameters
  ParseError,           // Query syntax
  QueryError,           // Execution failure
  InternalError,        // Server bug
  ErrorCode,            // Enum: 0-9
  errorFromCode,        // code + message -> typed error
} from '@neumann/client';
```

## Configuration Presets

```typescript
import {
  fastFailConfig,     // 2s connect, 5s query, no retry
  noRetryConfig,      // 30s timeouts, no retry
  highLatencyConfig,  // 30s connect, 120s query, 5 retries
  mergeClientConfig,  // merge partial config with defaults
} from '@neumann/client';

const client = await NeumannClient.connect('host:9200', fastFailConfig());
```
