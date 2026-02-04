# @neumann/client

TypeScript SDK for the Neumann database.

## Installation

```bash
npm install @neumann/client
```

For browser support with gRPC-Web:

```bash
npm install @neumann/client grpc-web
```

## Quick Start

```typescript
import { NeumannClient, isRowsResult, rowToObject } from '@neumann/client';

// Connect to a remote server
const client = await NeumannClient.connect('localhost:9200', {
  apiKey: 'your-api-key',
  tls: true,
});

// Execute a query
const result = await client.execute('SELECT users');

if (isRowsResult(result)) {
  for (const row of result.rows) {
    console.log(rowToObject(row));
  }
}

// Close the connection
client.close();
```

## Features

- **Dual environment support**: Node.js (gRPC) and browser (gRPC-Web)
- **Type-safe**: Full TypeScript support with discriminated unions
- **Streaming**: Support for streaming query results
- **Batch queries**: Execute multiple queries efficiently
- **Error handling**: Typed error hierarchy with error codes

## Connection Options

### Node.js (gRPC)

```typescript
const client = await NeumannClient.connect('localhost:9200', {
  apiKey: 'your-api-key',
  tls: true,
  metadata: { 'custom-header': 'value' },
});
```

### Browser (gRPC-Web)

```typescript
const client = await NeumannClient.connectWeb('https://api.example.com', {
  apiKey: 'your-api-key',
});
```

## Query Execution

### Single Query

```typescript
const result = await client.execute('SELECT users WHERE age > 21');
```

### Streaming Query

```typescript
for await (const result of client.executeStream('SELECT large_table')) {
  if (isRowsResult(result)) {
    console.log(`Batch of ${result.rows.length} rows`);
  }
}
```

### Batch Queries

```typescript
const results = await client.executeBatch([
  'SELECT users',
  'SELECT orders',
  'SELECT products',
]);
```

## Result Types

Query results use discriminated unions for type safety:

```typescript
import {
  isRowsResult,
  isNodesResult,
  isEdgesResult,
  isPathsResult,
  isSimilarResult,
  isErrorResult,
} from '@neumann/client';

const result = await client.execute(query);

switch (result.type) {
  case 'rows':
    for (const row of result.rows) {
      console.log(rowToObject(row));
    }
    break;
  case 'nodes':
    for (const node of result.nodes) {
      console.log(nodeToObject(node));
    }
    break;
  case 'similar':
    for (const item of result.items) {
      console.log(`${item.key}: ${item.score}`);
    }
    break;
  case 'error':
    console.error(result.message);
    break;
}
```

## Value Types

The SDK provides type-safe value handling:

```typescript
import {
  intValue,
  stringValue,
  valueToNative,
  valueFromNative,
} from '@neumann/client';

// Create typed values
const age = intValue(25);
const name = stringValue('Alice');

// Convert to native JavaScript types
const nativeAge = valueToNative(age); // number: 25

// Convert from native JavaScript values
const value = valueFromNative(42); // { type: 'int', data: 42 }
```

## Error Handling

```typescript
import {
  NeumannError,
  ConnectionError,
  AuthenticationError,
  QueryError,
  ErrorCode,
} from '@neumann/client';

try {
  const result = await client.execute(query);
} catch (error) {
  if (error instanceof ConnectionError) {
    console.error('Connection failed:', error.message);
  } else if (error instanceof AuthenticationError) {
    console.error('Auth failed:', error.message);
  } else if (error instanceof QueryError) {
    console.error('Query failed:', error.message);
  } else if (error instanceof NeumannError) {
    console.error(`Error [${ErrorCode[error.code]}]:`, error.message);
  }
}
```

## API Reference

### NeumannClient

| Method | Description |
| ------ | ----------- |
| `connect(address, options)` | Connect via gRPC (Node.js) |
| `connectWeb(address, options)` | Connect via gRPC-Web (browser) |
| `execute(query, options)` | Execute a single query |
| `executeStream(query, options)` | Stream query results |
| `executeBatch(queries, options)` | Execute multiple queries |
| `close()` | Close the connection |
| `isConnected` | Check connection status |
| `clientMode` | Get client mode (remote/embedded) |

### ConnectOptions

| Option | Type | Description |
| ------ | ---- | ----------- |
| `apiKey` | `string` | API key for authentication |
| `tls` | `boolean` | Enable TLS encryption |
| `metadata` | `Record<string, string>` | Custom headers |

### QueryOptions

| Option | Type | Description |
| ------ | ---- | ----------- |
| `identity` | `string` | Identity for vault access |

## Requirements

- Node.js 18.0.0 or later
- TypeScript 5.0+ (for type definitions)

## License

MIT
