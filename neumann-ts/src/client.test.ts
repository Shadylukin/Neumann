// SPDX-License-Identifier: MIT
import { describe, it, expect, vi } from 'vitest';
import {
  NeumannClient,
  convertProtoValue,
  convertProtoRow,
  convertProtoNode,
  convertProtoEdge,
  convertProtoPath,
  convertProtoSimilarItem,
  convertProtoArtifactInfo,
  convertProtoCheckpoint,
  convertProtoPageRankItem,
  convertProtoCentralityType,
  convertProtoCentralityItem,
  convertProtoCommunityItem,
  convertProtoCommunityMemberList,
  convertProtoPatternMatchBinding,
  convertProtoBindingEntry,
  convertProtoBindingValue,
  convertProtoPatternMatchStats,
  convertProtoConstraintItem,
  convertProtoAggregateValue,
  convertProtoChainResult,
  convertProtoUnifiedItem,
} from './client.js';
import {
  ConnectionError,
  NeumannError,
  AuthenticationError,
  PermissionDeniedError,
  NotFoundError,
  InvalidArgumentError,
  ParseError,
  QueryError,
  InternalError,
  ErrorCode,
  errorFromCode,
} from './types/errors.js';
import type { Row, Node, Edge } from './types/query-result.js';
import {
  nullValue,
  intValue,
  floatValue,
  stringValue,
  boolValue,
  bytesValue,
  valueToNative,
  valueFromNative,
} from './types/value.js';
import {
  isEmptyResult,
  isRowsResult,
  isNodesResult,
  isEdgesResult,
  isPathsResult,
  isSimilarResult,
  isErrorResult,
  isValueResult,
  isCountResult,
  isIdsResult,
  isTableListResult,
  isBlobResult,
  isBlobInfoResult,
  isArtifactListResult,
  isBlobStatsResult,
  isCheckpointListResult,
  isPageRankResult,
  isCentralityResult,
  isCommunitiesResult,
  isPatternMatchResult,
  isConstraintsResult,
  isAggregateResult,
  isBatchOperationResult,
  isGraphIndexesResult,
  isChainQueryResult,
  isUnifiedResult,
  rowToObject,
  nodeToObject,
  edgeToObject,
  copyRowValues,
  copyNodeProperties,
  copyEdgeProperties,
  copySimilarItemMetadata,
  copyUnifiedItemFields,
} from './types/query-result.js';
import {
  MAX_STRING_LENGTH,
  MAX_BYTES_LENGTH,
  validateIntValue,
  validateFloatValue,
  validateStringValue,
  validateBytesValue,
  safeIdToString,
  safeIdsToStrings,
} from './types/validation.js';

// Mock execute callback response
const mockExecuteResponse = { empty: {} };
const mockExecuteStreamResponses = [
  { row: { row: { columns: [{ name: 'x', value: { intValue: 1 } }] } } },
  { isFinal: true },
];
const mockBatchResponse = { results: [{ empty: {} }, { empty: {} }, { empty: {} }] };

// Mock paginated response
const mockPaginatedResponse = {
  result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: 1 } }] }] } },
  nextCursor: 'cursor123',
  hasMore: true,
  pageSize: 10,
  totalCount: 100,
};

// Mock close cursor response
const mockCloseCursorResponse = { success: true };

// Mock gRPC client
const mockGrpcClient = {
  Execute: vi.fn((request: unknown, metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
    callback(null, mockExecuteResponse);
  }),
  ExecuteStream: vi.fn(() => {
    let index = 0;
    return {
      [Symbol.asyncIterator]: () => ({
        next(): Promise<{ value: unknown; done: boolean }> {
          if (index < mockExecuteStreamResponses.length) {
            return Promise.resolve({ value: mockExecuteStreamResponses[index++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      }),
    };
  }),
  ExecuteBatch: vi.fn((request: unknown, metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
    callback(null, mockBatchResponse);
  }),
  ExecutePaginated: vi.fn(
    (request: unknown, metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
      callback(null, mockPaginatedResponse);
    }
  ),
  CloseCursor: vi.fn(
    (request: unknown, metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
      callback(null, mockCloseCursorResponse);
    }
  ),
};

// Mock grpc.ts module
vi.mock('./grpc.js', () => ({
  loadProto: vi.fn().mockResolvedValue({}),
  getQueryServiceClient: vi.fn(() => mockGrpcClient),
}));

// Mock @grpc/grpc-js
vi.mock('@grpc/grpc-js', () => {
  class MockMetadata {
    private _data: Map<string, string> = new Map();
    set(key: string, value: string): void {
      this._data.set(key, value);
    }
    get(key: string): string | undefined {
      return this._data.get(key);
    }
  }
  return {
    credentials: {
      createSsl: vi.fn(() => ({})),
      createInsecure: vi.fn(() => ({})),
    },
    Metadata: MockMetadata,
  };
});

// Mock grpc-web
vi.mock('grpc-web', () => {
  class MockGrpcWebClientBase {
    constructor() {}
  }
  return {
    GrpcWebClientBase: MockGrpcWebClientBase,
  };
});

describe('NeumannClient', () => {
  describe('query', () => {
    it('should execute a query using the query method', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.query('SELECT * FROM users');
      expect(result.type).toBe('empty');
      client.close();
    });

    it('should pass options to query', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.query('VAULT GET secret', { identity: 'alice' });
      expect(result.type).toBe('empty');
      client.close();
    });
  });

  describe('connect', () => {
    it('should create a connected client', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      expect(client.isConnected).toBe(true);
      expect(client.clientMode).toBe('remote');
      client.close();
    });

    it('should support TLS connections', async () => {
      const client = await NeumannClient.connect('localhost:50051', { tls: true });
      expect(client.isConnected).toBe(true);
      client.close();
    });

    it('should accept API key', async () => {
      const client = await NeumannClient.connect('localhost:50051', {
        apiKey: 'test-api-key',
      });
      expect(client.isConnected).toBe(true);
      client.close();
    });

    it('should accept custom metadata', async () => {
      const client = await NeumannClient.connect('localhost:50051', {
        metadata: { 'x-custom-header': 'value' },
      });
      expect(client.isConnected).toBe(true);
      client.close();
    });

    it('should throw ConnectionError when gRPC fails', async () => {
      // Temporarily make loadProto throw
      const grpcModule = await import('./grpc.js');
      const originalLoadProto = grpcModule.loadProto;
      (grpcModule as { loadProto: typeof originalLoadProto }).loadProto = vi.fn().mockRejectedValue(
        new Error('Connection refused')
      );

      await expect(NeumannClient.connect('localhost:50051')).rejects.toThrow(ConnectionError);

      // Restore
      (grpcModule as { loadProto: typeof originalLoadProto }).loadProto = originalLoadProto;
    });
  });

  describe('connectWeb', () => {
    it('should create a connected gRPC-Web client', async () => {
      const client = await NeumannClient.connectWeb('http://localhost:8080');
      expect(client.isConnected).toBe(true);
      expect(client.clientMode).toBe('remote');
      client.close();
    });

    it('should accept API key for web client', async () => {
      const client = await NeumannClient.connectWeb('http://localhost:8080', {
        apiKey: 'test-api-key',
      });
      expect(client.isConnected).toBe(true);
      client.close();
    });

    it('should throw ConnectionError when gRPC-Web fails', async () => {
      // Temporarily make the GrpcWebClientBase constructor throw
      const grpcWeb = await import('grpc-web');
      const originalBase = grpcWeb.GrpcWebClientBase;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (grpcWeb as Record<string, unknown>).GrpcWebClientBase = vi.fn().mockImplementation(() => {
        throw new Error('gRPC-Web initialization failed');
      });

      await expect(NeumannClient.connectWeb('http://localhost:8080')).rejects.toThrow(ConnectionError);

      // Restore
      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (grpcWeb as Record<string, unknown>).GrpcWebClientBase = originalBase;
    });
  });

  describe('close', () => {
    it('should disconnect the client', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      expect(client.isConnected).toBe(true);
      client.close();
      expect(client.isConnected).toBe(false);
    });
  });

  describe('execute', () => {
    it('should throw if not connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      client.close();
      await expect(client.execute('SELECT * FROM users')).rejects.toThrow(ConnectionError);
    });

    it('should execute a query when connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.execute('SELECT * FROM users');
      expect(result.type).toBe('empty');
      client.close();
    });

    it('should pass identity option', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.execute('VAULT GET secret', { identity: 'alice' });
      expect(result.type).toBe('empty');
      client.close();
    });
  });

  describe('executeStream', () => {
    it('should throw if not connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      client.close();
      await expect(async () => {
        for await (const _result of client.executeStream('SELECT * FROM users')) {
          // Should throw before yielding
        }
      }).rejects.toThrow(ConnectionError);
    });

    it('should yield results when connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const results: unknown[] = [];
      for await (const result of client.executeStream('SELECT * FROM users')) {
        results.push(result);
      }
      expect(results.length).toBeGreaterThan(0);
      client.close();
    });

    it('should throw on stream error', async () => {
      const errorResponses = [
        { error: { code: 1, message: 'Stream error' } },
      ];
      let index = 0;
      mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
        [Symbol.asyncIterator]: () => ({
          next(): Promise<{ value: unknown; done: boolean }> {
            if (index < errorResponses.length) {
              return Promise.resolve({ value: errorResponses[index++], done: false });
            }
            return Promise.resolve({ value: undefined, done: true });
          },
        }),
      }));

      const client = await NeumannClient.connect('localhost:50051');
      await expect(async () => {
        for await (const _result of client.executeStream('INVALID')) {
          // Should throw on error chunk
        }
      }).rejects.toThrow(InvalidArgumentError);
      client.close();
    });
  });

  describe('executeBatch', () => {
    it('should throw if not connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      client.close();
      await expect(client.executeBatch(['SELECT 1', 'SELECT 2'])).rejects.toThrow(
        ConnectionError
      );
    });

    it('should execute batch when connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const results = await client.executeBatch(['SELECT 1', 'SELECT 2', 'SELECT 3']);
      expect(results).toHaveLength(3);
      client.close();
    });

    it('should handle gRPC error in batch', async () => {
      mockGrpcClient.ExecuteBatch.mockImplementationOnce(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callback({ code: 3, details: 'Batch error' }, null);
        }
      );
      const client = await NeumannClient.connect('localhost:50051');
      await expect(client.executeBatch(['SELECT 1'])).rejects.toThrow(InvalidArgumentError);
      client.close();
    });
  });
});

describe('Proto Conversion Functions', () => {
  describe('convertProtoValue', () => {
    it('should convert null values', () => {
      const result = convertProtoValue(null);
      expect(result.type).toBe('null');
      expect(result.data).toBeNull();
    });

    it('should convert undefined values', () => {
      const result = convertProtoValue(undefined);
      expect(result.type).toBe('null');
    });

    it('should convert nullValue field', () => {
      const result = convertProtoValue({ nullValue: true });
      expect(result.type).toBe('null');
    });

    it('should convert int values', () => {
      const result = convertProtoValue({ intValue: 42 });
      expect(result.type).toBe('int');
      expect(result.data).toBe(42);
    });

    it('should convert float values', () => {
      const result = convertProtoValue({ floatValue: 3.14 });
      expect(result.type).toBe('float');
      expect(result.data).toBe(3.14);
    });

    it('should convert string values', () => {
      const result = convertProtoValue({ stringValue: 'hello' });
      expect(result.type).toBe('string');
      expect(result.data).toBe('hello');
    });

    it('should convert bool values', () => {
      const result = convertProtoValue({ boolValue: true });
      expect(result.type).toBe('bool');
      expect(result.data).toBe(true);
    });

    it('should convert bytes values', () => {
      const bytes = new Uint8Array([1, 2, 3]);
      const result = convertProtoValue({ bytesValue: bytes });
      expect(result.type).toBe('bytes');
      expect(result.data).toEqual(bytes);
    });

    it('should default to null for unknown types', () => {
      const result = convertProtoValue({ unknownField: 'value' });
      expect(result.type).toBe('null');
    });
  });

  describe('convertProtoRow', () => {
    it('should convert empty row', () => {
      const result = convertProtoRow({});
      expect(result.values.size).toBe(0);
    });

    it('should convert row with columns', () => {
      const result = convertProtoRow({
        columns: [
          { name: 'id', value: { intValue: 1 } },
          { name: 'name', value: { stringValue: 'Alice' } },
        ],
      });
      expect(result.values.size).toBe(2);
      expect(result.values.get('id')?.data).toBe(1);
      expect(result.values.get('name')?.data).toBe('Alice');
    });
  });

  describe('convertProtoNode', () => {
    it('should convert node with properties', () => {
      const result = convertProtoNode({
        id: 'node-1',
        label: 'Person',
        properties: [{ name: 'age', value: { intValue: 30 } }],
      });
      expect(result.id).toBe('node-1');
      expect(result.label).toBe('Person');
      expect(result.properties.get('age')?.data).toBe(30);
    });

    it('should convert node without properties', () => {
      const result = convertProtoNode({
        id: 'node-2',
        label: 'Item',
      });
      expect(result.id).toBe('node-2');
      expect(result.label).toBe('Item');
      expect(result.properties.size).toBe(0);
    });
  });

  describe('convertProtoEdge', () => {
    it('should convert edge with properties', () => {
      const result = convertProtoEdge({
        id: 'edge-1',
        edgeType: 'KNOWS',
        sourceId: 'node-1',
        targetId: 'node-2',
        properties: [{ name: 'since', value: { intValue: 2020 } }],
      });
      expect(result.id).toBe('edge-1');
      expect(result.edgeType).toBe('KNOWS');
      expect(result.source).toBe('node-1');
      expect(result.target).toBe('node-2');
      expect(result.properties.get('since')?.data).toBe(2020);
    });
  });

  describe('convertProtoPath', () => {
    it('should convert empty path', () => {
      const result = convertProtoPath({});
      expect(result.segments).toHaveLength(0);
    });

    it('should convert path with segments', () => {
      const result = convertProtoPath({
        segments: [
          {
            node: { id: 'n1', label: 'A', properties: [] },
            edge: {
              id: 'e1',
              edgeType: 'LINK',
              sourceId: 'n1',
              targetId: 'n2',
              properties: [],
            },
          },
          {
            node: { id: 'n2', label: 'B', properties: [] },
          },
        ],
      });
      expect(result.segments).toHaveLength(2);
      expect(result.segments[0]?.node.id).toBe('n1');
      expect(result.segments[0]?.edge?.id).toBe('e1');
      expect(result.segments[1]?.node.id).toBe('n2');
      expect(result.segments[1]?.edge).toBeUndefined();
    });
  });

  describe('convertProtoSimilarItem', () => {
    it('should convert similar item without metadata', () => {
      const result = convertProtoSimilarItem({
        key: 'item-1',
        score: 0.95,
      });
      expect(result.key).toBe('item-1');
      expect(result.score).toBe(0.95);
      expect(result.metadata).toBeUndefined();
    });

    it('should convert similar item with metadata', () => {
      const result = convertProtoSimilarItem({
        key: 'item-2',
        score: 0.85,
        metadata: [{ name: 'category', value: { stringValue: 'tech' } }],
      });
      expect(result.key).toBe('item-2');
      expect(result.score).toBe(0.85);
      expect(result.metadata?.get('category')?.data).toBe('tech');
    });
  });

  describe('convertProtoArtifactInfo', () => {
    it('should convert artifact info', () => {
      const result = convertProtoArtifactInfo({
        artifactId: 'art-123',
        filename: 'test.txt',
        size: 1024,
        checksum: 'abc123',
        contentType: 'text/plain',
        createdAt: 1700000000,
        tags: ['test', 'sample'],
      });
      expect(result.artifactId).toBe('art-123');
      expect(result.filename).toBe('test.txt');
      expect(result.size).toBe(1024);
      expect(result.checksum).toBe('abc123');
      expect(result.contentType).toBe('text/plain');
      expect(result.createdAt).toBe(1700000000);
      expect(result.tags).toEqual(['test', 'sample']);
    });

    it('should handle missing tags', () => {
      const result = convertProtoArtifactInfo({
        artifactId: 'art-456',
        filename: 'data.bin',
        size: 2048,
        checksum: 'def456',
        contentType: 'application/octet-stream',
        createdAt: 1700000001,
      });
      expect(result.tags).toEqual([]);
    });
  });
});

describe('Value Functions', () => {
  describe('nullValue', () => {
    it('should create null value', () => {
      const v = nullValue();
      expect(v.type).toBe('null');
      expect(v.data).toBeNull();
    });
  });

  describe('intValue', () => {
    it('should create int value', () => {
      const v = intValue(42);
      expect(v.type).toBe('int');
      expect(v.data).toBe(42);
    });

    it('should floor float to int', () => {
      const v = intValue(42.9);
      expect(v.data).toBe(42);
    });
  });

  describe('floatValue', () => {
    it('should create float value', () => {
      const v = floatValue(3.14159);
      expect(v.type).toBe('float');
      expect(v.data).toBe(3.14159);
    });
  });

  describe('stringValue', () => {
    it('should create string value', () => {
      const v = stringValue('hello world');
      expect(v.type).toBe('string');
      expect(v.data).toBe('hello world');
    });
  });

  describe('boolValue', () => {
    it('should create bool value true', () => {
      const v = boolValue(true);
      expect(v.type).toBe('bool');
      expect(v.data).toBe(true);
    });

    it('should create bool value false', () => {
      const v = boolValue(false);
      expect(v.type).toBe('bool');
      expect(v.data).toBe(false);
    });
  });

  describe('bytesValue', () => {
    it('should create bytes value', () => {
      const bytes = new Uint8Array([0x01, 0x02, 0x03]);
      const v = bytesValue(bytes);
      expect(v.type).toBe('bytes');
      expect(v.data).toEqual(bytes);
    });
  });

  describe('valueToNative', () => {
    it('should extract native value', () => {
      expect(valueToNative(intValue(42))).toBe(42);
      expect(valueToNative(stringValue('test'))).toBe('test');
      expect(valueToNative(nullValue())).toBeNull();
    });
  });

  describe('valueFromNative', () => {
    it('should convert null', () => {
      expect(valueFromNative(null).type).toBe('null');
    });

    it('should convert undefined', () => {
      expect(valueFromNative(undefined).type).toBe('null');
    });

    it('should convert boolean', () => {
      const v = valueFromNative(true);
      expect(v.type).toBe('bool');
      expect(v.data).toBe(true);
    });

    it('should convert integer', () => {
      const v = valueFromNative(42);
      expect(v.type).toBe('int');
      expect(v.data).toBe(42);
    });

    it('should convert float', () => {
      const v = valueFromNative(3.14);
      expect(v.type).toBe('float');
      expect(v.data).toBe(3.14);
    });

    it('should convert string', () => {
      const v = valueFromNative('hello');
      expect(v.type).toBe('string');
      expect(v.data).toBe('hello');
    });

    it('should convert Uint8Array', () => {
      const bytes = new Uint8Array([1, 2, 3]);
      const v = valueFromNative(bytes);
      expect(v.type).toBe('bytes');
      expect(v.data).toEqual(bytes);
    });

    it('should convert other objects to string', () => {
      const v = valueFromNative({ foo: 'bar' });
      expect(v.type).toBe('string');
    });
  });
});

describe('Error Classes', () => {
  describe('NeumannError', () => {
    it('should create with message and code', () => {
      const err = new NeumannError('test error', ErrorCode.INTERNAL);
      expect(err.message).toBe('test error');
      expect(err.code).toBe(ErrorCode.INTERNAL);
      expect(err.name).toBe('NeumannError');
    });

    it('should default to UNKNOWN code', () => {
      const err = new NeumannError('test');
      expect(err.code).toBe(ErrorCode.UNKNOWN);
    });

    it('should format toString correctly', () => {
      const err = new NeumannError('test error', ErrorCode.INTERNAL);
      expect(err.toString()).toContain('INTERNAL');
      expect(err.toString()).toContain('test error');
    });
  });

  describe('ConnectionError', () => {
    it('should create with UNAVAILABLE code', () => {
      const err = new ConnectionError('connection failed');
      expect(err.code).toBe(ErrorCode.UNAVAILABLE);
      expect(err.name).toBe('ConnectionError');
    });
  });

  describe('AuthenticationError', () => {
    it('should create with UNAUTHENTICATED code', () => {
      const err = new AuthenticationError();
      expect(err.code).toBe(ErrorCode.UNAUTHENTICATED);
      expect(err.name).toBe('AuthenticationError');
    });

    it('should accept custom message', () => {
      const err = new AuthenticationError('invalid token');
      expect(err.message).toBe('invalid token');
    });
  });

  describe('PermissionDeniedError', () => {
    it('should create with PERMISSION_DENIED code', () => {
      const err = new PermissionDeniedError();
      expect(err.code).toBe(ErrorCode.PERMISSION_DENIED);
      expect(err.name).toBe('PermissionDeniedError');
    });
  });

  describe('NotFoundError', () => {
    it('should create with NOT_FOUND code', () => {
      const err = new NotFoundError('user');
      expect(err.code).toBe(ErrorCode.NOT_FOUND);
      expect(err.message).toContain('user');
      expect(err.name).toBe('NotFoundError');
    });
  });

  describe('InvalidArgumentError', () => {
    it('should create with INVALID_ARGUMENT code', () => {
      const err = new InvalidArgumentError('bad input');
      expect(err.code).toBe(ErrorCode.INVALID_ARGUMENT);
      expect(err.name).toBe('InvalidArgumentError');
    });
  });

  describe('ParseError', () => {
    it('should create with PARSE_ERROR code', () => {
      const err = new ParseError('syntax error');
      expect(err.code).toBe(ErrorCode.PARSE_ERROR);
      expect(err.name).toBe('ParseError');
    });
  });

  describe('QueryError', () => {
    it('should create with QUERY_ERROR code', () => {
      const err = new QueryError('query failed');
      expect(err.code).toBe(ErrorCode.QUERY_ERROR);
      expect(err.name).toBe('QueryError');
    });
  });

  describe('InternalError', () => {
    it('should create with INTERNAL code', () => {
      const err = new InternalError();
      expect(err.code).toBe(ErrorCode.INTERNAL);
      expect(err.name).toBe('InternalError');
    });
  });

  describe('errorFromCode', () => {
    it('should create InvalidArgumentError for INVALID_ARGUMENT', () => {
      const err = errorFromCode(ErrorCode.INVALID_ARGUMENT, 'bad input');
      expect(err).toBeInstanceOf(InvalidArgumentError);
    });

    it('should create NotFoundError for NOT_FOUND', () => {
      const err = errorFromCode(ErrorCode.NOT_FOUND, 'resource');
      expect(err).toBeInstanceOf(NotFoundError);
    });

    it('should create PermissionDeniedError for PERMISSION_DENIED', () => {
      const err = errorFromCode(ErrorCode.PERMISSION_DENIED, 'denied');
      expect(err).toBeInstanceOf(PermissionDeniedError);
    });

    it('should create AuthenticationError for UNAUTHENTICATED', () => {
      const err = errorFromCode(ErrorCode.UNAUTHENTICATED, 'auth failed');
      expect(err).toBeInstanceOf(AuthenticationError);
    });

    it('should create ConnectionError for UNAVAILABLE', () => {
      const err = errorFromCode(ErrorCode.UNAVAILABLE, 'unavailable');
      expect(err).toBeInstanceOf(ConnectionError);
    });

    it('should create InternalError for INTERNAL', () => {
      const err = errorFromCode(ErrorCode.INTERNAL, 'internal error');
      expect(err).toBeInstanceOf(InternalError);
    });

    it('should create ParseError for PARSE_ERROR', () => {
      const err = errorFromCode(ErrorCode.PARSE_ERROR, 'parse error');
      expect(err).toBeInstanceOf(ParseError);
    });

    it('should create QueryError for QUERY_ERROR', () => {
      const err = errorFromCode(ErrorCode.QUERY_ERROR, 'query error');
      expect(err).toBeInstanceOf(QueryError);
    });

    it('should create NeumannError for UNKNOWN', () => {
      const err = errorFromCode(ErrorCode.UNKNOWN, 'unknown');
      expect(err).toBeInstanceOf(NeumannError);
      expect(err.constructor.name).toBe('NeumannError');
    });

    it('should accept numeric code', () => {
      const err = errorFromCode(7, 'internal');
      expect(err).toBeInstanceOf(InternalError);
    });
  });
});

describe('Type Guards', () => {
  describe('isEmptyResult', () => {
    it('should return true for empty result', () => {
      expect(isEmptyResult({ type: 'empty' })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isEmptyResult({ type: 'rows', rows: [] })).toBe(false);
    });
  });

  describe('isRowsResult', () => {
    it('should return true for rows result', () => {
      expect(isRowsResult({ type: 'rows', rows: [] })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isRowsResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isNodesResult', () => {
    it('should return true for nodes result', () => {
      expect(isNodesResult({ type: 'nodes', nodes: [] })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isNodesResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isEdgesResult', () => {
    it('should return true for edges result', () => {
      expect(isEdgesResult({ type: 'edges', edges: [] })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isEdgesResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isPathsResult', () => {
    it('should return true for paths result', () => {
      expect(isPathsResult({ type: 'paths', paths: [] })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isPathsResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isSimilarResult', () => {
    it('should return true for similar result', () => {
      expect(isSimilarResult({ type: 'similar', items: [] })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isSimilarResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isErrorResult', () => {
    it('should return true for error result', () => {
      expect(isErrorResult({ type: 'error', code: 1, message: 'err' })).toBe(true);
    });

    it('should return false for other types', () => {
      expect(isErrorResult({ type: 'empty' })).toBe(false);
    });
  });
});

describe('Conversion Utilities', () => {
  describe('rowToObject', () => {
    it('should convert row to plain object', () => {
      const row: Row = {
        values: new Map([
          ['id', intValue(1)],
          ['name', stringValue('Alice')],
        ]),
      };
      const obj = rowToObject(row);
      expect(obj['id']).toBe(1);
      expect(obj['name']).toBe('Alice');
    });

    it('should handle empty row', () => {
      const row: Row = { values: new Map() };
      const obj = rowToObject(row);
      expect(Object.keys(obj)).toHaveLength(0);
    });
  });

  describe('nodeToObject', () => {
    it('should convert node to plain object', () => {
      const node: Node = {
        id: 'n1',
        label: 'Person',
        properties: new Map([['age', intValue(30)]]),
      };
      const obj = nodeToObject(node);
      expect(obj['id']).toBe('n1');
      expect(obj['label']).toBe('Person');
      expect((obj['properties'] as Record<string, unknown>)['age']).toBe(30);
    });
  });

  describe('edgeToObject', () => {
    it('should convert edge to plain object', () => {
      const edge: Edge = {
        id: 'e1',
        edgeType: 'KNOWS',
        source: 'n1',
        target: 'n2',
        properties: new Map([['weight', floatValue(0.5)]]),
      };
      const obj = edgeToObject(edge);
      expect(obj['id']).toBe('e1');
      expect(obj['type']).toBe('KNOWS');
      expect(obj['source']).toBe('n1');
      expect(obj['target']).toBe('n2');
      expect((obj['properties'] as Record<string, unknown>)['weight']).toBe(0.5);
    });
  });
});

describe('Stream Chunk Types', () => {
  it('should handle node chunks', async () => {
    const nodeResponses = [
      { node: { node: { id: 'n1', label: 'Person', properties: [] } } },
      { isFinal: true },
    ];
    let index = 0;
    mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
      [Symbol.asyncIterator]: () => ({
        next(): Promise<{ value: unknown; done: boolean }> {
          if (index < nodeResponses.length) {
            return Promise.resolve({ value: nodeResponses[index++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      }),
    }));

    const client = await NeumannClient.connect('localhost:50051');
    const results: unknown[] = [];
    for await (const result of client.executeStream('MATCH (n)')) {
      results.push(result);
    }
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('type', 'nodes');
    client.close();
  });

  it('should handle edge chunks', async () => {
    const edgeResponses = [
      { edge: { edge: { id: 'e1', type: 'KNOWS', from: 'n1', to: 'n2' } } },
      { isFinal: true },
    ];
    let index = 0;
    mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
      [Symbol.asyncIterator]: () => ({
        next(): Promise<{ value: unknown; done: boolean }> {
          if (index < edgeResponses.length) {
            return Promise.resolve({ value: edgeResponses[index++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      }),
    }));

    const client = await NeumannClient.connect('localhost:50051');
    const results: unknown[] = [];
    for await (const result of client.executeStream('MATCH ()-[e]->()')) {
      results.push(result);
    }
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('type', 'edges');
    client.close();
  });

  it('should handle similarItem chunks', async () => {
    const similarItemResponses = [
      { similarItem: { item: { key: 'k1', score: 0.95 } } },
      { isFinal: true },
    ];
    let index = 0;
    mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
      [Symbol.asyncIterator]: () => ({
        next(): Promise<{ value: unknown; done: boolean }> {
          if (index < similarItemResponses.length) {
            return Promise.resolve({ value: similarItemResponses[index++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      }),
    }));

    const client = await NeumannClient.connect('localhost:50051');
    const results: unknown[] = [];
    for await (const result of client.executeStream('SIMILAR vec')) {
      results.push(result);
    }
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('type', 'similar');
    client.close();
  });

  it('should handle blobData chunks', async () => {
    const blobResponses = [
      { blobData: new Uint8Array([1, 2, 3, 4]) },
      { isFinal: true },
    ];
    let index = 0;
    mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
      [Symbol.asyncIterator]: () => ({
        next(): Promise<{ value: unknown; done: boolean }> {
          if (index < blobResponses.length) {
            return Promise.resolve({ value: blobResponses[index++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      }),
    }));

    const client = await NeumannClient.connect('localhost:50051');
    const results: unknown[] = [];
    for await (const result of client.executeStream('GET BLOB')) {
      results.push(result);
    }
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('type', 'blob');
    client.close();
  });

  it('should handle empty chunks as fallback', async () => {
    const emptyResponses = [
      { unknownField: true }, // No recognized field - should return empty
      { isFinal: true },
    ];
    let index = 0;
    mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
      [Symbol.asyncIterator]: () => ({
        next(): Promise<{ value: unknown; done: boolean }> {
          if (index < emptyResponses.length) {
            return Promise.resolve({ value: emptyResponses[index++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      }),
    }));

    const client = await NeumannClient.connect('localhost:50051');
    const results: unknown[] = [];
    for await (const result of client.executeStream('UNKNOWN')) {
      results.push(result);
    }
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('type', 'empty');
    client.close();
  });
});

describe('gRPC Error Handling', () => {
  it('should convert UNAUTHENTICATED to AuthenticationError', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 16, details: 'Auth failed' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow(AuthenticationError);
    client.close();
  });

  it('should convert PERMISSION_DENIED to PermissionDeniedError', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 7, details: 'Permission denied' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow(PermissionDeniedError);
    client.close();
  });

  it('should convert NOT_FOUND to NotFoundError', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 5, details: 'Not found' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow(NotFoundError);
    client.close();
  });

  it('should convert INVALID_ARGUMENT to InvalidArgumentError', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 3, details: 'Invalid argument' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow(InvalidArgumentError);
    client.close();
  });

  it('should convert UNAVAILABLE to ConnectionError', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 14, details: 'Service unavailable' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow(ConnectionError);
    client.close();
  });

  it('should convert unknown error codes to InternalError', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 99, details: 'Unknown error' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow(InternalError);
    client.close();
  });

  it('should use default message when details is empty', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 16, message: 'fallback' }, null);
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    await expect(client.execute('SELECT 1')).rejects.toThrow('Authentication failed');
    client.close();
  });
});

describe('Pagination', () => {
  describe('executePaginated', () => {
    it('should throw if not connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      client.close();
      await expect(client.executePaginated('SELECT * FROM users')).rejects.toThrow(
        ConnectionError
      );
    });

    it('should execute a paginated query when connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.executePaginated('SELECT * FROM users');
      expect(result.result.type).toBe('rows');
      expect(result.hasMore).toBe(true);
      expect(result.pageSize).toBe(10);
      expect(result.nextCursor).toBe('cursor123');
      expect(result.totalCount).toBe(100);
      client.close();
    });

    it('should pass pagination options', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.executePaginated('SELECT * FROM users', {
        pageSize: 20,
        cursor: 'prev_cursor',
        countTotal: true,
        cursorTtlSecs: 300,
        identity: 'alice',
      });
      expect(result.result.type).toBe('rows');
      client.close();
    });

    it('should handle gRPC error in executePaginated', async () => {
      mockGrpcClient.ExecutePaginated.mockImplementationOnce(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callback({ code: 3, details: 'Invalid query' }, null);
        }
      );
      const client = await NeumannClient.connect('localhost:50051');
      await expect(client.executePaginated('INVALID')).rejects.toThrow(InvalidArgumentError);
      client.close();
    });

    it('should handle response without optional fields', async () => {
      mockGrpcClient.ExecutePaginated.mockImplementationOnce(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callback(null, {
            result: { empty: {} },
            hasMore: false,
            pageSize: 10,
          });
        }
      );
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.executePaginated('SELECT');
      expect(result.hasMore).toBe(false);
      expect(result.nextCursor).toBeUndefined();
      expect(result.prevCursor).toBeUndefined();
      expect(result.totalCount).toBeUndefined();
      client.close();
    });

    it('should handle response with prevCursor', async () => {
      mockGrpcClient.ExecutePaginated.mockImplementationOnce(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callback(null, {
            result: { empty: {} },
            nextCursor: 'next',
            prevCursor: 'prev',
            hasMore: true,
            pageSize: 10,
          });
        }
      );
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.executePaginated('SELECT');
      expect(result.nextCursor).toBe('next');
      expect(result.prevCursor).toBe('prev');
      client.close();
    });
  });

  describe('closeCursor', () => {
    it('should throw if not connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      client.close();
      await expect(client.closeCursor('cursor123')).rejects.toThrow(ConnectionError);
    });

    it('should close a cursor when connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      const result = await client.closeCursor('cursor123');
      expect(result).toBe(true);
      client.close();
    });

    it('should handle gRPC error in closeCursor', async () => {
      mockGrpcClient.CloseCursor.mockImplementationOnce(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callback({ code: 5, details: 'Cursor not found' }, null);
        }
      );
      const client = await NeumannClient.connect('localhost:50051');
      await expect(client.closeCursor('invalid')).rejects.toThrow(NotFoundError);
      client.close();
    });
  });

  describe('executeAllPages', () => {
    it('should throw if not connected', async () => {
      const client = await NeumannClient.connect('localhost:50051');
      client.close();
      await expect(async () => {
        for await (const _result of client.executeAllPages('SELECT * FROM users')) {
          // Should throw before yielding
        }
      }).rejects.toThrow(ConnectionError);
    });

    it('should iterate through all pages', async () => {
      // Mock multiple pages then end
      let callCount = 0;
      mockGrpcClient.ExecutePaginated.mockImplementation(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callCount++;
          if (callCount === 1) {
            callback(null, {
              result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: 1 } }] }] } },
              nextCursor: 'cursor2',
              hasMore: true,
              pageSize: 10,
            });
          } else {
            callback(null, {
              result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: 2 } }] }] } },
              hasMore: false,
              pageSize: 10,
            });
          }
        }
      );

      const client = await NeumannClient.connect('localhost:50051');
      const results: unknown[] = [];
      for await (const result of client.executeAllPages('SELECT * FROM users')) {
        results.push(result);
      }
      expect(results.length).toBe(2);
      client.close();

      // Reset mock
      mockGrpcClient.ExecutePaginated.mockImplementation(
        (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
          callback(null, mockPaginatedResponse);
        }
      );
    });
  });
});

describe('New Type Guards', () => {
  describe('isValueResult', () => {
    it('should return true for value result', () => {
      expect(isValueResult({ type: 'value', value: 'test' })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isValueResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isCountResult', () => {
    it('should return true for count result', () => {
      expect(isCountResult({ type: 'count', count: 10 })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isCountResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isIdsResult', () => {
    it('should return true for ids result', () => {
      expect(isIdsResult({ type: 'ids', ids: ['1', '2'] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isIdsResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isTableListResult', () => {
    it('should return true for table list result', () => {
      expect(isTableListResult({ type: 'tableList', names: ['t1'] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isTableListResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isBlobResult', () => {
    it('should return true for blob result', () => {
      expect(isBlobResult({ type: 'blob', data: new Uint8Array([1]) })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isBlobResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isBlobInfoResult', () => {
    it('should return true for blob info result', () => {
      const info = {
        artifactId: 'a1',
        filename: 'f.txt',
        size: 100,
        checksum: 'abc',
        contentType: 'text/plain',
        createdAt: 123,
        tags: [],
      };
      expect(isBlobInfoResult({ type: 'blobInfo', info })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isBlobInfoResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isArtifactListResult', () => {
    it('should return true for artifact list result', () => {
      expect(isArtifactListResult({ type: 'artifactList', artifactIds: ['a1'] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isArtifactListResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isBlobStatsResult', () => {
    it('should return true for blob stats result', () => {
      expect(
        isBlobStatsResult({
          type: 'blobStats',
          artifactCount: 1,
          chunkCount: 2,
          totalBytes: 100,
          uniqueBytes: 80,
          dedupRatio: 0.8,
          orphanedChunks: 0,
        })
      ).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isBlobStatsResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isCheckpointListResult', () => {
    it('should return true for checkpoint list result', () => {
      expect(isCheckpointListResult({ type: 'checkpointList', checkpoints: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isCheckpointListResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isPageRankResult', () => {
    it('should return true for page rank result', () => {
      expect(isPageRankResult({ type: 'pageRank', items: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isPageRankResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isCentralityResult', () => {
    it('should return true for centrality result', () => {
      expect(isCentralityResult({ type: 'centrality', items: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isCentralityResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isCommunitiesResult', () => {
    it('should return true for communities result', () => {
      expect(isCommunitiesResult({ type: 'communities', items: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isCommunitiesResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isPatternMatchResult', () => {
    it('should return true for pattern match result', () => {
      expect(isPatternMatchResult({ type: 'patternMatch', matches: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isPatternMatchResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isConstraintsResult', () => {
    it('should return true for constraints result', () => {
      expect(isConstraintsResult({ type: 'constraints', items: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isConstraintsResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isAggregateResult', () => {
    it('should return true for aggregate result', () => {
      expect(isAggregateResult({ type: 'aggregate', value: { type: 'count', value: 10 } })).toBe(
        true
      );
    });
    it('should return false for other types', () => {
      expect(isAggregateResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isBatchOperationResult', () => {
    it('should return true for batch operation result', () => {
      expect(
        isBatchOperationResult({
          type: 'batchOperation',
          operation: 'insert',
          affectedCount: 5,
          createdIds: ['1'],
        })
      ).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isBatchOperationResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isGraphIndexesResult', () => {
    it('should return true for graph indexes result', () => {
      expect(isGraphIndexesResult({ type: 'graphIndexes', indexes: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isGraphIndexesResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isChainQueryResult', () => {
    it('should return true for chain query result', () => {
      expect(
        isChainQueryResult({ type: 'chain', result: { type: 'height', value: { height: 10 } } })
      ).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isChainQueryResult({ type: 'empty' })).toBe(false);
    });
  });

  describe('isUnifiedResult', () => {
    it('should return true for unified result', () => {
      expect(isUnifiedResult({ type: 'unified', description: 'test', items: [] })).toBe(true);
    });
    it('should return false for other types', () => {
      expect(isUnifiedResult({ type: 'empty' })).toBe(false);
    });
  });
});

describe('New Conversion Functions', () => {
  describe('convertProtoCheckpoint', () => {
    it('should convert checkpoint', () => {
      const result = convertProtoCheckpoint({
        id: 'cp1',
        name: 'checkpoint1',
        createdAt: 1700000000,
        isAuto: false,
      });
      expect(result.id).toBe('cp1');
      expect(result.name).toBe('checkpoint1');
      expect(result.createdAt).toBe(1700000000);
      expect(result.isAuto).toBe(false);
    });
  });

  describe('convertProtoPageRankItem', () => {
    it('should convert page rank item', () => {
      const result = convertProtoPageRankItem({ nodeId: 42, score: 0.85 });
      expect(result.nodeId).toBe('42');
      expect(result.score).toBe(0.85);
    });
  });

  describe('convertProtoCentralityType', () => {
    it('should convert betweenness', () => {
      expect(convertProtoCentralityType('CENTRALITY_TYPE_BETWEENNESS')).toBe('betweenness');
    });
    it('should convert closeness', () => {
      expect(convertProtoCentralityType('CENTRALITY_TYPE_CLOSENESS')).toBe('closeness');
    });
    it('should convert eigenvector', () => {
      expect(convertProtoCentralityType('CENTRALITY_TYPE_EIGENVECTOR')).toBe('eigenvector');
    });
    it('should default to betweenness for unknown', () => {
      expect(convertProtoCentralityType('UNKNOWN')).toBe('betweenness');
    });
  });

  describe('convertProtoCentralityItem', () => {
    it('should convert centrality item', () => {
      const result = convertProtoCentralityItem({ nodeId: 10, score: 0.5 });
      expect(result.nodeId).toBe('10');
      expect(result.score).toBe(0.5);
    });
  });

  describe('convertProtoCommunityItem', () => {
    it('should convert community item', () => {
      const result = convertProtoCommunityItem({ nodeId: 5, communityId: 2 });
      expect(result.nodeId).toBe('5');
      expect(result.communityId).toBe('2');
    });
  });

  describe('convertProtoCommunityMemberList', () => {
    it('should convert community member list', () => {
      const result = convertProtoCommunityMemberList({
        communityId: 1,
        memberNodeIds: [10, 20, 30],
      });
      expect(result.communityId).toBe('1');
      expect(result.memberNodeIds).toEqual(['10', '20', '30']);
    });
  });

  describe('convertProtoPatternMatchBinding', () => {
    it('should convert pattern match binding', () => {
      const result = convertProtoPatternMatchBinding({
        bindings: [{ variable: 'n', value: { node: { id: 1, label: 'Person' } } }],
      });
      expect(result.bindings.length).toBe(1);
      expect(result.bindings[0]?.variable).toBe('n');
    });
  });

  describe('convertProtoBindingEntry', () => {
    it('should convert binding entry', () => {
      const result = convertProtoBindingEntry({
        variable: 'x',
        value: { node: { id: 5, label: 'Test' } },
      });
      expect(result.variable).toBe('x');
      expect(result.value.type).toBe('node');
    });
  });

  describe('convertProtoBindingValue', () => {
    it('should convert node binding', () => {
      const result = convertProtoBindingValue({ node: { id: 1, label: 'Person' } });
      expect(result.type).toBe('node');
      if (result.type === 'node') {
        expect(result.value.id).toBe('1');
        expect(result.value.label).toBe('Person');
      }
    });

    it('should convert edge binding', () => {
      const result = convertProtoBindingValue({
        edge: { id: 2, edgeType: 'KNOWS', from: 1, to: 3 },
      });
      expect(result.type).toBe('edge');
      if (result.type === 'edge') {
        expect(result.value.id).toBe('2');
        expect(result.value.edgeType).toBe('KNOWS');
        expect(result.value.from).toBe('1');
        expect(result.value.to).toBe('3');
      }
    });

    it('should convert path binding', () => {
      const result = convertProtoBindingValue({
        path: { nodes: [1, 2, 3], edges: [10, 11], length: 2 },
      });
      expect(result.type).toBe('path');
      if (result.type === 'path') {
        expect(result.value.nodes).toEqual(['1', '2', '3']);
        expect(result.value.edges).toEqual(['10', '11']);
        expect(result.value.length).toBe(2);
      }
    });

    it('should default to empty node for unknown', () => {
      const result = convertProtoBindingValue({});
      expect(result.type).toBe('node');
    });
  });

  describe('convertProtoPatternMatchStats', () => {
    it('should convert pattern match stats', () => {
      const result = convertProtoPatternMatchStats({
        matchesFound: 10,
        nodesEvaluated: 100,
        edgesEvaluated: 50,
        truncated: true,
      });
      expect(result.matchesFound).toBe(10);
      expect(result.nodesEvaluated).toBe(100);
      expect(result.edgesEvaluated).toBe(50);
      expect(result.truncated).toBe(true);
    });
  });

  describe('convertProtoConstraintItem', () => {
    it('should convert constraint item', () => {
      const result = convertProtoConstraintItem({
        name: 'unique_email',
        target: 'users',
        property: 'email',
        constraintType: 'UNIQUE',
      });
      expect(result.name).toBe('unique_email');
      expect(result.target).toBe('users');
      expect(result.property).toBe('email');
      expect(result.constraintType).toBe('UNIQUE');
    });
  });

  describe('convertProtoAggregateValue', () => {
    it('should convert count', () => {
      const result = convertProtoAggregateValue({ count: 42 });
      expect(result.type).toBe('count');
      expect(result.value).toBe(42);
    });

    it('should convert sum', () => {
      const result = convertProtoAggregateValue({ sum: 100.5 });
      expect(result.type).toBe('sum');
      expect(result.value).toBe(100.5);
    });

    it('should convert avg', () => {
      const result = convertProtoAggregateValue({ avg: 25.5 });
      expect(result.type).toBe('avg');
      expect(result.value).toBe(25.5);
    });

    it('should convert min', () => {
      const result = convertProtoAggregateValue({ min: 1.0 });
      expect(result.type).toBe('min');
      expect(result.value).toBe(1.0);
    });

    it('should convert max', () => {
      const result = convertProtoAggregateValue({ max: 99.9 });
      expect(result.type).toBe('max');
      expect(result.value).toBe(99.9);
    });

    it('should default to count 0 for empty', () => {
      const result = convertProtoAggregateValue({});
      expect(result.type).toBe('count');
      expect(result.value).toBe(0);
    });
  });

  describe('convertProtoChainResult', () => {
    it('should convert transactionBegun', () => {
      const result = convertProtoChainResult({ transactionBegun: { txId: 'tx1' } });
      expect(result.type).toBe('transactionBegun');
      if (result.type === 'transactionBegun') {
        expect(result.value.txId).toBe('tx1');
      }
    });

    it('should convert committed', () => {
      const result = convertProtoChainResult({ committed: { blockHash: 'hash1', height: 10 } });
      expect(result.type).toBe('committed');
      if (result.type === 'committed') {
        expect(result.value.blockHash).toBe('hash1');
        expect(result.value.height).toBe(10);
      }
    });

    it('should convert rolledBack', () => {
      const result = convertProtoChainResult({ rolledBack: { toHeight: 5 } });
      expect(result.type).toBe('rolledBack');
      if (result.type === 'rolledBack') {
        expect(result.value.toHeight).toBe(5);
      }
    });

    it('should convert history', () => {
      const result = convertProtoChainResult({
        history: { entries: [{ height: 1, transactionType: 'PUT' }] },
      });
      expect(result.type).toBe('history');
      if (result.type === 'history') {
        expect(result.value.entries.length).toBe(1);
      }
    });

    it('should convert similar', () => {
      const result = convertProtoChainResult({
        similar: { items: [{ blockHash: 'h1', height: 5, similarity: 0.9 }] },
      });
      expect(result.type).toBe('similar');
      if (result.type === 'similar') {
        expect(result.value.items.length).toBe(1);
      }
    });

    it('should convert drift', () => {
      const result = convertProtoChainResult({
        drift: {
          fromHeight: 1,
          toHeight: 10,
          totalDrift: 0.5,
          avgDriftPerBlock: 0.05,
          maxDrift: 0.1,
        },
      });
      expect(result.type).toBe('drift');
      if (result.type === 'drift') {
        expect(result.value.fromHeight).toBe(1);
        expect(result.value.toHeight).toBe(10);
      }
    });

    it('should convert height', () => {
      const result = convertProtoChainResult({ height: { height: 100 } });
      expect(result.type).toBe('height');
      if (result.type === 'height') {
        expect(result.value.height).toBe(100);
      }
    });

    it('should convert tip', () => {
      const result = convertProtoChainResult({ tip: { hash: 'tiphash', height: 50 } });
      expect(result.type).toBe('tip');
      if (result.type === 'tip') {
        expect(result.value.hash).toBe('tiphash');
        expect(result.value.height).toBe(50);
      }
    });

    it('should convert block', () => {
      const result = convertProtoChainResult({
        block: {
          height: 10,
          hash: 'h',
          prevHash: 'ph',
          timestamp: 123,
          transactionCount: 5,
          proposer: 'node1',
        },
      });
      expect(result.type).toBe('block');
      if (result.type === 'block') {
        expect(result.value.height).toBe(10);
        expect(result.value.proposer).toBe('node1');
      }
    });

    it('should convert codebook', () => {
      const result = convertProtoChainResult({
        codebook: { scope: 'global', entryCount: 100, dimension: 128, domain: 'text' },
      });
      expect(result.type).toBe('codebook');
      if (result.type === 'codebook') {
        expect(result.value.scope).toBe('global');
        expect(result.value.domain).toBe('text');
      }
    });

    it('should convert transitionAnalysis', () => {
      const result = convertProtoChainResult({
        transitionAnalysis: {
          totalTransitions: 100,
          validTransitions: 95,
          invalidTransitions: 5,
          avgValidityScore: 0.95,
        },
      });
      expect(result.type).toBe('transitionAnalysis');
      if (result.type === 'transitionAnalysis') {
        expect(result.value.totalTransitions).toBe(100);
      }
    });

    it('should convert conflictResolution', () => {
      const result = convertProtoChainResult({
        conflictResolution: { strategy: 'semantic', conflictsResolved: 3 },
      });
      expect(result.type).toBe('conflictResolution');
      if (result.type === 'conflictResolution') {
        expect(result.value.strategy).toBe('semantic');
      }
    });

    it('should convert merge', () => {
      const result = convertProtoChainResult({ merge: { success: true, mergedCount: 10 } });
      expect(result.type).toBe('merge');
      if (result.type === 'merge') {
        expect(result.value.success).toBe(true);
        expect(result.value.mergedCount).toBe(10);
      }
    });

    it('should default to height 0 for empty', () => {
      const result = convertProtoChainResult({});
      expect(result.type).toBe('height');
      if (result.type === 'height') {
        expect(result.value.height).toBe(0);
      }
    });
  });

  describe('convertProtoUnifiedItem', () => {
    it('should convert unified item with fields', () => {
      const result = convertProtoUnifiedItem({
        entityType: 'user',
        key: 'u1',
        fields: { name: 'Alice', email: 'alice@example.com' },
        score: 0.9,
      });
      expect(result.entityType).toBe('user');
      expect(result.key).toBe('u1');
      expect(result.fields.get('name')?.data).toBe('Alice');
      expect(result.score).toBe(0.9);
    });

    it('should convert unified item without fields', () => {
      const result = convertProtoUnifiedItem({
        entityType: 'item',
        key: 'i1',
      });
      expect(result.entityType).toBe('item');
      expect(result.key).toBe('i1');
      expect(result.fields.size).toBe(0);
      expect(result.score).toBeUndefined();
    });
  });
});

describe('Response Type Conversion via execute()', () => {
  it('should handle value response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { value: { value: 'test-value' } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SELECT');
    expect(result.type).toBe('value');
    if (result.type === 'value') {
      expect(result.value).toBe('test-value');
    }
    client.close();
  });

  it('should handle artifactList response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { artifactList: { artifactIds: ['a1', 'a2'] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('LIST ARTIFACTS');
    expect(result.type).toBe('artifactList');
    if (result.type === 'artifactList') {
      expect(result.artifactIds).toEqual(['a1', 'a2']);
    }
    client.close();
  });

  it('should handle blobStats response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          blobStats: {
            artifactCount: 10,
            chunkCount: 50,
            totalBytes: 1000,
            uniqueBytes: 800,
            dedupRatio: 0.8,
            orphanedChunks: 2,
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('BLOB STATS');
    expect(result.type).toBe('blobStats');
    if (result.type === 'blobStats') {
      expect(result.artifactCount).toBe(10);
      expect(result.dedupRatio).toBe(0.8);
    }
    client.close();
  });

  it('should handle checkpointList response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          checkpointList: {
            checkpoints: [{ id: 'cp1', name: 'checkpoint1', createdAt: 1700000000, isAuto: false }],
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('LIST CHECKPOINTS');
    expect(result.type).toBe('checkpointList');
    if (result.type === 'checkpointList') {
      expect(result.checkpoints.length).toBe(1);
      expect(result.checkpoints[0]?.id).toBe('cp1');
    }
    client.close();
  });

  it('should handle pageRank response with all fields', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          pageRank: {
            items: [{ nodeId: 1, score: 0.85 }],
            iterations: 20,
            convergence: 0.001,
            converged: true,
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('PAGERANK');
    expect(result.type).toBe('pageRank');
    if (result.type === 'pageRank') {
      expect(result.items.length).toBe(1);
      expect(result.iterations).toBe(20);
      expect(result.convergence).toBe(0.001);
      expect(result.converged).toBe(true);
    }
    client.close();
  });

  it('should handle centrality response with all fields', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          centrality: {
            items: [{ nodeId: 1, score: 0.5 }],
            centralityType: 'CENTRALITY_TYPE_BETWEENNESS',
            iterations: 10,
            converged: true,
            sampleCount: 100,
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('CENTRALITY');
    expect(result.type).toBe('centrality');
    if (result.type === 'centrality') {
      expect(result.items.length).toBe(1);
      expect(result.centralityType).toBe('betweenness');
      expect(result.iterations).toBe(10);
      expect(result.converged).toBe(true);
      expect(result.sampleCount).toBe(100);
    }
    client.close();
  });

  it('should handle communities response with all fields', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          communities: {
            items: [{ nodeId: 1, communityId: 0 }],
            communityCount: 3,
            modularity: 0.75,
            passes: 5,
            iterations: 100,
            communities: [{ communityId: 0, memberNodeIds: [1, 2, 3] }],
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('COMMUNITIES');
    expect(result.type).toBe('communities');
    if (result.type === 'communities') {
      expect(result.items.length).toBe(1);
      expect(result.communityCount).toBe(3);
      expect(result.modularity).toBe(0.75);
      expect(result.passes).toBe(5);
      expect(result.iterations).toBe(100);
      expect(result.communities?.length).toBe(1);
    }
    client.close();
  });

  it('should handle patternMatch response with stats', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          patternMatch: {
            matches: [{ bindings: [{ variable: 'n', value: { node: { id: 1, label: 'Person' } } }] }],
            stats: { matchesFound: 5, nodesEvaluated: 100, edgesEvaluated: 50, truncated: false },
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('MATCH');
    expect(result.type).toBe('patternMatch');
    if (result.type === 'patternMatch') {
      expect(result.matches.length).toBe(1);
      expect(result.stats?.matchesFound).toBe(5);
    }
    client.close();
  });

  it('should handle constraints response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          constraints: {
            items: [{ name: 'unique_email', target: 'users', property: 'email', constraintType: 'UNIQUE' }],
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SHOW CONSTRAINTS');
    expect(result.type).toBe('constraints');
    if (result.type === 'constraints') {
      expect(result.items.length).toBe(1);
      expect(result.items[0]?.name).toBe('unique_email');
    }
    client.close();
  });

  it('should handle aggregate response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { aggregate: { count: 42 } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('COUNT');
    expect(result.type).toBe('aggregate');
    if (result.type === 'aggregate') {
      expect(result.value.type).toBe('count');
      expect(result.value.value).toBe(42);
    }
    client.close();
  });

  it('should handle batchOperation response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          batchOperation: { operation: 'insert', affectedCount: 10, createdIds: [1, 2, 3] },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('BATCH INSERT');
    expect(result.type).toBe('batchOperation');
    if (result.type === 'batchOperation') {
      expect(result.operation).toBe('insert');
      expect(result.affectedCount).toBe(10);
      expect(result.createdIds).toEqual(['1', '2', '3']);
    }
    client.close();
  });

  it('should handle graphIndexes response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { graphIndexes: { indexes: ['idx_label', 'idx_prop'] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SHOW INDEXES');
    expect(result.type).toBe('graphIndexes');
    if (result.type === 'graphIndexes') {
      expect(result.indexes).toEqual(['idx_label', 'idx_prop']);
    }
    client.close();
  });

  it('should handle chain response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { chain: { height: { height: 100 } } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('CHAIN HEIGHT');
    expect(result.type).toBe('chain');
    if (result.type === 'chain') {
      expect(result.result.type).toBe('height');
    }
    client.close();
  });

  it('should handle unified response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          unified: {
            description: 'Search results',
            items: [{ entityType: 'user', key: 'u1', fields: { name: 'Alice' }, score: 0.9 }],
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SEARCH');
    expect(result.type).toBe('unified');
    if (result.type === 'unified') {
      expect(result.description).toBe('Search results');
      expect(result.items.length).toBe(1);
    }
    client.close();
  });

  it('should handle error response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { error: { code: 1, message: 'Query error' } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('INVALID');
    expect(result.type).toBe('error');
    if (result.type === 'error') {
      expect(result.code).toBe(1);
      expect(result.message).toBe('Query error');
    }
    client.close();
  });

  it('should handle count response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { count: { count: 42 } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('COUNT');
    expect(result.type).toBe('count');
    if (result.type === 'count') {
      expect(result.count).toBe(42);
    }
    client.close();
  });

  it('should handle nodes response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { nodes: { nodes: [{ id: 'n1', label: 'Person' }] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('MATCH');
    expect(result.type).toBe('nodes');
    client.close();
  });

  it('should handle edges response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { edges: { edges: [{ id: 'e1', type: 'KNOWS', from: 'n1', to: 'n2' }] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('MATCH EDGES');
    expect(result.type).toBe('edges');
    client.close();
  });

  it('should handle path response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { path: { nodeIds: [1, 2, 3] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SHORTEST PATH');
    expect(result.type).toBe('paths');
    client.close();
  });

  it('should handle similar response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { similar: { items: [{ key: 'k1', score: 0.9 }] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SIMILAR');
    expect(result.type).toBe('similar');
    client.close();
  });

  it('should handle ids response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { ids: { ids: [1, 2, 3] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('GET IDS');
    expect(result.type).toBe('ids');
    if (result.type === 'ids') {
      expect(result.ids).toEqual(['1', '2', '3']);
    }
    client.close();
  });

  it('should handle tableList response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { tableList: { tables: ['users', 'orders'] } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('SHOW TABLES');
    expect(result.type).toBe('tableList');
    if (result.type === 'tableList') {
      expect(result.names).toEqual(['users', 'orders']);
    }
    client.close();
  });

  it('should handle blob response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, { blob: { data: new Uint8Array([1, 2, 3]) } });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('GET BLOB');
    expect(result.type).toBe('blob');
    if (result.type === 'blob') {
      expect(result.data).toEqual(new Uint8Array([1, 2, 3]));
    }
    client.close();
  });

  it('should handle artifactInfo response', async () => {
    mockGrpcClient.Execute.mockImplementationOnce(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, {
          artifactInfo: {
            artifactId: 'a1',
            filename: 'test.txt',
            size: 100,
            checksum: 'abc',
            contentType: 'text/plain',
            createdAt: 123,
            tags: ['tag1'],
          },
        });
      }
    );
    const client = await NeumannClient.connect('localhost:50051');
    const result = await client.execute('GET ARTIFACT INFO');
    expect(result.type).toBe('blobInfo');
    client.close();
  });
});

describe('Validation Utilities', () => {
  describe('validateIntValue', () => {
    it('should accept valid integers', () => {
      expect(validateIntValue(0)).toBe(0);
      expect(validateIntValue(42)).toBe(42);
      expect(validateIntValue(-100)).toBe(-100);
      expect(validateIntValue(Number.MAX_SAFE_INTEGER)).toBe(Number.MAX_SAFE_INTEGER);
      expect(validateIntValue(Number.MIN_SAFE_INTEGER)).toBe(Number.MIN_SAFE_INTEGER);
    });

    it('should reject NaN', () => {
      expect(() => validateIntValue(NaN)).toThrow(InvalidArgumentError);
    });

    it('should reject Infinity', () => {
      expect(() => validateIntValue(Infinity)).toThrow(InvalidArgumentError);
      expect(() => validateIntValue(-Infinity)).toThrow(InvalidArgumentError);
    });

    it('should reject values outside safe integer range', () => {
      expect(() => validateIntValue(Number.MAX_SAFE_INTEGER + 1)).toThrow(InvalidArgumentError);
      expect(() => validateIntValue(Number.MIN_SAFE_INTEGER - 1)).toThrow(InvalidArgumentError);
    });

    it('should include context in error message', () => {
      expect(() => validateIntValue(NaN, 'test field')).toThrow('test field');
    });
  });

  describe('validateFloatValue', () => {
    it('should accept valid floats', () => {
      expect(validateFloatValue(0)).toBe(0);
      expect(validateFloatValue(3.14159)).toBe(3.14159);
      expect(validateFloatValue(-273.15)).toBe(-273.15);
    });

    it('should reject NaN', () => {
      expect(() => validateFloatValue(NaN)).toThrow(InvalidArgumentError);
    });

    it('should reject Infinity', () => {
      expect(() => validateFloatValue(Infinity)).toThrow(InvalidArgumentError);
      expect(() => validateFloatValue(-Infinity)).toThrow(InvalidArgumentError);
    });

    it('should include context in error message', () => {
      expect(() => validateFloatValue(NaN, 'temperature')).toThrow('temperature');
    });
  });

  describe('validateStringValue', () => {
    it('should accept strings within limit', () => {
      expect(validateStringValue('')).toBe('');
      expect(validateStringValue('hello')).toBe('hello');
    });

    it('should reject strings exceeding limit', () => {
      const longString = 'x'.repeat(MAX_STRING_LENGTH + 1);
      expect(() => validateStringValue(longString)).toThrow(InvalidArgumentError);
    });

    it('should include context in error message', () => {
      const longString = 'x'.repeat(MAX_STRING_LENGTH + 1);
      expect(() => validateStringValue(longString, 'description')).toThrow('description');
    });
  });

  describe('validateBytesValue', () => {
    it('should accept bytes within limit', () => {
      const bytes = new Uint8Array([1, 2, 3]);
      expect(validateBytesValue(bytes)).toBe(bytes);
    });

    it('should reject bytes exceeding limit', () => {
      const largeBytes = new Uint8Array(MAX_BYTES_LENGTH + 1);
      expect(() => validateBytesValue(largeBytes)).toThrow(InvalidArgumentError);
    });

    it('should include context in error message', () => {
      const largeBytes = new Uint8Array(MAX_BYTES_LENGTH + 1);
      expect(() => validateBytesValue(largeBytes, 'payload')).toThrow('payload');
    });
  });

  describe('safeIdToString', () => {
    it('should convert safe integers to strings', () => {
      expect(safeIdToString(0)).toBe('0');
      expect(safeIdToString(42)).toBe('42');
      expect(safeIdToString(Number.MAX_SAFE_INTEGER)).toBe(String(Number.MAX_SAFE_INTEGER));
    });

    it('should reject unsafe integers', () => {
      expect(() => safeIdToString(Number.MAX_SAFE_INTEGER + 1)).toThrow(InvalidArgumentError);
      expect(() => safeIdToString(NaN)).toThrow(InvalidArgumentError);
    });
  });

  describe('safeIdsToStrings', () => {
    it('should convert array of safe integers', () => {
      expect(safeIdsToStrings([1, 2, 3])).toEqual(['1', '2', '3']);
    });

    it('should reject array with unsafe integers', () => {
      expect(() => safeIdsToStrings([1, Number.MAX_SAFE_INTEGER + 1, 3])).toThrow(
        InvalidArgumentError
      );
    });
  });
});

describe('Proto Value Validation', () => {
  it('should reject NaN in int values', () => {
    expect(() => convertProtoValue({ intValue: NaN })).toThrow(InvalidArgumentError);
  });

  it('should reject Infinity in int values', () => {
    expect(() => convertProtoValue({ intValue: Infinity })).toThrow(InvalidArgumentError);
  });

  it('should reject NaN in float values', () => {
    expect(() => convertProtoValue({ floatValue: NaN })).toThrow(InvalidArgumentError);
  });

  it('should reject Infinity in float values', () => {
    expect(() => convertProtoValue({ floatValue: Infinity })).toThrow(InvalidArgumentError);
  });
});

describe('Copy Helper Functions', () => {
  describe('copyRowValues', () => {
    it('should create mutable copy of row values', () => {
      const row: Row = { values: new Map([['id', intValue(1)]]) };
      const copy = copyRowValues(row);
      expect(copy).toBeInstanceOf(Map);
      expect(copy.get('id')?.data).toBe(1);
      copy.set('new', stringValue('test'));
      expect(copy.has('new')).toBe(true);
      expect(row.values.has('new')).toBe(false);
    });
  });

  describe('copyNodeProperties', () => {
    it('should create mutable copy of node properties', () => {
      const node: Node = {
        id: 'n1',
        label: 'Test',
        properties: new Map([['age', intValue(30)]]),
      };
      const copy = copyNodeProperties(node);
      expect(copy.get('age')?.data).toBe(30);
      copy.set('name', stringValue('Alice'));
      expect(copy.has('name')).toBe(true);
      expect(node.properties.has('name')).toBe(false);
    });
  });

  describe('copyEdgeProperties', () => {
    it('should create mutable copy of edge properties', () => {
      const edge: Edge = {
        id: 'e1',
        edgeType: 'KNOWS',
        source: 'n1',
        target: 'n2',
        properties: new Map([['weight', floatValue(0.5)]]),
      };
      const copy = copyEdgeProperties(edge);
      expect(copy.get('weight')?.data).toBe(0.5);
      copy.set('since', intValue(2020));
      expect(copy.has('since')).toBe(true);
      expect(edge.properties.has('since')).toBe(false);
    });
  });

  describe('copySimilarItemMetadata', () => {
    it('should return undefined for items without metadata', () => {
      const item = { key: 'k1', score: 0.9 };
      expect(copySimilarItemMetadata(item)).toBeUndefined();
    });

    it('should create mutable copy of metadata', () => {
      const item = {
        key: 'k1',
        score: 0.9,
        metadata: new Map([['tag', stringValue('test')]]),
      };
      const copy = copySimilarItemMetadata(item);
      expect(copy).toBeInstanceOf(Map);
      expect(copy?.get('tag')?.data).toBe('test');
    });
  });

  describe('copyUnifiedItemFields', () => {
    it('should create mutable copy of unified item fields', () => {
      const item = {
        entityType: 'user',
        key: 'u1',
        fields: new Map([['name', stringValue('Alice')]]),
      };
      const copy = copyUnifiedItemFields(item);
      expect(copy.get('name')?.data).toBe('Alice');
      copy.set('email', stringValue('alice@example.com'));
      expect(copy.has('email')).toBe(true);
      expect(item.fields.has('email')).toBe(false);
    });
  });
});

describe('UnifiedItem Field Conversion', () => {
  it('should convert mixed field types correctly', () => {
    const result = convertProtoUnifiedItem({
      entityType: 'user',
      key: 'u1',
      fields: {
        name: 'Alice',
        age: 30,
        active: true,
        score: 3.14,
      },
    });
    expect(result.fields.get('name')?.type).toBe('string');
    expect(result.fields.get('name')?.data).toBe('Alice');
    expect(result.fields.get('age')?.type).toBe('int');
    expect(result.fields.get('age')?.data).toBe(30);
    expect(result.fields.get('active')?.type).toBe('bool');
    expect(result.fields.get('active')?.data).toBe(true);
    expect(result.fields.get('score')?.type).toBe('float');
    expect(result.fields.get('score')?.data).toBe(3.14);
  });

  it('should handle null field values', () => {
    const result = convertProtoUnifiedItem({
      entityType: 'user',
      key: 'u1',
      fields: { nullable: null },
    });
    expect(result.fields.get('nullable')?.type).toBe('null');
  });
});

describe('Cursor Cleanup on Early Break', () => {
  it('should close cursor when breaking early from executeAllPages', async () => {
    let closeCount = 0;
    let callCount = 0;
    mockGrpcClient.ExecutePaginated.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callCount++;
        callback(null, {
          result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: callCount } }] }] } },
          nextCursor: 'cursor_' + callCount,
          hasMore: true,
          pageSize: 10,
        });
      }
    );
    mockGrpcClient.CloseCursor.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        closeCount++;
        callback(null, { success: true });
      }
    );

    const client = await NeumannClient.connect('localhost:50051');
    let iterations = 0;
    for await (const _result of client.executeAllPages('SELECT * FROM users')) {
      iterations++;
      if (iterations >= 2) break;
    }
    expect(iterations).toBe(2);
    expect(closeCount).toBe(1);
    client.close();

    // Reset mocks
    mockGrpcClient.ExecutePaginated.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, mockPaginatedResponse);
      }
    );
    mockGrpcClient.CloseCursor.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, mockCloseCursorResponse);
      }
    );
  });

  it('should ignore cursor cleanup errors', async () => {
    let callCount = 0;
    mockGrpcClient.ExecutePaginated.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callCount++;
        callback(null, {
          result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: callCount } }] }] } },
          nextCursor: 'cursor_' + callCount,
          hasMore: true,
          pageSize: 10,
        });
      }
    );
    mockGrpcClient.CloseCursor.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback({ code: 5, details: 'Cursor not found' }, null);
      }
    );

    const client = await NeumannClient.connect('localhost:50051');
    let iterations = 0;
    // Should not throw even though cursor close fails
    for await (const _result of client.executeAllPages('SELECT * FROM users')) {
      iterations++;
      if (iterations >= 1) break;
    }
    expect(iterations).toBe(1);
    client.close();

    // Reset mocks
    mockGrpcClient.ExecutePaginated.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, mockPaginatedResponse);
      }
    );
    mockGrpcClient.CloseCursor.mockImplementation(
      (_request: unknown, _metadata: unknown, callback: (err: unknown, response: unknown) => void) => {
        callback(null, mockCloseCursorResponse);
      }
    );
  });
});

describe('Stream Cleanup', () => {
  it('should call cancel on stream when breaking early', async () => {
    let cancelCalled = false;
    const streamResponses = [
      { row: { row: { columns: [{ name: 'x', value: { intValue: 1 } }] } } },
      { row: { row: { columns: [{ name: 'x', value: { intValue: 2 } }] } } },
      { isFinal: true },
    ];
    let index = 0;
    mockGrpcClient.ExecuteStream.mockImplementationOnce(() => {
      const stream = {
        [Symbol.asyncIterator]: () => ({
          next(): Promise<{ value: unknown; done: boolean }> {
            if (index < streamResponses.length) {
              return Promise.resolve({ value: streamResponses[index++], done: false });
            }
            return Promise.resolve({ value: undefined, done: true });
          },
        }),
        cancel: () => {
          cancelCalled = true;
        },
      };
      return stream;
    });

    const client = await NeumannClient.connect('localhost:50051');
    for await (const _result of client.executeStream('SELECT * FROM users')) {
      break; // Break immediately
    }
    expect(cancelCalled).toBe(true);
    client.close();
  });
});

describe('Safe ID Conversions', () => {
  it('should handle unsafe page rank node IDs', () => {
    expect(() => convertProtoPageRankItem({ nodeId: Number.MAX_SAFE_INTEGER + 1, score: 0.5 })).toThrow(
      InvalidArgumentError
    );
  });

  it('should handle unsafe centrality node IDs', () => {
    expect(() =>
      convertProtoCentralityItem({ nodeId: Number.MAX_SAFE_INTEGER + 1, score: 0.5 })
    ).toThrow(InvalidArgumentError);
  });

  it('should handle unsafe community node IDs', () => {
    expect(() =>
      convertProtoCommunityItem({ nodeId: Number.MAX_SAFE_INTEGER + 1, communityId: 1 })
    ).toThrow(InvalidArgumentError);
  });

  it('should handle unsafe community member IDs', () => {
    expect(() =>
      convertProtoCommunityMemberList({
        communityId: 1,
        memberNodeIds: [1, Number.MAX_SAFE_INTEGER + 1],
      })
    ).toThrow(InvalidArgumentError);
  });

  it('should handle unsafe binding value node ID', () => {
    expect(() =>
      convertProtoBindingValue({ node: { id: Number.MAX_SAFE_INTEGER + 1, label: 'Person' } })
    ).toThrow(InvalidArgumentError);
  });

  it('should handle unsafe binding value edge IDs', () => {
    expect(() =>
      convertProtoBindingValue({
        edge: { id: Number.MAX_SAFE_INTEGER + 1, edgeType: 'KNOWS', from: 1, to: 2 },
      })
    ).toThrow(InvalidArgumentError);
  });

  it('should handle unsafe binding value path node IDs', () => {
    expect(() =>
      convertProtoBindingValue({
        path: { nodes: [1, Number.MAX_SAFE_INTEGER + 1], edges: [10], length: 1 },
      })
    ).toThrow(InvalidArgumentError);
  });
});
