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
  rowToObject,
  nodeToObject,
  edgeToObject,
} from './types/query-result.js';

// Mock execute callback response
const mockExecuteResponse = { empty: {} };
const mockExecuteStreamResponses = [
  { row: { row: { columns: [{ name: 'x', value: { intValue: 1 } }] } } },
  { isFinal: true },
];
const mockBatchResponse = { results: [{ empty: {} }, { empty: {} }, { empty: {} }] };

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
};

// Mock grpc.ts module
vi.mock('./grpc.js', () => ({
  loadProto: vi.fn().mockResolvedValue({}),
  getQueryServiceClient: vi.fn(() => mockGrpcClient),
}));

// Mock @grpc/grpc-js
vi.mock('@grpc/grpc-js', () => ({
  credentials: {
    createSsl: vi.fn(() => ({})),
    createInsecure: vi.fn(() => ({})),
  },
  Metadata: vi.fn().mockImplementation(() => ({
    set: vi.fn(),
  })),
}));

// Mock grpc-web
vi.mock('grpc-web', () => ({
  GrpcWebClientBase: vi.fn().mockImplementation(() => ({})),
}));

describe('NeumannClient', () => {
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
