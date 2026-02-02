// SPDX-License-Identifier: MIT
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type * as grpc from '@grpc/grpc-js';
import { PointsClient, CollectionsClient } from './vector.js';
import type {
  PointsServiceClient,
  CollectionsServiceClient,
  UpsertPointsResponse,
  GetPointsResponse,
  DeletePointsResponse,
  QueryPointsResponse,
  ScrollPointsResponse,
  CreateCollectionResponse,
  GetCollectionResponse,
  DeleteCollectionResponse,
  ListCollectionsResponse,
  Point,
  ScoredPoint,
} from '../generated/vector.js';
import { NotFoundError, InvalidArgumentError, ConnectionError } from '../types/errors.js';

// Mock gRPC Metadata
function createMockMetadata(): grpc.Metadata {
  return {
    set: vi.fn(),
    get: vi.fn(),
    clone: vi.fn(),
  } as unknown as grpc.Metadata;
}

// Create mock service error
function createServiceError(code: number, details: string): grpc.ServiceError {
  return {
    code,
    details,
    message: details,
    name: 'ServiceError',
    metadata: {} as grpc.Metadata,
  } as grpc.ServiceError;
}

describe('PointsClient', () => {
  let client: PointsClient;
  let mockGrpcClient: PointsServiceClient;
  let mockMetadata: grpc.Metadata;

  beforeEach(() => {
    mockMetadata = createMockMetadata();
    mockGrpcClient = {
      Upsert: vi.fn(),
      Get: vi.fn(),
      Delete: vi.fn(),
      Query: vi.fn(),
      Scroll: vi.fn(),
    } as unknown as PointsServiceClient;
    client = new PointsClient(mockGrpcClient, mockMetadata);
  });

  describe('upsert', () => {
    it('should upsert points', async () => {
      vi.mocked(mockGrpcClient.Upsert).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: UpsertPointsResponse) => void) => {
          callback(null, { upserted: 2 });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.upsert('test-collection', [
        { id: 'p1', vector: [0.1, 0.2, 0.3] },
        { id: 'p2', vector: [0.4, 0.5, 0.6], payload: { name: 'test' } },
      ]);

      expect(result).toBe(2);
    });
  });

  describe('get', () => {
    it('should get points by IDs', async () => {
      const points: Point[] = [
        { id: 'p1', vector: [0.1, 0.2, 0.3] },
        { id: 'p2', vector: [0.4, 0.5, 0.6] },
      ];

      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetPointsResponse) => void) => {
          callback(null, { points });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.get('test-collection', ['p1', 'p2']);

      expect(result).toHaveLength(2);
      expect(result[0]?.id).toBe('p1');
      expect(result[0]?.vector).toEqual([0.1, 0.2, 0.3]);
    });

    it('should decode payload', async () => {
      const payload = new TextEncoder().encode(JSON.stringify('hello'));
      const points: Point[] = [
        { id: 'p1', vector: [0.1], payload: { name: payload } },
      ];

      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetPointsResponse) => void) => {
          callback(null, { points });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.get('test-collection', ['p1']);

      expect(result[0]?.payload?.name).toBe('hello');
    });
  });

  describe('delete', () => {
    it('should delete points by IDs', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: DeletePointsResponse) => void) => {
          callback(null, { deleted: 2 });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.delete('test-collection', ['p1', 'p2']);

      expect(result).toBe(2);
    });
  });

  describe('query', () => {
    it('should query similar points', async () => {
      const results: ScoredPoint[] = [
        { id: 'p1', score: 0.95 },
        { id: 'p2', score: 0.85 },
      ];

      vi.mocked(mockGrpcClient.Query).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: QueryPointsResponse) => void) => {
          callback(null, { results });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.query('test-collection', [0.1, 0.2, 0.3], { limit: 10 });

      expect(result).toHaveLength(2);
      expect(result[0]?.id).toBe('p1');
      expect(result[0]?.score).toBe(0.95);
    });

    it('should include vector when requested', async () => {
      const results: ScoredPoint[] = [
        { id: 'p1', score: 0.95, vector: [0.1, 0.2, 0.3] },
      ];

      vi.mocked(mockGrpcClient.Query).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: QueryPointsResponse) => void) => {
          callback(null, { results });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.query('test-collection', [0.1, 0.2, 0.3], { withVector: true });

      expect(result[0]?.vector).toEqual([0.1, 0.2, 0.3]);
    });
  });

  describe('scroll', () => {
    it('should scroll through points', async () => {
      const points: Point[] = [
        { id: 'p1', vector: [0.1, 0.2, 0.3] },
        { id: 'p2', vector: [0.4, 0.5, 0.6] },
      ];

      vi.mocked(mockGrpcClient.Scroll).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ScrollPointsResponse) => void) => {
          callback(null, { points, nextOffset: 'p3' });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.scroll('test-collection', { limit: 2 });

      expect(result.points).toHaveLength(2);
      expect(result.nextOffset).toBe('p3');
    });

    it('should return undefined nextOffset when no more pages', async () => {
      const points: Point[] = [
        { id: 'p1', vector: [0.1] },
      ];

      vi.mocked(mockGrpcClient.Scroll).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ScrollPointsResponse) => void) => {
          callback(null, { points, nextOffset: undefined });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.scroll('test-collection');

      expect(result.nextOffset).toBeUndefined();
    });
  });

  describe('scrollAll', () => {
    it('should iterate through all points', async () => {
      let callCount = 0;
      vi.mocked(mockGrpcClient.Scroll).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ScrollPointsResponse) => void) => {
          callCount++;
          if (callCount === 1) {
            callback(null, {
              points: [{ id: 'p1', vector: [0.1] }],
              nextOffset: 'p2',
            });
          } else {
            callback(null, {
              points: [{ id: 'p2', vector: [0.2] }],
              nextOffset: undefined,
            });
          }
          return {} as grpc.ClientUnaryCall;
        }
      );

      const allPoints = [];
      for await (const point of client.scrollAll('test-collection')) {
        allPoints.push(point);
      }

      expect(allPoints).toHaveLength(2);
      expect(allPoints[0]?.id).toBe('p1');
      expect(allPoints[1]?.id).toBe('p2');
    });
  });

  describe('error handling', () => {
    it('should handle NOT_FOUND error', async () => {
      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetPointsResponse) => void) => {
          callback(createServiceError(5, 'Collection not found'), { points: [] });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.get('missing', ['p1'])).rejects.toThrow(NotFoundError);
    });

    it('should handle INVALID_ARGUMENT error', async () => {
      vi.mocked(mockGrpcClient.Upsert).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: UpsertPointsResponse) => void) => {
          callback(createServiceError(3, 'Invalid vector dimension'), { upserted: 0 });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.upsert('test', [{ id: 'p1', vector: [] }])).rejects.toThrow(InvalidArgumentError);
    });

    it('should handle UNAVAILABLE error', async () => {
      vi.mocked(mockGrpcClient.Query).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: QueryPointsResponse) => void) => {
          callback(createServiceError(14, 'Service unavailable'), { results: [] });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.query('test', [0.1])).rejects.toThrow(ConnectionError);
    });
  });
});

describe('CollectionsClient', () => {
  let client: CollectionsClient;
  let mockGrpcClient: CollectionsServiceClient;
  let mockMetadata: grpc.Metadata;

  beforeEach(() => {
    mockMetadata = createMockMetadata();
    mockGrpcClient = {
      Create: vi.fn(),
      Get: vi.fn(),
      Delete: vi.fn(),
      List: vi.fn(),
    } as unknown as CollectionsServiceClient;
    client = new CollectionsClient(mockGrpcClient, mockMetadata);
  });

  describe('create', () => {
    it('should create a collection', async () => {
      vi.mocked(mockGrpcClient.Create).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: CreateCollectionResponse) => void) => {
          callback(null, { created: true });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.create('test-collection', 384, 'cosine');

      expect(result).toBe(true);
    });
  });

  describe('get', () => {
    it('should get collection info', async () => {
      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetCollectionResponse) => void) => {
          callback(null, {
            name: 'test-collection',
            pointsCount: 1000,
            dimension: 384,
            distance: 'cosine',
          });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.get('test-collection');

      expect(result.name).toBe('test-collection');
      expect(result.pointsCount).toBe(1000);
      expect(result.dimension).toBe(384);
      expect(result.distance).toBe('cosine');
    });
  });

  describe('delete', () => {
    it('should delete a collection', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: DeleteCollectionResponse) => void) => {
          callback(null, { deleted: true });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.delete('test-collection');

      expect(result).toBe(true);
    });
  });

  describe('list', () => {
    it('should list all collections', async () => {
      vi.mocked(mockGrpcClient.List).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ListCollectionsResponse) => void) => {
          callback(null, { collections: ['col1', 'col2', 'col3'] });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.list();

      expect(result).toEqual(['col1', 'col2', 'col3']);
    });
  });

  describe('exists', () => {
    it('should return true if collection exists', async () => {
      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetCollectionResponse) => void) => {
          callback(null, { name: 'test', pointsCount: 0, dimension: 384, distance: 'cosine' });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.exists('test');

      expect(result).toBe(true);
    });

    it('should return false if collection does not exist', async () => {
      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetCollectionResponse) => void) => {
          callback(createServiceError(5, 'Not found'), {} as GetCollectionResponse);
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.exists('missing');

      expect(result).toBe(false);
    });

    it('should propagate other errors', async () => {
      vi.mocked(mockGrpcClient.Get).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: GetCollectionResponse) => void) => {
          callback(createServiceError(14, 'Service unavailable'), {} as GetCollectionResponse);
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.exists('test')).rejects.toThrow(ConnectionError);
    });
  });

  describe('error handling', () => {
    it('should handle ALREADY_EXISTS error', async () => {
      vi.mocked(mockGrpcClient.Create).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: CreateCollectionResponse) => void) => {
          callback(createServiceError(6, 'Collection already exists'), { created: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.create('existing', 384)).rejects.toThrow(InvalidArgumentError);
    });
  });
});
