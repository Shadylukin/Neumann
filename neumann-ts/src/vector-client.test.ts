// SPDX-License-Identifier: MIT
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { VectorClient } from './vector-client.js';
import { ConnectionError } from './types/errors.js';

// Mock the grpc module
vi.mock('@grpc/grpc-js', () => ({
  credentials: {
    createInsecure: vi.fn(() => ({})),
    createSsl: vi.fn(() => ({})),
  },
  Metadata: vi.fn(() => ({
    set: vi.fn(),
    get: vi.fn(),
    clone: vi.fn(),
  })),
}));

// Mock the grpc module loading
vi.mock('./grpc.js', () => ({
  loadVectorProto: vi.fn(async () => ({
    PointsService: vi.fn(() => ({
      close: vi.fn(),
    })),
    CollectionsService: vi.fn(() => ({
      close: vi.fn(),
    })),
  })),
  getPointsServiceClient: vi.fn(() => ({
    Upsert: vi.fn(),
    Get: vi.fn(),
    Delete: vi.fn(),
    Query: vi.fn(),
    Scroll: vi.fn(),
    close: vi.fn(),
  })),
  getCollectionsServiceClient: vi.fn(() => ({
    Create: vi.fn(),
    Get: vi.fn(),
    Delete: vi.fn(),
    List: vi.fn(),
    close: vi.fn(),
  })),
}));

describe('VectorClient', () => {
  describe('connect', () => {
    it('should connect with default options', async () => {
      const client = await VectorClient.connect('localhost:9200');

      expect(client).toBeInstanceOf(VectorClient);
      expect(client.isConnected).toBe(true);

      client.close();
    });

    it('should connect with TLS', async () => {
      const client = await VectorClient.connect('localhost:9200', { tls: true });

      expect(client.isConnected).toBe(true);

      client.close();
    });

    it('should connect with API key', async () => {
      const client = await VectorClient.connect('localhost:9200', {
        apiKey: 'test-api-key',
      });

      expect(client.isConnected).toBe(true);

      client.close();
    });

    it('should connect with custom metadata', async () => {
      const client = await VectorClient.connect('localhost:9200', {
        metadata: { 'x-custom-header': 'value' },
      });

      expect(client.isConnected).toBe(true);

      client.close();
    });
  });

  describe('close', () => {
    it('should close the connection', async () => {
      const client = await VectorClient.connect('localhost:9200');
      expect(client.isConnected).toBe(true);

      client.close();

      expect(client.isConnected).toBe(false);
    });

    it('should throw ConnectionError when using closed client', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.listCollections()).rejects.toThrow(ConnectionError);
    });
  });

  describe('collection operations', () => {
    let client: VectorClient;

    beforeEach(async () => {
      client = await VectorClient.connect('localhost:9200');
    });

    afterEach(() => {
      client.close();
    });

    it('should have access to points client', () => {
      expect(client.points).toBeDefined();
    });

    it('should have access to collections client', () => {
      expect(client.collections).toBeDefined();
    });
  });

  describe('connection checks', () => {
    it('should throw when not connected for createCollection', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.createCollection('test', 384)).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for getCollection', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.getCollection('test')).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for deleteCollection', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.deleteCollection('test')).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for listCollections', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.listCollections()).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for collectionExists', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.collectionExists('test')).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for upsertPoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.upsertPoints('test', [])).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for getPoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.getPoints('test', ['p1'])).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for deletePoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.deletePoints('test', ['p1'])).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for queryPoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.queryPoints('test', [0.1])).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for scrollPoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.scrollPoints('test')).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for scrollAllPoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      const iterator = client.scrollAllPoints('test')[Symbol.asyncIterator]();
      await expect(iterator.next()).rejects.toThrow(ConnectionError);
    });

    it('should throw when not connected for countPoints', async () => {
      const client = await VectorClient.connect('localhost:9200');
      client.close();

      await expect(client.countPoints('test')).rejects.toThrow(ConnectionError);
    });
  });
});

