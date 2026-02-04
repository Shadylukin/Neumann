// SPDX-License-Identifier: MIT
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type * as grpc from '@grpc/grpc-js';
import { BlobClient } from './blob.js';
import type { BlobServiceClient, BlobUploadResponse, ArtifactInfo, BlobDownloadChunk } from '../generated/neumann.js';
import { ConnectionError, NotFoundError, InvalidArgumentError, InternalError } from '../types/errors.js';

// Mock gRPC Metadata
function createMockMetadata(): grpc.Metadata {
  return {
    set: vi.fn(),
    get: vi.fn(),
    clone: vi.fn(),
  } as unknown as grpc.Metadata;
}

// Mock writable stream for uploads
function createMockWritableStream(response: BlobUploadResponse) {
  const stream = {
    write: vi.fn(),
    end: vi.fn(),
    destroy: vi.fn(),
    on: vi.fn(),
  };

  // Capture the callback and call it when end() is called
  let callback: ((err: grpc.ServiceError | null, response: BlobUploadResponse) => void) | null = null;

  return {
    stream,
    setCallback: (cb: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
      callback = cb;
    },
    triggerResponse: () => {
      if (callback) {
        callback(null, response);
      }
    },
    triggerError: (err: grpc.ServiceError) => {
      if (callback) {
        callback(err, {} as BlobUploadResponse);
      }
    },
  };
}

// Mock readable stream for downloads
function createMockReadableStream(chunks: BlobDownloadChunk[]): grpc.ClientReadableStream<BlobDownloadChunk> {
  let index = 0;

  const stream = {
    async *[Symbol.asyncIterator]() {
      while (index < chunks.length) {
        yield chunks[index++];
      }
    },
  };

  return stream as unknown as grpc.ClientReadableStream<BlobDownloadChunk>;
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

describe('BlobClient', () => {
  let client: BlobClient;
  let mockGrpcClient: BlobServiceClient;
  let mockMetadata: grpc.Metadata;

  beforeEach(() => {
    mockMetadata = createMockMetadata();
    mockGrpcClient = {
      Upload: vi.fn(),
      Download: vi.fn(),
      Delete: vi.fn(),
      GetMetadata: vi.fn(),
    } as unknown as BlobServiceClient;
    client = new BlobClient(mockGrpcClient, mockMetadata);
  });

  describe('uploadBlob', () => {
    it('should upload a blob and return artifact info', async () => {
      const response: BlobUploadResponse = {
        artifactId: 'artifact-123',
        size: 1024,
        checksum: 'abc123',
      };

      const mockStream = createMockWritableStream(response);
      vi.mocked(mockGrpcClient.Upload).mockImplementation(
        (_metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
          mockStream.setCallback(callback);
          // Simulate async response after stream ends
          setTimeout(() => mockStream.triggerResponse(), 0);
          return mockStream.stream as unknown as grpc.ClientWritableStream<unknown>;
        }
      );

      const result = await client.uploadBlob('test.txt', Buffer.from('hello world'), {
        contentType: 'text/plain',
        tags: ['test'],
      });

      expect(result).toEqual({
        artifactId: 'artifact-123',
        size: 1024,
        checksum: 'abc123',
      });
      expect(mockStream.stream.write).toHaveBeenCalled();
      expect(mockStream.stream.end).toHaveBeenCalled();
    });

    it('should upload with createdBy option', async () => {
      const response: BlobUploadResponse = {
        artifactId: 'artifact-456',
        size: 512,
        checksum: 'def456',
      };

      const mockStream = createMockWritableStream(response);
      vi.mocked(mockGrpcClient.Upload).mockImplementation(
        (_metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
          mockStream.setCallback(callback);
          setTimeout(() => mockStream.triggerResponse(), 0);
          return mockStream.stream as unknown as grpc.ClientWritableStream<unknown>;
        }
      );

      const result = await client.uploadBlob('test.txt', Buffer.from('data'), {
        createdBy: 'user-123',
        linkedTo: ['artifact-1'],
        custom: { key: 'value' },
      });

      expect(result.artifactId).toBe('artifact-456');
      expect(mockStream.stream.write).toHaveBeenCalled();
    });

    it('should handle upload errors', async () => {
      const mockStream = createMockWritableStream({} as BlobUploadResponse);
      vi.mocked(mockGrpcClient.Upload).mockImplementation(
        (_metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
          mockStream.setCallback(callback);
          setTimeout(() => mockStream.triggerError(createServiceError(5, 'Not found')), 0);
          return mockStream.stream as unknown as grpc.ClientWritableStream<unknown>;
        }
      );

      await expect(client.uploadBlob('test.txt', Buffer.from('data'))).rejects.toThrow(NotFoundError);
    });
  });

  describe('uploadBlobStreaming', () => {
    it('should upload blob from async iterable', async () => {
      const response: BlobUploadResponse = {
        artifactId: 'artifact-stream',
        size: 2048,
        checksum: 'stream123',
      };

      const mockStream = createMockWritableStream(response);
      vi.mocked(mockGrpcClient.Upload).mockImplementation(
        (_metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
          mockStream.setCallback(callback);
          setTimeout(() => mockStream.triggerResponse(), 10);
          return mockStream.stream as unknown as grpc.ClientWritableStream<unknown>;
        }
      );

      async function* generateChunks(): AsyncIterable<Uint8Array> {
        yield new Uint8Array([1, 2, 3]);
        yield new Uint8Array([4, 5, 6]);
      }

      const result = await client.uploadBlobStreaming('stream.bin', generateChunks());

      expect(result.artifactId).toBe('artifact-stream');
      expect(mockStream.stream.write).toHaveBeenCalled();
    });

    it('should handle streaming upload errors', async () => {
      const mockStream = createMockWritableStream({} as BlobUploadResponse);
      vi.mocked(mockGrpcClient.Upload).mockImplementation(
        (_metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
          mockStream.setCallback(callback);
          setTimeout(() => mockStream.triggerError(createServiceError(14, 'Unavailable')), 0);
          return mockStream.stream as unknown as grpc.ClientWritableStream<unknown>;
        }
      );

      async function* generateChunks(): AsyncIterable<Uint8Array> {
        yield new Uint8Array([1, 2, 3]);
      }

      await expect(client.uploadBlobStreaming('stream.bin', generateChunks())).rejects.toThrow(ConnectionError);
    });

    it('should handle chunk iteration errors', async () => {
      const mockStream = createMockWritableStream({} as BlobUploadResponse);
      vi.mocked(mockGrpcClient.Upload).mockImplementation(
        (_metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: BlobUploadResponse) => void) => {
          mockStream.setCallback(callback);
          // Trigger error after a short delay to allow iteration to fail first
          setTimeout(() => mockStream.triggerError(createServiceError(13, 'Internal')), 50);
          return mockStream.stream as unknown as grpc.ClientWritableStream<unknown>;
        }
      );

      async function* failingGenerator(): AsyncIterable<Uint8Array> {
        yield new Uint8Array([1, 2, 3]);
        throw new Error('Iteration failed');
      }

      await expect(client.uploadBlobStreaming('stream.bin', failingGenerator())).rejects.toThrow();
      expect(mockStream.stream.destroy).toHaveBeenCalled();
    }, 10000);
  });

  describe('downloadBlob', () => {
    it('should download a blob as chunks', async () => {
      const chunks: BlobDownloadChunk[] = [
        { data: new Uint8Array([1, 2, 3]), isFinal: false },
        { data: new Uint8Array([4, 5, 6]), isFinal: false },
        { data: new Uint8Array([7, 8, 9]), isFinal: true },
      ];

      const mockStream = createMockReadableStream(chunks);
      vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream);

      const receivedChunks: Uint8Array[] = [];
      for await (const chunk of client.downloadBlob('artifact-123')) {
        receivedChunks.push(chunk);
      }

      expect(receivedChunks).toHaveLength(3);
      expect(receivedChunks[0]).toEqual(new Uint8Array([1, 2, 3]));
    });

    it('should skip empty chunks', async () => {
      const chunks: BlobDownloadChunk[] = [
        { data: new Uint8Array([1, 2, 3]), isFinal: false },
        { data: new Uint8Array([]), isFinal: false },
        { data: new Uint8Array([4, 5, 6]), isFinal: true },
      ];

      const mockStream = createMockReadableStream(chunks);
      vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream);

      const receivedChunks: Uint8Array[] = [];
      for await (const chunk of client.downloadBlob('artifact-123')) {
        receivedChunks.push(chunk);
      }

      expect(receivedChunks).toHaveLength(2);
    });

    it('should call cancel on stream when available', async () => {
      const chunks: BlobDownloadChunk[] = [
        { data: new Uint8Array([1, 2, 3]), isFinal: true },
      ];

      const cancelFn = vi.fn();
      const mockStream = {
        async *[Symbol.asyncIterator]() {
          for (const chunk of chunks) {
            yield chunk;
          }
        },
        cancel: cancelFn,
      };

      vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream as unknown as grpc.ClientReadableStream<BlobDownloadChunk>);

      const receivedChunks: Uint8Array[] = [];
      for await (const chunk of client.downloadBlob('artifact-123')) {
        receivedChunks.push(chunk);
      }

      expect(cancelFn).toHaveBeenCalled();
    });

    it('should handle cancel errors gracefully', async () => {
      const chunks: BlobDownloadChunk[] = [
        { data: new Uint8Array([1, 2, 3]), isFinal: true },
      ];

      const mockStream = {
        async *[Symbol.asyncIterator]() {
          for (const chunk of chunks) {
            yield chunk;
          }
        },
        cancel: () => { throw new Error('Cancel failed'); },
      };

      vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream as unknown as grpc.ClientReadableStream<BlobDownloadChunk>);

      // Should not throw even though cancel throws
      const receivedChunks: Uint8Array[] = [];
      for await (const chunk of client.downloadBlob('artifact-123')) {
        receivedChunks.push(chunk);
      }

      expect(receivedChunks).toHaveLength(1);
    });
  });

  describe('downloadBlobFull', () => {
    it('should download a complete blob', async () => {
      const chunks: BlobDownloadChunk[] = [
        { data: new Uint8Array([1, 2, 3]), isFinal: false },
        { data: new Uint8Array([4, 5, 6]), isFinal: true },
      ];

      const mockStream = createMockReadableStream(chunks);
      vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream);

      const result = await client.downloadBlobFull('artifact-123');

      expect(result).toBeInstanceOf(Buffer);
      expect(result).toEqual(Buffer.from([1, 2, 3, 4, 5, 6]));
    });
  });

  describe('deleteBlob', () => {
    it('should delete a blob', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback(null, { success: true });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.deleteBlob('artifact-123');

      expect(result).toBe(true);
    });

    it('should handle delete errors', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback(createServiceError(5, 'Not found'), { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow(NotFoundError);
    });
  });

  describe('getBlobMetadata', () => {
    it('should get blob metadata', async () => {
      const artifactInfo: ArtifactInfo = {
        id: 'artifact-123',
        filename: 'test.txt',
        contentType: 'text/plain',
        size: 1024,
        checksum: 'abc123',
        chunkCount: 1,
        created: Date.now(),
        modified: Date.now(),
        createdBy: 'user-1',
        tags: ['test'],
        linkedTo: [],
      };

      vi.mocked(mockGrpcClient.GetMetadata).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ArtifactInfo) => void) => {
          callback(null, artifactInfo);
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.getBlobMetadata('artifact-123');

      expect(result.id).toBe('artifact-123');
      expect(result.filename).toBe('test.txt');
      expect(result.contentType).toBe('text/plain');
    });

    it('should handle getBlobMetadata errors', async () => {
      vi.mocked(mockGrpcClient.GetMetadata).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ArtifactInfo) => void) => {
          callback(createServiceError(5, 'Artifact not found'), {} as ArtifactInfo);
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.getBlobMetadata('nonexistent')).rejects.toThrow(NotFoundError);
    });

    it('should handle metadata without custom field', async () => {
      const artifactInfo = {
        id: 'artifact-123',
        filename: 'test.txt',
        contentType: 'text/plain',
        size: 1024,
        checksum: 'abc123',
        chunkCount: 1,
        created: Date.now(),
        modified: Date.now(),
        createdBy: 'user-1',
        tags: ['test'],
        linkedTo: [],
        // custom is undefined
      } as ArtifactInfo;

      vi.mocked(mockGrpcClient.GetMetadata).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: ArtifactInfo) => void) => {
          callback(null, artifactInfo);
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.getBlobMetadata('artifact-123');

      expect(result.custom).toEqual({});
    });
  });

  describe('error handling', () => {
    it('should handle UNAVAILABLE error', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback(createServiceError(14, 'Service unavailable'), { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow(ConnectionError);
    });

    it('should handle INVALID_ARGUMENT error', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback(createServiceError(3, 'Invalid argument'), { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow(InvalidArgumentError);
    });

    it('should handle unknown errors as InternalError', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback(createServiceError(99, 'Unknown error'), { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow(InternalError);
    });

    it('should use default message when NOT_FOUND error has empty details', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback({ code: 5, details: '', message: '', name: 'Error', metadata: {} as grpc.Metadata } as grpc.ServiceError, { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow('Artifact not found');
    });

    it('should use default message when INVALID_ARGUMENT error has empty details', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback({ code: 3, details: '', message: '', name: 'Error', metadata: {} as grpc.Metadata } as grpc.ServiceError, { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow('Invalid argument');
    });

    it('should use default message when UNAVAILABLE error has empty details', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback({ code: 14, details: '', message: '', name: 'Error', metadata: {} as grpc.Metadata } as grpc.ServiceError, { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow('Service unavailable');
    });

    it('should use message fallback when unknown error has empty details', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback({ code: 99, details: '', message: 'fallback message', name: 'Error', metadata: {} as grpc.Metadata } as unknown as grpc.ServiceError, { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow('fallback message');
    });

    it('should use Internal error when unknown error has no message', async () => {
      vi.mocked(mockGrpcClient.Delete).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: { success: boolean }) => void) => {
          callback({ code: 99, details: '', message: '', name: 'Error', metadata: {} as grpc.Metadata } as unknown as grpc.ServiceError, { success: false });
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.deleteBlob('artifact-123')).rejects.toThrow('Internal error');
    });
  });
});
