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
  });
});
