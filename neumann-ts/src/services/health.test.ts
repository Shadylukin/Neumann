// SPDX-License-Identifier: MIT
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type * as grpc from '@grpc/grpc-js';
import { HealthClient, HealthStatus } from './health.js';
import type { HealthClient as GrpcHealthClient, HealthCheckResponse, ServingStatus } from '../generated/neumann.js';
import { ConnectionError, InternalError } from '../types/errors.js';

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

describe('HealthClient', () => {
  let client: HealthClient;
  let mockGrpcClient: GrpcHealthClient;
  let mockMetadata: grpc.Metadata;

  beforeEach(() => {
    mockMetadata = createMockMetadata();
    mockGrpcClient = {
      Check: vi.fn(),
    } as unknown as GrpcHealthClient;
    client = new HealthClient(mockGrpcClient, mockMetadata);
  });

  describe('check', () => {
    it('should return healthy status when serving', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 'SERVING_STATUS_SERVING' as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.check();

      expect(result.status).toBe(HealthStatus.Serving);
      expect(result.healthy).toBe(true);
    });

    it('should return not serving status', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 'SERVING_STATUS_NOT_SERVING' as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.check();

      expect(result.status).toBe(HealthStatus.NotServing);
      expect(result.healthy).toBe(false);
    });

    it('should return unknown status for unspecified', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 'SERVING_STATUS_UNSPECIFIED' as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.check();

      expect(result.status).toBe(HealthStatus.Unknown);
      expect(result.healthy).toBe(false);
    });

    it('should handle numeric status values', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 1 as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.check();

      expect(result.status).toBe(HealthStatus.Serving);
      expect(result.healthy).toBe(true);
    });

    it('should check specific service', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 'SERVING_STATUS_SERVING' as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.check('query');

      expect(result.healthy).toBe(true);
    });

    it('should handle errors', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(createServiceError(14, 'Service unavailable'), {} as HealthCheckResponse);
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.check()).rejects.toThrow(ConnectionError);
    });
  });

  describe('isHealthy', () => {
    it('should return true when healthy', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 'SERVING_STATUS_SERVING' as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.isHealthy();

      expect(result).toBe(true);
    });

    it('should return false when not healthy', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(null, { status: 'SERVING_STATUS_NOT_SERVING' as unknown as ServingStatus });
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.isHealthy();

      expect(result).toBe(false);
    });

    it('should return false on error', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(createServiceError(14, 'Service unavailable'), {} as HealthCheckResponse);
          return {} as grpc.ClientUnaryCall;
        }
      );

      const result = await client.isHealthy();

      expect(result).toBe(false);
    });
  });

  describe('error handling', () => {
    it('should handle unknown errors as InternalError', async () => {
      vi.mocked(mockGrpcClient.Check).mockImplementation(
        (_request: unknown, _metadata: grpc.Metadata, callback: (err: grpc.ServiceError | null, response: HealthCheckResponse) => void) => {
          callback(createServiceError(99, 'Unknown error'), {} as HealthCheckResponse);
          return {} as grpc.ClientUnaryCall;
        }
      );

      await expect(client.check()).rejects.toThrow(InternalError);
    });
  });
});
