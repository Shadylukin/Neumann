// SPDX-License-Identifier: MIT
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HealthClient, HealthStatus } from './health.js';
import { ConnectionError, InternalError } from '../types/errors.js';
// Mock gRPC Metadata
function createMockMetadata() {
    return {
        set: vi.fn(),
        get: vi.fn(),
        clone: vi.fn(),
    };
}
// Create mock service error
function createServiceError(code, details) {
    return {
        code,
        details,
        message: details,
        name: 'ServiceError',
        metadata: {},
    };
}
describe('HealthClient', () => {
    let client;
    let mockGrpcClient;
    let mockMetadata;
    beforeEach(() => {
        mockMetadata = createMockMetadata();
        mockGrpcClient = {
            Check: vi.fn(),
        };
        client = new HealthClient(mockGrpcClient, mockMetadata);
    });
    describe('check', () => {
        it('should return healthy status when serving', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_SERVING' });
                return {};
            });
            const result = await client.check();
            expect(result.status).toBe(HealthStatus.Serving);
            expect(result.healthy).toBe(true);
        });
        it('should return not serving status', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_NOT_SERVING' });
                return {};
            });
            const result = await client.check();
            expect(result.status).toBe(HealthStatus.NotServing);
            expect(result.healthy).toBe(false);
        });
        it('should return unknown status for unspecified', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_UNSPECIFIED' });
                return {};
            });
            const result = await client.check();
            expect(result.status).toBe(HealthStatus.Unknown);
            expect(result.healthy).toBe(false);
        });
        it('should handle numeric status values', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 1 });
                return {};
            });
            const result = await client.check();
            expect(result.status).toBe(HealthStatus.Serving);
            expect(result.healthy).toBe(true);
        });
        it('should check specific service', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_SERVING' });
                return {};
            });
            const result = await client.check('query');
            expect(result.healthy).toBe(true);
        });
        it('should handle errors', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), {});
                return {};
            });
            await expect(client.check()).rejects.toThrow(ConnectionError);
        });
    });
    describe('isHealthy', () => {
        it('should return true when healthy', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_SERVING' });
                return {};
            });
            const result = await client.isHealthy();
            expect(result).toBe(true);
        });
        it('should return false when not healthy', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_NOT_SERVING' });
                return {};
            });
            const result = await client.isHealthy();
            expect(result).toBe(false);
        });
        it('should return false on error', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), {});
                return {};
            });
            const result = await client.isHealthy();
            expect(result).toBe(false);
        });
    });
    describe('error handling', () => {
        it('should handle unknown errors as InternalError', async () => {
            vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(99, 'Unknown error'), {});
                return {};
            });
            await expect(client.check()).rejects.toThrow(InternalError);
        });
    });
});
//# sourceMappingURL=health.test.js.map