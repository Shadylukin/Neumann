"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// SPDX-License-Identifier: MIT
const vitest_1 = require("vitest");
const health_js_1 = require("./health.js");
const errors_js_1 = require("../types/errors.js");
// Mock gRPC Metadata
function createMockMetadata() {
    return {
        set: vitest_1.vi.fn(),
        get: vitest_1.vi.fn(),
        clone: vitest_1.vi.fn(),
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
(0, vitest_1.describe)('HealthClient', () => {
    let client;
    let mockGrpcClient;
    let mockMetadata;
    (0, vitest_1.beforeEach)(() => {
        mockMetadata = createMockMetadata();
        mockGrpcClient = {
            Check: vitest_1.vi.fn(),
        };
        client = new health_js_1.HealthClient(mockGrpcClient, mockMetadata);
    });
    (0, vitest_1.describe)('check', () => {
        (0, vitest_1.it)('should return healthy status when serving', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_SERVING' });
                return {};
            });
            const result = await client.check();
            (0, vitest_1.expect)(result.status).toBe(health_js_1.HealthStatus.Serving);
            (0, vitest_1.expect)(result.healthy).toBe(true);
        });
        (0, vitest_1.it)('should return not serving status', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_NOT_SERVING' });
                return {};
            });
            const result = await client.check();
            (0, vitest_1.expect)(result.status).toBe(health_js_1.HealthStatus.NotServing);
            (0, vitest_1.expect)(result.healthy).toBe(false);
        });
        (0, vitest_1.it)('should return unknown status for unspecified', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_UNSPECIFIED' });
                return {};
            });
            const result = await client.check();
            (0, vitest_1.expect)(result.status).toBe(health_js_1.HealthStatus.Unknown);
            (0, vitest_1.expect)(result.healthy).toBe(false);
        });
        (0, vitest_1.it)('should handle numeric status values', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 1 });
                return {};
            });
            const result = await client.check();
            (0, vitest_1.expect)(result.status).toBe(health_js_1.HealthStatus.Serving);
            (0, vitest_1.expect)(result.healthy).toBe(true);
        });
        (0, vitest_1.it)('should check specific service', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_SERVING' });
                return {};
            });
            const result = await client.check('query');
            (0, vitest_1.expect)(result.healthy).toBe(true);
        });
        (0, vitest_1.it)('should handle errors', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), {});
                return {};
            });
            await (0, vitest_1.expect)(client.check()).rejects.toThrow(errors_js_1.ConnectionError);
        });
    });
    (0, vitest_1.describe)('isHealthy', () => {
        (0, vitest_1.it)('should return true when healthy', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_SERVING' });
                return {};
            });
            const result = await client.isHealthy();
            (0, vitest_1.expect)(result).toBe(true);
        });
        (0, vitest_1.it)('should return false when not healthy', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(null, { status: 'SERVING_STATUS_NOT_SERVING' });
                return {};
            });
            const result = await client.isHealthy();
            (0, vitest_1.expect)(result).toBe(false);
        });
        (0, vitest_1.it)('should return false on error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), {});
                return {};
            });
            const result = await client.isHealthy();
            (0, vitest_1.expect)(result).toBe(false);
        });
    });
    (0, vitest_1.describe)('error handling', () => {
        (0, vitest_1.it)('should handle unknown errors as InternalError', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Check).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(99, 'Unknown error'), {});
                return {};
            });
            await (0, vitest_1.expect)(client.check()).rejects.toThrow(errors_js_1.InternalError);
        });
    });
});
//# sourceMappingURL=health.test.js.map