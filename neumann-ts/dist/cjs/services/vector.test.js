"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// SPDX-License-Identifier: MIT
const vitest_1 = require("vitest");
const vector_js_1 = require("./vector.js");
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
(0, vitest_1.describe)('PointsClient', () => {
    let client;
    let mockGrpcClient;
    let mockMetadata;
    (0, vitest_1.beforeEach)(() => {
        mockMetadata = createMockMetadata();
        mockGrpcClient = {
            Upsert: vitest_1.vi.fn(),
            Get: vitest_1.vi.fn(),
            Delete: vitest_1.vi.fn(),
            Query: vitest_1.vi.fn(),
            Scroll: vitest_1.vi.fn(),
        };
        client = new vector_js_1.PointsClient(mockGrpcClient, mockMetadata);
    });
    (0, vitest_1.describe)('upsert', () => {
        (0, vitest_1.it)('should upsert points', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Upsert).mockImplementation((_request, _metadata, callback) => {
                callback(null, { upserted: 2 });
                return {};
            });
            const result = await client.upsert('test-collection', [
                { id: 'p1', vector: [0.1, 0.2, 0.3] },
                { id: 'p2', vector: [0.4, 0.5, 0.6], payload: { name: 'test' } },
            ]);
            (0, vitest_1.expect)(result).toBe(2);
        });
    });
    (0, vitest_1.describe)('get', () => {
        (0, vitest_1.it)('should get points by IDs', async () => {
            const points = [
                { id: 'p1', vector: [0.1, 0.2, 0.3] },
                { id: 'p2', vector: [0.4, 0.5, 0.6] },
            ];
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(null, { points });
                return {};
            });
            const result = await client.get('test-collection', ['p1', 'p2']);
            (0, vitest_1.expect)(result).toHaveLength(2);
            (0, vitest_1.expect)(result[0]?.id).toBe('p1');
            (0, vitest_1.expect)(result[0]?.vector).toEqual([0.1, 0.2, 0.3]);
        });
        (0, vitest_1.it)('should decode payload', async () => {
            const payload = new TextEncoder().encode(JSON.stringify('hello'));
            const points = [
                { id: 'p1', vector: [0.1], payload: { name: payload } },
            ];
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(null, { points });
                return {};
            });
            const result = await client.get('test-collection', ['p1']);
            (0, vitest_1.expect)(result[0]?.payload?.name).toBe('hello');
        });
    });
    (0, vitest_1.describe)('delete', () => {
        (0, vitest_1.it)('should delete points by IDs', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(null, { deleted: 2 });
                return {};
            });
            const result = await client.delete('test-collection', ['p1', 'p2']);
            (0, vitest_1.expect)(result).toBe(2);
        });
    });
    (0, vitest_1.describe)('query', () => {
        (0, vitest_1.it)('should query similar points', async () => {
            const results = [
                { id: 'p1', score: 0.95 },
                { id: 'p2', score: 0.85 },
            ];
            vitest_1.vi.mocked(mockGrpcClient.Query).mockImplementation((_request, _metadata, callback) => {
                callback(null, { results });
                return {};
            });
            const result = await client.query('test-collection', [0.1, 0.2, 0.3], { limit: 10 });
            (0, vitest_1.expect)(result).toHaveLength(2);
            (0, vitest_1.expect)(result[0]?.id).toBe('p1');
            (0, vitest_1.expect)(result[0]?.score).toBe(0.95);
        });
        (0, vitest_1.it)('should include vector when requested', async () => {
            const results = [
                { id: 'p1', score: 0.95, vector: [0.1, 0.2, 0.3] },
            ];
            vitest_1.vi.mocked(mockGrpcClient.Query).mockImplementation((_request, _metadata, callback) => {
                callback(null, { results });
                return {};
            });
            const result = await client.query('test-collection', [0.1, 0.2, 0.3], { withVector: true });
            (0, vitest_1.expect)(result[0]?.vector).toEqual([0.1, 0.2, 0.3]);
        });
    });
    (0, vitest_1.describe)('scroll', () => {
        (0, vitest_1.it)('should scroll through points', async () => {
            const points = [
                { id: 'p1', vector: [0.1, 0.2, 0.3] },
                { id: 'p2', vector: [0.4, 0.5, 0.6] },
            ];
            vitest_1.vi.mocked(mockGrpcClient.Scroll).mockImplementation((_request, _metadata, callback) => {
                callback(null, { points, nextOffset: 'p3' });
                return {};
            });
            const result = await client.scroll('test-collection', { limit: 2 });
            (0, vitest_1.expect)(result.points).toHaveLength(2);
            (0, vitest_1.expect)(result.nextOffset).toBe('p3');
        });
        (0, vitest_1.it)('should return undefined nextOffset when no more pages', async () => {
            const points = [
                { id: 'p1', vector: [0.1] },
            ];
            vitest_1.vi.mocked(mockGrpcClient.Scroll).mockImplementation((_request, _metadata, callback) => {
                callback(null, { points, nextOffset: undefined });
                return {};
            });
            const result = await client.scroll('test-collection');
            (0, vitest_1.expect)(result.nextOffset).toBeUndefined();
        });
    });
    (0, vitest_1.describe)('scrollAll', () => {
        (0, vitest_1.it)('should iterate through all points', async () => {
            let callCount = 0;
            vitest_1.vi.mocked(mockGrpcClient.Scroll).mockImplementation((_request, _metadata, callback) => {
                callCount++;
                if (callCount === 1) {
                    callback(null, {
                        points: [{ id: 'p1', vector: [0.1] }],
                        nextOffset: 'p2',
                    });
                }
                else {
                    callback(null, {
                        points: [{ id: 'p2', vector: [0.2] }],
                        nextOffset: undefined,
                    });
                }
                return {};
            });
            const allPoints = [];
            for await (const point of client.scrollAll('test-collection')) {
                allPoints.push(point);
            }
            (0, vitest_1.expect)(allPoints).toHaveLength(2);
            (0, vitest_1.expect)(allPoints[0]?.id).toBe('p1');
            (0, vitest_1.expect)(allPoints[1]?.id).toBe('p2');
        });
    });
    (0, vitest_1.describe)('error handling', () => {
        (0, vitest_1.it)('should handle NOT_FOUND error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(5, 'Collection not found'), { points: [] });
                return {};
            });
            await (0, vitest_1.expect)(client.get('missing', ['p1'])).rejects.toThrow(errors_js_1.NotFoundError);
        });
        (0, vitest_1.it)('should handle INVALID_ARGUMENT error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Upsert).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(3, 'Invalid vector dimension'), { upserted: 0 });
                return {};
            });
            await (0, vitest_1.expect)(client.upsert('test', [{ id: 'p1', vector: [] }])).rejects.toThrow(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should handle UNAVAILABLE error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Query).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), { results: [] });
                return {};
            });
            await (0, vitest_1.expect)(client.query('test', [0.1])).rejects.toThrow(errors_js_1.ConnectionError);
        });
    });
});
(0, vitest_1.describe)('CollectionsClient', () => {
    let client;
    let mockGrpcClient;
    let mockMetadata;
    (0, vitest_1.beforeEach)(() => {
        mockMetadata = createMockMetadata();
        mockGrpcClient = {
            Create: vitest_1.vi.fn(),
            Get: vitest_1.vi.fn(),
            Delete: vitest_1.vi.fn(),
            List: vitest_1.vi.fn(),
        };
        client = new vector_js_1.CollectionsClient(mockGrpcClient, mockMetadata);
    });
    (0, vitest_1.describe)('create', () => {
        (0, vitest_1.it)('should create a collection', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Create).mockImplementation((_request, _metadata, callback) => {
                callback(null, { created: true });
                return {};
            });
            const result = await client.create('test-collection', 384, 'cosine');
            (0, vitest_1.expect)(result).toBe(true);
        });
    });
    (0, vitest_1.describe)('get', () => {
        (0, vitest_1.it)('should get collection info', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(null, {
                    name: 'test-collection',
                    pointsCount: 1000,
                    dimension: 384,
                    distance: 'cosine',
                });
                return {};
            });
            const result = await client.get('test-collection');
            (0, vitest_1.expect)(result.name).toBe('test-collection');
            (0, vitest_1.expect)(result.pointsCount).toBe(1000);
            (0, vitest_1.expect)(result.dimension).toBe(384);
            (0, vitest_1.expect)(result.distance).toBe('cosine');
        });
    });
    (0, vitest_1.describe)('delete', () => {
        (0, vitest_1.it)('should delete a collection', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(null, { deleted: true });
                return {};
            });
            const result = await client.delete('test-collection');
            (0, vitest_1.expect)(result).toBe(true);
        });
    });
    (0, vitest_1.describe)('list', () => {
        (0, vitest_1.it)('should list all collections', async () => {
            vitest_1.vi.mocked(mockGrpcClient.List).mockImplementation((_request, _metadata, callback) => {
                callback(null, { collections: ['col1', 'col2', 'col3'] });
                return {};
            });
            const result = await client.list();
            (0, vitest_1.expect)(result).toEqual(['col1', 'col2', 'col3']);
        });
    });
    (0, vitest_1.describe)('exists', () => {
        (0, vitest_1.it)('should return true if collection exists', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(null, { name: 'test', pointsCount: 0, dimension: 384, distance: 'cosine' });
                return {};
            });
            const result = await client.exists('test');
            (0, vitest_1.expect)(result).toBe(true);
        });
        (0, vitest_1.it)('should return false if collection does not exist', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(5, 'Not found'), {});
                return {};
            });
            const result = await client.exists('missing');
            (0, vitest_1.expect)(result).toBe(false);
        });
        (0, vitest_1.it)('should propagate other errors', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Get).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), {});
                return {};
            });
            await (0, vitest_1.expect)(client.exists('test')).rejects.toThrow(errors_js_1.ConnectionError);
        });
    });
    (0, vitest_1.describe)('error handling', () => {
        (0, vitest_1.it)('should handle ALREADY_EXISTS error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Create).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(6, 'Collection already exists'), { created: false });
                return {};
            });
            await (0, vitest_1.expect)(client.create('existing', 384)).rejects.toThrow(errors_js_1.InvalidArgumentError);
        });
    });
});
//# sourceMappingURL=vector.test.js.map