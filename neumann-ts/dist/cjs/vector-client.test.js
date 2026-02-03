"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// SPDX-License-Identifier: MIT
const vitest_1 = require("vitest");
const vector_client_js_1 = require("./vector-client.js");
const errors_js_1 = require("./types/errors.js");
// Mock the grpc module
vitest_1.vi.mock('@grpc/grpc-js', () => ({
    credentials: {
        createInsecure: vitest_1.vi.fn(() => ({})),
        createSsl: vitest_1.vi.fn(() => ({})),
    },
    Metadata: vitest_1.vi.fn(() => ({
        set: vitest_1.vi.fn(),
        get: vitest_1.vi.fn(),
        clone: vitest_1.vi.fn(),
    })),
}));
// Mock the grpc module loading
vitest_1.vi.mock('./grpc.js', () => ({
    loadVectorProto: vitest_1.vi.fn(async () => ({
        PointsService: vitest_1.vi.fn(() => ({
            close: vitest_1.vi.fn(),
        })),
        CollectionsService: vitest_1.vi.fn(() => ({
            close: vitest_1.vi.fn(),
        })),
    })),
    getPointsServiceClient: vitest_1.vi.fn(() => ({
        Upsert: vitest_1.vi.fn(),
        Get: vitest_1.vi.fn(),
        Delete: vitest_1.vi.fn(),
        Query: vitest_1.vi.fn(),
        Scroll: vitest_1.vi.fn(),
        close: vitest_1.vi.fn(),
    })),
    getCollectionsServiceClient: vitest_1.vi.fn(() => ({
        Create: vitest_1.vi.fn(),
        Get: vitest_1.vi.fn(),
        Delete: vitest_1.vi.fn(),
        List: vitest_1.vi.fn(),
        close: vitest_1.vi.fn(),
    })),
}));
(0, vitest_1.describe)('VectorClient', () => {
    (0, vitest_1.describe)('connect', () => {
        (0, vitest_1.it)('should connect with default options', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            (0, vitest_1.expect)(client).toBeInstanceOf(vector_client_js_1.VectorClient);
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should connect with TLS', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200', { tls: true });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should connect with API key', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200', {
                apiKey: 'test-api-key',
            });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should connect with custom metadata', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200', {
                metadata: { 'x-custom-header': 'value' },
            });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
    });
    (0, vitest_1.describe)('close', () => {
        (0, vitest_1.it)('should close the connection', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
            (0, vitest_1.expect)(client.isConnected).toBe(false);
        });
        (0, vitest_1.it)('should throw ConnectionError when using closed client', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.listCollections()).rejects.toThrow(errors_js_1.ConnectionError);
        });
    });
    (0, vitest_1.describe)('collection operations', () => {
        let client;
        (0, vitest_1.beforeEach)(async () => {
            client = await vector_client_js_1.VectorClient.connect('localhost:9200');
        });
        (0, vitest_1.afterEach)(() => {
            client.close();
        });
        (0, vitest_1.it)('should have access to points client', () => {
            (0, vitest_1.expect)(client.points).toBeDefined();
        });
        (0, vitest_1.it)('should have access to collections client', () => {
            (0, vitest_1.expect)(client.collections).toBeDefined();
        });
    });
    (0, vitest_1.describe)('connection checks', () => {
        (0, vitest_1.it)('should throw when not connected for createCollection', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.createCollection('test', 384)).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for getCollection', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.getCollection('test')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for deleteCollection', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.deleteCollection('test')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for listCollections', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.listCollections()).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for collectionExists', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.collectionExists('test')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for upsertPoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.upsertPoints('test', [])).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for getPoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.getPoints('test', ['p1'])).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for deletePoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.deletePoints('test', ['p1'])).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for queryPoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.queryPoints('test', [0.1])).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for scrollPoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.scrollPoints('test')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for scrollAllPoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            const iterator = client.scrollAllPoints('test')[Symbol.asyncIterator]();
            await (0, vitest_1.expect)(iterator.next()).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should throw when not connected for countPoints', async () => {
            const client = await vector_client_js_1.VectorClient.connect('localhost:9200');
            client.close();
            await (0, vitest_1.expect)(client.countPoints('test')).rejects.toThrow(errors_js_1.ConnectionError);
        });
    });
});
//# sourceMappingURL=vector-client.test.js.map