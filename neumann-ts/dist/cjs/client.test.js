"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
// SPDX-License-Identifier: MIT
const vitest_1 = require("vitest");
const client_js_1 = require("./client.js");
const errors_js_1 = require("./types/errors.js");
const value_js_1 = require("./types/value.js");
const query_result_js_1 = require("./types/query-result.js");
const validation_js_1 = require("./types/validation.js");
// Mock execute callback response
const mockExecuteResponse = { empty: {} };
const mockExecuteStreamResponses = [
    { row: { row: { columns: [{ name: 'x', value: { intValue: 1 } }] } } },
    { isFinal: true },
];
const mockBatchResponse = { results: [{ empty: {} }, { empty: {} }, { empty: {} }] };
// Mock paginated response
const mockPaginatedResponse = {
    result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: 1 } }] }] } },
    nextCursor: 'cursor123',
    hasMore: true,
    pageSize: 10,
    totalCount: 100,
};
// Mock close cursor response
const mockCloseCursorResponse = { success: true };
// Mock gRPC client
const mockGrpcClient = {
    Execute: vitest_1.vi.fn((request, metadata, callback) => {
        callback(null, mockExecuteResponse);
    }),
    ExecuteStream: vitest_1.vi.fn(() => {
        let index = 0;
        return {
            [Symbol.asyncIterator]: () => ({
                next() {
                    if (index < mockExecuteStreamResponses.length) {
                        return Promise.resolve({ value: mockExecuteStreamResponses[index++], done: false });
                    }
                    return Promise.resolve({ value: undefined, done: true });
                },
            }),
        };
    }),
    ExecuteBatch: vitest_1.vi.fn((request, metadata, callback) => {
        callback(null, mockBatchResponse);
    }),
    ExecutePaginated: vitest_1.vi.fn((request, metadata, callback) => {
        callback(null, mockPaginatedResponse);
    }),
    CloseCursor: vitest_1.vi.fn((request, metadata, callback) => {
        callback(null, mockCloseCursorResponse);
    }),
};
// Mock grpc.ts module
vitest_1.vi.mock('./grpc.js', () => ({
    loadProto: vitest_1.vi.fn().mockResolvedValue({}),
    getQueryServiceClient: vitest_1.vi.fn(() => mockGrpcClient),
}));
// Mock @grpc/grpc-js
vitest_1.vi.mock('@grpc/grpc-js', () => ({
    credentials: {
        createSsl: vitest_1.vi.fn(() => ({})),
        createInsecure: vitest_1.vi.fn(() => ({})),
    },
    Metadata: vitest_1.vi.fn().mockImplementation(() => ({
        set: vitest_1.vi.fn(),
    })),
}));
// Mock grpc-web
vitest_1.vi.mock('grpc-web', () => ({
    GrpcWebClientBase: vitest_1.vi.fn().mockImplementation(() => ({})),
}));
(0, vitest_1.describe)('NeumannClient', () => {
    (0, vitest_1.describe)('query', () => {
        (0, vitest_1.it)('should execute a query using the query method', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.query('SELECT * FROM users');
            (0, vitest_1.expect)(result.type).toBe('empty');
            client.close();
        });
        (0, vitest_1.it)('should pass options to query', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.query('VAULT GET secret', { identity: 'alice' });
            (0, vitest_1.expect)(result.type).toBe('empty');
            client.close();
        });
    });
    (0, vitest_1.describe)('connect', () => {
        (0, vitest_1.it)('should create a connected client', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            (0, vitest_1.expect)(client.clientMode).toBe('remote');
            client.close();
        });
        (0, vitest_1.it)('should support TLS connections', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051', { tls: true });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should accept API key', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051', {
                apiKey: 'test-api-key',
            });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should accept custom metadata', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051', {
                metadata: { 'x-custom-header': 'value' },
            });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should throw ConnectionError when gRPC fails', async () => {
            // Temporarily make loadProto throw
            const grpcModule = await Promise.resolve().then(() => __importStar(require('./grpc.js')));
            const originalLoadProto = grpcModule.loadProto;
            grpcModule.loadProto = vitest_1.vi.fn().mockRejectedValue(new Error('Connection refused'));
            await (0, vitest_1.expect)(client_js_1.NeumannClient.connect('localhost:50051')).rejects.toThrow(errors_js_1.ConnectionError);
            // Restore
            grpcModule.loadProto = originalLoadProto;
        });
    });
    (0, vitest_1.describe)('connectWeb', () => {
        (0, vitest_1.it)('should create a connected gRPC-Web client', async () => {
            const client = await client_js_1.NeumannClient.connectWeb('http://localhost:8080');
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            (0, vitest_1.expect)(client.clientMode).toBe('remote');
            client.close();
        });
        (0, vitest_1.it)('should accept API key for web client', async () => {
            const client = await client_js_1.NeumannClient.connectWeb('http://localhost:8080', {
                apiKey: 'test-api-key',
            });
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should throw ConnectionError when gRPC-Web fails', async () => {
            // Temporarily make the GrpcWebClientBase constructor throw
            const grpcWeb = await Promise.resolve().then(() => __importStar(require('grpc-web')));
            const originalBase = grpcWeb.GrpcWebClientBase;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
            grpcWeb.GrpcWebClientBase = vitest_1.vi.fn().mockImplementation(() => {
                throw new Error('gRPC-Web initialization failed');
            });
            await (0, vitest_1.expect)(client_js_1.NeumannClient.connectWeb('http://localhost:8080')).rejects.toThrow(errors_js_1.ConnectionError);
            // Restore
            // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
            grpcWeb.GrpcWebClientBase = originalBase;
        });
    });
    (0, vitest_1.describe)('close', () => {
        (0, vitest_1.it)('should disconnect the client', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            (0, vitest_1.expect)(client.isConnected).toBe(true);
            client.close();
            (0, vitest_1.expect)(client.isConnected).toBe(false);
        });
    });
    (0, vitest_1.describe)('execute', () => {
        (0, vitest_1.it)('should throw if not connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            client.close();
            await (0, vitest_1.expect)(client.execute('SELECT * FROM users')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should execute a query when connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.execute('SELECT * FROM users');
            (0, vitest_1.expect)(result.type).toBe('empty');
            client.close();
        });
        (0, vitest_1.it)('should pass identity option', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.execute('VAULT GET secret', { identity: 'alice' });
            (0, vitest_1.expect)(result.type).toBe('empty');
            client.close();
        });
    });
    (0, vitest_1.describe)('executeStream', () => {
        (0, vitest_1.it)('should throw if not connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            client.close();
            await (0, vitest_1.expect)(async () => {
                for await (const _result of client.executeStream('SELECT * FROM users')) {
                    // Should throw before yielding
                }
            }).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should yield results when connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const results = [];
            for await (const result of client.executeStream('SELECT * FROM users')) {
                results.push(result);
            }
            (0, vitest_1.expect)(results.length).toBeGreaterThan(0);
            client.close();
        });
        (0, vitest_1.it)('should throw on stream error', async () => {
            const errorResponses = [
                { error: { code: 1, message: 'Stream error' } },
            ];
            let index = 0;
            mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
                [Symbol.asyncIterator]: () => ({
                    next() {
                        if (index < errorResponses.length) {
                            return Promise.resolve({ value: errorResponses[index++], done: false });
                        }
                        return Promise.resolve({ value: undefined, done: true });
                    },
                }),
            }));
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            await (0, vitest_1.expect)(async () => {
                for await (const _result of client.executeStream('INVALID')) {
                    // Should throw on error chunk
                }
            }).rejects.toThrow(errors_js_1.InvalidArgumentError);
            client.close();
        });
    });
    (0, vitest_1.describe)('executeBatch', () => {
        (0, vitest_1.it)('should throw if not connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            client.close();
            await (0, vitest_1.expect)(client.executeBatch(['SELECT 1', 'SELECT 2'])).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should execute batch when connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const results = await client.executeBatch(['SELECT 1', 'SELECT 2', 'SELECT 3']);
            (0, vitest_1.expect)(results).toHaveLength(3);
            client.close();
        });
        (0, vitest_1.it)('should handle gRPC error in batch', async () => {
            mockGrpcClient.ExecuteBatch.mockImplementationOnce((_request, _metadata, callback) => {
                callback({ code: 3, details: 'Batch error' }, null);
            });
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            await (0, vitest_1.expect)(client.executeBatch(['SELECT 1'])).rejects.toThrow(errors_js_1.InvalidArgumentError);
            client.close();
        });
    });
});
(0, vitest_1.describe)('Proto Conversion Functions', () => {
    (0, vitest_1.describe)('convertProtoValue', () => {
        (0, vitest_1.it)('should convert null values', () => {
            const result = (0, client_js_1.convertProtoValue)(null);
            (0, vitest_1.expect)(result.type).toBe('null');
            (0, vitest_1.expect)(result.data).toBeNull();
        });
        (0, vitest_1.it)('should convert undefined values', () => {
            const result = (0, client_js_1.convertProtoValue)(undefined);
            (0, vitest_1.expect)(result.type).toBe('null');
        });
        (0, vitest_1.it)('should convert nullValue field', () => {
            const result = (0, client_js_1.convertProtoValue)({ nullValue: true });
            (0, vitest_1.expect)(result.type).toBe('null');
        });
        (0, vitest_1.it)('should convert int values', () => {
            const result = (0, client_js_1.convertProtoValue)({ intValue: 42 });
            (0, vitest_1.expect)(result.type).toBe('int');
            (0, vitest_1.expect)(result.data).toBe(42);
        });
        (0, vitest_1.it)('should convert float values', () => {
            const result = (0, client_js_1.convertProtoValue)({ floatValue: 3.14 });
            (0, vitest_1.expect)(result.type).toBe('float');
            (0, vitest_1.expect)(result.data).toBe(3.14);
        });
        (0, vitest_1.it)('should convert string values', () => {
            const result = (0, client_js_1.convertProtoValue)({ stringValue: 'hello' });
            (0, vitest_1.expect)(result.type).toBe('string');
            (0, vitest_1.expect)(result.data).toBe('hello');
        });
        (0, vitest_1.it)('should convert bool values', () => {
            const result = (0, client_js_1.convertProtoValue)({ boolValue: true });
            (0, vitest_1.expect)(result.type).toBe('bool');
            (0, vitest_1.expect)(result.data).toBe(true);
        });
        (0, vitest_1.it)('should convert bytes values', () => {
            const bytes = new Uint8Array([1, 2, 3]);
            const result = (0, client_js_1.convertProtoValue)({ bytesValue: bytes });
            (0, vitest_1.expect)(result.type).toBe('bytes');
            (0, vitest_1.expect)(result.data).toEqual(bytes);
        });
        (0, vitest_1.it)('should default to null for unknown types', () => {
            const result = (0, client_js_1.convertProtoValue)({ unknownField: 'value' });
            (0, vitest_1.expect)(result.type).toBe('null');
        });
    });
    (0, vitest_1.describe)('convertProtoRow', () => {
        (0, vitest_1.it)('should convert empty row', () => {
            const result = (0, client_js_1.convertProtoRow)({});
            (0, vitest_1.expect)(result.values.size).toBe(0);
        });
        (0, vitest_1.it)('should convert row with columns', () => {
            const result = (0, client_js_1.convertProtoRow)({
                columns: [
                    { name: 'id', value: { intValue: 1 } },
                    { name: 'name', value: { stringValue: 'Alice' } },
                ],
            });
            (0, vitest_1.expect)(result.values.size).toBe(2);
            (0, vitest_1.expect)(result.values.get('id')?.data).toBe(1);
            (0, vitest_1.expect)(result.values.get('name')?.data).toBe('Alice');
        });
    });
    (0, vitest_1.describe)('convertProtoNode', () => {
        (0, vitest_1.it)('should convert node with properties', () => {
            const result = (0, client_js_1.convertProtoNode)({
                id: 'node-1',
                label: 'Person',
                properties: [{ name: 'age', value: { intValue: 30 } }],
            });
            (0, vitest_1.expect)(result.id).toBe('node-1');
            (0, vitest_1.expect)(result.label).toBe('Person');
            (0, vitest_1.expect)(result.properties.get('age')?.data).toBe(30);
        });
        (0, vitest_1.it)('should convert node without properties', () => {
            const result = (0, client_js_1.convertProtoNode)({
                id: 'node-2',
                label: 'Item',
            });
            (0, vitest_1.expect)(result.id).toBe('node-2');
            (0, vitest_1.expect)(result.label).toBe('Item');
            (0, vitest_1.expect)(result.properties.size).toBe(0);
        });
    });
    (0, vitest_1.describe)('convertProtoEdge', () => {
        (0, vitest_1.it)('should convert edge with properties', () => {
            const result = (0, client_js_1.convertProtoEdge)({
                id: 'edge-1',
                edgeType: 'KNOWS',
                sourceId: 'node-1',
                targetId: 'node-2',
                properties: [{ name: 'since', value: { intValue: 2020 } }],
            });
            (0, vitest_1.expect)(result.id).toBe('edge-1');
            (0, vitest_1.expect)(result.edgeType).toBe('KNOWS');
            (0, vitest_1.expect)(result.source).toBe('node-1');
            (0, vitest_1.expect)(result.target).toBe('node-2');
            (0, vitest_1.expect)(result.properties.get('since')?.data).toBe(2020);
        });
    });
    (0, vitest_1.describe)('convertProtoPath', () => {
        (0, vitest_1.it)('should convert empty path', () => {
            const result = (0, client_js_1.convertProtoPath)({});
            (0, vitest_1.expect)(result.segments).toHaveLength(0);
        });
        (0, vitest_1.it)('should convert path with segments', () => {
            const result = (0, client_js_1.convertProtoPath)({
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
            (0, vitest_1.expect)(result.segments).toHaveLength(2);
            (0, vitest_1.expect)(result.segments[0]?.node.id).toBe('n1');
            (0, vitest_1.expect)(result.segments[0]?.edge?.id).toBe('e1');
            (0, vitest_1.expect)(result.segments[1]?.node.id).toBe('n2');
            (0, vitest_1.expect)(result.segments[1]?.edge).toBeUndefined();
        });
    });
    (0, vitest_1.describe)('convertProtoSimilarItem', () => {
        (0, vitest_1.it)('should convert similar item without metadata', () => {
            const result = (0, client_js_1.convertProtoSimilarItem)({
                key: 'item-1',
                score: 0.95,
            });
            (0, vitest_1.expect)(result.key).toBe('item-1');
            (0, vitest_1.expect)(result.score).toBe(0.95);
            (0, vitest_1.expect)(result.metadata).toBeUndefined();
        });
        (0, vitest_1.it)('should convert similar item with metadata', () => {
            const result = (0, client_js_1.convertProtoSimilarItem)({
                key: 'item-2',
                score: 0.85,
                metadata: [{ name: 'category', value: { stringValue: 'tech' } }],
            });
            (0, vitest_1.expect)(result.key).toBe('item-2');
            (0, vitest_1.expect)(result.score).toBe(0.85);
            (0, vitest_1.expect)(result.metadata?.get('category')?.data).toBe('tech');
        });
    });
    (0, vitest_1.describe)('convertProtoArtifactInfo', () => {
        (0, vitest_1.it)('should convert artifact info', () => {
            const result = (0, client_js_1.convertProtoArtifactInfo)({
                artifactId: 'art-123',
                filename: 'test.txt',
                size: 1024,
                checksum: 'abc123',
                contentType: 'text/plain',
                createdAt: 1700000000,
                tags: ['test', 'sample'],
            });
            (0, vitest_1.expect)(result.artifactId).toBe('art-123');
            (0, vitest_1.expect)(result.filename).toBe('test.txt');
            (0, vitest_1.expect)(result.size).toBe(1024);
            (0, vitest_1.expect)(result.checksum).toBe('abc123');
            (0, vitest_1.expect)(result.contentType).toBe('text/plain');
            (0, vitest_1.expect)(result.createdAt).toBe(1700000000);
            (0, vitest_1.expect)(result.tags).toEqual(['test', 'sample']);
        });
        (0, vitest_1.it)('should handle missing tags', () => {
            const result = (0, client_js_1.convertProtoArtifactInfo)({
                artifactId: 'art-456',
                filename: 'data.bin',
                size: 2048,
                checksum: 'def456',
                contentType: 'application/octet-stream',
                createdAt: 1700000001,
            });
            (0, vitest_1.expect)(result.tags).toEqual([]);
        });
    });
});
(0, vitest_1.describe)('Value Functions', () => {
    (0, vitest_1.describe)('nullValue', () => {
        (0, vitest_1.it)('should create null value', () => {
            const v = (0, value_js_1.nullValue)();
            (0, vitest_1.expect)(v.type).toBe('null');
            (0, vitest_1.expect)(v.data).toBeNull();
        });
    });
    (0, vitest_1.describe)('intValue', () => {
        (0, vitest_1.it)('should create int value', () => {
            const v = (0, value_js_1.intValue)(42);
            (0, vitest_1.expect)(v.type).toBe('int');
            (0, vitest_1.expect)(v.data).toBe(42);
        });
        (0, vitest_1.it)('should floor float to int', () => {
            const v = (0, value_js_1.intValue)(42.9);
            (0, vitest_1.expect)(v.data).toBe(42);
        });
    });
    (0, vitest_1.describe)('floatValue', () => {
        (0, vitest_1.it)('should create float value', () => {
            const v = (0, value_js_1.floatValue)(3.14159);
            (0, vitest_1.expect)(v.type).toBe('float');
            (0, vitest_1.expect)(v.data).toBe(3.14159);
        });
    });
    (0, vitest_1.describe)('stringValue', () => {
        (0, vitest_1.it)('should create string value', () => {
            const v = (0, value_js_1.stringValue)('hello world');
            (0, vitest_1.expect)(v.type).toBe('string');
            (0, vitest_1.expect)(v.data).toBe('hello world');
        });
    });
    (0, vitest_1.describe)('boolValue', () => {
        (0, vitest_1.it)('should create bool value true', () => {
            const v = (0, value_js_1.boolValue)(true);
            (0, vitest_1.expect)(v.type).toBe('bool');
            (0, vitest_1.expect)(v.data).toBe(true);
        });
        (0, vitest_1.it)('should create bool value false', () => {
            const v = (0, value_js_1.boolValue)(false);
            (0, vitest_1.expect)(v.type).toBe('bool');
            (0, vitest_1.expect)(v.data).toBe(false);
        });
    });
    (0, vitest_1.describe)('bytesValue', () => {
        (0, vitest_1.it)('should create bytes value', () => {
            const bytes = new Uint8Array([0x01, 0x02, 0x03]);
            const v = (0, value_js_1.bytesValue)(bytes);
            (0, vitest_1.expect)(v.type).toBe('bytes');
            (0, vitest_1.expect)(v.data).toEqual(bytes);
        });
    });
    (0, vitest_1.describe)('valueToNative', () => {
        (0, vitest_1.it)('should extract native value', () => {
            (0, vitest_1.expect)((0, value_js_1.valueToNative)((0, value_js_1.intValue)(42))).toBe(42);
            (0, vitest_1.expect)((0, value_js_1.valueToNative)((0, value_js_1.stringValue)('test'))).toBe('test');
            (0, vitest_1.expect)((0, value_js_1.valueToNative)((0, value_js_1.nullValue)())).toBeNull();
        });
    });
    (0, vitest_1.describe)('valueFromNative', () => {
        (0, vitest_1.it)('should convert null', () => {
            (0, vitest_1.expect)((0, value_js_1.valueFromNative)(null).type).toBe('null');
        });
        (0, vitest_1.it)('should convert undefined', () => {
            (0, vitest_1.expect)((0, value_js_1.valueFromNative)(undefined).type).toBe('null');
        });
        (0, vitest_1.it)('should convert boolean', () => {
            const v = (0, value_js_1.valueFromNative)(true);
            (0, vitest_1.expect)(v.type).toBe('bool');
            (0, vitest_1.expect)(v.data).toBe(true);
        });
        (0, vitest_1.it)('should convert integer', () => {
            const v = (0, value_js_1.valueFromNative)(42);
            (0, vitest_1.expect)(v.type).toBe('int');
            (0, vitest_1.expect)(v.data).toBe(42);
        });
        (0, vitest_1.it)('should convert float', () => {
            const v = (0, value_js_1.valueFromNative)(3.14);
            (0, vitest_1.expect)(v.type).toBe('float');
            (0, vitest_1.expect)(v.data).toBe(3.14);
        });
        (0, vitest_1.it)('should convert string', () => {
            const v = (0, value_js_1.valueFromNative)('hello');
            (0, vitest_1.expect)(v.type).toBe('string');
            (0, vitest_1.expect)(v.data).toBe('hello');
        });
        (0, vitest_1.it)('should convert Uint8Array', () => {
            const bytes = new Uint8Array([1, 2, 3]);
            const v = (0, value_js_1.valueFromNative)(bytes);
            (0, vitest_1.expect)(v.type).toBe('bytes');
            (0, vitest_1.expect)(v.data).toEqual(bytes);
        });
        (0, vitest_1.it)('should convert other objects to string', () => {
            const v = (0, value_js_1.valueFromNative)({ foo: 'bar' });
            (0, vitest_1.expect)(v.type).toBe('string');
        });
    });
});
(0, vitest_1.describe)('Error Classes', () => {
    (0, vitest_1.describe)('NeumannError', () => {
        (0, vitest_1.it)('should create with message and code', () => {
            const err = new errors_js_1.NeumannError('test error', errors_js_1.ErrorCode.INTERNAL);
            (0, vitest_1.expect)(err.message).toBe('test error');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.INTERNAL);
            (0, vitest_1.expect)(err.name).toBe('NeumannError');
        });
        (0, vitest_1.it)('should default to UNKNOWN code', () => {
            const err = new errors_js_1.NeumannError('test');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.UNKNOWN);
        });
        (0, vitest_1.it)('should format toString correctly', () => {
            const err = new errors_js_1.NeumannError('test error', errors_js_1.ErrorCode.INTERNAL);
            (0, vitest_1.expect)(err.toString()).toContain('INTERNAL');
            (0, vitest_1.expect)(err.toString()).toContain('test error');
        });
    });
    (0, vitest_1.describe)('ConnectionError', () => {
        (0, vitest_1.it)('should create with UNAVAILABLE code', () => {
            const err = new errors_js_1.ConnectionError('connection failed');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.UNAVAILABLE);
            (0, vitest_1.expect)(err.name).toBe('ConnectionError');
        });
    });
    (0, vitest_1.describe)('AuthenticationError', () => {
        (0, vitest_1.it)('should create with UNAUTHENTICATED code', () => {
            const err = new errors_js_1.AuthenticationError();
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.UNAUTHENTICATED);
            (0, vitest_1.expect)(err.name).toBe('AuthenticationError');
        });
        (0, vitest_1.it)('should accept custom message', () => {
            const err = new errors_js_1.AuthenticationError('invalid token');
            (0, vitest_1.expect)(err.message).toBe('invalid token');
        });
    });
    (0, vitest_1.describe)('PermissionDeniedError', () => {
        (0, vitest_1.it)('should create with PERMISSION_DENIED code', () => {
            const err = new errors_js_1.PermissionDeniedError();
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.PERMISSION_DENIED);
            (0, vitest_1.expect)(err.name).toBe('PermissionDeniedError');
        });
    });
    (0, vitest_1.describe)('NotFoundError', () => {
        (0, vitest_1.it)('should create with NOT_FOUND code', () => {
            const err = new errors_js_1.NotFoundError('user');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.NOT_FOUND);
            (0, vitest_1.expect)(err.message).toContain('user');
            (0, vitest_1.expect)(err.name).toBe('NotFoundError');
        });
    });
    (0, vitest_1.describe)('InvalidArgumentError', () => {
        (0, vitest_1.it)('should create with INVALID_ARGUMENT code', () => {
            const err = new errors_js_1.InvalidArgumentError('bad input');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.INVALID_ARGUMENT);
            (0, vitest_1.expect)(err.name).toBe('InvalidArgumentError');
        });
    });
    (0, vitest_1.describe)('ParseError', () => {
        (0, vitest_1.it)('should create with PARSE_ERROR code', () => {
            const err = new errors_js_1.ParseError('syntax error');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.PARSE_ERROR);
            (0, vitest_1.expect)(err.name).toBe('ParseError');
        });
    });
    (0, vitest_1.describe)('QueryError', () => {
        (0, vitest_1.it)('should create with QUERY_ERROR code', () => {
            const err = new errors_js_1.QueryError('query failed');
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.QUERY_ERROR);
            (0, vitest_1.expect)(err.name).toBe('QueryError');
        });
    });
    (0, vitest_1.describe)('InternalError', () => {
        (0, vitest_1.it)('should create with INTERNAL code', () => {
            const err = new errors_js_1.InternalError();
            (0, vitest_1.expect)(err.code).toBe(errors_js_1.ErrorCode.INTERNAL);
            (0, vitest_1.expect)(err.name).toBe('InternalError');
        });
    });
    (0, vitest_1.describe)('errorFromCode', () => {
        (0, vitest_1.it)('should create InvalidArgumentError for INVALID_ARGUMENT', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.INVALID_ARGUMENT, 'bad input');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should create NotFoundError for NOT_FOUND', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.NOT_FOUND, 'resource');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.NotFoundError);
        });
        (0, vitest_1.it)('should create PermissionDeniedError for PERMISSION_DENIED', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.PERMISSION_DENIED, 'denied');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.PermissionDeniedError);
        });
        (0, vitest_1.it)('should create AuthenticationError for UNAUTHENTICATED', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.UNAUTHENTICATED, 'auth failed');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.AuthenticationError);
        });
        (0, vitest_1.it)('should create ConnectionError for UNAVAILABLE', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.UNAVAILABLE, 'unavailable');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should create InternalError for INTERNAL', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.INTERNAL, 'internal error');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.InternalError);
        });
        (0, vitest_1.it)('should create ParseError for PARSE_ERROR', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.PARSE_ERROR, 'parse error');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.ParseError);
        });
        (0, vitest_1.it)('should create QueryError for QUERY_ERROR', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.QUERY_ERROR, 'query error');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.QueryError);
        });
        (0, vitest_1.it)('should create NeumannError for UNKNOWN', () => {
            const err = (0, errors_js_1.errorFromCode)(errors_js_1.ErrorCode.UNKNOWN, 'unknown');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.NeumannError);
            (0, vitest_1.expect)(err.constructor.name).toBe('NeumannError');
        });
        (0, vitest_1.it)('should accept numeric code', () => {
            const err = (0, errors_js_1.errorFromCode)(7, 'internal');
            (0, vitest_1.expect)(err).toBeInstanceOf(errors_js_1.InternalError);
        });
    });
});
(0, vitest_1.describe)('Type Guards', () => {
    (0, vitest_1.describe)('isEmptyResult', () => {
        (0, vitest_1.it)('should return true for empty result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isEmptyResult)({ type: 'empty' })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isEmptyResult)({ type: 'rows', rows: [] })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isRowsResult', () => {
        (0, vitest_1.it)('should return true for rows result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isRowsResult)({ type: 'rows', rows: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isRowsResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isNodesResult', () => {
        (0, vitest_1.it)('should return true for nodes result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isNodesResult)({ type: 'nodes', nodes: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isNodesResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isEdgesResult', () => {
        (0, vitest_1.it)('should return true for edges result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isEdgesResult)({ type: 'edges', edges: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isEdgesResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isPathsResult', () => {
        (0, vitest_1.it)('should return true for paths result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isPathsResult)({ type: 'paths', paths: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isPathsResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isSimilarResult', () => {
        (0, vitest_1.it)('should return true for similar result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isSimilarResult)({ type: 'similar', items: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isSimilarResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isErrorResult', () => {
        (0, vitest_1.it)('should return true for error result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isErrorResult)({ type: 'error', code: 1, message: 'err' })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isErrorResult)({ type: 'empty' })).toBe(false);
        });
    });
});
(0, vitest_1.describe)('Conversion Utilities', () => {
    (0, vitest_1.describe)('rowToObject', () => {
        (0, vitest_1.it)('should convert row to plain object', () => {
            const row = {
                values: new Map([
                    ['id', (0, value_js_1.intValue)(1)],
                    ['name', (0, value_js_1.stringValue)('Alice')],
                ]),
            };
            const obj = (0, query_result_js_1.rowToObject)(row);
            (0, vitest_1.expect)(obj['id']).toBe(1);
            (0, vitest_1.expect)(obj['name']).toBe('Alice');
        });
        (0, vitest_1.it)('should handle empty row', () => {
            const row = { values: new Map() };
            const obj = (0, query_result_js_1.rowToObject)(row);
            (0, vitest_1.expect)(Object.keys(obj)).toHaveLength(0);
        });
    });
    (0, vitest_1.describe)('nodeToObject', () => {
        (0, vitest_1.it)('should convert node to plain object', () => {
            const node = {
                id: 'n1',
                label: 'Person',
                properties: new Map([['age', (0, value_js_1.intValue)(30)]]),
            };
            const obj = (0, query_result_js_1.nodeToObject)(node);
            (0, vitest_1.expect)(obj['id']).toBe('n1');
            (0, vitest_1.expect)(obj['label']).toBe('Person');
            (0, vitest_1.expect)(obj['properties']['age']).toBe(30);
        });
    });
    (0, vitest_1.describe)('edgeToObject', () => {
        (0, vitest_1.it)('should convert edge to plain object', () => {
            const edge = {
                id: 'e1',
                edgeType: 'KNOWS',
                source: 'n1',
                target: 'n2',
                properties: new Map([['weight', (0, value_js_1.floatValue)(0.5)]]),
            };
            const obj = (0, query_result_js_1.edgeToObject)(edge);
            (0, vitest_1.expect)(obj['id']).toBe('e1');
            (0, vitest_1.expect)(obj['type']).toBe('KNOWS');
            (0, vitest_1.expect)(obj['source']).toBe('n1');
            (0, vitest_1.expect)(obj['target']).toBe('n2');
            (0, vitest_1.expect)(obj['properties']['weight']).toBe(0.5);
        });
    });
});
(0, vitest_1.describe)('Stream Chunk Types', () => {
    (0, vitest_1.it)('should handle node chunks', async () => {
        const nodeResponses = [
            { node: { node: { id: 'n1', label: 'Person', properties: [] } } },
            { isFinal: true },
        ];
        let index = 0;
        mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
            [Symbol.asyncIterator]: () => ({
                next() {
                    if (index < nodeResponses.length) {
                        return Promise.resolve({ value: nodeResponses[index++], done: false });
                    }
                    return Promise.resolve({ value: undefined, done: true });
                },
            }),
        }));
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const results = [];
        for await (const result of client.executeStream('MATCH (n)')) {
            results.push(result);
        }
        (0, vitest_1.expect)(results.length).toBeGreaterThan(0);
        (0, vitest_1.expect)(results[0]).toHaveProperty('type', 'nodes');
        client.close();
    });
    (0, vitest_1.it)('should handle edge chunks', async () => {
        const edgeResponses = [
            { edge: { edge: { id: 'e1', type: 'KNOWS', from: 'n1', to: 'n2' } } },
            { isFinal: true },
        ];
        let index = 0;
        mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
            [Symbol.asyncIterator]: () => ({
                next() {
                    if (index < edgeResponses.length) {
                        return Promise.resolve({ value: edgeResponses[index++], done: false });
                    }
                    return Promise.resolve({ value: undefined, done: true });
                },
            }),
        }));
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const results = [];
        for await (const result of client.executeStream('MATCH ()-[e]->()')) {
            results.push(result);
        }
        (0, vitest_1.expect)(results.length).toBeGreaterThan(0);
        (0, vitest_1.expect)(results[0]).toHaveProperty('type', 'edges');
        client.close();
    });
    (0, vitest_1.it)('should handle similarItem chunks', async () => {
        const similarItemResponses = [
            { similarItem: { item: { key: 'k1', score: 0.95 } } },
            { isFinal: true },
        ];
        let index = 0;
        mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
            [Symbol.asyncIterator]: () => ({
                next() {
                    if (index < similarItemResponses.length) {
                        return Promise.resolve({ value: similarItemResponses[index++], done: false });
                    }
                    return Promise.resolve({ value: undefined, done: true });
                },
            }),
        }));
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const results = [];
        for await (const result of client.executeStream('SIMILAR vec')) {
            results.push(result);
        }
        (0, vitest_1.expect)(results.length).toBeGreaterThan(0);
        (0, vitest_1.expect)(results[0]).toHaveProperty('type', 'similar');
        client.close();
    });
    (0, vitest_1.it)('should handle blobData chunks', async () => {
        const blobResponses = [
            { blobData: new Uint8Array([1, 2, 3, 4]) },
            { isFinal: true },
        ];
        let index = 0;
        mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
            [Symbol.asyncIterator]: () => ({
                next() {
                    if (index < blobResponses.length) {
                        return Promise.resolve({ value: blobResponses[index++], done: false });
                    }
                    return Promise.resolve({ value: undefined, done: true });
                },
            }),
        }));
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const results = [];
        for await (const result of client.executeStream('GET BLOB')) {
            results.push(result);
        }
        (0, vitest_1.expect)(results.length).toBeGreaterThan(0);
        (0, vitest_1.expect)(results[0]).toHaveProperty('type', 'blob');
        client.close();
    });
    (0, vitest_1.it)('should handle empty chunks as fallback', async () => {
        const emptyResponses = [
            { unknownField: true }, // No recognized field - should return empty
            { isFinal: true },
        ];
        let index = 0;
        mockGrpcClient.ExecuteStream.mockImplementationOnce(() => ({
            [Symbol.asyncIterator]: () => ({
                next() {
                    if (index < emptyResponses.length) {
                        return Promise.resolve({ value: emptyResponses[index++], done: false });
                    }
                    return Promise.resolve({ value: undefined, done: true });
                },
            }),
        }));
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const results = [];
        for await (const result of client.executeStream('UNKNOWN')) {
            results.push(result);
        }
        (0, vitest_1.expect)(results.length).toBeGreaterThan(0);
        (0, vitest_1.expect)(results[0]).toHaveProperty('type', 'empty');
        client.close();
    });
});
(0, vitest_1.describe)('gRPC Error Handling', () => {
    (0, vitest_1.it)('should convert UNAUTHENTICATED to AuthenticationError', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 16, details: 'Auth failed' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow(errors_js_1.AuthenticationError);
        client.close();
    });
    (0, vitest_1.it)('should convert PERMISSION_DENIED to PermissionDeniedError', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 7, details: 'Permission denied' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow(errors_js_1.PermissionDeniedError);
        client.close();
    });
    (0, vitest_1.it)('should convert NOT_FOUND to NotFoundError', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 5, details: 'Not found' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow(errors_js_1.NotFoundError);
        client.close();
    });
    (0, vitest_1.it)('should convert INVALID_ARGUMENT to InvalidArgumentError', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 3, details: 'Invalid argument' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow(errors_js_1.InvalidArgumentError);
        client.close();
    });
    (0, vitest_1.it)('should convert UNAVAILABLE to ConnectionError', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 14, details: 'Service unavailable' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow(errors_js_1.ConnectionError);
        client.close();
    });
    (0, vitest_1.it)('should convert unknown error codes to InternalError', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 99, details: 'Unknown error' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow(errors_js_1.InternalError);
        client.close();
    });
    (0, vitest_1.it)('should use default message when details is empty', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback({ code: 16, message: 'fallback' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        await (0, vitest_1.expect)(client.execute('SELECT 1')).rejects.toThrow('Authentication failed');
        client.close();
    });
});
(0, vitest_1.describe)('Pagination', () => {
    (0, vitest_1.describe)('executePaginated', () => {
        (0, vitest_1.it)('should throw if not connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            client.close();
            await (0, vitest_1.expect)(client.executePaginated('SELECT * FROM users')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should execute a paginated query when connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.executePaginated('SELECT * FROM users');
            (0, vitest_1.expect)(result.result.type).toBe('rows');
            (0, vitest_1.expect)(result.hasMore).toBe(true);
            (0, vitest_1.expect)(result.pageSize).toBe(10);
            (0, vitest_1.expect)(result.nextCursor).toBe('cursor123');
            (0, vitest_1.expect)(result.totalCount).toBe(100);
            client.close();
        });
        (0, vitest_1.it)('should pass pagination options', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.executePaginated('SELECT * FROM users', {
                pageSize: 20,
                cursor: 'prev_cursor',
                countTotal: true,
                cursorTtlSecs: 300,
                identity: 'alice',
            });
            (0, vitest_1.expect)(result.result.type).toBe('rows');
            client.close();
        });
        (0, vitest_1.it)('should handle gRPC error in executePaginated', async () => {
            mockGrpcClient.ExecutePaginated.mockImplementationOnce((_request, _metadata, callback) => {
                callback({ code: 3, details: 'Invalid query' }, null);
            });
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            await (0, vitest_1.expect)(client.executePaginated('INVALID')).rejects.toThrow(errors_js_1.InvalidArgumentError);
            client.close();
        });
        (0, vitest_1.it)('should handle response without optional fields', async () => {
            mockGrpcClient.ExecutePaginated.mockImplementationOnce((_request, _metadata, callback) => {
                callback(null, {
                    result: { empty: {} },
                    hasMore: false,
                    pageSize: 10,
                });
            });
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.executePaginated('SELECT');
            (0, vitest_1.expect)(result.hasMore).toBe(false);
            (0, vitest_1.expect)(result.nextCursor).toBeUndefined();
            (0, vitest_1.expect)(result.prevCursor).toBeUndefined();
            (0, vitest_1.expect)(result.totalCount).toBeUndefined();
            client.close();
        });
        (0, vitest_1.it)('should handle response with prevCursor', async () => {
            mockGrpcClient.ExecutePaginated.mockImplementationOnce((_request, _metadata, callback) => {
                callback(null, {
                    result: { empty: {} },
                    nextCursor: 'next',
                    prevCursor: 'prev',
                    hasMore: true,
                    pageSize: 10,
                });
            });
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.executePaginated('SELECT');
            (0, vitest_1.expect)(result.nextCursor).toBe('next');
            (0, vitest_1.expect)(result.prevCursor).toBe('prev');
            client.close();
        });
    });
    (0, vitest_1.describe)('closeCursor', () => {
        (0, vitest_1.it)('should throw if not connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            client.close();
            await (0, vitest_1.expect)(client.closeCursor('cursor123')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should close a cursor when connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const result = await client.closeCursor('cursor123');
            (0, vitest_1.expect)(result).toBe(true);
            client.close();
        });
        (0, vitest_1.it)('should handle gRPC error in closeCursor', async () => {
            mockGrpcClient.CloseCursor.mockImplementationOnce((_request, _metadata, callback) => {
                callback({ code: 5, details: 'Cursor not found' }, null);
            });
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            await (0, vitest_1.expect)(client.closeCursor('invalid')).rejects.toThrow(errors_js_1.NotFoundError);
            client.close();
        });
    });
    (0, vitest_1.describe)('executeAllPages', () => {
        (0, vitest_1.it)('should throw if not connected', async () => {
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            client.close();
            await (0, vitest_1.expect)(async () => {
                for await (const _result of client.executeAllPages('SELECT * FROM users')) {
                    // Should throw before yielding
                }
            }).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should iterate through all pages', async () => {
            // Mock multiple pages then end
            let callCount = 0;
            mockGrpcClient.ExecutePaginated.mockImplementation((_request, _metadata, callback) => {
                callCount++;
                if (callCount === 1) {
                    callback(null, {
                        result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: 1 } }] }] } },
                        nextCursor: 'cursor2',
                        hasMore: true,
                        pageSize: 10,
                    });
                }
                else {
                    callback(null, {
                        result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: 2 } }] }] } },
                        hasMore: false,
                        pageSize: 10,
                    });
                }
            });
            const client = await client_js_1.NeumannClient.connect('localhost:50051');
            const results = [];
            for await (const result of client.executeAllPages('SELECT * FROM users')) {
                results.push(result);
            }
            (0, vitest_1.expect)(results.length).toBe(2);
            client.close();
            // Reset mock
            mockGrpcClient.ExecutePaginated.mockImplementation((_request, _metadata, callback) => {
                callback(null, mockPaginatedResponse);
            });
        });
    });
});
(0, vitest_1.describe)('New Type Guards', () => {
    (0, vitest_1.describe)('isValueResult', () => {
        (0, vitest_1.it)('should return true for value result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isValueResult)({ type: 'value', value: 'test' })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isValueResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isCountResult', () => {
        (0, vitest_1.it)('should return true for count result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCountResult)({ type: 'count', count: 10 })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCountResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isIdsResult', () => {
        (0, vitest_1.it)('should return true for ids result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isIdsResult)({ type: 'ids', ids: ['1', '2'] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isIdsResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isTableListResult', () => {
        (0, vitest_1.it)('should return true for table list result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isTableListResult)({ type: 'tableList', names: ['t1'] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isTableListResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isBlobResult', () => {
        (0, vitest_1.it)('should return true for blob result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBlobResult)({ type: 'blob', data: new Uint8Array([1]) })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBlobResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isBlobInfoResult', () => {
        (0, vitest_1.it)('should return true for blob info result', () => {
            const info = {
                artifactId: 'a1',
                filename: 'f.txt',
                size: 100,
                checksum: 'abc',
                contentType: 'text/plain',
                createdAt: 123,
                tags: [],
            };
            (0, vitest_1.expect)((0, query_result_js_1.isBlobInfoResult)({ type: 'blobInfo', info })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBlobInfoResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isArtifactListResult', () => {
        (0, vitest_1.it)('should return true for artifact list result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isArtifactListResult)({ type: 'artifactList', artifactIds: ['a1'] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isArtifactListResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isBlobStatsResult', () => {
        (0, vitest_1.it)('should return true for blob stats result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBlobStatsResult)({
                type: 'blobStats',
                artifactCount: 1,
                chunkCount: 2,
                totalBytes: 100,
                uniqueBytes: 80,
                dedupRatio: 0.8,
                orphanedChunks: 0,
            })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBlobStatsResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isCheckpointListResult', () => {
        (0, vitest_1.it)('should return true for checkpoint list result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCheckpointListResult)({ type: 'checkpointList', checkpoints: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCheckpointListResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isPageRankResult', () => {
        (0, vitest_1.it)('should return true for page rank result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isPageRankResult)({ type: 'pageRank', items: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isPageRankResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isCentralityResult', () => {
        (0, vitest_1.it)('should return true for centrality result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCentralityResult)({ type: 'centrality', items: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCentralityResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isCommunitiesResult', () => {
        (0, vitest_1.it)('should return true for communities result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCommunitiesResult)({ type: 'communities', items: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isCommunitiesResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isPatternMatchResult', () => {
        (0, vitest_1.it)('should return true for pattern match result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isPatternMatchResult)({ type: 'patternMatch', matches: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isPatternMatchResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isConstraintsResult', () => {
        (0, vitest_1.it)('should return true for constraints result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isConstraintsResult)({ type: 'constraints', items: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isConstraintsResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isAggregateResult', () => {
        (0, vitest_1.it)('should return true for aggregate result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isAggregateResult)({ type: 'aggregate', value: { type: 'count', value: 10 } })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isAggregateResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isBatchOperationResult', () => {
        (0, vitest_1.it)('should return true for batch operation result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBatchOperationResult)({
                type: 'batchOperation',
                operation: 'insert',
                affectedCount: 5,
                createdIds: ['1'],
            })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isBatchOperationResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isGraphIndexesResult', () => {
        (0, vitest_1.it)('should return true for graph indexes result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isGraphIndexesResult)({ type: 'graphIndexes', indexes: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isGraphIndexesResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isChainQueryResult', () => {
        (0, vitest_1.it)('should return true for chain query result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isChainQueryResult)({ type: 'chain', result: { type: 'height', value: { height: 10 } } })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isChainQueryResult)({ type: 'empty' })).toBe(false);
        });
    });
    (0, vitest_1.describe)('isUnifiedResult', () => {
        (0, vitest_1.it)('should return true for unified result', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isUnifiedResult)({ type: 'unified', description: 'test', items: [] })).toBe(true);
        });
        (0, vitest_1.it)('should return false for other types', () => {
            (0, vitest_1.expect)((0, query_result_js_1.isUnifiedResult)({ type: 'empty' })).toBe(false);
        });
    });
});
(0, vitest_1.describe)('New Conversion Functions', () => {
    (0, vitest_1.describe)('convertProtoCheckpoint', () => {
        (0, vitest_1.it)('should convert checkpoint', () => {
            const result = (0, client_js_1.convertProtoCheckpoint)({
                id: 'cp1',
                name: 'checkpoint1',
                createdAt: 1700000000,
                isAuto: false,
            });
            (0, vitest_1.expect)(result.id).toBe('cp1');
            (0, vitest_1.expect)(result.name).toBe('checkpoint1');
            (0, vitest_1.expect)(result.createdAt).toBe(1700000000);
            (0, vitest_1.expect)(result.isAuto).toBe(false);
        });
    });
    (0, vitest_1.describe)('convertProtoPageRankItem', () => {
        (0, vitest_1.it)('should convert page rank item', () => {
            const result = (0, client_js_1.convertProtoPageRankItem)({ nodeId: 42, score: 0.85 });
            (0, vitest_1.expect)(result.nodeId).toBe('42');
            (0, vitest_1.expect)(result.score).toBe(0.85);
        });
    });
    (0, vitest_1.describe)('convertProtoCentralityType', () => {
        (0, vitest_1.it)('should convert betweenness', () => {
            (0, vitest_1.expect)((0, client_js_1.convertProtoCentralityType)('CENTRALITY_TYPE_BETWEENNESS')).toBe('betweenness');
        });
        (0, vitest_1.it)('should convert closeness', () => {
            (0, vitest_1.expect)((0, client_js_1.convertProtoCentralityType)('CENTRALITY_TYPE_CLOSENESS')).toBe('closeness');
        });
        (0, vitest_1.it)('should convert eigenvector', () => {
            (0, vitest_1.expect)((0, client_js_1.convertProtoCentralityType)('CENTRALITY_TYPE_EIGENVECTOR')).toBe('eigenvector');
        });
        (0, vitest_1.it)('should default to betweenness for unknown', () => {
            (0, vitest_1.expect)((0, client_js_1.convertProtoCentralityType)('UNKNOWN')).toBe('betweenness');
        });
    });
    (0, vitest_1.describe)('convertProtoCentralityItem', () => {
        (0, vitest_1.it)('should convert centrality item', () => {
            const result = (0, client_js_1.convertProtoCentralityItem)({ nodeId: 10, score: 0.5 });
            (0, vitest_1.expect)(result.nodeId).toBe('10');
            (0, vitest_1.expect)(result.score).toBe(0.5);
        });
    });
    (0, vitest_1.describe)('convertProtoCommunityItem', () => {
        (0, vitest_1.it)('should convert community item', () => {
            const result = (0, client_js_1.convertProtoCommunityItem)({ nodeId: 5, communityId: 2 });
            (0, vitest_1.expect)(result.nodeId).toBe('5');
            (0, vitest_1.expect)(result.communityId).toBe('2');
        });
    });
    (0, vitest_1.describe)('convertProtoCommunityMemberList', () => {
        (0, vitest_1.it)('should convert community member list', () => {
            const result = (0, client_js_1.convertProtoCommunityMemberList)({
                communityId: 1,
                memberNodeIds: [10, 20, 30],
            });
            (0, vitest_1.expect)(result.communityId).toBe('1');
            (0, vitest_1.expect)(result.memberNodeIds).toEqual(['10', '20', '30']);
        });
    });
    (0, vitest_1.describe)('convertProtoPatternMatchBinding', () => {
        (0, vitest_1.it)('should convert pattern match binding', () => {
            const result = (0, client_js_1.convertProtoPatternMatchBinding)({
                bindings: [{ variable: 'n', value: { node: { id: 1, label: 'Person' } } }],
            });
            (0, vitest_1.expect)(result.bindings.length).toBe(1);
            (0, vitest_1.expect)(result.bindings[0]?.variable).toBe('n');
        });
    });
    (0, vitest_1.describe)('convertProtoBindingEntry', () => {
        (0, vitest_1.it)('should convert binding entry', () => {
            const result = (0, client_js_1.convertProtoBindingEntry)({
                variable: 'x',
                value: { node: { id: 5, label: 'Test' } },
            });
            (0, vitest_1.expect)(result.variable).toBe('x');
            (0, vitest_1.expect)(result.value.type).toBe('node');
        });
    });
    (0, vitest_1.describe)('convertProtoBindingValue', () => {
        (0, vitest_1.it)('should convert node binding', () => {
            const result = (0, client_js_1.convertProtoBindingValue)({ node: { id: 1, label: 'Person' } });
            (0, vitest_1.expect)(result.type).toBe('node');
            if (result.type === 'node') {
                (0, vitest_1.expect)(result.value.id).toBe('1');
                (0, vitest_1.expect)(result.value.label).toBe('Person');
            }
        });
        (0, vitest_1.it)('should convert edge binding', () => {
            const result = (0, client_js_1.convertProtoBindingValue)({
                edge: { id: 2, edgeType: 'KNOWS', from: 1, to: 3 },
            });
            (0, vitest_1.expect)(result.type).toBe('edge');
            if (result.type === 'edge') {
                (0, vitest_1.expect)(result.value.id).toBe('2');
                (0, vitest_1.expect)(result.value.edgeType).toBe('KNOWS');
                (0, vitest_1.expect)(result.value.from).toBe('1');
                (0, vitest_1.expect)(result.value.to).toBe('3');
            }
        });
        (0, vitest_1.it)('should convert path binding', () => {
            const result = (0, client_js_1.convertProtoBindingValue)({
                path: { nodes: [1, 2, 3], edges: [10, 11], length: 2 },
            });
            (0, vitest_1.expect)(result.type).toBe('path');
            if (result.type === 'path') {
                (0, vitest_1.expect)(result.value.nodes).toEqual(['1', '2', '3']);
                (0, vitest_1.expect)(result.value.edges).toEqual(['10', '11']);
                (0, vitest_1.expect)(result.value.length).toBe(2);
            }
        });
        (0, vitest_1.it)('should default to empty node for unknown', () => {
            const result = (0, client_js_1.convertProtoBindingValue)({});
            (0, vitest_1.expect)(result.type).toBe('node');
        });
    });
    (0, vitest_1.describe)('convertProtoPatternMatchStats', () => {
        (0, vitest_1.it)('should convert pattern match stats', () => {
            const result = (0, client_js_1.convertProtoPatternMatchStats)({
                matchesFound: 10,
                nodesEvaluated: 100,
                edgesEvaluated: 50,
                truncated: true,
            });
            (0, vitest_1.expect)(result.matchesFound).toBe(10);
            (0, vitest_1.expect)(result.nodesEvaluated).toBe(100);
            (0, vitest_1.expect)(result.edgesEvaluated).toBe(50);
            (0, vitest_1.expect)(result.truncated).toBe(true);
        });
    });
    (0, vitest_1.describe)('convertProtoConstraintItem', () => {
        (0, vitest_1.it)('should convert constraint item', () => {
            const result = (0, client_js_1.convertProtoConstraintItem)({
                name: 'unique_email',
                target: 'users',
                property: 'email',
                constraintType: 'UNIQUE',
            });
            (0, vitest_1.expect)(result.name).toBe('unique_email');
            (0, vitest_1.expect)(result.target).toBe('users');
            (0, vitest_1.expect)(result.property).toBe('email');
            (0, vitest_1.expect)(result.constraintType).toBe('UNIQUE');
        });
    });
    (0, vitest_1.describe)('convertProtoAggregateValue', () => {
        (0, vitest_1.it)('should convert count', () => {
            const result = (0, client_js_1.convertProtoAggregateValue)({ count: 42 });
            (0, vitest_1.expect)(result.type).toBe('count');
            (0, vitest_1.expect)(result.value).toBe(42);
        });
        (0, vitest_1.it)('should convert sum', () => {
            const result = (0, client_js_1.convertProtoAggregateValue)({ sum: 100.5 });
            (0, vitest_1.expect)(result.type).toBe('sum');
            (0, vitest_1.expect)(result.value).toBe(100.5);
        });
        (0, vitest_1.it)('should convert avg', () => {
            const result = (0, client_js_1.convertProtoAggregateValue)({ avg: 25.5 });
            (0, vitest_1.expect)(result.type).toBe('avg');
            (0, vitest_1.expect)(result.value).toBe(25.5);
        });
        (0, vitest_1.it)('should convert min', () => {
            const result = (0, client_js_1.convertProtoAggregateValue)({ min: 1.0 });
            (0, vitest_1.expect)(result.type).toBe('min');
            (0, vitest_1.expect)(result.value).toBe(1.0);
        });
        (0, vitest_1.it)('should convert max', () => {
            const result = (0, client_js_1.convertProtoAggregateValue)({ max: 99.9 });
            (0, vitest_1.expect)(result.type).toBe('max');
            (0, vitest_1.expect)(result.value).toBe(99.9);
        });
        (0, vitest_1.it)('should default to count 0 for empty', () => {
            const result = (0, client_js_1.convertProtoAggregateValue)({});
            (0, vitest_1.expect)(result.type).toBe('count');
            (0, vitest_1.expect)(result.value).toBe(0);
        });
    });
    (0, vitest_1.describe)('convertProtoChainResult', () => {
        (0, vitest_1.it)('should convert transactionBegun', () => {
            const result = (0, client_js_1.convertProtoChainResult)({ transactionBegun: { txId: 'tx1' } });
            (0, vitest_1.expect)(result.type).toBe('transactionBegun');
            if (result.type === 'transactionBegun') {
                (0, vitest_1.expect)(result.value.txId).toBe('tx1');
            }
        });
        (0, vitest_1.it)('should convert committed', () => {
            const result = (0, client_js_1.convertProtoChainResult)({ committed: { blockHash: 'hash1', height: 10 } });
            (0, vitest_1.expect)(result.type).toBe('committed');
            if (result.type === 'committed') {
                (0, vitest_1.expect)(result.value.blockHash).toBe('hash1');
                (0, vitest_1.expect)(result.value.height).toBe(10);
            }
        });
        (0, vitest_1.it)('should convert rolledBack', () => {
            const result = (0, client_js_1.convertProtoChainResult)({ rolledBack: { toHeight: 5 } });
            (0, vitest_1.expect)(result.type).toBe('rolledBack');
            if (result.type === 'rolledBack') {
                (0, vitest_1.expect)(result.value.toHeight).toBe(5);
            }
        });
        (0, vitest_1.it)('should convert history', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                history: { entries: [{ height: 1, transactionType: 'PUT' }] },
            });
            (0, vitest_1.expect)(result.type).toBe('history');
            if (result.type === 'history') {
                (0, vitest_1.expect)(result.value.entries.length).toBe(1);
            }
        });
        (0, vitest_1.it)('should convert similar', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                similar: { items: [{ blockHash: 'h1', height: 5, similarity: 0.9 }] },
            });
            (0, vitest_1.expect)(result.type).toBe('similar');
            if (result.type === 'similar') {
                (0, vitest_1.expect)(result.value.items.length).toBe(1);
            }
        });
        (0, vitest_1.it)('should convert drift', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                drift: {
                    fromHeight: 1,
                    toHeight: 10,
                    totalDrift: 0.5,
                    avgDriftPerBlock: 0.05,
                    maxDrift: 0.1,
                },
            });
            (0, vitest_1.expect)(result.type).toBe('drift');
            if (result.type === 'drift') {
                (0, vitest_1.expect)(result.value.fromHeight).toBe(1);
                (0, vitest_1.expect)(result.value.toHeight).toBe(10);
            }
        });
        (0, vitest_1.it)('should convert height', () => {
            const result = (0, client_js_1.convertProtoChainResult)({ height: { height: 100 } });
            (0, vitest_1.expect)(result.type).toBe('height');
            if (result.type === 'height') {
                (0, vitest_1.expect)(result.value.height).toBe(100);
            }
        });
        (0, vitest_1.it)('should convert tip', () => {
            const result = (0, client_js_1.convertProtoChainResult)({ tip: { hash: 'tiphash', height: 50 } });
            (0, vitest_1.expect)(result.type).toBe('tip');
            if (result.type === 'tip') {
                (0, vitest_1.expect)(result.value.hash).toBe('tiphash');
                (0, vitest_1.expect)(result.value.height).toBe(50);
            }
        });
        (0, vitest_1.it)('should convert block', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                block: {
                    height: 10,
                    hash: 'h',
                    prevHash: 'ph',
                    timestamp: 123,
                    transactionCount: 5,
                    proposer: 'node1',
                },
            });
            (0, vitest_1.expect)(result.type).toBe('block');
            if (result.type === 'block') {
                (0, vitest_1.expect)(result.value.height).toBe(10);
                (0, vitest_1.expect)(result.value.proposer).toBe('node1');
            }
        });
        (0, vitest_1.it)('should convert codebook', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                codebook: { scope: 'global', entryCount: 100, dimension: 128, domain: 'text' },
            });
            (0, vitest_1.expect)(result.type).toBe('codebook');
            if (result.type === 'codebook') {
                (0, vitest_1.expect)(result.value.scope).toBe('global');
                (0, vitest_1.expect)(result.value.domain).toBe('text');
            }
        });
        (0, vitest_1.it)('should convert transitionAnalysis', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                transitionAnalysis: {
                    totalTransitions: 100,
                    validTransitions: 95,
                    invalidTransitions: 5,
                    avgValidityScore: 0.95,
                },
            });
            (0, vitest_1.expect)(result.type).toBe('transitionAnalysis');
            if (result.type === 'transitionAnalysis') {
                (0, vitest_1.expect)(result.value.totalTransitions).toBe(100);
            }
        });
        (0, vitest_1.it)('should convert conflictResolution', () => {
            const result = (0, client_js_1.convertProtoChainResult)({
                conflictResolution: { strategy: 'semantic', conflictsResolved: 3 },
            });
            (0, vitest_1.expect)(result.type).toBe('conflictResolution');
            if (result.type === 'conflictResolution') {
                (0, vitest_1.expect)(result.value.strategy).toBe('semantic');
            }
        });
        (0, vitest_1.it)('should convert merge', () => {
            const result = (0, client_js_1.convertProtoChainResult)({ merge: { success: true, mergedCount: 10 } });
            (0, vitest_1.expect)(result.type).toBe('merge');
            if (result.type === 'merge') {
                (0, vitest_1.expect)(result.value.success).toBe(true);
                (0, vitest_1.expect)(result.value.mergedCount).toBe(10);
            }
        });
        (0, vitest_1.it)('should default to height 0 for empty', () => {
            const result = (0, client_js_1.convertProtoChainResult)({});
            (0, vitest_1.expect)(result.type).toBe('height');
            if (result.type === 'height') {
                (0, vitest_1.expect)(result.value.height).toBe(0);
            }
        });
    });
    (0, vitest_1.describe)('convertProtoUnifiedItem', () => {
        (0, vitest_1.it)('should convert unified item with fields', () => {
            const result = (0, client_js_1.convertProtoUnifiedItem)({
                entityType: 'user',
                key: 'u1',
                fields: { name: 'Alice', email: 'alice@example.com' },
                score: 0.9,
            });
            (0, vitest_1.expect)(result.entityType).toBe('user');
            (0, vitest_1.expect)(result.key).toBe('u1');
            (0, vitest_1.expect)(result.fields.get('name')?.data).toBe('Alice');
            (0, vitest_1.expect)(result.score).toBe(0.9);
        });
        (0, vitest_1.it)('should convert unified item without fields', () => {
            const result = (0, client_js_1.convertProtoUnifiedItem)({
                entityType: 'item',
                key: 'i1',
            });
            (0, vitest_1.expect)(result.entityType).toBe('item');
            (0, vitest_1.expect)(result.key).toBe('i1');
            (0, vitest_1.expect)(result.fields.size).toBe(0);
            (0, vitest_1.expect)(result.score).toBeUndefined();
        });
    });
});
(0, vitest_1.describe)('Response Type Conversion via execute()', () => {
    (0, vitest_1.it)('should handle value response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { value: { value: 'test-value' } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SELECT');
        (0, vitest_1.expect)(result.type).toBe('value');
        if (result.type === 'value') {
            (0, vitest_1.expect)(result.value).toBe('test-value');
        }
        client.close();
    });
    (0, vitest_1.it)('should handle artifactList response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { artifactList: { artifactIds: ['a1', 'a2'] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('LIST ARTIFACTS');
        (0, vitest_1.expect)(result.type).toBe('artifactList');
        if (result.type === 'artifactList') {
            (0, vitest_1.expect)(result.artifactIds).toEqual(['a1', 'a2']);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle blobStats response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                blobStats: {
                    artifactCount: 10,
                    chunkCount: 50,
                    totalBytes: 1000,
                    uniqueBytes: 800,
                    dedupRatio: 0.8,
                    orphanedChunks: 2,
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('BLOB STATS');
        (0, vitest_1.expect)(result.type).toBe('blobStats');
        if (result.type === 'blobStats') {
            (0, vitest_1.expect)(result.artifactCount).toBe(10);
            (0, vitest_1.expect)(result.dedupRatio).toBe(0.8);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle checkpointList response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                checkpointList: {
                    checkpoints: [{ id: 'cp1', name: 'checkpoint1', createdAt: 1700000000, isAuto: false }],
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('LIST CHECKPOINTS');
        (0, vitest_1.expect)(result.type).toBe('checkpointList');
        if (result.type === 'checkpointList') {
            (0, vitest_1.expect)(result.checkpoints.length).toBe(1);
            (0, vitest_1.expect)(result.checkpoints[0]?.id).toBe('cp1');
        }
        client.close();
    });
    (0, vitest_1.it)('should handle pageRank response with all fields', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                pageRank: {
                    items: [{ nodeId: 1, score: 0.85 }],
                    iterations: 20,
                    convergence: 0.001,
                    converged: true,
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('PAGERANK');
        (0, vitest_1.expect)(result.type).toBe('pageRank');
        if (result.type === 'pageRank') {
            (0, vitest_1.expect)(result.items.length).toBe(1);
            (0, vitest_1.expect)(result.iterations).toBe(20);
            (0, vitest_1.expect)(result.convergence).toBe(0.001);
            (0, vitest_1.expect)(result.converged).toBe(true);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle centrality response with all fields', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                centrality: {
                    items: [{ nodeId: 1, score: 0.5 }],
                    centralityType: 'CENTRALITY_TYPE_BETWEENNESS',
                    iterations: 10,
                    converged: true,
                    sampleCount: 100,
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('CENTRALITY');
        (0, vitest_1.expect)(result.type).toBe('centrality');
        if (result.type === 'centrality') {
            (0, vitest_1.expect)(result.items.length).toBe(1);
            (0, vitest_1.expect)(result.centralityType).toBe('betweenness');
            (0, vitest_1.expect)(result.iterations).toBe(10);
            (0, vitest_1.expect)(result.converged).toBe(true);
            (0, vitest_1.expect)(result.sampleCount).toBe(100);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle communities response with all fields', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                communities: {
                    items: [{ nodeId: 1, communityId: 0 }],
                    communityCount: 3,
                    modularity: 0.75,
                    passes: 5,
                    iterations: 100,
                    communities: [{ communityId: 0, memberNodeIds: [1, 2, 3] }],
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('COMMUNITIES');
        (0, vitest_1.expect)(result.type).toBe('communities');
        if (result.type === 'communities') {
            (0, vitest_1.expect)(result.items.length).toBe(1);
            (0, vitest_1.expect)(result.communityCount).toBe(3);
            (0, vitest_1.expect)(result.modularity).toBe(0.75);
            (0, vitest_1.expect)(result.passes).toBe(5);
            (0, vitest_1.expect)(result.iterations).toBe(100);
            (0, vitest_1.expect)(result.communities?.length).toBe(1);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle patternMatch response with stats', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                patternMatch: {
                    matches: [{ bindings: [{ variable: 'n', value: { node: { id: 1, label: 'Person' } } }] }],
                    stats: { matchesFound: 5, nodesEvaluated: 100, edgesEvaluated: 50, truncated: false },
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('MATCH');
        (0, vitest_1.expect)(result.type).toBe('patternMatch');
        if (result.type === 'patternMatch') {
            (0, vitest_1.expect)(result.matches.length).toBe(1);
            (0, vitest_1.expect)(result.stats?.matchesFound).toBe(5);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle constraints response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                constraints: {
                    items: [{ name: 'unique_email', target: 'users', property: 'email', constraintType: 'UNIQUE' }],
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SHOW CONSTRAINTS');
        (0, vitest_1.expect)(result.type).toBe('constraints');
        if (result.type === 'constraints') {
            (0, vitest_1.expect)(result.items.length).toBe(1);
            (0, vitest_1.expect)(result.items[0]?.name).toBe('unique_email');
        }
        client.close();
    });
    (0, vitest_1.it)('should handle aggregate response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { aggregate: { count: 42 } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('COUNT');
        (0, vitest_1.expect)(result.type).toBe('aggregate');
        if (result.type === 'aggregate') {
            (0, vitest_1.expect)(result.value.type).toBe('count');
            (0, vitest_1.expect)(result.value.value).toBe(42);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle batchOperation response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                batchOperation: { operation: 'insert', affectedCount: 10, createdIds: [1, 2, 3] },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('BATCH INSERT');
        (0, vitest_1.expect)(result.type).toBe('batchOperation');
        if (result.type === 'batchOperation') {
            (0, vitest_1.expect)(result.operation).toBe('insert');
            (0, vitest_1.expect)(result.affectedCount).toBe(10);
            (0, vitest_1.expect)(result.createdIds).toEqual(['1', '2', '3']);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle graphIndexes response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { graphIndexes: { indexes: ['idx_label', 'idx_prop'] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SHOW INDEXES');
        (0, vitest_1.expect)(result.type).toBe('graphIndexes');
        if (result.type === 'graphIndexes') {
            (0, vitest_1.expect)(result.indexes).toEqual(['idx_label', 'idx_prop']);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle chain response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { chain: { height: { height: 100 } } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('CHAIN HEIGHT');
        (0, vitest_1.expect)(result.type).toBe('chain');
        if (result.type === 'chain') {
            (0, vitest_1.expect)(result.result.type).toBe('height');
        }
        client.close();
    });
    (0, vitest_1.it)('should handle unified response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                unified: {
                    description: 'Search results',
                    items: [{ entityType: 'user', key: 'u1', fields: { name: 'Alice' }, score: 0.9 }],
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SEARCH');
        (0, vitest_1.expect)(result.type).toBe('unified');
        if (result.type === 'unified') {
            (0, vitest_1.expect)(result.description).toBe('Search results');
            (0, vitest_1.expect)(result.items.length).toBe(1);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle error response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { error: { code: 1, message: 'Query error' } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('INVALID');
        (0, vitest_1.expect)(result.type).toBe('error');
        if (result.type === 'error') {
            (0, vitest_1.expect)(result.code).toBe(1);
            (0, vitest_1.expect)(result.message).toBe('Query error');
        }
        client.close();
    });
    (0, vitest_1.it)('should handle count response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { count: { count: 42 } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('COUNT');
        (0, vitest_1.expect)(result.type).toBe('count');
        if (result.type === 'count') {
            (0, vitest_1.expect)(result.count).toBe(42);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle nodes response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { nodes: { nodes: [{ id: 'n1', label: 'Person' }] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('MATCH');
        (0, vitest_1.expect)(result.type).toBe('nodes');
        client.close();
    });
    (0, vitest_1.it)('should handle edges response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { edges: { edges: [{ id: 'e1', type: 'KNOWS', from: 'n1', to: 'n2' }] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('MATCH EDGES');
        (0, vitest_1.expect)(result.type).toBe('edges');
        client.close();
    });
    (0, vitest_1.it)('should handle path response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { path: { nodeIds: [1, 2, 3] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SHORTEST PATH');
        (0, vitest_1.expect)(result.type).toBe('paths');
        client.close();
    });
    (0, vitest_1.it)('should handle similar response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { similar: { items: [{ key: 'k1', score: 0.9 }] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SIMILAR');
        (0, vitest_1.expect)(result.type).toBe('similar');
        client.close();
    });
    (0, vitest_1.it)('should handle ids response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { ids: { ids: [1, 2, 3] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('GET IDS');
        (0, vitest_1.expect)(result.type).toBe('ids');
        if (result.type === 'ids') {
            (0, vitest_1.expect)(result.ids).toEqual(['1', '2', '3']);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle tableList response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { tableList: { tables: ['users', 'orders'] } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('SHOW TABLES');
        (0, vitest_1.expect)(result.type).toBe('tableList');
        if (result.type === 'tableList') {
            (0, vitest_1.expect)(result.names).toEqual(['users', 'orders']);
        }
        client.close();
    });
    (0, vitest_1.it)('should handle blob response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, { blob: { data: new Uint8Array([1, 2, 3]) } });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('GET BLOB');
        (0, vitest_1.expect)(result.type).toBe('blob');
        if (result.type === 'blob') {
            (0, vitest_1.expect)(result.data).toEqual(new Uint8Array([1, 2, 3]));
        }
        client.close();
    });
    (0, vitest_1.it)('should handle artifactInfo response', async () => {
        mockGrpcClient.Execute.mockImplementationOnce((_request, _metadata, callback) => {
            callback(null, {
                artifactInfo: {
                    artifactId: 'a1',
                    filename: 'test.txt',
                    size: 100,
                    checksum: 'abc',
                    contentType: 'text/plain',
                    createdAt: 123,
                    tags: ['tag1'],
                },
            });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        const result = await client.execute('GET ARTIFACT INFO');
        (0, vitest_1.expect)(result.type).toBe('blobInfo');
        client.close();
    });
});
(0, vitest_1.describe)('Validation Utilities', () => {
    (0, vitest_1.describe)('validateIntValue', () => {
        (0, vitest_1.it)('should accept valid integers', () => {
            (0, vitest_1.expect)((0, validation_js_1.validateIntValue)(0)).toBe(0);
            (0, vitest_1.expect)((0, validation_js_1.validateIntValue)(42)).toBe(42);
            (0, vitest_1.expect)((0, validation_js_1.validateIntValue)(-100)).toBe(-100);
            (0, vitest_1.expect)((0, validation_js_1.validateIntValue)(Number.MAX_SAFE_INTEGER)).toBe(Number.MAX_SAFE_INTEGER);
            (0, vitest_1.expect)((0, validation_js_1.validateIntValue)(Number.MIN_SAFE_INTEGER)).toBe(Number.MIN_SAFE_INTEGER);
        });
        (0, vitest_1.it)('should reject NaN', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.validateIntValue)(NaN)).toThrow(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should reject Infinity', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.validateIntValue)(Infinity)).toThrow(errors_js_1.InvalidArgumentError);
            (0, vitest_1.expect)(() => (0, validation_js_1.validateIntValue)(-Infinity)).toThrow(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should reject values outside safe integer range', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.validateIntValue)(Number.MAX_SAFE_INTEGER + 1)).toThrow(errors_js_1.InvalidArgumentError);
            (0, vitest_1.expect)(() => (0, validation_js_1.validateIntValue)(Number.MIN_SAFE_INTEGER - 1)).toThrow(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should include context in error message', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.validateIntValue)(NaN, 'test field')).toThrow('test field');
        });
    });
    (0, vitest_1.describe)('validateFloatValue', () => {
        (0, vitest_1.it)('should accept valid floats', () => {
            (0, vitest_1.expect)((0, validation_js_1.validateFloatValue)(0)).toBe(0);
            (0, vitest_1.expect)((0, validation_js_1.validateFloatValue)(3.14159)).toBe(3.14159);
            (0, vitest_1.expect)((0, validation_js_1.validateFloatValue)(-273.15)).toBe(-273.15);
        });
        (0, vitest_1.it)('should reject NaN', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.validateFloatValue)(NaN)).toThrow(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should reject Infinity', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.validateFloatValue)(Infinity)).toThrow(errors_js_1.InvalidArgumentError);
            (0, vitest_1.expect)(() => (0, validation_js_1.validateFloatValue)(-Infinity)).toThrow(errors_js_1.InvalidArgumentError);
        });
    });
    (0, vitest_1.describe)('validateStringValue', () => {
        (0, vitest_1.it)('should accept strings within limit', () => {
            (0, vitest_1.expect)((0, validation_js_1.validateStringValue)('')).toBe('');
            (0, vitest_1.expect)((0, validation_js_1.validateStringValue)('hello')).toBe('hello');
        });
        (0, vitest_1.it)('should reject strings exceeding limit', () => {
            const longString = 'x'.repeat(validation_js_1.MAX_STRING_LENGTH + 1);
            (0, vitest_1.expect)(() => (0, validation_js_1.validateStringValue)(longString)).toThrow(errors_js_1.InvalidArgumentError);
        });
    });
    (0, vitest_1.describe)('validateBytesValue', () => {
        (0, vitest_1.it)('should accept bytes within limit', () => {
            const bytes = new Uint8Array([1, 2, 3]);
            (0, vitest_1.expect)((0, validation_js_1.validateBytesValue)(bytes)).toBe(bytes);
        });
        (0, vitest_1.it)('should reject bytes exceeding limit', () => {
            const largeBytes = new Uint8Array(validation_js_1.MAX_BYTES_LENGTH + 1);
            (0, vitest_1.expect)(() => (0, validation_js_1.validateBytesValue)(largeBytes)).toThrow(errors_js_1.InvalidArgumentError);
        });
    });
    (0, vitest_1.describe)('safeIdToString', () => {
        (0, vitest_1.it)('should convert safe integers to strings', () => {
            (0, vitest_1.expect)((0, validation_js_1.safeIdToString)(0)).toBe('0');
            (0, vitest_1.expect)((0, validation_js_1.safeIdToString)(42)).toBe('42');
            (0, vitest_1.expect)((0, validation_js_1.safeIdToString)(Number.MAX_SAFE_INTEGER)).toBe(String(Number.MAX_SAFE_INTEGER));
        });
        (0, vitest_1.it)('should reject unsafe integers', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.safeIdToString)(Number.MAX_SAFE_INTEGER + 1)).toThrow(errors_js_1.InvalidArgumentError);
            (0, vitest_1.expect)(() => (0, validation_js_1.safeIdToString)(NaN)).toThrow(errors_js_1.InvalidArgumentError);
        });
    });
    (0, vitest_1.describe)('safeIdsToStrings', () => {
        (0, vitest_1.it)('should convert array of safe integers', () => {
            (0, vitest_1.expect)((0, validation_js_1.safeIdsToStrings)([1, 2, 3])).toEqual(['1', '2', '3']);
        });
        (0, vitest_1.it)('should reject array with unsafe integers', () => {
            (0, vitest_1.expect)(() => (0, validation_js_1.safeIdsToStrings)([1, Number.MAX_SAFE_INTEGER + 1, 3])).toThrow(errors_js_1.InvalidArgumentError);
        });
    });
});
(0, vitest_1.describe)('Proto Value Validation', () => {
    (0, vitest_1.it)('should reject NaN in int values', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoValue)({ intValue: NaN })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should reject Infinity in int values', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoValue)({ intValue: Infinity })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should reject NaN in float values', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoValue)({ floatValue: NaN })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should reject Infinity in float values', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoValue)({ floatValue: Infinity })).toThrow(errors_js_1.InvalidArgumentError);
    });
});
(0, vitest_1.describe)('Copy Helper Functions', () => {
    (0, vitest_1.describe)('copyRowValues', () => {
        (0, vitest_1.it)('should create mutable copy of row values', () => {
            const row = { values: new Map([['id', (0, value_js_1.intValue)(1)]]) };
            const copy = (0, query_result_js_1.copyRowValues)(row);
            (0, vitest_1.expect)(copy).toBeInstanceOf(Map);
            (0, vitest_1.expect)(copy.get('id')?.data).toBe(1);
            copy.set('new', (0, value_js_1.stringValue)('test'));
            (0, vitest_1.expect)(copy.has('new')).toBe(true);
            (0, vitest_1.expect)(row.values.has('new')).toBe(false);
        });
    });
    (0, vitest_1.describe)('copyNodeProperties', () => {
        (0, vitest_1.it)('should create mutable copy of node properties', () => {
            const node = {
                id: 'n1',
                label: 'Test',
                properties: new Map([['age', (0, value_js_1.intValue)(30)]]),
            };
            const copy = (0, query_result_js_1.copyNodeProperties)(node);
            (0, vitest_1.expect)(copy.get('age')?.data).toBe(30);
            copy.set('name', (0, value_js_1.stringValue)('Alice'));
            (0, vitest_1.expect)(copy.has('name')).toBe(true);
            (0, vitest_1.expect)(node.properties.has('name')).toBe(false);
        });
    });
    (0, vitest_1.describe)('copyEdgeProperties', () => {
        (0, vitest_1.it)('should create mutable copy of edge properties', () => {
            const edge = {
                id: 'e1',
                edgeType: 'KNOWS',
                source: 'n1',
                target: 'n2',
                properties: new Map([['weight', (0, value_js_1.floatValue)(0.5)]]),
            };
            const copy = (0, query_result_js_1.copyEdgeProperties)(edge);
            (0, vitest_1.expect)(copy.get('weight')?.data).toBe(0.5);
            copy.set('since', (0, value_js_1.intValue)(2020));
            (0, vitest_1.expect)(copy.has('since')).toBe(true);
            (0, vitest_1.expect)(edge.properties.has('since')).toBe(false);
        });
    });
    (0, vitest_1.describe)('copySimilarItemMetadata', () => {
        (0, vitest_1.it)('should return undefined for items without metadata', () => {
            const item = { key: 'k1', score: 0.9 };
            (0, vitest_1.expect)((0, query_result_js_1.copySimilarItemMetadata)(item)).toBeUndefined();
        });
        (0, vitest_1.it)('should create mutable copy of metadata', () => {
            const item = {
                key: 'k1',
                score: 0.9,
                metadata: new Map([['tag', (0, value_js_1.stringValue)('test')]]),
            };
            const copy = (0, query_result_js_1.copySimilarItemMetadata)(item);
            (0, vitest_1.expect)(copy).toBeInstanceOf(Map);
            (0, vitest_1.expect)(copy?.get('tag')?.data).toBe('test');
        });
    });
    (0, vitest_1.describe)('copyUnifiedItemFields', () => {
        (0, vitest_1.it)('should create mutable copy of unified item fields', () => {
            const item = {
                entityType: 'user',
                key: 'u1',
                fields: new Map([['name', (0, value_js_1.stringValue)('Alice')]]),
            };
            const copy = (0, query_result_js_1.copyUnifiedItemFields)(item);
            (0, vitest_1.expect)(copy.get('name')?.data).toBe('Alice');
            copy.set('email', (0, value_js_1.stringValue)('alice@example.com'));
            (0, vitest_1.expect)(copy.has('email')).toBe(true);
            (0, vitest_1.expect)(item.fields.has('email')).toBe(false);
        });
    });
});
(0, vitest_1.describe)('UnifiedItem Field Conversion', () => {
    (0, vitest_1.it)('should convert mixed field types correctly', () => {
        const result = (0, client_js_1.convertProtoUnifiedItem)({
            entityType: 'user',
            key: 'u1',
            fields: {
                name: 'Alice',
                age: 30,
                active: true,
                score: 3.14,
            },
        });
        (0, vitest_1.expect)(result.fields.get('name')?.type).toBe('string');
        (0, vitest_1.expect)(result.fields.get('name')?.data).toBe('Alice');
        (0, vitest_1.expect)(result.fields.get('age')?.type).toBe('int');
        (0, vitest_1.expect)(result.fields.get('age')?.data).toBe(30);
        (0, vitest_1.expect)(result.fields.get('active')?.type).toBe('bool');
        (0, vitest_1.expect)(result.fields.get('active')?.data).toBe(true);
        (0, vitest_1.expect)(result.fields.get('score')?.type).toBe('float');
        (0, vitest_1.expect)(result.fields.get('score')?.data).toBe(3.14);
    });
    (0, vitest_1.it)('should handle null field values', () => {
        const result = (0, client_js_1.convertProtoUnifiedItem)({
            entityType: 'user',
            key: 'u1',
            fields: { nullable: null },
        });
        (0, vitest_1.expect)(result.fields.get('nullable')?.type).toBe('null');
    });
});
(0, vitest_1.describe)('Cursor Cleanup on Early Break', () => {
    (0, vitest_1.it)('should close cursor when breaking early from executeAllPages', async () => {
        let closeCount = 0;
        let callCount = 0;
        mockGrpcClient.ExecutePaginated.mockImplementation((_request, _metadata, callback) => {
            callCount++;
            callback(null, {
                result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: callCount } }] }] } },
                nextCursor: 'cursor_' + callCount,
                hasMore: true,
                pageSize: 10,
            });
        });
        mockGrpcClient.CloseCursor.mockImplementation((_request, _metadata, callback) => {
            closeCount++;
            callback(null, { success: true });
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        let iterations = 0;
        for await (const _result of client.executeAllPages('SELECT * FROM users')) {
            iterations++;
            if (iterations >= 2)
                break;
        }
        (0, vitest_1.expect)(iterations).toBe(2);
        (0, vitest_1.expect)(closeCount).toBe(1);
        client.close();
        // Reset mocks
        mockGrpcClient.ExecutePaginated.mockImplementation((_request, _metadata, callback) => {
            callback(null, mockPaginatedResponse);
        });
        mockGrpcClient.CloseCursor.mockImplementation((_request, _metadata, callback) => {
            callback(null, mockCloseCursorResponse);
        });
    });
    (0, vitest_1.it)('should ignore cursor cleanup errors', async () => {
        let callCount = 0;
        mockGrpcClient.ExecutePaginated.mockImplementation((_request, _metadata, callback) => {
            callCount++;
            callback(null, {
                result: { rows: { rows: [{ columns: [{ name: 'id', value: { intValue: callCount } }] }] } },
                nextCursor: 'cursor_' + callCount,
                hasMore: true,
                pageSize: 10,
            });
        });
        mockGrpcClient.CloseCursor.mockImplementation((_request, _metadata, callback) => {
            callback({ code: 5, details: 'Cursor not found' }, null);
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        let iterations = 0;
        // Should not throw even though cursor close fails
        for await (const _result of client.executeAllPages('SELECT * FROM users')) {
            iterations++;
            if (iterations >= 1)
                break;
        }
        (0, vitest_1.expect)(iterations).toBe(1);
        client.close();
        // Reset mocks
        mockGrpcClient.ExecutePaginated.mockImplementation((_request, _metadata, callback) => {
            callback(null, mockPaginatedResponse);
        });
        mockGrpcClient.CloseCursor.mockImplementation((_request, _metadata, callback) => {
            callback(null, mockCloseCursorResponse);
        });
    });
});
(0, vitest_1.describe)('Stream Cleanup', () => {
    (0, vitest_1.it)('should call cancel on stream when breaking early', async () => {
        let cancelCalled = false;
        const streamResponses = [
            { row: { row: { columns: [{ name: 'x', value: { intValue: 1 } }] } } },
            { row: { row: { columns: [{ name: 'x', value: { intValue: 2 } }] } } },
            { isFinal: true },
        ];
        let index = 0;
        mockGrpcClient.ExecuteStream.mockImplementationOnce(() => {
            const stream = {
                [Symbol.asyncIterator]: () => ({
                    next() {
                        if (index < streamResponses.length) {
                            return Promise.resolve({ value: streamResponses[index++], done: false });
                        }
                        return Promise.resolve({ value: undefined, done: true });
                    },
                }),
                cancel: () => {
                    cancelCalled = true;
                },
            };
            return stream;
        });
        const client = await client_js_1.NeumannClient.connect('localhost:50051');
        for await (const _result of client.executeStream('SELECT * FROM users')) {
            break; // Break immediately
        }
        (0, vitest_1.expect)(cancelCalled).toBe(true);
        client.close();
    });
});
(0, vitest_1.describe)('Safe ID Conversions', () => {
    (0, vitest_1.it)('should handle unsafe page rank node IDs', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoPageRankItem)({ nodeId: Number.MAX_SAFE_INTEGER + 1, score: 0.5 })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should handle unsafe centrality node IDs', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoCentralityItem)({ nodeId: Number.MAX_SAFE_INTEGER + 1, score: 0.5 })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should handle unsafe community node IDs', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoCommunityItem)({ nodeId: Number.MAX_SAFE_INTEGER + 1, communityId: 1 })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should handle unsafe community member IDs', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoCommunityMemberList)({
            communityId: 1,
            memberNodeIds: [1, Number.MAX_SAFE_INTEGER + 1],
        })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should handle unsafe binding value node ID', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoBindingValue)({ node: { id: Number.MAX_SAFE_INTEGER + 1, label: 'Person' } })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should handle unsafe binding value edge IDs', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoBindingValue)({
            edge: { id: Number.MAX_SAFE_INTEGER + 1, edgeType: 'KNOWS', from: 1, to: 2 },
        })).toThrow(errors_js_1.InvalidArgumentError);
    });
    (0, vitest_1.it)('should handle unsafe binding value path node IDs', () => {
        (0, vitest_1.expect)(() => (0, client_js_1.convertProtoBindingValue)({
            path: { nodes: [1, Number.MAX_SAFE_INTEGER + 1], edges: [10], length: 1 },
        })).toThrow(errors_js_1.InvalidArgumentError);
    });
});
//# sourceMappingURL=client.test.js.map