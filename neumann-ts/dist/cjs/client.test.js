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
// Mock gRPC modules
vitest_1.vi.mock('@grpc/grpc-js', () => ({
    credentials: {
        createSsl: vitest_1.vi.fn(() => ({})),
        createInsecure: vitest_1.vi.fn(() => ({})),
    },
    Channel: vitest_1.vi.fn().mockImplementation(() => ({
        getTarget: () => 'localhost:50051',
    })),
}));
vitest_1.vi.mock('grpc-web', () => ({
    GrpcWebClientBase: vitest_1.vi.fn().mockImplementation(() => ({})),
}));
(0, vitest_1.describe)('NeumannClient', () => {
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
            // Temporarily make the Channel constructor throw
            const grpc = await Promise.resolve().then(() => __importStar(require('@grpc/grpc-js')));
            const originalChannel = grpc.Channel;
            grpc.Channel = vitest_1.vi.fn().mockImplementation(() => {
                throw new Error('Connection refused');
            });
            await (0, vitest_1.expect)(client_js_1.NeumannClient.connect('localhost:50051')).rejects.toThrow(errors_js_1.ConnectionError);
            // Restore
            grpc.Channel = originalChannel;
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
            grpcWeb.GrpcWebClientBase = vitest_1.vi.fn().mockImplementation(() => {
                throw new Error('gRPC-Web initialization failed');
            });
            await (0, vitest_1.expect)(client_js_1.NeumannClient.connectWeb('http://localhost:8080')).rejects.toThrow(errors_js_1.ConnectionError);
            // Restore
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
//# sourceMappingURL=client.test.js.map