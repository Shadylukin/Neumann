"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// SPDX-License-Identifier: MIT
const vitest_1 = require("vitest");
const blob_js_1 = require("./blob.js");
const errors_js_1 = require("../types/errors.js");
// Mock gRPC Metadata
function createMockMetadata() {
    return {
        set: vitest_1.vi.fn(),
        get: vitest_1.vi.fn(),
        clone: vitest_1.vi.fn(),
    };
}
// Mock writable stream for uploads
function createMockWritableStream(response) {
    const stream = {
        write: vitest_1.vi.fn(),
        end: vitest_1.vi.fn(),
        destroy: vitest_1.vi.fn(),
        on: vitest_1.vi.fn(),
    };
    // Capture the callback and call it when end() is called
    let callback = null;
    return {
        stream,
        setCallback: (cb) => {
            callback = cb;
        },
        triggerResponse: () => {
            if (callback) {
                callback(null, response);
            }
        },
        triggerError: (err) => {
            if (callback) {
                callback(err, {});
            }
        },
    };
}
// Mock readable stream for downloads
function createMockReadableStream(chunks) {
    let index = 0;
    const stream = {
        async *[Symbol.asyncIterator]() {
            while (index < chunks.length) {
                yield chunks[index++];
            }
        },
    };
    return stream;
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
(0, vitest_1.describe)('BlobClient', () => {
    let client;
    let mockGrpcClient;
    let mockMetadata;
    (0, vitest_1.beforeEach)(() => {
        mockMetadata = createMockMetadata();
        mockGrpcClient = {
            Upload: vitest_1.vi.fn(),
            Download: vitest_1.vi.fn(),
            Delete: vitest_1.vi.fn(),
            GetMetadata: vitest_1.vi.fn(),
        };
        client = new blob_js_1.BlobClient(mockGrpcClient, mockMetadata);
    });
    (0, vitest_1.describe)('uploadBlob', () => {
        (0, vitest_1.it)('should upload a blob and return artifact info', async () => {
            const response = {
                artifactId: 'artifact-123',
                size: 1024,
                checksum: 'abc123',
            };
            const mockStream = createMockWritableStream(response);
            vitest_1.vi.mocked(mockGrpcClient.Upload).mockImplementation((_metadata, callback) => {
                mockStream.setCallback(callback);
                // Simulate async response after stream ends
                setTimeout(() => mockStream.triggerResponse(), 0);
                return mockStream.stream;
            });
            const result = await client.uploadBlob('test.txt', Buffer.from('hello world'), {
                contentType: 'text/plain',
                tags: ['test'],
            });
            (0, vitest_1.expect)(result).toEqual({
                artifactId: 'artifact-123',
                size: 1024,
                checksum: 'abc123',
            });
            (0, vitest_1.expect)(mockStream.stream.write).toHaveBeenCalled();
            (0, vitest_1.expect)(mockStream.stream.end).toHaveBeenCalled();
        });
        (0, vitest_1.it)('should handle upload errors', async () => {
            const mockStream = createMockWritableStream({});
            vitest_1.vi.mocked(mockGrpcClient.Upload).mockImplementation((_metadata, callback) => {
                mockStream.setCallback(callback);
                setTimeout(() => mockStream.triggerError(createServiceError(5, 'Not found')), 0);
                return mockStream.stream;
            });
            await (0, vitest_1.expect)(client.uploadBlob('test.txt', Buffer.from('data'))).rejects.toThrow(errors_js_1.NotFoundError);
        });
    });
    (0, vitest_1.describe)('downloadBlob', () => {
        (0, vitest_1.it)('should download a blob as chunks', async () => {
            const chunks = [
                { data: new Uint8Array([1, 2, 3]), isFinal: false },
                { data: new Uint8Array([4, 5, 6]), isFinal: false },
                { data: new Uint8Array([7, 8, 9]), isFinal: true },
            ];
            const mockStream = createMockReadableStream(chunks);
            vitest_1.vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream);
            const receivedChunks = [];
            for await (const chunk of client.downloadBlob('artifact-123')) {
                receivedChunks.push(chunk);
            }
            (0, vitest_1.expect)(receivedChunks).toHaveLength(3);
            (0, vitest_1.expect)(receivedChunks[0]).toEqual(new Uint8Array([1, 2, 3]));
        });
    });
    (0, vitest_1.describe)('downloadBlobFull', () => {
        (0, vitest_1.it)('should download a complete blob', async () => {
            const chunks = [
                { data: new Uint8Array([1, 2, 3]), isFinal: false },
                { data: new Uint8Array([4, 5, 6]), isFinal: true },
            ];
            const mockStream = createMockReadableStream(chunks);
            vitest_1.vi.mocked(mockGrpcClient.Download).mockReturnValue(mockStream);
            const result = await client.downloadBlobFull('artifact-123');
            (0, vitest_1.expect)(result).toBeInstanceOf(Buffer);
            (0, vitest_1.expect)(result).toEqual(Buffer.from([1, 2, 3, 4, 5, 6]));
        });
    });
    (0, vitest_1.describe)('deleteBlob', () => {
        (0, vitest_1.it)('should delete a blob', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(null, { success: true });
                return {};
            });
            const result = await client.deleteBlob('artifact-123');
            (0, vitest_1.expect)(result).toBe(true);
        });
        (0, vitest_1.it)('should handle delete errors', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(5, 'Not found'), { success: false });
                return {};
            });
            await (0, vitest_1.expect)(client.deleteBlob('artifact-123')).rejects.toThrow(errors_js_1.NotFoundError);
        });
    });
    (0, vitest_1.describe)('getBlobMetadata', () => {
        (0, vitest_1.it)('should get blob metadata', async () => {
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
            };
            vitest_1.vi.mocked(mockGrpcClient.GetMetadata).mockImplementation((_request, _metadata, callback) => {
                callback(null, artifactInfo);
                return {};
            });
            const result = await client.getBlobMetadata('artifact-123');
            (0, vitest_1.expect)(result.id).toBe('artifact-123');
            (0, vitest_1.expect)(result.filename).toBe('test.txt');
            (0, vitest_1.expect)(result.contentType).toBe('text/plain');
        });
    });
    (0, vitest_1.describe)('error handling', () => {
        (0, vitest_1.it)('should handle UNAVAILABLE error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(14, 'Service unavailable'), { success: false });
                return {};
            });
            await (0, vitest_1.expect)(client.deleteBlob('artifact-123')).rejects.toThrow(errors_js_1.ConnectionError);
        });
        (0, vitest_1.it)('should handle INVALID_ARGUMENT error', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(3, 'Invalid argument'), { success: false });
                return {};
            });
            await (0, vitest_1.expect)(client.deleteBlob('artifact-123')).rejects.toThrow(errors_js_1.InvalidArgumentError);
        });
        (0, vitest_1.it)('should handle unknown errors as InternalError', async () => {
            vitest_1.vi.mocked(mockGrpcClient.Delete).mockImplementation((_request, _metadata, callback) => {
                callback(createServiceError(99, 'Unknown error'), { success: false });
                return {};
            });
            await (0, vitest_1.expect)(client.deleteBlob('artifact-123')).rejects.toThrow(errors_js_1.InternalError);
        });
    });
});
//# sourceMappingURL=blob.test.js.map