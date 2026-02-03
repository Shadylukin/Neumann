import { ConnectionError, NotFoundError, InternalError, InvalidArgumentError, } from '../types/errors.js';
/**
 * Build upload metadata with proper typing for exactOptionalPropertyTypes.
 */
function buildUploadMetadata(filename, options) {
    const result = {
        filename,
        tags: options.tags ?? [],
        linkedTo: options.linkedTo ?? [],
        custom: options.custom ?? {},
    };
    if (options.contentType !== undefined) {
        result.contentType = options.contentType;
    }
    if (options.createdBy !== undefined) {
        result.createdBy = options.createdBy;
    }
    return result;
}
/**
 * Service client for blob/artifact operations.
 */
export class BlobClient {
    client;
    metadata;
    constructor(client, metadata) {
        this.client = client;
        this.metadata = metadata;
    }
    /**
     * Upload a blob from a buffer.
     *
     * @param filename - The filename for the artifact.
     * @param data - The blob data as a Buffer or Uint8Array.
     * @param options - Upload options.
     * @returns Upload result with artifact ID.
     */
    async uploadBlob(filename, data, options = {}) {
        return new Promise((resolve, reject) => {
            const call = this.client.Upload(this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve({
                    artifactId: response.artifactId,
                    size: response.size,
                    checksum: response.checksum,
                });
            });
            // Send metadata first
            call.write({
                metadata: buildUploadMetadata(filename, options),
            });
            // Send data in chunks (64KB chunks)
            const CHUNK_SIZE = 64 * 1024;
            const buffer = Buffer.from(data);
            for (let offset = 0; offset < buffer.length; offset += CHUNK_SIZE) {
                const chunk = buffer.subarray(offset, offset + CHUNK_SIZE);
                call.write({ chunk });
            }
            call.end();
        });
    }
    /**
     * Upload a blob from an async iterable (streaming).
     *
     * @param filename - The filename for the artifact.
     * @param chunks - Async iterable of data chunks.
     * @param options - Upload options.
     * @returns Upload result with artifact ID.
     */
    async uploadBlobStreaming(filename, chunks, options = {}) {
        return new Promise((resolve, reject) => {
            const call = this.client.Upload(this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve({
                    artifactId: response.artifactId,
                    size: response.size,
                    checksum: response.checksum,
                });
            });
            // Send metadata first
            call.write({
                metadata: buildUploadMetadata(filename, options),
            });
            // Stream chunks with proper error handling
            const streamChunks = async () => {
                try {
                    for await (const chunk of chunks) {
                        call.write({ chunk });
                    }
                    call.end();
                }
                catch (err) {
                    call.destroy(err instanceof Error ? err : new Error(String(err)));
                }
            };
            streamChunks().catch((err) => reject(this.handleError(err)));
        });
    }
    /**
     * Download a blob as an async iterable of chunks.
     * Automatically cancels the stream on early break or error.
     *
     * @param artifactId - The artifact ID to download.
     * @returns Async iterable of data chunks.
     */
    async *downloadBlob(artifactId) {
        const stream = this.client.Download({ artifactId }, this.metadata);
        try {
            for await (const chunk of stream) {
                if (chunk.data && chunk.data.length > 0) {
                    yield chunk.data;
                }
                if (chunk.isFinal) {
                    break;
                }
            }
        }
        finally {
            if (typeof stream.cancel === 'function') {
                try {
                    stream.cancel();
                }
                catch {
                    // Ignore cancel errors
                }
            }
        }
    }
    /**
     * Download a blob as a complete buffer.
     *
     * @param artifactId - The artifact ID to download.
     * @returns The complete blob data.
     */
    async downloadBlobFull(artifactId) {
        const chunks = [];
        for await (const chunk of this.downloadBlob(artifactId)) {
            chunks.push(chunk);
        }
        return Buffer.concat(chunks);
    }
    /**
     * Delete a blob.
     *
     * @param artifactId - The artifact ID to delete.
     * @returns True if deletion was successful.
     */
    async deleteBlob(artifactId) {
        return new Promise((resolve, reject) => {
            this.client.Delete({ artifactId }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.success);
            });
        });
    }
    /**
     * Get blob metadata.
     *
     * @param artifactId - The artifact ID.
     * @returns Artifact metadata.
     */
    async getBlobMetadata(artifactId) {
        return new Promise((resolve, reject) => {
            this.client.GetMetadata({ artifactId }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve({
                    id: response.id,
                    filename: response.filename,
                    contentType: response.contentType,
                    size: response.size,
                    checksum: response.checksum,
                    chunkCount: response.chunkCount,
                    created: new Date(response.created),
                    modified: new Date(response.modified),
                    createdBy: response.createdBy,
                    tags: response.tags,
                    linkedTo: response.linkedTo,
                    custom: response.custom ?? {},
                });
            });
        });
    }
    handleError(err) {
        const code = err.code;
        const NOT_FOUND = 5;
        const INVALID_ARGUMENT = 3;
        const UNAVAILABLE = 14;
        if (code === NOT_FOUND) {
            return new NotFoundError(err.details || 'Artifact not found');
        }
        if (code === INVALID_ARGUMENT) {
            return new InvalidArgumentError(err.details || 'Invalid argument');
        }
        if (code === UNAVAILABLE) {
            return new ConnectionError(err.details || 'Service unavailable');
        }
        return new InternalError(err.details || err.message || 'Internal error');
    }
}
//# sourceMappingURL=blob.js.map