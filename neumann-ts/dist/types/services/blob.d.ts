/**
 * BlobService client for artifact storage operations.
 */
import type * as grpc from '@grpc/grpc-js';
import type { BlobServiceClient } from '../generated/neumann.js';
/**
 * Options for blob upload.
 */
export interface BlobUploadOptions {
    /** Content type (MIME type). */
    contentType?: string;
    /** Creator identifier. */
    createdBy?: string;
    /** Tags for categorization. */
    tags?: string[];
    /** IDs of linked artifacts. */
    linkedTo?: string[];
    /** Custom metadata. */
    custom?: Record<string, string>;
}
/**
 * Result of a blob upload operation.
 */
export interface BlobUploadResult {
    /** Generated artifact ID. */
    artifactId: string;
    /** Total size in bytes. */
    size: number;
    /** SHA-256 checksum. */
    checksum: string;
}
/**
 * Artifact metadata.
 */
export interface ArtifactMetadata {
    id: string;
    filename: string;
    contentType: string;
    size: number;
    checksum: string;
    chunkCount: number;
    created: Date;
    modified: Date;
    createdBy: string;
    tags: string[];
    linkedTo: string[];
    custom: Record<string, string>;
}
/**
 * Service client for blob/artifact operations.
 */
export declare class BlobClient {
    private client;
    private metadata;
    constructor(client: BlobServiceClient, metadata: grpc.Metadata);
    /**
     * Upload a blob from a buffer.
     *
     * @param filename - The filename for the artifact.
     * @param data - The blob data as a Buffer or Uint8Array.
     * @param options - Upload options.
     * @returns Upload result with artifact ID.
     */
    uploadBlob(filename: string, data: Buffer | Uint8Array, options?: BlobUploadOptions): Promise<BlobUploadResult>;
    /**
     * Upload a blob from an async iterable (streaming).
     *
     * @param filename - The filename for the artifact.
     * @param chunks - Async iterable of data chunks.
     * @param options - Upload options.
     * @returns Upload result with artifact ID.
     */
    uploadBlobStreaming(filename: string, chunks: AsyncIterable<Uint8Array>, options?: BlobUploadOptions): Promise<BlobUploadResult>;
    /**
     * Download a blob as an async iterable of chunks.
     * Automatically cancels the stream on early break or error.
     *
     * @param artifactId - The artifact ID to download.
     * @returns Async iterable of data chunks.
     */
    downloadBlob(artifactId: string): AsyncIterable<Uint8Array>;
    /**
     * Download a blob as a complete buffer.
     *
     * @param artifactId - The artifact ID to download.
     * @returns The complete blob data.
     */
    downloadBlobFull(artifactId: string): Promise<Buffer>;
    /**
     * Delete a blob.
     *
     * @param artifactId - The artifact ID to delete.
     * @returns True if deletion was successful.
     */
    deleteBlob(artifactId: string): Promise<boolean>;
    /**
     * Get blob metadata.
     *
     * @param artifactId - The artifact ID.
     * @returns Artifact metadata.
     */
    getBlobMetadata(artifactId: string): Promise<ArtifactMetadata>;
    private handleError;
}
//# sourceMappingURL=blob.d.ts.map