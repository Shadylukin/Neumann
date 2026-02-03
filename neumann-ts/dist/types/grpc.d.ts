/**
 * gRPC module for Node.js environments.
 *
 * This module provides factory functions for creating gRPC service clients.
 * It uses dynamic proto loading with a portable path resolution that works
 * in both ESM and CommonJS builds.
 */
import * as grpc from '@grpc/grpc-js';
export type { QueryServiceClient, BlobServiceClient, HealthClient, QueryRequest, QueryResponse, QueryResponseChunk, BatchQueryRequest, BatchQueryResponse, BlobUploadRequest, BlobUploadResponse, BlobDownloadRequest, BlobDownloadChunk, BlobDeleteRequest, BlobDeleteResponse, BlobMetadataRequest, HealthCheckRequest, HealthCheckResponse, ArtifactInfo, ErrorInfo, ErrorCode, ServingStatus, } from './generated/neumann.js';
export type { PointsServiceClient, CollectionsServiceClient, Point, ScoredPoint, UpsertPointsRequest, UpsertPointsResponse, GetPointsRequest, GetPointsResponse, DeletePointsRequest, DeletePointsResponse, QueryPointsRequest, QueryPointsResponse, ScrollPointsRequest, ScrollPointsResponse, CreateCollectionRequest, CreateCollectionResponse, GetCollectionRequest, GetCollectionResponse, DeleteCollectionRequest, DeleteCollectionResponse, ListCollectionsRequest, ListCollectionsResponse, } from './generated/vector.js';
/**
 * Load the Neumann proto definition.
 * The proto is cached after the first load.
 */
export declare function loadProto(): Promise<grpc.GrpcObject>;
/**
 * Load the Vector proto definition.
 * The proto is cached after the first load.
 */
export declare function loadVectorProto(): Promise<grpc.GrpcObject>;
/**
 * Create a QueryService client.
 */
export declare function getQueryServiceClient(proto: grpc.GrpcObject, address: string, credentials: grpc.ChannelCredentials): grpc.Client;
/**
 * Create a BlobService client.
 */
export declare function getBlobServiceClient(proto: grpc.GrpcObject, address: string, credentials: grpc.ChannelCredentials): grpc.Client;
/**
 * Create a Health client.
 */
export declare function getHealthClient(proto: grpc.GrpcObject, address: string, credentials: grpc.ChannelCredentials): grpc.Client;
/**
 * Create a PointsService client.
 */
export declare function getPointsServiceClient(proto: grpc.GrpcObject, address: string, credentials: grpc.ChannelCredentials): grpc.Client;
/**
 * Create a CollectionsService client.
 */
export declare function getCollectionsServiceClient(proto: grpc.GrpcObject, address: string, credentials: grpc.ChannelCredentials): grpc.Client;
/**
 * Clean up temp proto files.
 */
export declare function cleanup(): void;
//# sourceMappingURL=grpc.d.ts.map