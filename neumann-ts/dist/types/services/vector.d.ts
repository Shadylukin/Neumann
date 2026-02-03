/**
 * Vector service clients for points and collections operations.
 */
import type * as grpc from '@grpc/grpc-js';
import type { PointsServiceClient, CollectionsServiceClient } from '../generated/vector.js';
/**
 * A vector point with payload.
 */
export interface VectorPoint {
    /** Unique point identifier. */
    id: string;
    /** Dense vector embedding. */
    vector: number[];
    /** Optional payload data. */
    payload?: Record<string, unknown> | undefined;
}
/**
 * A point with similarity score from a query.
 */
export interface ScoredVectorPoint {
    /** Unique point identifier. */
    id: string;
    /** Similarity score. */
    score: number;
    /** Optional payload data. */
    payload?: Record<string, unknown> | undefined;
    /** Optional vector (if requested). */
    vector?: number[] | undefined;
}
/**
 * Options for upserting points.
 */
export interface UpsertOptions {
    /** Wait for indexing to complete. */
    wait?: boolean;
}
/**
 * Options for getting points.
 */
export interface GetPointsOptions {
    /** Include payload in response. */
    withPayload?: boolean;
    /** Include vector in response. */
    withVector?: boolean;
}
/**
 * Options for querying similar points.
 */
export interface QueryOptions {
    /** Maximum number of results. */
    limit?: number;
    /** Number of results to skip. */
    offset?: number;
    /** Minimum similarity score threshold. */
    scoreThreshold?: number;
    /** Include payload in response. */
    withPayload?: boolean;
    /** Include vector in response. */
    withVector?: boolean;
}
/**
 * Options for scrolling through points.
 */
export interface ScrollOptions {
    /** Maximum number of points to return. */
    limit?: number | undefined;
    /** Offset point ID for pagination. */
    offsetId?: string | undefined;
    /** Include payload in response. */
    withPayload?: boolean | undefined;
    /** Include vector in response. */
    withVector?: boolean | undefined;
}
/**
 * Scroll result with pagination info.
 */
export interface ScrollResult {
    /** Points in this page. */
    points: VectorPoint[];
    /** Next offset ID for pagination, undefined if no more pages. */
    nextOffset?: string | undefined;
}
/**
 * Collection information.
 */
export interface CollectionInfo {
    /** Collection name. */
    name: string;
    /** Number of points in collection. */
    pointsCount: number;
    /** Vector dimension. */
    dimension: number;
    /** Distance metric. */
    distance: string;
}
/**
 * Distance metric for vector similarity.
 */
export type DistanceMetric = 'cosine' | 'euclidean' | 'dot';
/**
 * Service client for vector points operations.
 */
export declare class PointsClient {
    private client;
    private metadata;
    constructor(client: PointsServiceClient, metadata: grpc.Metadata);
    /**
     * Upsert points into a collection.
     *
     * @param collection - Target collection name.
     * @param points - Points to upsert.
     * @returns Number of points upserted.
     */
    upsert(collection: string, points: VectorPoint[]): Promise<number>;
    /**
     * Get points by IDs.
     *
     * @param collection - Target collection name.
     * @param ids - Point IDs to retrieve.
     * @param options - Get options.
     * @returns Retrieved points.
     */
    get(collection: string, ids: string[], options?: GetPointsOptions): Promise<VectorPoint[]>;
    /**
     * Delete points by IDs.
     *
     * @param collection - Target collection name.
     * @param ids - Point IDs to delete.
     * @returns Number of points deleted.
     */
    delete(collection: string, ids: string[]): Promise<number>;
    /**
     * Query for similar points.
     *
     * @param collection - Target collection name.
     * @param vector - Query vector.
     * @param options - Query options.
     * @returns Similar points with scores.
     */
    query(collection: string, vector: number[], options?: QueryOptions): Promise<ScoredVectorPoint[]>;
    /**
     * Scroll through points in a collection.
     *
     * @param collection - Target collection name.
     * @param options - Scroll options.
     * @returns Scroll result with points and next offset.
     */
    scroll(collection: string, options?: ScrollOptions): Promise<ScrollResult>;
    /**
     * Iterate through all points in a collection.
     *
     * @param collection - Target collection name.
     * @param options - Scroll options (limit per page).
     * @returns Async iterable of points.
     */
    scrollAll(collection: string, options?: Omit<ScrollOptions, 'offsetId'>): AsyncIterable<VectorPoint>;
    private convertPoint;
    private convertScoredPoint;
    private encodePayload;
    private decodePayload;
    private handleError;
}
/**
 * Service client for vector collections operations.
 */
export declare class CollectionsClient {
    private client;
    private metadata;
    constructor(client: CollectionsServiceClient, metadata: grpc.Metadata);
    /**
     * Create a new collection.
     *
     * @param name - Collection name.
     * @param dimension - Vector dimension.
     * @param distance - Distance metric.
     * @returns True if collection was created.
     */
    create(name: string, dimension: number, distance?: DistanceMetric): Promise<boolean>;
    /**
     * Get collection information.
     *
     * @param name - Collection name.
     * @returns Collection information.
     */
    get(name: string): Promise<CollectionInfo>;
    /**
     * Delete a collection.
     *
     * @param name - Collection name.
     * @returns True if collection was deleted.
     */
    delete(name: string): Promise<boolean>;
    /**
     * List all collections.
     *
     * @returns Array of collection names.
     */
    list(): Promise<string[]>;
    /**
     * Check if a collection exists.
     *
     * @param name - Collection name.
     * @returns True if collection exists.
     */
    exists(name: string): Promise<boolean>;
    private handleError;
}
//# sourceMappingURL=vector.d.ts.map