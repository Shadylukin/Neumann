import { PointsClient, CollectionsClient, type VectorPoint, type ScoredVectorPoint, type QueryOptions, type ScrollOptions, type ScrollResult, type CollectionInfo, type DistanceMetric, type GetPointsOptions } from './services/vector.js';
/**
 * Options for connecting to vector services.
 */
export interface VectorConnectOptions {
    /** API key for authentication. */
    apiKey?: string;
    /** Whether to use TLS encryption. */
    tls?: boolean;
    /** Custom metadata headers. */
    metadata?: Record<string, string>;
}
/**
 * Unified client for vector database operations.
 *
 * Provides a simplified interface for working with vector collections
 * and points, combining both Points and Collections services.
 *
 * @example
 * ```typescript
 * const vectors = await VectorClient.connect('localhost:9200');
 *
 * // Create a collection
 * await vectors.createCollection('documents', 384, 'cosine');
 *
 * // Upsert points
 * await vectors.upsertPoints('documents', [
 *   { id: 'doc1', vector: [...], payload: { title: 'Hello' } },
 * ]);
 *
 * // Query similar vectors
 * const results = await vectors.queryPoints('documents', queryVector, { limit: 10 });
 *
 * vectors.close();
 * ```
 */
export declare class VectorClient {
    private pointsClient;
    private collectionsClient;
    private grpcPointsClient;
    private grpcCollectionsClient;
    private connected;
    private constructor();
    /**
     * Connect to vector services via gRPC.
     *
     * @param address - Server address in format "host:port".
     * @param options - Connection options.
     * @returns A connected VectorClient.
     */
    static connect(address: string, options?: VectorConnectOptions): Promise<VectorClient>;
    /**
     * Check if client is connected.
     */
    get isConnected(): boolean;
    /**
     * Close the client connection.
     */
    close(): void;
    /**
     * Create a new collection.
     *
     * @param name - Collection name.
     * @param dimension - Vector dimension.
     * @param distance - Distance metric (default: 'cosine').
     * @returns True if collection was created.
     */
    createCollection(name: string, dimension: number, distance?: DistanceMetric): Promise<boolean>;
    /**
     * Get collection information.
     *
     * @param name - Collection name.
     * @returns Collection information.
     */
    getCollection(name: string): Promise<CollectionInfo>;
    /**
     * Delete a collection.
     *
     * @param name - Collection name.
     * @returns True if collection was deleted.
     */
    deleteCollection(name: string): Promise<boolean>;
    /**
     * List all collections.
     *
     * @returns Array of collection names.
     */
    listCollections(): Promise<string[]>;
    /**
     * Check if a collection exists.
     *
     * @param name - Collection name.
     * @returns True if collection exists.
     */
    collectionExists(name: string): Promise<boolean>;
    /**
     * Upsert points into a collection.
     *
     * @param collection - Target collection name.
     * @param points - Points to upsert.
     * @returns Number of points upserted.
     */
    upsertPoints(collection: string, points: VectorPoint[]): Promise<number>;
    /**
     * Get points by IDs.
     *
     * @param collection - Target collection name.
     * @param ids - Point IDs to retrieve.
     * @param options - Get options.
     * @returns Retrieved points.
     */
    getPoints(collection: string, ids: string[], options?: GetPointsOptions): Promise<VectorPoint[]>;
    /**
     * Delete points by IDs.
     *
     * @param collection - Target collection name.
     * @param ids - Point IDs to delete.
     * @returns Number of points deleted.
     */
    deletePoints(collection: string, ids: string[]): Promise<number>;
    /**
     * Query for similar points.
     *
     * @param collection - Target collection name.
     * @param vector - Query vector.
     * @param options - Query options.
     * @returns Similar points with scores.
     */
    queryPoints(collection: string, vector: number[], options?: QueryOptions): Promise<ScoredVectorPoint[]>;
    /**
     * Scroll through points in a collection.
     *
     * @param collection - Target collection name.
     * @param options - Scroll options.
     * @returns Scroll result with points and next offset.
     */
    scrollPoints(collection: string, options?: ScrollOptions): Promise<ScrollResult>;
    /**
     * Iterate through all points in a collection.
     *
     * @param collection - Target collection name.
     * @param options - Scroll options (limit per page).
     * @returns Async iterable of points.
     */
    scrollAllPoints(collection: string, options?: Omit<ScrollOptions, 'offsetId'>): AsyncIterable<VectorPoint>;
    /**
     * Get the number of points in a collection.
     *
     * @param collection - Collection name.
     * @returns Number of points.
     */
    countPoints(collection: string): Promise<number>;
    /**
     * Access the underlying PointsClient.
     */
    get points(): PointsClient;
    /**
     * Access the underlying CollectionsClient.
     */
    get collections(): CollectionsClient;
    private ensureConnected;
}
//# sourceMappingURL=vector-client.d.ts.map