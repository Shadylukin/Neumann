// SPDX-License-Identifier: MIT
/**
 * VectorClient provides a unified interface for vector operations,
 * combining Points and Collections services.
 */
import type * as grpc from '@grpc/grpc-js';
import { ConnectionError } from './types/errors.js';
import {
  PointsClient,
  CollectionsClient,
  type VectorPoint,
  type ScoredVectorPoint,
  type QueryOptions,
  type ScrollOptions,
  type ScrollResult,
  type CollectionInfo,
  type DistanceMetric,
  type GetPointsOptions,
} from './services/vector.js';
import type {
  PointsServiceClient,
  CollectionsServiceClient,
} from './generated/vector.js';

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
export class VectorClient {
  private pointsClient: PointsClient;
  private collectionsClient: CollectionsClient;
  private grpcPointsClient: grpc.Client;
  private grpcCollectionsClient: grpc.Client;
  private connected = true;

  private constructor(
    pointsClient: PointsClient,
    collectionsClient: CollectionsClient,
    grpcPointsClient: grpc.Client,
    grpcCollectionsClient: grpc.Client
  ) {
    this.pointsClient = pointsClient;
    this.collectionsClient = collectionsClient;
    this.grpcPointsClient = grpcPointsClient;
    this.grpcCollectionsClient = grpcCollectionsClient;
  }

  /**
   * Connect to vector services via gRPC.
   *
   * @param address - Server address in format "host:port".
   * @param options - Connection options.
   * @returns A connected VectorClient.
   */
  static async connect(
    address: string,
    options: VectorConnectOptions = {}
  ): Promise<VectorClient> {
    try {
      const grpc = await import('@grpc/grpc-js');
      const {
        loadVectorProto,
        getPointsServiceClient,
        getCollectionsServiceClient,
      } = await import('./grpc.js');

      const proto = await loadVectorProto();
      const credentials = options.tls
        ? grpc.credentials.createSsl()
        : grpc.credentials.createInsecure();

      const grpcPointsClient = getPointsServiceClient(
        proto,
        address,
        credentials
      ) as unknown as PointsServiceClient & grpc.Client;

      const grpcCollectionsClient = getCollectionsServiceClient(
        proto,
        address,
        credentials
      ) as unknown as CollectionsServiceClient & grpc.Client;

      // Setup metadata
      const metadata = new grpc.Metadata();
      if (options.apiKey) {
        metadata.set('x-api-key', options.apiKey);
      }
      if (options.metadata) {
        for (const [key, value] of Object.entries(options.metadata)) {
          metadata.set(key, value);
        }
      }

      const pointsClient = new PointsClient(
        grpcPointsClient as PointsServiceClient,
        metadata
      );
      const collectionsClient = new CollectionsClient(
        grpcCollectionsClient as CollectionsServiceClient,
        metadata
      );

      return new VectorClient(
        pointsClient,
        collectionsClient,
        grpcPointsClient,
        grpcCollectionsClient
      );
    } catch (err) {
      throw new ConnectionError(`Failed to connect to ${address}: ${String(err)}`);
    }
  }

  /**
   * Check if client is connected.
   */
  get isConnected(): boolean {
    return this.connected;
  }

  /**
   * Close the client connection.
   */
  close(): void {
    this.grpcPointsClient.close();
    this.grpcCollectionsClient.close();
    this.connected = false;
  }

  // === Collection Operations ===

  /**
   * Create a new collection.
   *
   * @param name - Collection name.
   * @param dimension - Vector dimension.
   * @param distance - Distance metric (default: 'cosine').
   * @returns True if collection was created.
   */
  async createCollection(
    name: string,
    dimension: number,
    distance: DistanceMetric = 'cosine'
  ): Promise<boolean> {
    this.ensureConnected();
    return this.collectionsClient.create(name, dimension, distance);
  }

  /**
   * Get collection information.
   *
   * @param name - Collection name.
   * @returns Collection information.
   */
  async getCollection(name: string): Promise<CollectionInfo> {
    this.ensureConnected();
    return this.collectionsClient.get(name);
  }

  /**
   * Delete a collection.
   *
   * @param name - Collection name.
   * @returns True if collection was deleted.
   */
  async deleteCollection(name: string): Promise<boolean> {
    this.ensureConnected();
    return this.collectionsClient.delete(name);
  }

  /**
   * List all collections.
   *
   * @returns Array of collection names.
   */
  async listCollections(): Promise<string[]> {
    this.ensureConnected();
    return this.collectionsClient.list();
  }

  /**
   * Check if a collection exists.
   *
   * @param name - Collection name.
   * @returns True if collection exists.
   */
  async collectionExists(name: string): Promise<boolean> {
    this.ensureConnected();
    return this.collectionsClient.exists(name);
  }

  // === Points Operations ===

  /**
   * Upsert points into a collection.
   *
   * @param collection - Target collection name.
   * @param points - Points to upsert.
   * @returns Number of points upserted.
   */
  async upsertPoints(
    collection: string,
    points: VectorPoint[]
  ): Promise<number> {
    this.ensureConnected();
    return this.pointsClient.upsert(collection, points);
  }

  /**
   * Get points by IDs.
   *
   * @param collection - Target collection name.
   * @param ids - Point IDs to retrieve.
   * @param options - Get options.
   * @returns Retrieved points.
   */
  async getPoints(
    collection: string,
    ids: string[],
    options?: GetPointsOptions
  ): Promise<VectorPoint[]> {
    this.ensureConnected();
    return this.pointsClient.get(collection, ids, options);
  }

  /**
   * Delete points by IDs.
   *
   * @param collection - Target collection name.
   * @param ids - Point IDs to delete.
   * @returns Number of points deleted.
   */
  async deletePoints(collection: string, ids: string[]): Promise<number> {
    this.ensureConnected();
    return this.pointsClient.delete(collection, ids);
  }

  /**
   * Query for similar points.
   *
   * @param collection - Target collection name.
   * @param vector - Query vector.
   * @param options - Query options.
   * @returns Similar points with scores.
   */
  async queryPoints(
    collection: string,
    vector: number[],
    options?: QueryOptions
  ): Promise<ScoredVectorPoint[]> {
    this.ensureConnected();
    return this.pointsClient.query(collection, vector, options);
  }

  /**
   * Scroll through points in a collection.
   *
   * @param collection - Target collection name.
   * @param options - Scroll options.
   * @returns Scroll result with points and next offset.
   */
  async scrollPoints(
    collection: string,
    options?: ScrollOptions
  ): Promise<ScrollResult> {
    this.ensureConnected();
    return this.pointsClient.scroll(collection, options);
  }

  /**
   * Iterate through all points in a collection.
   *
   * @param collection - Target collection name.
   * @param options - Scroll options (limit per page).
   * @returns Async iterable of points.
   */
  async *scrollAllPoints(
    collection: string,
    options?: Omit<ScrollOptions, 'offsetId'>
  ): AsyncIterable<VectorPoint> {
    this.ensureConnected();
    yield* this.pointsClient.scrollAll(collection, options);
  }

  // === Convenience Methods ===

  /**
   * Get the number of points in a collection.
   *
   * @param collection - Collection name.
   * @returns Number of points.
   */
  async countPoints(collection: string): Promise<number> {
    const info = await this.getCollection(collection);
    return info.pointsCount;
  }

  /**
   * Access the underlying PointsClient.
   */
  get points(): PointsClient {
    return this.pointsClient;
  }

  /**
   * Access the underlying CollectionsClient.
   */
  get collections(): CollectionsClient {
    return this.collectionsClient;
  }

  private ensureConnected(): void {
    if (!this.connected) {
      throw new ConnectionError('Client is not connected');
    }
  }
}
