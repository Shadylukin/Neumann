// SPDX-License-Identifier: MIT
/**
 * Vector service clients for points and collections operations.
 */
import type * as grpc from '@grpc/grpc-js';
import type {
  PointsServiceClient,
  CollectionsServiceClient,
  Point as ProtoPoint,
  ScoredPoint as ProtoScoredPoint,
  UpsertPointsResponse,
  GetPointsResponse,
  DeletePointsResponse,
  QueryPointsResponse,
  ScrollPointsResponse,
  CreateCollectionResponse,
  GetCollectionResponse,
  DeleteCollectionResponse,
  ListCollectionsResponse,
} from '../generated/vector.js';
import {
  ConnectionError,
  NotFoundError,
  InternalError,
  InvalidArgumentError,
} from '../types/errors.js';
import type { NeumannError } from '../types/errors.js';

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
export class PointsClient {
  private client: PointsServiceClient;
  private metadata: grpc.Metadata;

  constructor(client: PointsServiceClient, metadata: grpc.Metadata) {
    this.client = client;
    this.metadata = metadata;
  }

  /**
   * Upsert points into a collection.
   *
   * @param collection - Target collection name.
   * @param points - Points to upsert.
   * @returns Number of points upserted.
   */
  async upsert(collection: string, points: VectorPoint[]): Promise<number> {
    return new Promise((resolve, reject) => {
      const protoPoints: ProtoPoint[] = points.map((p) => ({
        id: p.id,
        vector: p.vector,
        payload: p.payload ? this.encodePayload(p.payload) : undefined,
      }));

      this.client.Upsert(
        { collection, points: protoPoints },
        this.metadata,
        (err: grpc.ServiceError | null, response: UpsertPointsResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.upserted);
        }
      );
    });
  }

  /**
   * Get points by IDs.
   *
   * @param collection - Target collection name.
   * @param ids - Point IDs to retrieve.
   * @param options - Get options.
   * @returns Retrieved points.
   */
  async get(
    collection: string,
    ids: string[],
    options: GetPointsOptions = {}
  ): Promise<VectorPoint[]> {
    return new Promise((resolve, reject) => {
      this.client.Get(
        {
          collection,
          ids,
          withPayload: options.withPayload ?? true,
          withVector: options.withVector ?? false,
        },
        this.metadata,
        (err: grpc.ServiceError | null, response: GetPointsResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.points.map((p) => this.convertPoint(p)));
        }
      );
    });
  }

  /**
   * Delete points by IDs.
   *
   * @param collection - Target collection name.
   * @param ids - Point IDs to delete.
   * @returns Number of points deleted.
   */
  async delete(collection: string, ids: string[]): Promise<number> {
    return new Promise((resolve, reject) => {
      this.client.Delete(
        { collection, ids },
        this.metadata,
        (err: grpc.ServiceError | null, response: DeletePointsResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.deleted);
        }
      );
    });
  }

  /**
   * Query for similar points.
   *
   * @param collection - Target collection name.
   * @param vector - Query vector.
   * @param options - Query options.
   * @returns Similar points with scores.
   */
  async query(
    collection: string,
    vector: number[],
    options: QueryOptions = {}
  ): Promise<ScoredVectorPoint[]> {
    return new Promise((resolve, reject) => {
      this.client.Query(
        {
          collection,
          vector,
          limit: options.limit ?? 10,
          offset: options.offset ?? 0,
          scoreThreshold: options.scoreThreshold,
          withPayload: options.withPayload ?? true,
          withVector: options.withVector ?? false,
        },
        this.metadata,
        (err: grpc.ServiceError | null, response: QueryPointsResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.results.map((p) => this.convertScoredPoint(p)));
        }
      );
    });
  }

  /**
   * Scroll through points in a collection.
   *
   * @param collection - Target collection name.
   * @param options - Scroll options.
   * @returns Scroll result with points and next offset.
   */
  async scroll(
    collection: string,
    options: ScrollOptions = {}
  ): Promise<ScrollResult> {
    return new Promise((resolve, reject) => {
      this.client.Scroll(
        {
          collection,
          offsetId: options.offsetId,
          limit: options.limit ?? 100,
          withPayload: options.withPayload ?? true,
          withVector: options.withVector ?? false,
        },
        this.metadata,
        (err: grpc.ServiceError | null, response: ScrollPointsResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve({
            points: response.points.map((p) => this.convertPoint(p)),
            nextOffset: response.nextOffset,
          });
        }
      );
    });
  }

  /**
   * Iterate through all points in a collection.
   *
   * @param collection - Target collection name.
   * @param options - Scroll options (limit per page).
   * @returns Async iterable of points.
   */
  async *scrollAll(
    collection: string,
    options: Omit<ScrollOptions, 'offsetId'> = {}
  ): AsyncIterable<VectorPoint> {
    let offsetId: string | undefined;

    while (true) {
      const result = await this.scroll(collection, { ...options, offsetId });
      for (const point of result.points) {
        yield point;
      }
      if (!result.nextOffset) {
        break;
      }
      offsetId = result.nextOffset;
    }
  }

  private convertPoint(proto: ProtoPoint): VectorPoint {
    return {
      id: proto.id,
      vector: proto.vector,
      payload: proto.payload ? this.decodePayload(proto.payload) : undefined,
    };
  }

  private convertScoredPoint(proto: ProtoScoredPoint): ScoredVectorPoint {
    return {
      id: proto.id,
      score: proto.score,
      payload: proto.payload ? this.decodePayload(proto.payload) : undefined,
      vector: proto.vector && proto.vector.length > 0 ? proto.vector : undefined,
    };
  }

  private encodePayload(
    payload: Record<string, unknown>
  ): Record<string, Uint8Array> {
    const result: Record<string, Uint8Array> = {};
    for (const [key, value] of Object.entries(payload)) {
      result[key] = new TextEncoder().encode(JSON.stringify(value));
    }
    return result;
  }

  private decodePayload(
    payload: Record<string, Uint8Array>
  ): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(payload)) {
      try {
        result[key] = JSON.parse(new TextDecoder().decode(value));
      } catch {
        result[key] = new TextDecoder().decode(value);
      }
    }
    return result;
  }

  private handleError(err: grpc.ServiceError): NeumannError {
    const code = err.code as number;
    const NOT_FOUND = 5;
    const INVALID_ARGUMENT = 3;
    const UNAVAILABLE = 14;

    if (code === NOT_FOUND) {
      return new NotFoundError(err.details || 'Collection not found');
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

/**
 * Service client for vector collections operations.
 */
export class CollectionsClient {
  private client: CollectionsServiceClient;
  private metadata: grpc.Metadata;

  constructor(client: CollectionsServiceClient, metadata: grpc.Metadata) {
    this.client = client;
    this.metadata = metadata;
  }

  /**
   * Create a new collection.
   *
   * @param name - Collection name.
   * @param dimension - Vector dimension.
   * @param distance - Distance metric.
   * @returns True if collection was created.
   */
  async create(
    name: string,
    dimension: number,
    distance: DistanceMetric = 'cosine'
  ): Promise<boolean> {
    return new Promise((resolve, reject) => {
      this.client.Create(
        { name, dimension, distance },
        this.metadata,
        (err: grpc.ServiceError | null, response: CreateCollectionResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.created);
        }
      );
    });
  }

  /**
   * Get collection information.
   *
   * @param name - Collection name.
   * @returns Collection information.
   */
  async get(name: string): Promise<CollectionInfo> {
    return new Promise((resolve, reject) => {
      this.client.Get(
        { name },
        this.metadata,
        (err: grpc.ServiceError | null, response: GetCollectionResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve({
            name: response.name,
            pointsCount: response.pointsCount,
            dimension: response.dimension,
            distance: response.distance,
          });
        }
      );
    });
  }

  /**
   * Delete a collection.
   *
   * @param name - Collection name.
   * @returns True if collection was deleted.
   */
  async delete(name: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      this.client.Delete(
        { name },
        this.metadata,
        (err: grpc.ServiceError | null, response: DeleteCollectionResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.deleted);
        }
      );
    });
  }

  /**
   * List all collections.
   *
   * @returns Array of collection names.
   */
  async list(): Promise<string[]> {
    return new Promise((resolve, reject) => {
      this.client.List(
        {},
        this.metadata,
        (err: grpc.ServiceError | null, response: ListCollectionsResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          resolve(response.collections);
        }
      );
    });
  }

  /**
   * Check if a collection exists.
   *
   * @param name - Collection name.
   * @returns True if collection exists.
   */
  async exists(name: string): Promise<boolean> {
    try {
      await this.get(name);
      return true;
    } catch (err) {
      if (err instanceof NotFoundError) {
        return false;
      }
      throw err;
    }
  }

  private handleError(err: grpc.ServiceError): NeumannError {
    const code = err.code as number;
    const NOT_FOUND = 5;
    const INVALID_ARGUMENT = 3;
    const ALREADY_EXISTS = 6;
    const UNAVAILABLE = 14;

    if (code === NOT_FOUND) {
      return new NotFoundError(err.details || 'Collection not found');
    }
    if (code === INVALID_ARGUMENT) {
      return new InvalidArgumentError(err.details || 'Invalid argument');
    }
    if (code === ALREADY_EXISTS) {
      return new InvalidArgumentError(err.details || 'Collection already exists');
    }
    if (code === UNAVAILABLE) {
      return new ConnectionError(err.details || 'Service unavailable');
    }
    return new InternalError(err.details || err.message || 'Internal error');
  }
}
