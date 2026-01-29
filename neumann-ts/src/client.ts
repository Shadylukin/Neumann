// SPDX-License-Identifier: MIT
import type {
  QueryResult,
  Row,
  Node,
  Edge,
  Path,
  PathSegment,
  SimilarItem,
  ArtifactInfo,
} from './types/query-result.js';
import type { Value } from './types/value.js';
import {
  ConnectionError,
  AuthenticationError,
  PermissionDeniedError,
  NotFoundError,
  InvalidArgumentError,
  InternalError,
  errorFromCode,
} from './types/errors.js';
import type { NeumannError } from './types/errors.js';
import {
  nullValue,
  intValue,
  floatValue,
  stringValue,
  boolValue,
  bytesValue,
} from './types/value.js';
import type * as grpc from '@grpc/grpc-js';
import type { QueryServiceClient } from './grpc.js';

/**
 * Options for connecting to a Neumann server.
 */
export interface ConnectOptions {
  /** API key for authentication. */
  apiKey?: string;
  /** Whether to use TLS encryption. */
  tls?: boolean;
  /** Custom metadata headers. */
  metadata?: Record<string, string>;
}

/**
 * Options for query execution.
 */
export interface QueryOptions {
  /** Identity for vault access. */
  identity?: string;
}

/**
 * Client mode.
 */
export type ClientMode = 'remote' | 'embedded';

/**
 * Client for Neumann database supporting both embedded and remote modes.
 */
export class NeumannClient {
  private mode: ClientMode;
  private connected = false;
  private client: unknown = null;
  private apiKey: string | undefined;
  private address: string | undefined;
  private grpcClient: QueryServiceClient | null = null;
  private grpcMetadata: grpc.Metadata | null = null;

  private constructor(mode: ClientMode) {
    this.mode = mode;
  }

  /**
   * Connect to a remote Neumann server via gRPC.
   *
   * @param address - Server address in format "host:port".
   * @param options - Connection options.
   * @returns A connected NeumannClient.
   */
  static async connect(address: string, options: ConnectOptions = {}): Promise<NeumannClient> {
    const client = new NeumannClient('remote');
    client.apiKey = options.apiKey;
    client.address = address;

    try {
      const grpc = await import('@grpc/grpc-js');
      const { loadProto, getQueryServiceClient } = await import('./grpc.js');

      const proto = await loadProto();
      const credentials = options.tls
        ? grpc.credentials.createSsl()
        : grpc.credentials.createInsecure();

      client.grpcClient = getQueryServiceClient(
        proto,
        address,
        credentials
      ) as QueryServiceClient;

      // Setup metadata for authentication
      client.grpcMetadata = new grpc.Metadata();
      if (options.apiKey) {
        client.grpcMetadata.set('x-api-key', options.apiKey);
      }
      if (options.metadata) {
        for (const [key, value] of Object.entries(options.metadata)) {
          client.grpcMetadata.set(key, value);
        }
      }

      client.connected = true;
    } catch (err) {
      throw new ConnectionError(`Failed to connect to ${address}: ${String(err)}`);
    }

    return client;
  }

  /**
   * Connect to a remote Neumann server via gRPC-Web (for browsers).
   *
   * @param address - Server address as a URL.
   * @param options - Connection options.
   * @returns A connected NeumannClient.
   */
  static async connectWeb(address: string, options: ConnectOptions = {}): Promise<NeumannClient> {
    const client = new NeumannClient('remote');
    client.apiKey = options.apiKey;
    client.address = address;

    try {
      // Dynamic import for gRPC-Web (browser environment)
      const grpcWeb = await import('grpc-web');
      client.client = new grpcWeb.GrpcWebClientBase({ format: 'binary' });
      client.connected = true;
    } catch (err) {
      throw new ConnectionError(`Failed to connect via gRPC-Web: ${String(err)}`);
    }

    return client;
  }

  /**
   * Check if client is connected.
   */
  get isConnected(): boolean {
    return this.connected;
  }

  /**
   * Get the client mode.
   */
  get clientMode(): ClientMode {
    return this.mode;
  }

  /**
   * Close the client connection.
   */
  close(): void {
    this.client = null;
    this.connected = false;
  }

  /**
   * Execute a query and return the result.
   *
   * @param query - The Neumann query to execute.
   * @param options - Query options.
   * @returns Query result.
   */
  async execute(query: string, options: QueryOptions = {}): Promise<QueryResult> {
    if (!this.connected || !this.grpcClient) {
      throw new ConnectionError('Client is not connected');
    }

    const request = {
      query,
      identity: options.identity ?? '',
    };

    return new Promise((resolve, reject) => {
      this.grpcClient!.Execute(
        request,
        this.grpcMetadata!,
        (err: grpc.ServiceError | null, response: unknown) => {
          if (err) {
            reject(this.handleGrpcError(err));
            return;
          }
          try {
            resolve(this.convertProtoResponse(response));
          } catch (e) {
            reject(e);
          }
        }
      );
    });
  }

  /**
   * Execute a streaming query.
   *
   * @param query - The Neumann query to execute.
   * @param options - Query options.
   * @returns Async iterator of query results.
   */
  async *executeStream(query: string, options: QueryOptions = {}): AsyncIterable<QueryResult> {
    if (!this.connected || !this.grpcClient) {
      throw new ConnectionError('Client is not connected');
    }

    const request = {
      query,
      identity: options.identity ?? '',
    };

    const stream = this.grpcClient.ExecuteStream(request, this.grpcMetadata!);

    for await (const chunk of stream as AsyncIterable<unknown>) {
      const c = chunk as {
        isFinal?: boolean;
        error?: { code: number; message: string };
        row?: unknown;
        node?: unknown;
        edge?: unknown;
        similarItem?: unknown;
        blobData?: Uint8Array;
      };

      if (c.isFinal) {
        break;
      }
      if (c.error) {
        throw errorFromCode(c.error.code, c.error.message);
      }
      yield this.convertProtoChunk(chunk);
    }
  }

  /**
   * Execute multiple queries in a batch.
   *
   * @param queries - List of queries to execute.
   * @param options - Query options.
   * @returns List of query results.
   */
  async executeBatch(queries: string[], options: QueryOptions = {}): Promise<QueryResult[]> {
    if (!this.connected || !this.grpcClient) {
      throw new ConnectionError('Client is not connected');
    }

    const request = {
      queries: queries.map((q) => ({
        query: q,
        identity: options.identity ?? '',
      })),
    };

    return new Promise((resolve, reject) => {
      this.grpcClient!.ExecuteBatch(
        request,
        this.grpcMetadata!,
        (err: grpc.ServiceError | null, response: unknown) => {
          if (err) {
            reject(this.handleGrpcError(err));
            return;
          }
          try {
            const r = response as { results?: unknown[] };
            const results = (r.results ?? []).map((res) => this.convertProtoResponse(res));
            resolve(results);
          } catch (e) {
            reject(e);
          }
        }
      );
    });
  }

  /**
   * Convert a proto QueryResponse to a QueryResult.
   */
  private convertProtoResponse(response: unknown): QueryResult {
    const r = response as {
      empty?: object;
      count?: { count: number };
      rows?: { rows: unknown[] };
      nodes?: { nodes: unknown[] };
      edges?: { edges: unknown[] };
      path?: { nodeIds: number[] };
      similar?: { items: unknown[] };
      ids?: { ids: string[] | number[] };
      tableList?: { tables: string[] };
      blob?: { data: Uint8Array };
      artifactInfo?: unknown;
      error?: { code: number; message: string };
    };

    if (r.error) {
      return { type: 'error', code: r.error.code, message: r.error.message };
    }
    if (r.empty !== undefined) {
      return { type: 'empty' };
    }
    if (r.count !== undefined) {
      return { type: 'count', count: r.count.count };
    }
    if (r.rows !== undefined) {
      return { type: 'rows', rows: r.rows.rows.map((row) => convertProtoRow(row)) };
    }
    if (r.nodes !== undefined) {
      return { type: 'nodes', nodes: r.nodes.nodes.map((node) => convertProtoNode(node)) };
    }
    if (r.edges !== undefined) {
      return { type: 'edges', edges: r.edges.edges.map((edge) => convertProtoEdge(edge)) };
    }
    if (r.path !== undefined) {
      return { type: 'paths', paths: [convertProtoPath(r.path)] };
    }
    if (r.similar !== undefined) {
      return { type: 'similar', items: r.similar.items.map((item) => convertProtoSimilarItem(item)) };
    }
    if (r.ids !== undefined) {
      return { type: 'ids', ids: r.ids.ids.map(String) };
    }
    if (r.tableList !== undefined) {
      return { type: 'tableList', names: r.tableList.tables };
    }
    if (r.blob !== undefined) {
      return { type: 'blob', data: r.blob.data };
    }
    if (r.artifactInfo !== undefined) {
      return { type: 'blobInfo', info: convertProtoArtifactInfo(r.artifactInfo) };
    }

    return { type: 'empty' };
  }

  /**
   * Convert a proto QueryResponseChunk to a QueryResult.
   */
  private convertProtoChunk(chunk: unknown): QueryResult {
    const c = chunk as {
      row?: { row: unknown };
      node?: { node: unknown };
      edge?: { edge: unknown };
      similarItem?: { item: unknown };
      blobData?: Uint8Array;
    };

    if (c.row?.row) {
      return { type: 'rows', rows: [convertProtoRow(c.row.row)] };
    }
    if (c.node?.node) {
      return { type: 'nodes', nodes: [convertProtoNode(c.node.node)] };
    }
    if (c.edge?.edge) {
      return { type: 'edges', edges: [convertProtoEdge(c.edge.edge)] };
    }
    if (c.similarItem?.item) {
      return { type: 'similar', items: [convertProtoSimilarItem(c.similarItem.item)] };
    }
    if (c.blobData) {
      return { type: 'blob', data: c.blobData };
    }

    return { type: 'empty' };
  }

  /**
   * Convert a gRPC error to a NeumannError.
   */
  private handleGrpcError(err: grpc.ServiceError): NeumannError {
    // gRPC status codes
    const code = err.code as number;
    const UNAUTHENTICATED = 16;
    const PERMISSION_DENIED = 7;
    const NOT_FOUND = 5;
    const INVALID_ARGUMENT = 3;
    const UNAVAILABLE = 14;

    if (code === UNAUTHENTICATED) {
      return new AuthenticationError(err.details || 'Authentication failed');
    }
    if (code === PERMISSION_DENIED) {
      return new PermissionDeniedError(err.details || 'Permission denied');
    }
    if (code === NOT_FOUND) {
      return new NotFoundError(err.details || 'Not found');
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
 * Convert a proto value to a Value.
 */
export function convertProtoValue(protoValue: unknown): Value {
  if (protoValue === null || protoValue === undefined) {
    return nullValue();
  }

  const v = protoValue as Record<string, unknown>;

  if ('nullValue' in v) {
    return nullValue();
  }
  if ('intValue' in v && typeof v.intValue === 'number') {
    return intValue(v.intValue);
  }
  if ('floatValue' in v && typeof v.floatValue === 'number') {
    return floatValue(v.floatValue);
  }
  if ('stringValue' in v && typeof v.stringValue === 'string') {
    return stringValue(v.stringValue);
  }
  if ('boolValue' in v && typeof v.boolValue === 'boolean') {
    return boolValue(v.boolValue);
  }
  if ('bytesValue' in v && v.bytesValue instanceof Uint8Array) {
    return bytesValue(v.bytesValue);
  }

  return nullValue();
}

/**
 * Convert a proto row to a Row.
 */
export function convertProtoRow(protoRow: unknown): Row {
  const values = new Map<string, Value>();
  const row = protoRow as { columns?: Array<{ name: string; value: unknown }> };

  if (row.columns) {
    for (const col of row.columns) {
      values.set(col.name, convertProtoValue(col.value));
    }
  }

  return { values };
}

/**
 * Convert a proto node to a Node.
 */
export function convertProtoNode(protoNode: unknown): Node {
  const properties = new Map<string, Value>();
  const node = protoNode as {
    id: string;
    label: string;
    properties?: Array<{ name: string; value: unknown }>;
  };

  if (node.properties) {
    for (const prop of node.properties) {
      properties.set(prop.name, convertProtoValue(prop.value));
    }
  }

  return {
    id: node.id,
    label: node.label,
    properties,
  };
}

/**
 * Convert a proto edge to an Edge.
 */
export function convertProtoEdge(protoEdge: unknown): Edge {
  const properties = new Map<string, Value>();
  const edge = protoEdge as {
    id: string;
    edgeType: string;
    sourceId: string;
    targetId: string;
    properties?: Array<{ name: string; value: unknown }>;
  };

  if (edge.properties) {
    for (const prop of edge.properties) {
      properties.set(prop.name, convertProtoValue(prop.value));
    }
  }

  return {
    id: edge.id,
    edgeType: edge.edgeType,
    source: edge.sourceId,
    target: edge.targetId,
    properties,
  };
}

/**
 * Convert a proto path to a Path.
 */
export function convertProtoPath(protoPath: unknown): Path {
  const segments: PathSegment[] = [];
  const path = protoPath as {
    segments?: Array<{ node: unknown; edge?: unknown }>;
  };

  if (path.segments) {
    for (const seg of path.segments) {
      const segment: PathSegment = {
        node: convertProtoNode(seg.node),
      };
      if (seg.edge) {
        segment.edge = convertProtoEdge(seg.edge);
      }
      segments.push(segment);
    }
  }

  return { segments };
}

/**
 * Convert a proto similar item to a SimilarItem.
 */
export function convertProtoSimilarItem(protoItem: unknown): SimilarItem {
  const item = protoItem as {
    key: string;
    score: number;
    metadata?: Array<{ name: string; value: unknown }>;
  };

  const result: SimilarItem = {
    key: item.key,
    score: item.score,
  };

  if (item.metadata && item.metadata.length > 0) {
    const metadata = new Map<string, Value>();
    for (const prop of item.metadata) {
      metadata.set(prop.name, convertProtoValue(prop.value));
    }
    result.metadata = metadata;
  }

  return result;
}

/**
 * Convert a proto artifact info to an ArtifactInfo.
 */
export function convertProtoArtifactInfo(protoInfo: unknown): ArtifactInfo {
  const info = protoInfo as {
    artifactId: string;
    filename: string;
    size: number;
    checksum: string;
    contentType: string;
    createdAt: number;
    tags?: string[];
  };

  return {
    artifactId: info.artifactId,
    filename: info.filename,
    size: info.size,
    checksum: info.checksum,
    contentType: info.contentType,
    createdAt: info.createdAt,
    tags: info.tags ?? [],
  };
}
