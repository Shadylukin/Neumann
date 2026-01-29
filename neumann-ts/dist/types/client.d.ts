import type { QueryResult, Row, Node, Edge, Path, SimilarItem, ArtifactInfo } from './types/query-result.js';
import type { Value } from './types/value.js';
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
export declare class NeumannClient {
    private mode;
    private connected;
    private client;
    private apiKey;
    private address;
    private constructor();
    /**
     * Connect to a remote Neumann server via gRPC.
     *
     * @param address - Server address in format "host:port".
     * @param options - Connection options.
     * @returns A connected NeumannClient.
     */
    static connect(address: string, options?: ConnectOptions): Promise<NeumannClient>;
    /**
     * Connect to a remote Neumann server via gRPC-Web (for browsers).
     *
     * @param address - Server address as a URL.
     * @param options - Connection options.
     * @returns A connected NeumannClient.
     */
    static connectWeb(address: string, options?: ConnectOptions): Promise<NeumannClient>;
    /**
     * Check if client is connected.
     */
    get isConnected(): boolean;
    /**
     * Get the client mode.
     */
    get clientMode(): ClientMode;
    /**
     * Close the client connection.
     */
    close(): void;
    /**
     * Execute a query and return the result.
     *
     * @param query - The Neumann query to execute.
     * @param options - Query options.
     * @returns Query result.
     */
    execute(query: string, options?: QueryOptions): Promise<QueryResult>;
    /**
     * Execute a streaming query.
     *
     * @param query - The Neumann query to execute.
     * @param options - Query options.
     * @returns Async iterator of query results.
     */
    executeStream(query: string, options?: QueryOptions): AsyncIterable<QueryResult>;
    /**
     * Execute multiple queries in a batch.
     *
     * @param queries - List of queries to execute.
     * @param options - Query options.
     * @returns List of query results.
     */
    executeBatch(queries: string[], options?: QueryOptions): Promise<QueryResult[]>;
}
/**
 * Convert a proto value to a Value.
 */
export declare function convertProtoValue(protoValue: unknown): Value;
/**
 * Convert a proto row to a Row.
 */
export declare function convertProtoRow(protoRow: unknown): Row;
/**
 * Convert a proto node to a Node.
 */
export declare function convertProtoNode(protoNode: unknown): Node;
/**
 * Convert a proto edge to an Edge.
 */
export declare function convertProtoEdge(protoEdge: unknown): Edge;
/**
 * Convert a proto path to a Path.
 */
export declare function convertProtoPath(protoPath: unknown): Path;
/**
 * Convert a proto similar item to a SimilarItem.
 */
export declare function convertProtoSimilarItem(protoItem: unknown): SimilarItem;
/**
 * Convert a proto artifact info to an ArtifactInfo.
 */
export declare function convertProtoArtifactInfo(protoInfo: unknown): ArtifactInfo;
//# sourceMappingURL=client.d.ts.map