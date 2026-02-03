import type { QueryResult, Row, Node, Edge, Path, SimilarItem, ArtifactInfo, PageRankItem, CentralityItem, CentralityType, CommunityItem, CommunityMemberList, PatternMatchBinding, PatternBindingEntry, PatternBindingValue, PatternMatchStats, ConstraintItem, AggregateValue, ChainSubResult, UnifiedItem, CheckpointInfo } from './types/query-result.js';
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
 * Options for paginated query execution.
 */
export interface PaginationOptions extends QueryOptions {
    /** Cursor for pagination continuation. */
    cursor?: string;
    /** Number of items per page. */
    pageSize?: number;
    /** Whether to count total results. */
    countTotal?: boolean;
    /** Cursor time-to-live in seconds. */
    cursorTtlSecs?: number;
}
/**
 * Result of a paginated query.
 */
export interface PaginatedResult<T extends QueryResult = QueryResult> {
    /** The query result for this page. */
    result: T;
    /** Cursor for the next page, if available. */
    nextCursor?: string;
    /** Cursor for the previous page, if available. */
    prevCursor?: string;
    /** Total count of results, if requested. */
    totalCount?: number;
    /** Whether there are more results. */
    hasMore: boolean;
    /** Number of items in this page. */
    pageSize: number;
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
    private grpcClient;
    private grpcMetadata;
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
    query(query: string, options?: QueryOptions): Promise<QueryResult>;
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
     * Automatically cancels the stream on early break or error.
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
    /**
     * Execute a paginated query and return the result with cursor information.
     *
     * @param query - The Neumann query to execute.
     * @param options - Pagination options.
     * @returns Paginated result with cursor information.
     */
    executePaginated(query: string, options?: PaginationOptions): Promise<PaginatedResult>;
    /**
     * Close a pagination cursor.
     *
     * @param cursor - The cursor to close.
     * @returns Whether the cursor was successfully closed.
     */
    closeCursor(cursor: string): Promise<boolean>;
    /**
     * Execute a query and iterate through all pages.
     * Automatically closes the cursor on early break or error.
     *
     * @param query - The Neumann query to execute.
     * @param options - Pagination options (cursor is ignored).
     * @returns Async iterator of query results from all pages.
     */
    executeAllPages(query: string, options?: Omit<PaginationOptions, 'cursor'>): AsyncIterable<QueryResult>;
    /**
     * Convert a proto QueryResponse to a QueryResult.
     */
    private convertProtoResponse;
    /**
     * Convert a proto QueryResponseChunk to a QueryResult.
     */
    private convertProtoChunk;
    /**
     * Assert that the client is connected and return the gRPC client and metadata.
     * Replaces non-null assertions with proper runtime checks.
     */
    private assertConnected;
    /**
     * Convert a gRPC error to a NeumannError.
     */
    private handleGrpcError;
}
/**
 * Convert a proto value to a Value.
 * Validates numeric values to prevent DoS via overflow.
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
/**
 * Convert a proto checkpoint to a CheckpointInfo.
 */
export declare function convertProtoCheckpoint(protoCheckpoint: unknown): CheckpointInfo;
/**
 * Convert a proto PageRank item to a PageRankItem.
 */
export declare function convertProtoPageRankItem(protoItem: unknown): PageRankItem;
/**
 * Convert a proto centrality type to CentralityType.
 */
export declare function convertProtoCentralityType(protoType: string): CentralityType;
/**
 * Convert a proto centrality item to a CentralityItem.
 */
export declare function convertProtoCentralityItem(protoItem: unknown): CentralityItem;
/**
 * Convert a proto community item to a CommunityItem.
 */
export declare function convertProtoCommunityItem(protoItem: unknown): CommunityItem;
/**
 * Convert a proto community member list to a CommunityMemberList.
 */
export declare function convertProtoCommunityMemberList(protoList: unknown): CommunityMemberList;
/**
 * Convert a proto pattern match binding to a PatternMatchBinding.
 */
export declare function convertProtoPatternMatchBinding(protoBinding: unknown): PatternMatchBinding;
/**
 * Convert a proto binding entry to a PatternBindingEntry.
 */
export declare function convertProtoBindingEntry(protoEntry: unknown): PatternBindingEntry;
/**
 * Convert a proto binding value to a PatternBindingValue.
 */
export declare function convertProtoBindingValue(protoValue: unknown): PatternBindingValue;
/**
 * Convert a proto pattern match stats to PatternMatchStats.
 */
export declare function convertProtoPatternMatchStats(protoStats: unknown): PatternMatchStats;
/**
 * Convert a proto constraint item to a ConstraintItem.
 */
export declare function convertProtoConstraintItem(protoItem: unknown): ConstraintItem;
/**
 * Convert a proto aggregate value to an AggregateValue.
 */
export declare function convertProtoAggregateValue(protoValue: unknown): AggregateValue;
/**
 * Convert a proto chain result to a ChainSubResult.
 */
export declare function convertProtoChainResult(protoChain: unknown): ChainSubResult;
/**
 * Convert a proto unified item to a UnifiedItem.
 * Uses proper value conversion to handle mixed field types.
 */
export declare function convertProtoUnifiedItem(protoItem: unknown): UnifiedItem;
//# sourceMappingURL=client.d.ts.map