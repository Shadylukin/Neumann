import { ConnectionError, AuthenticationError, PermissionDeniedError, NotFoundError, InvalidArgumentError, InternalError, errorFromCode, } from './types/errors.js';
import { nullValue, intValue, floatValue, stringValue, boolValue, bytesValue, valueFromNative, } from './types/value.js';
import { validateIntValue, validateFloatValue, validateStringValue, validateBytesValue, safeIdToString, safeIdsToStrings, } from './types/validation.js';
import { mergeClientConfig, toGrpcChannelOptions } from './config.js';
import { withRetry } from './retry.js';
/**
 * Client for Neumann database supporting both embedded and remote modes.
 */
export class NeumannClient {
    mode;
    connected = false;
    client = null;
    apiKey;
    address;
    grpcClient = null;
    grpcMetadata = null;
    config;
    constructor(mode, config) {
        this.mode = mode;
        this.config = config;
    }
    /**
     * Get the client configuration.
     */
    get clientConfig() {
        return this.config;
    }
    /**
     * Connect to a remote Neumann server via gRPC.
     *
     * @param address - Server address in format "host:port".
     * @param options - Connection options.
     * @returns A connected NeumannClient.
     */
    static async connect(address, options = {}) {
        const config = mergeClientConfig(options.config);
        const client = new NeumannClient('remote', config);
        client.apiKey = options.apiKey;
        client.address = address;
        try {
            const grpc = await import('@grpc/grpc-js');
            const { loadProto, getQueryServiceClient } = await import('./grpc.js');
            const proto = await loadProto();
            const credentials = options.tls
                ? grpc.credentials.createSsl()
                : grpc.credentials.createInsecure();
            // Apply channel options from config
            const channelOptions = toGrpcChannelOptions(config);
            client.grpcClient = getQueryServiceClient(proto, address, credentials, channelOptions);
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
        }
        catch (err) {
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
    static async connectWeb(address, options = {}) {
        const config = mergeClientConfig(options.config);
        const client = new NeumannClient('remote', config);
        client.apiKey = options.apiKey;
        client.address = address;
        try {
            // Dynamic import for gRPC-Web (browser environment)
            const grpcWeb = await import('grpc-web');
            client.client = new grpcWeb.GrpcWebClientBase({ format: 'binary' });
            client.connected = true;
        }
        catch (err) {
            throw new ConnectionError(`Failed to connect via gRPC-Web: ${String(err)}`);
        }
        return client;
    }
    /**
     * Check if client is connected.
     */
    get isConnected() {
        return this.connected;
    }
    /**
     * Get the client mode.
     */
    get clientMode() {
        return this.mode;
    }
    /**
     * Close the client connection.
     */
    close() {
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
    async query(query, options = {}) {
        return this.execute(query, options);
    }
    /**
     * Execute a query and return the result.
     *
     * @param query - The Neumann query to execute.
     * @param options - Query options.
     * @returns Query result.
     */
    async execute(query, options = {}) {
        const { grpcClient, metadata } = this.assertConnected();
        const request = {
            query,
            identity: options.identity ?? '',
        };
        const executeOnce = () => new Promise((resolve, reject) => {
            grpcClient.Execute(request, metadata, (err, response) => {
                if (err) {
                    reject(this.handleGrpcError(err));
                    return;
                }
                try {
                    resolve(this.convertProtoResponse(response));
                }
                catch (e) {
                    reject(e);
                }
            });
        });
        return withRetry(executeOnce, this.config.retry);
    }
    /**
     * Execute a streaming query.
     * Automatically cancels the stream on early break or error.
     *
     * @param query - The Neumann query to execute.
     * @param options - Query options.
     * @returns Async iterator of query results.
     */
    async *executeStream(query, options = {}) {
        const { grpcClient, metadata } = this.assertConnected();
        const request = {
            query,
            identity: options.identity ?? '',
        };
        const stream = grpcClient.ExecuteStream(request, metadata);
        try {
            for await (const chunk of stream) {
                const c = chunk;
                if (c.isFinal) {
                    break;
                }
                if (c.error) {
                    throw errorFromCode(c.error.code, c.error.message);
                }
                yield this.convertProtoChunk(chunk);
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
     * Execute multiple queries in a batch.
     *
     * @param queries - List of queries to execute.
     * @param options - Query options.
     * @returns List of query results.
     */
    async executeBatch(queries, options = {}) {
        const { grpcClient, metadata } = this.assertConnected();
        const request = {
            queries: queries.map((q) => ({
                query: q,
                identity: options.identity ?? '',
            })),
        };
        const executeBatchOnce = () => new Promise((resolve, reject) => {
            grpcClient.ExecuteBatch(request, metadata, (err, response) => {
                if (err) {
                    reject(this.handleGrpcError(err));
                    return;
                }
                try {
                    const r = response;
                    const results = (r.results ?? []).map((res) => this.convertProtoResponse(res));
                    resolve(results);
                }
                catch (e) {
                    reject(e);
                }
            });
        });
        return withRetry(executeBatchOnce, this.config.retry);
    }
    /**
     * Execute a paginated query and return the result with cursor information.
     *
     * @param query - The Neumann query to execute.
     * @param options - Pagination options.
     * @returns Paginated result with cursor information.
     */
    async executePaginated(query, options = {}) {
        const { grpcClient, metadata } = this.assertConnected();
        const request = {
            query,
            identity: options.identity ?? '',
            cursor: options.cursor ?? '',
            pageSize: options.pageSize ?? 0,
            countTotal: options.countTotal ?? false,
            cursorTtlSecs: options.cursorTtlSecs ?? 0,
        };
        const executePaginatedOnce = () => new Promise((resolve, reject) => {
            grpcClient.ExecutePaginated(request, metadata, (err, response) => {
                if (err) {
                    reject(this.handleGrpcError(err));
                    return;
                }
                try {
                    const r = response;
                    const result = this.convertProtoResponse(r.result);
                    const paginatedResult = {
                        result,
                        hasMore: r.hasMore ?? false,
                        pageSize: r.pageSize ?? 0,
                    };
                    if (r.nextCursor) {
                        paginatedResult.nextCursor = r.nextCursor;
                    }
                    if (r.prevCursor) {
                        paginatedResult.prevCursor = r.prevCursor;
                    }
                    if (r.totalCount !== undefined) {
                        paginatedResult.totalCount = r.totalCount;
                    }
                    resolve(paginatedResult);
                }
                catch (e) {
                    reject(e);
                }
            });
        });
        return withRetry(executePaginatedOnce, this.config.retry);
    }
    /**
     * Close a pagination cursor.
     *
     * @param cursor - The cursor to close.
     * @returns Whether the cursor was successfully closed.
     */
    async closeCursor(cursor) {
        const { grpcClient, metadata } = this.assertConnected();
        const closeCursorOnce = () => new Promise((resolve, reject) => {
            grpcClient.CloseCursor({ cursor }, metadata, (err, response) => {
                if (err) {
                    reject(this.handleGrpcError(err));
                    return;
                }
                const r = response;
                resolve(r.success ?? false);
            });
        });
        return withRetry(closeCursorOnce, this.config.retry);
    }
    /**
     * Execute a query and iterate through all pages.
     * Automatically closes the cursor on early break or error.
     *
     * @param query - The Neumann query to execute.
     * @param options - Pagination options (cursor is ignored).
     * @returns Async iterator of query results from all pages.
     */
    async *executeAllPages(query, options = {}) {
        let cursor;
        let cleanupNeeded = false;
        try {
            do {
                const paginatedOptions = { ...options };
                if (cursor) {
                    paginatedOptions.cursor = cursor;
                }
                const result = await this.executePaginated(query, paginatedOptions);
                cursor = result.nextCursor;
                cleanupNeeded = cursor !== undefined;
                yield result.result;
            } while (cursor);
            cleanupNeeded = false; // Completed normally, no cleanup needed
        }
        finally {
            if (cleanupNeeded && cursor) {
                try {
                    await this.closeCursor(cursor);
                }
                catch {
                    // Ignore cleanup errors
                }
            }
        }
    }
    /**
     * Convert a proto QueryResponse to a QueryResult.
     */
    convertProtoResponse(response) {
        const r = response;
        if (r.error) {
            return { type: 'error', code: r.error.code, message: r.error.message };
        }
        if (r.empty !== undefined) {
            return { type: 'empty' };
        }
        if (r.value !== undefined) {
            return { type: 'value', value: r.value.value };
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
            return {
                type: 'ids',
                ids: r.ids.ids.map((id, i) => typeof id === 'number' ? safeIdToString(id, `ids[${i}]`) : id),
            };
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
        if (r.artifactList !== undefined) {
            return { type: 'artifactList', artifactIds: r.artifactList.artifactIds };
        }
        if (r.blobStats !== undefined) {
            return {
                type: 'blobStats',
                artifactCount: r.blobStats.artifactCount,
                chunkCount: r.blobStats.chunkCount,
                totalBytes: r.blobStats.totalBytes,
                uniqueBytes: r.blobStats.uniqueBytes,
                dedupRatio: r.blobStats.dedupRatio,
                orphanedChunks: r.blobStats.orphanedChunks,
            };
        }
        if (r.checkpointList !== undefined) {
            return {
                type: 'checkpointList',
                checkpoints: r.checkpointList.checkpoints.map((cp) => convertProtoCheckpoint(cp)),
            };
        }
        if (r.pageRank !== undefined) {
            const result = {
                type: 'pageRank',
                items: r.pageRank.items.map((item) => convertProtoPageRankItem(item)),
            };
            if (r.pageRank.iterations !== undefined) {
                result.iterations = r.pageRank.iterations;
            }
            if (r.pageRank.convergence !== undefined) {
                result.convergence = r.pageRank.convergence;
            }
            if (r.pageRank.converged !== undefined) {
                result.converged = r.pageRank.converged;
            }
            return result;
        }
        if (r.centrality !== undefined) {
            const result = {
                type: 'centrality',
                items: r.centrality.items.map((item) => convertProtoCentralityItem(item)),
            };
            if (r.centrality.centralityType !== undefined) {
                result.centralityType = convertProtoCentralityType(r.centrality.centralityType);
            }
            if (r.centrality.iterations !== undefined) {
                result.iterations = r.centrality.iterations;
            }
            if (r.centrality.converged !== undefined) {
                result.converged = r.centrality.converged;
            }
            if (r.centrality.sampleCount !== undefined) {
                result.sampleCount = r.centrality.sampleCount;
            }
            return result;
        }
        if (r.communities !== undefined) {
            const result = {
                type: 'communities',
                items: r.communities.items.map((item) => convertProtoCommunityItem(item)),
            };
            if (r.communities.communityCount !== undefined) {
                result.communityCount = r.communities.communityCount;
            }
            if (r.communities.modularity !== undefined) {
                result.modularity = r.communities.modularity;
            }
            if (r.communities.passes !== undefined) {
                result.passes = r.communities.passes;
            }
            if (r.communities.iterations !== undefined) {
                result.iterations = r.communities.iterations;
            }
            if (r.communities.communities !== undefined) {
                result.communities = r.communities.communities.map((c) => convertProtoCommunityMemberList(c));
            }
            return result;
        }
        if (r.patternMatch !== undefined) {
            const result = {
                type: 'patternMatch',
                matches: r.patternMatch.matches.map((m) => convertProtoPatternMatchBinding(m)),
            };
            if (r.patternMatch.stats !== undefined) {
                result.stats = convertProtoPatternMatchStats(r.patternMatch.stats);
            }
            return result;
        }
        if (r.constraints !== undefined) {
            return {
                type: 'constraints',
                items: r.constraints.items.map((item) => convertProtoConstraintItem(item)),
            };
        }
        if (r.aggregate !== undefined) {
            return {
                type: 'aggregate',
                value: convertProtoAggregateValue(r.aggregate),
            };
        }
        if (r.batchOperation !== undefined) {
            return {
                type: 'batchOperation',
                operation: r.batchOperation.operation,
                affectedCount: r.batchOperation.affectedCount,
                createdIds: safeIdsToStrings(r.batchOperation.createdIds, 'createdIds'),
            };
        }
        if (r.graphIndexes !== undefined) {
            return { type: 'graphIndexes', indexes: r.graphIndexes.indexes };
        }
        if (r.chain !== undefined) {
            return { type: 'chain', result: convertProtoChainResult(r.chain) };
        }
        if (r.unified !== undefined) {
            return {
                type: 'unified',
                description: r.unified.description,
                items: r.unified.items.map((item) => convertProtoUnifiedItem(item)),
            };
        }
        return { type: 'empty' };
    }
    /**
     * Convert a proto QueryResponseChunk to a QueryResult.
     */
    convertProtoChunk(chunk) {
        const c = chunk;
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
     * Assert that the client is connected and return the gRPC client and metadata.
     * Replaces non-null assertions with proper runtime checks.
     */
    assertConnected() {
        if (!this.connected || !this.grpcClient || !this.grpcMetadata) {
            throw new ConnectionError('Client is not connected');
        }
        return { grpcClient: this.grpcClient, metadata: this.grpcMetadata };
    }
    /**
     * Convert a gRPC error to a NeumannError.
     */
    handleGrpcError(err) {
        // gRPC status codes
        const code = err.code;
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
 * Validates numeric values to prevent DoS via overflow.
 */
export function convertProtoValue(protoValue) {
    if (protoValue === null || protoValue === undefined) {
        return nullValue();
    }
    const v = protoValue;
    if ('nullValue' in v) {
        return nullValue();
    }
    if ('intValue' in v && typeof v.intValue === 'number') {
        return intValue(validateIntValue(v.intValue));
    }
    if ('floatValue' in v && typeof v.floatValue === 'number') {
        return floatValue(validateFloatValue(v.floatValue));
    }
    if ('stringValue' in v && typeof v.stringValue === 'string') {
        return stringValue(validateStringValue(v.stringValue));
    }
    if ('boolValue' in v && typeof v.boolValue === 'boolean') {
        return boolValue(v.boolValue);
    }
    if ('bytesValue' in v && v.bytesValue instanceof Uint8Array) {
        return bytesValue(validateBytesValue(v.bytesValue));
    }
    return nullValue();
}
/**
 * Convert a proto row to a Row.
 */
export function convertProtoRow(protoRow) {
    const values = new Map();
    const row = protoRow;
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
export function convertProtoNode(protoNode) {
    const properties = new Map();
    const node = protoNode;
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
export function convertProtoEdge(protoEdge) {
    const properties = new Map();
    const edge = protoEdge;
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
export function convertProtoPath(protoPath) {
    const segments = [];
    const path = protoPath;
    if (path.segments) {
        for (const seg of path.segments) {
            const segment = {
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
export function convertProtoSimilarItem(protoItem) {
    const item = protoItem;
    if (item.metadata && item.metadata.length > 0) {
        const metadata = new Map();
        for (const prop of item.metadata) {
            metadata.set(prop.name, convertProtoValue(prop.value));
        }
        return {
            key: item.key,
            score: item.score,
            metadata,
        };
    }
    return {
        key: item.key,
        score: item.score,
    };
}
/**
 * Convert a proto artifact info to an ArtifactInfo.
 */
export function convertProtoArtifactInfo(protoInfo) {
    const info = protoInfo;
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
/**
 * Convert a proto checkpoint to a CheckpointInfo.
 */
export function convertProtoCheckpoint(protoCheckpoint) {
    const cp = protoCheckpoint;
    return {
        id: cp.id,
        name: cp.name,
        createdAt: cp.createdAt,
        isAuto: cp.isAuto,
    };
}
/**
 * Convert a proto PageRank item to a PageRankItem.
 */
export function convertProtoPageRankItem(protoItem) {
    const item = protoItem;
    return {
        nodeId: safeIdToString(item.nodeId, 'nodeId'),
        score: item.score,
    };
}
/**
 * Convert a proto centrality type to CentralityType.
 */
export function convertProtoCentralityType(protoType) {
    const typeMap = {
        CENTRALITY_TYPE_BETWEENNESS: 'betweenness',
        CENTRALITY_TYPE_CLOSENESS: 'closeness',
        CENTRALITY_TYPE_EIGENVECTOR: 'eigenvector',
    };
    return typeMap[protoType] ?? 'betweenness';
}
/**
 * Convert a proto centrality item to a CentralityItem.
 */
export function convertProtoCentralityItem(protoItem) {
    const item = protoItem;
    return {
        nodeId: safeIdToString(item.nodeId, 'nodeId'),
        score: item.score,
    };
}
/**
 * Convert a proto community item to a CommunityItem.
 */
export function convertProtoCommunityItem(protoItem) {
    const item = protoItem;
    return {
        nodeId: safeIdToString(item.nodeId, 'nodeId'),
        communityId: safeIdToString(item.communityId, 'communityId'),
    };
}
/**
 * Convert a proto community member list to a CommunityMemberList.
 */
export function convertProtoCommunityMemberList(protoList) {
    const list = protoList;
    return {
        communityId: safeIdToString(list.communityId, 'communityId'),
        memberNodeIds: safeIdsToStrings(list.memberNodeIds, 'memberNodeIds'),
    };
}
/**
 * Convert a proto pattern match binding to a PatternMatchBinding.
 */
export function convertProtoPatternMatchBinding(protoBinding) {
    const binding = protoBinding;
    return {
        bindings: binding.bindings.map((b) => convertProtoBindingEntry(b)),
    };
}
/**
 * Convert a proto binding entry to a PatternBindingEntry.
 */
export function convertProtoBindingEntry(protoEntry) {
    const entry = protoEntry;
    return {
        variable: entry.variable,
        value: convertProtoBindingValue(entry.value),
    };
}
/**
 * Convert a proto binding value to a PatternBindingValue.
 */
export function convertProtoBindingValue(protoValue) {
    const value = protoValue;
    if (value.node) {
        return {
            type: 'node',
            value: {
                id: safeIdToString(value.node.id, 'node.id'),
                label: value.node.label,
            },
        };
    }
    if (value.edge) {
        return {
            type: 'edge',
            value: {
                id: safeIdToString(value.edge.id, 'edge.id'),
                edgeType: value.edge.edgeType,
                from: safeIdToString(value.edge.from, 'edge.from'),
                to: safeIdToString(value.edge.to, 'edge.to'),
            },
        };
    }
    if (value.path) {
        return {
            type: 'path',
            value: {
                nodes: safeIdsToStrings(value.path.nodes, 'path.nodes'),
                edges: safeIdsToStrings(value.path.edges, 'path.edges'),
                length: value.path.length,
            },
        };
    }
    // Default to empty node binding
    return { type: 'node', value: { id: '', label: '' } };
}
/**
 * Convert a proto pattern match stats to PatternMatchStats.
 */
export function convertProtoPatternMatchStats(protoStats) {
    const stats = protoStats;
    return {
        matchesFound: stats.matchesFound,
        nodesEvaluated: stats.nodesEvaluated,
        edgesEvaluated: stats.edgesEvaluated,
        truncated: stats.truncated,
    };
}
/**
 * Convert a proto constraint item to a ConstraintItem.
 */
export function convertProtoConstraintItem(protoItem) {
    const item = protoItem;
    return {
        name: item.name,
        target: item.target,
        property: item.property,
        constraintType: item.constraintType,
    };
}
/**
 * Convert a proto aggregate value to an AggregateValue.
 */
export function convertProtoAggregateValue(protoValue) {
    const value = protoValue;
    if (value.count !== undefined) {
        return { type: 'count', value: value.count };
    }
    if (value.sum !== undefined) {
        return { type: 'sum', value: value.sum };
    }
    if (value.avg !== undefined) {
        return { type: 'avg', value: value.avg };
    }
    if (value.min !== undefined) {
        return { type: 'min', value: value.min };
    }
    if (value.max !== undefined) {
        return { type: 'max', value: value.max };
    }
    return { type: 'count', value: 0 };
}
/**
 * Convert a proto chain result to a ChainSubResult.
 */
export function convertProtoChainResult(protoChain) {
    const chain = protoChain;
    if (chain.transactionBegun) {
        return { type: 'transactionBegun', value: { txId: chain.transactionBegun.txId } };
    }
    if (chain.committed) {
        return {
            type: 'committed',
            value: { blockHash: chain.committed.blockHash, height: chain.committed.height },
        };
    }
    if (chain.rolledBack) {
        return { type: 'rolledBack', value: { toHeight: chain.rolledBack.toHeight } };
    }
    if (chain.history) {
        return {
            type: 'history',
            value: {
                entries: chain.history.entries.map((e) => {
                    const entry = e;
                    const result = {
                        height: entry.height,
                        transactionType: entry.transactionType,
                    };
                    if (entry.data) {
                        result.data = entry.data;
                    }
                    return result;
                }),
            },
        };
    }
    if (chain.similar) {
        return {
            type: 'similar',
            value: {
                items: chain.similar.items.map((i) => {
                    const item = i;
                    return {
                        blockHash: item.blockHash,
                        height: item.height,
                        similarity: item.similarity,
                    };
                }),
            },
        };
    }
    if (chain.drift) {
        return {
            type: 'drift',
            value: {
                fromHeight: chain.drift.fromHeight,
                toHeight: chain.drift.toHeight,
                totalDrift: chain.drift.totalDrift,
                avgDriftPerBlock: chain.drift.avgDriftPerBlock,
                maxDrift: chain.drift.maxDrift,
            },
        };
    }
    if (chain.height) {
        return { type: 'height', value: { height: chain.height.height } };
    }
    if (chain.tip) {
        return { type: 'tip', value: { hash: chain.tip.hash, height: chain.tip.height } };
    }
    if (chain.block) {
        return {
            type: 'block',
            value: {
                height: chain.block.height,
                hash: chain.block.hash,
                prevHash: chain.block.prevHash,
                timestamp: chain.block.timestamp,
                transactionCount: chain.block.transactionCount,
                proposer: chain.block.proposer,
            },
        };
    }
    if (chain.codebook) {
        const result = {
            scope: chain.codebook.scope,
            entryCount: chain.codebook.entryCount,
            dimension: chain.codebook.dimension,
        };
        if (chain.codebook.domain) {
            result.domain = chain.codebook.domain;
        }
        return { type: 'codebook', value: result };
    }
    if (chain.transitionAnalysis) {
        return {
            type: 'transitionAnalysis',
            value: {
                totalTransitions: chain.transitionAnalysis.totalTransitions,
                validTransitions: chain.transitionAnalysis.validTransitions,
                invalidTransitions: chain.transitionAnalysis.invalidTransitions,
                avgValidityScore: chain.transitionAnalysis.avgValidityScore,
            },
        };
    }
    if (chain.conflictResolution) {
        return {
            type: 'conflictResolution',
            value: {
                strategy: chain.conflictResolution.strategy,
                conflictsResolved: chain.conflictResolution.conflictsResolved,
            },
        };
    }
    if (chain.merge) {
        return {
            type: 'merge',
            value: { success: chain.merge.success, mergedCount: chain.merge.mergedCount },
        };
    }
    // Default to height 0
    return { type: 'height', value: { height: 0 } };
}
/**
 * Convert a proto unified item to a UnifiedItem.
 * Uses proper value conversion to handle mixed field types.
 */
export function convertProtoUnifiedItem(protoItem) {
    const item = protoItem;
    const fields = new Map();
    if (item.fields) {
        for (const [key, value] of Object.entries(item.fields)) {
            fields.set(key, valueFromNative(value));
        }
    }
    const result = {
        entityType: item.entityType,
        key: item.key,
        fields,
    };
    if (item.score !== undefined) {
        result.score = item.score;
    }
    return result;
}
//# sourceMappingURL=client.js.map