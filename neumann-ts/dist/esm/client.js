import { ConnectionError } from './types/errors.js';
import { nullValue, intValue, floatValue, stringValue, boolValue, bytesValue, } from './types/value.js';
/**
 * Client for Neumann database supporting both embedded and remote modes.
 */
export class NeumannClient {
    mode;
    connected = false;
    client = null;
    apiKey;
    address;
    constructor(mode) {
        this.mode = mode;
    }
    /**
     * Connect to a remote Neumann server via gRPC.
     *
     * @param address - Server address in format "host:port".
     * @param options - Connection options.
     * @returns A connected NeumannClient.
     */
    static async connect(address, options = {}) {
        const client = new NeumannClient('remote');
        client.apiKey = options.apiKey;
        client.address = address;
        try {
            // Dynamic import for Node.js gRPC
            const grpc = await import('@grpc/grpc-js');
            const credentials = options.tls
                ? grpc.credentials.createSsl()
                : grpc.credentials.createInsecure();
            // Create gRPC channel
            const channel = new grpc.Channel(address, credentials, {});
            client.client = { channel, metadata: options.metadata };
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
        const client = new NeumannClient('remote');
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
    async execute(query, options = {}) {
        if (!this.connected) {
            throw new ConnectionError('Client is not connected');
        }
        const _clientInfo = this.client;
        // Build request object (will be used when proto stubs are generated)
        const _request = {
            query,
            identity: options.identity,
        };
        // Stub: return empty result as proto loading is not implemented
        // In production, this would use the generated proto stubs
        return await Promise.resolve({ type: 'empty' });
    }
    /**
     * Execute a streaming query.
     *
     * @param query - The Neumann query to execute.
     * @param options - Query options.
     * @returns Async iterator of query results.
     */
    async *executeStream(query, options = {}) {
        if (!this.connected) {
            throw new ConnectionError('Client is not connected');
        }
        // Build request object (will be used when proto stubs are generated)
        const _request = {
            query,
            identity: options.identity,
        };
        // Stub: yield empty result as proto loading is not implemented
        // In production, this would iterate over the gRPC stream
        yield await Promise.resolve({ type: 'empty' });
    }
    /**
     * Execute multiple queries in a batch.
     *
     * @param queries - List of queries to execute.
     * @param options - Query options.
     * @returns List of query results.
     */
    async executeBatch(queries, options = {}) {
        if (!this.connected) {
            throw new ConnectionError('Client is not connected');
        }
        // Build request objects (will be used when proto stubs are generated)
        const _requests = queries.map((q) => ({
            query: q,
            identity: options.identity,
        }));
        // Stub: return empty results as proto loading is not implemented
        // In production, this would use the generated proto stubs
        return await Promise.resolve(queries.map(() => ({ type: 'empty' })));
    }
}
/**
 * Convert a proto value to a Value.
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
    const result = {
        key: item.key,
        score: item.score,
    };
    if (item.metadata && item.metadata.length > 0) {
        const metadata = new Map();
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
//# sourceMappingURL=client.js.map