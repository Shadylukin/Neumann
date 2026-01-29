"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.NeumannClient = void 0;
exports.convertProtoValue = convertProtoValue;
exports.convertProtoRow = convertProtoRow;
exports.convertProtoNode = convertProtoNode;
exports.convertProtoEdge = convertProtoEdge;
exports.convertProtoPath = convertProtoPath;
exports.convertProtoSimilarItem = convertProtoSimilarItem;
exports.convertProtoArtifactInfo = convertProtoArtifactInfo;
const errors_js_1 = require("./types/errors.js");
const value_js_1 = require("./types/value.js");
/**
 * Client for Neumann database supporting both embedded and remote modes.
 */
class NeumannClient {
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
            const grpc = await Promise.resolve().then(() => __importStar(require('@grpc/grpc-js')));
            const credentials = options.tls
                ? grpc.credentials.createSsl()
                : grpc.credentials.createInsecure();
            // Create gRPC channel
            const channel = new grpc.Channel(address, credentials, {});
            client.client = { channel, metadata: options.metadata };
            client.connected = true;
        }
        catch (err) {
            throw new errors_js_1.ConnectionError(`Failed to connect to ${address}: ${String(err)}`);
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
            const grpcWeb = await Promise.resolve().then(() => __importStar(require('grpc-web')));
            client.client = new grpcWeb.GrpcWebClientBase({ format: 'binary' });
            client.connected = true;
        }
        catch (err) {
            throw new errors_js_1.ConnectionError(`Failed to connect via gRPC-Web: ${String(err)}`);
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
            throw new errors_js_1.ConnectionError('Client is not connected');
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
            throw new errors_js_1.ConnectionError('Client is not connected');
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
            throw new errors_js_1.ConnectionError('Client is not connected');
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
exports.NeumannClient = NeumannClient;
/**
 * Convert a proto value to a Value.
 */
function convertProtoValue(protoValue) {
    if (protoValue === null || protoValue === undefined) {
        return (0, value_js_1.nullValue)();
    }
    const v = protoValue;
    if ('nullValue' in v) {
        return (0, value_js_1.nullValue)();
    }
    if ('intValue' in v && typeof v.intValue === 'number') {
        return (0, value_js_1.intValue)(v.intValue);
    }
    if ('floatValue' in v && typeof v.floatValue === 'number') {
        return (0, value_js_1.floatValue)(v.floatValue);
    }
    if ('stringValue' in v && typeof v.stringValue === 'string') {
        return (0, value_js_1.stringValue)(v.stringValue);
    }
    if ('boolValue' in v && typeof v.boolValue === 'boolean') {
        return (0, value_js_1.boolValue)(v.boolValue);
    }
    if ('bytesValue' in v && v.bytesValue instanceof Uint8Array) {
        return (0, value_js_1.bytesValue)(v.bytesValue);
    }
    return (0, value_js_1.nullValue)();
}
/**
 * Convert a proto row to a Row.
 */
function convertProtoRow(protoRow) {
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
function convertProtoNode(protoNode) {
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
function convertProtoEdge(protoEdge) {
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
function convertProtoPath(protoPath) {
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
function convertProtoSimilarItem(protoItem) {
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
function convertProtoArtifactInfo(protoInfo) {
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