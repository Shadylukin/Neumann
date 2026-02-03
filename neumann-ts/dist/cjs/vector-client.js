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
exports.VectorClient = void 0;
const errors_js_1 = require("./types/errors.js");
const vector_js_1 = require("./services/vector.js");
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
class VectorClient {
    pointsClient;
    collectionsClient;
    grpcPointsClient;
    grpcCollectionsClient;
    connected = true;
    constructor(pointsClient, collectionsClient, grpcPointsClient, grpcCollectionsClient) {
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
    static async connect(address, options = {}) {
        try {
            const grpc = await Promise.resolve().then(() => __importStar(require('@grpc/grpc-js')));
            const { loadVectorProto, getPointsServiceClient, getCollectionsServiceClient, } = await Promise.resolve().then(() => __importStar(require('./grpc.js')));
            const proto = await loadVectorProto();
            const credentials = options.tls
                ? grpc.credentials.createSsl()
                : grpc.credentials.createInsecure();
            const grpcPointsClient = getPointsServiceClient(proto, address, credentials);
            const grpcCollectionsClient = getCollectionsServiceClient(proto, address, credentials);
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
            const pointsClient = new vector_js_1.PointsClient(grpcPointsClient, metadata);
            const collectionsClient = new vector_js_1.CollectionsClient(grpcCollectionsClient, metadata);
            return new VectorClient(pointsClient, collectionsClient, grpcPointsClient, grpcCollectionsClient);
        }
        catch (err) {
            throw new errors_js_1.ConnectionError(`Failed to connect to ${address}: ${String(err)}`);
        }
    }
    /**
     * Check if client is connected.
     */
    get isConnected() {
        return this.connected;
    }
    /**
     * Close the client connection.
     */
    close() {
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
    async createCollection(name, dimension, distance = 'cosine') {
        this.ensureConnected();
        return this.collectionsClient.create(name, dimension, distance);
    }
    /**
     * Get collection information.
     *
     * @param name - Collection name.
     * @returns Collection information.
     */
    async getCollection(name) {
        this.ensureConnected();
        return this.collectionsClient.get(name);
    }
    /**
     * Delete a collection.
     *
     * @param name - Collection name.
     * @returns True if collection was deleted.
     */
    async deleteCollection(name) {
        this.ensureConnected();
        return this.collectionsClient.delete(name);
    }
    /**
     * List all collections.
     *
     * @returns Array of collection names.
     */
    async listCollections() {
        this.ensureConnected();
        return this.collectionsClient.list();
    }
    /**
     * Check if a collection exists.
     *
     * @param name - Collection name.
     * @returns True if collection exists.
     */
    async collectionExists(name) {
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
    async upsertPoints(collection, points) {
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
    async getPoints(collection, ids, options) {
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
    async deletePoints(collection, ids) {
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
    async queryPoints(collection, vector, options) {
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
    async scrollPoints(collection, options) {
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
    async *scrollAllPoints(collection, options) {
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
    async countPoints(collection) {
        const info = await this.getCollection(collection);
        return info.pointsCount;
    }
    /**
     * Access the underlying PointsClient.
     */
    get points() {
        return this.pointsClient;
    }
    /**
     * Access the underlying CollectionsClient.
     */
    get collections() {
        return this.collectionsClient;
    }
    ensureConnected() {
        if (!this.connected) {
            throw new errors_js_1.ConnectionError('Client is not connected');
        }
    }
}
exports.VectorClient = VectorClient;
//# sourceMappingURL=vector-client.js.map