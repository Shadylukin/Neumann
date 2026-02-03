import { ConnectionError, NotFoundError, InternalError, InvalidArgumentError, } from '../types/errors.js';
/**
 * Service client for vector points operations.
 */
export class PointsClient {
    client;
    metadata;
    constructor(client, metadata) {
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
    async upsert(collection, points) {
        return new Promise((resolve, reject) => {
            const protoPoints = points.map((p) => ({
                id: p.id,
                vector: p.vector,
                payload: p.payload ? this.encodePayload(p.payload) : undefined,
            }));
            this.client.Upsert({ collection, points: protoPoints }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.upserted);
            });
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
    async get(collection, ids, options = {}) {
        return new Promise((resolve, reject) => {
            this.client.Get({
                collection,
                ids,
                withPayload: options.withPayload ?? true,
                withVector: options.withVector ?? false,
            }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.points.map((p) => this.convertPoint(p)));
            });
        });
    }
    /**
     * Delete points by IDs.
     *
     * @param collection - Target collection name.
     * @param ids - Point IDs to delete.
     * @returns Number of points deleted.
     */
    async delete(collection, ids) {
        return new Promise((resolve, reject) => {
            this.client.Delete({ collection, ids }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.deleted);
            });
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
    async query(collection, vector, options = {}) {
        return new Promise((resolve, reject) => {
            this.client.Query({
                collection,
                vector,
                limit: options.limit ?? 10,
                offset: options.offset ?? 0,
                scoreThreshold: options.scoreThreshold,
                withPayload: options.withPayload ?? true,
                withVector: options.withVector ?? false,
            }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.results.map((p) => this.convertScoredPoint(p)));
            });
        });
    }
    /**
     * Scroll through points in a collection.
     *
     * @param collection - Target collection name.
     * @param options - Scroll options.
     * @returns Scroll result with points and next offset.
     */
    async scroll(collection, options = {}) {
        return new Promise((resolve, reject) => {
            this.client.Scroll({
                collection,
                offsetId: options.offsetId,
                limit: options.limit ?? 100,
                withPayload: options.withPayload ?? true,
                withVector: options.withVector ?? false,
            }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve({
                    points: response.points.map((p) => this.convertPoint(p)),
                    nextOffset: response.nextOffset,
                });
            });
        });
    }
    /**
     * Iterate through all points in a collection.
     *
     * @param collection - Target collection name.
     * @param options - Scroll options (limit per page).
     * @returns Async iterable of points.
     */
    async *scrollAll(collection, options = {}) {
        let offsetId;
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
    convertPoint(proto) {
        return {
            id: proto.id,
            vector: proto.vector,
            payload: proto.payload ? this.decodePayload(proto.payload) : undefined,
        };
    }
    convertScoredPoint(proto) {
        return {
            id: proto.id,
            score: proto.score,
            payload: proto.payload ? this.decodePayload(proto.payload) : undefined,
            vector: proto.vector && proto.vector.length > 0 ? proto.vector : undefined,
        };
    }
    encodePayload(payload) {
        const result = {};
        for (const [key, value] of Object.entries(payload)) {
            result[key] = new TextEncoder().encode(JSON.stringify(value));
        }
        return result;
    }
    decodePayload(payload) {
        const result = {};
        for (const [key, value] of Object.entries(payload)) {
            try {
                result[key] = JSON.parse(new TextDecoder().decode(value));
            }
            catch {
                result[key] = new TextDecoder().decode(value);
            }
        }
        return result;
    }
    handleError(err) {
        const code = err.code;
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
    client;
    metadata;
    constructor(client, metadata) {
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
    async create(name, dimension, distance = 'cosine') {
        return new Promise((resolve, reject) => {
            this.client.Create({ name, dimension, distance }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.created);
            });
        });
    }
    /**
     * Get collection information.
     *
     * @param name - Collection name.
     * @returns Collection information.
     */
    async get(name) {
        return new Promise((resolve, reject) => {
            this.client.Get({ name }, this.metadata, (err, response) => {
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
            });
        });
    }
    /**
     * Delete a collection.
     *
     * @param name - Collection name.
     * @returns True if collection was deleted.
     */
    async delete(name) {
        return new Promise((resolve, reject) => {
            this.client.Delete({ name }, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.deleted);
            });
        });
    }
    /**
     * List all collections.
     *
     * @returns Array of collection names.
     */
    async list() {
        return new Promise((resolve, reject) => {
            this.client.List({}, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                resolve(response.collections);
            });
        });
    }
    /**
     * Check if a collection exists.
     *
     * @param name - Collection name.
     * @returns True if collection exists.
     */
    async exists(name) {
        try {
            await this.get(name);
            return true;
        }
        catch (err) {
            if (err instanceof NotFoundError) {
                return false;
            }
            throw err;
        }
    }
    handleError(err) {
        const code = err.code;
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
//# sourceMappingURL=vector.js.map