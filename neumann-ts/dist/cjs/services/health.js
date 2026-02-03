"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HealthClient = exports.HealthStatus = void 0;
const errors_js_1 = require("../types/errors.js");
/**
 * Health status enum.
 */
var HealthStatus;
(function (HealthStatus) {
    HealthStatus["Unknown"] = "UNKNOWN";
    HealthStatus["Serving"] = "SERVING";
    HealthStatus["NotServing"] = "NOT_SERVING";
})(HealthStatus || (exports.HealthStatus = HealthStatus = {}));
/**
 * Service client for health checks.
 */
class HealthClient {
    client;
    metadata;
    constructor(client, metadata) {
        this.client = client;
        this.metadata = metadata;
    }
    /**
     * Check the health of the server or a specific service.
     *
     * @param service - Optional service name to check.
     * @returns Health check result.
     */
    async check(service) {
        return new Promise((resolve, reject) => {
            const request = {};
            if (service !== undefined) {
                request.service = service;
            }
            this.client.Check(request, this.metadata, (err, response) => {
                if (err) {
                    reject(this.handleError(err));
                    return;
                }
                const status = this.mapStatus(response.status);
                resolve({
                    status,
                    healthy: status === HealthStatus.Serving,
                });
            });
        });
    }
    /**
     * Check if the server is healthy.
     *
     * @returns True if the server is serving.
     */
    async isHealthy() {
        try {
            const result = await this.check();
            return result.healthy;
        }
        catch {
            return false;
        }
    }
    mapStatus(status) {
        // Status comes as a string from proto-loader
        const statusStr = String(status);
        if (statusStr === 'SERVING_STATUS_SERVING' || statusStr === '1') {
            return HealthStatus.Serving;
        }
        if (statusStr === 'SERVING_STATUS_NOT_SERVING' || statusStr === '2') {
            return HealthStatus.NotServing;
        }
        return HealthStatus.Unknown;
    }
    handleError(err) {
        const code = err.code;
        const UNAVAILABLE = 14;
        if (code === UNAVAILABLE) {
            return new errors_js_1.ConnectionError(err.details || 'Service unavailable');
        }
        return new errors_js_1.InternalError(err.details || err.message || 'Internal error');
    }
}
exports.HealthClient = HealthClient;
//# sourceMappingURL=health.js.map