/**
 * Health service client for server health checks.
 */
import type * as grpc from '@grpc/grpc-js';
import type { HealthClient as GrpcHealthClient } from '../generated/neumann.js';
/**
 * Health status enum.
 */
export declare enum HealthStatus {
    Unknown = "UNKNOWN",
    Serving = "SERVING",
    NotServing = "NOT_SERVING"
}
/**
 * Health check result.
 */
export interface HealthCheckResult {
    /** Health status. */
    status: HealthStatus;
    /** Whether the service is healthy (serving). */
    healthy: boolean;
}
/**
 * Service client for health checks.
 */
export declare class HealthClient {
    private client;
    private metadata;
    constructor(client: GrpcHealthClient, metadata: grpc.Metadata);
    /**
     * Check the health of the server or a specific service.
     *
     * @param service - Optional service name to check.
     * @returns Health check result.
     */
    check(service?: string): Promise<HealthCheckResult>;
    /**
     * Check if the server is healthy.
     *
     * @returns True if the server is serving.
     */
    isHealthy(): Promise<boolean>;
    private mapStatus;
    private handleError;
}
//# sourceMappingURL=health.d.ts.map