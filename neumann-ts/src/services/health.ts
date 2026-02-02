// SPDX-License-Identifier: MIT
/**
 * Health service client for server health checks.
 */
import type * as grpc from '@grpc/grpc-js';
import type {
  HealthClient as GrpcHealthClient,
  HealthCheckResponse,
  ServingStatus,
} from '../generated/neumann.js';
import { ConnectionError, InternalError } from '../types/errors.js';
import type { NeumannError } from '../types/errors.js';

/**
 * Health status enum.
 */
export enum HealthStatus {
  Unknown = 'UNKNOWN',
  Serving = 'SERVING',
  NotServing = 'NOT_SERVING',
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
export class HealthClient {
  private client: GrpcHealthClient;
  private metadata: grpc.Metadata;

  constructor(client: GrpcHealthClient, metadata: grpc.Metadata) {
    this.client = client;
    this.metadata = metadata;
  }

  /**
   * Check the health of the server or a specific service.
   *
   * @param service - Optional service name to check.
   * @returns Health check result.
   */
  async check(service?: string): Promise<HealthCheckResult> {
    return new Promise((resolve, reject) => {
      const request: { service?: string } = {};
      if (service !== undefined) {
        request.service = service;
      }
      this.client.Check(
        request,
        this.metadata,
        (err: grpc.ServiceError | null, response: HealthCheckResponse) => {
          if (err) {
            reject(this.handleError(err));
            return;
          }
          const status = this.mapStatus(response.status);
          resolve({
            status,
            healthy: status === HealthStatus.Serving,
          });
        }
      );
    });
  }

  /**
   * Check if the server is healthy.
   *
   * @returns True if the server is serving.
   */
  async isHealthy(): Promise<boolean> {
    try {
      const result = await this.check();
      return result.healthy;
    } catch {
      return false;
    }
  }

  private mapStatus(status: ServingStatus): HealthStatus {
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

  private handleError(err: grpc.ServiceError): NeumannError {
    const code = err.code as number;
    const UNAVAILABLE = 14;

    if (code === UNAVAILABLE) {
      return new ConnectionError(err.details || 'Service unavailable');
    }
    return new InternalError(err.details || err.message || 'Internal error');
  }
}
