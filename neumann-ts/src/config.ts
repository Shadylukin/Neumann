// SPDX-License-Identifier: MIT
/**
 * Configuration classes for Neumann client.
 */

import type { RetryConfig, PartialRetryConfig } from './retry.js';
import { DEFAULT_RETRY_CONFIG, mergeRetryConfig } from './retry.js';

/**
 * Configuration for request timeouts.
 */
export interface TimeoutConfig {
  /** Default timeout for operations in seconds (default: 30). */
  defaultTimeoutS: number;
  /** Timeout for establishing connections in seconds (default: 10). */
  connectTimeoutS: number;
  /** Timeout for query operations in seconds (default: uses defaultTimeoutS). */
  queryTimeoutS?: number;
  /** Timeout for blob upload operations in seconds (default: 300). */
  blobUploadTimeoutS?: number;
  /** Timeout for blob download operations in seconds (default: 300). */
  blobDownloadTimeoutS?: number;
}

/**
 * Default timeout configuration.
 */
export const DEFAULT_TIMEOUT_CONFIG: TimeoutConfig = {
  defaultTimeoutS: 30,
  connectTimeoutS: 10,
  blobUploadTimeoutS: 300,
  blobDownloadTimeoutS: 300,
};

/**
 * Configuration for gRPC keepalive to detect dead connections.
 */
export interface KeepaliveConfig {
  /** Time between keepalive pings in milliseconds (default: 30000). */
  timeMs: number;
  /** Timeout for keepalive ping acknowledgement in milliseconds (default: 10000). */
  timeoutMs: number;
  /** Whether to send keepalive even when no active RPCs (default: true). */
  permitWithoutCalls: boolean;
}

/**
 * Default keepalive configuration.
 */
export const DEFAULT_KEEPALIVE_CONFIG: KeepaliveConfig = {
  timeMs: 30000,
  timeoutMs: 10000,
  permitWithoutCalls: true,
};

/**
 * Partial timeout configuration for overrides.
 */
export type PartialTimeoutConfig = Partial<TimeoutConfig>;

/**
 * Partial keepalive configuration for overrides.
 */
export type PartialKeepaliveConfig = Partial<KeepaliveConfig>;

/**
 * Complete client configuration.
 */
export interface ClientConfig {
  /** Timeout configuration. */
  timeout: TimeoutConfig;
  /** Retry configuration. */
  retry: RetryConfig;
  /** Keepalive configuration. */
  keepalive: KeepaliveConfig;
}

/**
 * Partial client configuration for overrides.
 */
export interface PartialClientConfig {
  timeout?: PartialTimeoutConfig;
  retry?: PartialRetryConfig;
  keepalive?: PartialKeepaliveConfig;
}

/**
 * Default client configuration.
 */
export const DEFAULT_CLIENT_CONFIG: ClientConfig = {
  timeout: DEFAULT_TIMEOUT_CONFIG,
  retry: DEFAULT_RETRY_CONFIG,
  keepalive: DEFAULT_KEEPALIVE_CONFIG,
};

/**
 * Merge partial config with defaults.
 */
export function mergeClientConfig(partial: PartialClientConfig = {}): ClientConfig {
  return {
    timeout: { ...DEFAULT_TIMEOUT_CONFIG, ...partial.timeout },
    retry: mergeRetryConfig(partial.retry ?? {}),
    keepalive: { ...DEFAULT_KEEPALIVE_CONFIG, ...partial.keepalive },
  };
}

/**
 * Create a configuration with retry disabled.
 */
export function noRetryConfig(): ClientConfig {
  return mergeClientConfig({
    retry: { maxAttempts: 1 },
  });
}

/**
 * Create a configuration for fast failure detection.
 */
export function fastFailConfig(): ClientConfig {
  return mergeClientConfig({
    timeout: {
      defaultTimeoutS: 5,
      connectTimeoutS: 2,
    },
    retry: { maxAttempts: 1 },
    keepalive: {
      timeMs: 10000,
      timeoutMs: 5000,
    },
  });
}

/**
 * Create a configuration for high-latency environments.
 */
export function highLatencyConfig(): ClientConfig {
  return mergeClientConfig({
    timeout: {
      defaultTimeoutS: 120,
      connectTimeoutS: 30,
    },
    retry: {
      maxAttempts: 5,
      initialBackoffMs: 500,
    },
    keepalive: {
      timeMs: 60000,
      timeoutMs: 30000,
    },
  });
}

/**
 * Convert timeout config to gRPC channel options.
 */
export function toGrpcChannelOptions(
  config: ClientConfig
): Record<string, string | number | boolean> {
  return {
    'grpc.keepalive_time_ms': config.keepalive.timeMs,
    'grpc.keepalive_timeout_ms': config.keepalive.timeoutMs,
    'grpc.keepalive_permit_without_calls': config.keepalive.permitWithoutCalls ? 1 : 0,
    'grpc.initial_reconnect_backoff_ms': config.retry.initialBackoffMs,
    'grpc.max_reconnect_backoff_ms': config.retry.maxBackoffMs,
  };
}
