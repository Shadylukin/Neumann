// SPDX-License-Identifier: MIT
/**
 * Configuration classes for Neumann client.
 */
import { DEFAULT_RETRY_CONFIG, mergeRetryConfig } from './retry.js';
/**
 * Default timeout configuration.
 */
export const DEFAULT_TIMEOUT_CONFIG = {
    defaultTimeoutS: 30,
    connectTimeoutS: 10,
    blobUploadTimeoutS: 300,
    blobDownloadTimeoutS: 300,
};
/**
 * Default keepalive configuration.
 */
export const DEFAULT_KEEPALIVE_CONFIG = {
    timeMs: 30000,
    timeoutMs: 10000,
    permitWithoutCalls: true,
};
/**
 * Default client configuration.
 */
export const DEFAULT_CLIENT_CONFIG = {
    timeout: DEFAULT_TIMEOUT_CONFIG,
    retry: DEFAULT_RETRY_CONFIG,
    keepalive: DEFAULT_KEEPALIVE_CONFIG,
};
/**
 * Merge partial config with defaults.
 */
export function mergeClientConfig(partial = {}) {
    return {
        timeout: { ...DEFAULT_TIMEOUT_CONFIG, ...partial.timeout },
        retry: mergeRetryConfig(partial.retry ?? {}),
        keepalive: { ...DEFAULT_KEEPALIVE_CONFIG, ...partial.keepalive },
    };
}
/**
 * Create a configuration with retry disabled.
 */
export function noRetryConfig() {
    return mergeClientConfig({
        retry: { maxAttempts: 1 },
    });
}
/**
 * Create a configuration for fast failure detection.
 */
export function fastFailConfig() {
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
export function highLatencyConfig() {
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
export function toGrpcChannelOptions(config) {
    return {
        'grpc.keepalive_time_ms': config.keepalive.timeMs,
        'grpc.keepalive_timeout_ms': config.keepalive.timeoutMs,
        'grpc.keepalive_permit_without_calls': config.keepalive.permitWithoutCalls ? 1 : 0,
        'grpc.initial_reconnect_backoff_ms': config.retry.initialBackoffMs,
        'grpc.max_reconnect_backoff_ms': config.retry.maxBackoffMs,
    };
}
//# sourceMappingURL=config.js.map