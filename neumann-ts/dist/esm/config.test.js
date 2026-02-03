// SPDX-License-Identifier: MIT
import { describe, it, expect } from 'vitest';
import { mergeClientConfig, noRetryConfig, fastFailConfig, highLatencyConfig, toGrpcChannelOptions, DEFAULT_TIMEOUT_CONFIG, DEFAULT_KEEPALIVE_CONFIG, DEFAULT_CLIENT_CONFIG, } from './config.js';
import { DEFAULT_RETRY_CONFIG } from './retry.js';
describe('config', () => {
    describe('DEFAULT_TIMEOUT_CONFIG', () => {
        it('has expected default values', () => {
            expect(DEFAULT_TIMEOUT_CONFIG.defaultTimeoutS).toBe(30);
            expect(DEFAULT_TIMEOUT_CONFIG.connectTimeoutS).toBe(10);
            expect(DEFAULT_TIMEOUT_CONFIG.blobUploadTimeoutS).toBe(300);
            expect(DEFAULT_TIMEOUT_CONFIG.blobDownloadTimeoutS).toBe(300);
        });
    });
    describe('DEFAULT_KEEPALIVE_CONFIG', () => {
        it('has expected default values', () => {
            expect(DEFAULT_KEEPALIVE_CONFIG.timeMs).toBe(30000);
            expect(DEFAULT_KEEPALIVE_CONFIG.timeoutMs).toBe(10000);
            expect(DEFAULT_KEEPALIVE_CONFIG.permitWithoutCalls).toBe(true);
        });
    });
    describe('DEFAULT_CLIENT_CONFIG', () => {
        it('combines all default configs', () => {
            expect(DEFAULT_CLIENT_CONFIG.timeout).toEqual(DEFAULT_TIMEOUT_CONFIG);
            expect(DEFAULT_CLIENT_CONFIG.retry).toEqual(DEFAULT_RETRY_CONFIG);
            expect(DEFAULT_CLIENT_CONFIG.keepalive).toEqual(DEFAULT_KEEPALIVE_CONFIG);
        });
    });
    describe('mergeClientConfig', () => {
        it('returns default config for undefined input', () => {
            expect(mergeClientConfig()).toEqual(DEFAULT_CLIENT_CONFIG);
        });
        it('returns default config for empty input', () => {
            expect(mergeClientConfig({})).toEqual(DEFAULT_CLIENT_CONFIG);
        });
        it('merges timeout overrides', () => {
            const partial = {
                timeout: { defaultTimeoutS: 60, connectTimeoutS: 20 },
            };
            const config = mergeClientConfig(partial);
            expect(config.timeout.defaultTimeoutS).toBe(60);
            expect(config.timeout.connectTimeoutS).toBe(20);
            expect(config.timeout.blobUploadTimeoutS).toBe(300);
            expect(config.retry).toEqual(DEFAULT_RETRY_CONFIG);
            expect(config.keepalive).toEqual(DEFAULT_KEEPALIVE_CONFIG);
        });
        it('merges retry overrides', () => {
            const partial = {
                retry: { maxAttempts: 5, initialBackoffMs: 500 },
            };
            const config = mergeClientConfig(partial);
            expect(config.retry.maxAttempts).toBe(5);
            expect(config.retry.initialBackoffMs).toBe(500);
            expect(config.retry.maxBackoffMs).toBe(DEFAULT_RETRY_CONFIG.maxBackoffMs);
            expect(config.timeout).toEqual(DEFAULT_TIMEOUT_CONFIG);
        });
        it('merges keepalive overrides', () => {
            const partial = {
                keepalive: { timeMs: 60000, permitWithoutCalls: false },
            };
            const config = mergeClientConfig(partial);
            expect(config.keepalive.timeMs).toBe(60000);
            expect(config.keepalive.timeoutMs).toBe(DEFAULT_KEEPALIVE_CONFIG.timeoutMs);
            expect(config.keepalive.permitWithoutCalls).toBe(false);
        });
        it('merges all sections together', () => {
            const partial = {
                timeout: { defaultTimeoutS: 60 },
                retry: { maxAttempts: 5 },
                keepalive: { timeMs: 60000 },
            };
            const config = mergeClientConfig(partial);
            expect(config.timeout.defaultTimeoutS).toBe(60);
            expect(config.timeout.connectTimeoutS).toBe(DEFAULT_TIMEOUT_CONFIG.connectTimeoutS);
            expect(config.retry.maxAttempts).toBe(5);
            expect(config.retry.initialBackoffMs).toBe(DEFAULT_RETRY_CONFIG.initialBackoffMs);
            expect(config.keepalive.timeMs).toBe(60000);
            expect(config.keepalive.timeoutMs).toBe(DEFAULT_KEEPALIVE_CONFIG.timeoutMs);
        });
    });
    describe('noRetryConfig', () => {
        it('sets maxAttempts to 1', () => {
            const config = noRetryConfig();
            expect(config.retry.maxAttempts).toBe(1);
        });
        it('preserves other defaults', () => {
            const config = noRetryConfig();
            expect(config.timeout).toEqual(DEFAULT_TIMEOUT_CONFIG);
            expect(config.keepalive).toEqual(DEFAULT_KEEPALIVE_CONFIG);
            expect(config.retry.initialBackoffMs).toBe(DEFAULT_RETRY_CONFIG.initialBackoffMs);
        });
    });
    describe('fastFailConfig', () => {
        it('uses short timeouts', () => {
            const config = fastFailConfig();
            expect(config.timeout.defaultTimeoutS).toBe(5);
            expect(config.timeout.connectTimeoutS).toBe(2);
        });
        it('disables retry', () => {
            const config = fastFailConfig();
            expect(config.retry.maxAttempts).toBe(1);
        });
        it('uses aggressive keepalive', () => {
            const config = fastFailConfig();
            expect(config.keepalive.timeMs).toBe(10000);
            expect(config.keepalive.timeoutMs).toBe(5000);
        });
    });
    describe('highLatencyConfig', () => {
        it('uses long timeouts', () => {
            const config = highLatencyConfig();
            expect(config.timeout.defaultTimeoutS).toBe(120);
            expect(config.timeout.connectTimeoutS).toBe(30);
        });
        it('uses more retry attempts', () => {
            const config = highLatencyConfig();
            expect(config.retry.maxAttempts).toBe(5);
            expect(config.retry.initialBackoffMs).toBe(500);
        });
        it('uses relaxed keepalive', () => {
            const config = highLatencyConfig();
            expect(config.keepalive.timeMs).toBe(60000);
            expect(config.keepalive.timeoutMs).toBe(30000);
        });
    });
    describe('toGrpcChannelOptions', () => {
        it('converts config to gRPC channel options', () => {
            const config = {
                timeout: DEFAULT_TIMEOUT_CONFIG,
                retry: DEFAULT_RETRY_CONFIG,
                keepalive: {
                    timeMs: 30000,
                    timeoutMs: 10000,
                    permitWithoutCalls: true,
                },
            };
            const options = toGrpcChannelOptions(config);
            expect(options['grpc.keepalive_time_ms']).toBe(30000);
            expect(options['grpc.keepalive_timeout_ms']).toBe(10000);
            expect(options['grpc.keepalive_permit_without_calls']).toBe(1);
            expect(options['grpc.initial_reconnect_backoff_ms']).toBe(100);
            expect(options['grpc.max_reconnect_backoff_ms']).toBe(10000);
        });
        it('converts permitWithoutCalls false to 0', () => {
            const config = {
                timeout: DEFAULT_TIMEOUT_CONFIG,
                retry: DEFAULT_RETRY_CONFIG,
                keepalive: {
                    timeMs: 30000,
                    timeoutMs: 10000,
                    permitWithoutCalls: false,
                },
            };
            const options = toGrpcChannelOptions(config);
            expect(options['grpc.keepalive_permit_without_calls']).toBe(0);
        });
        it('uses retry config for reconnect backoff', () => {
            const config = {
                timeout: DEFAULT_TIMEOUT_CONFIG,
                retry: {
                    ...DEFAULT_RETRY_CONFIG,
                    initialBackoffMs: 500,
                    maxBackoffMs: 5000,
                },
                keepalive: DEFAULT_KEEPALIVE_CONFIG,
            };
            const options = toGrpcChannelOptions(config);
            expect(options['grpc.initial_reconnect_backoff_ms']).toBe(500);
            expect(options['grpc.max_reconnect_backoff_ms']).toBe(5000);
        });
    });
});
//# sourceMappingURL=config.test.js.map