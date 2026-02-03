// SPDX-License-Identifier: MIT
/**
 * Retry logic with exponential backoff for transient failures.
 */
/**
 * gRPC status codes that are typically retryable.
 */
export const RETRYABLE_STATUS_CODES = {
    DEADLINE_EXCEEDED: 4,
    UNAVAILABLE: 14,
};
/**
 * Default retry configuration.
 */
export const DEFAULT_RETRY_CONFIG = {
    maxAttempts: 3,
    initialBackoffMs: 100,
    maxBackoffMs: 10000,
    backoffMultiplier: 2.0,
    retryableStatusCodes: [
        RETRYABLE_STATUS_CODES.UNAVAILABLE,
        RETRYABLE_STATUS_CODES.DEADLINE_EXCEEDED,
    ],
};
/**
 * Merge partial config with defaults.
 */
export function mergeRetryConfig(partial) {
    return { ...DEFAULT_RETRY_CONFIG, ...partial };
}
/**
 * Check if an error is retryable based on gRPC status code.
 */
export function isRetryable(error, config) {
    if (error === null || typeof error !== 'object') {
        return false;
    }
    const e = error;
    if (typeof e.code !== 'number') {
        return false;
    }
    return config.retryableStatusCodes.includes(e.code);
}
/**
 * Calculate backoff time with jitter.
 *
 * @param attempt - The current attempt number (0-indexed).
 * @param initialMs - Initial backoff in milliseconds.
 * @param maxMs - Maximum backoff in milliseconds.
 * @param multiplier - Exponential multiplier.
 * @returns Backoff time in milliseconds.
 */
export function calculateBackoff(attempt, initialMs, maxMs, multiplier) {
    const backoffMs = initialMs * Math.pow(multiplier, attempt);
    // Add jitter between 0.8 and 1.2
    const jitter = 0.8 + Math.random() * 0.4;
    return Math.min(backoffMs * jitter, maxMs);
}
/**
 * Sleep for a given number of milliseconds.
 */
function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
/**
 * Execute an async function with retry logic.
 *
 * @param fn - The async function to execute.
 * @param config - Retry configuration.
 * @returns The result of the function.
 * @throws The last error if all retries are exhausted.
 */
export async function withRetry(fn, config = DEFAULT_RETRY_CONFIG) {
    let lastError = null;
    for (let attempt = 0; attempt < config.maxAttempts; attempt++) {
        try {
            return await fn();
        }
        catch (error) {
            if (!isRetryable(error, config)) {
                throw error;
            }
            lastError = error;
            if (attempt < config.maxAttempts - 1) {
                const backoffMs = calculateBackoff(attempt, config.initialBackoffMs, config.maxBackoffMs, config.backoffMultiplier);
                await sleep(backoffMs);
            }
        }
    }
    if (lastError !== null) {
        throw lastError;
    }
    throw new Error('Retry loop completed without result');
}
/**
 * Create a retryable wrapper for an async function.
 *
 * @param fn - The async function to wrap.
 * @param config - Retry configuration.
 * @returns A wrapped function that will retry on transient failures.
 */
export function withRetryWrapper(fn, config = DEFAULT_RETRY_CONFIG) {
    return (...args) => withRetry(() => fn(...args), config);
}
//# sourceMappingURL=retry.js.map