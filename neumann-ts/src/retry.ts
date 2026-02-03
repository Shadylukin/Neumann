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
} as const;

/**
 * Configuration for retry behavior.
 */
export interface RetryConfig {
  /** Maximum number of retry attempts (default: 3). */
  maxAttempts: number;
  /** Initial backoff duration in milliseconds (default: 100). */
  initialBackoffMs: number;
  /** Maximum backoff duration in milliseconds (default: 10000). */
  maxBackoffMs: number;
  /** Multiplier for exponential backoff (default: 2.0). */
  backoffMultiplier: number;
  /** gRPC status codes that should trigger a retry. */
  retryableStatusCodes: number[];
}

/**
 * Default retry configuration.
 */
export const DEFAULT_RETRY_CONFIG: RetryConfig = {
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
 * Partial retry configuration for overrides.
 */
export type PartialRetryConfig = Partial<RetryConfig>;

/**
 * Merge partial config with defaults.
 */
export function mergeRetryConfig(partial: PartialRetryConfig): RetryConfig {
  return { ...DEFAULT_RETRY_CONFIG, ...partial };
}

/**
 * Check if an error is retryable based on gRPC status code.
 */
export function isRetryable(error: unknown, config: RetryConfig): boolean {
  if (error === null || typeof error !== 'object') {
    return false;
  }

  const e = error as { code?: number };
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
export function calculateBackoff(
  attempt: number,
  initialMs: number,
  maxMs: number,
  multiplier: number
): number {
  const backoffMs = initialMs * Math.pow(multiplier, attempt);
  // Add jitter between 0.8 and 1.2
  const jitter = 0.8 + Math.random() * 0.4;
  return Math.min(backoffMs * jitter, maxMs);
}

/**
 * Sleep for a given number of milliseconds.
 */
function sleep(ms: number): Promise<void> {
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
export async function withRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig = DEFAULT_RETRY_CONFIG
): Promise<T> {
  let lastError: unknown = null;

  for (let attempt = 0; attempt < config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (!isRetryable(error, config)) {
        throw error;
      }

      lastError = error;

      if (attempt < config.maxAttempts - 1) {
        const backoffMs = calculateBackoff(
          attempt,
          config.initialBackoffMs,
          config.maxBackoffMs,
          config.backoffMultiplier
        );
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
export function withRetryWrapper<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  config: RetryConfig = DEFAULT_RETRY_CONFIG
): (...args: TArgs) => Promise<TResult> {
  return (...args: TArgs) => withRetry(() => fn(...args), config);
}
