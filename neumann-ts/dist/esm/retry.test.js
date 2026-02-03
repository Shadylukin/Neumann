// SPDX-License-Identifier: MIT
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { withRetry, withRetryWrapper, isRetryable, calculateBackoff, mergeRetryConfig, DEFAULT_RETRY_CONFIG, RETRYABLE_STATUS_CODES, } from './retry.js';
describe('retry', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });
    afterEach(() => {
        vi.useRealTimers();
    });
    describe('isRetryable', () => {
        it('returns false for null', () => {
            expect(isRetryable(null, DEFAULT_RETRY_CONFIG)).toBe(false);
        });
        it('returns false for non-object', () => {
            expect(isRetryable('error', DEFAULT_RETRY_CONFIG)).toBe(false);
            expect(isRetryable(123, DEFAULT_RETRY_CONFIG)).toBe(false);
        });
        it('returns false for object without code', () => {
            expect(isRetryable({ message: 'error' }, DEFAULT_RETRY_CONFIG)).toBe(false);
        });
        it('returns false for non-numeric code', () => {
            expect(isRetryable({ code: 'UNAVAILABLE' }, DEFAULT_RETRY_CONFIG)).toBe(false);
        });
        it('returns true for UNAVAILABLE status code', () => {
            expect(isRetryable({ code: RETRYABLE_STATUS_CODES.UNAVAILABLE }, DEFAULT_RETRY_CONFIG)).toBe(true);
        });
        it('returns true for DEADLINE_EXCEEDED status code', () => {
            expect(isRetryable({ code: RETRYABLE_STATUS_CODES.DEADLINE_EXCEEDED }, DEFAULT_RETRY_CONFIG)).toBe(true);
        });
        it('returns false for non-retryable status code', () => {
            expect(isRetryable({ code: 5 }, DEFAULT_RETRY_CONFIG)).toBe(false); // NOT_FOUND
            expect(isRetryable({ code: 3 }, DEFAULT_RETRY_CONFIG)).toBe(false); // INVALID_ARGUMENT
        });
        it('respects custom retryable status codes', () => {
            const config = {
                ...DEFAULT_RETRY_CONFIG,
                retryableStatusCodes: [5], // Only NOT_FOUND
            };
            expect(isRetryable({ code: 5 }, config)).toBe(true);
            expect(isRetryable({ code: 14 }, config)).toBe(false); // UNAVAILABLE no longer retryable
        });
    });
    describe('calculateBackoff', () => {
        it('calculates exponential backoff', () => {
            // With jitter disabled (using Math.random mock)
            vi.spyOn(Math, 'random').mockReturnValue(0.5); // Returns 1.0 jitter factor (0.8 + 0.5 * 0.4)
            expect(calculateBackoff(0, 100, 10000, 2.0)).toBe(100);
            expect(calculateBackoff(1, 100, 10000, 2.0)).toBe(200);
            expect(calculateBackoff(2, 100, 10000, 2.0)).toBe(400);
            expect(calculateBackoff(3, 100, 10000, 2.0)).toBe(800);
        });
        it('caps at max backoff', () => {
            vi.spyOn(Math, 'random').mockReturnValue(0.5);
            expect(calculateBackoff(10, 100, 1000, 2.0)).toBe(1000);
            expect(calculateBackoff(20, 100, 1000, 2.0)).toBe(1000);
        });
        it('applies jitter between 0.8 and 1.2', () => {
            vi.spyOn(Math, 'random').mockReturnValue(0);
            expect(calculateBackoff(0, 100, 10000, 2.0)).toBe(80);
            vi.spyOn(Math, 'random').mockReturnValue(1);
            expect(calculateBackoff(0, 100, 10000, 2.0)).toBeCloseTo(120);
        });
    });
    describe('mergeRetryConfig', () => {
        it('returns default config for empty input', () => {
            expect(mergeRetryConfig({})).toEqual(DEFAULT_RETRY_CONFIG);
        });
        it('overrides specific fields', () => {
            const config = mergeRetryConfig({ maxAttempts: 5, initialBackoffMs: 200 });
            expect(config.maxAttempts).toBe(5);
            expect(config.initialBackoffMs).toBe(200);
            expect(config.maxBackoffMs).toBe(DEFAULT_RETRY_CONFIG.maxBackoffMs);
            expect(config.backoffMultiplier).toBe(DEFAULT_RETRY_CONFIG.backoffMultiplier);
        });
    });
    describe('withRetry', () => {
        it('returns result on success', async () => {
            const fn = vi.fn().mockResolvedValue('success');
            const result = await withRetry(fn, DEFAULT_RETRY_CONFIG);
            expect(result).toBe('success');
            expect(fn).toHaveBeenCalledTimes(1);
        });
        it('throws non-retryable error immediately', async () => {
            const error = { code: 5, message: 'NOT_FOUND' };
            const fn = vi.fn().mockRejectedValue(error);
            await expect(withRetry(fn, DEFAULT_RETRY_CONFIG)).rejects.toEqual(error);
            expect(fn).toHaveBeenCalledTimes(1);
        });
        it('retries on retryable error', async () => {
            const error = { code: 14, message: 'UNAVAILABLE' };
            const fn = vi
                .fn()
                .mockRejectedValueOnce(error)
                .mockRejectedValueOnce(error)
                .mockResolvedValueOnce('success');
            const resultPromise = withRetry(fn, DEFAULT_RETRY_CONFIG);
            // Advance timers for each retry backoff
            await vi.advanceTimersByTimeAsync(150); // First backoff
            await vi.advanceTimersByTimeAsync(300); // Second backoff
            const result = await resultPromise;
            expect(result).toBe('success');
            expect(fn).toHaveBeenCalledTimes(3);
        });
        it('throws after max attempts exhausted', async () => {
            vi.useRealTimers(); // Use real timers for this test
            const error = { code: 14, message: 'UNAVAILABLE' };
            const fn = vi.fn().mockRejectedValue(error);
            const config = {
                ...DEFAULT_RETRY_CONFIG,
                maxAttempts: 2,
                initialBackoffMs: 1,
                maxBackoffMs: 1,
            };
            await expect(withRetry(fn, config)).rejects.toEqual(error);
            expect(fn).toHaveBeenCalledTimes(2);
            vi.useFakeTimers(); // Restore fake timers
        });
        it('uses default config when not provided', async () => {
            const fn = vi.fn().mockResolvedValue('success');
            const result = await withRetry(fn);
            expect(result).toBe('success');
        });
    });
    describe('withRetryWrapper', () => {
        it('wraps function with retry logic', async () => {
            const error = { code: 14, message: 'UNAVAILABLE' };
            const originalFn = vi
                .fn()
                .mockRejectedValueOnce(error)
                .mockResolvedValueOnce('success');
            const wrappedFn = withRetryWrapper(originalFn, DEFAULT_RETRY_CONFIG);
            const resultPromise = wrappedFn('arg1', 'arg2');
            await vi.advanceTimersByTimeAsync(150);
            const result = await resultPromise;
            expect(result).toBe('success');
            expect(originalFn).toHaveBeenCalledTimes(2);
            expect(originalFn).toHaveBeenCalledWith('arg1', 'arg2');
        });
        it('preserves function arguments', async () => {
            const fn = vi.fn().mockResolvedValue('result');
            const wrapped = withRetryWrapper(fn);
            await wrapped(1, 'two', { three: 3 });
            expect(fn).toHaveBeenCalledWith(1, 'two', { three: 3 });
        });
    });
    describe('RETRYABLE_STATUS_CODES', () => {
        it('contains expected codes', () => {
            expect(RETRYABLE_STATUS_CODES.DEADLINE_EXCEEDED).toBe(4);
            expect(RETRYABLE_STATUS_CODES.UNAVAILABLE).toBe(14);
        });
    });
    describe('DEFAULT_RETRY_CONFIG', () => {
        it('has expected default values', () => {
            expect(DEFAULT_RETRY_CONFIG.maxAttempts).toBe(3);
            expect(DEFAULT_RETRY_CONFIG.initialBackoffMs).toBe(100);
            expect(DEFAULT_RETRY_CONFIG.maxBackoffMs).toBe(10000);
            expect(DEFAULT_RETRY_CONFIG.backoffMultiplier).toBe(2.0);
            expect(DEFAULT_RETRY_CONFIG.retryableStatusCodes).toContain(4);
            expect(DEFAULT_RETRY_CONFIG.retryableStatusCodes).toContain(14);
        });
    });
});
//# sourceMappingURL=retry.test.js.map