"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MAX_BYTES_LENGTH = exports.MAX_STRING_LENGTH = void 0;
exports.validateIntValue = validateIntValue;
exports.validateFloatValue = validateFloatValue;
exports.validateStringValue = validateStringValue;
exports.validateBytesValue = validateBytesValue;
exports.safeIdToString = safeIdToString;
exports.safeIdsToStrings = safeIdsToStrings;
// SPDX-License-Identifier: MIT
/**
 * Validation utilities for safe proto value handling.
 *
 * These functions provide bounds checking and validation to prevent
 * DoS attacks via overflow, resource exhaustion, and silent data loss.
 */
const errors_js_1 = require("./errors.js");
/** Maximum allowed string length (1MB). */
exports.MAX_STRING_LENGTH = 1024 * 1024;
/** Maximum allowed bytes length (16MB). */
exports.MAX_BYTES_LENGTH = 16 * 1024 * 1024;
/**
 * Validate an integer value from proto.
 * Throws if the value is NaN, Infinity, or outside JavaScript's safe integer range.
 *
 * @param n - The number to validate.
 * @param context - Optional context for error messages.
 * @returns The validated number.
 * @throws InvalidArgumentError if validation fails.
 */
function validateIntValue(n, context) {
    if (!Number.isFinite(n)) {
        const ctx = context ? ` (${context})` : '';
        throw new errors_js_1.InvalidArgumentError(`Invalid integer value: not finite${ctx}`);
    }
    if (!Number.isSafeInteger(n)) {
        const ctx = context ? ` (${context})` : '';
        throw new errors_js_1.InvalidArgumentError(`Integer value ${n} is outside safe range [-2^53+1, 2^53-1]${ctx}`);
    }
    return n;
}
/**
 * Validate a float value from proto.
 * Throws if the value is NaN or Infinity.
 *
 * @param n - The number to validate.
 * @param context - Optional context for error messages.
 * @returns The validated number.
 * @throws InvalidArgumentError if validation fails.
 */
function validateFloatValue(n, context) {
    if (!Number.isFinite(n)) {
        const ctx = context ? ` (${context})` : '';
        throw new errors_js_1.InvalidArgumentError(`Invalid float value: not finite${ctx}`);
    }
    return n;
}
/**
 * Validate a string value from proto.
 * Throws if the string exceeds the maximum allowed length.
 *
 * @param s - The string to validate.
 * @param context - Optional context for error messages.
 * @returns The validated string.
 * @throws InvalidArgumentError if validation fails.
 */
function validateStringValue(s, context) {
    if (s.length > exports.MAX_STRING_LENGTH) {
        const ctx = context ? ` (${context})` : '';
        throw new errors_js_1.InvalidArgumentError(`String value exceeds maximum length of ${exports.MAX_STRING_LENGTH} bytes${ctx}`);
    }
    return s;
}
/**
 * Validate a bytes value from proto.
 * Throws if the bytes exceed the maximum allowed length.
 *
 * @param b - The bytes to validate.
 * @param context - Optional context for error messages.
 * @returns The validated bytes.
 * @throws InvalidArgumentError if validation fails.
 */
function validateBytesValue(b, context) {
    if (b.length > exports.MAX_BYTES_LENGTH) {
        const ctx = context ? ` (${context})` : '';
        throw new errors_js_1.InvalidArgumentError(`Bytes value exceeds maximum length of ${exports.MAX_BYTES_LENGTH} bytes${ctx}`);
    }
    return b;
}
/**
 * Safely convert a numeric ID to a string.
 * Throws if the ID is outside JavaScript's safe integer range.
 *
 * @param id - The numeric ID to convert.
 * @param context - Optional context for error messages.
 * @returns The ID as a string.
 * @throws InvalidArgumentError if the ID is outside safe range.
 */
function safeIdToString(id, context) {
    if (!Number.isSafeInteger(id)) {
        const ctx = context ? ` for ${context}` : '';
        throw new errors_js_1.InvalidArgumentError(`ID value ${id} is outside safe integer range${ctx}`);
    }
    return String(id);
}
/**
 * Safely convert an array of numeric IDs to strings.
 * Throws if any ID is outside JavaScript's safe integer range.
 *
 * @param ids - The numeric IDs to convert.
 * @param context - Optional context for error messages.
 * @returns The IDs as strings.
 * @throws InvalidArgumentError if any ID is outside safe range.
 */
function safeIdsToStrings(ids, context) {
    return ids.map((id, index) => {
        const ctx = context ? `${context}[${index}]` : `index ${index}`;
        return safeIdToString(id, ctx);
    });
}
//# sourceMappingURL=validation.js.map