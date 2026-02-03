/** Maximum allowed string length (1MB). */
export declare const MAX_STRING_LENGTH: number;
/** Maximum allowed bytes length (16MB). */
export declare const MAX_BYTES_LENGTH: number;
/**
 * Validate an integer value from proto.
 * Throws if the value is NaN, Infinity, or outside JavaScript's safe integer range.
 *
 * @param n - The number to validate.
 * @param context - Optional context for error messages.
 * @returns The validated number.
 * @throws InvalidArgumentError if validation fails.
 */
export declare function validateIntValue(n: number, context?: string): number;
/**
 * Validate a float value from proto.
 * Throws if the value is NaN or Infinity.
 *
 * @param n - The number to validate.
 * @param context - Optional context for error messages.
 * @returns The validated number.
 * @throws InvalidArgumentError if validation fails.
 */
export declare function validateFloatValue(n: number, context?: string): number;
/**
 * Validate a string value from proto.
 * Throws if the string exceeds the maximum allowed length.
 *
 * @param s - The string to validate.
 * @param context - Optional context for error messages.
 * @returns The validated string.
 * @throws InvalidArgumentError if validation fails.
 */
export declare function validateStringValue(s: string, context?: string): string;
/**
 * Validate a bytes value from proto.
 * Throws if the bytes exceed the maximum allowed length.
 *
 * @param b - The bytes to validate.
 * @param context - Optional context for error messages.
 * @returns The validated bytes.
 * @throws InvalidArgumentError if validation fails.
 */
export declare function validateBytesValue(b: Uint8Array, context?: string): Uint8Array;
/**
 * Safely convert a numeric ID to a string.
 * Throws if the ID is outside JavaScript's safe integer range.
 *
 * @param id - The numeric ID to convert.
 * @param context - Optional context for error messages.
 * @returns The ID as a string.
 * @throws InvalidArgumentError if the ID is outside safe range.
 */
export declare function safeIdToString(id: number, context?: string): string;
/**
 * Safely convert an array of numeric IDs to strings.
 * Throws if any ID is outside JavaScript's safe integer range.
 *
 * @param ids - The numeric IDs to convert.
 * @param context - Optional context for error messages.
 * @returns The IDs as strings.
 * @throws InvalidArgumentError if any ID is outside safe range.
 */
export declare function safeIdsToStrings(ids: number[], context?: string): string[];
//# sourceMappingURL=validation.d.ts.map