/**
 * Scalar value types supported by Neumann.
 */
export type ScalarType = 'null' | 'int' | 'float' | 'string' | 'bool' | 'bytes';
/**
 * A typed scalar value.
 */
export interface Value {
    type: ScalarType;
    data: number | string | boolean | Uint8Array | null;
}
/**
 * Create a null value.
 */
export declare function nullValue(): Value;
/**
 * Create an integer value.
 */
export declare function intValue(v: number): Value;
/**
 * Create a float value.
 */
export declare function floatValue(v: number): Value;
/**
 * Create a string value.
 */
export declare function stringValue(v: string): Value;
/**
 * Create a boolean value.
 */
export declare function boolValue(v: boolean): Value;
/**
 * Create a bytes value.
 */
export declare function bytesValue(v: Uint8Array): Value;
/**
 * Convert a Value to its native JavaScript type.
 */
export declare function valueToNative(value: Value): number | string | boolean | Uint8Array | null;
/**
 * Create a Value from a native JavaScript value.
 */
export declare function valueFromNative(v: unknown): Value;
//# sourceMappingURL=value.d.ts.map