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
export function nullValue(): Value {
  return { type: 'null', data: null };
}

/**
 * Create an integer value.
 */
export function intValue(v: number): Value {
  return { type: 'int', data: Math.floor(v) };
}

/**
 * Create a float value.
 */
export function floatValue(v: number): Value {
  return { type: 'float', data: v };
}

/**
 * Create a string value.
 */
export function stringValue(v: string): Value {
  return { type: 'string', data: v };
}

/**
 * Create a boolean value.
 */
export function boolValue(v: boolean): Value {
  return { type: 'bool', data: v };
}

/**
 * Create a bytes value.
 */
export function bytesValue(v: Uint8Array): Value {
  return { type: 'bytes', data: v };
}

/**
 * Convert a Value to its native JavaScript type.
 */
export function valueToNative(value: Value): number | string | boolean | Uint8Array | null {
  return value.data;
}

/**
 * Create a Value from a native JavaScript value.
 */
export function valueFromNative(v: unknown): Value {
  if (v === null || v === undefined) {
    return nullValue();
  }
  if (typeof v === 'boolean') {
    return boolValue(v);
  }
  if (typeof v === 'number') {
    return Number.isInteger(v) ? intValue(v) : floatValue(v);
  }
  if (typeof v === 'string') {
    return stringValue(v);
  }
  if (v instanceof Uint8Array) {
    return bytesValue(v);
  }
  return stringValue(String(v));
}
