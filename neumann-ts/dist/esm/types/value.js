/**
 * Create a null value.
 */
export function nullValue() {
    return { type: 'null', data: null };
}
/**
 * Create an integer value.
 */
export function intValue(v) {
    return { type: 'int', data: Math.floor(v) };
}
/**
 * Create a float value.
 */
export function floatValue(v) {
    return { type: 'float', data: v };
}
/**
 * Create a string value.
 */
export function stringValue(v) {
    return { type: 'string', data: v };
}
/**
 * Create a boolean value.
 */
export function boolValue(v) {
    return { type: 'bool', data: v };
}
/**
 * Create a bytes value.
 */
export function bytesValue(v) {
    return { type: 'bytes', data: v };
}
/**
 * Convert a Value to its native JavaScript type.
 */
export function valueToNative(value) {
    return value.data;
}
/**
 * Create a Value from a native JavaScript value.
 */
export function valueFromNative(v) {
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
//# sourceMappingURL=value.js.map