// SPDX-License-Identifier: MIT
/**
 * Error codes from the Neumann server.
 */
export var ErrorCode;
(function (ErrorCode) {
    ErrorCode[ErrorCode["UNKNOWN"] = 0] = "UNKNOWN";
    ErrorCode[ErrorCode["INVALID_ARGUMENT"] = 1] = "INVALID_ARGUMENT";
    ErrorCode[ErrorCode["NOT_FOUND"] = 2] = "NOT_FOUND";
    ErrorCode[ErrorCode["PERMISSION_DENIED"] = 3] = "PERMISSION_DENIED";
    ErrorCode[ErrorCode["ALREADY_EXISTS"] = 4] = "ALREADY_EXISTS";
    ErrorCode[ErrorCode["UNAUTHENTICATED"] = 5] = "UNAUTHENTICATED";
    ErrorCode[ErrorCode["UNAVAILABLE"] = 6] = "UNAVAILABLE";
    ErrorCode[ErrorCode["INTERNAL"] = 7] = "INTERNAL";
    ErrorCode[ErrorCode["PARSE_ERROR"] = 8] = "PARSE_ERROR";
    ErrorCode[ErrorCode["QUERY_ERROR"] = 9] = "QUERY_ERROR";
})(ErrorCode || (ErrorCode = {}));
/**
 * Base error class for all Neumann errors.
 */
export class NeumannError extends Error {
    code;
    constructor(message, code = ErrorCode.UNKNOWN) {
        super(message);
        this.name = 'NeumannError';
        this.code = code;
        Object.setPrototypeOf(this, NeumannError.prototype);
    }
    toString() {
        return `[${ErrorCode[this.code]}] ${this.message}`;
    }
}
/**
 * Error connecting to the database.
 */
export class ConnectionError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.UNAVAILABLE);
        this.name = 'ConnectionError';
        Object.setPrototypeOf(this, ConnectionError.prototype);
    }
}
/**
 * Authentication failed.
 */
export class AuthenticationError extends NeumannError {
    constructor(message = 'Authentication failed') {
        super(message, ErrorCode.UNAUTHENTICATED);
        this.name = 'AuthenticationError';
        Object.setPrototypeOf(this, AuthenticationError.prototype);
    }
}
/**
 * Permission denied.
 */
export class PermissionDeniedError extends NeumannError {
    constructor(message = 'Permission denied') {
        super(message, ErrorCode.PERMISSION_DENIED);
        this.name = 'PermissionDeniedError';
        Object.setPrototypeOf(this, PermissionDeniedError.prototype);
    }
}
/**
 * Resource not found.
 */
export class NotFoundError extends NeumannError {
    constructor(resource) {
        super(`Not found: ${resource}`, ErrorCode.NOT_FOUND);
        this.name = 'NotFoundError';
        Object.setPrototypeOf(this, NotFoundError.prototype);
    }
}
/**
 * Invalid argument provided.
 */
export class InvalidArgumentError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.INVALID_ARGUMENT);
        this.name = 'InvalidArgumentError';
        Object.setPrototypeOf(this, InvalidArgumentError.prototype);
    }
}
/**
 * Query parsing failed.
 */
export class ParseError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.PARSE_ERROR);
        this.name = 'ParseError';
        Object.setPrototypeOf(this, ParseError.prototype);
    }
}
/**
 * Query execution failed.
 */
export class QueryError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.QUERY_ERROR);
        this.name = 'QueryError';
        Object.setPrototypeOf(this, QueryError.prototype);
    }
}
/**
 * Internal server error.
 */
export class InternalError extends NeumannError {
    constructor(message = 'Internal error') {
        super(message, ErrorCode.INTERNAL);
        this.name = 'InternalError';
        Object.setPrototypeOf(this, InternalError.prototype);
    }
}
/**
 * Create the appropriate error type from an error code.
 */
export function errorFromCode(code, message) {
    const errorCode = typeof code === 'number' ? code : code;
    switch (errorCode) {
        case ErrorCode.INVALID_ARGUMENT:
            return new InvalidArgumentError(message);
        case ErrorCode.NOT_FOUND:
            return new NotFoundError(message);
        case ErrorCode.PERMISSION_DENIED:
            return new PermissionDeniedError(message);
        case ErrorCode.UNAUTHENTICATED:
            return new AuthenticationError(message);
        case ErrorCode.UNAVAILABLE:
            return new ConnectionError(message);
        case ErrorCode.INTERNAL:
            return new InternalError(message);
        case ErrorCode.PARSE_ERROR:
            return new ParseError(message);
        case ErrorCode.QUERY_ERROR:
            return new QueryError(message);
        default:
            return new NeumannError(message, errorCode);
    }
}
//# sourceMappingURL=errors.js.map