"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.InternalError = exports.QueryError = exports.ParseError = exports.InvalidArgumentError = exports.NotFoundError = exports.PermissionDeniedError = exports.AuthenticationError = exports.ConnectionError = exports.NeumannError = exports.ErrorCode = void 0;
exports.errorFromCode = errorFromCode;
// SPDX-License-Identifier: MIT
/**
 * Error codes from the Neumann server.
 */
var ErrorCode;
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
})(ErrorCode || (exports.ErrorCode = ErrorCode = {}));
/**
 * Base error class for all Neumann errors.
 */
class NeumannError extends Error {
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
exports.NeumannError = NeumannError;
/**
 * Error connecting to the database.
 */
class ConnectionError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.UNAVAILABLE);
        this.name = 'ConnectionError';
        Object.setPrototypeOf(this, ConnectionError.prototype);
    }
}
exports.ConnectionError = ConnectionError;
/**
 * Authentication failed.
 */
class AuthenticationError extends NeumannError {
    constructor(message = 'Authentication failed') {
        super(message, ErrorCode.UNAUTHENTICATED);
        this.name = 'AuthenticationError';
        Object.setPrototypeOf(this, AuthenticationError.prototype);
    }
}
exports.AuthenticationError = AuthenticationError;
/**
 * Permission denied.
 */
class PermissionDeniedError extends NeumannError {
    constructor(message = 'Permission denied') {
        super(message, ErrorCode.PERMISSION_DENIED);
        this.name = 'PermissionDeniedError';
        Object.setPrototypeOf(this, PermissionDeniedError.prototype);
    }
}
exports.PermissionDeniedError = PermissionDeniedError;
/**
 * Resource not found.
 */
class NotFoundError extends NeumannError {
    constructor(resource) {
        super(`Not found: ${resource}`, ErrorCode.NOT_FOUND);
        this.name = 'NotFoundError';
        Object.setPrototypeOf(this, NotFoundError.prototype);
    }
}
exports.NotFoundError = NotFoundError;
/**
 * Invalid argument provided.
 */
class InvalidArgumentError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.INVALID_ARGUMENT);
        this.name = 'InvalidArgumentError';
        Object.setPrototypeOf(this, InvalidArgumentError.prototype);
    }
}
exports.InvalidArgumentError = InvalidArgumentError;
/**
 * Query parsing failed.
 */
class ParseError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.PARSE_ERROR);
        this.name = 'ParseError';
        Object.setPrototypeOf(this, ParseError.prototype);
    }
}
exports.ParseError = ParseError;
/**
 * Query execution failed.
 */
class QueryError extends NeumannError {
    constructor(message) {
        super(message, ErrorCode.QUERY_ERROR);
        this.name = 'QueryError';
        Object.setPrototypeOf(this, QueryError.prototype);
    }
}
exports.QueryError = QueryError;
/**
 * Internal server error.
 */
class InternalError extends NeumannError {
    constructor(message = 'Internal error') {
        super(message, ErrorCode.INTERNAL);
        this.name = 'InternalError';
        Object.setPrototypeOf(this, InternalError.prototype);
    }
}
exports.InternalError = InternalError;
/**
 * Create the appropriate error type from an error code.
 */
function errorFromCode(code, message) {
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