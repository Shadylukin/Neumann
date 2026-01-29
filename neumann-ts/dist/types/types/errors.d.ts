/**
 * Error codes from the Neumann server.
 */
export declare enum ErrorCode {
    UNKNOWN = 0,
    INVALID_ARGUMENT = 1,
    NOT_FOUND = 2,
    PERMISSION_DENIED = 3,
    ALREADY_EXISTS = 4,
    UNAUTHENTICATED = 5,
    UNAVAILABLE = 6,
    INTERNAL = 7,
    PARSE_ERROR = 8,
    QUERY_ERROR = 9
}
/**
 * Base error class for all Neumann errors.
 */
export declare class NeumannError extends Error {
    readonly code: ErrorCode;
    constructor(message: string, code?: ErrorCode);
    toString(): string;
}
/**
 * Error connecting to the database.
 */
export declare class ConnectionError extends NeumannError {
    constructor(message: string);
}
/**
 * Authentication failed.
 */
export declare class AuthenticationError extends NeumannError {
    constructor(message?: string);
}
/**
 * Permission denied.
 */
export declare class PermissionDeniedError extends NeumannError {
    constructor(message?: string);
}
/**
 * Resource not found.
 */
export declare class NotFoundError extends NeumannError {
    constructor(resource: string);
}
/**
 * Invalid argument provided.
 */
export declare class InvalidArgumentError extends NeumannError {
    constructor(message: string);
}
/**
 * Query parsing failed.
 */
export declare class ParseError extends NeumannError {
    constructor(message: string);
}
/**
 * Query execution failed.
 */
export declare class QueryError extends NeumannError {
    constructor(message: string);
}
/**
 * Internal server error.
 */
export declare class InternalError extends NeumannError {
    constructor(message?: string);
}
/**
 * Create the appropriate error type from an error code.
 */
export declare function errorFromCode(code: number | ErrorCode, message: string): NeumannError;
//# sourceMappingURL=errors.d.ts.map