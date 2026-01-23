/**
 * Error codes from the Neumann server.
 */
export enum ErrorCode {
  UNKNOWN = 0,
  INVALID_ARGUMENT = 1,
  NOT_FOUND = 2,
  PERMISSION_DENIED = 3,
  ALREADY_EXISTS = 4,
  UNAUTHENTICATED = 5,
  UNAVAILABLE = 6,
  INTERNAL = 7,
  PARSE_ERROR = 8,
  QUERY_ERROR = 9,
}

/**
 * Base error class for all Neumann errors.
 */
export class NeumannError extends Error {
  readonly code: ErrorCode;

  constructor(message: string, code: ErrorCode = ErrorCode.UNKNOWN) {
    super(message);
    this.name = 'NeumannError';
    this.code = code;
    Object.setPrototypeOf(this, NeumannError.prototype);
  }

  override toString(): string {
    return `[${ErrorCode[this.code]}] ${this.message}`;
  }
}

/**
 * Error connecting to the database.
 */
export class ConnectionError extends NeumannError {
  constructor(message: string) {
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
  constructor(resource: string) {
    super(`Not found: ${resource}`, ErrorCode.NOT_FOUND);
    this.name = 'NotFoundError';
    Object.setPrototypeOf(this, NotFoundError.prototype);
  }
}

/**
 * Invalid argument provided.
 */
export class InvalidArgumentError extends NeumannError {
  constructor(message: string) {
    super(message, ErrorCode.INVALID_ARGUMENT);
    this.name = 'InvalidArgumentError';
    Object.setPrototypeOf(this, InvalidArgumentError.prototype);
  }
}

/**
 * Query parsing failed.
 */
export class ParseError extends NeumannError {
  constructor(message: string) {
    super(message, ErrorCode.PARSE_ERROR);
    this.name = 'ParseError';
    Object.setPrototypeOf(this, ParseError.prototype);
  }
}

/**
 * Query execution failed.
 */
export class QueryError extends NeumannError {
  constructor(message: string) {
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
export function errorFromCode(code: number | ErrorCode, message: string): NeumannError {
  const errorCode = typeof code === 'number' ? (code as ErrorCode) : code;

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
