# SPDX-License-Identifier: MIT
"""Error types for Neumann database."""

from __future__ import annotations

from enum import Enum


class ErrorCode(Enum):
    """Error codes from the server."""

    UNKNOWN = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    PERMISSION_DENIED = 3
    ALREADY_EXISTS = 4
    UNAUTHENTICATED = 5
    UNAVAILABLE = 6
    INTERNAL = 7
    PARSE_ERROR = 8
    QUERY_ERROR = 9


class NeumannError(Exception):
    """Base exception for all Neumann errors."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.UNKNOWN) -> None:
        """Initialize with message and error code."""
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        """Return string representation."""
        return f"[{self.code.name}] {self.message}"


class ConnectionError(NeumannError):
    """Error connecting to the database."""

    def __init__(self, message: str) -> None:
        """Initialize connection error."""
        super().__init__(message, ErrorCode.UNAVAILABLE)


class AuthenticationError(NeumannError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed") -> None:
        """Initialize authentication error."""
        super().__init__(message, ErrorCode.UNAUTHENTICATED)


class PermissionError(NeumannError):
    """Permission denied."""

    def __init__(self, message: str = "Permission denied") -> None:
        """Initialize permission error."""
        super().__init__(message, ErrorCode.PERMISSION_DENIED)


class NotFoundError(NeumannError):
    """Resource not found."""

    def __init__(self, resource: str) -> None:
        """Initialize not found error."""
        super().__init__(f"Not found: {resource}", ErrorCode.NOT_FOUND)


class InvalidArgumentError(NeumannError):
    """Invalid argument provided."""

    def __init__(self, message: str) -> None:
        """Initialize invalid argument error."""
        super().__init__(message, ErrorCode.INVALID_ARGUMENT)


class ParseError(NeumannError):
    """Query parsing failed."""

    def __init__(self, message: str) -> None:
        """Initialize parse error."""
        super().__init__(message, ErrorCode.PARSE_ERROR)


class QueryError(NeumannError):
    """Query execution failed."""

    def __init__(self, message: str) -> None:
        """Initialize query error."""
        super().__init__(message, ErrorCode.QUERY_ERROR)


class InternalError(NeumannError):
    """Internal server error."""

    def __init__(self, message: str = "Internal error") -> None:
        """Initialize internal error."""
        super().__init__(message, ErrorCode.INTERNAL)


def error_from_code(code: int | ErrorCode, message: str) -> NeumannError:
    """Create the appropriate error type from an error code."""
    if isinstance(code, int):
        try:
            code = ErrorCode(code)
        except ValueError:
            code = ErrorCode.UNKNOWN

    error_classes: dict[ErrorCode, type[NeumannError]] = {
        ErrorCode.INVALID_ARGUMENT: InvalidArgumentError,
        ErrorCode.NOT_FOUND: NotFoundError,
        ErrorCode.PERMISSION_DENIED: PermissionError,
        ErrorCode.UNAUTHENTICATED: AuthenticationError,
        ErrorCode.UNAVAILABLE: ConnectionError,
        ErrorCode.INTERNAL: InternalError,
        ErrorCode.PARSE_ERROR: ParseError,
        ErrorCode.QUERY_ERROR: QueryError,
    }

    error_cls = error_classes.get(code, NeumannError)
    if error_cls in (NotFoundError,):
        return error_cls(message)
    return error_cls(message)
