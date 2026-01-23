"""Tests for Neumann error types."""

import pytest

from neumann.errors import (
    AuthenticationError,
    ConnectionError,
    ErrorCode,
    InternalError,
    InvalidArgumentError,
    NeumannError,
    NotFoundError,
    ParseError,
    PermissionError,
    QueryError,
    error_from_code,
)


class TestNeumannError:
    """Tests for NeumannError base class."""

    def test_base_error(self) -> None:
        """Test base error creation."""
        err = NeumannError("test message", ErrorCode.INTERNAL)
        assert err.message == "test message"
        assert err.code == ErrorCode.INTERNAL
        assert "[INTERNAL]" in str(err)

    def test_error_default_code(self) -> None:
        """Test error with default code."""
        err = NeumannError("test")
        assert err.code == ErrorCode.UNKNOWN


class TestSpecificErrors:
    """Tests for specific error types."""

    def test_connection_error(self) -> None:
        """Test connection error."""
        err = ConnectionError("failed to connect")
        assert err.code == ErrorCode.UNAVAILABLE
        assert "failed to connect" in str(err)

    def test_authentication_error(self) -> None:
        """Test authentication error."""
        err = AuthenticationError()
        assert err.code == ErrorCode.UNAUTHENTICATED
        assert "Authentication failed" in str(err)

    def test_permission_error(self) -> None:
        """Test permission error."""
        err = PermissionError()
        assert err.code == ErrorCode.PERMISSION_DENIED
        assert "Permission denied" in str(err)

    def test_not_found_error(self) -> None:
        """Test not found error."""
        err = NotFoundError("users/123")
        assert err.code == ErrorCode.NOT_FOUND
        assert "users/123" in str(err)

    def test_invalid_argument_error(self) -> None:
        """Test invalid argument error."""
        err = InvalidArgumentError("bad input")
        assert err.code == ErrorCode.INVALID_ARGUMENT
        assert "bad input" in str(err)

    def test_parse_error(self) -> None:
        """Test parse error."""
        err = ParseError("unexpected token")
        assert err.code == ErrorCode.PARSE_ERROR
        assert "unexpected token" in str(err)

    def test_query_error(self) -> None:
        """Test query error."""
        err = QueryError("table not found")
        assert err.code == ErrorCode.QUERY_ERROR
        assert "table not found" in str(err)

    def test_internal_error(self) -> None:
        """Test internal error."""
        err = InternalError()
        assert err.code == ErrorCode.INTERNAL
        assert "Internal error" in str(err)


class TestErrorFromCode:
    """Tests for error_from_code function."""

    def test_from_enum_code(self) -> None:
        """Test creating error from ErrorCode enum."""
        err = error_from_code(ErrorCode.NOT_FOUND, "resource missing")
        assert isinstance(err, NotFoundError)
        assert "resource missing" in str(err)

    def test_from_int_code(self) -> None:
        """Test creating error from integer code."""
        err = error_from_code(1, "invalid input")  # INVALID_ARGUMENT = 1
        assert isinstance(err, InvalidArgumentError)

    def test_unknown_int_code(self) -> None:
        """Test creating error from unknown integer code."""
        err = error_from_code(999, "unknown error")
        assert isinstance(err, NeumannError)
        assert err.code == ErrorCode.UNKNOWN

    def test_all_error_codes(self) -> None:
        """Test that all error codes map to appropriate types."""
        mappings = [
            (ErrorCode.INVALID_ARGUMENT, InvalidArgumentError),
            (ErrorCode.NOT_FOUND, NotFoundError),
            (ErrorCode.PERMISSION_DENIED, PermissionError),
            (ErrorCode.UNAUTHENTICATED, AuthenticationError),
            (ErrorCode.UNAVAILABLE, ConnectionError),
            (ErrorCode.INTERNAL, InternalError),
            (ErrorCode.PARSE_ERROR, ParseError),
            (ErrorCode.QUERY_ERROR, QueryError),
        ]

        for code, expected_type in mappings:
            err = error_from_code(code, "test")
            assert isinstance(
                err, expected_type
            ), f"Expected {expected_type.__name__} for {code}"


class TestErrorExceptionHandling:
    """Tests for exception handling patterns."""

    def test_catch_base_error(self) -> None:
        """Test catching all Neumann errors."""
        errors = [
            ConnectionError("test"),
            AuthenticationError(),
            NotFoundError("x"),
            ParseError("test"),
        ]

        for err in errors:
            with pytest.raises(NeumannError):
                raise err

    def test_error_inheritance(self) -> None:
        """Test error inheritance chain."""
        err = NotFoundError("test")
        assert isinstance(err, NeumannError)
        assert isinstance(err, Exception)
