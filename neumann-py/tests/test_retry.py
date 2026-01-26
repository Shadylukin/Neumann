"""Tests for retry logic."""

import asyncio
from unittest.mock import MagicMock

import pytest

from neumann.config import RetryConfig
from neumann.retry import (
    _calculate_backoff,
    _is_retryable,
    retry_call,
    retry_call_async,
    with_retry,
    with_retry_async,
)


class MockGrpcStatusCode:
    """Mock gRPC status code for testing."""

    def __init__(self, code: int) -> None:
        self.value = (code,)


class MockGrpcError(Exception):
    """Mock gRPC error for testing."""

    def __init__(self, code: int) -> None:
        self._code = MockGrpcStatusCode(code)

    def code(self) -> MockGrpcStatusCode:
        return self._code


class TestIsRetryable:
    """Tests for _is_retryable function."""

    def test_unavailable_is_retryable(self) -> None:
        """Test UNAVAILABLE (14) is retryable."""
        config = RetryConfig()
        error = MockGrpcError(14)
        assert _is_retryable(error, config) is True

    def test_deadline_exceeded_is_retryable(self) -> None:
        """Test DEADLINE_EXCEEDED (4) is retryable."""
        config = RetryConfig()
        error = MockGrpcError(4)
        assert _is_retryable(error, config) is True

    def test_invalid_argument_not_retryable(self) -> None:
        """Test INVALID_ARGUMENT (3) is not retryable."""
        config = RetryConfig()
        error = MockGrpcError(3)
        assert _is_retryable(error, config) is False

    def test_not_found_not_retryable(self) -> None:
        """Test NOT_FOUND (5) is not retryable."""
        config = RetryConfig()
        error = MockGrpcError(5)
        assert _is_retryable(error, config) is False

    def test_regular_exception_not_retryable(self) -> None:
        """Test regular exceptions are not retryable."""
        config = RetryConfig()
        error = ValueError("test")
        assert _is_retryable(error, config) is False

    def test_custom_retryable_codes(self) -> None:
        """Test custom retryable status codes."""
        config = RetryConfig(retryable_status_codes=(1, 2, 3))
        assert _is_retryable(MockGrpcError(1), config) is True
        assert _is_retryable(MockGrpcError(14), config) is False


class TestCalculateBackoff:
    """Tests for _calculate_backoff function."""

    def test_initial_backoff(self) -> None:
        """Test first attempt uses initial backoff."""
        backoff = _calculate_backoff(0, 100, 10000, 2.0)
        # With jitter between 0.8 and 1.2
        assert 80 <= backoff <= 120

    def test_exponential_growth(self) -> None:
        """Test backoff grows exponentially."""
        backoff0 = _calculate_backoff(0, 100, 10000, 2.0)
        backoff1 = _calculate_backoff(1, 100, 10000, 2.0)
        backoff2 = _calculate_backoff(2, 100, 10000, 2.0)

        # Each should roughly double (with jitter variation)
        assert backoff1 > backoff0 * 1.5
        assert backoff2 > backoff1 * 1.5

    def test_max_backoff_cap(self) -> None:
        """Test backoff is capped at max."""
        backoff = _calculate_backoff(10, 100, 1000, 2.0)
        assert backoff <= 1000 * 1.2  # Allow for jitter


class TestRetryCall:
    """Tests for retry_call function."""

    def test_success_first_try(self) -> None:
        """Test successful call on first try."""
        config = RetryConfig(max_attempts=3)
        mock = MagicMock(return_value="success")

        result = retry_call(mock, config)

        assert result == "success"
        assert mock.call_count == 1

    def test_success_after_retries(self) -> None:
        """Test success after transient failures."""
        config = RetryConfig(max_attempts=3, initial_backoff_ms=1)
        mock = MagicMock(side_effect=[MockGrpcError(14), MockGrpcError(14), "success"])

        result = retry_call(mock, config)

        assert result == "success"
        assert mock.call_count == 3

    def test_raises_non_retryable(self) -> None:
        """Test non-retryable error is raised immediately."""
        config = RetryConfig(max_attempts=3)
        mock = MagicMock(side_effect=ValueError("non-retryable"))

        with pytest.raises(ValueError, match="non-retryable"):
            retry_call(mock, config)

        assert mock.call_count == 1

    def test_raises_after_max_attempts(self) -> None:
        """Test error is raised after exhausting attempts."""
        config = RetryConfig(max_attempts=3, initial_backoff_ms=1)
        mock = MagicMock(side_effect=MockGrpcError(14))

        with pytest.raises(MockGrpcError):
            retry_call(mock, config)

        assert mock.call_count == 3

    def test_no_retry_config(self) -> None:
        """Test with max_attempts=1 (no retry)."""
        config = RetryConfig(max_attempts=1)
        mock = MagicMock(side_effect=MockGrpcError(14))

        with pytest.raises(MockGrpcError):
            retry_call(mock, config)

        assert mock.call_count == 1


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_decorator_success(self) -> None:
        """Test decorator on successful function."""
        config = RetryConfig(max_attempts=3)

        @with_retry(config)
        def success_func() -> str:
            return "success"

        result = success_func()
        assert result == "success"

    def test_decorator_with_args(self) -> None:
        """Test decorator preserves function arguments."""
        config = RetryConfig(max_attempts=3)

        @with_retry(config)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5


class TestRetryCallAsync:
    """Tests for async retry functions."""

    @pytest.mark.asyncio
    async def test_async_success_first_try(self) -> None:
        """Test async successful call on first try."""
        config = RetryConfig(max_attempts=3)
        call_count = 0

        async def mock_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_call_async(mock_func, config)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_success_after_retries(self) -> None:
        """Test async success after transient failures."""
        config = RetryConfig(max_attempts=3, initial_backoff_ms=1)
        call_count = 0

        async def mock_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise MockGrpcError(14)
            return "success"

        result = await retry_call_async(mock_func, config)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_raises_non_retryable(self) -> None:
        """Test async non-retryable error is raised immediately."""
        config = RetryConfig(max_attempts=3)
        call_count = 0

        async def mock_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("non-retryable")

        with pytest.raises(ValueError, match="non-retryable"):
            await retry_call_async(mock_func, config)

        assert call_count == 1


class TestWithRetryAsyncDecorator:
    """Tests for with_retry_async decorator."""

    @pytest.mark.asyncio
    async def test_async_decorator_success(self) -> None:
        """Test async decorator on successful function."""
        config = RetryConfig(max_attempts=3)

        @with_retry_async(config)
        async def success_func() -> str:
            return "success"

        result = await success_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_decorator_with_args(self) -> None:
        """Test async decorator preserves function arguments."""
        config = RetryConfig(max_attempts=3)

        @with_retry_async(config)
        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        result = await add(2, 3)
        assert result == 5
