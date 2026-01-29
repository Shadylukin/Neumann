# SPDX-License-Identifier: MIT
"""Retry logic with exponential backoff for transient failures."""

from __future__ import annotations

import asyncio
import random
import time
from functools import wraps
from typing import TYPE_CHECKING, Awaitable, Callable, ParamSpec, TypeVar

if TYPE_CHECKING:
    from neumann.config import RetryConfig

T = TypeVar("T")
P = ParamSpec("P")


def _is_retryable(e: Exception, config: RetryConfig) -> bool:
    """Check if exception is retryable based on gRPC status code."""
    if hasattr(e, "code") and callable(e.code):
        try:
            code = e.code()
            if hasattr(code, "value") and isinstance(code.value, tuple):
                return code.value[0] in config.retryable_status_codes
        except Exception:
            pass
    return False


def _calculate_backoff(
    attempt: int,
    initial_ms: int,
    max_ms: int,
    multiplier: float,
) -> float:
    """Calculate backoff time with jitter."""
    backoff_ms = initial_ms * (multiplier**attempt)
    jitter = random.uniform(0.8, 1.2)
    return min(backoff_ms * jitter, max_ms)


def with_retry(config: RetryConfig) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for sync functions with retry and exponential backoff."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable(e, config):
                        raise
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        sleep_ms = _calculate_backoff(
                            attempt,
                            config.initial_backoff_ms,
                            config.max_backoff_ms,
                            config.backoff_multiplier,
                        )
                        time.sleep(sleep_ms / 1000)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry loop completed without result")

        return wrapper

    return decorator


def with_retry_async(
    config: RetryConfig,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for async functions with retry and exponential backoff."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable(e, config):
                        raise
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        sleep_ms = _calculate_backoff(
                            attempt,
                            config.initial_backoff_ms,
                            config.max_backoff_ms,
                            config.backoff_multiplier,
                        )
                        await asyncio.sleep(sleep_ms / 1000)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry loop completed without result")

        return wrapper

    return decorator


def retry_call(
    func: Callable[[], T],
    config: RetryConfig,
) -> T:
    """Execute a function with retry logic.

    This is a non-decorator approach for cases where you need to retry
    a specific call inline.

    Args:
        func: Zero-argument callable to retry.
        config: Retry configuration.

    Returns:
        Result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            return func()
        except Exception as e:
            if not _is_retryable(e, config):
                raise
            last_exception = e
            if attempt < config.max_attempts - 1:
                sleep_ms = _calculate_backoff(
                    attempt,
                    config.initial_backoff_ms,
                    config.max_backoff_ms,
                    config.backoff_multiplier,
                )
                time.sleep(sleep_ms / 1000)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry loop completed without result")


async def retry_call_async(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: Zero-argument async callable to retry.
        config: Retry configuration.

    Returns:
        Result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            return await func()
        except Exception as e:
            if not _is_retryable(e, config):
                raise
            last_exception = e
            if attempt < config.max_attempts - 1:
                sleep_ms = _calculate_backoff(
                    attempt,
                    config.initial_backoff_ms,
                    config.max_backoff_ms,
                    config.backoff_multiplier,
                )
                await asyncio.sleep(sleep_ms / 1000)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry loop completed without result")
