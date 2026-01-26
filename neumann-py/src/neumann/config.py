"""Configuration classes for Neumann client."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    max_attempts: int = 3
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 10000
    backoff_multiplier: float = 2.0
    retryable_status_codes: tuple[int, ...] = (
        14,  # UNAVAILABLE
        4,  # DEADLINE_EXCEEDED (for idempotent ops)
    )


@dataclass
class TimeoutConfig:
    """Configuration for request timeouts."""

    default_timeout_s: float = 30.0
    connect_timeout_s: float = 10.0
    query_timeout_s: float | None = None
    blob_upload_timeout_s: float | None = 300.0
    blob_download_timeout_s: float | None = 300.0


@dataclass
class KeepaliveConfig:
    """Configuration for gRPC keepalive to detect dead connections."""

    time_ms: int = 30000  # Send keepalive ping every 30s
    timeout_ms: int = 10000  # Wait 10s for ping ack
    permit_without_calls: bool = True  # Send even when no active RPCs


@dataclass
class ClientConfig:
    """Complete client configuration."""

    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    keepalive: KeepaliveConfig = field(default_factory=KeepaliveConfig)

    @classmethod
    def default(cls) -> ClientConfig:
        """Create default configuration."""
        return cls()

    @classmethod
    def no_retry(cls) -> ClientConfig:
        """Create configuration with retry disabled."""
        return cls(retry=RetryConfig(max_attempts=1))

    @classmethod
    def fast_fail(cls) -> ClientConfig:
        """Create configuration for fast failure detection."""
        return cls(
            timeout=TimeoutConfig(default_timeout_s=5.0, connect_timeout_s=2.0),
            retry=RetryConfig(max_attempts=1),
            keepalive=KeepaliveConfig(time_ms=10000, timeout_ms=5000),
        )

    @classmethod
    def high_latency(cls) -> ClientConfig:
        """Create configuration for high-latency environments."""
        return cls(
            timeout=TimeoutConfig(default_timeout_s=120.0, connect_timeout_s=30.0),
            retry=RetryConfig(max_attempts=5, initial_backoff_ms=500),
            keepalive=KeepaliveConfig(time_ms=60000, timeout_ms=30000),
        )
