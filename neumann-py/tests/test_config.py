"""Tests for client configuration."""

from neumann.config import (
    ClientConfig,
    KeepaliveConfig,
    RetryConfig,
    TimeoutConfig,
)


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_default_values(self) -> None:
        """Test default timeout values."""
        config = TimeoutConfig()
        assert config.default_timeout_s == 30.0
        assert config.connect_timeout_s == 10.0
        assert config.query_timeout_s is None
        assert config.blob_upload_timeout_s == 300.0
        assert config.blob_download_timeout_s == 300.0

    def test_custom_values(self) -> None:
        """Test custom timeout values."""
        config = TimeoutConfig(
            default_timeout_s=60.0,
            connect_timeout_s=5.0,
            query_timeout_s=120.0,
        )
        assert config.default_timeout_s == 60.0
        assert config.connect_timeout_s == 5.0
        assert config.query_timeout_s == 120.0


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self) -> None:
        """Test default retry values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_backoff_ms == 100
        assert config.max_backoff_ms == 10000
        assert config.backoff_multiplier == 2.0
        assert 14 in config.retryable_status_codes  # UNAVAILABLE
        assert 4 in config.retryable_status_codes  # DEADLINE_EXCEEDED

    def test_single_attempt(self) -> None:
        """Test single attempt (no retry)."""
        config = RetryConfig(max_attempts=1)
        assert config.max_attempts == 1

    def test_custom_backoff(self) -> None:
        """Test custom backoff configuration."""
        config = RetryConfig(
            initial_backoff_ms=50,
            max_backoff_ms=5000,
            backoff_multiplier=1.5,
        )
        assert config.initial_backoff_ms == 50
        assert config.max_backoff_ms == 5000
        assert config.backoff_multiplier == 1.5


class TestKeepaliveConfig:
    """Tests for KeepaliveConfig."""

    def test_default_values(self) -> None:
        """Test default keepalive values."""
        config = KeepaliveConfig()
        assert config.time_ms == 30000
        assert config.timeout_ms == 10000
        assert config.permit_without_calls is True

    def test_custom_values(self) -> None:
        """Test custom keepalive values."""
        config = KeepaliveConfig(
            time_ms=60000,
            timeout_ms=20000,
            permit_without_calls=False,
        )
        assert config.time_ms == 60000
        assert config.timeout_ms == 20000
        assert config.permit_without_calls is False


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_factory(self) -> None:
        """Test default configuration factory."""
        config = ClientConfig.default()
        assert config.timeout.default_timeout_s == 30.0
        assert config.retry.max_attempts == 3
        assert config.keepalive.time_ms == 30000

    def test_no_retry_factory(self) -> None:
        """Test no-retry configuration factory."""
        config = ClientConfig.no_retry()
        assert config.retry.max_attempts == 1
        assert config.timeout.default_timeout_s == 30.0

    def test_fast_fail_factory(self) -> None:
        """Test fast-fail configuration factory."""
        config = ClientConfig.fast_fail()
        assert config.timeout.default_timeout_s == 5.0
        assert config.timeout.connect_timeout_s == 2.0
        assert config.retry.max_attempts == 1
        assert config.keepalive.time_ms == 10000

    def test_high_latency_factory(self) -> None:
        """Test high-latency configuration factory."""
        config = ClientConfig.high_latency()
        assert config.timeout.default_timeout_s == 120.0
        assert config.timeout.connect_timeout_s == 30.0
        assert config.retry.max_attempts == 5
        assert config.retry.initial_backoff_ms == 500
        assert config.keepalive.time_ms == 60000

    def test_custom_composition(self) -> None:
        """Test custom configuration composition."""
        config = ClientConfig(
            timeout=TimeoutConfig(default_timeout_s=45.0),
            retry=RetryConfig(max_attempts=4),
            keepalive=KeepaliveConfig(time_ms=20000),
        )
        assert config.timeout.default_timeout_s == 45.0
        assert config.retry.max_attempts == 4
        assert config.keepalive.time_ms == 20000

    def test_dataclass_immutability(self) -> None:
        """Test that configs are independent after creation."""
        config1 = ClientConfig.default()
        config2 = ClientConfig.default()
        # Modifying nested objects should not affect other instances
        assert config1.timeout is not config2.timeout
        assert config1.retry is not config2.retry
        assert config1.keepalive is not config2.keepalive
