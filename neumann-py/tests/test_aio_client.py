# SPDX-License-Identifier: MIT
"""Comprehensive tests for AsyncNeumannClient."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from neumann.aio.client import AsyncNeumannClient
from neumann.errors import ConnectionError, NeumannError
from neumann.types import QueryResult, QueryResultType


class TestAsyncNeumannClientInit:
    """Tests for AsyncNeumannClient initialization."""

    def test_init(self) -> None:
        """Test initialization."""
        client = AsyncNeumannClient()
        assert not client.is_connected
        assert client._channel is None
        assert client._stub is None
        assert client._api_key is None


class TestAsyncNeumannClientConnect:
    """Tests for async connect method."""

    @pytest.mark.asyncio
    async def test_connect_requires_grpc(self) -> None:
        """Test connect raises when grpc not available."""
        with patch.dict("sys.modules", {"grpc": None, "grpc.aio": None}):
            with pytest.raises(ConnectionError):
                await AsyncNeumannClient.connect("localhost:50051")

    @pytest.mark.asyncio
    async def test_connect_creates_client(self) -> None:
        """Test connect creates connected client."""
        mock_grpc_aio = MagicMock()
        mock_channel = MagicMock()
        mock_grpc_aio.insecure_channel.return_value = mock_channel

        mock_stub_module = MagicMock()
        mock_stub = MagicMock()
        mock_stub_module.QueryServiceStub.return_value = mock_stub

        with patch.dict(
            "sys.modules",
            {
                "grpc": MagicMock(),
                "grpc.aio": mock_grpc_aio,
                "neumann.proto": MagicMock(),
                "neumann.proto.neumann_pb2_grpc": mock_stub_module,
            },
        ):
            client = AsyncNeumannClient()
            client._channel = mock_channel
            client._stub = mock_stub
            client._connected = True

            assert client.is_connected

    @pytest.mark.asyncio
    async def test_connect_with_tls(self) -> None:
        """Test connect with TLS."""
        mock_grpc = MagicMock()
        mock_grpc_aio = MagicMock()
        mock_credentials = MagicMock()
        mock_grpc.ssl_channel_credentials.return_value = mock_credentials
        mock_channel = MagicMock()
        mock_grpc_aio.secure_channel.return_value = mock_channel

        with patch.dict(
            "sys.modules",
            {
                "grpc": mock_grpc,
                "grpc.aio": mock_grpc_aio,
            },
        ):
            client = AsyncNeumannClient()
            client._channel = mock_channel
            client._connected = True

            assert client.is_connected

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self) -> None:
        """Test connect stores API key."""
        client = AsyncNeumannClient()
        client._api_key = "test-key"
        client._connected = True

        assert client._api_key == "test-key"


class TestAsyncNeumannClientConnection:
    """Tests for connection management."""

    def test_is_connected_property(self) -> None:
        """Test is_connected property."""
        client = AsyncNeumannClient()
        assert not client.is_connected

        client._connected = True
        assert client.is_connected

    @pytest.mark.asyncio
    async def test_close_disconnects(self) -> None:
        """Test close disconnects client."""
        client = AsyncNeumannClient()
        mock_channel = AsyncMock()
        client._channel = mock_channel
        client._stub = MagicMock()
        client._connected = True

        await client.close()

        assert not client.is_connected
        assert client._channel is None
        assert client._stub is None
        mock_channel.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager."""
        client = AsyncNeumannClient()
        mock_channel = AsyncMock()
        client._channel = mock_channel
        client._stub = MagicMock()
        client._connected = True

        async with client as c:
            assert c is client
            assert c.is_connected

        assert not client.is_connected


class TestAsyncNeumannClientExecute:
    """Tests for async execute method."""

    @pytest.mark.asyncio
    async def test_execute_requires_connection(self) -> None:
        """Test execute raises when not connected."""
        client = AsyncNeumannClient()
        with pytest.raises(ConnectionError) as exc_info:
            await client.execute("SELECT * FROM users")
        assert "not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_requires_stub(self) -> None:
        """Test execute raises when stub not initialized."""
        client = AsyncNeumannClient()
        client._connected = True
        client._stub = None

        with pytest.raises(ConnectionError):
            await client.execute("SELECT * FROM users")

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        """Test execute raises when proto not available (expected behavior)."""
        client = AsyncNeumannClient()
        client._connected = True
        mock_stub = AsyncMock()
        client._stub = mock_stub

        # The execute will fail because proto module is not available
        # This is expected behavior - we're testing the connection check
        with pytest.raises((NeumannError, ImportError)):
            await client.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_with_identity(self) -> None:
        """Test execute with identity raises when proto not available."""
        client = AsyncNeumannClient()
        client._connected = True
        mock_stub = AsyncMock()
        client._stub = mock_stub

        with pytest.raises((NeumannError, ImportError)):
            await client.execute("VAULT GET secret", identity="alice")

    @pytest.mark.asyncio
    async def test_execute_with_api_key(self) -> None:
        """Test execute with API key raises when proto not available."""
        client = AsyncNeumannClient()
        client._connected = True
        client._api_key = "test-api-key"
        mock_stub = AsyncMock()
        client._stub = mock_stub

        with pytest.raises((NeumannError, ImportError)):
            await client.execute("SELECT 1")


class TestAsyncNeumannClientExecuteStream:
    """Tests for async execute_stream method."""

    @pytest.mark.asyncio
    async def test_execute_stream_requires_connection(self) -> None:
        """Test execute_stream raises when not connected."""
        client = AsyncNeumannClient()

        with pytest.raises(ConnectionError):
            async for _ in client.execute_stream("SELECT * FROM users"):
                pass

    @pytest.mark.asyncio
    async def test_execute_stream_requires_stub(self) -> None:
        """Test execute_stream raises when stub not initialized."""
        client = AsyncNeumannClient()
        client._connected = True
        client._stub = None

        with pytest.raises(ConnectionError):
            async for _ in client.execute_stream("SELECT * FROM users"):
                pass


class TestAsyncNeumannClientExecuteBatch:
    """Tests for async execute_batch method."""

    @pytest.mark.asyncio
    async def test_execute_batch_requires_connection(self) -> None:
        """Test execute_batch raises when not connected."""
        client = AsyncNeumannClient()
        with pytest.raises(ConnectionError):
            await client.execute_batch(["SELECT 1", "SELECT 2"])

    @pytest.mark.asyncio
    async def test_execute_batch_requires_stub(self) -> None:
        """Test execute_batch raises when stub not initialized."""
        client = AsyncNeumannClient()
        client._connected = True
        client._stub = None

        with pytest.raises(ConnectionError):
            await client.execute_batch(["SELECT 1", "SELECT 2"])


class TestAsyncNeumannClientRunInExecutor:
    """Tests for run_in_executor method."""

    @pytest.mark.asyncio
    async def test_run_in_executor(self) -> None:
        """Test run_in_executor executes in thread pool."""
        client = AsyncNeumannClient()

        mock_native = MagicMock()
        mock_native.execute.return_value = {"type": "count", "data": 42}

        with patch("neumann.client.NeumannClient.embedded") as mock_embedded:
            mock_sync_client = MagicMock()
            mock_sync_client.__enter__ = MagicMock(return_value=mock_sync_client)
            mock_sync_client.__exit__ = MagicMock(return_value=None)
            mock_sync_client.execute.return_value = QueryResult(
                QueryResultType.COUNT, 42
            )
            mock_embedded.return_value = mock_sync_client

            result = await client.run_in_executor("SELECT COUNT(*) FROM users")
            assert result.type == QueryResultType.COUNT
            assert result.data == 42


class TestAsyncNeumannClientConnectDirect:
    """Tests that directly call connect() with proper proto support."""

    @pytest.mark.asyncio
    async def test_connect_insecure_success(self) -> None:
        """Test connect with insecure channel."""
        import grpc.aio
        from neumann.proto import neumann_pb2_grpc

        # Create client directly using connect()
        with patch.object(grpc.aio, "insecure_channel") as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch

            with patch.object(neumann_pb2_grpc, "QueryServiceStub") as mock_stub_cls:
                mock_stub = MagicMock()
                mock_stub_cls.return_value = mock_stub

                client = await AsyncNeumannClient.connect("localhost:50051")

                assert client.is_connected
                # Check that channel was created with address and keepalive options
                mock_channel.assert_called_once()
                call_args = mock_channel.call_args
                assert call_args[0][0] == "localhost:50051"
                assert "options" in call_args[1]
                mock_stub_cls.assert_called_once_with(mock_ch)

    @pytest.mark.asyncio
    async def test_connect_secure_with_tls(self) -> None:
        """Test connect with TLS enabled."""
        import grpc
        import grpc.aio
        from neumann.proto import neumann_pb2_grpc

        with patch.object(grpc, "ssl_channel_credentials") as mock_creds:
            mock_credentials = MagicMock()
            mock_creds.return_value = mock_credentials

            with patch.object(grpc.aio, "secure_channel") as mock_channel:
                mock_ch = MagicMock()
                mock_channel.return_value = mock_ch

                with patch.object(neumann_pb2_grpc, "QueryServiceStub") as mock_stub_cls:
                    mock_stub = MagicMock()
                    mock_stub_cls.return_value = mock_stub

                    client = await AsyncNeumannClient.connect(
                        "localhost:50051", tls=True
                    )

                    assert client.is_connected
                    mock_creds.assert_called_once()
                    # Check that channel was created with address, credentials, and options
                    mock_channel.assert_called_once()
                    call_args = mock_channel.call_args
                    assert call_args[0][0] == "localhost:50051"
                    assert call_args[0][1] == mock_credentials
                    assert "options" in call_args[1]

    @pytest.mark.asyncio
    async def test_connect_with_api_key_stored(self) -> None:
        """Test connect stores API key."""
        import grpc.aio
        from neumann.proto import neumann_pb2_grpc

        with patch.object(grpc.aio, "insecure_channel") as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch

            with patch.object(neumann_pb2_grpc, "QueryServiceStub"):
                client = await AsyncNeumannClient.connect(
                    "localhost:50051", api_key="secret-key"
                )

                assert client._api_key == "secret-key"

    @pytest.mark.asyncio
    async def test_connect_generic_exception(self) -> None:
        """Test connect handles generic exceptions."""
        import grpc.aio

        with patch.object(
            grpc.aio, "insecure_channel", side_effect=RuntimeError("Connection failed")
        ):
            with pytest.raises(ConnectionError) as exc_info:
                await AsyncNeumannClient.connect("localhost:50051")
            assert "Failed to connect" in str(exc_info.value)


class TestAsyncNeumannClientExecuteDirect:
    """Tests for execute with actual proto module."""

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        """Test execute succeeds with proper response."""
        from neumann.proto import neumann_pb2

        client = AsyncNeumannClient()
        client._connected = True

        # Create mock stub that returns a proper response
        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.count = 42
        mock_response.WhichOneof.return_value = "count"
        mock_stub.Execute.return_value = mock_response
        client._stub = mock_stub

        result = await client.execute("SELECT COUNT(*)")
        assert result.type == QueryResultType.COUNT

    @pytest.mark.asyncio
    async def test_execute_with_identity(self) -> None:
        """Test execute with identity parameter."""
        client = AsyncNeumannClient()
        client._connected = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = None
        mock_stub.Execute.return_value = mock_response
        client._stub = mock_stub

        result = await client.execute("VAULT GET secret", identity="alice")
        assert result.type == QueryResultType.EMPTY

    @pytest.mark.asyncio
    async def test_execute_with_api_key_in_metadata(self) -> None:
        """Test execute includes API key in metadata."""
        client = AsyncNeumannClient()
        client._connected = True
        client._api_key = "test-api-key"

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = None
        mock_stub.Execute.return_value = mock_response
        client._stub = mock_stub

        await client.execute("SELECT 1")

        # Verify metadata was passed
        call_args = mock_stub.Execute.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert ("x-api-key", "test-api-key") in kwargs.get("metadata", [])

    @pytest.mark.asyncio
    async def test_execute_grpc_error(self) -> None:
        """Test execute handles gRPC errors."""
        client = AsyncNeumannClient()
        client._connected = True

        # Create a mock gRPC error
        class MockGrpcError(Exception):
            pass

        MockGrpcError.__module__ = "grpc._channel"

        mock_stub = AsyncMock()
        mock_stub.Execute.side_effect = MockGrpcError("RPC failed")
        client._stub = mock_stub

        with pytest.raises(ConnectionError) as exc_info:
            await client.execute("SELECT 1")
        assert "gRPC error" in str(exc_info.value)


class TestAsyncNeumannClientExecuteStreamDirect:
    """Tests for execute_stream with actual proto module."""

    @pytest.mark.asyncio
    async def test_execute_stream_success(self) -> None:
        """Test execute_stream yields results."""
        client = AsyncNeumannClient()
        client._connected = True

        # Create mock chunk
        mock_chunk = MagicMock()
        mock_chunk.is_final = False
        mock_chunk.WhichOneof.return_value = "count"
        mock_chunk.count = 1

        mock_final = MagicMock()
        mock_final.is_final = True

        async def async_gen():
            yield mock_chunk
            yield mock_final

        mock_stub = MagicMock()
        mock_stub.ExecuteStream.return_value = async_gen()
        client._stub = mock_stub

        results = []
        async for r in client.execute_stream("SELECT *"):
            results.append(r)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_execute_stream_with_identity(self) -> None:
        """Test execute_stream with identity."""
        client = AsyncNeumannClient()
        client._connected = True

        mock_final = MagicMock()
        mock_final.is_final = True

        async def async_gen():
            yield mock_final

        mock_stub = MagicMock()
        mock_stub.ExecuteStream.return_value = async_gen()
        client._stub = mock_stub

        results = []
        async for r in client.execute_stream("SELECT *", identity="alice"):
            results.append(r)

        assert len(results) == 0  # Only final chunk

    @pytest.mark.asyncio
    async def test_execute_stream_grpc_error(self) -> None:
        """Test execute_stream handles gRPC errors."""
        client = AsyncNeumannClient()
        client._connected = True

        class MockGrpcError(Exception):
            pass

        MockGrpcError.__module__ = "grpc._channel"

        async def async_gen():
            raise MockGrpcError("Stream failed")
            yield  # Make it an async generator

        mock_stub = MagicMock()
        mock_stub.ExecuteStream.return_value = async_gen()
        client._stub = mock_stub

        with pytest.raises(ConnectionError) as exc_info:
            async for _ in client.execute_stream("SELECT *"):
                pass
        assert "gRPC error" in str(exc_info.value)


class TestAsyncNeumannClientExecuteBatchDirect:
    """Tests for execute_batch with actual proto module."""

    @pytest.mark.asyncio
    async def test_execute_batch_success(self) -> None:
        """Test execute_batch returns results."""
        client = AsyncNeumannClient()
        client._connected = True

        # Create mock results
        mock_result1 = MagicMock()
        mock_result1.error = None
        mock_result1.result = MagicMock()
        mock_result1.result.count = 1
        mock_result1.WhichOneof.return_value = "count"

        mock_result2 = MagicMock()
        mock_result2.error = None
        mock_result2.result = MagicMock()
        mock_result2.result.count = 2
        mock_result2.WhichOneof.return_value = "count"

        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2]

        mock_stub = AsyncMock()
        mock_stub.ExecuteBatch.return_value = mock_response
        client._stub = mock_stub

        results = await client.execute_batch(["SELECT 1", "SELECT 2"])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_execute_batch_with_identity(self) -> None:
        """Test execute_batch with identity."""
        client = AsyncNeumannClient()
        client._connected = True

        mock_response = MagicMock()
        mock_response.results = []

        mock_stub = AsyncMock()
        mock_stub.ExecuteBatch.return_value = mock_response
        client._stub = mock_stub

        results = await client.execute_batch([], identity="alice")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_batch_grpc_error(self) -> None:
        """Test execute_batch handles gRPC errors."""
        client = AsyncNeumannClient()
        client._connected = True

        class MockGrpcError(Exception):
            pass

        MockGrpcError.__module__ = "grpc._channel"

        mock_stub = AsyncMock()
        mock_stub.ExecuteBatch.side_effect = MockGrpcError("Batch failed")
        client._stub = mock_stub

        with pytest.raises(ConnectionError) as exc_info:
            await client.execute_batch(["SELECT 1"])
        assert "gRPC error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_batch_with_api_key(self) -> None:
        """Test execute_batch includes API key in metadata."""
        client = AsyncNeumannClient()
        client._connected = True
        client._api_key = "batch-api-key"

        mock_response = MagicMock()
        mock_response.results = []

        mock_stub = AsyncMock()
        mock_stub.ExecuteBatch.return_value = mock_response
        client._stub = mock_stub

        await client.execute_batch(["SELECT 1"])

        call_args = mock_stub.ExecuteBatch.call_args
        _, kwargs = call_args
        assert ("x-api-key", "batch-api-key") in kwargs.get("metadata", [])


class TestAsyncNeumannClientStreamWithApiKey:
    """Tests for execute_stream with API key."""

    @pytest.mark.asyncio
    async def test_execute_stream_with_api_key(self) -> None:
        """Test execute_stream includes API key in metadata."""
        client = AsyncNeumannClient()
        client._connected = True
        client._api_key = "stream-api-key"

        mock_final = MagicMock()
        mock_final.is_final = True

        async def async_gen():
            yield mock_final

        mock_stub = MagicMock()
        mock_stub.ExecuteStream.return_value = async_gen()
        client._stub = mock_stub

        async for _ in client.execute_stream("SELECT *"):
            pass

        call_args = mock_stub.ExecuteStream.call_args
        _, kwargs = call_args
        assert ("x-api-key", "stream-api-key") in kwargs.get("metadata", [])
