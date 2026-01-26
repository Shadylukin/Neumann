"""Async client for Neumann database."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

from neumann.config import ClientConfig
from neumann.errors import ConnectionError, NeumannError
from neumann.retry import retry_call_async
from neumann.types import QueryResult

if TYPE_CHECKING:
    from types import TracebackType


class AsyncNeumannClient:
    """Async client for Neumann database supporting remote mode via gRPC.

    Note: Embedded mode is not supported in async client due to PyO3 limitations.
    Use the synchronous NeumannClient for embedded mode.
    """

    def __init__(self) -> None:
        """Initialize async client (internal use - use class methods)."""
        self._channel: object | None = None
        self._stub: object | None = None
        self._api_key: str | None = None
        self._config: ClientConfig = ClientConfig.default()
        self._connected = False

    @classmethod
    async def connect(
        cls,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
        config: ClientConfig | None = None,
    ) -> AsyncNeumannClient:
        """Connect to a remote Neumann server via async gRPC.

        Args:
            address: Server address in format "host:port".
            api_key: Optional API key for authentication.
            tls: Whether to use TLS encryption.
            config: Optional client configuration for timeouts, retries, and keepalive.

        Returns:
            A connected AsyncNeumannClient.

        Raises:
            ConnectionError: If connection fails.
        """
        client = cls()
        client._api_key = api_key
        client._config = config or ClientConfig.default()

        try:
            import grpc.aio

            # Channel options with keepalive
            options = [
                ("grpc.keepalive_time_ms", client._config.keepalive.time_ms),
                ("grpc.keepalive_timeout_ms", client._config.keepalive.timeout_ms),
                (
                    "grpc.keepalive_permit_without_calls",
                    1 if client._config.keepalive.permit_without_calls else 0,
                ),
                ("grpc.http2.min_time_between_pings_ms", client._config.keepalive.time_ms),
            ]

            if tls:
                credentials = grpc.ssl_channel_credentials()
                client._channel = grpc.aio.secure_channel(address, credentials, options=options)
            else:
                client._channel = grpc.aio.insecure_channel(address, options=options)

            from neumann.proto import neumann_pb2_grpc

            client._stub = neumann_pb2_grpc.QueryServiceStub(client._channel)
            client._connected = True
        except ImportError as e:
            raise ConnectionError(
                "gRPC not available. Install with: pip install grpcio"
            ) from e
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {address}: {e}") from e

        return client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def close(self) -> None:
        """Close the client connection."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
        self._stub = None
        self._connected = False

    async def __aenter__(self) -> AsyncNeumannClient:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def execute(self, query: str, *, identity: str | None = None) -> QueryResult:
        """Execute a query and return the result.

        Args:
            query: The Neumann query to execute.
            identity: Optional identity for vault access.

        Returns:
            QueryResult containing the query results.

        Raises:
            NeumannError: If query execution fails.
        """
        if not self._connected or self._stub is None:
            raise ConnectionError("Client is not connected")

        try:
            from neumann.proto import neumann_pb2

            request = neumann_pb2.QueryRequest(query=query)
            if identity:
                request.identity = identity

            metadata = []
            if self._api_key:
                metadata.append(("x-api-key", self._api_key))

            timeout = (
                self._config.timeout.query_timeout_s
                or self._config.timeout.default_timeout_s
            )

            async def do_execute() -> object:
                return await self._stub.Execute(
                    request, timeout=timeout, metadata=metadata or None
                )

            response = await retry_call_async(do_execute, self._config.retry)

            # Import conversion utilities from sync client
            from neumann.client import NeumannClient

            converter = NeumannClient.__new__(NeumannClient)
            return converter._convert_proto_result(response)
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    async def execute_stream(
        self, query: str, *, identity: str | None = None
    ) -> AsyncIterator[QueryResult]:
        """Execute a streaming query.

        Args:
            query: The Neumann query to execute.
            identity: Optional identity for vault access.

        Yields:
            QueryResult items from the stream.

        Raises:
            NeumannError: If query execution fails.
        """
        if not self._connected or self._stub is None:
            raise ConnectionError("Client is not connected")

        try:
            from neumann.proto import neumann_pb2

            request = neumann_pb2.QueryRequest(query=query)
            if identity:
                request.identity = identity

            metadata = []
            if self._api_key:
                metadata.append(("x-api-key", self._api_key))

            timeout = (
                self._config.timeout.query_timeout_s
                or self._config.timeout.default_timeout_s
            )

            # Import conversion utilities
            from neumann.client import NeumannClient

            converter = NeumannClient.__new__(NeumannClient)

            stream = self._stub.ExecuteStream(
                request, timeout=timeout, metadata=metadata or None
            )
            async for chunk in stream:
                if chunk.is_final:
                    break
                yield converter._convert_proto_chunk(chunk)
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    async def execute_batch(
        self, queries: list[str], *, identity: str | None = None
    ) -> list[QueryResult]:
        """Execute multiple queries in a batch.

        Args:
            queries: List of queries to execute.
            identity: Optional identity for vault access.

        Returns:
            List of QueryResults, one per query.

        Raises:
            NeumannError: If batch execution fails.
        """
        if not self._connected or self._stub is None:
            raise ConnectionError("Client is not connected")

        try:
            from neumann.proto import neumann_pb2

            query_requests = []
            for q in queries:
                req = neumann_pb2.QueryRequest(query=q)
                if identity:
                    req.identity = identity
                query_requests.append(req)

            request = neumann_pb2.BatchQueryRequest(queries=query_requests)

            metadata = []
            if self._api_key:
                metadata.append(("x-api-key", self._api_key))

            timeout = (
                self._config.timeout.query_timeout_s
                or self._config.timeout.default_timeout_s
            )

            # Import conversion utilities
            from neumann.client import NeumannClient

            converter = NeumannClient.__new__(NeumannClient)

            async def do_batch() -> object:
                return await self._stub.ExecuteBatch(
                    request, timeout=timeout, metadata=metadata or None
                )

            response = await retry_call_async(do_batch, self._config.retry)
            return [converter._convert_proto_result(r) for r in response.results]
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    async def run_in_executor(
        self, query: str, *, identity: str | None = None
    ) -> QueryResult:
        """Run a query using a thread pool executor.

        This is useful when you need to use the synchronous embedded client
        from async code.

        Args:
            query: The Neumann query to execute.
            identity: Optional identity for vault access.

        Returns:
            QueryResult containing the query results.
        """
        from neumann.client import NeumannClient

        # Create a sync client for embedded mode
        def run_sync() -> QueryResult:
            with NeumannClient.embedded() as client:
                return client.execute(query, identity=identity)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_sync)
