# SPDX-License-Identifier: MIT
"""BlobService client for artifact storage operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from neumann.config import ClientConfig, RetryConfig
from neumann.errors import (
    ConnectionError,
)
from neumann.retry import retry_call

if TYPE_CHECKING:
    from types import TracebackType


@dataclass
class BlobUploadOptions:
    """Options for blob upload."""

    content_type: str | None = None
    created_by: str | None = None
    tags: list[str] = field(default_factory=list)
    linked_to: list[str] = field(default_factory=list)
    custom: dict[str, str] = field(default_factory=dict)


@dataclass
class BlobUploadResult:
    """Result of a blob upload operation."""

    artifact_id: str
    size: int
    checksum: str


@dataclass
class ArtifactMetadata:
    """Artifact metadata."""

    id: str
    filename: str
    content_type: str
    size: int
    checksum: str
    chunk_count: int
    created: datetime
    modified: datetime
    created_by: str
    tags: list[str]
    linked_to: list[str]
    custom: dict[str, str]


class BlobClient:
    """Service client for blob/artifact operations."""

    CHUNK_SIZE = 64 * 1024  # 64KB chunks

    def __init__(
        self,
        stub: Any,
        metadata: list[tuple[str, str]] | None = None,
        retry_config: RetryConfig | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self._stub: Any = stub
        self._metadata: list[tuple[str, str]] = metadata or []
        self._retry_config = retry_config or RetryConfig()
        self._timeout_s = timeout_s

    def upload_blob(
        self,
        filename: str,
        data: bytes,
        options: BlobUploadOptions | None = None,
    ) -> BlobUploadResult:
        """Upload a blob from bytes.

        Args:
            filename: The filename for the artifact.
            data: The blob data as bytes.
            options: Upload options.

        Returns:
            Upload result with artifact ID.
        """
        options = options or BlobUploadOptions()

        def generate_requests() -> Iterator[Any]:
            from neumann.proto import neumann_pb2

            # Send metadata first
            meta = neumann_pb2.BlobUploadMetadata(
                filename=filename,
                tags=options.tags,
                linked_to=options.linked_to,
                custom=options.custom,
            )
            if options.content_type:
                meta.content_type = options.content_type
            if options.created_by:
                meta.created_by = options.created_by

            yield neumann_pb2.BlobUploadRequest(metadata=meta)

            # Send data in chunks
            for offset in range(0, len(data), self.CHUNK_SIZE):
                chunk = data[offset : offset + self.CHUNK_SIZE]
                yield neumann_pb2.BlobUploadRequest(chunk=chunk)

        def do_upload() -> Any:
            return self._stub.Upload(
                generate_requests(),
                timeout=self._timeout_s,
                metadata=self._metadata or None,
            )

        response = retry_call(do_upload, self._retry_config)
        return BlobUploadResult(
            artifact_id=str(response.artifact_id),
            size=int(response.size),
            checksum=str(response.checksum),
        )

    def upload_blob_streaming(
        self,
        filename: str,
        chunks: Iterator[bytes],
        options: BlobUploadOptions | None = None,
    ) -> BlobUploadResult:
        """Upload a blob from an iterator of chunks.

        Args:
            filename: The filename for the artifact.
            chunks: Iterator of data chunks.
            options: Upload options.

        Returns:
            Upload result with artifact ID.
        """
        options = options or BlobUploadOptions()

        def generate_requests() -> Iterator[Any]:
            from neumann.proto import neumann_pb2

            # Send metadata first
            meta = neumann_pb2.BlobUploadMetadata(
                filename=filename,
                tags=options.tags,
                linked_to=options.linked_to,
                custom=options.custom,
            )
            if options.content_type:
                meta.content_type = options.content_type
            if options.created_by:
                meta.created_by = options.created_by

            yield neumann_pb2.BlobUploadRequest(metadata=meta)

            # Stream chunks
            for chunk in chunks:
                yield neumann_pb2.BlobUploadRequest(chunk=chunk)

        def do_upload() -> Any:
            return self._stub.Upload(
                generate_requests(),
                timeout=self._timeout_s,
                metadata=self._metadata or None,
            )

        response = retry_call(do_upload, self._retry_config)
        return BlobUploadResult(
            artifact_id=str(response.artifact_id),
            size=int(response.size),
            checksum=str(response.checksum),
        )

    def download_blob(self, artifact_id: str) -> Iterator[bytes]:
        """Download a blob as an iterator of chunks.

        Args:
            artifact_id: The artifact ID to download.

        Yields:
            Data chunks.
        """
        from neumann.proto import neumann_pb2

        request = neumann_pb2.BlobDownloadRequest(artifact_id=artifact_id)
        stream = self._stub.Download(
            request, timeout=self._timeout_s, metadata=self._metadata or None
        )

        for chunk in stream:
            if chunk.data:
                yield bytes(chunk.data)
            if chunk.is_final:
                break

    def download_blob_full(self, artifact_id: str) -> bytes:
        """Download a blob as complete bytes.

        Args:
            artifact_id: The artifact ID to download.

        Returns:
            The complete blob data.
        """
        chunks = []
        for chunk in self.download_blob(artifact_id):
            chunks.append(chunk)
        return b"".join(chunks)

    def delete_blob(self, artifact_id: str) -> bool:
        """Delete a blob.

        Args:
            artifact_id: The artifact ID to delete.

        Returns:
            True if deletion was successful.
        """
        from neumann.proto import neumann_pb2

        request = neumann_pb2.BlobDeleteRequest(artifact_id=artifact_id)

        def do_delete() -> Any:
            return self._stub.Delete(request, metadata=self._metadata or None)

        response = retry_call(do_delete, self._retry_config)
        return cast(bool, response.success)

    def get_blob_metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Get blob metadata.

        Args:
            artifact_id: The artifact ID.

        Returns:
            Artifact metadata.
        """
        from neumann.proto import neumann_pb2

        request = neumann_pb2.BlobMetadataRequest(artifact_id=artifact_id)

        def do_get() -> Any:
            return self._stub.GetMetadata(request, metadata=self._metadata or None)

        response = retry_call(do_get, self._retry_config)
        return ArtifactMetadata(
            id=str(response.id),
            filename=str(response.filename),
            content_type=str(response.content_type),
            size=int(response.size),
            checksum=str(response.checksum),
            chunk_count=int(response.chunk_count),
            created=datetime.fromtimestamp(float(response.created) / 1000.0),
            modified=datetime.fromtimestamp(float(response.modified) / 1000.0),
            created_by=str(response.created_by),
            tags=[str(t) for t in response.tags],
            linked_to=[str(link) for link in response.linked_to],
            custom={str(k): str(v) for k, v in response.custom.items()} if response.custom else {},
        )


class BlobServiceClient:
    """High-level blob client for artifact storage."""

    def __init__(
        self,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
        config: ClientConfig | None = None,
    ) -> None:
        """Initialize blob client (internal use - use connect())."""
        self._address = address
        self._api_key = api_key
        self._tls = tls
        self._config = config or ClientConfig.default()
        self._channel: Any = None
        self._blob_stub: Any = None
        self._blob: BlobClient | None = None
        self._connected = False

    @classmethod
    def connect(
        cls,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
        config: ClientConfig | None = None,
    ) -> BlobServiceClient:
        """Connect to a remote Neumann server's blob service.

        Args:
            address: Server address in format "host:port".
            api_key: Optional API key for authentication.
            tls: Whether to use TLS encryption.
            config: Optional client configuration.

        Returns:
            A connected BlobServiceClient.

        Raises:
            ConnectionError: If connection fails.
        """
        client = cls(address, api_key=api_key, tls=tls, config=config)

        try:
            import grpc

            options = [
                ("grpc.keepalive_time_ms", client._config.keepalive.time_ms),
                ("grpc.keepalive_timeout_ms", client._config.keepalive.timeout_ms),
                (
                    "grpc.keepalive_permit_without_calls",
                    1 if client._config.keepalive.permit_without_calls else 0,
                ),
            ]

            if tls:
                credentials = grpc.ssl_channel_credentials()
                client._channel = grpc.secure_channel(address, credentials, options=options)
            else:
                client._channel = grpc.insecure_channel(address, options=options)

            from neumann.proto import neumann_pb2_grpc

            client._blob_stub = neumann_pb2_grpc.BlobServiceStub(client._channel)

            metadata = []
            if api_key:
                metadata.append(("x-api-key", api_key))

            timeout_s = client._config.timeout.blob_upload_timeout_s or 300.0
            client._blob = BlobClient(client._blob_stub, metadata, client._config.retry, timeout_s)
            client._connected = True

        except ImportError as e:
            raise ConnectionError("gRPC not available. Install with: pip install grpcio") from e
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {address}: {e}") from e

        return client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def close(self) -> None:
        """Close the client connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        self._blob = None
        self._connected = False

    def __enter__(self) -> BlobServiceClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def _assert_connected(self) -> BlobClient:
        """Assert that the client is connected."""
        if self._blob is None:
            raise ConnectionError("Client is not connected")
        return self._blob

    # Convenience methods that delegate to blob client

    def upload_blob(
        self,
        filename: str,
        data: bytes,
        options: BlobUploadOptions | None = None,
    ) -> BlobUploadResult:
        """Upload a blob from bytes."""
        return self._assert_connected().upload_blob(filename, data, options)

    def upload_blob_streaming(
        self,
        filename: str,
        chunks: Iterator[bytes],
        options: BlobUploadOptions | None = None,
    ) -> BlobUploadResult:
        """Upload a blob from an iterator of chunks."""
        return self._assert_connected().upload_blob_streaming(filename, chunks, options)

    def download_blob(self, artifact_id: str) -> Iterator[bytes]:
        """Download a blob as an iterator of chunks."""
        return self._assert_connected().download_blob(artifact_id)

    def download_blob_full(self, artifact_id: str) -> bytes:
        """Download a blob as complete bytes."""
        return self._assert_connected().download_blob_full(artifact_id)

    def delete_blob(self, artifact_id: str) -> bool:
        """Delete a blob."""
        return self._assert_connected().delete_blob(artifact_id)

    def get_blob_metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Get blob metadata."""
        return self._assert_connected().get_blob_metadata(artifact_id)
