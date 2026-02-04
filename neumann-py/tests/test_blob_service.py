# SPDX-License-Identifier: MIT
"""Tests for blob service clients."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from neumann.config import RetryConfig
from neumann.services.blob import (
    ArtifactMetadata,
    BlobClient,
    BlobServiceClient,
    BlobUploadOptions,
    BlobUploadResult,
)


class TestBlobUploadOptions:
    """Tests for BlobUploadOptions dataclass."""

    def test_default_values(self) -> None:
        """Test default option values."""
        options = BlobUploadOptions()
        assert options.content_type is None
        assert options.created_by is None
        assert options.tags == []
        assert options.linked_to == []
        assert options.custom == {}

    def test_custom_values(self) -> None:
        """Test custom option values."""
        options = BlobUploadOptions(
            content_type="text/plain",
            created_by="user:alice",
            tags=["important", "document"],
            linked_to=["artifact-123"],
            custom={"version": "1.0"},
        )
        assert options.content_type == "text/plain"
        assert options.created_by == "user:alice"
        assert len(options.tags) == 2
        assert len(options.linked_to) == 1
        assert options.custom["version"] == "1.0"


class TestBlobUploadResult:
    """Tests for BlobUploadResult dataclass."""

    def test_create(self) -> None:
        """Test upload result creation."""
        result = BlobUploadResult(
            artifact_id="art-123",
            size=1024,
            checksum="abc123def456",
        )
        assert result.artifact_id == "art-123"
        assert result.size == 1024
        assert result.checksum == "abc123def456"


class TestArtifactMetadata:
    """Tests for ArtifactMetadata dataclass."""

    def test_create(self) -> None:
        """Test metadata creation."""
        now = datetime.now()
        metadata = ArtifactMetadata(
            id="art-123",
            filename="test.txt",
            content_type="text/plain",
            size=1024,
            checksum="abc123",
            chunk_count=1,
            created=now,
            modified=now,
            created_by="user:alice",
            tags=["test"],
            linked_to=[],
            custom={},
        )
        assert metadata.id == "art-123"
        assert metadata.filename == "test.txt"
        assert metadata.size == 1024
        assert metadata.created == now


class TestBlobClientDownload:
    """Tests for BlobClient download operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = BlobClient(
            self.mock_stub,
            metadata=[("x-api-key", "test")],
            retry_config=RetryConfig(),
        )

    def test_download_blob(self) -> None:
        """Test downloading a blob."""
        chunk1 = MagicMock()
        chunk1.data = b"hello "
        chunk1.is_final = False

        chunk2 = MagicMock()
        chunk2.data = b"world"
        chunk2.is_final = True

        self.mock_stub.Download.return_value = iter([chunk1, chunk2])

        chunks = list(self.client.download_blob("art-123"))

        assert len(chunks) == 2
        assert chunks[0] == b"hello "
        assert chunks[1] == b"world"

    def test_download_blob_full(self) -> None:
        """Test downloading a blob as complete bytes."""
        chunk1 = MagicMock()
        chunk1.data = b"hello "
        chunk1.is_final = False

        chunk2 = MagicMock()
        chunk2.data = b"world"
        chunk2.is_final = True

        self.mock_stub.Download.return_value = iter([chunk1, chunk2])

        result = self.client.download_blob_full("art-123")

        assert result == b"hello world"


class TestBlobServiceClientNotConnected:
    """Tests for BlobServiceClient error handling."""

    def test_assert_connected_raises(self) -> None:
        """Test that operations raise when not connected."""
        from neumann.errors import ConnectionError

        client = BlobServiceClient("localhost:50051")

        with pytest.raises(ConnectionError, match="not connected"):
            client.upload_blob("test.txt", b"data")


class TestBlobClientStreaming:
    """Tests for streaming functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = BlobClient(self.mock_stub)

    def test_upload_streaming_iterator(self) -> None:
        """Test uploading from an iterator."""
        mock_response = MagicMock()
        mock_response.artifact_id = "art-123"
        mock_response.size = 15
        mock_response.checksum = "abc123"
        self.mock_stub.Upload.return_value = mock_response

        chunks = iter([b"hello", b" ", b"world"])
        result = self.client.upload_blob_streaming("test.txt", chunks)

        assert result.artifact_id == "art-123"

    def test_download_handles_empty_chunks(self) -> None:
        """Test downloading handles empty data chunks."""
        chunk1 = MagicMock()
        chunk1.data = b""
        chunk1.is_final = False

        chunk2 = MagicMock()
        chunk2.data = b"data"
        chunk2.is_final = True

        self.mock_stub.Download.return_value = iter([chunk1, chunk2])

        chunks = list(self.client.download_blob("art-123"))

        # Empty chunks should be skipped
        assert len(chunks) == 1
        assert chunks[0] == b"data"


class TestBlobClientUpload:
    """Tests for BlobClient upload operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = BlobClient(self.mock_stub, timeout_s=60.0)

    def test_upload_blob_basic(self) -> None:
        """Test basic blob upload."""
        mock_response = MagicMock()
        mock_response.artifact_id = "art-123"
        mock_response.size = 1024
        mock_response.checksum = "sha256-abc"
        self.mock_stub.Upload.return_value = mock_response

        result = self.client.upload_blob("test.txt", b"Hello World")

        assert result.artifact_id == "art-123"
        assert result.size == 1024
        assert result.checksum == "sha256-abc"
        self.mock_stub.Upload.assert_called_once()

    def test_upload_blob_with_options(self) -> None:
        """Test upload with all options set."""
        mock_response = MagicMock()
        mock_response.artifact_id = "art-456"
        mock_response.size = 2048
        mock_response.checksum = "sha256-def"
        self.mock_stub.Upload.return_value = mock_response

        options = BlobUploadOptions(
            content_type="application/json",
            created_by="user:alice",
            tags=["important", "backup"],
            linked_to=["art-123"],
            custom={"version": "2.0"},
        )

        result = self.client.upload_blob("data.json", b'{"key": "value"}', options)

        assert result.artifact_id == "art-456"

    def test_upload_blob_large_data(self) -> None:
        """Test upload handles chunking for large data."""
        mock_response = MagicMock()
        mock_response.artifact_id = "art-big"
        mock_response.size = 128 * 1024
        mock_response.checksum = "sha256-big"
        self.mock_stub.Upload.return_value = mock_response

        # Create data larger than CHUNK_SIZE (64KB)
        large_data = b"x" * (128 * 1024)
        result = self.client.upload_blob("large.bin", large_data)

        assert result.size == 128 * 1024

    def test_upload_streaming_with_options(self) -> None:
        """Test streaming upload with options."""
        mock_response = MagicMock()
        mock_response.artifact_id = "art-stream"
        mock_response.size = 100
        mock_response.checksum = "sha256-stream"
        self.mock_stub.Upload.return_value = mock_response

        options = BlobUploadOptions(
            content_type="text/plain",
            created_by="user:bob",
        )

        chunks = iter([b"chunk1", b"chunk2"])
        result = self.client.upload_blob_streaming("stream.txt", chunks, options)

        assert result.artifact_id == "art-stream"


class TestBlobClientDelete:
    """Tests for BlobClient delete operation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = BlobClient(self.mock_stub)

    def test_delete_blob_success(self) -> None:
        """Test successful blob deletion."""
        mock_response = MagicMock()
        mock_response.success = True
        self.mock_stub.Delete.return_value = mock_response

        result = self.client.delete_blob("art-123")

        assert result is True
        self.mock_stub.Delete.assert_called_once()

    def test_delete_blob_not_found(self) -> None:
        """Test deletion when blob not found."""
        mock_response = MagicMock()
        mock_response.success = False
        self.mock_stub.Delete.return_value = mock_response

        result = self.client.delete_blob("nonexistent")

        assert result is False


class TestBlobClientMetadata:
    """Tests for BlobClient metadata operation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = BlobClient(self.mock_stub)

    def test_get_blob_metadata(self) -> None:
        """Test getting blob metadata."""
        mock_response = MagicMock()
        mock_response.id = "art-123"
        mock_response.filename = "test.txt"
        mock_response.content_type = "text/plain"
        mock_response.size = 1024
        mock_response.checksum = "sha256-abc"
        mock_response.chunk_count = 1
        mock_response.created = 1704067200000  # 2024-01-01 00:00:00 UTC
        mock_response.modified = 1704153600000  # 2024-01-02 00:00:00 UTC
        mock_response.created_by = "user:alice"
        mock_response.tags = ["important"]
        mock_response.linked_to = ["art-456"]
        mock_response.custom = {"version": "1.0"}
        self.mock_stub.GetMetadata.return_value = mock_response

        result = self.client.get_blob_metadata("art-123")

        assert result.id == "art-123"
        assert result.filename == "test.txt"
        assert result.content_type == "text/plain"
        assert result.size == 1024
        assert result.checksum == "sha256-abc"
        assert result.chunk_count == 1
        assert result.created_by == "user:alice"
        assert result.tags == ["important"]
        assert result.linked_to == ["art-456"]
        assert result.custom == {"version": "1.0"}

    def test_get_blob_metadata_empty_custom(self) -> None:
        """Test getting metadata with empty custom fields."""
        mock_response = MagicMock()
        mock_response.id = "art-123"
        mock_response.filename = "test.txt"
        mock_response.content_type = ""
        mock_response.size = 0
        mock_response.checksum = ""
        mock_response.chunk_count = 0
        mock_response.created = 0
        mock_response.modified = 0
        mock_response.created_by = ""
        mock_response.tags = []
        mock_response.linked_to = []
        mock_response.custom = None
        self.mock_stub.GetMetadata.return_value = mock_response

        result = self.client.get_blob_metadata("art-123")

        assert result.custom == {}


class TestBlobServiceClientConnect:
    """Tests for BlobServiceClient connection handling."""

    def test_init_not_connected(self) -> None:
        """Test client is not connected after init."""
        client = BlobServiceClient("localhost:50051")

        assert not client.is_connected
        assert client._channel is None

    def test_connect_insecure(self) -> None:
        """Test connecting without TLS."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.neumann_pb2_grpc.BlobServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            client = BlobServiceClient.connect("localhost:50051")

            assert client.is_connected
            mock_channel.assert_called_once()
            client.close()

    def test_connect_with_tls(self) -> None:
        """Test connecting with TLS."""
        with (
            patch("grpc.ssl_channel_credentials") as mock_creds,
            patch("grpc.secure_channel") as mock_channel,
            patch("neumann.proto.neumann_pb2_grpc.BlobServiceStub"),
        ):
            mock_creds.return_value = MagicMock()
            mock_channel.return_value = MagicMock()

            client = BlobServiceClient.connect("localhost:50051", tls=True)

            assert client.is_connected
            mock_creds.assert_called_once()
            client.close()

    def test_connect_with_api_key(self) -> None:
        """Test connecting with API key."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.neumann_pb2_grpc.BlobServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            client = BlobServiceClient.connect(
                "localhost:50051",
                api_key="test-key",
            )

            assert client.is_connected
            client.close()

    def test_close(self) -> None:
        """Test close clears state."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.neumann_pb2_grpc.BlobServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            client = BlobServiceClient.connect("localhost:50051")
            client.close()

            assert not client.is_connected
            assert client._channel is None
            assert client._blob is None

    def test_context_manager(self) -> None:
        """Test client as context manager."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.neumann_pb2_grpc.BlobServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            with BlobServiceClient.connect("localhost:50051") as client:
                assert client.is_connected

            assert not client.is_connected


class TestBlobServiceClientConvenienceMethods:
    """Tests for BlobServiceClient convenience methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_blob = MagicMock(spec=BlobClient)

        with (
            patch("grpc.insecure_channel"),
            patch("neumann.proto.neumann_pb2_grpc.BlobServiceStub"),
        ):
            self.client = BlobServiceClient.connect("localhost:50051")
            self.client._blob = self.mock_blob

    def teardown_method(self) -> None:
        """Clean up."""
        self.client.close()

    def test_upload_blob_delegates(self) -> None:
        """Test upload_blob delegates to blob client."""
        self.mock_blob.upload_blob.return_value = BlobUploadResult(
            artifact_id="art-123", size=100, checksum="abc"
        )

        result = self.client.upload_blob("test.txt", b"data")

        assert result.artifact_id == "art-123"
        self.mock_blob.upload_blob.assert_called_once()

    def test_upload_blob_streaming_delegates(self) -> None:
        """Test upload_blob_streaming delegates to blob client."""
        self.mock_blob.upload_blob_streaming.return_value = BlobUploadResult(
            artifact_id="art-456", size=200, checksum="def"
        )

        chunks = iter([b"a", b"b"])
        result = self.client.upload_blob_streaming("test.txt", chunks)

        assert result.artifact_id == "art-456"

    def test_download_blob_delegates(self) -> None:
        """Test download_blob delegates to blob client."""
        self.mock_blob.download_blob.return_value = iter([b"data"])

        result = list(self.client.download_blob("art-123"))

        assert result == [b"data"]

    def test_download_blob_full_delegates(self) -> None:
        """Test download_blob_full delegates to blob client."""
        self.mock_blob.download_blob_full.return_value = b"full data"

        result = self.client.download_blob_full("art-123")

        assert result == b"full data"

    def test_delete_blob_delegates(self) -> None:
        """Test delete_blob delegates to blob client."""
        self.mock_blob.delete_blob.return_value = True

        result = self.client.delete_blob("art-123")

        assert result is True

    def test_get_blob_metadata_delegates(self) -> None:
        """Test get_blob_metadata delegates to blob client."""
        now = datetime.now()
        self.mock_blob.get_blob_metadata.return_value = ArtifactMetadata(
            id="art-123",
            filename="test.txt",
            content_type="text/plain",
            size=100,
            checksum="abc",
            chunk_count=1,
            created=now,
            modified=now,
            created_by="user",
            tags=[],
            linked_to=[],
            custom={},
        )

        result = self.client.get_blob_metadata("art-123")

        assert result.id == "art-123"
