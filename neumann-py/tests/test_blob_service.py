# SPDX-License-Identifier: MIT
"""Tests for blob service clients."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from neumann.config import ClientConfig, RetryConfig
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
