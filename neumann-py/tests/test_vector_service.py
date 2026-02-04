# SPDX-License-Identifier: MIT
"""Tests for vector service clients."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from neumann.services.vector import (
    CollectionInfo,
    CollectionsClient,
    DistanceMetric,
    PointsClient,
    ScoredVectorPoint,
    ScrollResult,
    VectorPoint,
)


class TestVectorPoint:
    """Tests for VectorPoint dataclass."""

    def test_create_basic(self) -> None:
        """Test basic point creation."""
        point = VectorPoint(id="p1", vector=[0.1, 0.2, 0.3])
        assert point.id == "p1"
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload is None

    def test_create_with_payload(self) -> None:
        """Test point creation with payload."""
        point = VectorPoint(id="p1", vector=[0.1, 0.2, 0.3], payload={"name": "test", "count": 42})
        assert point.payload == {"name": "test", "count": 42}


class TestScoredVectorPoint:
    """Tests for ScoredVectorPoint dataclass."""

    def test_create_basic(self) -> None:
        """Test basic scored point creation."""
        point = ScoredVectorPoint(id="p1", score=0.95)
        assert point.id == "p1"
        assert point.score == 0.95
        assert point.payload is None
        assert point.vector is None

    def test_create_full(self) -> None:
        """Test scored point creation with all fields."""
        point = ScoredVectorPoint(
            id="p1",
            score=0.95,
            payload={"name": "test"},
            vector=[0.1, 0.2, 0.3],
        )
        assert point.payload == {"name": "test"}
        assert point.vector == [0.1, 0.2, 0.3]


class TestCollectionInfo:
    """Tests for CollectionInfo dataclass."""

    def test_create(self) -> None:
        """Test collection info creation."""
        info = CollectionInfo(
            name="test_collection", points_count=1000, dimension=384, distance="cosine"
        )
        assert info.name == "test_collection"
        assert info.points_count == 1000
        assert info.dimension == 384
        assert info.distance == "cosine"


class TestScrollResult:
    """Tests for ScrollResult dataclass."""

    def test_create_with_next_offset(self) -> None:
        """Test scroll result with next offset."""
        points = [VectorPoint(id="p1", vector=[0.1])]
        result = ScrollResult(points=points, next_offset="p2")
        assert len(result.points) == 1
        assert result.next_offset == "p2"

    def test_create_without_next_offset(self) -> None:
        """Test scroll result without next offset (last page)."""
        points = [VectorPoint(id="p1", vector=[0.1])]
        result = ScrollResult(points=points)
        assert result.next_offset is None


class TestPointsClientMocked:
    """Tests for PointsClient with mocked proto."""

    def test_convert_point(self) -> None:
        """Test point conversion."""
        mock_stub = MagicMock()
        client = PointsClient(mock_stub)

        mock_proto = MagicMock()
        mock_proto.id = "p1"
        mock_proto.vector = [0.1, 0.2]
        mock_proto.payload = {"name": json.dumps("test").encode()}

        result = client._convert_point(mock_proto)

        assert result.id == "p1"
        assert result.vector == [0.1, 0.2]
        assert result.payload == {"name": "test"}

    def test_convert_scored_point(self) -> None:
        """Test scored point conversion."""
        mock_stub = MagicMock()
        client = PointsClient(mock_stub)

        mock_proto = MagicMock()
        mock_proto.id = "p1"
        mock_proto.score = 0.95
        mock_proto.payload = {}
        mock_proto.vector = [0.1, 0.2]

        result = client._convert_scored_point(mock_proto)

        assert result.id == "p1"
        assert result.score == 0.95
        assert result.vector == [0.1, 0.2]

    def test_convert_point_with_invalid_json(self) -> None:
        """Test point conversion with non-JSON payload."""
        mock_stub = MagicMock()
        client = PointsClient(mock_stub)

        mock_proto = MagicMock()
        mock_proto.id = "p1"
        mock_proto.vector = [0.1]
        mock_proto.payload = {"raw": b"not json"}

        result = client._convert_point(mock_proto)

        assert result.payload == {"raw": "not json"}


class TestCollectionsClientMocked:
    """Tests for CollectionsClient with mocked proto."""

    def test_exists_false(self) -> None:
        """Test exists returns False when collection doesn't exist."""
        from neumann.errors import NotFoundError

        mock_stub = MagicMock()
        mock_stub.Get.side_effect = NotFoundError("not found")
        client = CollectionsClient(mock_stub)

        # Mock the internal get call
        with patch.object(client, "get", side_effect=NotFoundError("not found")):
            result = client.exists("nonexistent")
            assert result is False


class TestDistanceMetric:
    """Tests for DistanceMetric type."""

    def test_valid_metrics(self) -> None:
        """Test valid distance metrics."""
        # Type checking at runtime
        metrics: list[DistanceMetric] = ["cosine", "euclidean", "dot"]
        assert len(metrics) == 3
