# SPDX-License-Identifier: MIT
"""Tests for vector service clients."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from neumann.services.vector import (
    CollectionInfo,
    CollectionsClient,
    DistanceMetric,
    PointsClient,
    ScoredVectorPoint,
    ScrollResult,
    VectorClient,
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


class TestPointsClientOperations:
    """Tests for PointsClient CRUD operations with mocked proto."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = PointsClient(self.mock_stub, metadata=[("x-api-key", "test")])

    def test_upsert_points(self) -> None:
        """Test upserting points."""
        mock_response = MagicMock()
        mock_response.upserted = 3
        self.mock_stub.Upsert.return_value = mock_response

        points = [
            VectorPoint(id="p1", vector=[0.1, 0.2], payload={"name": "test"}),
            VectorPoint(id="p2", vector=[0.3, 0.4], payload=None),
            VectorPoint(id="p3", vector=[0.5, 0.6]),
        ]

        result = self.client.upsert("test_collection", points)

        assert result == 3
        self.mock_stub.Upsert.assert_called_once()

    def test_upsert_points_empty_payload(self) -> None:
        """Test upserting points with empty payload."""
        mock_response = MagicMock()
        mock_response.upserted = 1
        self.mock_stub.Upsert.return_value = mock_response

        points = [VectorPoint(id="p1", vector=[0.1, 0.2])]

        result = self.client.upsert("test_collection", points)

        assert result == 1

    def test_get_points(self) -> None:
        """Test getting points by IDs."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.vector = [0.1, 0.2]
        mock_point.payload = {}

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        self.mock_stub.Get.return_value = mock_response

        result = self.client.get("test_collection", ["p1"])

        assert len(result) == 1
        assert result[0].id == "p1"
        self.mock_stub.Get.assert_called_once()

    def test_get_points_with_options(self) -> None:
        """Test getting points with payload and vector options."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.vector = [0.1, 0.2]
        mock_point.payload = {b"key": b'"value"'}

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        self.mock_stub.Get.return_value = mock_response

        result = self.client.get(
            "test_collection",
            ["p1"],
            with_payload=False,
            with_vector=True,
        )

        assert len(result) == 1

    def test_delete_points(self) -> None:
        """Test deleting points."""
        mock_response = MagicMock()
        mock_response.deleted = 2
        self.mock_stub.Delete.return_value = mock_response

        result = self.client.delete("test_collection", ["p1", "p2"])

        assert result == 2
        self.mock_stub.Delete.assert_called_once()

    def test_query_points(self) -> None:
        """Test querying similar points."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.95
        mock_point.payload = {}
        mock_point.vector = [0.1, 0.2]

        mock_response = MagicMock()
        mock_response.results = [mock_point]
        self.mock_stub.Query.return_value = mock_response

        result = self.client.query(
            "test_collection",
            [0.1, 0.2],
            limit=10,
            offset=0,
        )

        assert len(result) == 1
        assert result[0].score == 0.95

    def test_query_points_with_threshold(self) -> None:
        """Test querying with score threshold."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.95
        mock_point.payload = {}
        mock_point.vector = []

        mock_response = MagicMock()
        mock_response.results = [mock_point]
        self.mock_stub.Query.return_value = mock_response

        result = self.client.query(
            "test_collection",
            [0.1, 0.2],
            score_threshold=0.5,
            with_payload=True,
            with_vector=False,
        )

        assert len(result) == 1

    def test_scroll_points(self) -> None:
        """Test scrolling through points."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.vector = [0.1, 0.2]
        mock_point.payload = {}

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_response.next_offset = "p2"
        self.mock_stub.Scroll.return_value = mock_response

        result = self.client.scroll("test_collection", limit=100)

        assert len(result.points) == 1
        assert result.next_offset == "p2"

    def test_scroll_points_with_offset(self) -> None:
        """Test scrolling with offset ID."""
        mock_point = MagicMock()
        mock_point.id = "p2"
        mock_point.vector = [0.3, 0.4]
        mock_point.payload = {}

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_response.next_offset = ""  # No more pages
        self.mock_stub.Scroll.return_value = mock_response

        result = self.client.scroll(
            "test_collection",
            offset_id="p1",
            with_payload=False,
            with_vector=True,
        )

        assert result.next_offset is None

    def test_convert_point_no_payload(self) -> None:
        """Test converting point with no payload."""
        mock_proto = MagicMock()
        mock_proto.id = "p1"
        mock_proto.vector = [0.1, 0.2]
        mock_proto.payload = {}

        result = self.client._convert_point(mock_proto)

        assert result.id == "p1"
        assert result.payload is None

    def test_convert_scored_point_no_vector(self) -> None:
        """Test converting scored point without vector."""
        mock_proto = MagicMock()
        mock_proto.id = "p1"
        mock_proto.score = 0.9
        mock_proto.payload = {}
        mock_proto.vector = []

        result = self.client._convert_scored_point(mock_proto)

        assert result.vector is None


class TestCollectionsClientOperations:
    """Tests for CollectionsClient CRUD operations with mocked proto."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_stub = MagicMock()
        self.client = CollectionsClient(self.mock_stub)

    def test_create_collection(self) -> None:
        """Test creating a collection."""
        mock_response = MagicMock()
        mock_response.created = True
        self.mock_stub.Create.return_value = mock_response

        result = self.client.create("test_collection", 384, "cosine")

        assert result is True
        self.mock_stub.Create.assert_called_once()

    def test_create_collection_euclidean(self) -> None:
        """Test creating collection with euclidean distance."""
        mock_response = MagicMock()
        mock_response.created = True
        self.mock_stub.Create.return_value = mock_response

        result = self.client.create("test_collection", 128, "euclidean")

        assert result is True

    def test_get_collection(self) -> None:
        """Test getting collection info."""
        mock_response = MagicMock()
        mock_response.name = "test_collection"
        mock_response.points_count = 1000
        mock_response.dimension = 384
        mock_response.distance = "cosine"
        self.mock_stub.Get.return_value = mock_response

        result = self.client.get("test_collection")

        assert result.name == "test_collection"
        assert result.points_count == 1000
        assert result.dimension == 384
        assert result.distance == "cosine"

    def test_delete_collection(self) -> None:
        """Test deleting a collection."""
        mock_response = MagicMock()
        mock_response.deleted = True
        self.mock_stub.Delete.return_value = mock_response

        result = self.client.delete("test_collection")

        assert result is True

    def test_list_collections(self) -> None:
        """Test listing collections."""
        mock_response = MagicMock()
        mock_response.collections = ["collection1", "collection2"]
        self.mock_stub.List.return_value = mock_response

        result = self.client.list()

        assert len(result) == 2
        assert "collection1" in result
        assert "collection2" in result

    def test_list_collections_empty(self) -> None:
        """Test listing collections when empty."""
        mock_response = MagicMock()
        mock_response.collections = []
        self.mock_stub.List.return_value = mock_response

        result = self.client.list()

        assert len(result) == 0

    def test_exists_true(self) -> None:
        """Test exists returns True when collection exists."""
        mock_response = MagicMock()
        mock_response.name = "test_collection"
        mock_response.points_count = 0
        mock_response.dimension = 384
        mock_response.distance = "cosine"
        self.mock_stub.Get.return_value = mock_response

        result = self.client.exists("test_collection")

        assert result is True


class TestVectorClientConnect:
    """Tests for VectorClient connection handling."""

    def test_init_not_connected(self) -> None:
        """Test client is not connected after init."""
        client = VectorClient("localhost:50051")

        assert not client.is_connected
        assert client._channel is None

    def test_points_raises_when_not_connected(self) -> None:
        """Test points property raises when not connected."""
        from neumann.errors import ConnectionError

        client = VectorClient("localhost:50051")

        with pytest.raises(ConnectionError, match="not connected"):
            _ = client.points

    def test_collections_raises_when_not_connected(self) -> None:
        """Test collections property raises when not connected."""
        from neumann.errors import ConnectionError

        client = VectorClient("localhost:50051")

        with pytest.raises(ConnectionError, match="not connected"):
            _ = client.collections

    def test_connect_with_tls(self) -> None:
        """Test connect with TLS creates secure channel."""
        with (
            patch("grpc.ssl_channel_credentials") as mock_creds,
            patch("grpc.secure_channel") as mock_channel,
            patch("neumann.proto.vector_pb2_grpc.PointsServiceStub"),
            patch("neumann.proto.vector_pb2_grpc.CollectionsServiceStub"),
        ):
            mock_creds.return_value = MagicMock()
            mock_channel.return_value = MagicMock()

            client = VectorClient.connect("localhost:50051", tls=True)

            mock_creds.assert_called_once()
            mock_channel.assert_called_once()
            assert client.is_connected

            client.close()

    def test_connect_with_api_key(self) -> None:
        """Test connect with API key sets metadata."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.vector_pb2_grpc.PointsServiceStub"),
            patch("neumann.proto.vector_pb2_grpc.CollectionsServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            client = VectorClient.connect(
                "localhost:50051",
                api_key="test-key",
            )

            assert client.is_connected
            client.close()

    def test_connect_grpc_import_error(self) -> None:
        """Test connect raises when gRPC not available."""
        from neumann.errors import ConnectionError

        with patch.dict("sys.modules", {"grpc": None}), pytest.raises(ConnectionError):
            VectorClient.connect("localhost:50051")

    def test_close_clears_state(self) -> None:
        """Test close clears all state."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.vector_pb2_grpc.PointsServiceStub"),
            patch("neumann.proto.vector_pb2_grpc.CollectionsServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            client = VectorClient.connect("localhost:50051")
            assert client.is_connected

            client.close()

            assert not client.is_connected
            assert client._channel is None
            assert client._points is None
            assert client._collections is None

    def test_context_manager(self) -> None:
        """Test client as context manager."""
        with (
            patch("grpc.insecure_channel") as mock_channel,
            patch("neumann.proto.vector_pb2_grpc.PointsServiceStub"),
            patch("neumann.proto.vector_pb2_grpc.CollectionsServiceStub"),
        ):
            mock_channel.return_value = MagicMock()

            with VectorClient.connect("localhost:50051") as client:
                assert client.is_connected

            assert not client.is_connected


class TestVectorClientConvenienceMethods:
    """Tests for VectorClient convenience methods."""

    def setup_method(self) -> None:
        """Set up test fixtures with mocked connection."""
        self.mock_points = MagicMock(spec=PointsClient)
        self.mock_collections = MagicMock(spec=CollectionsClient)

        with (
            patch("grpc.insecure_channel"),
            patch("neumann.proto.vector_pb2_grpc.PointsServiceStub"),
            patch("neumann.proto.vector_pb2_grpc.CollectionsServiceStub"),
        ):
            self.client = VectorClient.connect("localhost:50051")
            self.client._points = self.mock_points
            self.client._collections = self.mock_collections

    def teardown_method(self) -> None:
        """Clean up."""
        self.client.close()

    def test_create_collection(self) -> None:
        """Test create_collection delegates to collections client."""
        self.mock_collections.create.return_value = True

        result = self.client.create_collection("test", 384, "cosine")

        assert result is True
        self.mock_collections.create.assert_called_once_with("test", 384, "cosine")

    def test_get_collection(self) -> None:
        """Test get_collection delegates to collections client."""
        self.mock_collections.get.return_value = CollectionInfo(
            name="test", points_count=100, dimension=384, distance="cosine"
        )

        result = self.client.get_collection("test")

        assert result.name == "test"
        self.mock_collections.get.assert_called_once_with("test")

    def test_delete_collection(self) -> None:
        """Test delete_collection delegates to collections client."""
        self.mock_collections.delete.return_value = True

        result = self.client.delete_collection("test")

        assert result is True

    def test_list_collections(self) -> None:
        """Test list_collections delegates to collections client."""
        self.mock_collections.list.return_value = ["a", "b"]

        result = self.client.list_collections()

        assert result == ["a", "b"]

    def test_collection_exists(self) -> None:
        """Test collection_exists delegates to collections client."""
        self.mock_collections.exists.return_value = True

        result = self.client.collection_exists("test")

        assert result is True

    def test_upsert_points(self) -> None:
        """Test upsert_points delegates to points client."""
        self.mock_points.upsert.return_value = 5

        points = [VectorPoint(id="p1", vector=[0.1])]
        result = self.client.upsert_points("test", points)

        assert result == 5
        self.mock_points.upsert.assert_called_once()

    def test_get_points(self) -> None:
        """Test get_points delegates to points client."""
        self.mock_points.get.return_value = [VectorPoint(id="p1", vector=[0.1])]

        result = self.client.get_points("test", ["p1"], with_payload=True, with_vector=False)

        assert len(result) == 1

    def test_delete_points(self) -> None:
        """Test delete_points delegates to points client."""
        self.mock_points.delete.return_value = 2

        result = self.client.delete_points("test", ["p1", "p2"])

        assert result == 2

    def test_query_points(self) -> None:
        """Test query_points delegates to points client."""
        self.mock_points.query.return_value = [ScoredVectorPoint(id="p1", score=0.9)]

        result = self.client.query_points(
            "test",
            [0.1, 0.2],
            limit=5,
            offset=0,
            score_threshold=0.5,
            with_payload=True,
            with_vector=False,
        )

        assert len(result) == 1

    def test_scroll_points(self) -> None:
        """Test scroll_points delegates to points client."""
        self.mock_points.scroll.return_value = ScrollResult(
            points=[VectorPoint(id="p1", vector=[0.1])],
            next_offset="p2",
        )

        result = self.client.scroll_points(
            "test",
            offset_id="p0",
            limit=50,
            with_payload=False,
            with_vector=True,
        )

        assert result.next_offset == "p2"

    def test_count_points(self) -> None:
        """Test count_points uses collection info."""
        self.mock_collections.get.return_value = CollectionInfo(
            name="test", points_count=500, dimension=384, distance="cosine"
        )

        result = self.client.count_points("test")

        assert result == 500
