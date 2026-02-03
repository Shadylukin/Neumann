# SPDX-License-Identifier: MIT
"""Vector service clients for points and collections operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from neumann.config import ClientConfig, RetryConfig
from neumann.errors import (
    ConnectionError,
    NotFoundError,
)
from neumann.retry import retry_call

if TYPE_CHECKING:
    from types import TracebackType


DistanceMetric = Literal["cosine", "euclidean", "dot"]


@dataclass
class VectorPoint:
    """A vector point with payload."""

    id: str
    vector: list[float]
    payload: dict[str, Any] | None = None


@dataclass
class ScoredVectorPoint:
    """A point with similarity score from a query."""

    id: str
    score: float
    payload: dict[str, Any] | None = None
    vector: list[float] | None = None


@dataclass
class CollectionInfo:
    """Collection information."""

    name: str
    points_count: int
    dimension: int
    distance: str


@dataclass
class ScrollResult:
    """Scroll result with pagination info."""

    points: list[VectorPoint]
    next_offset: str | None = None


class PointsClient:
    """Service client for vector points operations."""

    def __init__(
        self,
        stub: Any,
        metadata: list[tuple[str, str]] | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._stub: Any = stub
        self._metadata: list[tuple[str, str]] = metadata or []
        self._retry_config = retry_config or RetryConfig()

    def upsert(
        self,
        collection: str,
        points: list[VectorPoint],
    ) -> int:
        """Upsert points into a collection.

        Args:
            collection: Target collection name.
            points: Points to upsert.

        Returns:
            Number of points upserted.
        """
        from neumann.proto import vector_pb2

        proto_points = []
        for p in points:
            payload = {}
            if p.payload:
                for k, v in p.payload.items():
                    payload[k] = json.dumps(v).encode()
            proto_points.append(vector_pb2.Point(id=p.id, vector=p.vector, payload=payload))

        request = vector_pb2.UpsertPointsRequest(collection=collection, points=proto_points)

        def do_upsert() -> Any:
            return self._stub.Upsert(request, metadata=self._metadata or None)

        response = retry_call(do_upsert, self._retry_config)
        return cast(int, response.upserted)

    def get(
        self,
        collection: str,
        ids: list[str],
        *,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[VectorPoint]:
        """Get points by IDs.

        Args:
            collection: Target collection name.
            ids: Point IDs to retrieve.
            with_payload: Include payload in response.
            with_vector: Include vector in response.

        Returns:
            Retrieved points.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.GetPointsRequest(
            collection=collection,
            ids=ids,
            with_payload=with_payload,
            with_vector=with_vector,
        )

        def do_get() -> Any:
            return self._stub.Get(request, metadata=self._metadata or None)

        response = retry_call(do_get, self._retry_config)
        return [self._convert_point(p) for p in response.points]

    def delete(self, collection: str, ids: list[str]) -> int:
        """Delete points by IDs.

        Args:
            collection: Target collection name.
            ids: Point IDs to delete.

        Returns:
            Number of points deleted.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.DeletePointsRequest(collection=collection, ids=ids)

        def do_delete() -> Any:
            return self._stub.Delete(request, metadata=self._metadata or None)

        response = retry_call(do_delete, self._retry_config)
        return cast(int, response.deleted)

    def query(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int = 10,
        offset: int = 0,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[ScoredVectorPoint]:
        """Query for similar points.

        Args:
            collection: Target collection name.
            vector: Query vector.
            limit: Maximum number of results.
            offset: Number of results to skip.
            score_threshold: Minimum similarity score threshold.
            with_payload: Include payload in response.
            with_vector: Include vector in response.

        Returns:
            Similar points with scores.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.QueryPointsRequest(
            collection=collection,
            vector=vector,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vector=with_vector,
        )
        if score_threshold is not None:
            request.score_threshold = score_threshold

        def do_query() -> Any:
            return self._stub.Query(request, metadata=self._metadata or None)

        response = retry_call(do_query, self._retry_config)
        return [self._convert_scored_point(p) for p in response.results]

    def scroll(
        self,
        collection: str,
        *,
        offset_id: str | None = None,
        limit: int = 100,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> ScrollResult:
        """Scroll through points in a collection.

        Args:
            collection: Target collection name.
            offset_id: Offset point ID for pagination.
            limit: Maximum number of points to return.
            with_payload: Include payload in response.
            with_vector: Include vector in response.

        Returns:
            Scroll result with points and next offset.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.ScrollPointsRequest(
            collection=collection,
            limit=limit,
            with_payload=with_payload,
            with_vector=with_vector,
        )
        if offset_id is not None:
            request.offset_id = offset_id

        def do_scroll() -> Any:
            return self._stub.Scroll(request, metadata=self._metadata or None)

        response = retry_call(do_scroll, self._retry_config)
        points = [self._convert_point(p) for p in response.points]
        next_offset: str | None = response.next_offset if response.next_offset else None
        return ScrollResult(points=points, next_offset=next_offset)

    def _convert_point(self, proto: Any) -> VectorPoint:
        """Convert proto point to VectorPoint."""
        payload: dict[str, Any] | None = None
        if proto.payload:
            payload = {}
            for k, v in proto.payload.items():
                try:
                    payload[k] = json.loads(v.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload[k] = v.decode()
        return VectorPoint(id=str(proto.id), vector=list(proto.vector), payload=payload)

    def _convert_scored_point(self, proto: Any) -> ScoredVectorPoint:
        """Convert proto scored point to ScoredVectorPoint."""
        payload: dict[str, Any] | None = None
        if proto.payload:
            payload = {}
            for k, v in proto.payload.items():
                try:
                    payload[k] = json.loads(v.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload[k] = v.decode()
        vector: list[float] | None = list(proto.vector) if proto.vector else None
        return ScoredVectorPoint(
            id=str(proto.id), score=float(proto.score), payload=payload, vector=vector
        )


class CollectionsClient:
    """Service client for vector collections operations."""

    def __init__(
        self,
        stub: Any,
        metadata: list[tuple[str, str]] | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._stub: Any = stub
        self._metadata: list[tuple[str, str]] = metadata or []
        self._retry_config = retry_config or RetryConfig()

    def create(
        self,
        name: str,
        dimension: int,
        distance: DistanceMetric = "cosine",
    ) -> bool:
        """Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            distance: Distance metric.

        Returns:
            True if collection was created.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.CreateCollectionRequest(
            name=name, dimension=dimension, distance=distance
        )

        def do_create() -> Any:
            return self._stub.Create(request, metadata=self._metadata or None)

        response = retry_call(do_create, self._retry_config)
        return cast(bool, response.created)

    def get(self, name: str) -> CollectionInfo:
        """Get collection information.

        Args:
            name: Collection name.

        Returns:
            Collection information.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.GetCollectionRequest(name=name)

        def do_get() -> Any:
            return self._stub.Get(request, metadata=self._metadata or None)

        response = retry_call(do_get, self._retry_config)
        return CollectionInfo(
            name=str(response.name),
            points_count=int(response.points_count),
            dimension=int(response.dimension),
            distance=str(response.distance),
        )

    def delete(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name.

        Returns:
            True if collection was deleted.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.DeleteCollectionRequest(name=name)

        def do_delete() -> Any:
            return self._stub.Delete(request, metadata=self._metadata or None)

        response = retry_call(do_delete, self._retry_config)
        return cast(bool, response.deleted)

    def list(self) -> list[str]:
        """List all collections.

        Returns:
            Array of collection names.
        """
        from neumann.proto import vector_pb2

        request = vector_pb2.ListCollectionsRequest()

        def do_list() -> Any:
            return self._stub.List(request, metadata=self._metadata or None)

        response = retry_call(do_list, self._retry_config)
        return [str(c) for c in response.collections]

    def exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name.

        Returns:
            True if collection exists.
        """
        try:
            self.get(name)
            return True
        except NotFoundError:
            return False


class VectorClient:
    """High-level vector client combining points and collections operations."""

    def __init__(
        self,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
        config: ClientConfig | None = None,
    ) -> None:
        """Initialize vector client (internal use - use connect())."""
        self._address = address
        self._api_key = api_key
        self._tls = tls
        self._config = config or ClientConfig.default()
        self._channel: Any = None
        self._points_stub: Any = None
        self._collections_stub: Any = None
        self._points: PointsClient | None = None
        self._collections: CollectionsClient | None = None
        self._connected = False

    @classmethod
    def connect(
        cls,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
        config: ClientConfig | None = None,
    ) -> VectorClient:
        """Connect to a remote Neumann server's vector service.

        Args:
            address: Server address in format "host:port".
            api_key: Optional API key for authentication.
            tls: Whether to use TLS encryption.
            config: Optional client configuration.

        Returns:
            A connected VectorClient.

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

            from neumann.proto import vector_pb2_grpc

            client._points_stub = vector_pb2_grpc.PointsServiceStub(client._channel)
            client._collections_stub = vector_pb2_grpc.CollectionsServiceStub(client._channel)

            metadata = []
            if api_key:
                metadata.append(("x-api-key", api_key))

            client._points = PointsClient(client._points_stub, metadata, client._config.retry)
            client._collections = CollectionsClient(
                client._collections_stub, metadata, client._config.retry
            )
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

    @property
    def points(self) -> PointsClient:
        """Get the points service client."""
        if self._points is None:
            raise ConnectionError("Client is not connected")
        return self._points

    @property
    def collections(self) -> CollectionsClient:
        """Get the collections service client."""
        if self._collections is None:
            raise ConnectionError("Client is not connected")
        return self._collections

    def close(self) -> None:
        """Close the client connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        self._points = None
        self._collections = None
        self._connected = False

    def __enter__(self) -> VectorClient:
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

    # Convenience methods that delegate to service clients

    def create_collection(
        self,
        name: str,
        dimension: int,
        distance: DistanceMetric = "cosine",
    ) -> bool:
        """Create a new collection."""
        return self.collections.create(name, dimension, distance)

    def get_collection(self, name: str) -> CollectionInfo:
        """Get collection information."""
        return self.collections.get(name)

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        return self.collections.delete(name)

    def list_collections(self) -> list[str]:
        """List all collections."""
        return self.collections.list()

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return self.collections.exists(name)

    def upsert_points(self, collection: str, points: list[VectorPoint]) -> int:
        """Upsert points into a collection."""
        return self.points.upsert(collection, points)

    def get_points(
        self,
        collection: str,
        ids: list[str],
        *,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[VectorPoint]:
        """Get points by IDs."""
        return self.points.get(collection, ids, with_payload=with_payload, with_vector=with_vector)

    def delete_points(self, collection: str, ids: list[str]) -> int:
        """Delete points by IDs."""
        return self.points.delete(collection, ids)

    def query_points(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int = 10,
        offset: int = 0,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[ScoredVectorPoint]:
        """Query for similar points."""
        return self.points.query(
            collection,
            vector,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vector=with_vector,
        )

    def scroll_points(
        self,
        collection: str,
        *,
        offset_id: str | None = None,
        limit: int = 100,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> ScrollResult:
        """Scroll through points in a collection."""
        return self.points.scroll(
            collection,
            offset_id=offset_id,
            limit=limit,
            with_payload=with_payload,
            with_vector=with_vector,
        )

    def count_points(self, collection: str) -> int:
        """Get number of points in a collection."""
        info = self.collections.get(collection)
        return info.points_count
