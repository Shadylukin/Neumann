"""Neumann database client with dual-mode support (embedded and remote)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from neumann.errors import (
    ConnectionError,
    NeumannError,
    error_from_code,
)
from neumann.types import (
    ArtifactInfo,
    Edge,
    Node,
    Path,
    PathSegment,
    QueryResult,
    QueryResultType,
    Row,
    ScalarType,
    SimilarItem,
    Value,
)

if TYPE_CHECKING:
    from types import TracebackType


class NeumannClient:
    """Client for Neumann database supporting both embedded and remote modes."""

    def __init__(self, mode: str = "remote") -> None:
        """Initialize client (internal use - use class methods to create)."""
        self._mode = mode
        self._native: object | None = None
        self._channel: object | None = None
        self._stub: object | None = None
        self._api_key: str | None = None
        self._connected = False

    @classmethod
    def embedded(cls, path: str | None = None) -> NeumannClient:
        """Create an embedded (in-process) client using PyO3 bindings.

        Args:
            path: Optional path for persistent storage. If None, uses in-memory storage.

        Returns:
            A connected NeumannClient in embedded mode.

        Raises:
            ConnectionError: If native module is not available.
        """
        client = cls(mode="embedded")
        try:
            from neumann import _native

            if path:
                client._native = _native.QueryRouter.with_path(path)
            else:
                client._native = _native.QueryRouter()
            client._connected = True
        except ImportError as e:
            raise ConnectionError(
                "Native module not available. Install with: pip install neumann-db[native]"
            ) from e
        return client

    @classmethod
    def connect(
        cls,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
    ) -> NeumannClient:
        """Connect to a remote Neumann server via gRPC.

        Args:
            address: Server address in format "host:port".
            api_key: Optional API key for authentication.
            tls: Whether to use TLS encryption.

        Returns:
            A connected NeumannClient in remote mode.

        Raises:
            ConnectionError: If connection fails.
        """
        client = cls(mode="remote")
        client._api_key = api_key

        try:
            import grpc

            if tls:
                credentials = grpc.ssl_channel_credentials()
                client._channel = grpc.secure_channel(address, credentials)
            else:
                client._channel = grpc.insecure_channel(address)

            # Import the generated proto stubs
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

    @property
    def mode(self) -> str:
        """Get client mode ('embedded' or 'remote')."""
        return self._mode

    def close(self) -> None:
        """Close the client connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        self._native = None
        self._stub = None
        self._connected = False

    def __enter__(self) -> NeumannClient:
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

    def execute(self, query: str, *, identity: str | None = None) -> QueryResult:
        """Execute a query and return the result.

        Args:
            query: The Neumann query to execute.
            identity: Optional identity for vault access.

        Returns:
            QueryResult containing the query results.

        Raises:
            NeumannError: If query execution fails.
        """
        if not self._connected:
            raise ConnectionError("Client is not connected")

        if self._mode == "embedded":
            return self._execute_embedded(query, identity)
        else:
            return self._execute_remote(query, identity)

    def _execute_embedded(self, query: str, identity: str | None) -> QueryResult:
        """Execute query in embedded mode."""
        if self._native is None:
            raise ConnectionError("Native module not initialized")

        try:
            result = self._native.execute(query, identity)
            return self._convert_native_result(result)
        except Exception as e:
            raise NeumannError(str(e)) from e

    def _execute_remote(self, query: str, identity: str | None) -> QueryResult:
        """Execute query in remote mode."""
        if self._stub is None:
            raise ConnectionError("gRPC stub not initialized")

        try:
            from neumann.proto import neumann_pb2

            request = neumann_pb2.QueryRequest(query=query)
            if identity:
                request.identity = identity

            metadata = []
            if self._api_key:
                metadata.append(("x-api-key", self._api_key))

            response = self._stub.Execute(request, metadata=metadata or None)
            return self._convert_proto_result(response)
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    def execute_stream(
        self, query: str, *, identity: str | None = None
    ) -> Iterator[QueryResult]:
        """Execute a streaming query.

        Args:
            query: The Neumann query to execute.
            identity: Optional identity for vault access.

        Yields:
            QueryResult items from the stream.

        Raises:
            NeumannError: If query execution fails.
        """
        if not self._connected:
            raise ConnectionError("Client is not connected")

        if self._mode == "embedded":
            # Embedded mode doesn't support true streaming, return single result
            yield self.execute(query, identity=identity)
        else:
            yield from self._execute_stream_remote(query, identity)

    def _execute_stream_remote(
        self, query: str, identity: str | None
    ) -> Iterator[QueryResult]:
        """Execute streaming query in remote mode."""
        if self._stub is None:
            raise ConnectionError("gRPC stub not initialized")

        try:
            from neumann.proto import neumann_pb2

            request = neumann_pb2.QueryRequest(query=query)
            if identity:
                request.identity = identity

            metadata = []
            if self._api_key:
                metadata.append(("x-api-key", self._api_key))

            stream = self._stub.ExecuteStream(request, metadata=metadata or None)
            for chunk in stream:
                if chunk.is_final:
                    break
                yield self._convert_proto_chunk(chunk)
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    def execute_batch(
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
        if not self._connected:
            raise ConnectionError("Client is not connected")

        if self._mode == "embedded":
            return [self.execute(q, identity=identity) for q in queries]
        else:
            return self._execute_batch_remote(queries, identity)

    def _execute_batch_remote(
        self, queries: list[str], identity: str | None
    ) -> list[QueryResult]:
        """Execute batch in remote mode."""
        if self._stub is None:
            raise ConnectionError("gRPC stub not initialized")

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

            response = self._stub.ExecuteBatch(request, metadata=metadata or None)
            return [self._convert_proto_result(r) for r in response.results]
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    def _convert_native_result(self, result: object) -> QueryResult:
        """Convert native result to QueryResult."""
        # The native module returns a dict-like object
        if result is None:
            return QueryResult(QueryResultType.EMPTY)

        result_dict = dict(result) if hasattr(result, "items") else {"type": "empty"}
        result_type = result_dict.get("type", "empty")

        if result_type == "empty":
            return QueryResult(QueryResultType.EMPTY)
        elif result_type == "value":
            return QueryResult(QueryResultType.VALUE, result_dict.get("data"))
        elif result_type == "count":
            return QueryResult(QueryResultType.COUNT, result_dict.get("data"))
        elif result_type == "rows":
            rows = [self._convert_native_row(r) for r in result_dict.get("data", [])]
            return QueryResult(QueryResultType.ROWS, rows)
        elif result_type == "nodes":
            nodes = [self._convert_native_node(n) for n in result_dict.get("data", [])]
            return QueryResult(QueryResultType.NODES, nodes)
        elif result_type == "edges":
            edges = [self._convert_native_edge(e) for e in result_dict.get("data", [])]
            return QueryResult(QueryResultType.EDGES, edges)
        elif result_type == "similar":
            items = [
                self._convert_native_similar(s) for s in result_dict.get("data", [])
            ]
            return QueryResult(QueryResultType.SIMILAR, items)
        elif result_type == "ids":
            return QueryResult(QueryResultType.IDS, result_dict.get("data", []))
        elif result_type == "table_list":
            return QueryResult(QueryResultType.TABLE_LIST, result_dict.get("data", []))
        elif result_type == "blob":
            return QueryResult(QueryResultType.BLOB, result_dict.get("data"))
        elif result_type == "error":
            return QueryResult(QueryResultType.ERROR, result_dict.get("message"))
        else:
            return QueryResult(QueryResultType.EMPTY)

    def _convert_native_row(self, row: dict) -> Row:
        """Convert native row to Row."""
        values = {}
        for k, v in row.items():
            values[k] = self._convert_native_value(v)
        return Row(values=values)

    def _convert_native_node(self, node: dict) -> Node:
        """Convert native node to Node."""
        props = {}
        for k, v in node.get("properties", {}).items():
            props[k] = self._convert_native_value(v)
        return Node(
            id=node.get("id", ""),
            label=node.get("label", ""),
            properties=props,
        )

    def _convert_native_edge(self, edge: dict) -> Edge:
        """Convert native edge to Edge."""
        props = {}
        for k, v in edge.get("properties", {}).items():
            props[k] = self._convert_native_value(v)
        return Edge(
            id=edge.get("id", ""),
            edge_type=edge.get("type", ""),
            source=edge.get("source", ""),
            target=edge.get("target", ""),
            properties=props,
        )

    def _convert_native_similar(self, item: dict) -> SimilarItem:
        """Convert native similar item to SimilarItem."""
        meta = {}
        for k, v in item.get("metadata", {}).items():
            meta[k] = self._convert_native_value(v)
        return SimilarItem(
            key=item.get("key", ""),
            score=float(item.get("score", 0.0)),
            metadata=meta,
        )

    def _convert_native_value(self, value: object) -> Value:
        """Convert native value to Value."""
        if value is None:
            return Value.null()
        elif isinstance(value, bool):
            return Value.bool_(value)
        elif isinstance(value, int):
            return Value.int_(value)
        elif isinstance(value, float):
            return Value.float_(value)
        elif isinstance(value, str):
            return Value.string(value)
        elif isinstance(value, bytes):
            return Value.bytes_(value)
        else:
            return Value.string(str(value))

    def _convert_proto_result(self, response: object) -> QueryResult:
        """Convert proto response to QueryResult."""
        # Handle error response
        if hasattr(response, "error") and response.error:
            err = response.error
            raise error_from_code(err.code, err.message)

        if not hasattr(response, "result") or not response.result:
            return QueryResult(QueryResultType.EMPTY)

        result = response.result
        which = response.WhichOneof("result")

        if which == "empty":
            return QueryResult(QueryResultType.EMPTY)
        elif which == "value":
            return QueryResult(QueryResultType.VALUE, result.value)
        elif which == "count":
            return QueryResult(QueryResultType.COUNT, result.count)
        elif which == "rows":
            rows = [self._convert_proto_row(r) for r in result.rows]
            return QueryResult(QueryResultType.ROWS, rows)
        elif which == "nodes":
            nodes = [self._convert_proto_node(n) for n in result.nodes]
            return QueryResult(QueryResultType.NODES, nodes)
        elif which == "edges":
            edges = [self._convert_proto_edge(e) for e in result.edges]
            return QueryResult(QueryResultType.EDGES, edges)
        elif which == "paths":
            paths = [self._convert_proto_path(p) for p in result.paths]
            return QueryResult(QueryResultType.PATHS, paths)
        elif which == "similar_items":
            items = [self._convert_proto_similar(s) for s in result.items]
            return QueryResult(QueryResultType.SIMILAR, items)
        elif which == "ids":
            return QueryResult(QueryResultType.IDS, list(result.ids))
        elif which == "table_list":
            return QueryResult(QueryResultType.TABLE_LIST, list(result.names))
        elif which == "blob_data":
            return QueryResult(QueryResultType.BLOB, result)
        elif which == "blob_info":
            info = ArtifactInfo(
                artifact_id=result.artifact_id,
                filename=result.filename,
                size=result.size,
                checksum=result.checksum,
                content_type=result.content_type,
                created_at=result.created_at,
                tags=list(result.tags),
            )
            return QueryResult(QueryResultType.BLOB_INFO, info)
        else:
            return QueryResult(QueryResultType.EMPTY)

    def _convert_proto_row(self, row: object) -> Row:
        """Convert proto row to Row."""
        values = {}
        for col in row.columns:
            values[col.name] = self._convert_proto_value(col.value)
        return Row(values=values)

    def _convert_proto_node(self, node: object) -> Node:
        """Convert proto node to Node."""
        props = {}
        for prop in node.properties:
            props[prop.name] = self._convert_proto_value(prop.value)
        return Node(id=node.id, label=node.label, properties=props)

    def _convert_proto_edge(self, edge: object) -> Edge:
        """Convert proto edge to Edge."""
        props = {}
        for prop in edge.properties:
            props[prop.name] = self._convert_proto_value(prop.value)
        return Edge(
            id=edge.id,
            edge_type=edge.edge_type,
            source=edge.source_id,
            target=edge.target_id,
            properties=props,
        )

    def _convert_proto_path(self, path: object) -> Path:
        """Convert proto path to Path."""
        segments = []
        for seg in path.segments:
            node = self._convert_proto_node(seg.node) if seg.node else None
            edge = self._convert_proto_edge(seg.edge) if seg.edge else None
            if node:
                segments.append(PathSegment(node=node, edge=edge))
        return Path(segments=segments)

    def _convert_proto_similar(self, item: object) -> SimilarItem:
        """Convert proto similar item to SimilarItem."""
        meta = {}
        if hasattr(item, "metadata"):
            for prop in item.metadata:
                meta[prop.name] = self._convert_proto_value(prop.value)
        return SimilarItem(key=item.key, score=item.score, metadata=meta)

    def _convert_proto_value(self, value: object) -> Value:
        """Convert proto value to Value."""
        which = value.WhichOneof("value") if hasattr(value, "WhichOneof") else None

        if which == "null_value" or which is None:
            return Value.null()
        elif which == "int_value":
            return Value.int_(value.int_value)
        elif which == "float_value":
            return Value.float_(value.float_value)
        elif which == "string_value":
            return Value.string(value.string_value)
        elif which == "bool_value":
            return Value.bool_(value.bool_value)
        elif which == "bytes_value":
            return Value.bytes_(value.bytes_value)
        else:
            return Value.null()

    def _convert_proto_chunk(self, chunk: object) -> QueryResult:
        """Convert proto streaming chunk to QueryResult."""
        which = chunk.WhichOneof("chunk") if hasattr(chunk, "WhichOneof") else None

        if which == "row":
            row = self._convert_proto_row(chunk.row.row)
            return QueryResult(QueryResultType.ROWS, [row])
        elif which == "node":
            node = self._convert_proto_node(chunk.node.node)
            return QueryResult(QueryResultType.NODES, [node])
        elif which == "edge":
            edge = self._convert_proto_edge(chunk.edge.edge)
            return QueryResult(QueryResultType.EDGES, [edge])
        elif which == "similar_item":
            item = self._convert_proto_similar(chunk.similar_item.item)
            return QueryResult(QueryResultType.SIMILAR, [item])
        elif which == "blob_data":
            return QueryResult(QueryResultType.BLOB, chunk.blob_data)
        elif which == "error":
            raise error_from_code(chunk.error.code, chunk.error.message)
        else:
            return QueryResult(QueryResultType.EMPTY)
