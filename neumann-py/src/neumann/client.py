# SPDX-License-Identifier: MIT
"""Neumann database client with dual-mode support (embedded and remote)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from neumann.config import ClientConfig
from neumann.errors import (
    ConnectionError,
    NeumannError,
    error_from_code,
)
from neumann.retry import retry_call
from neumann.types import (
    ArtifactInfo,
    BlobStats,
    ChainBlockInfo,
    ChainCodebookInfo,
    ChainCommitted,
    ChainConflictResolution,
    ChainDrift,
    ChainHeight,
    ChainHistory,
    ChainHistoryEntry,
    ChainMergeResult,
    ChainRolledBack,
    ChainSimilar,
    ChainSimilarItem,
    ChainTip,
    ChainTransactionBegun,
    ChainTransitionAnalysis,
    CheckpointInfo,
    Edge,
    Node,
    Path,
    PathSegment,
    QueryResult,
    QueryResultType,
    Row,
    SimilarItem,
    UnifiedItem,
    UnifiedResult,
    Value,
)

if TYPE_CHECKING:
    from types import TracebackType


class NeumannClient:
    """Client for Neumann database supporting both embedded and remote modes."""

    def __init__(self, mode: str = "remote") -> None:
        """Initialize client (internal use - use class methods to create)."""
        self._mode = mode
        self._native: Any = None
        self._channel: Any = None
        self._stub: Any = None
        self._api_key: str | None = None
        self._config: ClientConfig = ClientConfig.default()
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
            from neumann import _native  # type: ignore[attr-defined]

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
        config: ClientConfig | None = None,
    ) -> NeumannClient:
        """Connect to a remote Neumann server via gRPC.

        Args:
            address: Server address in format "host:port".
            api_key: Optional API key for authentication.
            tls: Whether to use TLS encryption.
            config: Optional client configuration for timeouts, retries, and keepalive.

        Returns:
            A connected NeumannClient in remote mode.

        Raises:
            ConnectionError: If connection fails.
        """
        client = cls(mode="remote")
        client._api_key = api_key
        client._config = config or ClientConfig.default()

        try:
            import grpc

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
                client._channel = grpc.secure_channel(address, credentials, options=options)
            else:
                client._channel = grpc.insecure_channel(address, options=options)

            # Import the generated proto stubs
            from neumann.proto import neumann_pb2_grpc

            client._stub = neumann_pb2_grpc.QueryServiceStub(client._channel)
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

    def query(self, query: str, *, identity: str | None = None) -> QueryResult:
        """Execute a query and return the result.

        This is an alias for execute().

        Args:
            query: The Neumann query to execute.
            identity: Optional identity for vault access.

        Returns:
            QueryResult containing the query results.

        Raises:
            NeumannError: If query execution fails.
        """
        return self.execute(query, identity=identity)

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

            timeout = self._config.timeout.query_timeout_s or self._config.timeout.default_timeout_s

            def do_execute() -> Any:
                return self._stub.Execute(request, timeout=timeout, metadata=metadata or None)

            response = retry_call(do_execute, self._config.retry)
            return self._convert_proto_result(response)
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    def execute_stream(self, query: str, *, identity: str | None = None) -> Iterator[QueryResult]:
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

    def _execute_stream_remote(self, query: str, identity: str | None) -> Iterator[QueryResult]:
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

            timeout = self._config.timeout.query_timeout_s or self._config.timeout.default_timeout_s

            stream = self._stub.ExecuteStream(request, timeout=timeout, metadata=metadata or None)
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

    def _execute_batch_remote(self, queries: list[str], identity: str | None) -> list[QueryResult]:
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

            timeout = self._config.timeout.query_timeout_s or self._config.timeout.default_timeout_s

            def do_batch() -> Any:
                return self._stub.ExecuteBatch(request, timeout=timeout, metadata=metadata or None)

            response = retry_call(do_batch, self._config.retry)
            return [self._convert_proto_result(r) for r in response.results]
        except Exception as e:
            if "grpc" in str(type(e).__module__):
                raise ConnectionError(f"gRPC error: {e}") from e
            raise NeumannError(str(e)) from e

    def _convert_native_result(self, result: Any) -> QueryResult:
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
            items = [self._convert_native_similar(s) for s in result_dict.get("data", [])]
            return QueryResult(QueryResultType.SIMILAR, items)
        elif result_type == "ids":
            return QueryResult(QueryResultType.IDS, result_dict.get("data", []))
        elif result_type == "table_list":
            return QueryResult(QueryResultType.TABLE_LIST, result_dict.get("data", []))
        elif result_type == "blob":
            return QueryResult(QueryResultType.BLOB, result_dict.get("data"))
        elif result_type == "blob_stats":
            data = result_dict.get("data", {})
            stats = BlobStats(
                artifact_count=data.get("artifact_count", 0),
                chunk_count=data.get("chunk_count", 0),
                total_bytes=data.get("total_bytes", 0),
                unique_bytes=data.get("unique_bytes", 0),
                dedup_ratio=data.get("dedup_ratio", 0.0),
                orphaned_chunks=data.get("orphaned_chunks", 0),
            )
            return QueryResult(QueryResultType.BLOB_STATS, stats)
        elif result_type == "artifact_list":
            return QueryResult(QueryResultType.ARTIFACT_LIST, result_dict.get("data", []))
        elif result_type == "checkpoint_list":
            checkpoints = [self._convert_native_checkpoint(c) for c in result_dict.get("data", [])]
            return QueryResult(QueryResultType.CHECKPOINT_LIST, checkpoints)
        elif result_type == "unified":
            data = result_dict.get("data", {})
            unified_items: list[UnifiedItem] = [
                self._convert_native_unified_item(i) for i in data.get("items", [])
            ]
            unified = UnifiedResult(
                description=data.get("description", ""),
                items=unified_items,
            )
            return QueryResult(QueryResultType.UNIFIED, unified)
        elif result_type == "error":
            return QueryResult(QueryResultType.ERROR, result_dict.get("message"))
        else:
            return self._convert_native_chain_result(result_type, result_dict)

    def _convert_native_row(self, row: dict[str, Any]) -> Row:
        """Convert native row to Row."""
        values: dict[str, Value] = {}
        for k, v in row.items():
            values[k] = self._convert_native_value(v)
        return Row(values=values)

    def _convert_native_node(self, node: dict[str, Any]) -> Node:
        """Convert native node to Node."""
        props: dict[str, Value] = {}
        for k, v in node.get("properties", {}).items():
            props[k] = self._convert_native_value(v)
        return Node(
            id=node.get("id", ""),
            label=node.get("label", ""),
            properties=props,
        )

    def _convert_native_edge(self, edge: dict[str, Any]) -> Edge:
        """Convert native edge to Edge."""
        props: dict[str, Value] = {}
        for k, v in edge.get("properties", {}).items():
            props[k] = self._convert_native_value(v)
        return Edge(
            id=edge.get("id", ""),
            edge_type=edge.get("type", ""),
            source=edge.get("source", ""),
            target=edge.get("target", ""),
            properties=props,
        )

    def _convert_native_similar(self, item: dict[str, Any]) -> SimilarItem:
        """Convert native similar item to SimilarItem."""
        meta: dict[str, Value] = {}
        for k, v in item.get("metadata", {}).items():
            meta[k] = self._convert_native_value(v)
        return SimilarItem(
            key=item.get("key", ""),
            score=float(item.get("score", 0.0)),
            metadata=meta,
        )

    def _convert_native_value(self, value: Any) -> Value:
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

    def _convert_native_checkpoint(self, checkpoint: dict[str, Any]) -> CheckpointInfo:
        """Convert native checkpoint to CheckpointInfo."""
        return CheckpointInfo(
            id=checkpoint.get("id", ""),
            name=checkpoint.get("name", ""),
            created_at=checkpoint.get("created_at", 0),
            is_auto=checkpoint.get("is_auto", False),
        )

    def _convert_native_unified_item(self, item: dict[str, Any]) -> UnifiedItem:
        """Convert native unified item to UnifiedItem."""
        return UnifiedItem(
            entity_type=item.get("entity_type", ""),
            key=item.get("key", ""),
            fields=item.get("fields", {}),
            score=item.get("score"),
        )

    def _convert_native_chain_result(
        self, result_type: str, result_dict: dict[str, Any]
    ) -> QueryResult:
        """Convert native chain result to QueryResult."""
        data = result_dict.get("data", {})

        if result_type == "chain_transaction_begun":
            return QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id=data.get("tx_id", "")),
            )
        elif result_type == "chain_committed":
            return QueryResult(
                QueryResultType.CHAIN_COMMITTED,
                ChainCommitted(
                    block_hash=data.get("block_hash", ""),
                    height=data.get("height", 0),
                ),
            )
        elif result_type == "chain_rolled_back":
            return QueryResult(
                QueryResultType.CHAIN_ROLLED_BACK,
                ChainRolledBack(to_height=data.get("to_height", 0)),
            )
        elif result_type == "chain_history":
            entries = [
                ChainHistoryEntry(
                    height=e.get("height", 0),
                    transaction_type=e.get("transaction_type", ""),
                    data=e.get("data"),
                )
                for e in data.get("entries", [])
            ]
            return QueryResult(QueryResultType.CHAIN_HISTORY, ChainHistory(entries=entries))
        elif result_type == "chain_similar":
            items = [
                ChainSimilarItem(
                    block_hash=i.get("block_hash", ""),
                    height=i.get("height", 0),
                    similarity=i.get("similarity", 0.0),
                )
                for i in data.get("items", [])
            ]
            return QueryResult(QueryResultType.CHAIN_SIMILAR, ChainSimilar(items=items))
        elif result_type == "chain_drift":
            return QueryResult(
                QueryResultType.CHAIN_DRIFT,
                ChainDrift(
                    from_height=data.get("from_height", 0),
                    to_height=data.get("to_height", 0),
                    total_drift=data.get("total_drift", 0.0),
                    avg_drift_per_block=data.get("avg_drift_per_block", 0.0),
                    max_drift=data.get("max_drift", 0.0),
                ),
            )
        elif result_type == "chain_height":
            return QueryResult(
                QueryResultType.CHAIN_HEIGHT,
                ChainHeight(height=data.get("height", 0)),
            )
        elif result_type == "chain_tip":
            return QueryResult(
                QueryResultType.CHAIN_TIP,
                ChainTip(hash=data.get("hash", ""), height=data.get("height", 0)),
            )
        elif result_type == "chain_block":
            return QueryResult(
                QueryResultType.CHAIN_BLOCK,
                ChainBlockInfo(
                    height=data.get("height", 0),
                    hash=data.get("hash", ""),
                    prev_hash=data.get("prev_hash", ""),
                    timestamp=data.get("timestamp", 0),
                    transaction_count=data.get("transaction_count", 0),
                    proposer=data.get("proposer", ""),
                ),
            )
        elif result_type == "chain_codebook":
            return QueryResult(
                QueryResultType.CHAIN_CODEBOOK,
                ChainCodebookInfo(
                    scope=data.get("scope", ""),
                    entry_count=data.get("entry_count", 0),
                    dimension=data.get("dimension", 0),
                    domain=data.get("domain"),
                ),
            )
        elif result_type == "chain_transition_analysis":
            return QueryResult(
                QueryResultType.CHAIN_TRANSITION_ANALYSIS,
                ChainTransitionAnalysis(
                    total_transitions=data.get("total_transitions", 0),
                    valid_transitions=data.get("valid_transitions", 0),
                    invalid_transitions=data.get("invalid_transitions", 0),
                    avg_validity_score=data.get("avg_validity_score", 0.0),
                ),
            )
        elif result_type == "chain_conflict_resolution":
            return QueryResult(
                QueryResultType.CHAIN_CONFLICT_RESOLUTION,
                ChainConflictResolution(
                    strategy=data.get("strategy", ""),
                    conflicts_resolved=data.get("conflicts_resolved", 0),
                ),
            )
        elif result_type == "chain_merge":
            return QueryResult(
                QueryResultType.CHAIN_MERGE,
                ChainMergeResult(
                    success=data.get("success", False),
                    merged_count=data.get("merged_count", 0),
                ),
            )
        else:
            return QueryResult(QueryResultType.EMPTY)

    def _convert_proto_result(self, response: Any) -> QueryResult:
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
        elif which == "blob_stats":
            stats = BlobStats(
                artifact_count=result.artifact_count,
                chunk_count=result.chunk_count,
                total_bytes=result.total_bytes,
                unique_bytes=result.unique_bytes,
                dedup_ratio=result.dedup_ratio,
                orphaned_chunks=result.orphaned_chunks,
            )
            return QueryResult(QueryResultType.BLOB_STATS, stats)
        elif which == "artifact_list":
            return QueryResult(QueryResultType.ARTIFACT_LIST, list(result.artifact_ids))
        elif which == "checkpoint_list":
            checkpoints = [self._convert_proto_checkpoint(c) for c in result.checkpoints]
            return QueryResult(QueryResultType.CHECKPOINT_LIST, checkpoints)
        elif which == "unified":
            unified_items: list[UnifiedItem] = [
                self._convert_proto_unified_item(i) for i in result.items
            ]
            unified = UnifiedResult(description=result.description, items=unified_items)
            return QueryResult(QueryResultType.UNIFIED, unified)
        elif which in (
            "chain_transaction_begun",
            "chain_committed",
            "chain_rolled_back",
            "chain_history",
            "chain_similar",
            "chain_drift",
            "chain_height",
            "chain_tip",
            "chain_block",
            "chain_codebook",
            "chain_transition_analysis",
            "chain_conflict_resolution",
            "chain_merge",
        ):
            return self._convert_proto_chain_result(which, result)
        else:
            return QueryResult(QueryResultType.EMPTY)

    def _convert_proto_row(self, row: Any) -> Row:
        """Convert proto row to Row."""
        values = {}
        for col in row.columns:
            values[col.name] = self._convert_proto_value(col.value)
        return Row(values=values)

    def _convert_proto_node(self, node: Any) -> Node:
        """Convert proto node to Node."""
        props = {}
        for prop in node.properties:
            props[prop.name] = self._convert_proto_value(prop.value)
        return Node(id=node.id, label=node.label, properties=props)

    def _convert_proto_edge(self, edge: Any) -> Edge:
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

    def _convert_proto_path(self, path: Any) -> Path:
        """Convert proto path to Path."""
        segments = []
        for seg in path.segments:
            node = self._convert_proto_node(seg.node) if seg.node else None
            edge = self._convert_proto_edge(seg.edge) if seg.edge else None
            if node:
                segments.append(PathSegment(node=node, edge=edge))
        return Path(segments=segments)

    def _convert_proto_similar(self, item: Any) -> SimilarItem:
        """Convert proto similar item to SimilarItem."""
        meta = {}
        if hasattr(item, "metadata"):
            for prop in item.metadata:
                meta[prop.name] = self._convert_proto_value(prop.value)
        return SimilarItem(key=item.key, score=item.score, metadata=meta)

    def _convert_proto_value(self, value: Any) -> Value:
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

    def _convert_proto_checkpoint(self, checkpoint: Any) -> CheckpointInfo:
        """Convert proto checkpoint to CheckpointInfo."""
        return CheckpointInfo(
            id=checkpoint.id,
            name=checkpoint.name,
            created_at=checkpoint.created_at,
            is_auto=checkpoint.is_auto,
        )

    def _convert_proto_unified_item(self, item: Any) -> UnifiedItem:
        """Convert proto unified item to UnifiedItem."""
        fields = {}
        if hasattr(item, "fields"):
            for k, v in item.fields.items():
                fields[k] = v
        return UnifiedItem(
            entity_type=item.entity_type,
            key=item.key,
            fields=fields,
            score=item.score if hasattr(item, "score") and item.score else None,
        )

    def _convert_proto_chain_result(self, which: str, result: Any) -> QueryResult:
        """Convert proto chain result to QueryResult."""
        if which == "chain_transaction_begun":
            return QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id=result.tx_id),
            )
        elif which == "chain_committed":
            return QueryResult(
                QueryResultType.CHAIN_COMMITTED,
                ChainCommitted(block_hash=result.block_hash, height=result.height),
            )
        elif which == "chain_rolled_back":
            return QueryResult(
                QueryResultType.CHAIN_ROLLED_BACK,
                ChainRolledBack(to_height=result.to_height),
            )
        elif which == "chain_history":
            entries = [
                ChainHistoryEntry(
                    height=e.height,
                    transaction_type=e.transaction_type,
                    data=e.data if hasattr(e, "data") else None,
                )
                for e in result.entries
            ]
            return QueryResult(QueryResultType.CHAIN_HISTORY, ChainHistory(entries=entries))
        elif which == "chain_similar":
            items = [
                ChainSimilarItem(
                    block_hash=i.block_hash,
                    height=i.height,
                    similarity=i.similarity,
                )
                for i in result.items
            ]
            return QueryResult(QueryResultType.CHAIN_SIMILAR, ChainSimilar(items=items))
        elif which == "chain_drift":
            return QueryResult(
                QueryResultType.CHAIN_DRIFT,
                ChainDrift(
                    from_height=result.from_height,
                    to_height=result.to_height,
                    total_drift=result.total_drift,
                    avg_drift_per_block=result.avg_drift_per_block,
                    max_drift=result.max_drift,
                ),
            )
        elif which == "chain_height":
            return QueryResult(QueryResultType.CHAIN_HEIGHT, ChainHeight(height=result.height))
        elif which == "chain_tip":
            return QueryResult(
                QueryResultType.CHAIN_TIP,
                ChainTip(hash=result.hash, height=result.height),
            )
        elif which == "chain_block":
            return QueryResult(
                QueryResultType.CHAIN_BLOCK,
                ChainBlockInfo(
                    height=result.height,
                    hash=result.hash,
                    prev_hash=result.prev_hash,
                    timestamp=result.timestamp,
                    transaction_count=result.transaction_count,
                    proposer=result.proposer,
                ),
            )
        elif which == "chain_codebook":
            return QueryResult(
                QueryResultType.CHAIN_CODEBOOK,
                ChainCodebookInfo(
                    scope=result.scope,
                    entry_count=result.entry_count,
                    dimension=result.dimension,
                    domain=result.domain if hasattr(result, "domain") else None,
                ),
            )
        elif which == "chain_transition_analysis":
            return QueryResult(
                QueryResultType.CHAIN_TRANSITION_ANALYSIS,
                ChainTransitionAnalysis(
                    total_transitions=result.total_transitions,
                    valid_transitions=result.valid_transitions,
                    invalid_transitions=result.invalid_transitions,
                    avg_validity_score=result.avg_validity_score,
                ),
            )
        elif which == "chain_conflict_resolution":
            return QueryResult(
                QueryResultType.CHAIN_CONFLICT_RESOLUTION,
                ChainConflictResolution(
                    strategy=result.strategy,
                    conflicts_resolved=result.conflicts_resolved,
                ),
            )
        elif which == "chain_merge":
            return QueryResult(
                QueryResultType.CHAIN_MERGE,
                ChainMergeResult(success=result.success, merged_count=result.merged_count),
            )
        else:
            return QueryResult(QueryResultType.EMPTY)

    def _convert_proto_chunk(self, chunk: Any) -> QueryResult:
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
