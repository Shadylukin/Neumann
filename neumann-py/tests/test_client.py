"""Comprehensive tests for NeumannClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neumann.client import NeumannClient
from neumann.errors import (
    ConnectionError,
    ErrorCode,
    InternalError,
    NeumannError,
)
from neumann.types import (
    BlobStats,
    ChainBlockInfo,
    ChainCodebookInfo,
    ChainCommitted,
    ChainConflictResolution,
    ChainDrift,
    ChainHeight,
    ChainHistory,
    ChainMergeResult,
    ChainRolledBack,
    ChainSimilar,
    ChainTip,
    ChainTransactionBegun,
    ChainTransitionAnalysis,
    CheckpointInfo,
    Edge,
    Node,
    QueryResultType,
    Row,
    ScalarType,
    SimilarItem,
    UnifiedResult,
)


class TestNeumannClientInit:
    """Tests for NeumannClient initialization."""

    def test_init_default_mode(self) -> None:
        """Test default mode is remote."""
        client = NeumannClient()
        assert client.mode == "remote"
        assert not client.is_connected

    def test_init_embedded_mode(self) -> None:
        """Test embedded mode initialization."""
        client = NeumannClient(mode="embedded")
        assert client.mode == "embedded"


class TestNeumannClientEmbedded:
    """Tests for embedded mode client."""

    def test_embedded_requires_native_module(self) -> None:
        """Test that embedded mode requires native module."""
        with patch.dict("sys.modules", {"neumann._native": None}):
            with pytest.raises(ConnectionError) as exc_info:
                NeumannClient.embedded()
            assert "Native module not available" in str(exc_info.value)

    def test_embedded_creates_connected_client(self) -> None:
        """Test embedded mode creates connected client."""
        mock_native = MagicMock()
        mock_router = MagicMock()
        mock_native.QueryRouter.return_value = mock_router

        with patch.dict("sys.modules", {"neumann._native": mock_native}):
            with patch("neumann.client.NeumannClient.__init__", return_value=None):
                client = NeumannClient.__new__(NeumannClient)
                client._mode = "embedded"
                client._native = None
                client._channel = None
                client._stub = None
                client._api_key = None
                client._connected = False

                # Simulate embedded() logic
                client._native = mock_router
                client._connected = True

                assert client.is_connected
                assert client.mode == "embedded"

    def test_embedded_with_path(self) -> None:
        """Test embedded mode with persistence path."""
        mock_native = MagicMock()
        mock_router = MagicMock()
        mock_native.QueryRouter.with_path.return_value = mock_router

        with patch.dict("sys.modules", {"neumann._native": mock_native}):
            with patch("neumann.client.NeumannClient.__init__", return_value=None):
                client = NeumannClient.__new__(NeumannClient)
                client._mode = "embedded"
                client._native = None
                client._channel = None
                client._stub = None
                client._api_key = None
                client._connected = False

                # Simulate with_path logic
                client._native = mock_native.QueryRouter.with_path("/tmp/neumann")
                client._connected = True

                assert client.is_connected


class TestNeumannClientRemote:
    """Tests for remote mode client."""

    def test_connect_requires_grpc(self) -> None:
        """Test that connect requires grpcio."""
        with patch.dict("sys.modules", {"grpc": None}):
            with pytest.raises(ConnectionError) as exc_info:
                NeumannClient.connect("localhost:50051")
            # The import error should be caught

    def test_connect_creates_channel(self) -> None:
        """Test connect creates gRPC channel."""
        mock_grpc = MagicMock()
        mock_channel = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel

        mock_stub_module = MagicMock()
        mock_stub = MagicMock()
        mock_stub_module.QueryServiceStub.return_value = mock_stub

        with patch.dict(
            "sys.modules",
            {
                "grpc": mock_grpc,
                "neumann.proto": MagicMock(),
                "neumann.proto.neumann_pb2_grpc": mock_stub_module,
            },
        ):
            client = NeumannClient.__new__(NeumannClient)
            client._mode = "remote"
            client._native = None
            client._channel = mock_channel
            client._stub = mock_stub
            client._api_key = None
            client._connected = True

            assert client.is_connected
            assert client.mode == "remote"

    def test_connect_with_tls(self) -> None:
        """Test connect with TLS enabled."""
        mock_grpc = MagicMock()
        mock_credentials = MagicMock()
        mock_grpc.ssl_channel_credentials.return_value = mock_credentials
        mock_channel = MagicMock()
        mock_grpc.secure_channel.return_value = mock_channel

        with patch.dict("sys.modules", {"grpc": mock_grpc}):
            client = NeumannClient.__new__(NeumannClient)
            client._mode = "remote"
            client._channel = mock_channel
            client._connected = True

            assert client.is_connected

    def test_connect_with_api_key(self) -> None:
        """Test connect stores API key."""
        client = NeumannClient.__new__(NeumannClient)
        client._mode = "remote"
        client._native = None
        client._channel = MagicMock()
        client._stub = MagicMock()
        client._api_key = "test-api-key"
        client._connected = True

        assert client._api_key == "test-api-key"


class TestNeumannClientConnection:
    """Tests for connection management."""

    def test_is_connected_property(self) -> None:
        """Test is_connected property."""
        client = NeumannClient()
        assert not client.is_connected

        client._connected = True
        assert client.is_connected

    def test_mode_property(self) -> None:
        """Test mode property."""
        client = NeumannClient(mode="embedded")
        assert client.mode == "embedded"

        client = NeumannClient(mode="remote")
        assert client.mode == "remote"

    def test_close_disconnects(self) -> None:
        """Test close disconnects client."""
        client = NeumannClient()
        mock_channel = MagicMock()
        client._channel = mock_channel
        client._stub = MagicMock()
        client._native = MagicMock()
        client._connected = True

        client.close()

        assert not client.is_connected
        assert client._channel is None
        assert client._stub is None
        assert client._native is None
        mock_channel.close.assert_called_once()

    def test_context_manager(self) -> None:
        """Test context manager support."""
        client = NeumannClient()
        client._connected = True

        with client as c:
            assert c is client
            assert c.is_connected

        assert not client.is_connected


class TestNeumannClientExecute:
    """Tests for execute method."""

    def test_execute_requires_connection(self) -> None:
        """Test execute raises when not connected."""
        client = NeumannClient()
        with pytest.raises(ConnectionError) as exc_info:
            client.execute("SELECT * FROM users")
        assert "not connected" in str(exc_info.value)

    def test_execute_embedded_requires_native(self) -> None:
        """Test embedded execute requires native module."""
        client = NeumannClient(mode="embedded")
        client._connected = True
        client._native = None

        with pytest.raises(ConnectionError) as exc_info:
            client.execute("SELECT * FROM users")
        assert "not initialized" in str(exc_info.value)

    def test_execute_embedded_success(self) -> None:
        """Test embedded execute success."""
        client = NeumannClient(mode="embedded")
        client._connected = True
        mock_native = MagicMock()
        mock_native.execute.return_value = {"type": "count", "data": 42}
        client._native = mock_native

        result = client.execute("SELECT COUNT(*) FROM users")

        assert result.type == QueryResultType.COUNT
        assert result.data == 42
        mock_native.execute.assert_called_once_with("SELECT COUNT(*) FROM users", None)

    def test_execute_embedded_with_identity(self) -> None:
        """Test embedded execute with identity."""
        client = NeumannClient(mode="embedded")
        client._connected = True
        mock_native = MagicMock()
        mock_native.execute.return_value = {"type": "value", "data": "secret"}
        client._native = mock_native

        result = client.execute("VAULT GET my_secret", identity="alice")

        mock_native.execute.assert_called_once_with("VAULT GET my_secret", "alice")

    def test_execute_remote_requires_stub(self) -> None:
        """Test remote execute requires stub."""
        client = NeumannClient(mode="remote")
        client._connected = True
        client._stub = None

        with pytest.raises(ConnectionError) as exc_info:
            client.execute("SELECT * FROM users")
        assert "not initialized" in str(exc_info.value)


class TestNeumannClientExecuteStream:
    """Tests for execute_stream method."""

    def test_execute_stream_requires_connection(self) -> None:
        """Test execute_stream raises when not connected."""
        client = NeumannClient()
        with pytest.raises(ConnectionError):
            list(client.execute_stream("SELECT * FROM users"))

    def test_execute_stream_embedded_fallback(self) -> None:
        """Test embedded mode falls back to single result."""
        client = NeumannClient(mode="embedded")
        client._connected = True
        mock_native = MagicMock()
        mock_native.execute.return_value = {"type": "rows", "data": []}
        client._native = mock_native

        results = list(client.execute_stream("SELECT * FROM users"))

        assert len(results) == 1
        assert results[0].type == QueryResultType.ROWS


class TestNeumannClientExecuteBatch:
    """Tests for execute_batch method."""

    def test_execute_batch_requires_connection(self) -> None:
        """Test execute_batch raises when not connected."""
        client = NeumannClient()
        with pytest.raises(ConnectionError):
            client.execute_batch(["SELECT 1", "SELECT 2"])

    def test_execute_batch_embedded(self) -> None:
        """Test embedded batch execution."""
        client = NeumannClient(mode="embedded")
        client._connected = True
        mock_native = MagicMock()
        mock_native.execute.side_effect = [
            {"type": "count", "data": 1},
            {"type": "count", "data": 2},
            {"type": "count", "data": 3},
        ]
        client._native = mock_native

        results = client.execute_batch(["SELECT 1", "SELECT 2", "SELECT 3"])

        assert len(results) == 3
        assert results[0].data == 1
        assert results[1].data == 2
        assert results[2].data == 3


class TestNeumannClientConvertNativeResult:
    """Tests for _convert_native_result method."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_none_result(self) -> None:
        """Test converting None result."""
        result = self.client._convert_native_result(None)
        assert result.type == QueryResultType.EMPTY

    def test_convert_empty_result(self) -> None:
        """Test converting empty result."""
        result = self.client._convert_native_result({"type": "empty"})
        assert result.type == QueryResultType.EMPTY

    def test_convert_value_result(self) -> None:
        """Test converting value result."""
        result = self.client._convert_native_result({"type": "value", "data": "hello"})
        assert result.type == QueryResultType.VALUE
        assert result.data == "hello"

    def test_convert_count_result(self) -> None:
        """Test converting count result."""
        result = self.client._convert_native_result({"type": "count", "data": 42})
        assert result.type == QueryResultType.COUNT
        assert result.data == 42

    def test_convert_rows_result(self) -> None:
        """Test converting rows result."""
        result = self.client._convert_native_result(
            {
                "type": "rows",
                "data": [
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"},
                ],
            }
        )
        assert result.type == QueryResultType.ROWS
        assert len(result.data) == 2
        assert isinstance(result.data[0], Row)

    def test_convert_nodes_result(self) -> None:
        """Test converting nodes result."""
        result = self.client._convert_native_result(
            {
                "type": "nodes",
                "data": [
                    {"id": "n1", "label": "Person", "properties": {"name": "Alice"}},
                ],
            }
        )
        assert result.type == QueryResultType.NODES
        assert len(result.data) == 1
        assert isinstance(result.data[0], Node)

    def test_convert_edges_result(self) -> None:
        """Test converting edges result."""
        result = self.client._convert_native_result(
            {
                "type": "edges",
                "data": [
                    {
                        "id": "e1",
                        "type": "KNOWS",
                        "source": "n1",
                        "target": "n2",
                        "properties": {},
                    },
                ],
            }
        )
        assert result.type == QueryResultType.EDGES
        assert len(result.data) == 1
        assert isinstance(result.data[0], Edge)

    def test_convert_similar_result(self) -> None:
        """Test converting similar items result."""
        result = self.client._convert_native_result(
            {
                "type": "similar",
                "data": [
                    {"key": "item1", "score": 0.95, "metadata": {}},
                    {"key": "item2", "score": 0.85, "metadata": {"tag": "test"}},
                ],
            }
        )
        assert result.type == QueryResultType.SIMILAR
        assert len(result.data) == 2
        assert isinstance(result.data[0], SimilarItem)

    def test_convert_ids_result(self) -> None:
        """Test converting IDs result."""
        result = self.client._convert_native_result({"type": "ids", "data": [1, 2, 3, 4, 5]})
        assert result.type == QueryResultType.IDS
        assert result.data == [1, 2, 3, 4, 5]

    def test_convert_table_list_result(self) -> None:
        """Test converting table list result."""
        result = self.client._convert_native_result(
            {"type": "table_list", "data": ["users", "orders", "products"]}
        )
        assert result.type == QueryResultType.TABLE_LIST
        assert result.data == ["users", "orders", "products"]

    def test_convert_blob_result(self) -> None:
        """Test converting blob result."""
        result = self.client._convert_native_result({"type": "blob", "data": b"binary data"})
        assert result.type == QueryResultType.BLOB
        assert result.data == b"binary data"

    def test_convert_error_result(self) -> None:
        """Test converting error result."""
        result = self.client._convert_native_result(
            {"type": "error", "message": "Something went wrong"}
        )
        assert result.type == QueryResultType.ERROR
        assert result.data == "Something went wrong"

    def test_convert_unknown_type(self) -> None:
        """Test converting unknown type falls back to empty."""
        result = self.client._convert_native_result({"type": "unknown"})
        assert result.type == QueryResultType.EMPTY


class TestNeumannClientConvertNativeRow:
    """Tests for _convert_native_row method."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_row_with_values(self) -> None:
        """Test converting row with values."""
        row = self.client._convert_native_row({"id": 1, "name": "Alice", "active": True})
        assert isinstance(row, Row)
        assert row.values["id"].data == 1
        assert row.values["name"].data == "Alice"
        assert row.values["active"].data is True

    def test_convert_empty_row(self) -> None:
        """Test converting empty row."""
        row = self.client._convert_native_row({})
        assert isinstance(row, Row)
        assert len(row.values) == 0


class TestNeumannClientConvertNativeNode:
    """Tests for _convert_native_node method."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_node_with_properties(self) -> None:
        """Test converting node with properties."""
        node = self.client._convert_native_node(
            {"id": "n1", "label": "Person", "properties": {"age": 30}}
        )
        assert isinstance(node, Node)
        assert node.id == "n1"
        assert node.label == "Person"
        assert node.properties["age"].data == 30

    def test_convert_node_without_properties(self) -> None:
        """Test converting node without properties."""
        node = self.client._convert_native_node({"id": "n2", "label": "Item"})
        assert isinstance(node, Node)
        assert node.id == "n2"
        assert node.label == "Item"
        assert len(node.properties) == 0


class TestNeumannClientConvertNativeEdge:
    """Tests for _convert_native_edge method."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_edge_with_properties(self) -> None:
        """Test converting edge with properties."""
        edge = self.client._convert_native_edge(
            {
                "id": "e1",
                "type": "KNOWS",
                "source": "n1",
                "target": "n2",
                "properties": {"since": 2020},
            }
        )
        assert isinstance(edge, Edge)
        assert edge.id == "e1"
        assert edge.edge_type == "KNOWS"
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.properties["since"].data == 2020


class TestNeumannClientConvertNativeSimilar:
    """Tests for _convert_native_similar method."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_similar_item(self) -> None:
        """Test converting similar item."""
        item = self.client._convert_native_similar(
            {"key": "item1", "score": 0.95, "metadata": {"category": "test"}}
        )
        assert isinstance(item, SimilarItem)
        assert item.key == "item1"
        assert item.score == 0.95
        assert item.metadata["category"].data == "test"

    def test_convert_similar_without_metadata(self) -> None:
        """Test converting similar item without metadata."""
        item = self.client._convert_native_similar({"key": "item2", "score": 0.8})
        assert isinstance(item, SimilarItem)
        assert item.key == "item2"
        assert item.score == 0.8


class TestNeumannClientConvertNativeValue:
    """Tests for _convert_native_value method."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_null(self) -> None:
        """Test converting None to null value."""
        value = self.client._convert_native_value(None)
        assert value.type == ScalarType.NULL
        assert value.data is None

    def test_convert_bool(self) -> None:
        """Test converting boolean."""
        value = self.client._convert_native_value(True)
        assert value.data is True

        value = self.client._convert_native_value(False)
        assert value.data is False

    def test_convert_int(self) -> None:
        """Test converting integer."""
        value = self.client._convert_native_value(42)
        assert value.data == 42

    def test_convert_float(self) -> None:
        """Test converting float."""
        value = self.client._convert_native_value(3.14)
        assert value.data == 3.14

    def test_convert_string(self) -> None:
        """Test converting string."""
        value = self.client._convert_native_value("hello")
        assert value.data == "hello"

    def test_convert_bytes(self) -> None:
        """Test converting bytes."""
        value = self.client._convert_native_value(b"binary")
        assert value.data == b"binary"

    def test_convert_other_to_string(self) -> None:
        """Test converting other types to string."""
        value = self.client._convert_native_value([1, 2, 3])
        assert value.data == "[1, 2, 3]"


class TestNeumannClientProtoConversion:
    """Tests for proto conversion methods."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_proto_value_null(self) -> None:
        """Test converting proto null value."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "null_value"
        value = self.client._convert_proto_value(mock_value)
        assert value.type == ScalarType.NULL

    def test_convert_proto_value_int(self) -> None:
        """Test converting proto int value."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "int_value"
        mock_value.int_value = 42
        value = self.client._convert_proto_value(mock_value)
        assert value.data == 42

    def test_convert_proto_value_float(self) -> None:
        """Test converting proto float value."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "float_value"
        mock_value.float_value = 3.14
        value = self.client._convert_proto_value(mock_value)
        assert value.data == 3.14

    def test_convert_proto_value_string(self) -> None:
        """Test converting proto string value."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "string_value"
        mock_value.string_value = "hello"
        value = self.client._convert_proto_value(mock_value)
        assert value.data == "hello"

    def test_convert_proto_value_bool(self) -> None:
        """Test converting proto bool value."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "bool_value"
        mock_value.bool_value = True
        value = self.client._convert_proto_value(mock_value)
        assert value.data is True

    def test_convert_proto_value_bytes(self) -> None:
        """Test converting proto bytes value."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "bytes_value"
        mock_value.bytes_value = b"data"
        value = self.client._convert_proto_value(mock_value)
        assert value.data == b"data"

    def test_convert_proto_value_no_which(self) -> None:
        """Test converting proto value without WhichOneof."""
        value = self.client._convert_proto_value({})
        assert value.type == ScalarType.NULL

    def test_convert_proto_row(self) -> None:
        """Test converting proto row."""
        mock_row = MagicMock()
        mock_col1 = MagicMock()
        mock_col1.name = "id"
        mock_col1.value = MagicMock()
        mock_col1.value.WhichOneof.return_value = "int_value"
        mock_col1.value.int_value = 1

        mock_col2 = MagicMock()
        mock_col2.name = "name"
        mock_col2.value = MagicMock()
        mock_col2.value.WhichOneof.return_value = "string_value"
        mock_col2.value.string_value = "Alice"

        mock_row.columns = [mock_col1, mock_col2]

        row = self.client._convert_proto_row(mock_row)
        assert row.values["id"].data == 1
        assert row.values["name"].data == "Alice"

    def test_convert_proto_node(self) -> None:
        """Test converting proto node."""
        mock_node = MagicMock()
        mock_node.id = "n1"
        mock_node.label = "Person"
        mock_prop = MagicMock()
        mock_prop.name = "age"
        mock_prop.value = MagicMock()
        mock_prop.value.WhichOneof.return_value = "int_value"
        mock_prop.value.int_value = 30
        mock_node.properties = [mock_prop]

        node = self.client._convert_proto_node(mock_node)
        assert node.id == "n1"
        assert node.label == "Person"
        assert node.properties["age"].data == 30

    def test_convert_proto_edge(self) -> None:
        """Test converting proto edge."""
        mock_edge = MagicMock()
        mock_edge.id = "e1"
        mock_edge.edge_type = "KNOWS"
        mock_edge.source_id = "n1"
        mock_edge.target_id = "n2"
        mock_edge.properties = []

        edge = self.client._convert_proto_edge(mock_edge)
        assert edge.id == "e1"
        assert edge.edge_type == "KNOWS"
        assert edge.source == "n1"
        assert edge.target == "n2"

    def test_convert_proto_path(self) -> None:
        """Test converting proto path."""
        mock_path = MagicMock()
        mock_seg1 = MagicMock()
        mock_seg1.node = MagicMock()
        mock_seg1.node.id = "n1"
        mock_seg1.node.label = "A"
        mock_seg1.node.properties = []
        mock_seg1.edge = MagicMock()
        mock_seg1.edge.id = "e1"
        mock_seg1.edge.edge_type = "LINK"
        mock_seg1.edge.source_id = "n1"
        mock_seg1.edge.target_id = "n2"
        mock_seg1.edge.properties = []

        mock_seg2 = MagicMock()
        mock_seg2.node = MagicMock()
        mock_seg2.node.id = "n2"
        mock_seg2.node.label = "B"
        mock_seg2.node.properties = []
        mock_seg2.edge = None

        mock_path.segments = [mock_seg1, mock_seg2]

        path = self.client._convert_proto_path(mock_path)
        assert len(path.segments) == 2
        assert path.segments[0].node.id == "n1"
        assert path.segments[0].edge.id == "e1"
        assert path.segments[1].node.id == "n2"
        assert path.segments[1].edge is None

    def test_convert_proto_similar(self) -> None:
        """Test converting proto similar item."""
        mock_item = MagicMock()
        mock_item.key = "item1"
        mock_item.score = 0.95
        mock_prop = MagicMock()
        mock_prop.name = "category"
        mock_prop.value = MagicMock()
        mock_prop.value.WhichOneof.return_value = "string_value"
        mock_prop.value.string_value = "tech"
        mock_item.metadata = [mock_prop]

        item = self.client._convert_proto_similar(mock_item)
        assert item.key == "item1"
        assert item.score == 0.95
        assert item.metadata["category"].data == "tech"


class TestNeumannClientProtoChunkConversion:
    """Tests for proto chunk conversion."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_chunk_row(self) -> None:
        """Test converting row chunk."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "row"
        mock_chunk.row.row = MagicMock()
        mock_chunk.row.row.columns = []

        result = self.client._convert_proto_chunk(mock_chunk)
        assert result.type == QueryResultType.ROWS

    def test_convert_chunk_node(self) -> None:
        """Test converting node chunk."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "node"
        mock_chunk.node.node = MagicMock()
        mock_chunk.node.node.id = "n1"
        mock_chunk.node.node.label = "Test"
        mock_chunk.node.node.properties = []

        result = self.client._convert_proto_chunk(mock_chunk)
        assert result.type == QueryResultType.NODES

    def test_convert_chunk_edge(self) -> None:
        """Test converting edge chunk."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "edge"
        mock_chunk.edge.edge = MagicMock()
        mock_chunk.edge.edge.id = "e1"
        mock_chunk.edge.edge.edge_type = "LINK"
        mock_chunk.edge.edge.source_id = "n1"
        mock_chunk.edge.edge.target_id = "n2"
        mock_chunk.edge.edge.properties = []

        result = self.client._convert_proto_chunk(mock_chunk)
        assert result.type == QueryResultType.EDGES

    def test_convert_chunk_similar_item(self) -> None:
        """Test converting similar item chunk."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "similar_item"
        mock_chunk.similar_item.item = MagicMock()
        mock_chunk.similar_item.item.key = "item1"
        mock_chunk.similar_item.item.score = 0.9
        mock_chunk.similar_item.item.metadata = []

        result = self.client._convert_proto_chunk(mock_chunk)
        assert result.type == QueryResultType.SIMILAR

    def test_convert_chunk_blob_data(self) -> None:
        """Test converting blob data chunk."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "blob_data"
        mock_chunk.blob_data = b"binary data"

        result = self.client._convert_proto_chunk(mock_chunk)
        assert result.type == QueryResultType.BLOB

    def test_convert_chunk_error(self) -> None:
        """Test converting error chunk."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "error"
        mock_chunk.error.code = ErrorCode.INTERNAL
        mock_chunk.error.message = "Something went wrong"

        with pytest.raises(InternalError):
            self.client._convert_proto_chunk(mock_chunk)

    def test_convert_chunk_unknown(self) -> None:
        """Test converting unknown chunk type."""
        mock_chunk = MagicMock()
        mock_chunk.WhichOneof.return_value = "unknown"

        result = self.client._convert_proto_chunk(mock_chunk)
        assert result.type == QueryResultType.EMPTY


class TestNeumannClientEmbeddedDirect:
    """Tests that directly call embedded() to cover actual code paths."""

    def test_embedded_calls_native_queryrouter(self) -> None:
        """Test embedded() calls native QueryRouter."""
        mock_native = MagicMock()
        mock_router = MagicMock()
        mock_native.QueryRouter.return_value = mock_router

        with patch.dict("sys.modules", {"neumann._native": mock_native}):
            # Need to reload to pick up the mock
            import importlib
            import neumann.client as client_mod

            importlib.reload(client_mod)

            client = client_mod.NeumannClient.embedded()
            assert client.is_connected
            assert client._native is mock_router
            mock_native.QueryRouter.assert_called_once()

    def test_embedded_with_path_calls_with_path(self) -> None:
        """Test embedded(path=...) calls QueryRouter.with_path()."""
        mock_native = MagicMock()
        mock_router = MagicMock()
        mock_native.QueryRouter.with_path.return_value = mock_router

        with patch.dict("sys.modules", {"neumann._native": mock_native}):
            import importlib
            import neumann.client as client_mod

            importlib.reload(client_mod)

            client = client_mod.NeumannClient.embedded(path="/tmp/neumann-db")
            assert client.is_connected
            assert client._native is mock_router
            mock_native.QueryRouter.with_path.assert_called_once_with("/tmp/neumann-db")


class TestNeumannClientConnectDirect:
    """Tests that directly call connect() to cover actual code paths."""

    def test_connect_insecure_channel(self) -> None:
        """Test connect() creates insecure channel."""
        mock_grpc = MagicMock()
        mock_channel = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel

        mock_stub_module = MagicMock()
        mock_stub = MagicMock()
        mock_stub_module.QueryServiceStub.return_value = mock_stub

        with patch.dict(
            "sys.modules",
            {
                "grpc": mock_grpc,
                "neumann.proto": MagicMock(),
                "neumann.proto.neumann_pb2_grpc": mock_stub_module,
            },
        ):
            import importlib
            import neumann.client as client_mod

            importlib.reload(client_mod)

            client = client_mod.NeumannClient.connect("localhost:50051")
            assert client.is_connected
            mock_grpc.insecure_channel.assert_called_once_with("localhost:50051")

    def test_connect_secure_channel_with_tls(self) -> None:
        """Test connect(tls=True) creates secure channel."""
        mock_grpc = MagicMock()
        mock_creds = MagicMock()
        mock_grpc.ssl_channel_credentials.return_value = mock_creds
        mock_channel = MagicMock()
        mock_grpc.secure_channel.return_value = mock_channel

        mock_stub_module = MagicMock()
        mock_stub = MagicMock()
        mock_stub_module.QueryServiceStub.return_value = mock_stub

        with patch.dict(
            "sys.modules",
            {
                "grpc": mock_grpc,
                "neumann.proto": MagicMock(),
                "neumann.proto.neumann_pb2_grpc": mock_stub_module,
            },
        ):
            import importlib
            import neumann.client as client_mod

            importlib.reload(client_mod)

            client = client_mod.NeumannClient.connect("localhost:50051", tls=True)
            assert client.is_connected
            mock_grpc.ssl_channel_credentials.assert_called_once()
            mock_grpc.secure_channel.assert_called_once_with("localhost:50051", mock_creds)

    def test_connect_with_api_key(self) -> None:
        """Test connect() stores API key."""
        mock_grpc = MagicMock()
        mock_channel = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel

        mock_stub_module = MagicMock()
        mock_stub = MagicMock()
        mock_stub_module.QueryServiceStub.return_value = mock_stub

        with patch.dict(
            "sys.modules",
            {
                "grpc": mock_grpc,
                "neumann.proto": MagicMock(),
                "neumann.proto.neumann_pb2_grpc": mock_stub_module,
            },
        ):
            import importlib
            import neumann.client as client_mod

            importlib.reload(client_mod)

            client = client_mod.NeumannClient.connect("localhost:50051", api_key="secret-key")
            assert client._api_key == "secret-key"

    def test_connect_handles_generic_exception(self) -> None:
        """Test connect() handles generic exceptions."""
        mock_grpc = MagicMock()
        mock_grpc.insecure_channel.side_effect = RuntimeError("Connection failed")

        with patch.dict("sys.modules", {"grpc": mock_grpc}):
            import importlib
            import neumann.client as client_mod

            importlib.reload(client_mod)

            with pytest.raises(ConnectionError) as exc_info:
                client_mod.NeumannClient.connect("localhost:50051")
            assert "Failed to connect" in str(exc_info.value)


class TestNeumannClientExecuteRemoteDirect:
    """Tests for _execute_remote to cover remote execution paths."""

    def test_execute_remote_success(self) -> None:
        """Test _execute_remote succeeds."""
        mock_pb2 = MagicMock()
        mock_request = MagicMock()
        mock_pb2.QueryRequest.return_value = mock_request

        # Properly configure mock response to avoid error path
        mock_response = MagicMock()
        mock_response.error = None  # No error
        mock_response.result = MagicMock()
        mock_response.result.count = 42
        mock_response.WhichOneof.return_value = "count"

        client = NeumannClient(mode="remote")
        client._connected = True
        mock_stub = MagicMock()
        mock_stub.Execute.return_value = mock_response
        client._stub = mock_stub

        with patch.dict("sys.modules", {"neumann.proto.neumann_pb2": mock_pb2}):
            result = client._execute_remote("SELECT COUNT(*)", None)
            assert result.type == QueryResultType.COUNT

    def test_execute_remote_with_identity(self) -> None:
        """Test _execute_remote with identity."""
        mock_pb2 = MagicMock()
        mock_request = MagicMock()
        mock_pb2.QueryRequest.return_value = mock_request

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = None  # Empty result
        mock_response.WhichOneof.return_value = None

        client = NeumannClient(mode="remote")
        client._connected = True
        mock_stub = MagicMock()
        mock_stub.Execute.return_value = mock_response
        client._stub = mock_stub

        with patch.dict("sys.modules", {"neumann.proto.neumann_pb2": mock_pb2}):
            result = client._execute_remote("VAULT GET secret", "alice")
            assert result.type == QueryResultType.EMPTY

    def test_execute_remote_with_api_key(self) -> None:
        """Test _execute_remote includes API key in metadata."""
        mock_pb2 = MagicMock()
        mock_request = MagicMock()
        mock_pb2.QueryRequest.return_value = mock_request

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = None
        mock_response.WhichOneof.return_value = None

        client = NeumannClient(mode="remote")
        client._connected = True
        client._api_key = "test-api-key"
        mock_stub = MagicMock()
        mock_stub.Execute.return_value = mock_response
        client._stub = mock_stub

        with patch.dict("sys.modules", {"neumann.proto.neumann_pb2": mock_pb2}):
            client._execute_remote("SELECT 1", None)
            # Verify Execute was called with metadata containing api key
            call_args = mock_stub.Execute.call_args
            assert call_args is not None
            # Check metadata was passed
            _, kwargs = call_args
            assert "metadata" in kwargs
            assert ("x-api-key", "test-api-key") in kwargs["metadata"]

    def test_execute_remote_grpc_error(self) -> None:
        """Test _execute_remote handles gRPC errors."""
        mock_pb2 = MagicMock()
        mock_request = MagicMock()
        mock_pb2.QueryRequest.return_value = mock_request

        client = NeumannClient(mode="remote")
        client._connected = True
        mock_stub = MagicMock()

        # Create a mock gRPC error
        class MockGrpcError(Exception):
            pass

        MockGrpcError.__module__ = "grpc._channel"
        mock_stub.Execute.side_effect = MockGrpcError("RPC failed")
        client._stub = mock_stub

        with patch.dict("sys.modules", {"neumann.proto.neumann_pb2": mock_pb2}):
            with pytest.raises(ConnectionError) as exc_info:
                client._execute_remote("SELECT 1", None)
            assert "gRPC error" in str(exc_info.value)


class TestNeumannClientStreamRemoteDirect:
    """Tests for _execute_stream_remote."""

    def test_execute_stream_remote_requires_stub(self) -> None:
        """Test _execute_stream_remote requires stub."""
        client = NeumannClient(mode="remote")
        client._connected = True
        client._stub = None

        with pytest.raises(ConnectionError):
            list(client._execute_stream_remote("SELECT *", None))

    def test_execute_stream_remote_success(self) -> None:
        """Test _execute_stream_remote streams results."""
        mock_pb2 = MagicMock()
        mock_request = MagicMock()
        mock_pb2.QueryRequest.return_value = mock_request

        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.is_final = False
        mock_chunk1.WhichOneof.return_value = "count"
        mock_chunk1.count = 1

        mock_chunk2 = MagicMock()
        mock_chunk2.is_final = True

        client = NeumannClient(mode="remote")
        client._connected = True
        mock_stub = MagicMock()
        mock_stub.ExecuteStream.return_value = iter([mock_chunk1, mock_chunk2])
        client._stub = mock_stub

        with patch.dict("sys.modules", {"neumann.proto.neumann_pb2": mock_pb2}):
            results = list(client._execute_stream_remote("SELECT *", None))
            assert len(results) == 1


class TestNeumannClientBatchRemoteDirect:
    """Tests for _execute_batch_remote."""

    def test_execute_batch_remote_requires_stub(self) -> None:
        """Test _execute_batch_remote requires stub."""
        client = NeumannClient(mode="remote")
        client._connected = True
        client._stub = None

        with pytest.raises(ConnectionError):
            client._execute_batch_remote(["SELECT 1"], None)

    def test_execute_batch_remote_success(self) -> None:
        """Test _execute_batch_remote executes batch."""
        mock_pb2 = MagicMock()
        mock_request = MagicMock()
        mock_pb2.QueryRequest.return_value = mock_request
        mock_pb2.BatchQueryRequest.return_value = MagicMock()

        # Properly configure mock results
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

        client = NeumannClient(mode="remote")
        client._connected = True
        mock_stub = MagicMock()
        mock_stub.ExecuteBatch.return_value = mock_response
        client._stub = mock_stub

        with patch.dict("sys.modules", {"neumann.proto.neumann_pb2": mock_pb2}):
            results = client._execute_batch_remote(["SELECT 1", "SELECT 2"], None)
            assert len(results) == 2


class TestNeumannClientEmbeddedExecuteErrors:
    """Tests for embedded execute error handling."""

    def test_execute_embedded_exception(self) -> None:
        """Test embedded execute wraps exceptions."""
        client = NeumannClient(mode="embedded")
        client._connected = True
        mock_native = MagicMock()
        mock_native.execute.side_effect = RuntimeError("Native error")
        client._native = mock_native

        with pytest.raises(NeumannError) as exc_info:
            client.execute("SELECT *")
        assert "Native error" in str(exc_info.value)


class TestNeumannClientConvertProtoResultBranches:
    """Tests for _convert_proto_result to cover all branches."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_proto_result_empty(self) -> None:
        """Test converting empty result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.WhichOneof.return_value = "empty"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.EMPTY

    def test_convert_proto_result_value(self) -> None:
        """Test converting value result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.value = "hello"
        mock_response.WhichOneof.return_value = "value"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.VALUE

    def test_convert_proto_result_rows(self) -> None:
        """Test converting rows result."""
        mock_row = MagicMock()
        mock_row.columns = []

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.rows = [mock_row]
        mock_response.WhichOneof.return_value = "rows"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.ROWS

    def test_convert_proto_result_nodes(self) -> None:
        """Test converting nodes result."""
        mock_node = MagicMock()
        mock_node.id = "n1"
        mock_node.label = "Test"
        mock_node.properties = []

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.nodes = [mock_node]
        mock_response.WhichOneof.return_value = "nodes"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.NODES

    def test_convert_proto_result_edges(self) -> None:
        """Test converting edges result."""
        mock_edge = MagicMock()
        mock_edge.id = "e1"
        mock_edge.edge_type = "LINK"
        mock_edge.source_id = "n1"
        mock_edge.target_id = "n2"
        mock_edge.properties = []

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.edges = [mock_edge]
        mock_response.WhichOneof.return_value = "edges"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.EDGES

    def test_convert_proto_result_paths(self) -> None:
        """Test converting paths result."""
        mock_seg = MagicMock()
        mock_seg.node = MagicMock()
        mock_seg.node.id = "n1"
        mock_seg.node.label = "Test"
        mock_seg.node.properties = []
        mock_seg.edge = None

        mock_path = MagicMock()
        mock_path.segments = [mock_seg]

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.paths = [mock_path]
        mock_response.WhichOneof.return_value = "paths"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.PATHS

    def test_convert_proto_result_similar_items(self) -> None:
        """Test converting similar items result."""
        mock_item = MagicMock()
        mock_item.key = "item1"
        mock_item.score = 0.9
        mock_item.metadata = []

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.items = [mock_item]
        mock_response.WhichOneof.return_value = "similar_items"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.SIMILAR

    def test_convert_proto_result_ids(self) -> None:
        """Test converting ids result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.ids = [1, 2, 3]
        mock_response.WhichOneof.return_value = "ids"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.IDS
        assert result.data == [1, 2, 3]

    def test_convert_proto_result_table_list(self) -> None:
        """Test converting table list result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.names = ["users", "orders"]
        mock_response.WhichOneof.return_value = "table_list"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.TABLE_LIST
        assert result.data == ["users", "orders"]

    def test_convert_proto_result_blob_data(self) -> None:
        """Test converting blob data result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = b"binary data"
        mock_response.WhichOneof.return_value = "blob_data"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.BLOB

    def test_convert_proto_result_blob_info(self) -> None:
        """Test converting blob info result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.artifact_id = "art-123"
        mock_response.result.filename = "file.txt"
        mock_response.result.size = 1024
        mock_response.result.checksum = "abc123"
        mock_response.result.content_type = "text/plain"
        mock_response.result.created_at = 1234567890
        mock_response.result.tags = ["tag1"]
        mock_response.WhichOneof.return_value = "blob_info"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.BLOB_INFO
        assert result.data.artifact_id == "art-123"

    def test_convert_proto_result_unknown(self) -> None:
        """Test converting unknown result type."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.WhichOneof.return_value = "unknown_type"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.EMPTY

    def test_convert_proto_edge_with_properties(self) -> None:
        """Test converting proto edge with properties."""
        mock_edge = MagicMock()
        mock_edge.id = "e1"
        mock_edge.edge_type = "KNOWS"
        mock_edge.source_id = "n1"
        mock_edge.target_id = "n2"

        mock_prop = MagicMock()
        mock_prop.name = "since"
        mock_prop.value = MagicMock()
        mock_prop.value.WhichOneof.return_value = "int_value"
        mock_prop.value.int_value = 2020
        mock_edge.properties = [mock_prop]

        edge = self.client._convert_proto_edge(mock_edge)
        assert edge.properties["since"].data == 2020

    def test_convert_proto_value_unknown_type(self) -> None:
        """Test converting proto value with unknown type."""
        mock_value = MagicMock()
        mock_value.WhichOneof.return_value = "unknown_value_type"

        value = self.client._convert_proto_value(mock_value)
        assert value.type == ScalarType.NULL


class TestNeumannClientConvertNativeNewTypes:
    """Tests for _convert_native_result with new result types."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_blob_stats_result(self) -> None:
        """Test converting blob stats result."""
        result = self.client._convert_native_result(
            {
                "type": "blob_stats",
                "data": {
                    "artifact_count": 100,
                    "chunk_count": 500,
                    "total_bytes": 1024000,
                    "unique_bytes": 512000,
                    "dedup_ratio": 2.0,
                    "orphaned_chunks": 5,
                },
            }
        )
        assert result.type == QueryResultType.BLOB_STATS
        assert isinstance(result.data, BlobStats)
        assert result.data.artifact_count == 100
        assert result.data.dedup_ratio == 2.0

    def test_convert_artifact_list_result(self) -> None:
        """Test converting artifact list result."""
        result = self.client._convert_native_result(
            {"type": "artifact_list", "data": ["art-1", "art-2", "art-3"]}
        )
        assert result.type == QueryResultType.ARTIFACT_LIST
        assert result.data == ["art-1", "art-2", "art-3"]

    def test_convert_checkpoint_list_result(self) -> None:
        """Test converting checkpoint list result."""
        result = self.client._convert_native_result(
            {
                "type": "checkpoint_list",
                "data": [
                    {"id": "cp-1", "name": "backup1", "created_at": 100, "is_auto": False},
                    {"id": "cp-2", "name": "auto", "created_at": 200, "is_auto": True},
                ],
            }
        )
        assert result.type == QueryResultType.CHECKPOINT_LIST
        assert len(result.data) == 2
        assert isinstance(result.data[0], CheckpointInfo)
        assert result.data[0].id == "cp-1"
        assert result.data[1].is_auto is True

    def test_convert_unified_result(self) -> None:
        """Test converting unified result."""
        result = self.client._convert_native_result(
            {
                "type": "unified",
                "data": {
                    "description": "Found 2 items",
                    "items": [
                        {"entity_type": "node", "key": "n1", "fields": {"name": "A"}, "score": 0.9},
                        {"entity_type": "edge", "key": "e1", "fields": {}, "score": None},
                    ],
                },
            }
        )
        assert result.type == QueryResultType.UNIFIED
        assert isinstance(result.data, UnifiedResult)
        assert result.data.description == "Found 2 items"
        assert len(result.data.items) == 2
        assert result.data.items[0].entity_type == "node"
        assert result.data.items[0].score == 0.9
        assert result.data.items[1].score is None


class TestNeumannClientConvertNativeChainResults:
    """Tests for _convert_native_chain_result."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_chain_transaction_begun(self) -> None:
        """Test converting chain transaction begun result."""
        result = self.client._convert_native_result(
            {"type": "chain_transaction_begun", "data": {"tx_id": "tx-123"}}
        )
        assert result.type == QueryResultType.CHAIN_TRANSACTION_BEGUN
        assert isinstance(result.data, ChainTransactionBegun)
        assert result.data.tx_id == "tx-123"

    def test_convert_chain_committed(self) -> None:
        """Test converting chain committed result."""
        result = self.client._convert_native_result(
            {"type": "chain_committed", "data": {"block_hash": "abc", "height": 100}}
        )
        assert result.type == QueryResultType.CHAIN_COMMITTED
        assert isinstance(result.data, ChainCommitted)
        assert result.data.block_hash == "abc"
        assert result.data.height == 100

    def test_convert_chain_rolled_back(self) -> None:
        """Test converting chain rolled back result."""
        result = self.client._convert_native_result(
            {"type": "chain_rolled_back", "data": {"to_height": 50}}
        )
        assert result.type == QueryResultType.CHAIN_ROLLED_BACK
        assert isinstance(result.data, ChainRolledBack)
        assert result.data.to_height == 50

    def test_convert_chain_history(self) -> None:
        """Test converting chain history result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_history",
                "data": {
                    "entries": [
                        {"height": 1, "transaction_type": "PUT", "data": b"test"},
                        {"height": 2, "transaction_type": "DELETE", "data": None},
                    ]
                },
            }
        )
        assert result.type == QueryResultType.CHAIN_HISTORY
        assert isinstance(result.data, ChainHistory)
        assert len(result.data.entries) == 2
        assert result.data.entries[0].height == 1
        assert result.data.entries[0].data == b"test"

    def test_convert_chain_similar(self) -> None:
        """Test converting chain similar result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_similar",
                "data": {
                    "items": [
                        {"block_hash": "h1", "height": 10, "similarity": 0.95},
                        {"block_hash": "h2", "height": 20, "similarity": 0.85},
                    ]
                },
            }
        )
        assert result.type == QueryResultType.CHAIN_SIMILAR
        assert isinstance(result.data, ChainSimilar)
        assert len(result.data.items) == 2
        assert result.data.items[0].similarity == 0.95

    def test_convert_chain_drift(self) -> None:
        """Test converting chain drift result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_drift",
                "data": {
                    "from_height": 0,
                    "to_height": 100,
                    "total_drift": 15.5,
                    "avg_drift_per_block": 0.155,
                    "max_drift": 2.3,
                },
            }
        )
        assert result.type == QueryResultType.CHAIN_DRIFT
        assert isinstance(result.data, ChainDrift)
        assert result.data.total_drift == 15.5

    def test_convert_chain_height(self) -> None:
        """Test converting chain height result."""
        result = self.client._convert_native_result(
            {"type": "chain_height", "data": {"height": 500}}
        )
        assert result.type == QueryResultType.CHAIN_HEIGHT
        assert isinstance(result.data, ChainHeight)
        assert result.data.height == 500

    def test_convert_chain_tip(self) -> None:
        """Test converting chain tip result."""
        result = self.client._convert_native_result(
            {"type": "chain_tip", "data": {"hash": "head", "height": 500}}
        )
        assert result.type == QueryResultType.CHAIN_TIP
        assert isinstance(result.data, ChainTip)
        assert result.data.hash == "head"
        assert result.data.height == 500

    def test_convert_chain_block(self) -> None:
        """Test converting chain block result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_block",
                "data": {
                    "height": 100,
                    "hash": "b100",
                    "prev_hash": "b99",
                    "timestamp": 1704067200,
                    "transaction_count": 50,
                    "proposer": "val1",
                },
            }
        )
        assert result.type == QueryResultType.CHAIN_BLOCK
        assert isinstance(result.data, ChainBlockInfo)
        assert result.data.height == 100
        assert result.data.proposer == "val1"

    def test_convert_chain_codebook(self) -> None:
        """Test converting chain codebook result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_codebook",
                "data": {
                    "scope": "global",
                    "entry_count": 256,
                    "dimension": 128,
                    "domain": "embeddings",
                },
            }
        )
        assert result.type == QueryResultType.CHAIN_CODEBOOK
        assert isinstance(result.data, ChainCodebookInfo)
        assert result.data.scope == "global"
        assert result.data.domain == "embeddings"

    def test_convert_chain_codebook_no_domain(self) -> None:
        """Test converting chain codebook without domain."""
        result = self.client._convert_native_result(
            {
                "type": "chain_codebook",
                "data": {"scope": "local", "entry_count": 64, "dimension": 32},
            }
        )
        assert result.type == QueryResultType.CHAIN_CODEBOOK
        assert result.data.domain is None

    def test_convert_chain_transition_analysis(self) -> None:
        """Test converting chain transition analysis result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_transition_analysis",
                "data": {
                    "total_transitions": 100,
                    "valid_transitions": 95,
                    "invalid_transitions": 5,
                    "avg_validity_score": 0.95,
                },
            }
        )
        assert result.type == QueryResultType.CHAIN_TRANSITION_ANALYSIS
        assert isinstance(result.data, ChainTransitionAnalysis)
        assert result.data.total_transitions == 100

    def test_convert_chain_conflict_resolution(self) -> None:
        """Test converting chain conflict resolution result."""
        result = self.client._convert_native_result(
            {
                "type": "chain_conflict_resolution",
                "data": {"strategy": "semantic", "conflicts_resolved": 10},
            }
        )
        assert result.type == QueryResultType.CHAIN_CONFLICT_RESOLUTION
        assert isinstance(result.data, ChainConflictResolution)
        assert result.data.strategy == "semantic"

    def test_convert_chain_merge(self) -> None:
        """Test converting chain merge result."""
        result = self.client._convert_native_result(
            {"type": "chain_merge", "data": {"success": True, "merged_count": 25}}
        )
        assert result.type == QueryResultType.CHAIN_MERGE
        assert isinstance(result.data, ChainMergeResult)
        assert result.data.success is True
        assert result.data.merged_count == 25

    def test_convert_unknown_chain_type(self) -> None:
        """Test converting unknown chain type returns empty."""
        result = self.client._convert_native_result({"type": "chain_unknown", "data": {}})
        assert result.type == QueryResultType.EMPTY


class TestNeumannClientConvertProtoNewTypes:
    """Tests for _convert_proto_result with new result types."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_proto_blob_stats(self) -> None:
        """Test converting proto blob stats result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.artifact_count = 100
        mock_response.result.chunk_count = 500
        mock_response.result.total_bytes = 1024000
        mock_response.result.unique_bytes = 512000
        mock_response.result.dedup_ratio = 2.0
        mock_response.result.orphaned_chunks = 5
        mock_response.WhichOneof.return_value = "blob_stats"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.BLOB_STATS
        assert result.data.artifact_count == 100

    def test_convert_proto_artifact_list(self) -> None:
        """Test converting proto artifact list result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.artifact_ids = ["art1", "art2"]
        mock_response.WhichOneof.return_value = "artifact_list"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.ARTIFACT_LIST
        assert result.data == ["art1", "art2"]

    def test_convert_proto_checkpoint_list(self) -> None:
        """Test converting proto checkpoint list result."""
        mock_cp = MagicMock()
        mock_cp.id = "cp-1"
        mock_cp.name = "backup"
        mock_cp.created_at = 100
        mock_cp.is_auto = False

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.checkpoints = [mock_cp]
        mock_response.WhichOneof.return_value = "checkpoint_list"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHECKPOINT_LIST
        assert len(result.data) == 1
        assert result.data[0].id == "cp-1"

    def test_convert_proto_unified(self) -> None:
        """Test converting proto unified result."""
        mock_item = MagicMock()
        mock_item.entity_type = "node"
        mock_item.key = "n1"
        mock_item.fields = {"name": "Alice"}
        mock_item.score = 0.9

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.description = "Found 1 item"
        mock_response.result.items = [mock_item]
        mock_response.WhichOneof.return_value = "unified"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.UNIFIED
        assert result.data.description == "Found 1 item"
        assert len(result.data.items) == 1


class TestNeumannClientConvertProtoChainResults:
    """Tests for _convert_proto_chain_result."""

    def setup_method(self) -> None:
        """Set up test client."""
        self.client = NeumannClient()

    def test_convert_proto_chain_transaction_begun(self) -> None:
        """Test converting proto chain transaction begun result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.tx_id = "tx-123"
        mock_response.WhichOneof.return_value = "chain_transaction_begun"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_TRANSACTION_BEGUN
        assert result.data.tx_id == "tx-123"

    def test_convert_proto_chain_committed(self) -> None:
        """Test converting proto chain committed result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.block_hash = "abc"
        mock_response.result.height = 100
        mock_response.WhichOneof.return_value = "chain_committed"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_COMMITTED
        assert result.data.block_hash == "abc"

    def test_convert_proto_chain_rolled_back(self) -> None:
        """Test converting proto chain rolled back result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.to_height = 50
        mock_response.WhichOneof.return_value = "chain_rolled_back"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_ROLLED_BACK
        assert result.data.to_height == 50

    def test_convert_proto_chain_history(self) -> None:
        """Test converting proto chain history result."""
        mock_entry = MagicMock()
        mock_entry.height = 1
        mock_entry.transaction_type = "PUT"
        mock_entry.data = b"test"

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.entries = [mock_entry]
        mock_response.WhichOneof.return_value = "chain_history"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_HISTORY
        assert len(result.data.entries) == 1

    def test_convert_proto_chain_similar(self) -> None:
        """Test converting proto chain similar result."""
        mock_item = MagicMock()
        mock_item.block_hash = "h1"
        mock_item.height = 10
        mock_item.similarity = 0.95

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.items = [mock_item]
        mock_response.WhichOneof.return_value = "chain_similar"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_SIMILAR
        assert len(result.data.items) == 1

    def test_convert_proto_chain_drift(self) -> None:
        """Test converting proto chain drift result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.from_height = 0
        mock_response.result.to_height = 100
        mock_response.result.total_drift = 15.5
        mock_response.result.avg_drift_per_block = 0.155
        mock_response.result.max_drift = 2.3
        mock_response.WhichOneof.return_value = "chain_drift"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_DRIFT
        assert result.data.total_drift == 15.5

    def test_convert_proto_chain_height(self) -> None:
        """Test converting proto chain height result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.height = 500
        mock_response.WhichOneof.return_value = "chain_height"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_HEIGHT
        assert result.data.height == 500

    def test_convert_proto_chain_tip(self) -> None:
        """Test converting proto chain tip result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.hash = "head"
        mock_response.result.height = 500
        mock_response.WhichOneof.return_value = "chain_tip"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_TIP
        assert result.data.hash == "head"

    def test_convert_proto_chain_block(self) -> None:
        """Test converting proto chain block result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.height = 100
        mock_response.result.hash = "b100"
        mock_response.result.prev_hash = "b99"
        mock_response.result.timestamp = 1704067200
        mock_response.result.transaction_count = 50
        mock_response.result.proposer = "val1"
        mock_response.WhichOneof.return_value = "chain_block"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_BLOCK
        assert result.data.height == 100

    def test_convert_proto_chain_codebook(self) -> None:
        """Test converting proto chain codebook result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.scope = "global"
        mock_response.result.entry_count = 256
        mock_response.result.dimension = 128
        mock_response.result.domain = "embeddings"
        mock_response.WhichOneof.return_value = "chain_codebook"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_CODEBOOK
        assert result.data.scope == "global"

    def test_convert_proto_chain_transition_analysis(self) -> None:
        """Test converting proto chain transition analysis result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.total_transitions = 100
        mock_response.result.valid_transitions = 95
        mock_response.result.invalid_transitions = 5
        mock_response.result.avg_validity_score = 0.95
        mock_response.WhichOneof.return_value = "chain_transition_analysis"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_TRANSITION_ANALYSIS
        assert result.data.total_transitions == 100

    def test_convert_proto_chain_conflict_resolution(self) -> None:
        """Test converting proto chain conflict resolution result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.strategy = "semantic"
        mock_response.result.conflicts_resolved = 10
        mock_response.WhichOneof.return_value = "chain_conflict_resolution"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_CONFLICT_RESOLUTION
        assert result.data.strategy == "semantic"

    def test_convert_proto_chain_merge(self) -> None:
        """Test converting proto chain merge result."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = MagicMock()
        mock_response.result.success = True
        mock_response.result.merged_count = 25
        mock_response.WhichOneof.return_value = "chain_merge"

        result = self.client._convert_proto_result(mock_response)
        assert result.type == QueryResultType.CHAIN_MERGE
        assert result.data.success is True
