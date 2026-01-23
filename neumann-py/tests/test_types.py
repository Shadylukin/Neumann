"""Tests for Neumann data types."""

import pytest

from neumann.types import (
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


class TestValue:
    """Tests for Value class."""

    def test_null_value(self) -> None:
        """Test null value creation."""
        v = Value.null()
        assert v.type == ScalarType.NULL
        assert v.data is None
        assert v.as_python() is None

    def test_int_value(self) -> None:
        """Test integer value creation."""
        v = Value.int_(42)
        assert v.type == ScalarType.INT
        assert v.data == 42
        assert v.as_python() == 42

    def test_float_value(self) -> None:
        """Test float value creation."""
        v = Value.float_(3.14)
        assert v.type == ScalarType.FLOAT
        assert v.data == 3.14
        assert v.as_python() == 3.14

    def test_string_value(self) -> None:
        """Test string value creation."""
        v = Value.string("hello")
        assert v.type == ScalarType.STRING
        assert v.data == "hello"
        assert v.as_python() == "hello"

    def test_bool_value(self) -> None:
        """Test boolean value creation."""
        v = Value.bool_(True)
        assert v.type == ScalarType.BOOL
        assert v.data is True
        assert v.as_python() is True

    def test_bytes_value(self) -> None:
        """Test bytes value creation."""
        v = Value.bytes_(b"data")
        assert v.type == ScalarType.BYTES
        assert v.data == b"data"
        assert v.as_python() == b"data"

    def test_value_is_frozen(self) -> None:
        """Test that Value is immutable."""
        v = Value.int_(42)
        with pytest.raises(AttributeError):
            v.data = 100  # type: ignore


class TestRow:
    """Tests for Row class."""

    def test_empty_row(self) -> None:
        """Test empty row creation."""
        row = Row()
        assert len(row.values) == 0
        assert row.get("missing") is None

    def test_row_with_values(self) -> None:
        """Test row with values."""
        row = Row(
            values={
                "name": Value.string("Alice"),
                "age": Value.int_(30),
            }
        )
        assert row.get("name") == Value.string("Alice")
        assert row.get_string("name") == "Alice"
        assert row.get_int("age") == 30

    def test_row_to_dict(self) -> None:
        """Test row conversion to dict."""
        row = Row(
            values={
                "name": Value.string("Bob"),
                "active": Value.bool_(True),
            }
        )
        d = row.to_dict()
        assert d == {"name": "Bob", "active": True}

    def test_row_get_typed_methods(self) -> None:
        """Test typed getter methods."""
        row = Row(
            values={
                "int_val": Value.int_(100),
                "float_val": Value.float_(1.5),
                "str_val": Value.string("test"),
                "bool_val": Value.bool_(False),
            }
        )
        assert row.get_int("int_val") == 100
        assert row.get_float("float_val") == 1.5
        assert row.get_string("str_val") == "test"
        assert row.get_bool("bool_val") is False

        # Wrong type returns None
        assert row.get_int("str_val") is None
        assert row.get_string("int_val") is None


class TestNode:
    """Tests for Node class."""

    def test_node_creation(self) -> None:
        """Test node creation."""
        node = Node(
            id="n1",
            label="Person",
            properties={"name": Value.string("Alice")},
        )
        assert node.id == "n1"
        assert node.label == "Person"
        assert node.get_property("name") == Value.string("Alice")

    def test_node_to_dict(self) -> None:
        """Test node conversion to dict."""
        node = Node(
            id="n2",
            label="User",
            properties={"age": Value.int_(25)},
        )
        d = node.to_dict()
        assert d == {
            "id": "n2",
            "label": "User",
            "properties": {"age": 25},
        }


class TestEdge:
    """Tests for Edge class."""

    def test_edge_creation(self) -> None:
        """Test edge creation."""
        edge = Edge(
            id="e1",
            edge_type="KNOWS",
            source="n1",
            target="n2",
            properties={"since": Value.int_(2020)},
        )
        assert edge.id == "e1"
        assert edge.edge_type == "KNOWS"
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.get_property("since") == Value.int_(2020)

    def test_edge_to_dict(self) -> None:
        """Test edge conversion to dict."""
        edge = Edge(
            id="e2",
            edge_type="FOLLOWS",
            source="a",
            target="b",
        )
        d = edge.to_dict()
        assert d == {
            "id": "e2",
            "type": "FOLLOWS",
            "source": "a",
            "target": "b",
            "properties": {},
        }


class TestPath:
    """Tests for Path class."""

    def test_empty_path(self) -> None:
        """Test empty path."""
        path = Path()
        assert len(path) == 0
        assert path.nodes == []
        assert path.edges == []

    def test_path_with_segments(self) -> None:
        """Test path with segments."""
        n1 = Node(id="1", label="A")
        n2 = Node(id="2", label="B")
        e1 = Edge(id="e1", edge_type="R", source="1", target="2")

        path = Path(
            segments=[
                PathSegment(node=n1, edge=e1),
                PathSegment(node=n2),
            ]
        )

        assert len(path) == 2
        assert path.nodes == [n1, n2]
        assert path.edges == [e1]


class TestSimilarItem:
    """Tests for SimilarItem class."""

    def test_similar_item(self) -> None:
        """Test similar item creation."""
        item = SimilarItem(
            key="vec1",
            score=0.95,
            metadata={"source": Value.string("doc1")},
        )
        assert item.key == "vec1"
        assert item.score == 0.95
        assert item.metadata["source"] == Value.string("doc1")


class TestQueryResult:
    """Tests for QueryResult class."""

    def test_empty_result(self) -> None:
        """Test empty result."""
        result = QueryResult(QueryResultType.EMPTY)
        assert result.is_empty
        assert not result.is_error
        assert result.rows == []

    def test_rows_result(self) -> None:
        """Test rows result."""
        rows = [Row(values={"x": Value.int_(1)})]
        result = QueryResult(QueryResultType.ROWS, rows)
        assert result.type == QueryResultType.ROWS
        assert result.rows == rows
        assert result.nodes == []

    def test_count_result(self) -> None:
        """Test count result."""
        result = QueryResult(QueryResultType.COUNT, 42)
        assert result.count == 42

    def test_error_result(self) -> None:
        """Test error result."""
        result = QueryResult(QueryResultType.ERROR, "something went wrong")
        assert result.is_error
        assert result.error_message == "something went wrong"

    def test_ids_result(self) -> None:
        """Test IDs result."""
        result = QueryResult(QueryResultType.IDS, ["id1", "id2", "id3"])
        assert result.ids == ["id1", "id2", "id3"]

    def test_table_list_result(self) -> None:
        """Test table list result."""
        result = QueryResult(QueryResultType.TABLE_LIST, ["users", "products"])
        assert result.table_names == ["users", "products"]

    def test_nodes_result(self) -> None:
        """Test nodes result."""
        nodes = [Node(id="n1", label="Person")]
        result = QueryResult(QueryResultType.NODES, nodes)
        assert result.nodes == nodes

    def test_edges_result(self) -> None:
        """Test edges result."""
        edges = [Edge(id="e1", edge_type="KNOWS", source="n1", target="n2")]
        result = QueryResult(QueryResultType.EDGES, edges)
        assert result.edges == edges

    def test_paths_result(self) -> None:
        """Test paths result."""
        n1 = Node(id="n1", label="A")
        path = Path(segments=[PathSegment(node=n1)])
        result = QueryResult(QueryResultType.PATHS, [path])
        assert result.paths == [path]

    def test_similar_items_result(self) -> None:
        """Test similar items result."""
        items = [SimilarItem(key="k1", score=0.9)]
        result = QueryResult(QueryResultType.SIMILAR, items)
        assert result.similar_items == items

    def test_blob_data_result(self) -> None:
        """Test blob data result."""
        result = QueryResult(QueryResultType.BLOB, b"binary data")
        assert result.blob_data == b"binary data"

    def test_blob_info_result(self) -> None:
        """Test blob info result."""
        from neumann.types import ArtifactInfo

        info = ArtifactInfo(
            artifact_id="art-1",
            filename="file.txt",
            size=100,
            checksum="abc",
            content_type="text/plain",
            created_at=1234567890,
            tags=["tag1"],
        )
        result = QueryResult(QueryResultType.BLOB_INFO, info)
        assert result.blob_info == info

    def test_result_wrong_type_accessors(self) -> None:
        """Test accessors return empty/None for wrong types."""
        # COUNT result
        count_result = QueryResult(QueryResultType.COUNT, 42)
        assert count_result.nodes == []
        assert count_result.edges == []
        assert count_result.paths == []
        assert count_result.similar_items == []
        assert count_result.ids == []
        assert count_result.table_names == []
        assert count_result.blob_data is None
        assert count_result.blob_info is None
        assert count_result.error_message is None

    def test_result_none_data_accessors(self) -> None:
        """Test accessors return empty/None when data is None."""
        # ROWS with None data
        rows_result = QueryResult(QueryResultType.ROWS, None)
        assert rows_result.rows == []

        # COUNT with None data
        count_result = QueryResult(QueryResultType.COUNT, None)
        assert count_result.count is None


class TestRowGetters:
    """Tests for Row getter methods with edge cases."""

    def test_get_int_wrong_type(self) -> None:
        """Test get_int returns None for wrong type."""
        row = Row(values={"name": Value.string("Alice")})
        assert row.get_int("name") is None

    def test_get_float_wrong_type(self) -> None:
        """Test get_float returns None for wrong type."""
        row = Row(values={"name": Value.string("Alice")})
        assert row.get_float("name") is None

    def test_get_string_wrong_type(self) -> None:
        """Test get_string returns None for wrong type."""
        row = Row(values={"count": Value.int_(42)})
        assert row.get_string("count") is None

    def test_get_bool_wrong_type(self) -> None:
        """Test get_bool returns None for wrong type."""
        row = Row(values={"name": Value.string("Alice")})
        assert row.get_bool("name") is None

    def test_get_missing_column(self) -> None:
        """Test getters return None for missing columns."""
        row = Row(values={})
        assert row.get_int("missing") is None
        assert row.get_float("missing") is None
        assert row.get_string("missing") is None
        assert row.get_bool("missing") is None
