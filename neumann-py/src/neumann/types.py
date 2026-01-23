"""Core data types for Neumann database."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScalarType(Enum):
    """Scalar value types."""

    NULL = "null"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    BYTES = "bytes"


@dataclass(frozen=True)
class Value:
    """A typed scalar value."""

    type: ScalarType
    data: int | float | str | bool | bytes | None

    @classmethod
    def null(cls) -> Value:
        """Create a null value."""
        return cls(ScalarType.NULL, None)

    @classmethod
    def int_(cls, v: int) -> Value:
        """Create an integer value."""
        return cls(ScalarType.INT, v)

    @classmethod
    def float_(cls, v: float) -> Value:
        """Create a float value."""
        return cls(ScalarType.FLOAT, v)

    @classmethod
    def string(cls, v: str) -> Value:
        """Create a string value."""
        return cls(ScalarType.STRING, v)

    @classmethod
    def bool_(cls, v: bool) -> Value:
        """Create a boolean value."""
        return cls(ScalarType.BOOL, v)

    @classmethod
    def bytes_(cls, v: bytes) -> Value:
        """Create a bytes value."""
        return cls(ScalarType.BYTES, v)

    def as_python(self) -> int | float | str | bool | bytes | None:
        """Convert to native Python type."""
        return self.data


@dataclass
class Row:
    """A row from a relational query result."""

    values: dict[str, Value] = field(default_factory=dict)

    def get(self, column: str) -> Value | None:
        """Get a value by column name."""
        return self.values.get(column)

    def get_int(self, column: str) -> int | None:
        """Get an integer value by column name."""
        val = self.values.get(column)
        if val and val.type == ScalarType.INT:
            return int(val.data) if val.data is not None else None
        return None

    def get_float(self, column: str) -> float | None:
        """Get a float value by column name."""
        val = self.values.get(column)
        if val and val.type == ScalarType.FLOAT:
            return float(val.data) if val.data is not None else None
        return None

    def get_string(self, column: str) -> str | None:
        """Get a string value by column name."""
        val = self.values.get(column)
        if val and val.type == ScalarType.STRING:
            return str(val.data) if val.data is not None else None
        return None

    def get_bool(self, column: str) -> bool | None:
        """Get a boolean value by column name."""
        val = self.values.get(column)
        if val and val.type == ScalarType.BOOL:
            return bool(val.data) if val.data is not None else None
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert row to a dictionary with Python native types."""
        return {k: v.as_python() for k, v in self.values.items()}


@dataclass
class Node:
    """A graph node."""

    id: str
    label: str
    properties: dict[str, Value] = field(default_factory=dict)

    def get_property(self, name: str) -> Value | None:
        """Get a property by name."""
        return self.properties.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert node to a dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "properties": {k: v.as_python() for k, v in self.properties.items()},
        }


@dataclass
class Edge:
    """A graph edge."""

    id: str
    edge_type: str
    source: str
    target: str
    properties: dict[str, Value] = field(default_factory=dict)

    def get_property(self, name: str) -> Value | None:
        """Get a property by name."""
        return self.properties.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert edge to a dictionary."""
        return {
            "id": self.id,
            "type": self.edge_type,
            "source": self.source,
            "target": self.target,
            "properties": {k: v.as_python() for k, v in self.properties.items()},
        }


@dataclass
class PathSegment:
    """A segment in a graph path (node + optional edge)."""

    node: Node
    edge: Edge | None = None


@dataclass
class Path:
    """A path through a graph."""

    segments: list[PathSegment] = field(default_factory=list)

    @property
    def nodes(self) -> list[Node]:
        """Get all nodes in the path."""
        return [s.node for s in self.segments]

    @property
    def edges(self) -> list[Edge]:
        """Get all edges in the path."""
        return [s.edge for s in self.segments if s.edge is not None]

    def __len__(self) -> int:
        """Return number of segments."""
        return len(self.segments)


@dataclass
class SimilarItem:
    """A similarity search result."""

    key: str
    score: float
    metadata: dict[str, Value] = field(default_factory=dict)


@dataclass
class ArtifactInfo:
    """Metadata for a blob artifact."""

    artifact_id: str
    filename: str
    size: int
    checksum: str
    content_type: str
    created_at: int
    tags: list[str] = field(default_factory=list)


class QueryResultType(Enum):
    """Types of query results."""

    EMPTY = "empty"
    VALUE = "value"
    COUNT = "count"
    ROWS = "rows"
    NODES = "nodes"
    EDGES = "edges"
    PATHS = "paths"
    SIMILAR = "similar"
    IDS = "ids"
    TABLE_LIST = "table_list"
    BLOB = "blob"
    BLOB_INFO = "blob_info"
    CHAIN_COMMITTED = "chain_committed"
    CHAIN_TRANSACTION = "chain_transaction"
    UNIFIED = "unified"
    ERROR = "error"


@dataclass
class QueryResult:
    """Result from a query execution."""

    type: QueryResultType
    data: Any = None

    @property
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return self.type == QueryResultType.EMPTY

    @property
    def is_error(self) -> bool:
        """Check if result is an error."""
        return self.type == QueryResultType.ERROR

    @property
    def value(self) -> str | None:
        """Get value if result is a single value."""
        if self.type == QueryResultType.VALUE:
            return str(self.data) if self.data else None
        return None

    @property
    def count(self) -> int | None:
        """Get count if result is a count."""
        if self.type == QueryResultType.COUNT:
            return int(self.data) if self.data else None
        return None

    @property
    def rows(self) -> list[Row]:
        """Get rows if result is rows."""
        if self.type == QueryResultType.ROWS:
            return self.data if self.data else []
        return []

    @property
    def nodes(self) -> list[Node]:
        """Get nodes if result is nodes."""
        if self.type == QueryResultType.NODES:
            return self.data if self.data else []
        return []

    @property
    def edges(self) -> list[Edge]:
        """Get edges if result is edges."""
        if self.type == QueryResultType.EDGES:
            return self.data if self.data else []
        return []

    @property
    def paths(self) -> list[Path]:
        """Get paths if result is paths."""
        if self.type == QueryResultType.PATHS:
            return self.data if self.data else []
        return []

    @property
    def similar_items(self) -> list[SimilarItem]:
        """Get similar items if result is similar."""
        if self.type == QueryResultType.SIMILAR:
            return self.data if self.data else []
        return []

    @property
    def ids(self) -> list[str]:
        """Get IDs if result is IDs."""
        if self.type == QueryResultType.IDS:
            return self.data if self.data else []
        return []

    @property
    def table_names(self) -> list[str]:
        """Get table names if result is a table list."""
        if self.type == QueryResultType.TABLE_LIST:
            return self.data if self.data else []
        return []

    @property
    def blob_data(self) -> bytes | None:
        """Get blob data if result is blob."""
        if self.type == QueryResultType.BLOB:
            return self.data if self.data else None
        return None

    @property
    def blob_info(self) -> ArtifactInfo | None:
        """Get blob info if result is blob info."""
        if self.type == QueryResultType.BLOB_INFO:
            return self.data if self.data else None
        return None

    @property
    def error_message(self) -> str | None:
        """Get error message if result is error."""
        if self.type == QueryResultType.ERROR:
            return str(self.data) if self.data else None
        return None
