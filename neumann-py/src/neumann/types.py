# SPDX-License-Identifier: MIT
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


@dataclass(frozen=True)
class UnifiedItem:
    """An item from a unified query result."""

    entity_type: str
    key: str
    fields: dict[str, str] = field(default_factory=dict)
    score: float | None = None


@dataclass(frozen=True)
class UnifiedResult:
    """Result from a unified cross-engine query."""

    description: str
    items: list[UnifiedItem] = field(default_factory=list)


@dataclass(frozen=True)
class BlobStats:
    """Statistics for blob storage."""

    artifact_count: int
    chunk_count: int
    total_bytes: int
    unique_bytes: int
    dedup_ratio: float
    orphaned_chunks: int


@dataclass(frozen=True)
class CheckpointInfo:
    """Information about a checkpoint."""

    id: str
    name: str
    created_at: int
    is_auto: bool


@dataclass(frozen=True)
class ChainTransactionBegun:
    """Result when a chain transaction begins."""

    tx_id: str


@dataclass(frozen=True)
class ChainCommitted:
    """Result when a chain commit succeeds."""

    block_hash: str
    height: int


@dataclass(frozen=True)
class ChainRolledBack:
    """Result when a chain rollback completes."""

    to_height: int


@dataclass(frozen=True)
class ChainHistoryEntry:
    """An entry in chain history."""

    height: int
    transaction_type: str
    data: bytes | None = None


@dataclass(frozen=True)
class ChainHistory:
    """Chain history result."""

    entries: list[ChainHistoryEntry] = field(default_factory=list)


@dataclass(frozen=True)
class ChainSimilarItem:
    """A similar item found in the chain."""

    block_hash: str
    height: int
    similarity: float


@dataclass(frozen=True)
class ChainSimilar:
    """Chain similarity search result."""

    items: list[ChainSimilarItem] = field(default_factory=list)


@dataclass(frozen=True)
class ChainDrift:
    """Chain drift analysis result."""

    from_height: int
    to_height: int
    total_drift: float
    avg_drift_per_block: float
    max_drift: float


@dataclass(frozen=True)
class ChainHeight:
    """Current chain height."""

    height: int


@dataclass(frozen=True)
class ChainTip:
    """Current chain tip information."""

    hash: str
    height: int


@dataclass(frozen=True)
class ChainBlockInfo:
    """Information about a chain block."""

    height: int
    hash: str
    prev_hash: str
    timestamp: int
    transaction_count: int
    proposer: str


@dataclass(frozen=True)
class ChainCodebookInfo:
    """Information about a chain codebook."""

    scope: str
    entry_count: int
    dimension: int
    domain: str | None = None


@dataclass(frozen=True)
class ChainTransitionAnalysis:
    """Analysis of chain transitions."""

    total_transitions: int
    valid_transitions: int
    invalid_transitions: int
    avg_validity_score: float


@dataclass(frozen=True)
class ChainConflictResolution:
    """Result of chain conflict resolution."""

    strategy: str
    conflicts_resolved: int


@dataclass(frozen=True)
class ChainMergeResult:
    """Result of a chain merge operation."""

    success: bool
    merged_count: int


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
    BLOB_STATS = "blob_stats"
    ARTIFACT_LIST = "artifact_list"
    CHECKPOINT_LIST = "checkpoint_list"
    CHAIN_COMMITTED = "chain_committed"
    CHAIN_TRANSACTION = "chain_transaction"
    CHAIN_TRANSACTION_BEGUN = "chain_transaction_begun"
    CHAIN_ROLLED_BACK = "chain_rolled_back"
    CHAIN_HISTORY = "chain_history"
    CHAIN_SIMILAR = "chain_similar"
    CHAIN_DRIFT = "chain_drift"
    CHAIN_HEIGHT = "chain_height"
    CHAIN_TIP = "chain_tip"
    CHAIN_BLOCK = "chain_block"
    CHAIN_CODEBOOK = "chain_codebook"
    CHAIN_TRANSITION_ANALYSIS = "chain_transition_analysis"
    CHAIN_CONFLICT_RESOLUTION = "chain_conflict_resolution"
    CHAIN_MERGE = "chain_merge"
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

    @property
    def unified_result(self) -> UnifiedResult | None:
        """Get unified result if result is unified."""
        if self.type == QueryResultType.UNIFIED:
            return self.data if self.data else None
        return None

    @property
    def unified_items(self) -> list[UnifiedItem]:
        """Get unified items if result is unified."""
        if self.type == QueryResultType.UNIFIED and self.data:
            return self.data.items if hasattr(self.data, "items") else []
        return []

    @property
    def unified_description(self) -> str | None:
        """Get unified description if result is unified."""
        if self.type == QueryResultType.UNIFIED and self.data:
            return self.data.description if hasattr(self.data, "description") else None
        return None

    @property
    def artifact_ids(self) -> list[str]:
        """Get artifact IDs if result is artifact list."""
        if self.type == QueryResultType.ARTIFACT_LIST:
            return self.data if self.data else []
        return []

    @property
    def blob_stats(self) -> BlobStats | None:
        """Get blob stats if result is blob stats."""
        if self.type == QueryResultType.BLOB_STATS:
            return self.data if self.data else None
        return None

    @property
    def checkpoints(self) -> list[CheckpointInfo]:
        """Get checkpoints if result is checkpoint list."""
        if self.type == QueryResultType.CHECKPOINT_LIST:
            return self.data if self.data else []
        return []

    @property
    def chain_transaction_begun(self) -> ChainTransactionBegun | None:
        """Get chain transaction begun if result is chain transaction begun."""
        if self.type == QueryResultType.CHAIN_TRANSACTION_BEGUN:
            return self.data if self.data else None
        return None

    @property
    def chain_committed(self) -> ChainCommitted | None:
        """Get chain committed if result is chain committed."""
        if self.type == QueryResultType.CHAIN_COMMITTED:
            return self.data if self.data else None
        return None

    @property
    def chain_rolled_back(self) -> ChainRolledBack | None:
        """Get chain rolled back if result is chain rolled back."""
        if self.type == QueryResultType.CHAIN_ROLLED_BACK:
            return self.data if self.data else None
        return None

    @property
    def chain_history(self) -> ChainHistory | None:
        """Get chain history if result is chain history."""
        if self.type == QueryResultType.CHAIN_HISTORY:
            return self.data if self.data else None
        return None

    @property
    def chain_similar(self) -> ChainSimilar | None:
        """Get chain similar if result is chain similar."""
        if self.type == QueryResultType.CHAIN_SIMILAR:
            return self.data if self.data else None
        return None

    @property
    def chain_drift(self) -> ChainDrift | None:
        """Get chain drift if result is chain drift."""
        if self.type == QueryResultType.CHAIN_DRIFT:
            return self.data if self.data else None
        return None

    @property
    def chain_height(self) -> ChainHeight | None:
        """Get chain height if result is chain height."""
        if self.type == QueryResultType.CHAIN_HEIGHT:
            return self.data if self.data else None
        return None

    @property
    def chain_tip(self) -> ChainTip | None:
        """Get chain tip if result is chain tip."""
        if self.type == QueryResultType.CHAIN_TIP:
            return self.data if self.data else None
        return None

    @property
    def chain_block(self) -> ChainBlockInfo | None:
        """Get chain block if result is chain block."""
        if self.type == QueryResultType.CHAIN_BLOCK:
            return self.data if self.data else None
        return None

    @property
    def chain_codebook(self) -> ChainCodebookInfo | None:
        """Get chain codebook if result is chain codebook."""
        if self.type == QueryResultType.CHAIN_CODEBOOK:
            return self.data if self.data else None
        return None

    @property
    def chain_transition_analysis(self) -> ChainTransitionAnalysis | None:
        """Get chain transition analysis if result is chain transition analysis."""
        if self.type == QueryResultType.CHAIN_TRANSITION_ANALYSIS:
            return self.data if self.data else None
        return None

    @property
    def chain_conflict_resolution(self) -> ChainConflictResolution | None:
        """Get chain conflict resolution if result is chain conflict resolution."""
        if self.type == QueryResultType.CHAIN_CONFLICT_RESOLUTION:
            return self.data if self.data else None
        return None

    @property
    def chain_merge(self) -> ChainMergeResult | None:
        """Get chain merge result if result is chain merge."""
        if self.type == QueryResultType.CHAIN_MERGE:
            return self.data if self.data else None
        return None
