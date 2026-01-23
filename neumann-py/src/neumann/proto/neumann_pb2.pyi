from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CODE_UNSPECIFIED: _ClassVar[ErrorCode]
    ERROR_CODE_INVALID_QUERY: _ClassVar[ErrorCode]
    ERROR_CODE_NOT_FOUND: _ClassVar[ErrorCode]
    ERROR_CODE_PERMISSION_DENIED: _ClassVar[ErrorCode]
    ERROR_CODE_ALREADY_EXISTS: _ClassVar[ErrorCode]
    ERROR_CODE_INTERNAL: _ClassVar[ErrorCode]
    ERROR_CODE_UNAVAILABLE: _ClassVar[ErrorCode]
    ERROR_CODE_INVALID_ARGUMENT: _ClassVar[ErrorCode]
    ERROR_CODE_UNAUTHENTICATED: _ClassVar[ErrorCode]

class ServingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVING_STATUS_UNSPECIFIED: _ClassVar[ServingStatus]
    SERVING_STATUS_SERVING: _ClassVar[ServingStatus]
    SERVING_STATUS_NOT_SERVING: _ClassVar[ServingStatus]
ERROR_CODE_UNSPECIFIED: ErrorCode
ERROR_CODE_INVALID_QUERY: ErrorCode
ERROR_CODE_NOT_FOUND: ErrorCode
ERROR_CODE_PERMISSION_DENIED: ErrorCode
ERROR_CODE_ALREADY_EXISTS: ErrorCode
ERROR_CODE_INTERNAL: ErrorCode
ERROR_CODE_UNAVAILABLE: ErrorCode
ERROR_CODE_INVALID_ARGUMENT: ErrorCode
ERROR_CODE_UNAUTHENTICATED: ErrorCode
SERVING_STATUS_UNSPECIFIED: ServingStatus
SERVING_STATUS_SERVING: ServingStatus
SERVING_STATUS_NOT_SERVING: ServingStatus

class QueryRequest(_message.Message):
    __slots__ = ("query", "identity")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    query: str
    identity: str
    def __init__(self, query: _Optional[str] = ..., identity: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("empty", "value", "count", "ids", "rows", "nodes", "edges", "path", "similar", "unified", "table_list", "blob", "artifact_info", "artifact_list", "blob_stats", "checkpoint_list", "chain", "error")
    EMPTY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    SIMILAR_FIELD_NUMBER: _ClassVar[int]
    UNIFIED_FIELD_NUMBER: _ClassVar[int]
    TABLE_LIST_FIELD_NUMBER: _ClassVar[int]
    BLOB_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_INFO_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_LIST_FIELD_NUMBER: _ClassVar[int]
    BLOB_STATS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_LIST_FIELD_NUMBER: _ClassVar[int]
    CHAIN_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    empty: EmptyResult
    value: StringValue
    count: CountResult
    ids: IdsResult
    rows: RowsResult
    nodes: NodesResult
    edges: EdgesResult
    path: PathResult
    similar: SimilarResult
    unified: UnifiedQueryResult
    table_list: TableListResult
    blob: BlobResult
    artifact_info: ArtifactInfo
    artifact_list: ArtifactListResult
    blob_stats: BlobStatsResult
    checkpoint_list: CheckpointListResult
    chain: ChainQueryResult
    error: ErrorInfo
    def __init__(self, empty: _Optional[_Union[EmptyResult, _Mapping]] = ..., value: _Optional[_Union[StringValue, _Mapping]] = ..., count: _Optional[_Union[CountResult, _Mapping]] = ..., ids: _Optional[_Union[IdsResult, _Mapping]] = ..., rows: _Optional[_Union[RowsResult, _Mapping]] = ..., nodes: _Optional[_Union[NodesResult, _Mapping]] = ..., edges: _Optional[_Union[EdgesResult, _Mapping]] = ..., path: _Optional[_Union[PathResult, _Mapping]] = ..., similar: _Optional[_Union[SimilarResult, _Mapping]] = ..., unified: _Optional[_Union[UnifiedQueryResult, _Mapping]] = ..., table_list: _Optional[_Union[TableListResult, _Mapping]] = ..., blob: _Optional[_Union[BlobResult, _Mapping]] = ..., artifact_info: _Optional[_Union[ArtifactInfo, _Mapping]] = ..., artifact_list: _Optional[_Union[ArtifactListResult, _Mapping]] = ..., blob_stats: _Optional[_Union[BlobStatsResult, _Mapping]] = ..., checkpoint_list: _Optional[_Union[CheckpointListResult, _Mapping]] = ..., chain: _Optional[_Union[ChainQueryResult, _Mapping]] = ..., error: _Optional[_Union[ErrorInfo, _Mapping]] = ...) -> None: ...

class QueryResponseChunk(_message.Message):
    __slots__ = ("row", "node", "edge", "similar_item", "blob_data", "error", "is_final")
    ROW_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    EDGE_FIELD_NUMBER: _ClassVar[int]
    SIMILAR_ITEM_FIELD_NUMBER: _ClassVar[int]
    BLOB_DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    row: RowChunk
    node: NodeChunk
    edge: EdgeChunk
    similar_item: SimilarChunk
    blob_data: bytes
    error: ErrorInfo
    is_final: bool
    def __init__(self, row: _Optional[_Union[RowChunk, _Mapping]] = ..., node: _Optional[_Union[NodeChunk, _Mapping]] = ..., edge: _Optional[_Union[EdgeChunk, _Mapping]] = ..., similar_item: _Optional[_Union[SimilarChunk, _Mapping]] = ..., blob_data: _Optional[bytes] = ..., error: _Optional[_Union[ErrorInfo, _Mapping]] = ..., is_final: bool = ...) -> None: ...

class BatchQueryRequest(_message.Message):
    __slots__ = ("queries",)
    QUERIES_FIELD_NUMBER: _ClassVar[int]
    queries: _containers.RepeatedCompositeFieldContainer[QueryRequest]
    def __init__(self, queries: _Optional[_Iterable[_Union[QueryRequest, _Mapping]]] = ...) -> None: ...

class BatchQueryResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[QueryResponse]
    def __init__(self, results: _Optional[_Iterable[_Union[QueryResponse, _Mapping]]] = ...) -> None: ...

class EmptyResult(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StringValue(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: str
    def __init__(self, value: _Optional[str] = ...) -> None: ...

class CountResult(_message.Message):
    __slots__ = ("count",)
    COUNT_FIELD_NUMBER: _ClassVar[int]
    count: int
    def __init__(self, count: _Optional[int] = ...) -> None: ...

class IdsResult(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, ids: _Optional[_Iterable[int]] = ...) -> None: ...

class RowsResult(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[Row]
    def __init__(self, rows: _Optional[_Iterable[_Union[Row, _Mapping]]] = ...) -> None: ...

class Row(_message.Message):
    __slots__ = ("id", "values")
    ID_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    id: int
    values: _containers.RepeatedCompositeFieldContainer[ColumnValue]
    def __init__(self, id: _Optional[int] = ..., values: _Optional[_Iterable[_Union[ColumnValue, _Mapping]]] = ...) -> None: ...

class ColumnValue(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: Value
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ("null", "int_value", "float_value", "string_value", "bool_value")
    NULL_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    null: bool
    int_value: int
    float_value: float
    string_value: str
    bool_value: bool
    def __init__(self, null: bool = ..., int_value: _Optional[int] = ..., float_value: _Optional[float] = ..., string_value: _Optional[str] = ..., bool_value: bool = ...) -> None: ...

class NodesResult(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("id", "label", "properties")
    class PropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    id: int
    label: str
    properties: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[int] = ..., label: _Optional[str] = ..., properties: _Optional[_Mapping[str, str]] = ...) -> None: ...

class NodeChunk(_message.Message):
    __slots__ = ("node",)
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: Node
    def __init__(self, node: _Optional[_Union[Node, _Mapping]] = ...) -> None: ...

class EdgesResult(_message.Message):
    __slots__ = ("edges",)
    EDGES_FIELD_NUMBER: _ClassVar[int]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    def __init__(self, edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ...) -> None: ...

class Edge(_message.Message):
    __slots__ = ("id", "to", "label")
    ID_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    id: int
    to: int
    label: str
    def __init__(self, id: _Optional[int] = ..., to: _Optional[int] = ..., label: _Optional[str] = ..., **kwargs) -> None: ...

class EdgeChunk(_message.Message):
    __slots__ = ("edge",)
    EDGE_FIELD_NUMBER: _ClassVar[int]
    edge: Edge
    def __init__(self, edge: _Optional[_Union[Edge, _Mapping]] = ...) -> None: ...

class PathResult(_message.Message):
    __slots__ = ("node_ids",)
    NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    node_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, node_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class SimilarResult(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[SimilarItem]
    def __init__(self, items: _Optional[_Iterable[_Union[SimilarItem, _Mapping]]] = ...) -> None: ...

class SimilarItem(_message.Message):
    __slots__ = ("key", "score")
    KEY_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    key: str
    score: float
    def __init__(self, key: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class SimilarChunk(_message.Message):
    __slots__ = ("item",)
    ITEM_FIELD_NUMBER: _ClassVar[int]
    item: SimilarItem
    def __init__(self, item: _Optional[_Union[SimilarItem, _Mapping]] = ...) -> None: ...

class RowChunk(_message.Message):
    __slots__ = ("row",)
    ROW_FIELD_NUMBER: _ClassVar[int]
    row: Row
    def __init__(self, row: _Optional[_Union[Row, _Mapping]] = ...) -> None: ...

class UnifiedQueryResult(_message.Message):
    __slots__ = ("description", "items")
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    description: str
    items: _containers.RepeatedCompositeFieldContainer[UnifiedItem]
    def __init__(self, description: _Optional[str] = ..., items: _Optional[_Iterable[_Union[UnifiedItem, _Mapping]]] = ...) -> None: ...

class UnifiedItem(_message.Message):
    __slots__ = ("entity_type", "key", "fields", "score")
    class FieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    entity_type: str
    key: str
    fields: _containers.ScalarMap[str, str]
    score: float
    def __init__(self, entity_type: _Optional[str] = ..., key: _Optional[str] = ..., fields: _Optional[_Mapping[str, str]] = ..., score: _Optional[float] = ...) -> None: ...

class TableListResult(_message.Message):
    __slots__ = ("tables",)
    TABLES_FIELD_NUMBER: _ClassVar[int]
    tables: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, tables: _Optional[_Iterable[str]] = ...) -> None: ...

class BlobResult(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class ArtifactInfo(_message.Message):
    __slots__ = ("id", "filename", "content_type", "size", "checksum", "chunk_count", "created", "modified", "created_by", "tags", "linked_to", "custom")
    class CustomEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    LINKED_TO_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_NUMBER: _ClassVar[int]
    id: str
    filename: str
    content_type: str
    size: int
    checksum: str
    chunk_count: int
    created: int
    modified: int
    created_by: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    linked_to: _containers.RepeatedScalarFieldContainer[str]
    custom: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., filename: _Optional[str] = ..., content_type: _Optional[str] = ..., size: _Optional[int] = ..., checksum: _Optional[str] = ..., chunk_count: _Optional[int] = ..., created: _Optional[int] = ..., modified: _Optional[int] = ..., created_by: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., linked_to: _Optional[_Iterable[str]] = ..., custom: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ArtifactListResult(_message.Message):
    __slots__ = ("artifact_ids",)
    ARTIFACT_IDS_FIELD_NUMBER: _ClassVar[int]
    artifact_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, artifact_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class BlobStatsResult(_message.Message):
    __slots__ = ("artifact_count", "chunk_count", "total_bytes", "unique_bytes", "dedup_ratio", "orphaned_chunks")
    ARTIFACT_COUNT_FIELD_NUMBER: _ClassVar[int]
    CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEDUP_RATIO_FIELD_NUMBER: _ClassVar[int]
    ORPHANED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    artifact_count: int
    chunk_count: int
    total_bytes: int
    unique_bytes: int
    dedup_ratio: float
    orphaned_chunks: int
    def __init__(self, artifact_count: _Optional[int] = ..., chunk_count: _Optional[int] = ..., total_bytes: _Optional[int] = ..., unique_bytes: _Optional[int] = ..., dedup_ratio: _Optional[float] = ..., orphaned_chunks: _Optional[int] = ...) -> None: ...

class CheckpointListResult(_message.Message):
    __slots__ = ("checkpoints",)
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    checkpoints: _containers.RepeatedCompositeFieldContainer[CheckpointInfo]
    def __init__(self, checkpoints: _Optional[_Iterable[_Union[CheckpointInfo, _Mapping]]] = ...) -> None: ...

class CheckpointInfo(_message.Message):
    __slots__ = ("id", "name", "created_at", "is_auto")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_AUTO_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    created_at: int
    is_auto: bool
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., created_at: _Optional[int] = ..., is_auto: bool = ...) -> None: ...

class ChainQueryResult(_message.Message):
    __slots__ = ("transaction_begun", "committed", "rolled_back", "history", "similar", "drift", "height", "tip", "block", "codebook", "transition_analysis", "conflict_resolution", "merge")
    TRANSACTION_BEGUN_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_FIELD_NUMBER: _ClassVar[int]
    ROLLED_BACK_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    SIMILAR_FIELD_NUMBER: _ClassVar[int]
    DRIFT_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TIP_FIELD_NUMBER: _ClassVar[int]
    BLOCK_FIELD_NUMBER: _ClassVar[int]
    CODEBOOK_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_ANALYSIS_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    MERGE_FIELD_NUMBER: _ClassVar[int]
    transaction_begun: ChainTransactionBegun
    committed: ChainCommitted
    rolled_back: ChainRolledBack
    history: ChainHistory
    similar: ChainSimilar
    drift: ChainDrift
    height: ChainHeight
    tip: ChainTip
    block: ChainBlockInfo
    codebook: ChainCodebookInfo
    transition_analysis: ChainTransitionAnalysis
    conflict_resolution: ChainConflictResolution
    merge: ChainMergeResult
    def __init__(self, transaction_begun: _Optional[_Union[ChainTransactionBegun, _Mapping]] = ..., committed: _Optional[_Union[ChainCommitted, _Mapping]] = ..., rolled_back: _Optional[_Union[ChainRolledBack, _Mapping]] = ..., history: _Optional[_Union[ChainHistory, _Mapping]] = ..., similar: _Optional[_Union[ChainSimilar, _Mapping]] = ..., drift: _Optional[_Union[ChainDrift, _Mapping]] = ..., height: _Optional[_Union[ChainHeight, _Mapping]] = ..., tip: _Optional[_Union[ChainTip, _Mapping]] = ..., block: _Optional[_Union[ChainBlockInfo, _Mapping]] = ..., codebook: _Optional[_Union[ChainCodebookInfo, _Mapping]] = ..., transition_analysis: _Optional[_Union[ChainTransitionAnalysis, _Mapping]] = ..., conflict_resolution: _Optional[_Union[ChainConflictResolution, _Mapping]] = ..., merge: _Optional[_Union[ChainMergeResult, _Mapping]] = ...) -> None: ...

class ChainTransactionBegun(_message.Message):
    __slots__ = ("tx_id",)
    TX_ID_FIELD_NUMBER: _ClassVar[int]
    tx_id: str
    def __init__(self, tx_id: _Optional[str] = ...) -> None: ...

class ChainCommitted(_message.Message):
    __slots__ = ("block_hash", "height")
    BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    block_hash: str
    height: int
    def __init__(self, block_hash: _Optional[str] = ..., height: _Optional[int] = ...) -> None: ...

class ChainRolledBack(_message.Message):
    __slots__ = ("to_height",)
    TO_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    to_height: int
    def __init__(self, to_height: _Optional[int] = ...) -> None: ...

class ChainHistory(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ChainHistoryEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[ChainHistoryEntry, _Mapping]]] = ...) -> None: ...

class ChainHistoryEntry(_message.Message):
    __slots__ = ("height", "transaction_type", "data")
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TRANSACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    height: int
    transaction_type: str
    data: bytes
    def __init__(self, height: _Optional[int] = ..., transaction_type: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class ChainSimilar(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[ChainSimilarItem]
    def __init__(self, items: _Optional[_Iterable[_Union[ChainSimilarItem, _Mapping]]] = ...) -> None: ...

class ChainSimilarItem(_message.Message):
    __slots__ = ("block_hash", "height", "similarity")
    BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    block_hash: str
    height: int
    similarity: float
    def __init__(self, block_hash: _Optional[str] = ..., height: _Optional[int] = ..., similarity: _Optional[float] = ...) -> None: ...

class ChainDrift(_message.Message):
    __slots__ = ("from_height", "to_height", "total_drift", "avg_drift_per_block", "max_drift")
    FROM_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TO_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DRIFT_FIELD_NUMBER: _ClassVar[int]
    AVG_DRIFT_PER_BLOCK_FIELD_NUMBER: _ClassVar[int]
    MAX_DRIFT_FIELD_NUMBER: _ClassVar[int]
    from_height: int
    to_height: int
    total_drift: float
    avg_drift_per_block: float
    max_drift: float
    def __init__(self, from_height: _Optional[int] = ..., to_height: _Optional[int] = ..., total_drift: _Optional[float] = ..., avg_drift_per_block: _Optional[float] = ..., max_drift: _Optional[float] = ...) -> None: ...

class ChainHeight(_message.Message):
    __slots__ = ("height",)
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    height: int
    def __init__(self, height: _Optional[int] = ...) -> None: ...

class ChainTip(_message.Message):
    __slots__ = ("hash", "height")
    HASH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    hash: str
    height: int
    def __init__(self, hash: _Optional[str] = ..., height: _Optional[int] = ...) -> None: ...

class ChainBlockInfo(_message.Message):
    __slots__ = ("height", "hash", "prev_hash", "timestamp", "transaction_count", "proposer")
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    HASH_FIELD_NUMBER: _ClassVar[int]
    PREV_HASH_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TRANSACTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    PROPOSER_FIELD_NUMBER: _ClassVar[int]
    height: int
    hash: str
    prev_hash: str
    timestamp: int
    transaction_count: int
    proposer: str
    def __init__(self, height: _Optional[int] = ..., hash: _Optional[str] = ..., prev_hash: _Optional[str] = ..., timestamp: _Optional[int] = ..., transaction_count: _Optional[int] = ..., proposer: _Optional[str] = ...) -> None: ...

class ChainCodebookInfo(_message.Message):
    __slots__ = ("scope", "entry_count", "dimension", "domain")
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    scope: str
    entry_count: int
    dimension: int
    domain: str
    def __init__(self, scope: _Optional[str] = ..., entry_count: _Optional[int] = ..., dimension: _Optional[int] = ..., domain: _Optional[str] = ...) -> None: ...

class ChainTransitionAnalysis(_message.Message):
    __slots__ = ("total_transitions", "valid_transitions", "invalid_transitions", "avg_validity_score")
    TOTAL_TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    VALID_TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    INVALID_TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    AVG_VALIDITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    total_transitions: int
    valid_transitions: int
    invalid_transitions: int
    avg_validity_score: float
    def __init__(self, total_transitions: _Optional[int] = ..., valid_transitions: _Optional[int] = ..., invalid_transitions: _Optional[int] = ..., avg_validity_score: _Optional[float] = ...) -> None: ...

class ChainConflictResolution(_message.Message):
    __slots__ = ("strategy", "conflicts_resolved")
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    CONFLICTS_RESOLVED_FIELD_NUMBER: _ClassVar[int]
    strategy: str
    conflicts_resolved: int
    def __init__(self, strategy: _Optional[str] = ..., conflicts_resolved: _Optional[int] = ...) -> None: ...

class ChainMergeResult(_message.Message):
    __slots__ = ("success", "merged_count")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MERGED_COUNT_FIELD_NUMBER: _ClassVar[int]
    success: bool
    merged_count: int
    def __init__(self, success: bool = ..., merged_count: _Optional[int] = ...) -> None: ...

class ErrorInfo(_message.Message):
    __slots__ = ("code", "message", "details")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    message: str
    details: str
    def __init__(self, code: _Optional[_Union[ErrorCode, str]] = ..., message: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class BlobUploadRequest(_message.Message):
    __slots__ = ("metadata", "chunk")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    metadata: BlobUploadMetadata
    chunk: bytes
    def __init__(self, metadata: _Optional[_Union[BlobUploadMetadata, _Mapping]] = ..., chunk: _Optional[bytes] = ...) -> None: ...

class BlobUploadMetadata(_message.Message):
    __slots__ = ("filename", "content_type", "created_by", "tags", "linked_to", "custom")
    class CustomEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    LINKED_TO_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_NUMBER: _ClassVar[int]
    filename: str
    content_type: str
    created_by: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    linked_to: _containers.RepeatedScalarFieldContainer[str]
    custom: _containers.ScalarMap[str, str]
    def __init__(self, filename: _Optional[str] = ..., content_type: _Optional[str] = ..., created_by: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., linked_to: _Optional[_Iterable[str]] = ..., custom: _Optional[_Mapping[str, str]] = ...) -> None: ...

class BlobUploadResponse(_message.Message):
    __slots__ = ("artifact_id", "size", "checksum")
    ARTIFACT_ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    artifact_id: str
    size: int
    checksum: str
    def __init__(self, artifact_id: _Optional[str] = ..., size: _Optional[int] = ..., checksum: _Optional[str] = ...) -> None: ...

class BlobDownloadRequest(_message.Message):
    __slots__ = ("artifact_id",)
    ARTIFACT_ID_FIELD_NUMBER: _ClassVar[int]
    artifact_id: str
    def __init__(self, artifact_id: _Optional[str] = ...) -> None: ...

class BlobDownloadChunk(_message.Message):
    __slots__ = ("data", "is_final")
    DATA_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    is_final: bool
    def __init__(self, data: _Optional[bytes] = ..., is_final: bool = ...) -> None: ...

class BlobDeleteRequest(_message.Message):
    __slots__ = ("artifact_id",)
    ARTIFACT_ID_FIELD_NUMBER: _ClassVar[int]
    artifact_id: str
    def __init__(self, artifact_id: _Optional[str] = ...) -> None: ...

class BlobDeleteResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class BlobMetadataRequest(_message.Message):
    __slots__ = ("artifact_id",)
    ARTIFACT_ID_FIELD_NUMBER: _ClassVar[int]
    artifact_id: str
    def __init__(self, artifact_id: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("service",)
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: _Optional[str] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: ServingStatus
    def __init__(self, status: _Optional[_Union[ServingStatus, str]] = ...) -> None: ...
