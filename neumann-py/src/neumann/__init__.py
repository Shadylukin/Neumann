# SPDX-License-Identifier: MIT
"""Neumann database Python SDK.

This package provides a Python client for the Neumann database with support
for both embedded (in-process via PyO3) and remote (gRPC) modes.

Basic usage:

    # Remote connection (sync)
    from neumann import NeumannClient

    with NeumannClient.connect("localhost:50051", api_key="...") as client:
        result = client.query("SELECT users")
        for row in result.rows:
            print(row.to_dict())

    # Embedded mode (requires native module)
    client = NeumannClient.embedded()
    client.execute("CREATE TABLE users (name:string, age:int)")

    # Async client - preferred pattern
    from neumann.aio import AsyncNeumannClient

    async with AsyncNeumannClient("localhost:50051") as client:
        # Unified query across all engines
        results = await client.query('''
            FIND NODE user
            WHERE role = 'engineer'
            SIMILAR TO embedding
            CONNECTED TO 'user:alice'
        ''')

    # Or using connect() explicitly
    async with await AsyncNeumannClient.connect("localhost:50051") as client:
        result = await client.query("SELECT users")
"""

from neumann.client import NeumannClient
from neumann.config import (
    ClientConfig,
    KeepaliveConfig,
    RetryConfig,
    TimeoutConfig,
)
from neumann.errors import (
    AuthenticationError,
    ConnectionError,
    ErrorCode,
    InternalError,
    InvalidArgumentError,
    NeumannError,
    NotFoundError,
    ParseError,
    PermissionError,
    QueryError,
)
from neumann.services import (
    ArtifactMetadata,
    BlobClient,
    BlobServiceClient,
    BlobUploadOptions,
    BlobUploadResult,
    CollectionInfo,
    CollectionsClient,
    DistanceMetric,
    PointsClient,
    ScoredVectorPoint,
    ScrollResult,
    VectorClient,
    VectorPoint,
)
from neumann.transaction import Transaction
from neumann.types import (
    AggregateResult,
    ArtifactInfo,
    BatchOperationResult,
    BlobStats,
    CentralityItem,
    CentralityResult,
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
    CommunitiesResult,
    CommunityItem,
    ConstraintItem,
    ConstraintsResult,
    Edge,
    EdgeBinding,
    GraphIndexesResult,
    Node,
    NodeBinding,
    PageRankItem,
    PageRankResult,
    Path,
    PathBinding,
    PathSegment,
    PatternMatchResult,
    PatternMatchStats,
    QueryResult,
    QueryResultType,
    Row,
    ScalarType,
    SimilarItem,
    UnifiedItem,
    UnifiedResult,
    Value,
)

__version__ = "0.2.0"

__all__ = [
    # Client
    "NeumannClient",
    "Transaction",
    # Vector services
    "VectorClient",
    "VectorPoint",
    "ScoredVectorPoint",
    "CollectionInfo",
    "DistanceMetric",
    "PointsClient",
    "CollectionsClient",
    # Blob services
    "BlobClient",
    "BlobServiceClient",
    "BlobUploadOptions",
    "BlobUploadResult",
    "ArtifactMetadata",
    "ScrollResult",
    # Configuration
    "ClientConfig",
    "TimeoutConfig",
    "RetryConfig",
    "KeepaliveConfig",
    # Types
    "QueryResult",
    "QueryResultType",
    "Row",
    "Node",
    "Edge",
    "Path",
    "PathSegment",
    "SimilarItem",
    "ArtifactInfo",
    "BlobStats",
    "CheckpointInfo",
    "UnifiedItem",
    "UnifiedResult",
    # Graph algorithm results
    "PageRankItem",
    "PageRankResult",
    "CentralityItem",
    "CentralityResult",
    "CommunityItem",
    "CommunitiesResult",
    "ConstraintItem",
    "ConstraintsResult",
    "AggregateResult",
    "BatchOperationResult",
    "GraphIndexesResult",
    # Pattern matching
    "NodeBinding",
    "EdgeBinding",
    "PathBinding",
    "PatternMatchStats",
    "PatternMatchResult",
    # Chain results
    "ChainTransactionBegun",
    "ChainCommitted",
    "ChainRolledBack",
    "ChainHistoryEntry",
    "ChainHistory",
    "ChainSimilarItem",
    "ChainSimilar",
    "ChainDrift",
    "ChainHeight",
    "ChainTip",
    "ChainBlockInfo",
    "ChainCodebookInfo",
    "ChainTransitionAnalysis",
    "ChainConflictResolution",
    "ChainMergeResult",
    "Value",
    "ScalarType",
    # Errors
    "NeumannError",
    "ErrorCode",
    "ConnectionError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "InvalidArgumentError",
    "ParseError",
    "QueryError",
    "InternalError",
]
