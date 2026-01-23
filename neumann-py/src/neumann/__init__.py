"""Neumann database Python SDK.

This package provides a Python client for the Neumann database with support
for both embedded (in-process via PyO3) and remote (gRPC) modes.

Basic usage:

    # Remote connection
    from neumann import NeumannClient

    client = NeumannClient.connect("localhost:9200", api_key="...")
    result = client.execute("SELECT users")
    for row in result.rows:
        print(row.to_dict())

    # Embedded mode (requires native module)
    client = NeumannClient.embedded()
    client.execute("CREATE TABLE users (name:string, age:int)")

    # Async client
    from neumann.aio import AsyncNeumannClient

    async with await AsyncNeumannClient.connect("localhost:9200") as client:
        result = await client.execute("SELECT users")
"""

from neumann.client import NeumannClient
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
from neumann.transaction import Transaction
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

__version__ = "0.1.0"

__all__ = [
    # Client
    "NeumannClient",
    "Transaction",
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
