"""Protocol buffer generated files for Neumann gRPC.

This module contains generated protobuf files. To regenerate them:

1. Install grpcio-tools:
   pip install grpcio-tools

2. Generate Python files from proto:
   python -m grpc_tools.protoc \\
       -I../../neumann_server/proto \\
       --python_out=. \\
       --grpc_python_out=. \\
       --pyi_out=. \\
       ../../neumann_server/proto/neumann.proto \\
       ../../neumann_server/proto/vector.proto
"""

from __future__ import annotations

from . import neumann_pb2, neumann_pb2_grpc, vector_pb2, vector_pb2_grpc

__all__ = ["neumann_pb2", "neumann_pb2_grpc", "vector_pb2", "vector_pb2_grpc"]
