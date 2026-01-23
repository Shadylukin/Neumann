"""Protocol buffer generated files for Neumann gRPC.

This module should contain generated protobuf files. To generate them:

1. Install grpcio-tools:
   pip install grpcio-tools

2. Generate Python files from proto:
   python -m grpc_tools.protoc \\
       -I../../neumann_server/proto \\
       --python_out=. \\
       --grpc_python_out=. \\
       --pyi_out=. \\
       ../../neumann_server/proto/neumann.proto

Until the files are generated, imports will raise an error with instructions.
"""

from __future__ import annotations

import sys
from types import ModuleType


class _ProtoStub(ModuleType):
    """Stub module that raises helpful error on attribute access."""

    def __getattr__(self, name: str) -> None:
        raise ImportError(
            f"Cannot import '{name}' from neumann.proto. "
            "Protocol buffer files have not been generated. "
            "Please run the following command from neumann-py/src/neumann/proto/:\n\n"
            "  python -m grpc_tools.protoc \\\n"
            "      -I../../../../neumann_server/proto \\\n"
            "      --python_out=. \\\n"
            "      --grpc_python_out=. \\\n"
            "      --pyi_out=. \\\n"
            "      ../../../../neumann_server/proto/neumann.proto\n"
        )


# Replace this module with stub if generated files don't exist
try:
    from . import neumann_pb2 as _test_import

    del _test_import
except ImportError:
    # Generated files not present - install stub
    sys.modules[__name__] = _ProtoStub(__name__)
