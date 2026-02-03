# SPDX-License-Identifier: MIT
"""Async transaction support for Neumann database."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from neumann.errors import NeumannError
from neumann.types import QueryResult, QueryResultType

if TYPE_CHECKING:
    from types import TracebackType

    from neumann.aio.client import AsyncNeumannClient


class AsyncTransaction:
    """Async transaction context manager for Neumann database.

    Provides automatic transaction management with commit/rollback semantics.

    Example:
        async with await AsyncNeumannClient.connect("localhost:50051") as client:
            async with AsyncTransaction(client) as tx:
                await tx.execute("INSERT INTO users VALUES (1, 'alice')")
                await tx.execute("INSERT INTO orders VALUES (1, 1, 100)")
                # Commits automatically on successful exit
    """

    def __init__(
        self,
        client: AsyncNeumannClient,
        *,
        identity: str | None = None,
        auto_commit: bool = True,
    ) -> None:
        """Initialize transaction.

        Args:
            client: The async client to use for the transaction.
            identity: Optional identity for vault access.
            auto_commit: Whether to automatically commit on successful exit (default: True).
        """
        self._client = client
        self._identity = identity
        self._auto_commit = auto_commit
        self._tx_id: str | None = None
        self._active = False
        self._committed = False
        self._rolled_back = False

    @property
    def tx_id(self) -> str | None:
        """Get the transaction ID."""
        return self._tx_id

    @property
    def is_active(self) -> bool:
        """Check if transaction is active."""
        return self._active

    @property
    def is_committed(self) -> bool:
        """Check if transaction was committed."""
        return self._committed

    @property
    def is_rolled_back(self) -> bool:
        """Check if transaction was rolled back."""
        return self._rolled_back

    async def begin(self) -> str:
        """Begin the transaction.

        Returns:
            The transaction ID.

        Raises:
            NeumannError: If transaction cannot be started.
        """
        if self._active:
            raise NeumannError("Transaction already active")

        result = await self._client.execute("CHAIN BEGIN", identity=self._identity)

        if result.type == QueryResultType.CHAIN_TRANSACTION_BEGUN:
            self._tx_id = str(result.data.tx_id)
            self._active = True
            return self._tx_id
        elif result.type == QueryResultType.ERROR:
            raise NeumannError(f"Failed to begin transaction: {result.data}")
        else:
            raise NeumannError(f"Unexpected result type: {result.type}")

    async def execute(self, query: str) -> QueryResult:
        """Execute a query within the transaction.

        Args:
            query: The query to execute.

        Returns:
            Query result.

        Raises:
            NeumannError: If transaction is not active.
        """
        if not self._active:
            raise NeumannError("Transaction is not active")

        return await self._client.execute(query, identity=self._identity)

    async def commit(self) -> tuple[str, int]:
        """Commit the transaction.

        Returns:
            Tuple of (block_hash, height).

        Raises:
            NeumannError: If commit fails or transaction is not active.
        """
        if not self._active:
            raise NeumannError("Transaction is not active")

        result = await self._client.execute("CHAIN COMMIT", identity=self._identity)

        if result.type == QueryResultType.CHAIN_COMMITTED:
            self._active = False
            self._committed = True
            return (str(result.data.block_hash), int(result.data.height))
        elif result.type == QueryResultType.ERROR:
            raise NeumannError(f"Failed to commit transaction: {result.data}")
        else:
            raise NeumannError(f"Unexpected result type: {result.type}")

    async def rollback(self) -> int:
        """Rollback the transaction.

        Returns:
            The height rolled back to.

        Raises:
            NeumannError: If rollback fails or transaction is not active.
        """
        if not self._active:
            raise NeumannError("Transaction is not active")

        result = await self._client.execute("CHAIN ROLLBACK", identity=self._identity)

        if result.type == QueryResultType.CHAIN_ROLLED_BACK:
            self._active = False
            self._rolled_back = True
            return int(result.data.to_height)
        elif result.type == QueryResultType.ERROR:
            raise NeumannError(f"Failed to rollback transaction: {result.data}")
        else:
            raise NeumannError(f"Unexpected result type: {result.type}")

    async def __aenter__(self) -> AsyncTransaction:
        """Async context manager entry - begins transaction."""
        await self.begin()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit - commits or rolls back."""
        if not self._active:
            return

        if exc_type is not None:
            # Exception occurred - rollback
            with contextlib.suppress(Exception):
                await self.rollback()
        elif self._auto_commit:
            # No exception and auto_commit enabled - commit
            await self.commit()


class TransactionBuilder:
    """Builder for creating transactions with custom options.

    Example:
        tx = (
            TransactionBuilder(client)
            .with_identity("user:alice")
            .with_auto_commit(False)
            .build()
        )

        async with tx:
            await tx.execute("INSERT INTO users VALUES (1, 'alice')")
            await tx.commit()  # Manual commit required
    """

    def __init__(self, client: AsyncNeumannClient) -> None:
        """Initialize transaction builder.

        Args:
            client: The async client to use for the transaction.
        """
        self._client = client
        self._identity: str | None = None
        self._auto_commit = True

    def with_identity(self, identity: str) -> TransactionBuilder:
        """Set the identity for vault access.

        Args:
            identity: The identity to use.

        Returns:
            Self for chaining.
        """
        self._identity = identity
        return self

    def with_auto_commit(self, auto_commit: bool) -> TransactionBuilder:
        """Set whether to auto-commit on successful exit.

        Args:
            auto_commit: Whether to auto-commit.

        Returns:
            Self for chaining.
        """
        self._auto_commit = auto_commit
        return self

    def build(self) -> AsyncTransaction:
        """Build the transaction.

        Returns:
            The configured AsyncTransaction.
        """
        return AsyncTransaction(
            self._client,
            identity=self._identity,
            auto_commit=self._auto_commit,
        )
