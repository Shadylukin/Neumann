# SPDX-License-Identifier: MIT
"""Transaction support for Neumann database."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neumann.errors import NeumannError
from neumann.types import QueryResult

if TYPE_CHECKING:
    from types import TracebackType

    from neumann.client import NeumannClient


class Transaction:
    """A database transaction with automatic commit/rollback.

    Use as a context manager for automatic transaction management:

        with client.transaction() as tx:
            tx.execute("INSERT users name='Alice'")
            tx.execute("INSERT users name='Bob'")
            # Auto-commits on successful exit

    On exception, the transaction is automatically rolled back.
    """

    def __init__(self, client: NeumannClient) -> None:
        """Initialize a transaction.

        Args:
            client: The NeumannClient to use for the transaction.
        """
        self._client = client
        self._started = False
        self._committed = False
        self._rolled_back = False
        self._tx_id: str | None = None

    @property
    def is_active(self) -> bool:
        """Check if transaction is currently active."""
        return self._started and not self._committed and not self._rolled_back

    def begin(self) -> Transaction:
        """Begin the transaction.

        Returns:
            Self for method chaining.

        Raises:
            NeumannError: If transaction is already started or client error.
        """
        if self._started:
            raise NeumannError("Transaction already started")

        result = self._client.execute("BEGIN")
        if result.is_error:
            raise NeumannError(result.error_message or "Failed to begin transaction")

        self._started = True

        # Extract transaction ID if returned
        if result.value:
            self._tx_id = result.value

        return self

    def commit(self) -> None:
        """Commit the transaction.

        Raises:
            NeumannError: If transaction is not active or commit fails.
        """
        if not self.is_active:
            raise NeumannError("Transaction is not active")

        result = self._client.execute("COMMIT")
        if result.is_error:
            raise NeumannError(result.error_message or "Failed to commit transaction")

        self._committed = True

    def rollback(self) -> None:
        """Rollback the transaction.

        Raises:
            NeumannError: If transaction is not active or rollback fails.
        """
        if not self.is_active:
            raise NeumannError("Transaction is not active")

        result = self._client.execute("ROLLBACK")
        if result.is_error:
            raise NeumannError(result.error_message or "Failed to rollback transaction")

        self._rolled_back = True

    def execute(self, query: str) -> QueryResult:
        """Execute a query within this transaction.

        Args:
            query: The query to execute.

        Returns:
            QueryResult from the query.

        Raises:
            NeumannError: If transaction is not active or query fails.
        """
        if not self.is_active:
            raise NeumannError("Transaction is not active")

        return self._client.execute(query)

    def __enter__(self) -> Transaction:
        """Context manager entry - begins the transaction."""
        return self.begin()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit - commits or rolls back.

        On normal exit, commits the transaction.
        On exception, rolls back the transaction.
        """
        if not self.is_active:
            return False

        if exc_type is not None:
            # Exception occurred - rollback
            try:
                self.rollback()
            except NeumannError:
                pass  # Ignore rollback errors
            return False  # Don't suppress the exception

        # Normal exit - commit
        try:
            self.commit()
        except NeumannError:
            # Commit failed - try to rollback
            try:
                self.rollback()
            except NeumannError:
                pass
            raise

        return False
