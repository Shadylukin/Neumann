# SPDX-License-Identifier: MIT
"""Tests for async transaction support."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from neumann.aio.transaction import AsyncTransaction, TransactionBuilder
from neumann.errors import NeumannError
from neumann.types import (
    ChainCommitted,
    ChainRolledBack,
    ChainTransactionBegun,
    QueryResult,
    QueryResultType,
)


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock async client."""
    client = AsyncMock()
    client.execute = AsyncMock()
    return client


class TestAsyncTransaction:
    """Tests for AsyncTransaction."""

    @pytest.mark.asyncio
    async def test_begin_transaction(self, mock_client: AsyncMock) -> None:
        """Test beginning a transaction."""
        mock_client.execute.return_value = QueryResult(
            QueryResultType.CHAIN_TRANSACTION_BEGUN,
            ChainTransactionBegun(tx_id="tx-123"),
        )

        tx = AsyncTransaction(mock_client)
        tx_id = await tx.begin()

        assert tx_id == "tx-123"
        assert tx.tx_id == "tx-123"
        assert tx.is_active
        assert not tx.is_committed
        assert not tx.is_rolled_back
        mock_client.execute.assert_called_once_with("CHAIN BEGIN", identity=None)

    @pytest.mark.asyncio
    async def test_begin_already_active(self, mock_client: AsyncMock) -> None:
        """Test beginning when transaction already active."""
        mock_client.execute.return_value = QueryResult(
            QueryResultType.CHAIN_TRANSACTION_BEGUN,
            ChainTransactionBegun(tx_id="tx-123"),
        )

        tx = AsyncTransaction(mock_client)
        await tx.begin()

        with pytest.raises(NeumannError, match="already active"):
            await tx.begin()

    @pytest.mark.asyncio
    async def test_execute_query(self, mock_client: AsyncMock) -> None:
        """Test executing a query within transaction."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(QueryResultType.COUNT, 1),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()
        result = await tx.execute("INSERT INTO users VALUES (1, 'alice')")

        assert result.type == QueryResultType.COUNT
        assert mock_client.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_not_active(self, mock_client: AsyncMock) -> None:
        """Test executing when transaction not active."""
        tx = AsyncTransaction(mock_client)

        with pytest.raises(NeumannError, match="not active"):
            await tx.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_commit_transaction(self, mock_client: AsyncMock) -> None:
        """Test committing a transaction."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.CHAIN_COMMITTED,
                ChainCommitted(block_hash="hash-abc", height=42),
            ),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()
        block_hash, height = await tx.commit()

        assert block_hash == "hash-abc"
        assert height == 42
        assert not tx.is_active
        assert tx.is_committed
        assert not tx.is_rolled_back

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, mock_client: AsyncMock) -> None:
        """Test rolling back a transaction."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.CHAIN_ROLLED_BACK,
                ChainRolledBack(to_height=41),
            ),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()
        to_height = await tx.rollback()

        assert to_height == 41
        assert not tx.is_active
        assert not tx.is_committed
        assert tx.is_rolled_back

    @pytest.mark.asyncio
    async def test_context_manager_commit(self, mock_client: AsyncMock) -> None:
        """Test context manager auto-commits on success."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(QueryResultType.COUNT, 1),
            QueryResult(
                QueryResultType.CHAIN_COMMITTED,
                ChainCommitted(block_hash="hash-abc", height=42),
            ),
        ]

        async with AsyncTransaction(mock_client) as tx:
            await tx.execute("INSERT INTO users VALUES (1, 'alice')")

        assert tx.is_committed
        assert mock_client.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_context_manager_rollback_on_exception(self, mock_client: AsyncMock) -> None:
        """Test context manager rolls back on exception."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.CHAIN_ROLLED_BACK,
                ChainRolledBack(to_height=41),
            ),
        ]

        with pytest.raises(ValueError, match="test error"):
            async with AsyncTransaction(mock_client) as tx:
                raise ValueError("test error")

        assert tx.is_rolled_back

    @pytest.mark.asyncio
    async def test_no_auto_commit(self, mock_client: AsyncMock) -> None:
        """Test disabling auto-commit."""
        mock_client.execute.return_value = QueryResult(
            QueryResultType.CHAIN_TRANSACTION_BEGUN,
            ChainTransactionBegun(tx_id="tx-123"),
        )

        async with AsyncTransaction(mock_client, auto_commit=False) as tx:
            pass  # No commit should be called

        # Only begin was called, no commit
        mock_client.execute.assert_called_once()
        assert tx.is_active  # Still active because no auto-commit

    @pytest.mark.asyncio
    async def test_with_identity(self, mock_client: AsyncMock) -> None:
        """Test transaction with identity."""
        mock_client.execute.return_value = QueryResult(
            QueryResultType.CHAIN_TRANSACTION_BEGUN,
            ChainTransactionBegun(tx_id="tx-123"),
        )

        tx = AsyncTransaction(mock_client, identity="user:alice")
        await tx.begin()

        mock_client.execute.assert_called_once_with("CHAIN BEGIN", identity="user:alice")


class TestTransactionBuilder:
    """Tests for TransactionBuilder."""

    def test_build_default(self) -> None:
        """Test building transaction with defaults."""
        mock_client = AsyncMock()
        tx = TransactionBuilder(mock_client).build()

        assert tx._client is mock_client
        assert tx._identity is None
        assert tx._auto_commit is True

    def test_with_identity(self) -> None:
        """Test setting identity."""
        mock_client = AsyncMock()
        tx = TransactionBuilder(mock_client).with_identity("user:alice").build()

        assert tx._identity == "user:alice"

    def test_with_auto_commit(self) -> None:
        """Test setting auto-commit."""
        mock_client = AsyncMock()
        tx = TransactionBuilder(mock_client).with_auto_commit(False).build()

        assert tx._auto_commit is False

    def test_chaining(self) -> None:
        """Test method chaining."""
        mock_client = AsyncMock()
        tx = (
            TransactionBuilder(mock_client)
            .with_identity("user:alice")
            .with_auto_commit(False)
            .build()
        )

        assert tx._identity == "user:alice"
        assert tx._auto_commit is False


class TestAsyncTransactionErrorHandling:
    """Tests for AsyncTransaction error handling."""

    @pytest.mark.asyncio
    async def test_begin_returns_error(self, mock_client: AsyncMock) -> None:
        """Test begin when server returns error result type."""
        mock_client.execute.return_value = QueryResult(
            QueryResultType.ERROR,
            "Transaction begin failed",
        )

        tx = AsyncTransaction(mock_client)

        with pytest.raises(NeumannError, match="Failed to begin transaction"):
            await tx.begin()

    @pytest.mark.asyncio
    async def test_begin_unexpected_result_type(self, mock_client: AsyncMock) -> None:
        """Test begin with unexpected result type."""
        mock_client.execute.return_value = QueryResult(
            QueryResultType.COUNT,
            1,
        )

        tx = AsyncTransaction(mock_client)

        with pytest.raises(NeumannError, match="Unexpected result type"):
            await tx.begin()

    @pytest.mark.asyncio
    async def test_commit_when_not_active(self, mock_client: AsyncMock) -> None:
        """Test commit when transaction not active."""
        tx = AsyncTransaction(mock_client)

        with pytest.raises(NeumannError, match="not active"):
            await tx.commit()

    @pytest.mark.asyncio
    async def test_commit_returns_error(self, mock_client: AsyncMock) -> None:
        """Test commit when server returns error."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.ERROR,
                "Commit failed",
            ),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()

        with pytest.raises(NeumannError, match="Failed to commit transaction"):
            await tx.commit()

    @pytest.mark.asyncio
    async def test_commit_unexpected_result_type(self, mock_client: AsyncMock) -> None:
        """Test commit with unexpected result type."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.COUNT,
                1,
            ),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()

        with pytest.raises(NeumannError, match="Unexpected result type"):
            await tx.commit()

    @pytest.mark.asyncio
    async def test_rollback_when_not_active(self, mock_client: AsyncMock) -> None:
        """Test rollback when transaction not active."""
        tx = AsyncTransaction(mock_client)

        with pytest.raises(NeumannError, match="not active"):
            await tx.rollback()

    @pytest.mark.asyncio
    async def test_rollback_returns_error(self, mock_client: AsyncMock) -> None:
        """Test rollback when server returns error."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.ERROR,
                "Rollback failed",
            ),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()

        with pytest.raises(NeumannError, match="Failed to rollback transaction"):
            await tx.rollback()

    @pytest.mark.asyncio
    async def test_rollback_unexpected_result_type(self, mock_client: AsyncMock) -> None:
        """Test rollback with unexpected result type."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            QueryResult(
                QueryResultType.COUNT,
                1,
            ),
        ]

        tx = AsyncTransaction(mock_client)
        await tx.begin()

        with pytest.raises(NeumannError, match="Unexpected result type"):
            await tx.rollback()

    @pytest.mark.asyncio
    async def test_context_exit_when_not_active(self, mock_client: AsyncMock) -> None:
        """Test context manager exit when transaction not active."""
        tx = AsyncTransaction(mock_client)
        tx._active = False  # Not active

        # Should not raise - just return
        await tx.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_context_exit_exception_with_rollback_failure(
        self, mock_client: AsyncMock
    ) -> None:
        """Test context manager handles rollback failure silently."""
        mock_client.execute.side_effect = [
            QueryResult(
                QueryResultType.CHAIN_TRANSACTION_BEGUN,
                ChainTransactionBegun(tx_id="tx-123"),
            ),
            Exception("Rollback failed due to network error"),
        ]

        # Should suppress the rollback error but propagate original exception
        with pytest.raises(ValueError, match="original error"):
            async with AsyncTransaction(mock_client):
                raise ValueError("original error")
