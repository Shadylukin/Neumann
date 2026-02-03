# SPDX-License-Identifier: MIT
"""Tests for Transaction class."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neumann.transaction import Transaction
from neumann.errors import NeumannError
from neumann.types import QueryResult, QueryResultType


class TestTransactionInit:
    """Tests for Transaction initialization."""

    def test_init(self) -> None:
        """Test transaction initialization."""
        client = MagicMock()
        tx = Transaction(client)

        assert tx._client is client
        assert not tx._started
        assert not tx._committed
        assert not tx._rolled_back
        assert tx._tx_id is None

    def test_is_active_initial(self) -> None:
        """Test is_active returns False initially."""
        client = MagicMock()
        tx = Transaction(client)

        assert not tx.is_active


class TestTransactionBegin:
    """Tests for Transaction.begin()."""

    def test_begin_success(self) -> None:
        """Test successful begin."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        result = tx.begin()

        assert result is tx
        assert tx._started
        assert tx.is_active
        client.execute.assert_called_once_with("BEGIN")

    def test_begin_with_tx_id(self) -> None:
        """Test begin returns transaction ID."""
        client = MagicMock()
        result = QueryResult(QueryResultType.VALUE, "tx-123")
        client.execute.return_value = result

        tx = Transaction(client)
        tx.begin()

        assert tx._tx_id == "tx-123"

    def test_begin_already_started(self) -> None:
        """Test begin raises if already started."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()

        with pytest.raises(NeumannError) as exc_info:
            tx.begin()
        assert "already started" in str(exc_info.value)

    def test_begin_error(self) -> None:
        """Test begin raises on error result."""
        client = MagicMock()
        result = QueryResult(QueryResultType.ERROR, "Begin failed")
        client.execute.return_value = result

        tx = Transaction(client)

        with pytest.raises(NeumannError) as exc_info:
            tx.begin()
        assert "Begin failed" in str(exc_info.value)


class TestTransactionCommit:
    """Tests for Transaction.commit()."""

    def test_commit_success(self) -> None:
        """Test successful commit."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.commit()

        assert tx._committed
        assert not tx.is_active
        assert client.execute.call_count == 2
        client.execute.assert_called_with("COMMIT")

    def test_commit_not_active(self) -> None:
        """Test commit raises if not active."""
        client = MagicMock()
        tx = Transaction(client)

        with pytest.raises(NeumannError) as exc_info:
            tx.commit()
        assert "not active" in str(exc_info.value)

    def test_commit_already_committed(self) -> None:
        """Test commit raises if already committed."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.commit()

        with pytest.raises(NeumannError) as exc_info:
            tx.commit()
        assert "not active" in str(exc_info.value)

    def test_commit_error(self) -> None:
        """Test commit raises on error result."""
        client = MagicMock()
        client.execute.side_effect = [
            QueryResult(QueryResultType.EMPTY),  # BEGIN
            QueryResult(QueryResultType.ERROR, "Commit failed"),  # COMMIT
        ]

        tx = Transaction(client)
        tx.begin()

        with pytest.raises(NeumannError) as exc_info:
            tx.commit()
        assert "Commit failed" in str(exc_info.value)


class TestTransactionRollback:
    """Tests for Transaction.rollback()."""

    def test_rollback_success(self) -> None:
        """Test successful rollback."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.rollback()

        assert tx._rolled_back
        assert not tx.is_active
        assert client.execute.call_count == 2
        client.execute.assert_called_with("ROLLBACK")

    def test_rollback_not_active(self) -> None:
        """Test rollback raises if not active."""
        client = MagicMock()
        tx = Transaction(client)

        with pytest.raises(NeumannError) as exc_info:
            tx.rollback()
        assert "not active" in str(exc_info.value)

    def test_rollback_already_rolled_back(self) -> None:
        """Test rollback raises if already rolled back."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.rollback()

        with pytest.raises(NeumannError) as exc_info:
            tx.rollback()
        assert "not active" in str(exc_info.value)

    def test_rollback_error(self) -> None:
        """Test rollback raises on error result."""
        client = MagicMock()
        client.execute.side_effect = [
            QueryResult(QueryResultType.EMPTY),  # BEGIN
            QueryResult(QueryResultType.ERROR, "Rollback failed"),  # ROLLBACK
        ]

        tx = Transaction(client)
        tx.begin()

        with pytest.raises(NeumannError) as exc_info:
            tx.rollback()
        assert "Rollback failed" in str(exc_info.value)


class TestTransactionExecute:
    """Tests for Transaction.execute()."""

    def test_execute_success(self) -> None:
        """Test successful execute within transaction."""
        client = MagicMock()
        client.execute.side_effect = [
            QueryResult(QueryResultType.EMPTY),  # BEGIN
            QueryResult(QueryResultType.COUNT, 5),  # User query
        ]

        tx = Transaction(client)
        tx.begin()
        result = tx.execute("SELECT COUNT(*) FROM users")

        assert result.type == QueryResultType.COUNT
        assert result.data == 5

    def test_execute_not_active(self) -> None:
        """Test execute raises if not active."""
        client = MagicMock()
        tx = Transaction(client)

        with pytest.raises(NeumannError) as exc_info:
            tx.execute("SELECT 1")
        assert "not active" in str(exc_info.value)

    def test_execute_after_commit(self) -> None:
        """Test execute raises after commit."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.commit()

        with pytest.raises(NeumannError) as exc_info:
            tx.execute("SELECT 1")
        assert "not active" in str(exc_info.value)

    def test_execute_after_rollback(self) -> None:
        """Test execute raises after rollback."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.rollback()

        with pytest.raises(NeumannError) as exc_info:
            tx.execute("SELECT 1")
        assert "not active" in str(exc_info.value)


class TestTransactionContextManager:
    """Tests for Transaction context manager."""

    def test_context_manager_success(self) -> None:
        """Test context manager commits on success."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        with Transaction(client) as tx:
            assert tx.is_active
            tx.execute("INSERT users name='Alice'")

        assert tx._committed
        assert not tx.is_active
        # BEGIN, INSERT, COMMIT
        assert client.execute.call_count == 3

    def test_context_manager_exception(self) -> None:
        """Test context manager rolls back on exception."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        with pytest.raises(ValueError):
            with Transaction(client) as tx:
                tx.execute("INSERT users name='Alice'")
                raise ValueError("Test error")

        assert tx._rolled_back
        assert not tx.is_active
        # BEGIN, INSERT, ROLLBACK
        assert client.execute.call_count == 3

    def test_context_manager_commit_failure(self) -> None:
        """Test context manager rolls back on commit failure."""
        client = MagicMock()
        client.execute.side_effect = [
            QueryResult(QueryResultType.EMPTY),  # BEGIN
            QueryResult(QueryResultType.EMPTY),  # INSERT
            QueryResult(QueryResultType.ERROR, "Commit failed"),  # COMMIT
            QueryResult(QueryResultType.EMPTY),  # ROLLBACK
        ]

        with pytest.raises(NeumannError):
            with Transaction(client) as tx:
                tx.execute("INSERT users name='Alice'")

        # Should have attempted rollback after commit failure
        assert client.execute.call_count == 4

    def test_context_manager_rollback_failure_on_exception(self) -> None:
        """Test context manager handles rollback failure on exception."""
        client = MagicMock()
        client.execute.side_effect = [
            QueryResult(QueryResultType.EMPTY),  # BEGIN
            QueryResult(QueryResultType.ERROR, "Rollback failed"),  # ROLLBACK
        ]

        with pytest.raises(ValueError):
            with Transaction(client) as tx:
                raise ValueError("Test error")

        # Rollback error is ignored, original exception is raised

    def test_context_manager_not_active_on_exit(self) -> None:
        """Test context manager handles not active on exit."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.commit()

        # Manually entering exit should not raise
        result = tx.__exit__(None, None, None)
        assert result is None


class TestTransactionIsActive:
    """Tests for is_active property."""

    def test_is_active_not_started(self) -> None:
        """Test is_active when not started."""
        client = MagicMock()
        tx = Transaction(client)

        assert not tx.is_active

    def test_is_active_after_begin(self) -> None:
        """Test is_active after begin."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()

        assert tx.is_active

    def test_is_active_after_commit(self) -> None:
        """Test is_active after commit."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.commit()

        assert not tx.is_active

    def test_is_active_after_rollback(self) -> None:
        """Test is_active after rollback."""
        client = MagicMock()
        client.execute.return_value = QueryResult(QueryResultType.EMPTY)

        tx = Transaction(client)
        tx.begin()
        tx.rollback()

        assert not tx.is_active
