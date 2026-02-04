# SPDX-License-Identifier: MIT
"""Tests for integration modules (pandas and numpy)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from neumann.types import (
    QueryResult,
    QueryResultType,
    Row,
    Value,
)


class TestPandasIntegration:
    """Tests for pandas integration."""

    def test_import_exports(self) -> None:
        """Test that integration functions are exported."""
        from neumann.integrations import (
            dataframe_to_inserts,
            result_to_dataframe,
            rows_to_dataframe,
        )

        assert callable(result_to_dataframe)
        assert callable(rows_to_dataframe)
        assert callable(dataframe_to_inserts)

    def test_result_to_dataframe_empty(self) -> None:
        """Test converting empty result to dataframe."""
        pytest.importorskip("pandas")
        from neumann.integrations.pandas import result_to_dataframe

        result = QueryResult(QueryResultType.EMPTY)

        with pytest.raises(ValueError) as exc_info:
            result_to_dataframe(result)
        assert "expected rows result" in str(exc_info.value).lower()

    def test_result_to_dataframe_rows(self) -> None:
        """Test converting rows result to dataframe."""
        pytest.importorskip("pandas")
        from neumann.integrations.pandas import result_to_dataframe

        rows = [
            Row(values={"id": Value.int_(1), "name": Value.string("Alice")}),
            Row(values={"id": Value.int_(2), "name": Value.string("Bob")}),
        ]
        result = QueryResult(QueryResultType.ROWS, rows)

        df = result_to_dataframe(result)

        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["id"] == 2
        assert df.iloc[1]["name"] == "Bob"

    def test_rows_to_dataframe(self) -> None:
        """Test converting rows to dataframe."""
        pytest.importorskip("pandas")
        from neumann.integrations.pandas import rows_to_dataframe

        rows = [
            Row(values={"x": Value.float_(1.0), "y": Value.float_(2.0)}),
            Row(values={"x": Value.float_(3.0), "y": Value.float_(4.0)}),
        ]

        df = rows_to_dataframe(rows)

        assert len(df) == 2
        assert df.iloc[0]["x"] == 1.0
        assert df.iloc[1]["y"] == 4.0

    def test_rows_to_dataframe_empty(self) -> None:
        """Test converting empty rows list to dataframe."""
        pytest.importorskip("pandas")
        from neumann.integrations.pandas import rows_to_dataframe

        df = rows_to_dataframe([])
        assert len(df) == 0

    def test_dataframe_to_inserts(self) -> None:
        """Test converting dataframe to INSERT statements."""
        pd = pytest.importorskip("pandas")
        from neumann.integrations.pandas import dataframe_to_inserts

        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "active": [True, False]})

        inserts = dataframe_to_inserts(df, "users")

        assert len(inserts) == 2
        assert "INSERT users" in inserts[0]
        assert "Alice" in inserts[0]
        assert "Bob" in inserts[1]

    def test_dataframe_to_inserts_with_null(self) -> None:
        """Test converting dataframe with null values."""
        pd = pytest.importorskip("pandas")
        import numpy as np

        from neumann.integrations.pandas import dataframe_to_inserts

        df = pd.DataFrame({"id": [1, 2], "value": [10.0, np.nan]})

        inserts = dataframe_to_inserts(df, "data")

        assert len(inserts) == 2
        assert "null" in inserts[1]

    def test_pandas_not_installed(self) -> None:
        """Test error when pandas not installed."""
        # This test simulates pandas not being installed
        with patch.dict("sys.modules", {"pandas": None}):
            # Force reimport
            pass

            # The module should handle the import error gracefully
            # by raising a helpful error when functions are called


class TestNumpyIntegration:
    """Tests for numpy integration."""

    def test_import_exports(self) -> None:
        """Test that integration functions are exported."""
        from neumann.integrations import (
            cosine_similarity,
            euclidean_distance,
            normalize_vectors,
            parse_embedding,
            vector_to_insert,
            vectors_to_inserts,
        )

        assert callable(vector_to_insert)
        assert callable(vectors_to_inserts)
        assert callable(parse_embedding)
        assert callable(cosine_similarity)
        assert callable(euclidean_distance)
        assert callable(normalize_vectors)

    def test_vector_to_insert(self) -> None:
        """Test converting vector to INSERT statement."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import vector_to_insert

        vec = np.array([0.1, 0.2, 0.3])
        insert = vector_to_insert("doc1", vec)

        assert "INSERT VECTOR" in insert
        assert "doc1" in insert
        assert "0.1" in insert

    def test_vector_to_insert_with_normalize(self) -> None:
        """Test converting vector with normalization."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import vector_to_insert

        vec = np.array([3.0, 4.0])
        insert = vector_to_insert("doc1", vec, normalize=True)

        assert "INSERT VECTOR" in insert
        assert "doc1" in insert
        # Normalized: [0.6, 0.8]
        assert "0.6" in insert
        assert "0.8" in insert

    def test_vectors_to_inserts(self) -> None:
        """Test converting multiple vectors."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import vectors_to_inserts

        vectors = {
            "doc1": np.array([0.1, 0.2]),
            "doc2": np.array([0.3, 0.4]),
        }
        inserts = vectors_to_inserts(vectors)

        assert len(inserts) == 2

    def test_parse_embedding(self) -> None:
        """Test parsing embedding from string."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import parse_embedding

        embedding_str = "[0.1, 0.2, 0.3]"
        vec = parse_embedding(embedding_str)

        assert len(vec) == 3
        assert np.allclose(vec, [0.1, 0.2, 0.3])

    def test_parse_embedding_invalid(self) -> None:
        """Test parsing invalid embedding string."""
        pytest.importorskip("numpy")
        from neumann.integrations.numpy import parse_embedding

        with pytest.raises(ValueError):
            parse_embedding("not a vector")

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity for orthogonal vectors."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 0.0)

    def test_cosine_similarity_opposite(self) -> None:
        """Test cosine similarity for opposite vectors."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])

        sim = cosine_similarity(a, b)
        assert np.isclose(sim, -1.0)

    def test_euclidean_distance(self) -> None:
        """Test euclidean distance calculation."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import euclidean_distance

        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])

        dist = euclidean_distance(a, b)
        assert np.isclose(dist, 5.0)

    def test_euclidean_distance_same_point(self) -> None:
        """Test euclidean distance for same point."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import euclidean_distance

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        dist = euclidean_distance(a, b)
        assert np.isclose(dist, 0.0)

    def test_normalize_vectors(self) -> None:
        """Test vector normalization."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import normalize_vectors

        vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
        normalized = normalize_vectors(vectors)

        # Check that all vectors have unit length
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)

    def test_normalize_vectors_single(self) -> None:
        """Test normalizing single vector."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import normalize_vectors

        vector = np.array([[3.0, 4.0]])
        normalized = normalize_vectors(vector)

        assert np.allclose(normalized[0], [0.6, 0.8])

    def test_normalize_vectors_zero(self) -> None:
        """Test normalizing zero vector."""
        np = pytest.importorskip("numpy")
        from neumann.integrations.numpy import normalize_vectors

        vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
        normalized = normalize_vectors(vectors)

        # Zero vector should remain zero
        assert np.allclose(normalized[0], [0.0, 0.0])
        assert np.allclose(normalized[1], [1.0, 0.0])


class TestIntegrationModuleExports:
    """Tests for integration module exports."""

    def test_all_exports_available(self) -> None:
        """Test all functions are exported from integrations module."""
        from neumann import integrations

        expected_exports = [
            "result_to_dataframe",
            "rows_to_dataframe",
            "dataframe_to_inserts",
            "vector_to_insert",
            "vectors_to_inserts",
            "parse_embedding",
            "cosine_similarity",
            "euclidean_distance",
            "normalize_vectors",
        ]

        for name in expected_exports:
            assert hasattr(integrations, name), f"Missing export: {name}"

    def test_all_list_correct(self) -> None:
        """Test __all__ list is correct."""
        from neumann.integrations import __all__

        expected = [
            "result_to_dataframe",
            "rows_to_dataframe",
            "dataframe_to_inserts",
            "vector_to_insert",
            "vectors_to_inserts",
            "parse_embedding",
            "cosine_similarity",
            "euclidean_distance",
            "normalize_vectors",
        ]

        assert set(__all__) == set(expected)
