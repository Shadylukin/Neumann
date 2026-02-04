# SPDX-License-Identifier: MIT
"""Pandas integration for Neumann database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neumann.types import QueryResult, QueryResultType, Row

if TYPE_CHECKING:
    import pandas as pd


def result_to_dataframe(result: QueryResult) -> pd.DataFrame:
    """Convert a QueryResult with rows to a pandas DataFrame.

    Args:
        result: A QueryResult containing rows.

    Returns:
        A pandas DataFrame with the row data.

    Raises:
        ValueError: If result is not of type ROWS.
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for DataFrame conversion. "
            "Install with: pip install neumann-db[pandas]"
        ) from e

    if result.type != QueryResultType.ROWS:
        raise ValueError(f"Expected ROWS result, got {result.type.name}")

    data: list[dict[str, Any]] = [row.to_dict() for row in result.rows]
    return pd.DataFrame(data)


def rows_to_dataframe(rows: list[Row]) -> pd.DataFrame:
    """Convert a list of Row objects to a pandas DataFrame.

    Args:
        rows: List of Row objects.

    Returns:
        A pandas DataFrame with the row data.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for DataFrame conversion. "
            "Install with: pip install neumann-db[pandas]"
        ) from e

    data: list[dict[str, Any]] = [row.to_dict() for row in rows]
    return pd.DataFrame(data)


def dataframe_to_inserts(
    df: pd.DataFrame,
    table: str,
    *,
    column_mapping: dict[str, str] | None = None,
) -> list[str]:
    """Convert a pandas DataFrame to INSERT statements.

    Args:
        df: The pandas DataFrame to convert.
        table: The target table name.
        column_mapping: Optional mapping from DataFrame columns to table columns.

    Returns:
        List of INSERT query strings.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for DataFrame conversion. "
            "Install with: pip install neumann-db[pandas]"
        ) from e

    queries = []
    mapping = column_mapping or {}

    for _, row in df.iterrows():
        assignments = []
        for col in df.columns:
            target_col = mapping.get(col, col)
            value = row[col]

            if pd.isna(value):
                assignments.append(f"{target_col}=null")
            elif isinstance(value, bool):
                assignments.append(f"{target_col}={str(value).lower()}")
            elif isinstance(value, int | float):
                assignments.append(f"{target_col}={value}")
            elif isinstance(value, str):
                escaped = value.replace('"', '\\"')
                assignments.append(f'{target_col}="{escaped}"')
            else:
                escaped = str(value).replace('"', '\\"')
                assignments.append(f'{target_col}="{escaped}"')

        query = f"INSERT {table} {', '.join(assignments)}"
        queries.append(query)

    return queries
