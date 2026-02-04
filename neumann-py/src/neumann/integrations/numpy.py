# SPDX-License-Identifier: MIT
"""NumPy integration for Neumann database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def vector_to_insert(
    key: str,
    vector: npt.ArrayLike,
    *,
    normalize: bool = False,
) -> str:
    """Convert a NumPy array to a vector INSERT statement.

    Args:
        key: The key/identifier for the vector.
        vector: The vector data as a NumPy array or array-like.
        normalize: Whether to L2-normalize the vector.

    Returns:
        An INSERT query string for the vector.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for vector operations. Install with: pip install neumann-db[numpy]"
        ) from e

    arr = np.asarray(vector, dtype=np.float32)

    if normalize:
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

    vector_str = ",".join(f"{x:.6f}" for x in arr.flatten())
    return f'INSERT VECTOR key="{key}", embedding=[{vector_str}]'


def vectors_to_inserts(
    vectors: dict[str, npt.ArrayLike],
    *,
    normalize: bool = False,
) -> list[str]:
    """Convert multiple vectors to INSERT statements.

    Args:
        vectors: Dictionary mapping keys to vector data.
        normalize: Whether to L2-normalize the vectors.

    Returns:
        List of INSERT query strings.

    Raises:
        ImportError: If numpy is not installed.
    """
    return [vector_to_insert(key, vec, normalize=normalize) for key, vec in vectors.items()]


def parse_embedding(embedding_str: str) -> np.ndarray:
    """Parse an embedding string to a NumPy array.

    Args:
        embedding_str: String representation of embedding (e.g., "[1.0, 2.0, 3.0]").

    Returns:
        NumPy array of the embedding.

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If parsing fails.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for vector operations. Install with: pip install neumann-db[numpy]"
        ) from e

    # Remove brackets and split
    cleaned = embedding_str.strip("[]")
    values = [float(x.strip()) for x in cleaned.split(",")]
    return np.array(values, dtype=np.float32)


def cosine_similarity(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for vector operations. Install with: pip install neumann-db[numpy]"
        ) from e

    a_arr = np.asarray(a, dtype=np.float32).flatten()
    b_arr = np.asarray(b, dtype=np.float32).flatten()

    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def euclidean_distance(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for vector operations. Install with: pip install neumann-db[numpy]"
        ) from e

    a_arr = np.asarray(a, dtype=np.float32).flatten()
    b_arr = np.asarray(b, dtype=np.float32).flatten()

    return float(np.linalg.norm(a_arr - b_arr))


def normalize_vectors(vectors: npt.ArrayLike) -> np.ndarray:
    """L2-normalize a batch of vectors.

    Args:
        vectors: 2D array of shape (n_vectors, dim).

    Returns:
        Normalized vectors with unit L2 norm.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for vector operations. Install with: pip install neumann-db[numpy]"
        ) from e

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return cast("np.ndarray[Any, Any]", arr / norms)
