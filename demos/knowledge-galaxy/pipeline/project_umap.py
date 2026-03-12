"""Project high-dimensional embeddings to 3D coordinates using UMAP.

Takes the dense embedding matrix (N, 384) and produces a 3D projection
suitable for rendering in a Three.js scene. Coordinates are normalized
to the [-500, 500] range.
"""

import argparse
import json
import logging
import sys

import numpy as np
import umap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCENE_RANGE = 500.0


def project_to_3d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Run UMAP to project embeddings into 3D space.

    Args:
        embeddings: Dense embedding matrix of shape (N, D).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.

    Returns:
        numpy array of shape (N, 3) with coordinates in [-500, 500].
    """
    logger.info(
        "Running UMAP (n_components=3, n_neighbors=%d, min_dist=%.2f) on %d points",
        n_neighbors,
        min_dist,
        embeddings.shape[0],
    )
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    # Normalize each axis to [-SCENE_RANGE, SCENE_RANGE]
    for axis in range(3):
        col = coords[:, axis]
        col_min, col_max = col.min(), col.max()
        span = col_max - col_min
        if span > 0:
            coords[:, axis] = (col - col_min) / span * 2 * SCENE_RANGE - SCENE_RANGE
        else:
            coords[:, axis] = 0.0

    logger.info("Projected to shape %s, range [%.1f, %.1f]", coords.shape, -SCENE_RANGE, SCENE_RANGE)
    return coords


def main() -> None:
    """Entry point for the UMAP projection CLI."""
    parser = argparse.ArgumentParser(
        description="Project embeddings to 3D using UMAP"
    )
    parser.add_argument(
        "--input",
        default="embeddings.npy",
        help="Input numpy embeddings file (default: embeddings.npy)",
    )
    parser.add_argument(
        "--output",
        default="coords_3d.npy",
        help="Output numpy 3D coordinates file (default: coords_3d.npy)",
    )
    args = parser.parse_args()

    try:
        embeddings = np.load(args.input)
        logger.info("Loaded embeddings: shape %s", embeddings.shape)

        coords = project_to_3d(embeddings)

        # Save as numpy
        np.save(args.output, coords)
        logger.info("Saved 3D coordinates to %s", args.output)

        # Also export as JSON for frontend consumption
        json_path = args.output.replace(".npy", ".json")
        coords_list = coords.tolist()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coords_list, f)
        logger.info("Saved JSON coordinates to %s (%d points)", json_path, len(coords_list))

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
