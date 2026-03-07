"""Load 3D spatial coordinates into a running galaxy-server instance.

This script runs AFTER the galaxy-server has started. It reads the UMAP-
projected 3D coordinates and paper metadata, then POSTs each point to the
server's spatial index endpoint for efficient 3D range queries.
"""

import argparse
import json
import logging
import sys

import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_id(url: str) -> str:
    """Strip the OpenAlex URL prefix to get a short identifier."""
    return url.replace("https://openalex.org/", "")


def load_papers(path: str) -> list[dict]:
    """Load papers from a JSONL file."""
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    logger.info("Loaded %d papers from %s", len(papers), path)
    return papers


def insert_spatial_points(
    papers: list[dict],
    coords: np.ndarray,
    rest_url: str,
) -> None:
    """POST each paper's 3D coordinates to the galaxy-server spatial index.

    Each point is sent as an individual request to the spatial3d insert
    endpoint.
    """
    endpoint = f"{rest_url}/collections/galaxy/spatial3d/insert"
    logger.info("Inserting %d spatial points via %s", len(papers), endpoint)

    success_count = 0
    error_count = 0

    for i, paper in enumerate(tqdm(papers, desc="Inserting spatial points", unit="point")):
        if i >= coords.shape[0]:
            logger.warning("Ran out of coordinates at index %d", i)
            break

        short_id = paper.get("short_id", normalize_id(paper.get("id", "")))
        x, y, z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])

        payload = {
            "key": f"paper:{short_id}",
            "x": x,
            "y": y,
            "z": z,
            "w": 1.0,
            "h": 1.0,
            "d": 1.0,
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            success_count += 1
        except requests.RequestException as exc:
            error_count += 1
            if error_count <= 5:
                logger.warning("Failed to insert point %s: %s", short_id, exc)
            elif error_count == 6:
                logger.warning("Suppressing further error messages...")

    logger.info(
        "Spatial insert complete: %d succeeded, %d failed",
        success_count,
        error_count,
    )


def main() -> None:
    """Entry point for the spatial loader CLI."""
    parser = argparse.ArgumentParser(
        description="Load 3D coordinates into a running galaxy-server"
    )
    parser.add_argument(
        "--coords",
        default="coords_3d.npy",
        help="Input numpy 3D coordinates file (default: coords_3d.npy)",
    )
    parser.add_argument(
        "--papers",
        default="papers.jsonl",
        help="Input JSONL papers file (default: papers.jsonl)",
    )
    parser.add_argument(
        "--rest-url",
        default="http://localhost:8080",
        help="Galaxy-server REST URL (default: http://localhost:8080)",
    )
    args = parser.parse_args()

    try:
        papers = load_papers(args.papers)
        if not papers:
            logger.error("No papers found in %s", args.papers)
            sys.exit(1)

        coords = np.load(args.coords)
        logger.info("Loaded coordinates: shape %s", coords.shape)

        if coords.shape[0] != len(papers):
            logger.warning(
                "Mismatch: %d papers but %d coordinates",
                len(papers),
                coords.shape[0],
            )

        insert_spatial_points(papers, coords, args.rest_url)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except requests.ConnectionError:
        logger.error(
            "Cannot connect to galaxy-server at %s. "
            "Make sure the server is running first.",
            args.rest_url,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
