"""Load papers and embeddings into a Neumann database.

Creates entities for each paper with metadata and embeddings, then
establishes citation edges between papers that are both in the dataset.
Outputs a galaxy.db file that the galaxy-server can load.
"""

import argparse
import json
import logging
import sys

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_id(url: str) -> str:
    """Strip the OpenAlex URL prefix to get a short identifier."""
    return url.replace("https://openalex.org/", "")


def escape_sql_string(s: str) -> str:
    """Escape a string for safe inclusion in Neumann query literals."""
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s


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


def format_embedding(vec: np.ndarray) -> str:
    """Format a numpy vector as a Neumann embedding literal [x, y, z, ...]."""
    parts = ", ".join(f"{v:.6f}" for v in vec)
    return f"[{parts}]"


def create_entities(client, papers: list[dict], embeddings: np.ndarray) -> None:
    """Create paper entities with metadata and embeddings."""
    logger.info("Creating %d paper entities", len(papers))
    for i, paper in enumerate(tqdm(papers, desc="Creating entities", unit="paper")):
        short_id = paper.get("short_id", normalize_id(paper.get("id", "")))
        title_safe = escape_sql_string(paper.get("title", "") or "")
        category_safe = escape_sql_string(paper.get("category", "Other"))
        year = paper.get("publication_year", 0) or 0
        cited_by_count = paper.get("cited_by_count", 0) or 0
        authors = paper.get("authors", [])
        authors_safe = escape_sql_string(", ".join(authors[:5]))

        query = (
            f"ENTITY CREATE 'paper:{short_id}' {{ "
            f"title: '{title_safe}', "
            f"category: '{category_safe}', "
            f"year: {year}, "
            f"cited_by_count: {cited_by_count}, "
            f"authors: '{authors_safe}' "
            f"}}"
        )
        client.execute(query)

        if i < embeddings.shape[0]:
            embedding_str = format_embedding(embeddings[i])
            embed_query = f"EMBED STORE 'paper:{short_id}' {embedding_str}"
            client.execute(embed_query)


def create_citations(client, papers: list[dict]) -> None:
    """Create citation edges between papers that are both in the dataset."""
    paper_ids = set()
    for paper in papers:
        short_id = paper.get("short_id", normalize_id(paper.get("id", "")))
        paper_ids.add(short_id)

    logger.info("Creating citation edges (only between papers in dataset)")
    citation_count = 0
    for paper in tqdm(papers, desc="Creating citations", unit="paper"):
        src = paper.get("short_id", normalize_id(paper.get("id", "")))
        for ref_url in paper.get("referenced_works", []):
            dst = normalize_id(ref_url)
            if dst in paper_ids:
                query = f"ENTITY CONNECT 'paper:{src}' -> 'paper:{dst}' : cites"
                client.execute(query)
                citation_count += 1

    logger.info("Created %d citation edges", citation_count)


def main() -> None:
    """Entry point for the Neumann loader CLI."""
    parser = argparse.ArgumentParser(
        description="Load papers and embeddings into a Neumann database"
    )
    parser.add_argument(
        "--papers",
        default="papers.jsonl",
        help="Input JSONL papers file (default: papers.jsonl)",
    )
    parser.add_argument(
        "--embeddings",
        default="embeddings.npy",
        help="Input numpy embeddings file (default: embeddings.npy)",
    )
    parser.add_argument(
        "--output",
        default="galaxy.db",
        help="Output Neumann database file (default: galaxy.db)",
    )
    args = parser.parse_args()

    try:
        from neumann import NeumannClient
    except ImportError:
        logger.error(
            "neumann Python package not found. "
            "Build and install it first: pip install ./bindings/python"
        )
        sys.exit(1)

    try:
        papers = load_papers(args.papers)
        if not papers:
            logger.error("No papers found in %s", args.papers)
            sys.exit(1)

        embeddings = np.load(args.embeddings)
        logger.info("Loaded embeddings: shape %s", embeddings.shape)

        if embeddings.shape[0] != len(papers):
            logger.warning(
                "Mismatch: %d papers but %d embeddings. Using min(%d, %d).",
                len(papers),
                embeddings.shape[0],
                len(papers),
                embeddings.shape[0],
            )

        logger.info("Creating in-process Neumann database")
        client = NeumannClient.embedded()

        create_entities(client, papers, embeddings)
        create_citations(client, papers)

        client.save(args.output)
        logger.info("Saved database to %s", args.output)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
