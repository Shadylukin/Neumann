"""Generate dense embeddings for paper titles and abstracts.

Loads papers from a JSONL file and produces 384-dimensional embeddings
using the all-MiniLM-L6-v2 sentence-transformer model. Results are saved
as a numpy array of shape (N, 384).
"""

import argparse
import json
import logging
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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


def embed_papers(
    papers: list[dict], model_name: str, batch_size: int
) -> np.ndarray:
    """Embed paper titles and abstracts into dense vectors.

    Concatenates each paper's title and abstract, then encodes the
    combined text using the specified sentence-transformer model.

    Returns:
        numpy array of shape (N, 384).
    """
    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)

    texts = []
    for paper in papers:
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        combined = (title + " " + abstract).strip()
        texts.append(combined if combined else "untitled")

    logger.info("Embedding %d texts in batches of %d", len(texts), batch_size)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

    result = np.vstack(all_embeddings)
    logger.info("Embedding matrix shape: %s", result.shape)
    return result


def main() -> None:
    """Entry point for the embedding CLI."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for papers using sentence-transformers"
    )
    parser.add_argument(
        "--input",
        default="papers.jsonl",
        help="Input JSONL file path (default: papers.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="embeddings.npy",
        help="Output numpy file path (default: embeddings.npy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256)",
    )
    args = parser.parse_args()

    try:
        papers = load_papers(args.input)
        if not papers:
            logger.error("No papers found in %s", args.input)
            sys.exit(1)

        embeddings = embed_papers(
            papers, model_name="all-MiniLM-L6-v2", batch_size=args.batch_size
        )
        np.save(args.output, embeddings)
        logger.info("Saved embeddings to %s", args.output)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
