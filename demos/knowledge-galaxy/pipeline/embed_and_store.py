"""Embed paper titles using sentence-transformers and store in Neumann.

Reads the currently loaded graph nodes from the galaxy server, embeds
their titles with all-MiniLM-L6-v2 (384-dim), and stores the embeddings
via EMBED STORE so that FIND NODE SIMILAR TO works.
"""

import argparse
import logging
import sys

import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_papers(web_url: str) -> list[dict]:
    """Fetch all loaded papers from the galaxy server."""
    resp = requests.post(
        f"{web_url}/api/galaxy",
        json={"query": "NODE LIST LIMIT 50000"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        logger.error("API error: %s", data["error"])
        return []

    papers = []
    for node in data.get("items", []):
        props = node.get("properties", {})
        label = node.get("label", "")
        title = props.get("title", "")
        authors = props.get("authors", "")
        category = props.get("category", "")
        if label and title:
            papers.append({
                "label": label,
                "title": title,
                "authors": authors,
                "category": category,
                "text": f"{title}. {authors}",
            })
    return papers


def store_embeddings(
    papers: list[dict],
    embeddings: list[list[float]],
    web_url: str,
) -> None:
    """Store embeddings via EMBED STORE on the galaxy server."""
    query_url = f"{web_url}/api/execute"
    errors = 0

    for paper, emb in zip(
        tqdm(papers, desc="Storing embeddings", unit="emb"),
        embeddings,
    ):
        key = f"paper:{paper['label']}"
        vec_str = ", ".join(f"{v:.6f}" for v in emb)
        query = f"EMBED STORE '{key}' [{vec_str}]"

        try:
            resp = requests.post(
                query_url,
                json={"query": query},
                timeout=30,
            )
            data = resp.json()
            if resp.status_code != 200 or data.get("error"):
                errors += 1
                if errors <= 3:
                    logger.warning(
                        "EMBED STORE failed for %s: %s",
                        key,
                        data.get("error", resp.text[:200]),
                    )
        except requests.RequestException as exc:
            errors += 1
            if errors <= 3:
                logger.warning("EMBED STORE error: %s", exc)

    logger.info(
        "Embeddings stored: %d succeeded, %d failed",
        len(papers) - errors,
        errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed paper titles and store in Neumann"
    )
    parser.add_argument(
        "--web-url",
        default="http://localhost:9000",
        help="Galaxy server web URL (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )
    args = parser.parse_args()

    # Fetch papers from server
    papers = fetch_papers(args.web_url)
    if not papers:
        logger.error("No papers found in the server")
        sys.exit(1)
    logger.info("Fetched %d papers from server", len(papers))

    # Load model and embed
    logger.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model)
    logger.info("Model loaded (dim=%d)", model.get_sentence_embedding_dimension())

    texts = [p["text"] for p in papers]
    logger.info("Embedding %d texts...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    logger.info("Embedding complete, storing in Neumann...")
    store_embeddings(papers, embeddings.tolist(), args.web_url)
    logger.info("Done! FIND NODE SIMILAR TO should now work.")


if __name__ == "__main__":
    main()
