"""Fetch Computer Science papers from the OpenAlex API.

Uses cursor-based pagination to retrieve article metadata including titles,
abstracts, authors, topics, and citation information. Papers are filtered
to the Computer Science field (fields/17) and saved as JSONL.
"""

import argparse
import json
import logging
import os
import sys
import time

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"

CATEGORY_MAP = {
    "Artificial Intelligence": "AI",
    "Computer Vision and Pattern Recognition": "CV",
    "Computation and Language": "NLP",
    "Machine Learning": "ML",
    "Software Engineering": "SE",
    "Databases": "DB",
    "Computer Networks and Communications": "Systems",
    "Theoretical Computer Science": "Theory",
    "Information Systems": "DB",
    "Human-Computer Interaction": "SE",
    "Computer Science Applications": "AI",
    "Computational Theory and Mathematics": "Theory",
    "Hardware and Architecture": "Systems",
    "Signal Processing": "CV",
}


def normalize_id(url: str) -> str:
    """Strip the OpenAlex URL prefix to get a short identifier.

    Example: 'https://openalex.org/W12345' -> 'W12345'
    """
    return url.replace("https://openalex.org/", "")


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plain-text abstract from an OpenAlex inverted index.

    The inverted index maps each word to a list of positions. We invert
    it back into a sequential list of words.
    """
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda wp: wp[0])
    return " ".join(word for _, word in word_positions)


def classify_paper(paper: dict) -> str:
    """Map a paper's primary topic subfield to a short category label."""
    topic = paper.get("primary_topic")
    if not topic:
        return "Other"
    subfield = topic.get("subfield", {})
    display_name = subfield.get("display_name", "") if subfield else ""
    return CATEGORY_MAP.get(display_name, "Other")


def build_params(api_key: str | None, cursor: str) -> dict:
    """Build query parameters for the OpenAlex works endpoint."""
    params = {
        "filter": "type:article,primary_topic.field.id:fields/17",
        "select": (
            "id,title,abstract_inverted_index,publication_year,"
            "authorships,primary_topic,referenced_works,cited_by_count"
        ),
        "per_page": 200,
        "cursor": cursor,
    }
    if api_key:
        params["api_key"] = api_key
    return params


def fetch_with_backoff(url: str, params: dict, max_retries: int = 5) -> requests.Response:
    """GET request with exponential backoff on 429 (rate-limited) responses."""
    delay = 1.0
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            retry_after = resp.headers.get("X-RateLimit-Reset")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after) - time.time())
                except (ValueError, TypeError):
                    pass
            logger.warning(
                "Rate-limited (429). Retrying in %.1f s (attempt %d/%d)",
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp

    # Final attempt without catching
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def verify_field(api_key: str | None) -> None:
    """Verify that the Computer Science field exists before bulk fetching."""
    params = {"select": "id,display_name"}
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(f"{BASE_URL}/fields/17", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    logger.info("Verified field: %s (%s)", data.get("display_name"), data.get("id"))


def fetch_papers(api_key: str | None, max_papers: int, output: str) -> None:
    """Fetch papers using cursor-based pagination and save to JSONL."""
    verify_field(api_key)

    cursor = "*"
    total_fetched = 0
    works_url = f"{BASE_URL}/works"

    with open(output, "w", encoding="utf-8") as f, tqdm(
        total=max_papers, desc="Fetching papers", unit="paper"
    ) as pbar:
        while total_fetched < max_papers:
            params = build_params(api_key, cursor)
            resp = fetch_with_backoff(works_url, params)
            data = resp.json()

            results = data.get("results", [])
            if not results:
                logger.info("No more results returned. Stopping.")
                break

            for paper in results:
                if total_fetched >= max_papers:
                    break

                record = {
                    "id": paper.get("id", ""),
                    "short_id": normalize_id(paper.get("id", "")),
                    "title": paper.get("title", ""),
                    "abstract": reconstruct_abstract(
                        paper.get("abstract_inverted_index")
                    ),
                    "publication_year": paper.get("publication_year"),
                    "cited_by_count": paper.get("cited_by_count", 0),
                    "category": classify_paper(paper),
                    "authors": [
                        a.get("author", {}).get("display_name", "")
                        for a in (paper.get("authorships") or [])
                        if a.get("author", {}).get("display_name")
                    ],
                    "referenced_works": paper.get("referenced_works", []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_fetched += 1
                pbar.update(1)

            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                logger.info("No next cursor. Pagination complete.")
                break
            cursor = next_cursor

    logger.info("Fetched %d papers -> %s", total_fetched, output)


def main() -> None:
    """Entry point for the paper fetcher CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch CS papers from OpenAlex API"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENALEX_API_KEY"),
        help="OpenAlex API key (or set OPENALEX_API_KEY env var)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=50000,
        help="Maximum number of papers to fetch (default: 50000)",
    )
    parser.add_argument(
        "--output",
        default="papers.jsonl",
        help="Output JSONL file path (default: papers.jsonl)",
    )
    args = parser.parse_args()

    try:
        fetch_papers(args.api_key, args.max_papers, args.output)
    except requests.HTTPError as exc:
        logger.error("HTTP error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
