"""Quick loader: fetch real CS papers from OpenAlex and load into galaxy-server.

This is a self-contained script that:
1. Fetches papers from the OpenAlex API (no API key needed)
2. Loads them as graph nodes via the /api/query endpoint
3. Inserts 3D positions (clustered by category) via the spatial REST API
4. Creates citation edges between papers in the dataset

No ML dependencies required (no sentence-transformers, no UMAP).
Positions are deterministic based on category clustering.
"""

import argparse
import json
import logging
import math
import sys
import time

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org"

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

# 3D cluster centers for each category (spread around a sphere)
CATEGORIES = ["AI", "ML", "CV", "NLP", "SE", "DB", "Systems", "Theory", "Other"]
CATEGORY_CENTERS = {}
for i, cat in enumerate(CATEGORIES):
    angle = (i / len(CATEGORIES)) * 2 * math.pi
    CATEGORY_CENTERS[cat] = (
        math.cos(angle) * 200,
        (i % 3 - 1) * 80,
        math.sin(angle) * 200,
    )


def normalize_id(url: str) -> str:
    return url.replace("https://openalex.org/", "")


def reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda wp: wp[0])
    return " ".join(word for _, word in word_positions)


def classify_paper(paper: dict) -> str:
    topic = paper.get("primary_topic")
    if not topic:
        return "Other"
    subfield = topic.get("subfield", {})
    display_name = subfield.get("display_name", "") if subfield else ""
    return CATEGORY_MAP.get(display_name, "Other")


def fetch_papers(max_papers: int) -> list[dict]:
    """Fetch CS papers from OpenAlex with cursor-based pagination."""
    cursor = "*"
    papers = []
    works_url = f"{OPENALEX_URL}/works"

    with tqdm(total=max_papers, desc="Fetching papers", unit="paper") as pbar:
        while len(papers) < max_papers:
            params = {
                "filter": "type:article,primary_topic.field.id:fields/17",
                "select": (
                    "id,title,abstract_inverted_index,publication_year,"
                    "authorships,primary_topic,referenced_works,cited_by_count"
                ),
                "per_page": 200,
                "cursor": cursor,
            }

            delay = 1.0
            for attempt in range(5):
                resp = requests.get(works_url, params=params, timeout=30)
                if resp.status_code == 429:
                    logger.warning("Rate limited, waiting %.1fs", delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                break
            else:
                resp = requests.get(works_url, params=params, timeout=30)
                resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for paper in results:
                if len(papers) >= max_papers:
                    break

                short_id = normalize_id(paper.get("id", ""))
                title = paper.get("title", "") or ""
                abstract_text = reconstruct_abstract(
                    paper.get("abstract_inverted_index")
                )
                category = classify_paper(paper)
                year = paper.get("publication_year") or 0
                cited_by = paper.get("cited_by_count", 0) or 0
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in (paper.get("authorships") or [])
                    if a.get("author", {}).get("display_name")
                ]
                refs = [
                    normalize_id(r)
                    for r in (paper.get("referenced_works") or [])
                ]

                papers.append({
                    "short_id": short_id,
                    "title": title,
                    "abstract": abstract_text[:200],
                    "category": category,
                    "year": year,
                    "cited_by_count": cited_by,
                    "authors": authors[:5],
                    "referenced_works": refs,
                })
                pbar.update(1)

            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor

    logger.info("Fetched %d papers from OpenAlex", len(papers))
    return papers


def escape_sql(s: str) -> str:
    """Escape a string for Neumann query literals."""
    import re
    s = re.sub(r"<[^>]+>", "", s)  # Strip HTML tags from OpenAlex data
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", " ")


# Offset added to all coordinates to keep them positive (parser doesn't
# handle negative number literals in property values).  The frontend
# subtracts this same offset to center the scene.
COORD_OFFSET = 500.0


def generate_position(category: str, index: int) -> tuple[float, float, float]:
    """Generate a deterministic 3D position based on category clustering."""
    cx, cy, cz = CATEGORY_CENTERS.get(category, (0, 0, 0))

    # Use index as seed for deterministic spread within cluster
    # Golden ratio based distribution for even spread
    phi = (1 + math.sqrt(5)) / 2
    theta = 2 * math.pi * index * phi
    r = 30 + (index * 7.31) % 100  # radius spread
    y_offset = ((index * 13.37) % 200) - 100

    x = cx + math.cos(theta) * r + COORD_OFFSET
    y = cy + y_offset + COORD_OFFSET
    z = cz + math.sin(theta) * r + COORD_OFFSET

    return (x, y, z)


def execute_query(url: str, query: str) -> dict | None:
    """Execute a query and return the parsed JSON, or None on error."""
    try:
        resp = requests.post(url, json={"query": query}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("error"):
            return None
        return data
    except requests.RequestException:
        return None


def load_into_server(
    papers: list[dict],
    web_url: str,
    rest_url: str,
) -> None:
    """Load papers into the running galaxy-server."""

    query_url = f"{web_url}/api/execute"
    spatial_url = f"{rest_url}/collections/galaxy/spatial3d/insert"

    # Phase 1: Create nodes (include 3D positions as properties)
    # Track label -> integer ID mapping for edge creation
    label_to_id: dict[str, int] = {}
    logger.info("Creating %d paper nodes via %s", len(papers), query_url)
    node_errors = 0
    for i, paper in enumerate(tqdm(papers, desc="Creating nodes", unit="node")):
        sid = paper["short_id"]
        title = escape_sql(paper["title"])
        cat = escape_sql(paper["category"])
        year = paper["year"]
        cited = paper["cited_by_count"]
        authors_str = escape_sql(", ".join(paper["authors"]))
        x, y, z = generate_position(paper["category"], i)

        query = (
            f"NODE CREATE {sid} {{ "
            f"title: '{title}', "
            f"category: '{cat}', "
            f"year: {year}, "
            f"cited_by_count: {cited}, "
            f"authors: '{authors_str}', "
            f"x: {x:.2f}, "
            f"y: {y:.2f}, "
            f"z: {z:.2f} "
            f"}}"
        )
        result = execute_query(query_url, query)
        if result and result.get("items"):
            node_id = result["items"][0]
            label_to_id[sid] = node_id
        else:
            node_errors += 1
            if node_errors <= 3:
                logger.warning("Node create failed for %s", sid)

    logger.info(
        "Nodes created: %d succeeded, %d failed",
        len(label_to_id), node_errors,
    )

    # Phase 2: Insert 3D positions
    logger.info("Inserting spatial positions via %s", spatial_url)
    spatial_errors = 0
    for i, paper in enumerate(tqdm(papers, desc="Inserting positions", unit="point")):
        x, y, z = generate_position(paper["category"], i)

        payload = {
            "key": f"paper:{paper['short_id']}",
            "x": x,
            "y": y,
            "z": z,
            "w": 1.0,
            "h": 1.0,
            "d": 1.0,
        }
        try:
            resp = requests.post(spatial_url, json=payload, timeout=10)
            if resp.status_code != 200:
                spatial_errors += 1
                if spatial_errors <= 3:
                    logger.warning("Spatial insert failed (%d): %s", resp.status_code, resp.text[:200])
        except requests.RequestException as exc:
            spatial_errors += 1
            if spatial_errors <= 3:
                logger.warning("Spatial insert error: %s", exc)

    logger.info(
        "Spatial: %d succeeded, %d failed",
        len(papers) - spatial_errors,
        spatial_errors,
    )

    # Phase 3: Create citation edges using integer node IDs
    logger.info("Creating citation edges")
    edge_count = 0
    edge_errors = 0
    for paper in tqdm(papers, desc="Creating edges", unit="paper"):
        src_label = paper["short_id"]
        src_id = label_to_id.get(src_label)
        if src_id is None:
            continue
        for ref_id in paper["referenced_works"]:
            dst_id = label_to_id.get(ref_id)
            if dst_id is not None:
                query = f"EDGE CREATE {src_id} -> {dst_id} : cites"
                result = execute_query(query_url, query)
                if result:
                    edge_count += 1
                else:
                    edge_errors += 1

    logger.info("Edges: %d created, %d failed", edge_count, edge_errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch real CS papers and load into galaxy-server"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=2000,
        help="Number of papers to fetch (default: 2000)",
    )
    parser.add_argument(
        "--web-url",
        default="http://localhost:9000",
        help="Galaxy server web URL (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--rest-url",
        default="http://localhost:8080",
        help="Galaxy server REST URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--save-jsonl",
        default=None,
        help="Also save papers to a JSONL file",
    )
    args = parser.parse_args()

    papers = fetch_papers(args.max_papers)
    if not papers:
        logger.error("No papers fetched")
        sys.exit(1)

    if args.save_jsonl:
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("Saved %d papers to %s", len(papers), args.save_jsonl)

    load_into_server(papers, args.web_url, args.rest_url)
    logger.info("Done! Refresh http://localhost:5173 to see the galaxy.")


if __name__ == "__main__":
    main()
