"""Fetch the fresh task (F8): new arXiv papers and their primary category.

The papers were submitted from 2026-10-10 to 2026-10-15. This is after the release
dates of almost all systems under test, so no system saw them in training. arXiv
metadata is CC0, so we keep the file in git (fixtures/).

  uv run python src/fetch_arxiv.py
"""
import json
import pathlib
import time
import xml.etree.ElementTree as ET

import httpx

from tasks import ARXIV_CATEGORIES

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "arxiv-fresh-2026-10.jsonl"
API = "https://export.arxiv.org/api/query"
WINDOW = "[202610100000 TO 202610152359]"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PER_CATEGORY = 40


def fetch(category):
    params = {
        "search_query": f"cat:{category} AND submittedDate:{WINDOW}",
        "max_results": 400,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    r = httpx.get(API, params=params, timeout=120, follow_redirects=True)
    r.raise_for_status()
    papers = []
    for e in ET.fromstring(r.text).findall("a:entry", NS):
        primary = e.find("arxiv:primary_category", NS).get("term")
        if primary != category:
            continue  # a cross-list: the paper has another primary category
        papers.append({
            "arxiv_id": e.find("a:id", NS).text.rsplit("/", 1)[-1],
            "published": e.find("a:published", NS).text,
            "title": " ".join(e.find("a:title", NS).text.split()),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
            "primary": primary,
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
        })
    return papers[:PER_CATEGORY]


def main():
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for category in ARXIV_CATEGORIES:
            papers = fetch(category)
            print(f"{category}: {len(papers)} papers")
            for p in papers:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            time.sleep(3)  # the arXiv API asks for 3 seconds between calls


if __name__ == "__main__":
    main()
