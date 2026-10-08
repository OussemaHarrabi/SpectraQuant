"""Citation check (Agent L): compare docs/research/novelty-risk.md and literature-matrix.csv
claims against the real arXiv abstract page of each cited work.

Uses the raw HTML already fetched into docs/results/verification/raw/abs_<id>.html by the
fetch step (HTTP GET https://arxiv.org/abs/<id>). Prints title/date/authors/comments and the
abstract, so every novelty verdict can be checked against the source.
"""
import csv
import html
import re
import sys

ids = {
    # novelty-risk P1..P6
    "P1": "2609.33923", "P2": "2210.17323", "P3": "1911.03852", "P4a": "2402.14866",
    "P4b": "2607.07964", "P5a": "2607.23047", "P5b": "2509.15455", "P6": "2311.12023",
    # novelty-risk R1..R5
    "R1": "2602.02001", "R2": "2505.08022", "R3": "1802.05957", "R4": "2406.06623",
    "R5a": "2312.05821", "R5b": "2403.07378",
    # novelty-risk C1..C6
    "C1": "2507.09616", "C3": "2602.22268", "C4": "2609.24298",
    # neighbours NOT cited anywhere in docs/ (checked with grep)
    "UNCITED-SVDQuant": "2411.05007", "UNCITED-JoLT": "2607.12550",
}

matrix = {r["arxiv_id"]: r for r in csv.DictReader(
    open("docs/results/verification/snapshot/literature-matrix.csv", encoding="utf-8")) if r["arxiv_id"]}


def field(b, pat):
    m = re.search(pat, b, re.S | re.I)
    return html.unescape(re.sub("<[^>]+>", " ", m.group(1))).strip() if m else None


only = sys.argv[1:] or list(ids)
for tag in only:
    aid = ids[tag]
    b = open(f"docs/results/verification/raw/abs_{aid}.html", encoding="utf-8", errors="replace").read()
    title = field(b, r'<meta\s+name="citation_title"\s+content="([^"]+)"')
    date = field(b, r'<meta\s+name="citation_date"\s+content="([^"]+)"')
    authors = re.findall(r'<meta\s+name="citation_author"\s+content="([^"]+)"', b)
    comments = field(b, r'<td class="tablecell comments[^"]*">(.*?)</td>')
    abstract = re.sub(r"\s+", " ", field(b, r'<blockquote class="abstract[^"]*">(.*?)</blockquote>') or "")
    mrow = matrix.get(aid)
    print("=" * 118)
    print(f"[{tag}] arXiv:{aid}")
    print(f"  fetched title : {title}")
    print(f"  matrix title  : {mrow['title'] if mrow else '(absent from matrix)'}")
    print(f"  fetched date  : {date}   matrix year: {mrow['year'] if mrow else '-'}")
    print(f"  fetched authors: {authors[:4]}")
    print(f"  fetched comments: {comments}")
    print(f"  matrix venue  : {mrow['venue'] if mrow else '-'}  | venue_verified: "
          f"{mrow['venue_verified'] if mrow else '-'}")
    print(f"  abstract      : {abstract[:1200]}")
