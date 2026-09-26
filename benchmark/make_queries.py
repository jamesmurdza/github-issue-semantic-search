"""Build the benchmark's queries, benchmark/queries.json.

Every query has one right answer, an issue, and is one of four kinds:

  duplicate   The title of an issue a maintainer marked as a duplicate. The
              right answer is the issue it duplicates, and the duplicate itself
              is hidden from the search. A real user's words for a problem
              someone had already reported: the closest thing here to real use.
  title       An issue's title, with the issue's body hidden, so the search has
              to find the thread through its comments.
  paraphrase  The issue's problem in everyday words, avoiding its distinctive
              terms. Written by an agent that read the issue (writer_brief.md).
  specific    A concrete detail from the issue, an error message or a function
              name, as someone holding it would type it. Same agent.

It takes two commands, because an agent writes queries in between:

  uv run python -m benchmark.make_queries sample <repo> <out.json>
      50 random issues, as the query writer reads them. The agent writes two
      queries for each into written_queries.json, following writer_brief.md.
  uv run python -m benchmark.make_queries build
      duplicates from GitHub, titles from the index and the written queries,
      into queries.json.

Both read the index at QDRANT_URL, which must hold REPOS, synced by
`gh-issue sync`. `build` also needs GITHUB_TOKEN.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Literal, TypedDict

import cyclopts
import httpx
from qdrant_client import QdrantClient, models

from gh_issue.index.store import COLLECTION
from gh_issue.settings import qdrant_url

REPOS = ["inngest/inngest", "pydantic/pydantic-ai"]
HERE = Path(__file__).parent
QUERIES = HERE / "queries.json"
WRITTEN = HERE / "written_queries.json"

TITLE_QUERIES = 150
SEED = 2026

Kind = Literal["duplicate", "title", "paraphrase", "specific"]


class Hidden(TypedDict, total=False):
    """Chunks the search must not see for this query: all of `issue`, or only
    its chunks of `kind`."""

    issue: int
    kind: str


class Query(TypedDict):
    repo: str
    kind: Kind
    text: str
    # The issue number of the right answer.
    target: int
    hide: Hidden | None


class Written(TypedDict):
    repo: str
    issue: int
    paraphrase: str
    specific: str


app = cyclopts.App(help=__doc__)


def indexed_chunks(repo: str, fields: list[str]) -> list[dict[str, Any]]:
    """The payload `fields` of every chunk of `repo` in the index."""
    client = QdrantClient(url=qdrant_url(), timeout=60)
    where = models.Filter(
        must=[models.FieldCondition(key="repo", match=models.MatchValue(value=repo))]
    )
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        page, offset = client.scroll(
            COLLECTION, scroll_filter=where, limit=1000, offset=offset, with_payload=fields
        )
        payloads.extend(point.payload for point in page if point.payload)
        if offset is None:
            return payloads


@app.command
def sample(repo: str, out: Path, count: int = 50) -> None:
    """Random issues from the index, as the query writer reads them: the title,
    the body and the first three comments, each cut short."""
    threads: dict[int, dict[str, Any]] = {}
    for chunk in indexed_chunks(repo, ["issue_number", "kind", "title", "text", "created_at"]):
        thread = threads.setdefault(
            chunk["issue_number"], {"title": chunk["title"], "body": "", "comments": []}
        )
        if chunk["kind"] == "body":
            # The body chunk starts with the title; the writer sees it once.
            thread["body"] = chunk["text"].removeprefix(f"{chunk['title']}\n\n")
        else:
            thread["comments"].append((chunk["created_at"], chunk["text"]))
    picked = random.Random(SEED).sample(sorted(threads), count)
    issues = [
        {
            "number": number,
            "title": threads[number]["title"],
            "body": threads[number]["body"][:2500],
            "first_comments": [text[:600] for _, text in sorted(threads[number]["comments"])[:3]],
        }
        for number in picked
    ]
    out.write_text(json.dumps(issues, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(issues)} issues from {repo} written to {out}")


DUPLICATES = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        timelineItems(itemTypes: [MARKED_AS_DUPLICATE_EVENT], first: 10) {
          nodes { ... on MarkedAsDuplicateEvent {
            duplicate { ... on Issue { number repository { nameWithOwner } } }
            canonical { ... on Issue { number repository { nameWithOwner } } }
          } }
        }
      }
    }
  }
}
"""


def duplicate_pairs(repo: str) -> set[tuple[int, int]]:
    """(duplicate, original) for every issue marked a duplicate of another
    issue in the same repo. The event shows on both issues, hence the set."""
    owner, name = repo.split("/")
    pairs: set[tuple[int, int]] = set()
    after = None
    with httpx.Client(
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}, timeout=60
    ) as github:
        while True:
            response = github.post(
                "https://api.github.com/graphql",
                json={
                    "query": DUPLICATES,
                    "variables": {"owner": owner, "name": name, "after": after},
                },
            )
            response.raise_for_status()
            issues = response.json()["data"]["repository"]["issues"]
            for issue in issues["nodes"]:
                for event in issue["timelineItems"]["nodes"]:
                    duplicate, original = event.get("duplicate") or {}, event.get("canonical") or {}
                    same_repo = (
                        duplicate.get("repository", {}).get("nameWithOwner")
                        == original.get("repository", {}).get("nameWithOwner")
                        == repo
                    )
                    if same_repo and duplicate["number"] != original["number"]:
                        pairs.add((duplicate["number"], original["number"]))
            if not issues["pageInfo"]["hasNextPage"]:
                return pairs
            after = issues["pageInfo"]["endCursor"]


def queries_for(repo: str, written: list[Written]) -> list[Query]:
    chunks = indexed_chunks(repo, ["issue_number", "kind", "title"])
    titles = {chunk["issue_number"]: chunk["title"] for chunk in chunks}
    queries: list[Query] = []

    for duplicate, original in sorted(duplicate_pairs(repo)):
        if duplicate in titles and original in titles:
            queries.append(
                {
                    "repo": repo,
                    "kind": "duplicate",
                    "text": titles[duplicate],
                    "target": original,
                    "hide": {"issue": duplicate},
                }
            )

    # Only issues with comments: with the body hidden, nothing else is left.
    discussed = sorted({chunk["issue_number"] for chunk in chunks if chunk["kind"] == "comment"})
    for number in random.Random(SEED).sample(discussed, min(TITLE_QUERIES, len(discussed))):
        queries.append(
            {
                "repo": repo,
                "kind": "title",
                "text": titles[number],
                "target": number,
                "hide": {"issue": number, "kind": "body"},
            }
        )

    for item in written:
        if item["repo"] == repo:
            for kind in ("paraphrase", "specific"):
                queries.append(
                    {
                        "repo": repo,
                        "kind": kind,
                        "text": item[kind],
                        "target": item["issue"],
                        "hide": None,
                    }
                )
    return queries


@app.command
def build() -> None:
    """Every repo's queries into queries.json."""
    written: list[Written] = json.loads(WRITTEN.read_text(encoding="utf-8"))
    queries = [query for repo in REPOS for query in queries_for(repo, written)]
    QUERIES.write_text(json.dumps(queries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for query in queries:
        key = f"{query['repo']} {query['kind']}"
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        print(f"{count:5}  {key}")
    print(f"{len(queries):5}  queries written to {QUERIES}")


if __name__ == "__main__":
    app()
