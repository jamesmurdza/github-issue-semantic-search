"""The text `gh-issue search`, `status` and `config` print. Plain ASCII, so it
reads the same in any terminal and survives being piped to a file."""

import datetime

from gh_issue.config import SETTINGS, Shown
from gh_issue.heartbeat import WorkerStatus
from gh_issue.index.backfills import BackfillProgress
from gh_issue.index.store import SearchHit
from gh_issue.query import Status

SNIPPET = 240


def ago(then: str, now: datetime.datetime) -> str:
    seconds = int((now - datetime.datetime.fromisoformat(then)).total_seconds())
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    days = seconds // 86400
    return f"{days} day ago" if days == 1 else f"{days} days ago"


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET else flat[: SNIPPET - 3].rstrip() + "..."


def format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No matches."
    blocks = []
    for position, hit in enumerate(hits, start=1):
        number = hit["issue_number"]
        # A body is stored as title and body together, which is what gets
        # embedded; the first line shows the title already.
        text = hit["text"]
        if hit["kind"] == "body":
            text = text.removeprefix(f"{hit['title']}\n\n")
        blocks.append(
            "\n".join(
                [
                    f"{position}. {hit['repo']}#{number} ({hit['state']}) {hit['title']}",
                    f"   {hit['kind']} by {hit['author']} on {hit['created_at'][:10]}",
                    f"   {_snippet(text)}",
                    f"   https://github.com/{hit['repo']}/issues/{number}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _worker_line(worker: WorkerStatus, now: datetime.datetime) -> str:
    start = "Start one with: gh-issue worker"
    if worker["last_seen"] is None:
        return f"Worker: not running. {start}"
    seen = ago(worker["last_seen"], now)
    if not worker["running"]:
        return f"Worker: not running (last heartbeat {seen}). {start}"
    if not worker["connected"]:
        state = (worker["state"] or "unknown").lower()
        return f"Worker: running, but not connected to Inngest ({state}; last heartbeat {seen})"
    where = "the local dev server" if worker["mode"] == "dev" else "Inngest Cloud"
    return f"Worker: connected to {where} (last heartbeat {seen})"


def _backfill(progress: BackfillProgress | None, now: datetime.datetime) -> str:
    if progress is None:
        return "-"
    if progress["finished"]:
        return "done"
    if progress["next"] is None:
        return f"queued {ago(progress['at'], now)}"
    return f"paging, last page {ago(progress['at'], now)}"


def format_status(status: Status, now: datetime.datetime) -> str:
    lines = [_worker_line(status["worker"], now), ""]
    repos = status["repos"]
    if not repos:
        lines.append("Nothing synced yet. Start with: gh-issue sync owner/name")
        return "\n".join(lines)

    header = ("REPO", "ISSUES", "CHUNKS", "BACKFILL", "LAST SWEEP", "FAILED")
    rows = [
        (
            repo["repo"],
            f"{repo['issues']:,}",
            f"{repo['chunks']:,}",
            _backfill(repo["backfill"], now),
            ago(repo["caught_up_to"], now) if repo["caught_up_to"] else "never",
            str(repo["failed"]),
        )
        for repo in repos
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    right = {1, 2, 5}  # the number columns
    for row in [header, *rows]:
        cells = [
            cell.rjust(widths[i]) if i in right else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())

    failed = sum(repo["failed"] for repo in repos)
    if failed:
        lines += ["", f"{failed} issue(s) failed every retry; `gh-issue sync` retries them."]
    return "\n".join(lines)


def format_config(path: str, rows: list[Shown]) -> str:
    """Each setting, its value and where the value came from. A setting that is
    not set says what it is for instead."""
    values = [row["value"] or "not set" for row in rows]
    key_width = max(len(row["key"]) for row in rows) + 3
    value_width = max(len(value) for value in values) + 2
    lines = [f"Settings, from {path} and the environment (which wins):", ""]
    for row, value in zip(rows, values, strict=True):
        about = row["source"] or SETTINGS[row["key"]]["help"]
        lines.append(f"  {row['key']:<{key_width}}{value:<{value_width}}{about}")
    lines += ["", "Change one: gh-issue config set <setting> [value]"]
    return "\n".join(lines)
