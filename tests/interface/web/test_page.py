"""The page in a real browser, against the real server.

No GitHub token is set, so nothing here reaches GitHub: its list shows the
sentence for a missing token, which is also the proof that one list failing
leaves the other working.
"""

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from playwright.sync_api import Page, expect, sync_playwright

from gh_issue.index.store import Store
from gh_issue.interface.web.app import make_app

MIGRATIONS = (
    "after upgrading the server fails to start because the database migration files are missing"
)


@pytest.fixture
def served(indexed: Store, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The app on a free local port, served by uvicorn as the command does."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(make_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline and thread.is_alive(), "the server never started"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def page() -> Iterator[Page]:
    """A page in a real Chromium, opened and closed inside the test.

    pytest-playwright offers the same fixture, but keeps one browser for the
    whole run, and Playwright's event loop with it. Tried: an asyncio.run() and
    an async test after a browser test both failed, and this suite has both.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            yield browser.new_page()
        finally:
            browser.close()


def test_the_repos_to_choose_from_are_the_ones_in_the_index(served: str, page: Page) -> None:
    page.goto(served)

    repo = page.get_by_role("combobox", name="Repo")
    expect(repo.get_by_role("option")).to_have_text(["other/repo", "owner/name"])
    expect(repo).to_have_value("other/repo")


def test_a_question_shows_the_comment_that_answers_it_first(served: str, page: Page) -> None:
    page.goto(served)
    question = page.get_by_role("searchbox", name="Question")
    expect(question).to_be_focused()

    page.get_by_role("combobox", name="Repo").select_option("owner/name")
    question.fill(MIGRATIONS)
    question.press("Enter")

    first = page.get_by_role("region", name="gh-issue").get_by_role("listitem").first
    expect(first).to_contain_text("App freezes when I scroll fast")
    expect(first).to_contain_text("#412")
    expect(first).to_contain_text("Comment by maintainer, 2 Jan 2026")
    expect(first).to_contain_text("cannot find the sqlite migrations folder")
    expect(first.get_by_role("link")).to_have_attribute(
        "href", "https://github.com/owner/name/issues/412#issuecomment-9981"
    )

    github = page.get_by_role("region", name="GitHub search")
    expect(github).to_contain_text("github.token is not set")
    expect(github.get_by_role("listitem")).to_have_count(0)


def test_a_comment_that_matched_is_shown_whole(served: str, page: Page) -> None:
    """Its last words are often the answer. Seen on inngest/inngest: three
    lines of a comment were shown, the error message it was found by was on
    the fourth, and half of that line showed through under the third.

    The window is narrow here so that the test comment takes more than three
    lines. The server has already cut it to a snippet's length.
    """
    page.set_viewport_size({"width": 320, "height": 800})
    page.goto(served)
    page.get_by_role("combobox", name="Repo").select_option("owner/name")
    question = page.get_by_role("searchbox", name="Question")
    question.fill(MIGRATIONS)
    question.press("Enter")

    comment = page.get_by_role("region", name="gh-issue").get_by_role("listitem").first
    post = comment.get_by_text("Downgrading to 1.3.3 fixed it for me.")
    expect(post).to_be_visible()
    lines = post.evaluate(
        """node => {
            const style = getComputedStyle(node);
            const padding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
            return Math.round((node.scrollHeight - padding) / parseFloat(style.lineHeight));
        }"""
    )
    assert lines > 3, "the window is not narrow enough to test this"
    assert post.evaluate("node => node.scrollHeight <= node.clientHeight")


def test_choosing_another_repo_asks_the_same_question_there(served: str, page: Page) -> None:
    """Otherwise the lists would go on showing the repo that was left."""
    page.goto(served)
    repo = page.get_by_role("combobox", name="Repo")
    repo.select_option("owner/name")
    question = page.get_by_role("searchbox", name="Question")
    question.fill(MIGRATIONS)
    question.press("Enter")
    first = page.get_by_role("region", name="gh-issue").get_by_role("listitem").first
    expect(first).to_contain_text("#412")

    repo.select_option("other/repo")

    expect(first).to_contain_text("#99")
    expect(first).to_contain_text("Server does not start: the sqlite migrations folder is missing")
    expect(first.get_by_role("link")).to_have_attribute(
        "href", "https://github.com/other/repo/issues/99"
    )
