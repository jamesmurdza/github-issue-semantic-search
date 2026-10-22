"""The page in a real browser, against the real server.

Nothing here reaches GitHub. Where a test needs GitHub's list, the server's
GitHub client answers from the test; where it doesn't, no token is set and
GitHub's list shows the sentence for that, which is also the proof that one
list failing leaves the other working.
"""

import re
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from playwright.sync_api import Locator, Page, ViewportSize, expect, sync_playwright

from gh_issue.index.store import Store
from gh_issue.interface.web.app import make_app
from tests.interface.web.conftest import GitHubAnswers, github_item

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
            page = browser.new_page()
            # The page asks github.com for the picture of a comment's author.
            page.route("https://github.com/**", lambda route: route.abort())
            yield page
        finally:
            browser.close()


def one_issue_from_github(request: httpx.Request) -> httpx.Response:
    """GitHub finds issue 7, which is not the one with the answer, and no other."""
    found = [github_item(7, "Login fails after the token expires", state="closed")]
    return httpx.Response(200, json={"total_count": 1, "search_type": "hybrid", "items": found})


def ask(page: Page, question: str, repo: str = "owner/name") -> None:
    page.get_by_role("combobox", name="Repo").select_option(repo)
    box = page.get_by_role("textbox", name="Question")
    box.fill(question)
    box.press("Enter")


def github_list(page: Page) -> Locator:
    return page.get_by_role("region", name="Original")


def our_list(page: Page) -> Locator:
    return page.get_by_role("region", name="Ours")


def rows(region: Locator) -> Locator:
    return region.get_by_role("list", name="Results").get_by_role("listitem")


def tags(region: Locator) -> Locator:
    """The tags that say a result is not in GitHub's list, where one is shown.
    Each of our rows keeps a place for its tag, so that the row does not move
    when GitHub's answer arrives."""
    return region.get_by_text("Not in GitHub").filter(visible=True)


def middle(locator: Locator) -> tuple[float, float]:
    box = locator.bounding_box()
    assert box is not None
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


# Before a question ------------------------------------------------------------------


def test_the_repos_to_choose_from_are_the_ones_in_the_index(served: str, page: Page) -> None:
    page.goto(served)

    repo = page.get_by_role("combobox", name="Repo")
    expect(repo.get_by_role("option")).to_have_text(["other/repo", "owner/name"])
    expect(repo).to_have_value("other/repo")


def test_the_page_opens_on_the_logo_and_the_question_in_the_middle(served: str, page: Page) -> None:
    """Two empty lists said nothing, so they wait for a question."""
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(served)

    expect(page).to_have_title("Issue Search")
    question = page.get_by_role("textbox", name="Question")
    expect(question).to_be_focused()
    expect(github_list(page)).to_be_hidden()
    expect(our_list(page)).to_be_hidden()
    expect(page.get_by_role("paragraph").filter(visible=True)).to_have_count(0)

    # The box is one thing: the question, and under it the repo and the button.
    box = page.get_by_role("search")
    expect(box.get_by_role("combobox", name="Repo")).to_be_visible()
    expect(box.get_by_role("button", name="Search")).to_be_visible()
    across, down = middle(box)
    assert abs(across - 640) <= 16
    assert abs(down - 360) <= 16

    # The logo is over it, in the middle too.
    logo = page.get_by_role("heading", name="Issue Search", level=1)
    expect(logo).to_be_visible()
    at = page.evaluate(
        """() => {
            const logo = document.querySelector("h1").getBoundingClientRect();
            const box = document.querySelector("[role=search]").getBoundingClientRect();
            return { middle: logo.left + logo.width / 2, gap: box.top - logo.bottom };
        }"""
    )
    assert abs(at["middle"] - 640) <= 16
    assert 0 < at["gap"] <= 60


def test_the_lists_appear_once_a_question_is_asked(served: str, page: Page) -> None:
    """Above the question, which moves to the bottom of the window and stays
    there, as the box does where people type to a chat."""
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(github_list(page)).to_be_visible()
    expect(our_list(page)).to_be_visible()
    expect(rows(our_list(page)).first).to_contain_text("#412")
    page.wait_for_function(
        """() => {
            const box = document.querySelector("[role=search]").getBoundingClientRect();
            const lists = document.querySelector("#lists").getBoundingClientRect();
            return innerHeight - box.bottom <= 24 && lists.bottom <= box.top;
        }""",
        timeout=5000,
    )


def test_a_click_on_search_asks_as_enter_does(served: str, page: Page) -> None:
    page.goto(served)
    page.get_by_role("combobox", name="Repo").select_option("owner/name")
    page.get_by_role("textbox", name="Question").fill(MIGRATIONS)

    page.get_by_role("button", name="Search").click()

    expect(rows(our_list(page)).first).to_contain_text("#412")


def test_on_a_narrow_window_the_question_stays_in_reach_as_the_page_scrolls(
    served: str, page: Page
) -> None:
    """The lists are one above the other there and the page scrolls, so the
    question holds to the bottom of the window."""
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(served)
    ask(page, MIGRATIONS)
    expect(rows(our_list(page))).to_have_count(3)
    assert page.evaluate("document.documentElement.scrollHeight > innerHeight")

    rows(our_list(page)).last.scroll_into_view_if_needed()

    assert page.evaluate("scrollY") > 0
    expect(page.get_by_role("textbox", name="Question")).to_be_in_viewport(ratio=1)
    expect(page.get_by_role("button", name="Search")).to_be_in_viewport(ratio=1)
    # The bar it is in covers the window from side to side. Seen: it stopped
    # short of the right edge, and the list showed beside it.
    bar = page.evaluate(
        """() => {
            const bar = document.querySelector(".asking").getBoundingClientRect();
            return [Math.round(bar.left), Math.round(bar.right)];
        }"""
    )
    assert bar == [0, 375]


def test_the_logo_goes_to_the_top_left_once_a_question_is_asked(served: str, page: Page) -> None:
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(rows(our_list(page)).first).to_contain_text("#412")
    page.wait_for_function(
        """() => {
            const logo = document.querySelector("h1").getBoundingClientRect();
            const lists = document.querySelector("#lists").getBoundingClientRect();
            const level = Math.abs(logo.left - lists.left) <= 2;
            return logo.top <= 40 && level && logo.bottom <= lists.top;
        }""",
        timeout=5000,
    )


def test_each_list_says_what_it_searches(served: str, page: Page) -> None:
    """The same three things are listed on both sides, each with a tick or a
    cross, and one differs: that is the comparison, readable without sound.
    They are a list of facts and must not look like something to press."""
    page.goto(served)
    ask(page, MIGRATIONS)

    theirs = github_list(page).get_by_role("list", name="What it searches")
    expect(theirs.get_by_role("listitem")).to_have_text(["Titles", "Bodies", "not Comments"])
    expect(theirs.get_by_role("listitem").last).to_have_css("text-decoration-line", "line-through")
    ours = our_list(page).get_by_role("list", name="What it searches")
    expect(ours.get_by_role("listitem")).to_have_text(["Titles", "Bodies", "Comments"])
    expect(ours.get_by_role("listitem").last).to_have_css("text-decoration-line", "none")
    for item in (theirs.get_by_role("listitem").first, ours.get_by_role("listitem").last):
        expect(item).to_have_css("border-top-width", "0px")
        expect(item).to_have_css("background-color", "rgba(0, 0, 0, 0)")


def test_the_page_does_not_name_the_tool(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """While nothing is wrong, that is. A message about something to fix still
    names the command to type."""
    github_answers(one_issue_from_github)
    page.goto(served)
    expect(page.get_by_role("main")).not_to_contain_text("gh-issue")

    ask(page, MIGRATIONS)

    expect(rows(our_list(page)).first).to_be_visible()
    expect(rows(github_list(page)).first).to_be_visible()
    expect(page.get_by_role("main")).not_to_contain_text("gh-issue")


# Our list ---------------------------------------------------------------------------


def test_a_question_shows_the_comment_that_answers_it_first(served: str, page: Page) -> None:
    page.goto(served)

    ask(page, MIGRATIONS)

    first = rows(our_list(page)).first
    expect(first).to_contain_text("App freezes when I scroll fast")
    expect(first).to_contain_text("#412")
    expect(first).to_contain_text("maintainer commented on Jan 2, 2026")
    expect(first).to_contain_text("cannot find the sqlite migrations folder")
    expect(first.get_by_role("link")).to_have_attribute(
        "href", "https://github.com/owner/name/issues/412#issuecomment-9981"
    )

    expect(github_list(page)).to_contain_text("github.token is not set")
    expect(rows(github_list(page))).to_have_count(0)
    # With no list from GitHub there is nothing to compare ours with.
    expect(tags(our_list(page))).to_have_count(0)
    # The author's picture could not be fetched, and leaves no broken image.
    expect(first.locator("img")).to_have_count(0)


def test_a_comment_that_matched_is_shown_whole(served: str, page: Page) -> None:
    """Its last words are often the answer. Seen on inngest/inngest: three
    lines of a comment were shown, the error message it was found by was on
    the fourth, and half of that line showed through under the third.

    The window is narrow here so that the test comment takes more than three
    lines. The server has already cut it to a snippet's length.
    """
    page.set_viewport_size({"width": 320, "height": 800})
    page.goto(served)
    ask(page, MIGRATIONS)

    comment = rows(our_list(page)).first
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
    ask(page, MIGRATIONS)
    first = rows(our_list(page)).first
    expect(first).to_contain_text("#412")

    page.get_by_role("combobox", name="Repo").select_option("other/repo")

    expect(first).to_contain_text("#99")
    expect(first).to_contain_text("Server does not start: the sqlite migrations folder is missing")
    expect(first.get_by_role("link")).to_have_attribute(
        "href", "https://github.com/other/repo/issues/99"
    )


def test_the_answering_comment_is_drawn_as_a_comment(served: str, page: Page) -> None:
    """With its author's picture and the words GitHub puts above a comment, so
    that it reads as a comment at a glance."""
    dot = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
    )
    page.route(
        "https://github.com/maintainer.png**",
        lambda route: route.fulfill(status=200, content_type="image/png", body=dot),
    )
    page.goto(served)

    ask(page, MIGRATIONS)

    first = rows(our_list(page)).first
    expect(first.locator("img")).to_have_attribute(
        "src", "https://github.com/maintainer.png?size=64"
    )
    expect(first.locator("img")).to_be_visible()
    expect(first).to_contain_text("maintainer commented on Jan 2, 2026")


def test_the_words_of_the_question_are_marked_in_the_post(served: str, page: Page) -> None:
    """They show where to look: in a long comment, the line that matched."""
    page.goto(served)

    ask(page, MIGRATIONS)

    first = rows(our_list(page)).first
    expect(first.locator("mark")).to_have_text(["migrations", "server", "starts"])


def test_code_in_a_post_is_shown_as_code(served: str, page: Page) -> None:
    """An error message is usually in backticks, and it is usually the answer."""
    page.goto(served)

    ask(page, "the webhook signature is rejected after rotating the key", repo="other/repo")

    first = rows(our_list(page)).first
    expect(first).to_contain_text("#100")
    expect(first.locator("code")).to_have_text(
        ["InvalidSignature", "curl -X POST localhost:8288/hook"]
    )
    expect(first).not_to_contain_text("`")


def test_only_the_first_result_shows_its_post_at_length(served: str, page: Page) -> None:
    """One answer stands out. The rest are a line each, so the list stays short
    enough to see next to GitHub's."""
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(rows(our_list(page)).locator(".rank")).to_have_text(["1", "2", "3"])
    later = rows(our_list(page)).filter(has_text="Invoices show the wrong currency")
    post = later.get_by_text("The invoice total is shown in dollars")
    expect(post).to_be_visible()
    assert post.evaluate(
        "node => node.getBoundingClientRect().height"
        " <= parseFloat(getComputedStyle(node).lineHeight) + 1"
    )


def test_a_row_is_laid_out_as_github_lays_out_an_issue(served: str, page: Page) -> None:
    """The title, and under it the number and who opened it. The state icon
    stays on the title's first line however long the title is."""
    page.set_viewport_size({"width": 320, "height": 800})
    page.goto(served)

    ask(page, MIGRATIONS)

    row = rows(our_list(page)).filter(has_text="Login fails after the token expires")
    expect(row).to_contain_text("reporter opened on Jan 1, 2026")
    expect(row.locator(".title")).to_have_css("font-size", "17px")
    expect(row.locator(".details")).to_have_css("font-size", "14px")
    # Measured in one go: the page is still moving up as the lists arrive.
    at = row.evaluate(
        """item => {
            const box = (selector) => item.querySelector(selector).getBoundingClientRect();
            const title = box(".title");
            const icon = box(".icon");
            return {
                line: parseFloat(getComputedStyle(item.querySelector(".title")).lineHeight),
                titleTop: title.top,
                titleHeight: title.height,
                numberTop: box(".number").top,
                iconMiddle: icon.top + icon.height / 2,
            };
        }"""
    )
    assert at["titleHeight"] > at["line"] * 1.5, "the window is not narrow enough to wrap the title"
    assert at["numberTop"] >= at["titleTop"] + at["titleHeight"] - 1
    assert abs(at["iconMiddle"] - (at["titleTop"] + at["line"] / 2)) <= 3


# GitHub's list, and ours against it -----------------------------------------------


def test_githubs_list_shows_what_githubs_own_list_shows(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    github_answers(one_issue_from_github)
    page.goto(served)

    ask(page, MIGRATIONS)

    theirs = github_list(page)
    row = rows(theirs).first
    expect(row).to_contain_text("Login fails after the token expires")
    expect(row).to_contain_text("#7")
    expect(row).to_contain_text("someone opened on Mar 3, 2026")
    expect(row).to_contain_text("4 comments")
    expect(row.get_by_role("link")).to_have_attribute(
        "href", "https://github.com/owner/name/issues/7"
    )
    # Hybrid is the mode that was asked for, and the page does not label it.
    expect(theirs).not_to_contain_text("mode")
    # The box ends where its one result does.
    spare = theirs.evaluate(
        """section => section.getBoundingClientRect().height
            - section.querySelector("header").getBoundingClientRect().height
            - section.querySelector(".results").getBoundingClientRect().height"""
    )
    assert spare <= 4


def test_githubs_list_says_so_when_github_ran_another_mode(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """The list is headed "Semantic search reads". If GitHub fell back to
    matching words, that heading would be wrong and nothing would say so."""
    found = [github_item(7, "Login fails after the token expires", state="closed")]
    github_answers(
        lambda request: httpx.Response(
            200, json={"total_count": 1, "search_type": "lexical", "items": found}
        )
    )
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(rows(github_list(page)).first).to_contain_text("#7")
    expect(github_list(page)).to_contain_text("lexical mode")


def test_a_long_list_scrolls_inside_its_box_and_the_page_stays_still(
    served: str, page: Page
) -> None:
    """Both lists stay side by side, under the question, however long one is."""
    page.set_viewport_size({"width": 1280, "height": 420})
    page.goto(served)

    ask(page, MIGRATIONS)

    ours = our_list(page)
    expect(rows(ours)).to_have_count(3)
    results = ours.get_by_role("list", name="Results")
    expect(results).to_have_css("overflow-y", "auto")
    assert results.evaluate("list => list.scrollHeight > list.clientHeight"), (
        "the window is not short enough for the list to overflow"
    )
    assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight")

    rows(ours).last.scroll_into_view_if_needed()

    expect(rows(ours).last).to_be_in_viewport()
    assert results.evaluate("list => list.scrollTop > 0")
    assert page.evaluate("window.scrollY") == 0
    # The list's own heading stays where it was.
    expect(ours.get_by_role("heading", name="Ours")).to_be_in_viewport()


def test_a_result_github_did_not_return_says_so(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """This is the point of the page, so it is not left for the reader to work
    out by comparing two lists. GitHub's whole answer here is issue 7."""
    github_answers(one_issue_from_github)
    page.goto(served)

    ask(page, MIGRATIONS)

    ours = rows(our_list(page))
    expect(ours.filter(has_text="#412")).to_contain_text("Not in GitHub's results")
    expect(ours.filter(has_text="Invoices show the wrong currency")).to_contain_text(
        "Not in GitHub's results"
    )
    in_both = ours.filter(has_text="Login fails after the token expires")
    expect(in_both).to_be_visible()
    expect(tags(in_both)).to_have_count(0)


def test_when_github_has_more_results_than_are_shown_it_says_top(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """Issue 412 may be further down GitHub's list, and the page can't know."""
    shown = [
        github_item(number, f"Another issue {number}") for number in (7, 1001, 1002, 1003, 1004)
    ]
    github_answers(
        lambda request: httpx.Response(
            200, json={"total_count": 23, "search_type": "hybrid", "items": shown}
        )
    )
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(rows(our_list(page)).filter(has_text="#412")).to_contain_text("Not in GitHub's top 5")


def test_a_list_that_is_slow_shows_rows_to_come(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """And the other list does not wait for it."""
    let_go = threading.Event()

    def held(request: httpx.Request) -> httpx.Response:
        let_go.wait(timeout=20)
        return one_issue_from_github(request)

    github_answers(held)
    page.goto(served)

    ask(page, MIGRATIONS)

    theirs = github_list(page)
    expect(theirs.locator(".placeholder")).to_have_count(3)
    expect(rows(our_list(page)).first).to_contain_text("#412")
    expect(rows(theirs)).to_have_count(0)
    let_go.set()
    expect(rows(theirs).first).to_contain_text("#7")
    expect(theirs.locator(".placeholder")).to_have_count(0)


def test_a_quick_answer_is_shown_without_rows_to_come_first(served: str, page: Page) -> None:
    """Our list answers in about 50 ms. Grey rows for that long are a flash."""
    page.goto(served)
    page.evaluate(
        """() => {
            window.flashed = 0;
            new MutationObserver(() => {
                if (document.querySelector("#ours .placeholder")) window.flashed += 1;
            }).observe(document.querySelector("#ours .results"), { childList: true });
        }"""
    )

    ask(page, MIGRATIONS)

    expect(rows(our_list(page)).first).to_contain_text("#412")
    assert page.evaluate("window.flashed") == 0


@pytest.mark.parametrize(
    "window",
    [
        pytest.param({"width": 900, "height": 900}, id="side-by-side"),
        pytest.param({"width": 375, "height": 812}, id="one-above-the-other"),
    ],
)
def test_our_rows_stay_where_they_are_when_githubs_answer_arrives(
    served: str, page: Page, github_answers: GitHubAnswers, window: ViewportSize
) -> None:
    """GitHub answers about a second after we do, and the tags come with it.
    Seen: each tagged row grew, by up to 24 px where the tag took a new line,
    and everything under it moved while it was being read."""
    let_go = threading.Event()

    def held(request: httpx.Request) -> httpx.Response:
        let_go.wait(timeout=20)
        return one_issue_from_github(request)

    github_answers(held)
    page.set_viewport_size(window)
    page.goto(served)
    ask(page, MIGRATIONS)
    ours = our_list(page)
    expect(rows(ours)).to_have_count(3)
    page.wait_for_function("() => document.getAnimations().length === 0")
    heights = (
        "() => [...document.querySelectorAll('#ours .result')]"
        ".map(row => row.getBoundingClientRect().height)"
    )
    before = page.evaluate(heights)

    let_go.set()

    expect(tags(ours)).to_have_count(2)
    assert page.evaluate(heights) == before


def test_a_row_that_github_has_too_shows_no_gap_where_a_tag_would_be(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """Seen on a narrow window: the tag's place was kept as an empty line
    between who opened the issue and its first words."""
    github_answers(one_issue_from_github)
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(tags(our_list(page))).to_have_count(2)
    in_both = rows(our_list(page)).filter(has_text="Login fails after the token expires")
    gap = in_both.evaluate(
        """row => row.querySelector(".post").getBoundingClientRect().top
            - row.querySelector(".who").getBoundingClientRect().bottom"""
    )
    assert gap <= 8


def test_an_answer_to_an_earlier_question_is_dropped(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """Ask, then ask again before GitHub has answered the first: the first
    answer arrives last, and must not replace the second."""
    let_go = threading.Event()

    def answer(request: httpx.Request) -> httpx.Response:
        if "first question" in request.url.params["q"]:
            let_go.wait(timeout=20)
            return httpx.Response(
                200,
                json={"total_count": 1, "items": [github_item(1001, "For the first question")]},
            )
        return httpx.Response(
            200, json={"total_count": 1, "items": [github_item(1002, "For the second question")]}
        )

    github_answers(answer)
    page.goto(served)
    theirs = github_list(page)

    with page.expect_response(lambda response: "first+question" in response.url) as late:
        ask(page, "the first question")
        ask(page, "the second question")
        expect(rows(theirs).first).to_contain_text("#1002")
        let_go.set()
    late.value.finished()
    # One more trip to the page, so that it has had the late answer in hand.
    page.evaluate("() => new Promise(requestAnimationFrame)")

    expect(rows(theirs)).to_have_text([re.compile("For the second question")])


def test_markup_in_a_title_is_shown_as_the_characters_it_is(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """Titles, logins and posts are other people's text."""
    title = '<img src=x onerror="window.broken = true"><b>Urgent</b> & more'
    github_answers(
        lambda request: httpx.Response(
            200, json={"total_count": 1, "items": [github_item(7, title)]}
        )
    )
    page.goto(served)

    ask(page, MIGRATIONS)

    row = rows(github_list(page)).first
    expect(row).to_contain_text(title)
    expect(row.locator("img, b")).to_have_count(0)
    assert page.evaluate("window.broken") is None


def test_the_first_result_shows_up_to_three_lines_of_an_issue_body(served: str, page: Page) -> None:
    page.goto(served)

    ask(page, "the invoice shows dollars, not euros")

    first = rows(our_list(page)).first
    expect(first).to_contain_text("#55")
    lines = first.get_by_text("The invoice total is shown in dollars").evaluate(
        "node => node.getBoundingClientRect().height"
        " / parseFloat(getComputedStyle(node).lineHeight)"
    )
    assert 2.5 < lines <= 3.1


def test_marked_words_are_dark_enough_to_read_in_a_grey_line(served: str, page: Page) -> None:
    """A line of a later result is grey, and grey on the mark's yellow was
    3.86 to 1."""
    page.goto(served)

    ask(page, "server fails to start and the webhook signature is wrong", repo="other/repo")

    later = rows(our_list(page)).nth(1).locator("mark").first
    expect(later).to_have_css("color", "rgb(31, 35, 40)")


def test_the_mode_is_not_left_showing_when_the_next_search_fails(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if "first question" in request.url.params["q"]:
            return httpx.Response(
                200, json={"total_count": 0, "search_type": "lexical", "items": []}
            )
        return httpx.Response(500)

    github_answers(answer)
    page.goto(served)
    theirs = github_list(page)
    ask(page, "the first question")
    expect(theirs.get_by_text("lexical mode")).to_be_visible()

    ask(page, "the second question")

    expect(theirs).to_contain_text("GitHub's search failed (HTTP 500).")
    expect(theirs.get_by_text("lexical mode")).to_have_count(0)


def test_when_githubs_limit_is_spent_the_page_waits_then_asks_again(
    served: str, page: Page, github_answers: GitHubAnswers
) -> None:
    """A search made too soon answers itself, which saves a take when recording."""
    asked = 0

    def limited_once(request: httpx.Request) -> httpx.Response:
        nonlocal asked
        asked += 1
        if asked == 1:
            headers = {
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(int(time.time()) + 2),
            }
            return httpx.Response(403, headers=headers)
        return one_issue_from_github(request)

    made = github_answers(limited_once)
    page.goto(served)
    # The sentence is on the page for a second or two. The page notes down what
    # the list said, so that the test does not have to look at the right moment.
    page.evaluate(
        """() => {
            window.said = [];
            const status = document.querySelector("#github .message");
            new MutationObserver(() => window.said.push(status.textContent)).observe(status, {
                childList: true,
                subtree: true,
                characterData: true,
            });
        }"""
    )

    ask(page, MIGRATIONS)

    theirs = github_list(page)
    expect(rows(theirs).first).to_contain_text("#7", timeout=10000)
    waiting = re.compile(
        r"GitHub allows 10 of these searches a minute\. Trying again in \d seconds?\."
    )
    said: list[str] = page.evaluate("window.said")
    assert any(waiting.fullmatch(text) for text in said), said
    assert len(made) == 2


# What the page keeps -----------------------------------------------------------------


def test_a_search_is_kept_in_the_address(served: str, page: Page) -> None:
    """So a reload, a bookmark, or a tab opened ahead of a recording has it."""
    page.goto(served)

    ask(page, MIGRATIONS)

    expect(page).to_have_url(re.compile(r"/\?repo=owner%2Fname&q=after\+upgrading\+the\+server\+"))


def test_an_address_with_a_search_opens_on_its_results(served: str, page: Page) -> None:
    page.goto(f"{served}/?repo=owner%2Fname&q=" + MIGRATIONS.replace(" ", "+"))

    expect(page.get_by_role("combobox", name="Repo")).to_have_value("owner/name")
    expect(page.get_by_role("textbox", name="Question")).to_have_value(MIGRATIONS)
    expect(rows(our_list(page)).first).to_contain_text("#412")


def test_the_repo_chosen_last_is_chosen_on_the_next_visit(served: str, page: Page) -> None:
    """The first repo in the alphabet is rarely the one being shown."""
    page.goto(served)
    repo = page.get_by_role("combobox", name="Repo")
    expect(repo).to_have_value("other/repo")
    repo.select_option("owner/name")

    page.goto(served)

    expect(repo).to_have_value("owner/name")


def test_a_long_question_takes_up_to_three_lines_then_scrolls(served: str, page: Page) -> None:
    """Seen with the box one line tall: only the question's last word,
    "missing", was on show."""
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(served)
    box = page.get_by_role("textbox", name="Question")

    box.fill("The server fails to start")
    one = box.evaluate("node => node.clientHeight")
    box.fill(MIGRATIONS + " " + MIGRATIONS)

    at = box.evaluate(
        """node => ({
            line: parseFloat(getComputedStyle(node).lineHeight),
            shown: node.clientHeight,
            needed: node.scrollHeight,
            scrolls: getComputedStyle(node).overflowY,
        })"""
    )
    assert one == at["line"]
    assert at["shown"] == 3 * at["line"]
    assert at["needed"] > at["shown"]
    assert at["scrolls"] == "auto"


def test_an_address_with_a_search_opens_with_the_question_at_the_bottom(
    served: str, page: Page
) -> None:
    """Not in the middle and then sliding away, which is how it looked."""
    page.add_init_script(
        """document.addEventListener("DOMContentLoaded", () => {
            const box = document.querySelector("[role=search]").getBoundingClientRect();
            window.firstGap = innerHeight - box.bottom;
        });"""
    )
    page.set_viewport_size({"width": 1280, "height": 720})

    page.goto(f"{served}/?repo=owner%2Fname&q=" + MIGRATIONS.replace(" ", "+"))

    expect(rows(our_list(page)).first).to_contain_text("#412")
    assert page.evaluate("window.firstGap") <= 24


def test_a_question_of_many_lines_leaves_room_for_the_lists(served: str, page: Page) -> None:
    """Seen without a limit: a twelve-line question left the lists no height."""
    page.set_viewport_size({"width": 760, "height": 420})
    page.goto(served)

    ask(page, " ".join([MIGRATIONS] * 6))

    expect(rows(our_list(page)).first).to_be_visible()
    # The box is still sliding down to the bottom when the first row shows, and
    # the lists only have their room once it is there. Seen: 29 px, measured
    # before the slide had begun, against 168 px after it.
    page.wait_for_function("() => document.getAnimations().length === 0")
    box = page.get_by_role("textbox", name="Question")
    assert box.evaluate("node => node.clientHeight") <= 3 * 28
    assert page.locator("#lists").evaluate("node => node.getBoundingClientRect().height") > 100


def test_the_box_fits_the_question_again_when_the_window_narrows(served: str, page: Page) -> None:
    """For a browser that cannot size the box to its text by itself, which is
    what this one is made into here."""
    page.add_init_script(
        """const supports = CSS.supports.bind(CSS);
        CSS.supports = (...asked) => asked[0] !== "field-sizing" && supports(...asked);"""
    )
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(served)
    page.add_style_tag(content="textarea { field-sizing: fixed !important; }")
    box = page.get_by_role("textbox", name="Question")
    box.fill(MIGRATIONS)
    assert box.evaluate("node => node.clientHeight") == 28

    page.set_viewport_size({"width": 375, "height": 812})

    page.wait_for_function(
        "() => document.querySelector('#question').clientHeight === 3 * 28", timeout=5000
    )
