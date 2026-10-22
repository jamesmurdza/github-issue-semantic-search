"""What the page shows of a post: its words, with the Markdown markers taken
out and code left in one pair of backticks.

Each case here is something a real issue or comment does. The ones marked
"was" are what an earlier version made of them.
"""

import time

import pytest

from gh_issue.interface.web.plain import excerpt


def shown(text: str) -> str:
    return excerpt(text, 280)


@pytest.mark.parametrize(
    ("post", "expected"),
    [
        pytest.param(
            "I ran ```npm install``` and got an error.\nThen I tried again.",
            "I ran `npm install` and got an error. Then I tried again.",
            # was: "I ran `Then I tried again.`"
            id="three-backticks-inside-a-line-are-a-span-not-a-block",
        ),
        pytest.param(
            "~~~\nmake build\n~~~\nDone.", "`make build` Done.", id="a-block-fenced-with-tildes"
        ),
        pytest.param(
            "> ```\n> make build\n> ```\nDone.",
            "`make build` Done.",
            id="a-quoted-block-loses-its-quote-arrows",
        ),
        pytest.param(
            "1. Run:\n\n      ```\n      make build\n      ```\n\n"
            "**Expected behavior**\nIt builds.",
            "1. Run: `make build` Expected behavior: It builds.",
            # was: the closing fence was read as an opening one, and everything
            # after it as code. Seen on inngest/inngest.
            id="a-block-indented-under-a-list-item",
        ),
        pytest.param(
            "I get: ```\nKilled: 9\n```\n**To Reproduce**\nInstall it.",
            "I get: `Killed: 9` To Reproduce: Install it.",
            id="a-block-opened-at-the-end-of-a-line",
        ),
        pytest.param(
            "Set KEY=<your-key>, call Promise<void> and get <class 'ValueError'>.",
            "Set KEY=<your-key>, call Promise<void> and get <class 'ValueError'>.",
            # was: "Set KEY=, call Promise and get ."
            id="angle-brackets-that-are-not-html-stay",
        ),
        pytest.param(
            "a<br>b <details><summary>Logs</summary>text</details>",
            "a b Logs text",
            id="html-tags-leave-a-space-not-a-join",
        ),
        pytest.param(
            "See <https://example.com/x>.",
            "See https://example.com/x.",
            id="a-link-in-angle-brackets-keeps-its-address",
        ),
        pytest.param(
            "Before ![image](https://example.com/y.png) after",
            "Before after",
            id="an-image-is-left-out-with-its-alt-text",
        ),
        pytest.param(
            "I got this error:\r\n```txt\r\nScan error on column 5\r\n```\r\n\r\n"
            "**Expected behavior**\r\nIt works.",
            "I got this error: `Scan error on column 5` Expected behavior: It works.",
            # was: "... `Scan error on column 5 ''' **Expected behavior** It works.`"
            # Posts written on github.com arrive with these line endings.
            id="windows-line-endings",
        ),
        pytest.param(
            "See [the docs](https://example.com/docs) and [setup](#setup).",
            "See the docs and setup.",
            id="links-keep-their-words",
        ),
        pytest.param(
            "Read rows[0][1], then call handlers[0](event).",
            "Read rows[0][1], then call handlers[0](event).",
            # was: "Read rows[0][1], then call handlers0."
            id="square-brackets-that-are-not-links-stay",
        ),
        pytest.param(
            "Footnote.\n\n[1]: https://example.com",
            "Footnote.",
            id="a-link-target-on-its-own-line",
        ),
        pytest.param(
            "- [x] I searched\n- [ ] I read the docs",
            "- I searched - I read the docs",
            id="a-checklist-without-its-boxes",
        ),
        pytest.param(
            "| Name | Value |\n|---|---|\n| cli | 1.4.0 |",
            "Name Value cli 1.4.0",
            id="a-table-without-its-lines",
        ),
        pytest.param("~~old~~ new", "old new", id="struck-through"),
        pytest.param(
            "Title\n=====\ntext\n\n---\nmore",
            "Title text more",
            id="an-underlined-heading-and-a-rule",
        ),
        pytest.param("a &lt;b&gt; &amp; c", "a <b> & c", id="html-entities"),
        pytest.param(
            "\\*not bold\\* and 2 \\* 3", "*not bold* and 2 * 3", id="escaped-markers-stay"
        ),
        pytest.param(
            "a **statically\nlinked** binary",
            "a statically linked binary",
            id="bold-across-a-line-break",
        ),
        pytest.param(
            "set workflow_version=0 and snake_case_name",
            "set workflow_version=0 and snake_case_name",
            id="underscores-inside-names-stay",
        ),
        pytest.param(
            "`**kwargs` and `<port>` and `# not a heading`",
            "`**kwargs` and `<port>` and `# not a heading`",
            id="code-is-left-as-written",
        ),
        pytest.param("a \x007\x00 b", "a 7 b", id="a-nul-in-a-post"),
    ],
)
def test_a_post_is_shown_as_its_words(post: str, expected: str) -> None:
    assert shown(post) == expected


def test_a_cut_just_inside_code_does_not_leave_an_empty_piece_of_code() -> None:
    """The page shows what follows a backtick as code, and "..." alone in a
    code box reads as a glitch."""
    cut = shown("x" * 275 + " `code here` and more words after it")

    assert cut.endswith("x...")
    assert "`" not in cut


@pytest.mark.parametrize(
    "post",
    [
        pytest.param("# a " + "# " * 32000, id="hashes"),  # was 17 s
        pytest.param("[" * 65000, id="brackets"),  # was 15 s
        pytest.param("<a" * 32000, id="tags"),  # was 5 s
        pytest.param("`" * 65000, id="backticks"),  # was 3 s
    ],
)
def test_a_long_post_made_to_be_slow_is_still_quick(post: str) -> None:
    """A post can be 65,536 characters and the excerpt needs 280 of them. A
    search waits for five of these."""
    started = time.perf_counter()

    shown(post)

    assert time.perf_counter() - started < 0.5
