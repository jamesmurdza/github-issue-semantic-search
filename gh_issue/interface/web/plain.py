"""A post's Markdown as plain words, for the page.

The page puts other people's text on screen as text, never as HTML, so a
post's Markdown would show as its markers: hashes, stars, brackets, arrows.
This takes the markers out and leaves the words.

Code is the exception. It keeps one pair of backticks around it, a fenced
block too, which is how the page knows to show it as code. No other backtick
is left in the text.

It is a reading aid for a few lines of excerpt, not a Markdown parser: a list
keeps its dashes, and a table becomes its cells in a row.
"""

import html
import re

from gh_issue.utils import snippet

# No excerpt needs more of a post than its start, and this bounds the work.
# Some patterns below take time that grows with the square of a line's length:
# measured on posts of 65,000 characters made for it, 17 s for one of hashes
# and 15 s for one of brackets. Ordinary posts of that size took 8 ms.
READ = 4000

# Code: a fenced block that starts a line, indented or quoted or not, or a
# span between two runs of backticks, which may cross a line break but not a
# blank line. They are found in one pass from the start of the post, so that a
# block opened at the end of a line of text is taken whole as a span, and its
# closing fence is not then read as the opening of another block.
_CODE = re.compile(
    r"^(?P<lead>[ \t]*(?:>[ \t]?)*)(?P<fence>```|~~~)[^\n]*\n"
    r"(?P<block>.*?)"
    r"(?:^[ \t]*(?:>[ \t]?)*(?P=fence)[ \t]*$|\Z)"
    r"|(?<!`)(?P<ticks>`+)(?!`)(?P<span>(?:[^\n]|\n(?!\s*\n))+?)(?<!`)(?P=ticks)(?!`)",
    re.MULTILINE | re.DOTALL,
)
_KEPT = re.compile("\x00(\\d+)\x00")

_ESCAPED = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")
_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_AUTOLINK = re.compile(r"<((?:https?://|mailto:)[^>\s]+)>")
# Only what is HTML is taken out: `Promise<void>` and `<your-key>` are a
# writer's own angle brackets, and they stay.
_HTML_TAG = re.compile(
    r"</?(?:a|b|blockquote|br|center|code|del|details|div|em|h[1-6]|hr|i|img|ins|kbd|li|ol|p"
    r"|picture|pre|s|small|source|span|strong|sub|summary|sup|table|tbody|td|th|thead|tr|u|ul"
    r"|video)(?:\s[^>\n]*)?/?>",
    re.IGNORECASE,
)
_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
# A link is taken as one only when what is in the round brackets is an
# address. `rows[0][1]` and `handlers[0](event)` are a writer's own brackets.
_LINK = re.compile(
    r"\[([^\]\n]+)\]\((?:[a-z][a-z0-9+.-]*:|/|#|\.{1,2}/|www\.)[^)\n]*\)", re.IGNORECASE
)
_LINK_TARGET = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]+\S.*$", re.MULTILINE)
_TABLE_RULE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$", re.MULTILINE
)
_TABLE_ROW = re.compile(r"^[ \t]*\|(.+)\|[ \t]*$", re.MULTILINE)
# A rule, or the line under a heading that is underlined rather than hashed.
_RULE = re.compile(r"^[ \t]{0,3}(?:={2,}|-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_BOLD_LINE = re.compile(r"^[ \t]*(?:\*\*|__)([^*_\n]+)(?:\*\*|__)[ \t]*$", re.MULTILINE)
_QUOTE = re.compile(r"^[ \t]{0,3}(?:>[ \t]?)+", re.MULTILINE)
_TASK = re.compile(r"^([ \t]*[-*+][ \t]+)\[[ xX]\][ \t]+", re.MULTILINE)
# Bold may run over a line break, but not over a blank line.
_BOLD = re.compile(
    r"\*\*((?:[^*\n]|\n(?!\s*\n))+?)\*\*|(?<!\w)__((?:[^_\n]|\n(?!\s*\n))+?)__(?!\w)"
)
_EMPHASIS = re.compile(
    r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])|(?<!\w)_(?!\s)([^_\n]+?)(?<!\s)_(?!\w)"
)
_STRUCK = re.compile(r"~~([^~\n]+)~~")


def _label(match: re.Match[str]) -> str:
    """A heading, or a line that is all bold, as its words. They get a colon so
    that they don't run into the sentence after them once the lines are joined."""
    words: str = match.group(1).strip()
    return words if words.endswith((".", ":", "!", "?")) else f"{words}:"


def _words(match: re.Match[str]) -> str:
    words: str = match.group(1) or match.group(2)
    return words


def _cells(match: re.Match[str]) -> str:
    return " ".join(cell.strip() for cell in match.group(1).split("|"))


def _code(text: str) -> str:
    """Code on one line, between the pair of backticks the page looks for."""
    inner = " ".join(text.replace("`", "'").split())
    return f"`{inner}`" if inner else ""


def plain(text: str) -> str:
    """The words of the start of a post, with its code in backticks."""
    # What is set aside, to be put back untouched at the end.
    kept: list[str] = []

    def keep(value: str) -> str:
        kept.append(value)
        return f"\x00{len(kept) - 1}\x00"

    def code(match: re.Match[str]) -> str:
        if match.group("fence") is None:
            return keep(_code(match.group("span")))
        block: str = match.group("block")
        if ">" in match.group("lead"):
            block = _QUOTE.sub("", block)
        return keep(_code(block)) + "\n"

    def escaped(match: re.Match[str]) -> str:
        character: str = match.group(1)
        return keep("'" if character == "`" else character)

    # The marker for something set aside is a NUL, so the post may have none.
    # And a post written on github.com ends its lines the Windows way, which
    # the patterns that look for the end of a line would not see as one.
    text = text[:READ].replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # Code comes out first, so that nothing below reads it as Markdown, and
    # then the characters a writer escaped to stop that themselves.
    text = _CODE.sub(code, text)
    text = _ESCAPED.sub(escaped, text)

    text = _HTML_COMMENT.sub("", text)
    text = _AUTOLINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _LINK_TARGET.sub("", text)
    text = _TABLE_RULE.sub("", text)
    text = _TABLE_ROW.sub(_cells, text)
    text = _RULE.sub("", text)
    text = _HEADING.sub(_label, text)
    text = _BOLD_LINE.sub(_label, text)
    text = _QUOTE.sub("", text)
    text = _TASK.sub(r"\1", text)
    text = _BOLD.sub(_words, text)
    text = _EMPHASIS.sub(_words, text)
    text = _STRUCK.sub(r"\1", text)
    text = html.unescape(text.replace("`", ""))
    return _KEPT.sub(lambda match: kept[int(match.group(1))], text)


def excerpt(text: str, length: int) -> str:
    """`text` as the page shows it: plain, on one line, and cut to `length`."""
    cut = snippet(plain(text), length)
    # A cut that lands just inside a piece of code leaves its opening backtick
    # with nothing after it, which the page would show as a box of dots.
    if cut.endswith("`...") and cut.count("`") % 2:
        cut = cut[:-4].rstrip() + "..."
    return cut
