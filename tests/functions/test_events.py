from gh_issue.functions import events
from gh_issue.functions.client import client


def test_the_three_event_names_are_distinct() -> None:
    """A copy-paste here is not a typo, it is an infinite loop: the backfill
    sends itself its own trigger, so it must not share a name with anything."""
    names = {
        events.EVENT_ISSUE_CHANGED,
        events.EVENT_BACKFILL_PAGE,
        events.EVENT_SWEEP_REQUESTED,
    }
    assert len(names) == 3


def test_the_inngest_app_is_named() -> None:
    assert client.app_id == "gh-issue"
