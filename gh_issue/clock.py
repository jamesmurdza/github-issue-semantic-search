import datetime


def now() -> str:
    """UTC to the second, in the format GitHub's `since` takes.

    Truncation rounds down, so a sweep window that starts here overlaps the
    previous one slightly rather than leaving a gap.

    Inside an Inngest function, call this only inside a step. The body replays
    once per step, so a read in plain Python happens again on every pass and
    the value that sticks is whatever the last pass saw.
    """
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
