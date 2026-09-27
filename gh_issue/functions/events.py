"""The names of the events the functions send and are triggered by.

They live here rather than next to the functions that send them, because three
different triggers emit `repo/issue.changed` and only one function consumes it.
Keeping the strings in one place is what makes that a single indexing path
instead of three that happen to agree.
"""

# The backfill, the webhook transform and the hourly sweep all emit this one.
EVENT_ISSUE_CHANGED = "repo/issue.changed"
# Sent by the backfill to itself, carrying the cursor for the next page.
EVENT_BACKFILL_PAGE = "repo/backfill.page"
# Fanned out by the cron so each repo sweeps in its own run.
EVENT_SWEEP_REQUESTED = "repo/sweep.requested"
