# Brief for the query writer

The paraphrase and specific queries in `written_queries.json` were written by a
separate Claude agent, one run per repo. It saw this brief and one input file
(50 issues from `make_queries sample`), nothing else: no search engine, no other
issues, no results. The `{…}` parts were filled in per repo; everything else is
word for word.

- inngest/inngest: *a workflow/durable-functions platform*; the example
  distinctive words were *product names, error strings, identifiers, config keys,
  version numbers*; the example common words *"error", "function", "event",
  "run", "local", "not working"*; the example details *a function / CLI flag /
  config key / API name*.
- pydantic/pydantic-ai: *a Python agent framework for LLMs*; *product names,
  error strings, identifiers, class or function names, config keys, version
  numbers*; *"error", "agent", "model", "tool", "stream", "not working"*; *a
  class / function / parameter / API name*.

---

You are writing search queries for a retrieval benchmark over GitHub issues.
Your queries will be used to test a search engine; you must not run any
searches, open any other files, or look anything up. Work only from the one
input file.

Input: `{input file}`
It is a JSON list of 50 GitHub issues from the repo {repo} ({what the repo is}).
Each has "number", "title", "body" and "first_comments".

For EACH issue, write two queries that a developer might type into a search box
over this repo's issue history when they have the problem the issue describes,
before they know the issue exists:

1. "paraphrase": the problem described in the developer's own everyday words,
   the way someone who has just hit it would describe it. Do NOT reuse the
   distinctive words of the title or the body ({distinctive words}). Common
   words like {common words} are fine. 4 to 12 words.

2. "specific": what a developer would type when they have a concrete detail in
   hand: a fragment of the exact error message, {details}, a status code or a
   version number that appears in the issue. Use the issue's own wording for
   that detail. 3 to 12 words. If the issue has no concrete detail like that,
   use its two or three most distinctive terms.

Rules:
- One query of each kind per issue; never mention the issue number.
- Do not try to make queries easy or hard; write what a real user would
  plausibly type.
- Base each query only on that issue's own text.

Write the result as a JSON list to: `{output file}`
with one object per issue, in input order:
`{"issue": <number>, "paraphrase": "...", "specific": "..."}`

Make sure the file is valid JSON with exactly 50 objects, then reply with just
"done: 50 issues" (or what went wrong).
