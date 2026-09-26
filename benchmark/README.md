# Search benchmark

`gh-issue search` runs two searches at once and fuses them. This benchmark is
how that design, and its settings, were chosen: 620 queries over two real
repos, each with one right answer, run against the index `gh-issue sync` builds.

## The question

Every chunk (an issue body, or one comment) is stored with two vectors:

- **dense**: bge-small's embedding. It finds text that *means* the same thing,
  even in other words.
- **BM25**: a weight for each word stem the text contains. It finds text that
  *uses the same words*, and rare words count for more.

Search can use either alone, or both, merged by Qdrant into one list. Qdrant
merges two ways:

- **RRF** (reciprocal rank fusion) scores a chunk by its *rank* in each list,
  and ignores how far ahead of the next one it was.
- **DBSF** (distribution-based score fusion) rescales each list's *scores* by how
  spread out they are, then adds them. A chunk far ahead in one list stays far
  ahead.

So the benchmark asks: which of the four ranks the right issue highest? Then:
does counting one side more (weighted RRF) help? How many candidates should each
side bring? And what is BGE's query instruction worth?

## The index

Both repos, synced with `gh-issue sync` on 2026-09-26:

| repo | issues | chunks |
|---|---:|---:|
| inngest/inngest | 315 | 1,250 |
| pydantic/pydantic-ai | 3,367 | 14,963 |

inngest/inngest is the repo this example is written around. pydantic/pydantic-ai
is ten times larger, and its maintainers mark duplicates, which inngest's do not.

## The queries

Every query has one right answer, an issue. [`queries.json`](queries.json) holds
them all, built by [`make_queries.py`](make_queries.py).

| kind | what it is | example | right answer |
|---|---|---|---|
| duplicate | the title of an issue a maintainer marked as a duplicate. The duplicate itself is hidden. | "Support for Claude 3.7 thinking" | the issue it duplicates: "Reasoning response support" |
| title | an issue's title, with that issue's body hidden, so it can only be found through its comments | "Permission denied running npx inngest-cli@latest on Windows" | the issue |
| paraphrase | the problem in everyday words, avoiding the issue's distinctive terms | "local server caps parallel tasks at a couple dozen, how to raise" | the issue |
| specific | a concrete detail from the issue: an error message, a function name | "local concurrency limit 25 steps throttling" | the issue |

Duplicates are the closest thing here to real use: one user's words for a
problem another user already reported. Titles test finding a thread by its
discussion. Paraphrase and specific queries were written by a separate agent that
read 50 random issues per repo and nothing else; it did not see any search
results. [`writer_brief.md`](writer_brief.md) is exactly what it was told, and
[`written_queries.json`](written_queries.json) is what it wrote.

## The score

Search shows one hit per issue, so a query's **rank** is where the right issue
lands once each issue is counted once.

**MRR@10** (mean reciprocal rank) turns ranks into one number: a query scores 1
if the right issue is first, 1/2 if second, 1/3 if third, down to 1/10, and 0 if
it is not in the top ten. The mean over queries is the score. 1.0 would be
perfect.

Queries fall into seven groups: a repo and a kind (inngest has no marked duplicates).
Each score below is the mean of the groups' scores, so the 150 title queries of
a repo count as much as its 50 paraphrases.

The queries are a sample, and a different sample would score a little
differently. The last column is the gap to the shipped search, with a 95%
interval from re-drawing the queries at random 2,000 times ("paired bootstrap").
An interval that does not include 0 is a gap that luck of the sample does not
explain.

## Results

| method | MRR@10 | right issue first | in the top 10 | vs DBSF, 95% interval |
|---|---:|---:|---:|---|
| dense, without the query instruction | 0.513 | 45% | 67% | -0.127 [-0.154, -0.101] |
| dense | 0.569 | 50% | 72% | -0.071 [-0.096, -0.045] |
| BM25 | 0.578 | 52% | 70% | -0.063 [-0.085, -0.042] |
| RRF | 0.629 | 56% | 77% | -0.012 [-0.026, +0.001] |
| RRF, k=60 | 0.615 | 55% | 77% | -0.025 [-0.039, -0.011] |
| RRF, dense counts 3x | 0.616 | 54% | 78% | -0.025 [-0.041, -0.008] |
| RRF, dense counts 2x | 0.616 | 53% | 78% | -0.024 [-0.040, -0.008] |
| RRF, BM25 counts 2x | 0.620 | 55% | 77% | -0.020 [-0.033, -0.007] |
| RRF, BM25 counts 3x | 0.620 | 55% | 76% | -0.020 [-0.033, -0.007] |
| DBSF (shipped) | 0.640 | 57% | 78% |  |
| DBSF, 25 candidates a side | 0.637 | 57% | 78% | -0.004 [-0.013, +0.006] |
| DBSF, 50 candidates a side | 0.641 | 58% | 78% | +0.001 [-0.005, +0.006] |
| DBSF, 200 candidates a side | 0.642 | 58% | 77% | +0.001 [-0.002, +0.006] |

MRR@10 by group:

| repo | kind | queries | dense | BM25 | RRF | DBSF (shipped) |
|---|---|---:|---:|---:|---:|---:|
| inngest/inngest | paraphrase | 50 | 0.502 | 0.464 | 0.616 | **0.624** |
| inngest/inngest | specific | 50 | 0.919 | 0.951 | 0.948 | **0.958** |
| inngest/inngest | title | 150 | 0.735 | 0.698 | 0.729 | **0.737** |
| pydantic/pydantic-ai | duplicate | 120 | 0.351 | 0.297 | 0.367 | **0.369** |
| pydantic/pydantic-ai | paraphrase | 50 | 0.228 | 0.360 | 0.414 | **0.441** |
| pydantic/pydantic-ai | specific | 50 | 0.858 | 0.908 | 0.901 | **0.915** |
| pydantic/pydantic-ai | title | 150 | 0.390 | 0.365 | 0.425 | **0.438** |

What this says:

- **Neither search alone is enough.** Dense and BM25 score about the same
  (0.569 and 0.578), but they fail on different queries: dense is ahead on
  duplicates, titles and inngest's paraphrases, BM25 on specific queries and
  pydantic-ai's paraphrases. Fused, both ways score higher than either.
- **DBSF is ahead of RRF, but only a little.** It is first in all seven
  groups, 0.002 to 0.027 ahead of the next best, but its overall gap to RRF is
  0.012 and the interval,
  [-0.026, +0.001], just reaches zero: a consistent lead, not a proven one.
  RRF only sees ranks, so a chunk far ahead in one list counts no more than
  one barely ahead; and it ties constantly (a chunk 5th in one list and absent
  from the other scores exactly what a chunk 5th in the other list does), so
  its results depend on how ties are broken. Breaking them by point id rather
  than by issue number put RRF at 0.628, [-0.026, -0.001]. DBSF's scores tie
  only on identical text, and it scored 0.640 either way.
- **Leaning on one side does not help.** Counting dense or BM25 two or three
  times as much scored below plain RRF (0.616 to 0.620, against 0.629), and so
  did RRF's textbook constant k=60 (0.615; Qdrant's default is 2).
- **50 candidates a side is enough.** 50, 100 and 200 score the same, within
  0.002; 25 is a little lower. The shipped 100 is the middle of that range.
- **BGE's query instruction is worth 0.056 to dense search** (0.513 without it,
  0.569 with). It is the only change here that is not about fusion.

## Why dense search trails BM25

Three reasons, the first one fixed.

1. **The query instruction was missing.** BGE models are trained with
   "Represent this sentence for searching relevant passages: " in front of
   short queries, and nothing in front of stored text. fastembed does not add
   it, so every query was embedded as if it were a stored comment. Adding it
   (`gh_issue/embed.py`) took dense from 0.513 to 0.569.
2. **The model reads only the first 512 tokens** of a chunk. Long comments,
   usually pasted logs and code, are embedded by their opening alone; BM25
   counts every word. From `benchmark.chunk_lengths`:

   | repo | chunks | median tokens | longer than 512 | tokens past 512 |
   |---|---:|---:|---:|---:|
   | inngest/inngest | 1,250 | 70 | 10% | 44% |
   | pydantic/pydantic-ai | 14,963 | 142 | 22% | 40% |

   Splitting long comments into several chunks would let the model see all of
   them. That is a possible later change; it was not needed to ship.
3. **These threads are full of exact names.** Error strings, class names and
   config keys are words BM25 matches exactly, and a small embedding model
   blurs them: on the specific queries BM25 scores 0.951 and 0.908 against
   dense's 0.919 and 0.858.

## What shipped

DBSF, with 100 candidates from each side, and BGE's instruction in front
of every query (`gh_issue/embed.py`, `gh_issue/index/store.py`). The runner
checks that its "DBSF (shipped)" row is the query `gh-issue search` actually
sends: on every query that hides nothing, the two list the same issues in the
same order, with the same scores.

## Run it

It needs Docker, the Inngest dev server and a GitHub token, like `gh-issue`
itself. Syncing pydantic-ai takes about 45 minutes, most of it embedding.

```bash
# 1. An index of its own, so your own stays untouched
docker run -d --name gh-issue-bench -p 6343:6333 qdrant/qdrant:v1.19.1
export QDRANT_URL=http://localhost:6343 GH_ISSUE_HOME=./.gh-issue-bench
export GITHUB_TOKEN=$(gh auth token) INNGEST_DEV=1

# 2. Index both repos: start a worker, sync, and wait until `gh-issue status`
#    says done and its counts stop moving
uv run gh-issue worker &
uv run gh-issue sync inngest/inngest pydantic/pydantic-ai

# 3. Score every method (a few minutes)
uv run python -m benchmark.run
uv run python -m benchmark.chunk_lengths

# Optional: new queries. `build` redraws duplicates from GitHub and titles from
# the index; new written queries need `sample`, and an agent given writer_brief.md.
uv run python -m benchmark.make_queries build
```

Two runs on the same index print the same tables. Qdrant lists equal scores in
any order (the same bare link posted on two issues scores the same), so search
orders ties itself, by repo and then issue number, and the runner uses the same
order. The repos keep changing, though, so a rerun on a later day scores a
little differently.

## Limits

- One embedding model. bge-small is fixed in this example; a larger model, or
  one that reads more than 512 tokens, may close the gap to BM25.
- Two repos, both developer tools with code-heavy threads. Prose-heavy issue
  trackers may favour dense search more.
- The written queries come from an agent that had just read the issue, so even
  its paraphrases share some words with it. Real users' wording is unknown.
- Equal scores go to the lower issue number, as in `gh-issue search`. In a
  duplicate query the original usually is the lower number, so a tie helps
  there; this matters for RRF, which ties often, and hardly for DBSF.
- Only the marked answer counts. A title query that finds a different thread
  about the same bug, or a duplicate query that finds another duplicate of the
  same original, scores as a miss.
