# gh-issue

Semantic search over a GitHub repository's issue history.

## Running Qdrant

The index lives in Qdrant, started from this directory:

```bash
docker compose up -d
```

The dashboard is at http://localhost:6333/dashboard. Copy `.env.example` to
`.env` and leave `QDRANT_URL` pointing at it.

The tests do not need this — they open Qdrant in embedded mode in a temporary
directory, so `uv run pytest` works with nothing running.
