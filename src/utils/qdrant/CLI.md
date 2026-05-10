# Qdrant Inspection CLI

`qdrant-check` is a read-only command-line tool for inspecting the configured
Qdrant review collection. It uses the same Qdrant environment settings as the
ingestion code and does not create, update, or delete points.

The tool reads these settings from the environment:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`
- `QDRANT_DENSE_VECTOR_NAME`
- `QDRANT_SPARSE_VECTOR_NAME`
- `FASTEMBED_CACHE_PATH`

Vectors are hidden in command output by default.

## Commands

### Summary

Show collection health, counts, vector configuration, sparse vector
configuration, and payload schema.

```bash
uv run qdrant-check summary
```

Print the same information as JSON:

```bash
uv run qdrant-check summary --json
```

### Browse

Browse stored review chunks without loading vectors.

```bash
uv run qdrant-check browse --limit 10
```

Filter by source and rating:

```bash
uv run qdrant-check browse --source eurogamer --rating 4 --limit 5
```

Filter by exact clean game name:

```bash
uv run qdrant-check browse --game "Lego Voyagers"
```

Print matching points as JSON:

```bash
uv run qdrant-check browse --source ign --json
```

### Search

Run a hybrid dense+sparse text search using the same embedding models as
ingestion.

```bash
uv run qdrant-check search "space adventure game" --limit 5
```

Filter search results:

```bash
uv run qdrant-check search "horror combat" --source eurogamer --rating 4 --limit 5
```

Print search results as JSON:

```bash
uv run qdrant-check search "space adventure game" --json --limit 2
```

## Shared Options

`browse` and `search` support the same filtering and output options:

- `--limit <number>`: maximum number of points to return. Default: `10`.
- `--source <name>`: filter by the `web` payload field, for example `eurogamer`
  or `ign`.
- `--game <name>`: filter by exact clean `game_name` payload value.
- `--rating <number>`: filter by exact numeric `rating` payload value.
- `--json`: print results as JSON instead of readable terminal text.

## Output Fields

Readable output includes:

- point ID
- score, for search results
- title
- game name
- rating
- source
- URL
- chunk index
- shortened review chunk text

JSON output also includes `metadata`, a nested object for review and game
metadata such as `title`, `subtitle`, `game_name`, `rating`, `url`, and `web`.
