# Movie Recommender

A FastAPI movie recommendation service built for a class competition. The project uses a local retrieval stack over a TMDB top-1000 dataset, then makes one final LLM call to choose the best movie and write a short recommendation blurb.

## Project Layout

```text
agentic-movie-recommender/
  main.py     # FastAPI entrypoint
  llm.py      # final LLM selection flow
  retrieval.py
  fts_retrieval.py
  semantic_retrieval.py
  scripts/    # local prep, TMDB enrichment, LLM augmentation, and benchmarking commands
  data/       # dataset, retrieval artifacts, eval cases, benchmark outputs
  ui/         # local Streamlit debug UI
```

## Current Runtime Flow

1. `POST /recommend` enters through `main.py`
2. `retrieval.py` builds a broad local candidate pool using:
   - SQLite + FTS5 lexical retrieval
   - precomputed embedding similarity
   - a simplified semantic-first rerank with explicit genre/avoid filtering
   - light diversification
3. `retrieval.py` returns:
   - a shortlist of up to 12 candidates
   - a lean prompt profile with only:
     - `target_genres`
     - `tone` (when the request clearly expresses one)
     - `avoid`
4. The final LLM sees the raw request, compact shortlist, avoid/watch-history hints, and returns:
   - `tmdb_id`
   - a recommendation description capped at 500 characters
5. If the final LLM fails, the app falls back to a deterministic local choice

## Dependencies

Install the project dependencies from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables:

- `OLLAMA_API_KEY` required for the final recommendation LLM call
- `OLLAMA_API_KEY` also powers offline LLM augmentation
- `TMDB_API_KEY` optional for rebuilding the enriched dataset locally

Set the required key in the same shell before running the API:

```bash
export OLLAMA_API_KEY=your_ollama_api_key_here
```

## Running Locally

### 1. Prepare local data/artifacts if needed

This repo already includes the generated retrieval artifacts under `data/`, so this step is **not part of normal day-to-day usage**. In the common case, you can install dependencies and start `uvicorn` immediately.

Run local preparation only when you explicitly want to:

- rebuild SQLite / embedding artifacts
- refresh TMDB-enriched metadata
- refresh offline LLM augmentation fields
- recover from stale or mismatched artifact metadata

```bash
python -m scripts.prepare_local_runtime
```

What it does:

- checks whether TMDB enrichment is stale, and only refreshes it if TMDB credentials are set and rows still need enrichment
- checks whether LLM augmentation is stale, and only refreshes it if `OLLAMA_API_KEY` is set and rows still need augmentation
- rebuilds the retrieval database and text-embedding artifacts only when the dataset changed or the artifact metadata is stale:
  - `data/movies.sqlite`
  - `data/movies.sqlite.meta.json`
  - `data/movie_embeddings.npy`
  - `data/movie_embedding_ids.json`
  - `data/movie_embedding_meta.json`

By default, `prepare_local_runtime` is idempotent: it skips TMDB enrichment, LLM augmentation, and artifact rebuilds if they are already current.

Useful variants:

```bash
python -m scripts.prepare_local_runtime --skip-tmdb
python -m scripts.prepare_local_runtime --skip-augmentation
python -m scripts.prepare_local_runtime --skip-tmdb --skip-augmentation
python -m scripts.prepare_local_runtime --refresh-augmentation
python -m scripts.prepare_local_runtime --skip-artifacts
python -m scripts.prepare_local_runtime --augment-workers 2
python -m scripts.prepare_local_runtime --augment-model gemma4:31b-cloud
```

Recommended usage patterns:

- normal local API work: do **not** run `prepare_local_runtime`
- only rebuild retrieval artifacts from the current CSV:
  ```bash
  python -m scripts.prepare_local_runtime --skip-tmdb --skip-augmentation
  ```
- refresh augmentation but avoid touching TMDB enrichment:
  ```bash
  python -m scripts.prepare_local_runtime --skip-tmdb --refresh-augmentation
  ```

You do not need to run `scripts.text_artifacts` separately unless you specifically want those lower-level maintenance commands, and you do not need to rerun `scripts.llm_augment` for already-complete rows because the augmentation script resumes only missing fields.

If you want TMDB enrichment in that step:

```bash
export TMDB_API_KEY=your_tmdb_api_key_here
python -m scripts.prepare_local_runtime
```

### 2. Start the API

```bash
uvicorn main:app --reload
```

The default local URL is:

```text
http://127.0.0.1:8000
```

### 3. Send a test request

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "preferences": "I want a grounded sci-fi movie with no superpowers.",
    "history": []
  }'
```

### 4. Optional local UI

The Streamlit app is a local debugging tool for sending API requests and inspecting the retrieval shortlist.

```bash
streamlit run ui/streamlit_ui.py
```

## Deployment To Leapcell

The project is deployable to Leapcell as-is.

### What gets deployed

- root runtime files like `main.py`, `llm.py`, and `retrieval.py`
- `data/` retrieval artifacts and dataset
- root config files like `requirements.txt` and `leapcell.yaml`

### Deploy steps

1. Push the repo to GitHub.
2. In Leapcell, create a new service from that GitHub repo.
3. If needed, set the service root to this project directory.
4. Let Leapcell use the root `leapcell.yaml`.
5. Add the `OLLAMA_API_KEY` secret in Leapcell.
6. Deploy.

The current `leapcell.yaml`:

- installs a lean runtime dependency set from `requirements-leapcell.txt`
- does **not** download or warm the sentence-transformer model during image build
- starts the API with:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

Why this matters:

- Leapcell image builds are resource-constrained.
- Installing `sentence-transformers` pulls in a much heavier stack and warming the model during build can exceed Leapcell limits.
- The deployed app can still run without that package: semantic retrieval will simply stay unavailable on Leapcell, and the runtime will fall back to lexical / metadata retrieval plus the final LLM choice.

Keep using the full `requirements.txt` for local development and offline artifact generation. The lightweight `requirements-leapcell.txt` exists only for hosted deployment.

After deployment, test:

```bash
curl https://YOUR-LEAPCELL-URL/
```

and:

```bash
curl -X POST https://YOUR-LEAPCELL-URL/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "preferences": "I want a good mystery thriller with a smart twist.",
    "history": []
  }'
```

## Benchmarking And Evaluation

The repo includes:

- `data/evaluation_cases.json`
- `scripts/benchmark_recommender.py`

Run the benchmark from the repo root:

```bash
python -m scripts.benchmark_recommender
```

It evaluates both:

- `llm.py`
- `llm_baseline.py`

and writes:

- `data/benchmark_results.csv`
- `data/benchmark_summary.csv`

### Benchmark outputs

`data/benchmark_results.csv` includes one row per case per agent, with fields such as:

- `agent`
- `input_prompt`
- `movie_id`
- `movie_title`
- `description`
- `used_llm`
- `retrieval_mode`
- `runtime`

`data/benchmark_summary.csv` contains aggregated comparison metrics.

### Evaluation goals

The benchmark keeps the existing scoring emphasis:

- hard constraint pass rate
- history-repeat avoidance
- retrieval mode visibility
- description and recommendation quality proxies
- total runtime

## Offline Augmentation

The local prep pipeline now has three layers:

1. TMDB enrichment for factual metadata such as collection info, related titles, alternative titles, spoken languages, and ratings
2. LLM augmentation for semantic fields such as:
   - `essence`
   - `tone_tags_json`
   - `audience_tags_json`
   - `source_tags_json`
   - `keywords_augmented_json`
3. Retrieval artifact rebuilds for:
   - SQLite FTS
   - sentence-transformer embeddings

The deployed API does not run any of these offline steps at request time.

The offline augmenter uses Ollama cloud directly. The current default model is `gemma4:31b-cloud`. You can override the model with `--model` or `AUGMENT_MODEL`.

### Run augmentation directly

```bash
export OLLAMA_API_KEY=your_ollama_api_key_here
python -m scripts.llm_augment
```

The augmentation job is resumable:

- rerunning it only sends rows with missing augmentation fields back to the LLM
- already-complete rows are skipped
- progress is saved batch by batch to `data/tmdb_top1000_movies_enriched.csv`

Important operational warning:

- this step can take a long time on the full 1000-movie dataset
- Ollama cloud may return `429 too many concurrent requests` or read timeouts even at low concurrency
- a completed-looking run may still leave some rows missing until a rerun finishes them
- this step is offline-only and should not be run on normal API startup or deployment

If you hit Ollama rate limits or timeouts, keep the worker count low and rerun:

```bash
python -m scripts.llm_augment --workers 1
```

If you want to avoid rerunning other prep layers while finishing augmentation, use:

```bash
python -m scripts.prepare_local_runtime --skip-tmdb --refresh-augmentation --augment-workers 1
```

If the augmentation eventually completes and you want the retrieval stack to use the new fields, rebuild artifacts afterward:

```bash
python -m scripts.prepare_local_runtime --skip-tmdb --skip-augmentation
```

## Offline Scripts

Useful maintenance commands. These are optional standalone helpers; `python -m scripts.prepare_local_runtime` already covers the normal end-to-end local prep flow, including rebuilding the text embeddings.

```bash
python -m scripts.tmdb_enrichment
python -m scripts.llm_augment
python -m scripts.text_artifacts --target index
python -m scripts.text_artifacts --target embeddings
python -m scripts.prepare_local_runtime
python -m scripts.benchmark_recommender
```

`scripts/tmdb_enrichment.py`, `scripts/llm_augment.py`, and `scripts/text_artifacts.py` are offline-only and are not used by the deployed API.

## Notes

- The deployed API does not expose internal debug fields like `used_llm`.
- The Streamlit app is for local inspection only; it is not part of the production deployment.
- Watch history is used primarily for exclusion and anti-repeat behavior, not as guaranteed taste evidence.
- Retrieval no longer uses a separate pre-retrieval LLM step; the only live model call in the request path is the final recommendation-selection call.
- `scripts.prepare_local_runtime` is a maintenance command, not a normal startup step.
- Offline LLM augmentation is the slowest and least reliable prep layer; expect long runtimes and occasional provider errors, and rely on resumable reruns rather than assuming one pass will always finish cleanly.
