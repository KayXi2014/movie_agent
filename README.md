# Movie Recommender

A FastAPI movie recommendation service built for a class competition. The project uses a local retrieval stack over a TMDB top-1000 dataset, optionally asks a small LLM for compact intent hints, then routes the final LLM through an adaptive shortlist based on retrieval confidence.

## Project Layout

```text
agentic-movie-recommender/
  main.py     # FastAPI entrypoint
  llm.py      # final LLM selection flow
  retrieval.py
  semantic_retrieval.py
  scripts/    # local prep, TMDB enrichment, LLM augmentation, and benchmarking commands
  data/       # dataset, retrieval artifacts, eval cases, benchmark outputs
  ui/         # local Streamlit debug UI
```

## Current Runtime Flow

1. `POST /recommend` enters through `main.py`
2. By default, `llm.py` starts a short intent LLM call in parallel with local retrieval.
   - The intent call only returns compact retrieval hints such as genres, avoid terms, tone, quality preference, country/language, release year, and story setting.
   - If it times out or returns invalid JSON, it is ignored completely.
   - Set `ENABLE_LLM_INTENT=0` to disable this stage.
3. `retrieval.py` builds a broad local candidate pool using:
   - in-memory weighted BM25 lexical retrieval over title, genres, keywords, tags, director/cast, essence, and capped overview terms
   - optional hosted Hugging Face semantic recall for fuzzy/vibe-style requests only
   - literal local constraints such as title/history matches, exact genre aliases, avoid terms, known directors/cast, release-year ranges, runtime requests, and dataset-derived country/language terms
   - optional LLM intent hints for fuzzy mood/style, quality preference, and ambiguous setting intent
   - simple rank fusion from BM25, optional semantic recall, seed-title related IDs, constraint-preserved candidates, and IMDb-backed quality stabilization
   - broad-request quality floors so vague genre requests do not surface obscure low-vote titles unless the match is unusually strong
   - light diversification
4. `retrieval.py` returns:
   - a shortlist of up to 10 candidates
   - a prompt profile with parsed request signals
   - a confidence bundle with `confidence`, `route`, `convergence`, `top_fulfillment`, score gaps, and binary contradiction flags such as genre, avoid, person, year, runtime, or seed misses
5. `llm.py` routes the final stage:
   - high confidence: skip judging and ask the LLM to describe the top candidate only
   - medium confidence: ask the LLM to judge the top 5
   - low confidence: ask the LLM to judge the top 8
6. The final LLM returns:
   - `tmdb_id`
   - a recommendation description capped at 500 characters
7. If the final LLM fails, the app falls back to a deterministic local choice

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
- `HF_TOKEN` optional; only needed when `ENABLE_HF_SEMANTIC_RETRIEVAL=1`
- `ENABLE_LLM_INTENT` optional; defaults to `1`, set to `0` to disable the short pre-retrieval intent LLM
- `INTENT_LLM_TIMEOUT_S` optional; defaults to `4.0`
- `ENABLE_HF_SEMANTIC_RETRIEVAL` optional; defaults to `0`
- `HF_SEMANTIC_TIMEOUT_S` optional; defaults to `2.0`

Set the required key in the same shell before running the API:

```bash
export OLLAMA_API_KEY=your_ollama_api_key_here
```

Optional intent routing controls:

```bash
export INTENT_LLM_TIMEOUT_S=4
# export ENABLE_LLM_INTENT=0  # disable the intent LLM if needed
```

Optional semantic recall controls:

```bash
export HF_TOKEN=your_huggingface_token_here
export ENABLE_HF_SEMANTIC_RETRIEVAL=1
export HF_SEMANTIC_TIMEOUT_S=2
```

Semantic recall is intentionally narrow: it only runs for fuzzy requests such as “like Dune,” “dystopian sci-fi,” “Tarantino-style,” “vibe,” “feel,” or weak lexical cases. If Hugging Face is unavailable or times out, retrieval continues with weighted BM25.

Runtime constraints are local and literal. Requests like “short,” “quick,” “under 90 minutes,” or “less than 2 hours” prefer matching runtimes, and “short/quick/light” requests exclude movies over 180 minutes when enough candidates remain. Requests like “long,” “epic,” or “over 2 hours” prefer longer movies without hard-failing if the dataset has too few exact matches.

## Running Locally

### 1. Prepare local data/artifacts if needed

This repo already includes the generated retrieval artifacts under `data/`, so this step is **not part of normal day-to-day usage**. In the common case, you can install dependencies and start `uvicorn` immediately.

Run local preparation only when you explicitly want to:

- refresh TMDB-enriched metadata
- apply IMDb ratings from `data/IMDB_ratings.tsv`
- refresh offline LLM augmentation fields
- update the CSV before optionally rebuilding hosted-HF movie embeddings

```bash
python -m scripts.prepare_local_runtime
```

What it does:

- checks whether TMDB enrichment is stale, and only refreshes it if TMDB credentials are set and rows still need enrichment
- applies `data/IMDB_ratings.tsv` into the active dataset only when IMDb rows differ from the CSV
- checks whether LLM augmentation is stale, and only refreshes it if `OLLAMA_API_KEY` is set and rows still need augmentation

By default, `prepare_local_runtime` is idempotent: it skips TMDB enrichment, IMDb overlay, and LLM augmentation if they are already current.

Useful variants:

```bash
python -m scripts.prepare_local_runtime --skip-tmdb
python -m scripts.prepare_local_runtime --skip-imdb
python -m scripts.prepare_local_runtime --skip-augmentation
python -m scripts.prepare_local_runtime --skip-tmdb --skip-augmentation
python -m scripts.prepare_local_runtime --refresh-augmentation
python -m scripts.prepare_local_runtime --augment-workers 2
python -m scripts.prepare_local_runtime --augment-model gemma4:31b-cloud
```

Recommended usage patterns:

- normal local API work: do **not** run `prepare_local_runtime`
- check the current CSV without touching optional providers:
  ```bash
  python -m scripts.prepare_local_runtime --skip-tmdb --skip-imdb --skip-augmentation
  ```
- apply IMDb TSV ratings without touching TMDB or LLM augmentation:
  ```bash
  python -m scripts.prepare_local_runtime --skip-tmdb --skip-augmentation
  ```
- refresh augmentation but avoid touching TMDB enrichment:
  ```bash
  python -m scripts.prepare_local_runtime --skip-tmdb --refresh-augmentation
  ```

You do not need to run `scripts.text_artifacts` separately unless you specifically want to rebuild optional semantic embeddings, and you do not need to rerun `scripts.llm_augment` for already-complete rows because the augmentation script resumes only missing fields.

To rebuild optional Hugging Face semantic embeddings:

```bash
export HF_TOKEN=your_huggingface_token_here
python -m scripts.text_artifacts --target embeddings
```

Do not enable `ENABLE_HF_SEMANTIC_RETRIEVAL=1` in deployment unless the embedding artifacts match the active CSV and `HF_TOKEN` is configured.

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

The Streamlit app is a local debugging tool for sending API requests and inspecting retrieval confidence, adaptive routing, and the shortlist that would be sent to the final LLM.

```bash
streamlit run ui/streamlit_ui.py
```

### 5. Use `llm.py` directly without the API

You can call the recommendation agent directly when you do not need FastAPI, HTTP, or JSON request handling. This is useful for notebooks, scripts, quick local experiments, and competition benchmarks.

Interactive mode:

```bash
export OLLAMA_API_KEY=your_ollama_api_key_here
python llm.py
```

Programmatic use:

```python
from llm import get_recommendation

result = get_recommendation(
    "Recommend a fast-paced movie for someone who dislikes slow films",
    ["The Dark Knight Rises"],
)

print(result["tmdb_id"])
print(result["description"])
print(result["used_llm"])
```

`get_recommendation(preferences, history)` returns the same core output shape used by the API: `tmdb_id`, `description`, and `used_llm`. The `history` argument can be a list of movie titles; IDs are not required. This direct path still uses the same retrieval, optional intent LLM, final LLM, and deterministic fallback logic as the API.

## Deployment To Leapcell

The project is deployable to Leapcell as-is.

### What gets deployed

- root runtime files like `main.py`, `llm.py`, and `retrieval.py`
- `data/` retrieval artifacts and dataset
- root config files like `requirements-leapcell.txt` and `leapcell.yaml`

### Deploy steps

1. Push the repo to GitHub.
2. In Leapcell, create a new service from that GitHub repo.
3. If needed, set the service root to this project directory.
4. Let Leapcell use the root `leapcell.yaml`.
5. Add the `OLLAMA_API_KEY` secret in Leapcell.
6. Optional: set `ENABLE_LLM_INTENT=0` if you want hosted requests to skip the short intent LLM stage.
7. Optional: set `HF_TOKEN` and `ENABLE_HF_SEMANTIC_RETRIEVAL=1` only if you have prebuilt matching HF embedding artifacts and want fuzzy-query semantic recall.
8. Deploy.

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
- Runtime retrieval defaults to weighted BM25 lexical scoring and does not load local sentence-transformer models.
- Optional semantic recall uses hosted Hugging Face query embeddings only when `ENABLE_HF_SEMANTIC_RETRIEVAL=1`; it is disabled by default for predictable hosted latency.
- The intent LLM uses Ollama cloud and is controlled by `ENABLE_LLM_INTENT`, which defaults to enabled. If it misses its timeout, the request continues with local retrieval only.

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

The local prep pipeline now has three CSV-maintenance layers:

1. TMDB enrichment for factual metadata such as collection info, related titles, alternative titles, spoken languages, and ratings
2. IMDb rating overlay from `data/IMDB_ratings.tsv`
3. LLM augmentation for semantic fields such as:
   - `essence`
   - `tone_tags_json`
   - `audience_tags_json`
   - `source_tags_json`
   - `keywords_augmented_json`

Optional hosted-HF semantic embeddings are rebuilt separately with `scripts.text_artifacts`; SQLite text artifacts are no longer used.

The deployed API does not run any of these offline steps at request time.

The offline augmenter uses Ollama cloud directly. The current default model is `gemma4:31b-cloud`. You can override the model with `--model` or `AUGMENT_MODEL`.

### IMDb ratings

The prep pipeline reads `data/IMDB_ratings.tsv`, which should use the IMDb `title.ratings.tsv` shape:

```tsv
tconst	averageRating	numVotes
tt0000001	5.7	2209
```

It joins `tconst` to the movie dataset's `imdb_id` column, then writes `imdb_rating` and `imdb_votes` into the active movie CSV. Apply it through the normal prep command:

```bash
python -m scripts.prepare_local_runtime --skip-tmdb --skip-augmentation
```

The step is idempotent: if the active movie CSV already has those IMDb values, it skips writing. Use `--skip-imdb` when you explicitly want to avoid touching IMDb fields. Manual IMDb override CSVs are not used; update `data/IMDB_ratings.tsv` or the source dataset instead.

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

If the augmentation eventually completes and you use optional semantic recall, rebuild embeddings afterward:

```bash
python -m scripts.text_artifacts --target embeddings
```

## Offline Scripts

Useful maintenance commands. These are optional standalone helpers; `python -m scripts.prepare_local_runtime` covers CSV prep, and runtime lexical retrieval is in-memory weighted BM25 over the active CSV.

```bash
python -m scripts.tmdb_enrichment
python -m scripts.imdb_enrichment
python -m scripts.llm_augment
python -m scripts.text_artifacts --target embeddings
python -m scripts.prepare_local_runtime
python -m scripts.benchmark_recommender
```

`scripts/tmdb_enrichment.py`, `scripts/llm_augment.py`, and `scripts/text_artifacts.py` are offline-only and are not used by the deployed API.

## Notes

- The deployed API does not expose internal debug fields like `used_llm`.
- The Streamlit app is for local inspection only; it is not part of the production deployment.
- Watch history is used primarily for exclusion and anti-repeat behavior, not as guaranteed taste evidence.
- Retrieval uses a short pre-retrieval intent LLM by default; set `ENABLE_LLM_INTENT=0` if you want the only live model call in the request path to be the final recommendation-selection/description call.
- Optional semantic recall is disabled by default; enable it with `ENABLE_HF_SEMANTIC_RETRIEVAL=1` only when `HF_TOKEN` and matching `data/movie_embeddings.npy` artifacts are available.
- Add or refresh IMDb quality data in `data/IMDB_ratings.tsv`; when present, IMDb rating/vote fields become the preferred quality signal, with TMDB as fallback.
- Semantic embeddings intentionally use compact factual text only: tone/audience/source tags, TMDB keywords, genres, overview, and tagline. LLM-generated `essence` and augmented keywords are kept out of embedding text because they can be verbose or inferred enough to confuse dense semantic recall.
- `scripts.prepare_local_runtime` is a maintenance command, not a normal startup step.
- Offline LLM augmentation is the slowest and least reliable prep layer; expect long runtimes and occasional provider errors, and rely on resumable reruns rather than assuming one pass will always finish cleanly.
