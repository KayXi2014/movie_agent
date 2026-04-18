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
  scripts/    # local prep, enrichment, and benchmarking commands
  data/       # dataset, retrieval artifacts, eval cases, benchmark outputs
  ui/         # local Streamlit debug UI
```

## Current Runtime Flow

1. `POST /recommend` enters through `main.py`
2. `llm.py` asks a tiny LLM step for retrieval hints
3. `retrieval.py` builds a broad local candidate pool using:
   - SQLite + FTS5 lexical retrieval
   - precomputed embedding similarity
   - light reranking and diversification
4. The final LLM sees a compact candidate list and returns:
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
- `TMDB_API_KEY` optional for rebuilding the enriched dataset locally

Set the required key in the same shell before running the API:

```bash
export OLLAMA_API_KEY=your_ollama_api_key_here
```

## Running Locally

### 1. Prepare retrieval artifacts if needed

This repo already includes the generated retrieval artifacts under `data/`, so this step is only required if you want to rebuild them.

```bash
python -m scripts.prepare_local_runtime
```

What it does:

- optionally rebuilds `data/tmdb_top1000_movies_enriched.csv` if `TMDB_API_KEY` is set
- rebuilds:
  - `data/movies.sqlite`
  - `data/movies.sqlite.meta.json`
  - `data/movie_embeddings.npy`
  - `data/movie_embedding_ids.json`
  - `data/movie_embedding_meta.json`

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

- installs dependencies
- warms the sentence-transformer model during build
- starts the API with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

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

## Offline Scripts

Useful maintenance commands:

```bash
python -m scripts.build_enriched_dataset
python -m scripts.build_retrieval_index
python -m scripts.build_movie_embeddings
python -m scripts.prepare_local_runtime
python -m scripts.benchmark_recommender
```

`scripts/tmdb_client.py` is offline-only and is not used by the deployed API.

## Notes

- The deployed API does not expose internal debug fields like `used_llm`.
- The Streamlit app is for local inspection only; it is not part of the production deployment.
- Watch history is used primarily for exclusion and anti-repeat behavior, not as guaranteed taste evidence.
