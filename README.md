# Movie Recommender – Starter

A minimal API that recommends a movie based on a user's stated preferences. It picks from the 40 most-voted movies in the TMDB dataset and uses an LLM to choose the best match and write a short pitch.

---

## Running locally

You can run your API on your own laptop to start, for testing purposes.

**1. Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your API key**
You will need to obtain an API key from [ollama.com/settings/keys](https://ollama.com/settings/keys). A free account is included with Ollama.

You need to bring this API key into your terminal environment, by running the command:

```bash
export OLLAMA_API_KEY=your_ollama_api_key_here
```

**3. Start the server**

```bash
uvicorn main:app --reload
```

You should see it output:

``` 
INFO:     Will watch for changes in these directories: ['/path/to/your/app']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Note the port -- 8000 by default, althought it may be something else if 8000 is occupied.
The server will automatically reload to reflect your changes if you edit the files in that directory.

**4. Send a test request**

You can now make requests to your agent by `curl`-ing it, for example:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "preferences": "I love superheroes and feel-good buddy cop stories.",
    "history": [{"tmdb_id": 24428, "name": "The Avengers"}]
  }'
```

Note that the port (8000 here) must be the same one that your app is listening on.

**5. Optional: Run the Streamlit UI**

This repo includes a separate Streamlit frontend in `streamlit_ui.py` so you can use the recommender from a browser form instead of scripts.

Start it in a second terminal (with the same virtual environment activated):

```bash
streamlit run streamlit_ui.py
```

Then open the Streamlit URL shown in your terminal (usually `http://localhost:8501`).

The UI lets you enter:
- `user_id`
- free-text `preferences`
- watch history rows (`tmdb_id` + `name`)

By default it sends requests to `http://127.0.0.1:8000/recommend`, which matches the local FastAPI server.

---

## Deploying to Leapcell

[Leapcell](https://leapcell.io/) is a provider that will host your API so that anyone can send requests to it. There are many services like this, but Leapcell is easy to use and has a very generous free tier.

To play the game in class, you will need to have deployed your app publicly so that other students can access it.

You will need a free leapcell account, and you will need to [connect it to your Github account](https://docs.leapcell.io/service/connect-to-github/).

To deploy your app to Leapcell, follow the steps:


**1. Push your code to GitHub.**

**2. Create a new service on Leapcell.**

Connect your GitHub repo. Leapcell will detect `leapcell.yaml` and use it for build and run:

```yaml
build:
  buildCommand: pip install -r requirements.txt
run:
  runCommand: uvicorn main:app --host 0.0.0.0 --port 8080
```

**3. Set your API key secret in the Leapcell dashboard.**

Go to your service's **Environment Variables** settings and add `OLLAMA_API_KEY` with your key as the value. Do not commit the key to your repo.

**4. Deploy.**

Leapcell will install dependencies, start the server, and give you a public URL. Submit that URL as your API endpoint.

---

## Improving the baseline

Some ideas to get you started:

- Expand the candidate pool beyond the top 40 (e.g. filter by genre first, then rank).
- Include genre, keywords, or cast in the prompt to give the LLM more signal.
- Use the watch history to steer away from movies too similar to ones already seen.
- Experiment with prompt phrasing — chain-of-thought or few-shot examples often improve output quality.
- Cache responses for identical inputs to stay safely under the 5-second deadline.

## What Changed From `llm_baseline.py`

The baseline recommender sends one large prompt to the LLM using only the 40 most-voted movies in the dataset. It includes limited metadata per movie and asks the model to handle retrieval, ranking, and blurb generation in one shot.

The new `llm.py` uses a hybrid pipeline instead:

- It uses the full TMDB top-1000 dataset instead of only the top 40 most-voted movies, which expands coverage beyond blockbuster-heavy titles.
- It starts with a broad heuristic prefilter over the full dataset, so the LLM no longer has to search the entire candidate set by itself.
- It runs a first LLM pass to extract a structured user profile from the request and watch history. That profile includes target genres, preferred tones, preferred themes, negative constraints, and history-derived signals such as liked actors or directors.
- It then reranks candidates using that structured profile, which makes the system less rigid than pure keyword scoring and better at handling natural-language requests like “feel-good buddy cop”, “not another Marvel team-up”, or “dark fantasy for adults”.
- It resolves watch history primarily by `tmdb_id`, with title matching as a fallback. This makes history handling more robust even if the movie name is entered inconsistently.
- It uses more metadata from the dataset, including `keywords`, `tagline`, `director`, `top_cast`, `original_language`, and production information, not just title, genres, and overview.
- It sends only a short ranked shortlist into the final recommendation prompt, which gives the LLM a cleaner decision set and reduces prompt clutter.
- It adds response caching and a reusable Ollama client so repeated requests are faster and more stable.
- It adds stronger fallback behavior. If profile extraction or final selection fails, the system still returns the top deterministic candidate from the reranked shortlist instead of crashing.

In short, `llm_baseline.py` is a one-shot prompt over a small popularity-skewed candidate pool, while `llm.py` is a two-stage recommendation system: prefilter first, extract a taste profile, rerank with that profile, then let the LLM make the final pick from a much better shortlist.

## Evaluation Setup

The repo includes an offline benchmark in `benchmark_recommender.py` and a fixed evaluation set in `evaluation_cases.json`.

The benchmark compares the baseline-style top-40 retriever against the new pipeline using several signals:

- constraint pass rate: avoid watched titles and avoid explicit blocked titles or franchises
- genre and theme match: whether the recommendation matches the target genres and metadata keywords in each case
- discourage penalties: whether the result drifts into genres or themes the prompt explicitly asks to avoid
- history-aware behavior: whether cases that depend on prior watch history preserve useful taste signals
- diversity: how many unique titles are recommended across the whole benchmark
- average movie rating: a weak proxy for general movie quality, used only as supporting evidence

This benchmark is meant to be a repeatable offline comparison tool, while final recommendation quality should still be validated with blind human preference testing on sampled requests.


---

## Key libraries

### FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is the web framework. It handles routing HTTP requests to Python functions.

The app is created with one line:

```python
app = FastAPI(title="Movie Recommender")
```

Routes are declared with decorators. The `@app.post("/recommend")` decorator means: when the server receives a `POST` request to `/recommend`, call the `recommend()` function.

```python
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    ...
```

FastAPI automatically reads the JSON body of the incoming request, validates it against `RecommendRequest`, and serializes the return value to JSON using `RecommendResponse`. You do not need to call `json.loads` or `json.dumps` yourself.

### Pydantic

[Pydantic](https://docs.pydantic.dev/) is what FastAPI uses under the hood to define and enforce data shapes. You declare a class that inherits from `BaseModel`, and Pydantic will automatically parse and validate incoming data against it.

There are three models in this project:

```python
class WatchHistoryItem(BaseModel):
    tmdb_id: int
    name: str

class RecommendRequest(BaseModel):
    user_id: int
    preferences: str
    history: list[WatchHistoryItem] = []   # optional, defaults to empty list

class RecommendResponse(BaseModel):
    tmdb_id: int
    user_id: int
    description: str
```

If a request arrives with a missing required field (e.g. no `user_id`), or the wrong type (e.g. `user_id` is a string that can't be cast to int), FastAPI will automatically return a `422 Unprocessable Entity` error — you never see that case inside `recommend()`.

### Ollama (`ollama`)

The [`ollama`](https://pypi.org/project/ollama/) package is the official Python SDK. It handles authentication and the HTTP call to the Ollama cloud API.

```python
import ollama

client = ollama.Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)
```

Making a call looks like this:

```python
response = client.chat(
    model="gemini-3-flash-preview",
    messages=[{"role": "user", "content": prompt}],
    format="json",
)
result = json.loads(response.message.content)
```

Setting `format="json"` is Ollama's **JSON mode** — it instructs the model to return valid JSON, so you can call `json.loads` directly without any cleanup.
