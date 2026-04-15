import asyncio
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import TOP_MOVIES, get_recommendation

# ---------------------------------------------------------------------------
# DO NOT EDIT: FastAPI app and request/response schemas
#
# These define the API contract. Changing them will break the grader.
# ---------------------------------------------------------------------------

app = FastAPI(title="Movie Recommender")

TIMEOUT_SECONDS = 20
VALID_IDS = set(TOP_MOVIES["tmdb_id"].astype(int))


class WatchHistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendRequest(BaseModel):
    user_id: int
    preferences: str
    history: list[WatchHistoryItem] = []


class RecommendResponse(BaseModel):
    tmdb_id: int
    user_id: int
    description: str


# ---------------------------------------------------------------------------
# DO NOT EDIT: Endpoint
#
# Calls get_recommendation() from llm.py and enforces the output contract.
# Edit llm.py instead.
# ---------------------------------------------------------------------------


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    if not os.environ.get("OLLAMA_API_KEY"):
        raise HTTPException(status_code=500, detail="OLLAMA_API_KEY not set")

    history_ids = {h.tmdb_id for h in request.history}
    history_payload = [{"tmdb_id": h.tmdb_id, "name": h.name} for h in request.history]

    # Rule 3: must respond within 5 seconds
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(get_recommendation, request.preferences, history_payload),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Timed out after {TIMEOUT_SECONDS}s")
    except json.JSONDecodeError as e:
        # Rule 6: LLM returned something that isn't valid JSON
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    # Rule 6: response must be a dict with the expected keys
    if not isinstance(result, dict) or "tmdb_id" not in result:
        raise HTTPException(status_code=502, detail="LLM response missing tmdb_id field")

    tmdb_id = int(result.get("tmdb_id", -1))

    # Rule 5: tmdb_id must be in the candidate list
    if tmdb_id not in VALID_IDS:
        raise HTTPException(status_code=502, detail=f"tmdb_id {tmdb_id} is not in the candidate list")

    # Rule 4: must not recommend something the user has already seen
    if tmdb_id in history_ids:
        raise HTTPException(status_code=502, detail=f"tmdb_id {tmdb_id} is already in the user's watch history")

    description = str(result.get("description", ""))[:500]

    return RecommendResponse(
        tmdb_id=tmdb_id,
        user_id=request.user_id,
        description=description,
    )


@app.get("/")
def health():
    return {"status": "ok", "candidates": len(TOP_MOVIES)}
