"""
Movie recommendation logic.

The API contract lives in main.py. This module focuses on:
- retrieving relevant candidates from the full dataset
- using watch history as a taste signal while avoiding repeats
- asking the model to choose from a short ranked shortlist
- caching repeat requests to reduce latency
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

import ollama
import pandas as pd

MODEL = os.environ.get("OLLAMA_MODEL", "gemini-3-flash-preview")
LLM_CLIENT_TIMEOUT_SECONDS = 20
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_MOVIES = pd.read_csv(DATA_PATH).fillna("")

TEXT_COLUMNS = [
    "title",
    "original_title",
    "genres",
    "overview",
    "tagline",
    "keywords",
    "director",
    "top_cast",
    "original_language",
    "production_countries",
]
SHORTLIST_SIZE = 7
MAX_KEYWORDS = 8
TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "like",
    "movie",
    "movies",
    "of",
    "on",
    "or",
    "something",
    "that",
    "the",
    "to",
    "want",
    "watch",
    "with",
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _tokenize(value: Any) -> set[str]:
    return {token for token in TOKEN_RE.findall(_normalize_text(value)) if token not in STOP_WORDS}


def _split_csvish(value: Any) -> set[str]:
    items = set()
    for part in str(value or "").split(","):
        cleaned = part.strip().lower()
        if cleaned:
            items.add(cleaned)
    return items


def _title_root(title: Any) -> str:
    text = _normalize_text(title)
    if not text:
        return ""
    for delimiter in (" - ", ":"):
        if delimiter in text:
            text = text.split(delimiter, 1)[0]
    text = re.sub(r"\b(part|chapter|episode|vol|volume)\b.*$", "", text).strip()
    text = re.sub(r"\b\d+\b$", "", text).strip()
    return text


def _prepare_movies(df: pd.DataFrame) -> pd.DataFrame:
    movies = df.copy()
    movies["normalized_title"] = movies["title"].map(_normalize_text)
    movies["title_root"] = movies["title"].map(_title_root)
    movies["genres_set"] = movies["genres"].map(_split_csvish)
    movies["keywords_set"] = movies["keywords"].map(_split_csvish)
    movies["genres_tokens"] = movies["genres"].map(_tokenize)
    movies["keywords_tokens"] = movies["keywords"].map(_tokenize)
    movies["cast_set"] = movies["top_cast"].map(_split_csvish)
    movies["director_set"] = movies["director"].map(_split_csvish)
    movies["overview_tokens"] = movies["overview"].map(_tokenize)
    movies["tagline_tokens"] = movies["tagline"].map(_tokenize)
    movies["title_tokens"] = movies["title"].map(_tokenize)
    movies["search_blob"] = movies[TEXT_COLUMNS].agg(" ".join, axis=1).map(_normalize_text)
    return movies


MOVIES = _prepare_movies(TOP_MOVIES)
MOVIES_BY_TITLE = MOVIES.groupby("normalized_title", sort=False)
MOVIES_BY_TMDB_ID = MOVIES.set_index("tmdb_id", drop=False)


@lru_cache(maxsize=1)
def _get_client() -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=LLM_CLIENT_TIMEOUT_SECONDS,
    )


def _normalize_history_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        tmdb_id = item.get("tmdb_id")
        name = item.get("name", "")
    else:
        tmdb_id = None
        name = item

    normalized_name = " ".join(str(name or "").split())
    if tmdb_id is None and not normalized_name:
        return None

    normalized_item: dict[str, Any] = {"name": normalized_name}
    if tmdb_id is not None:
        try:
            normalized_item["tmdb_id"] = int(tmdb_id)
        except (TypeError, ValueError):
            pass
    return normalized_item


def _history_rows(history: tuple[tuple[int | None, str], ...]) -> pd.DataFrame:
    rows = []
    seen_ids: set[int] = set()

    for tmdb_id, name in history:
        if tmdb_id is not None and tmdb_id in MOVIES_BY_TMDB_ID.index:
            row = MOVIES_BY_TMDB_ID.loc[tmdb_id]
            if int(row.tmdb_id) not in seen_ids:
                rows.append(row.to_frame().T)
                seen_ids.add(int(row.tmdb_id))
            continue

        title = _normalize_text(name)
        if title in MOVIES_BY_TITLE.groups:
            group = MOVIES_BY_TITLE.get_group(title)
            unseen = group[~group["tmdb_id"].isin(seen_ids)]
            if not unseen.empty:
                rows.append(unseen)
                seen_ids.update(unseen["tmdb_id"].astype(int).tolist())

    if not rows:
        return MOVIES.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tmdb_id"])


def _derive_user_profile(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    history_df = _history_rows(history)
    preference_tokens = _tokenize(preferences)
    preferred_genre_tokens = set()

    liked_genres: set[str] = set()
    liked_keywords: set[str] = set()
    liked_cast: set[str] = set()
    liked_directors: set[str] = set()
    watched_roots: set[str] = set()

    for row in history_df.itertuples():
        liked_genres.update(row.genres_set)
        liked_keywords.update(sorted(row.keywords_set)[:MAX_KEYWORDS])
        liked_cast.update(sorted(row.cast_set)[:3])
        liked_directors.update(row.director_set)
        if row.title_root:
            watched_roots.add(row.title_root)

    for genre_text in MOVIES["genres"].unique():
        genre_tokens = _tokenize(genre_text)
        if genre_tokens and genre_tokens.issubset(preference_tokens):
            preferred_genre_tokens.update(genre_tokens)

    return {
        "preference_tokens": preference_tokens,
        "preferred_genre_tokens": preferred_genre_tokens,
        "history_titles": {_normalize_text(name) for _, name in history if _normalize_text(name)},
        "history_tmdb_ids": {tmdb_id for tmdb_id, _ in history if tmdb_id is not None},
        "liked_genres": liked_genres,
        "liked_keywords": liked_keywords,
        "liked_cast": liked_cast,
        "liked_directors": liked_directors,
        "watched_roots": watched_roots,
        "history_df": history_df,
    }


def _score_movie(row: pd.Series, profile: dict[str, Any]) -> float:
    score = 0.0

    if int(row["tmdb_id"]) in profile["history_tmdb_ids"]:
        return -10_000.0

    if row["normalized_title"] in profile["history_titles"]:
        return -10_000.0

    token_matches = profile["preference_tokens"].intersection(row["title_tokens"])
    token_matches |= profile["preference_tokens"].intersection(row["overview_tokens"])
    token_matches |= profile["preference_tokens"].intersection(row["tagline_tokens"])
    token_matches |= profile["preference_tokens"].intersection(row["keywords_tokens"])
    score += 2.2 * len(token_matches)

    explicit_genre_matches = profile["preferred_genre_tokens"].intersection(row["genres_tokens"])
    score += 3.5 * len(explicit_genre_matches)

    genre_matches = profile["liked_genres"].intersection(row["genres_set"])
    score += 3.0 * len(genre_matches)

    keyword_matches = profile["liked_keywords"].intersection(row["keywords_set"])
    score += 1.2 * len(keyword_matches)

    cast_matches = profile["liked_cast"].intersection(row["cast_set"])
    score += 1.5 * len(cast_matches)

    director_matches = profile["liked_directors"].intersection(row["director_set"])
    score += 2.5 * len(director_matches)

    if row["title_root"] and row["title_root"] in profile["watched_roots"]:
        score -= 6.0

    score += min(float(row["vote_average"]) or 0.0, 10.0) * 0.25
    score += min(float(row["vote_count"]) or 0.0, 5000.0) / 5000.0
    return score


def _build_shortlist(preferences: str, history: tuple[tuple[int | None, str], ...]) -> list[dict[str, Any]]:
    profile = _derive_user_profile(preferences, history)
    candidates = MOVIES.copy()
    candidates["score"] = candidates.apply(_score_movie, axis=1, profile=profile)
    candidates = candidates.sort_values(["score", "vote_average", "vote_count"], ascending=False)
    shortlist = []

    for row in candidates.head(100).itertuples():
        if int(row.tmdb_id) in profile["history_tmdb_ids"] or row.normalized_title in profile["history_titles"]:
            continue
        shortlist.append(
            {
                "tmdb_id": int(row.tmdb_id),
                "title": row.title,
                "year": int(row.year),
                "genres": row.genres,
                "overview": str(row.overview)[:350],
                "tagline": str(row.tagline)[:160],
                "keywords": ", ".join(sorted(row.keywords_set)[:MAX_KEYWORDS]),
                "director": row.director,
                "top_cast": ", ".join(sorted(row.cast_set)[:4]),
                "score": round(float(row.score), 2),
            }
        )
        if len(shortlist) >= SHORTLIST_SIZE:
            break

    return shortlist


def _build_selection_prompt(preferences: str, history: tuple[tuple[int | None, str], ...], shortlist: list[dict[str, Any]]) -> str:
    profile = _derive_user_profile(preferences, history)
    history_text = ", ".join(f'"{name}"' for _, name in history if name) if history else "none"
    
    # Extract liked directors and cast with associated films
    liked_dirs_with_films = {}
    liked_cast_with_films = {}
    for row in profile["history_df"].itertuples():
        for d in row.director_set:
            if d not in liked_dirs_with_films:
                liked_dirs_with_films[d] = []
            liked_dirs_with_films[d].append(row.title)
        for c in sorted(row.cast_set)[:3]:
            if c not in liked_cast_with_films:
                liked_cast_with_films[c] = []
            liked_cast_with_films[c].append(row.title)
    
    director_context = ""
    if liked_dirs_with_films:
        dirs_str = ", ".join(f"{d} ({', '.join(liked_dirs_with_films[d])})" for d in list(liked_dirs_with_films.keys())[:3])
        director_context = f"\nFavored directors from watch history: {dirs_str}"
    
    cast_context = ""
    if liked_cast_with_films:
        cast_str = ", ".join(f"{c} ({', '.join(liked_cast_with_films[c])})" for c in list(liked_cast_with_films.keys())[:3])
        cast_context = f"\nFavored actors from watch history: {cast_str}"
    
    shortlist_text = "\n".join(
        (
            f'- tmdb_id={movie["tmdb_id"]} | "{movie["title"]}" ({movie["year"]})'
            f' | genres: {movie["genres"]}'
            f' | director: {movie["director"] or "unknown"}'
            f' | cast: {movie["top_cast"] or "unknown"}'
            f' | overview: {movie["overview"] or "unknown"}'
        )
        for movie in shortlist
    )
    return f"""You are a movie recommendation assistant. Your task is to pick ONE movie that will resonate with this user.

User preferences:
"{preferences}"

Watch history (films already watched, do NOT recommend):
{history_text}{director_context}{cast_context}

Instructions:
1. Infer the user's taste from their preferences and what they've enjoyed before.
2. Look for movies that share genre, tone, themes, or creative talent (directors/cast) with their history.
3. Pick exactly one movie from the shortlist below.
4. Write a description that:
   - Opens with a SHORT STORY HOOK (what the movie is about, 1-2 sentences)
   - If the film shares a director or actor from their watch history, mention this connection briefly
   - If NO director/actor match exists, OMIT any mention of cast/director entirely
   - Keep total length under 500 characters

Shortlist:
{shortlist_text}

Provide ONLY valid JSON (no markdown, no explanation):
{{
  "tmdb_id": <integer>,
  "description": "<story hook (1-2 sentences) + reason it matches their taste. Mention shared directors/actors ONLY if they appear in watch history.>"
}}"""


def _fallback_description(movie: dict[str, Any], preferences: str, profile: dict[str, Any] | None = None) -> str:
    # Start with story hook from overview or tagline, preserving complete sentences
    story_hook = ""
    if movie["overview"]:
        text = movie["overview"]
        # Try to find a complete sentence within a reasonable length
        max_length = 280
        if len(text) > max_length:
            text = text[:max_length]
            # Prefer to cut at sentence end (period + space)
            period_idx = text.rfind('. ')
            if period_idx > 100:  # Make sure we have at least a meaningful chunk
                text = text[:period_idx + 1]
            else:
                # Fallback to last word boundary
                last_space = text.rfind(' ')
                if last_space > 100:
                    text = text[:last_space]
        story_hook = text.rstrip('.').rstrip() + "."
    elif movie["tagline"]:
        text = movie["tagline"][:150]
        if len(text) >= 150:
            last_space = text.rfind(' ')
            if last_space > 60:
                text = text[:last_space]
        story_hook = text.rstrip('.').rstrip() + "."
    
    # Add cast/director connection ONLY if there's an actual match
    connection = ""
    if profile:
        # Check for director match
        if movie["director"]:
            director_names = [d.strip().lower() for d in movie["director"].split(",")][:1]
            liked_dirs_lower = {d.lower() for d in profile.get("liked_directors", set())}
            for d_name in director_names:
                if d_name in liked_dirs_lower:
                    connection = f" From director {d_name.title()}, whose work you love."
                    break
        
        # Check for cast match if no director match
        if not connection and movie["top_cast"]:
            cast_names = [c.strip().lower() for c in movie["top_cast"].split(",")][:2]
            liked_cast_lower = {c.lower() for c in profile.get("liked_cast", set())}
            for c_name in cast_names:
                if c_name in liked_cast_lower:
                    connection = f" Stars {c_name.title()}, one of your favorites."
                    break
    
    # Add why it matches preferences, truncating at word boundary
    pref_text = preferences.strip().strip('\"').strip()  # Remove any surrounding quotes
    if len(pref_text) > 100:
        pref_text = pref_text[:100]
        last_space = pref_text.rfind(' ')
        if last_space > 40:
            pref_text = pref_text[:last_space]
    reason = f"It matches your taste for {pref_text}."
    
    description = f"{story_hook}{connection} {reason}"
    return description[:500]


def _choose_with_llm(preferences: str, history: tuple[tuple[int | None, str], ...], shortlist: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = _build_selection_prompt(preferences, history, shortlist)
    logger.info(
        "LLM request starting: preferences_len=%d history_count=%d shortlist_count=%d",
        len(preferences),
        len(history),
        len(shortlist),
    )
    response = _get_client().chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    try:
        payload = json.loads(response.message.content)
    except json.JSONDecodeError as e:
        logger.warning("LLM returned invalid JSON on first parse attempt: %s", e)
        # If JSON parsing fails, try to extract valid JSON manually
        content = response.message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.exception("LLM response still invalid after cleanup; raw content starts with: %r", content[:200])
            raise ValueError(f"LLM returned invalid JSON: {e}")
    
    valid_ids = {movie["tmdb_id"] for movie in shortlist}
    tmdb_id = int(payload.get("tmdb_id", -1))
    if tmdb_id not in valid_ids:
        logger.error(
            "LLM selected tmdb_id outside shortlist: tmdb_id=%s valid_ids=%s",
            tmdb_id,
            sorted(valid_ids),
        )
        raise ValueError(f"Model selected tmdb_id {tmdb_id}, which is outside the shortlist")
    payload["description"] = str(payload.get("description", ""))[:500]
    logger.info("LLM request succeeded: tmdb_id=%s description_len=%d", tmdb_id, len(payload["description"]))
    return payload


@lru_cache(maxsize=256)
def _get_recommendation_cached(preferences: str, history: tuple[str, ...]) -> dict[str, Any]:
    shortlist = _build_shortlist(preferences, history)
    if not shortlist:
        raise ValueError("No candidate movies available")

    profile = _derive_user_profile(preferences, history)
    try:
        return _choose_with_llm(preferences, history, shortlist)
    except Exception as exc:
        logger.exception(
            "Falling back to heuristic recommendation: preferences_len=%d history_count=%d shortlist_count=%d error=%s",
            len(preferences),
            len(history),
            len(shortlist),
            exc,
        )
        best = shortlist[0]
        return {
            "tmdb_id": best["tmdb_id"],
            "description": _fallback_description(best, preferences, profile),
        }


def get_recommendation(preferences: str, history: list[Any]) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    normalized_preferences = " ".join(preferences.split())
    normalized_history = []
    for item in history:
        normalized_item = _normalize_history_item(item)
        if normalized_item is None:
            continue
        normalized_history.append((normalized_item.get("tmdb_id"), normalized_item["name"]))

    normalized_history = tuple(sorted(set(normalized_history), key=lambda item: (item[0] is None, item[0], item[1].lower())))
    return _get_recommendation_cached(normalized_preferences, normalized_history)


if __name__ == "__main__":
    print("Movie recommender – type your preferences and press Enter.")
    print("For watch history, enter comma-separated movie titles (or leave blank).")
    print()

    preferences = input("Preferences: ").strip()
    history_raw = input("Watch history (optional): ").strip()
    history = [t.strip() for t in history_raw.split(",")] if history_raw else []

    print("\nThinking...\n")
    result = get_recommendation(preferences, history)

    match = TOP_MOVIES[TOP_MOVIES["tmdb_id"] == result["tmdb_id"]]
    title = match.iloc[0]["title"] if not match.empty else "unknown"

    print(f"Recommendation: {title} (tmdb_id={result['tmdb_id']})")
    print(f"\n{result['description']}")
