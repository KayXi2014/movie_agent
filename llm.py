"""
Movie recommendation logic.

Pipeline:
1. Broad heuristic prefilter over the full dataset
2. LLM extracts a compact structured taste profile
3. Profile-guided reranking of candidates
4. LLM picks one movie from a short shortlist and writes the final blurb
5. Deterministic fallback if any LLM step fails
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
LLM_CLIENT_TIMEOUT_SECONDS = 10
PREFILTER_SIZE = 80
SHORTLIST_SIZE = 7
MAX_KEYWORDS = 8
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
    "love",
    "movie",
    "movies",
    "of",
    "on",
    "or",
    "something",
    "stories",
    "story",
    "that",
    "the",
    "to",
    "want",
    "watch",
    "with",
}
PHRASE_HINTS = {
    "superhero": ["hero", "superhero", "marvel", "dc", "vigilante", "powers"],
    "buddy cop": ["buddy cop", "buddy comedy", "cop", "police", "detective", "partners"],
    "rom-com": ["romance", "comedy", "chemistry", "romantic", "funny"],
    "feel-good": ["uplifting", "heartwarming", "friendship", "hopeful", "fun"],
    "coming-of-age": ["coming of age", "teen", "growing up", "youth"],
    "dark fantasy": ["dark", "ominous", "fantasy", "adult"],
    "mind-bending": ["mind-bending", "cerebral", "twist", "psychological", "ambitious"],
}
NEGATIVE_PATTERNS = {
    "not another marvel": ["marvel", "avengers", "captain america", "iron man", "guardians"],
    "not an action movie": ["action", "battle", "explosion", "war"],
    "not kid-friendly": ["family", "animation", "kid", "children"],
    "not another superhero": ["superhero", "marvel", "dc", "hero"],
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


def _normalize_history(history: list[Any]) -> tuple[tuple[int | None, str], ...]:
    normalized = []
    for item in history:
        normalized_item = _normalize_history_item(item)
        if normalized_item is None:
            continue
        normalized.append((normalized_item.get("tmdb_id"), normalized_item["name"]))
    return tuple(sorted(set(normalized), key=lambda item: (item[0] is None, item[0], item[1].lower())))


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


def _extract_phrase_hints(preferences: str) -> tuple[set[str], set[str]]:
    pref_lower = _normalize_text(preferences)
    positive = set()
    negative = set()

    for phrase, hints in PHRASE_HINTS.items():
        if phrase in pref_lower:
            positive.update(hints)

    for phrase, hints in NEGATIVE_PATTERNS.items():
        if phrase in pref_lower:
            negative.update(hints)

    if " not " in f" {pref_lower} ":
        if "action" in pref_lower:
            negative.add("action")
        if "kid" in pref_lower or "family" in pref_lower:
            negative.update({"family", "animation"})
        if "marvel" in pref_lower or "superhero" in pref_lower:
            negative.update({"marvel", "avengers", "superhero"})

    return positive, negative


def _derive_history_signals(history_df: pd.DataFrame) -> dict[str, set[str]]:
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

    return {
        "liked_genres": liked_genres,
        "liked_keywords": liked_keywords,
        "liked_cast": liked_cast,
        "liked_directors": liked_directors,
        "watched_roots": watched_roots,
    }


def _history_summary_text(history_df: pd.DataFrame) -> str:
    if history_df.empty:
        return "none"
    lines = []
    for row in history_df.head(4).itertuples():
        lines.append(
            f'- "{row.title}" | genres: {row.genres or "unknown"} | director: {row.director or "unknown"}'
            f' | cast: {", ".join(sorted(row.cast_set)[:4]) or "unknown"}'
            f' | keywords: {", ".join(sorted(row.keywords_set)[:6]) or "unknown"}'
        )
    return "\n".join(lines)


def _candidate_lines(candidates: pd.DataFrame, limit: int) -> str:
    lines = []
    for row in candidates.head(limit).itertuples():
        lines.append(
            f'- tmdb_id={int(row.tmdb_id)} | "{row.title}" ({int(row.year)})'
            f' | genres: {row.genres or "unknown"}'
            f' | keywords: {", ".join(sorted(row.keywords_set)[:6]) or "unknown"}'
            f' | director: {row.director or "unknown"}'
            f' | cast: {", ".join(sorted(row.cast_set)[:4]) or "unknown"}'
            f' | overview: {str(row.overview)[:220] or "unknown"}'
        )
    return "\n".join(lines)


def _prefilter_score(row: pd.Series, profile: dict[str, Any]) -> float:
    if int(row["tmdb_id"]) in profile["history_tmdb_ids"]:
        return -10_000.0
    if row["normalized_title"] in profile["history_titles"]:
        return -10_000.0

    score = 0.0
    positive_tokens = profile["raw_preference_tokens"] | profile["phrase_positive_tokens"]
    negative_tokens = profile["phrase_negative_tokens"]

    token_matches = positive_tokens.intersection(row["title_tokens"])
    token_matches |= positive_tokens.intersection(row["overview_tokens"])
    token_matches |= positive_tokens.intersection(row["tagline_tokens"])
    token_matches |= positive_tokens.intersection(row["keywords_tokens"])
    token_matches |= positive_tokens.intersection(row["genres_tokens"])
    score += 1.8 * len(token_matches)

    genre_matches = profile["liked_genres"].intersection(row["genres_set"])
    score += 2.0 * len(genre_matches)

    if row["title_root"] and row["title_root"] in profile["watched_roots"]:
        score -= 8.0

    blob = row["search_blob"]
    for token in negative_tokens:
        if token and token in blob:
            score -= 3.0

    score += min(float(row["vote_average"]) or 0.0, 10.0) * 0.2
    score += min(float(row["vote_count"]) or 0.0, 5000.0) / 8000.0
    return score


def _default_profile(preferences: str, history: tuple[tuple[int | None, str], ...], history_df: pd.DataFrame) -> dict[str, Any]:
    preference_tokens = _tokenize(preferences)
    phrase_positive_tokens, phrase_negative_tokens = _extract_phrase_hints(preferences)
    history_signals = _derive_history_signals(history_df)
    preferred_genres = sorted(history_signals["liked_genres"])[:4]
    avoid = sorted(phrase_negative_tokens)[:8]
    themes = sorted((preference_tokens | phrase_positive_tokens) - phrase_negative_tokens)[:10]

    return {
        "target_genres": preferred_genres,
        "preferred_tones": [],
        "preferred_themes": themes,
        "avoid": avoid,
        "history_signals": {
            "liked_genres": sorted(history_signals["liked_genres"])[:6],
            "liked_directors": sorted(history_signals["liked_directors"])[:4],
            "liked_cast": sorted(history_signals["liked_cast"])[:6],
        },
    }


def _parse_profile_payload(payload: dict[str, Any], default_profile: dict[str, Any]) -> dict[str, Any]:
    def _clean_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip().lower() for item in value if str(item).strip()]

    parsed = {
        "target_genres": _clean_list(payload.get("target_genres")) or default_profile["target_genres"],
        "preferred_tones": _clean_list(payload.get("preferred_tones")),
        "preferred_themes": _clean_list(payload.get("preferred_themes")) or default_profile["preferred_themes"],
        "avoid": _clean_list(payload.get("avoid")) or default_profile["avoid"],
        "history_signals": default_profile["history_signals"],
    }

    history_signals = payload.get("history_signals")
    if isinstance(history_signals, dict):
        parsed["history_signals"] = {
            "liked_genres": _clean_list(history_signals.get("liked_genres")) or default_profile["history_signals"]["liked_genres"],
            "liked_directors": _clean_list(history_signals.get("liked_directors")) or default_profile["history_signals"]["liked_directors"],
            "liked_cast": _clean_list(history_signals.get("liked_cast")) or default_profile["history_signals"]["liked_cast"],
        }
    return parsed


def _extract_user_profile(preferences: str, history: tuple[tuple[int | None, str], ...], candidates: pd.DataFrame) -> dict[str, Any]:
    history_df = _history_rows(history)
    default_profile = _default_profile(preferences, history, history_df)

    prompt = f"""You extract a structured movie-taste profile.

User preferences:
"{preferences}"

Watch history:
{_history_summary_text(history_df)}

Candidate sample:
{_candidate_lines(candidates, 20)}

Return ONLY valid JSON in this format:
{{
  "target_genres": ["genre"],
  "preferred_tones": ["tone"],
  "preferred_themes": ["theme"],
  "avoid": ["thing to avoid"],
  "history_signals": {{
    "liked_genres": ["genre"],
    "liked_directors": ["director"],
    "liked_cast": ["actor"]
  }}
}}

Rules:
- Keep lists short and specific.
- Include negative constraints if the user says "not", "avoid", or implies fatigue with a franchise/style.
- Use watch history as taste evidence.
- Do not invent movies outside the provided context."""

    try:
        response = _get_client().chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        payload = json.loads(response.message.content)
        return _parse_profile_payload(payload, default_profile)
    except Exception:
        logger.exception("Profile extraction failed; using deterministic profile fallback")
        return default_profile


def _build_base_profile(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    history_df = _history_rows(history)
    raw_preference_tokens = _tokenize(preferences)
    phrase_positive_tokens, phrase_negative_tokens = _extract_phrase_hints(preferences)
    history_signals = _derive_history_signals(history_df)

    return {
        "history_df": history_df,
        "history_titles": {_normalize_text(name) for _, name in history if _normalize_text(name)},
        "history_tmdb_ids": {tmdb_id for tmdb_id, _ in history if tmdb_id is not None},
        "raw_preference_tokens": raw_preference_tokens,
        "phrase_positive_tokens": phrase_positive_tokens,
        "phrase_negative_tokens": phrase_negative_tokens,
        **history_signals,
    }


def _prefilter_candidates(preferences: str, history: tuple[tuple[int | None, str], ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    profile = _build_base_profile(preferences, history)
    candidates = MOVIES.copy()
    candidates["prefilter_score"] = candidates.apply(_prefilter_score, axis=1, profile=profile)
    candidates = candidates.sort_values(["prefilter_score", "vote_average", "vote_count"], ascending=False)
    return candidates.head(PREFILTER_SIZE).copy(), profile


def _profile_guided_score(row: pd.Series, base_profile: dict[str, Any], user_profile: dict[str, Any]) -> float:
    if int(row["tmdb_id"]) in base_profile["history_tmdb_ids"]:
        return -10_000.0
    if row["normalized_title"] in base_profile["history_titles"]:
        return -10_000.0

    score = float(row["prefilter_score"])
    blob = row["search_blob"]

    for genre in user_profile["target_genres"]:
        if genre in row["genres_set"] or genre in blob:
            score += 3.0

    for tone in user_profile["preferred_tones"]:
        if tone in blob:
            score += 2.0

    for theme in user_profile["preferred_themes"]:
        if theme in blob:
            score += 1.8

    for avoid in user_profile["avoid"]:
        if avoid in blob:
            score -= 4.0

    liked_genres = set(user_profile["history_signals"]["liked_genres"]) | base_profile["liked_genres"]
    liked_directors = set(user_profile["history_signals"]["liked_directors"]) | base_profile["liked_directors"]
    liked_cast = set(user_profile["history_signals"]["liked_cast"]) | base_profile["liked_cast"]

    score += 2.0 * len(liked_genres.intersection(row["genres_set"]))
    score += 3.0 * len(liked_directors.intersection(row["director_set"]))
    score += 2.0 * len(liked_cast.intersection(row["cast_set"]))

    if row["title_root"] and row["title_root"] in base_profile["watched_roots"]:
        score -= 10.0

    return score


def _build_shortlist(preferences: str, history: tuple[tuple[int | None, str], ...]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prefiltered, base_profile = _prefilter_candidates(preferences, history)
    user_profile = _extract_user_profile(preferences, history, prefiltered)
    reranked = prefiltered.copy()
    reranked["score"] = reranked.apply(_profile_guided_score, axis=1, base_profile=base_profile, user_profile=user_profile)
    reranked = reranked.sort_values(["score", "vote_average", "vote_count"], ascending=False)

    shortlist = []
    for row in reranked.head(40).itertuples():
        if int(row.tmdb_id) in base_profile["history_tmdb_ids"] or row.normalized_title in base_profile["history_titles"]:
            continue
        shortlist.append(
            {
                "tmdb_id": int(row.tmdb_id),
                "title": row.title,
                "year": int(row.year),
                "genres": row.genres,
                "overview": str(row.overview)[:260],
                "keywords": ", ".join(sorted(row.keywords_set)[:MAX_KEYWORDS]),
                "director": row.director,
                "top_cast": ", ".join(sorted(row.cast_set)[:4]),
                "score": round(float(row.score), 2),
            }
        )
        if len(shortlist) >= SHORTLIST_SIZE:
            break

    return shortlist, user_profile, base_profile


def _build_selection_prompt(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    shortlist: list[dict[str, Any]],
    user_profile: dict[str, Any],
) -> str:
    history_text = ", ".join(f'"{name}"' for _, name in history if name) if history else "none"
    avoid_text = ", ".join(user_profile["avoid"]) if user_profile["avoid"] else "none"
    profile_text = json.dumps(user_profile, ensure_ascii=True)
    shortlist_text = "\n".join(
        (
            f'- tmdb_id={movie["tmdb_id"]} | "{movie["title"]}" ({movie["year"]})'
            f' | score={movie["score"]}'
            f' | genres: {movie["genres"]}'
            f' | director: {movie["director"] or "unknown"}'
            f' | cast: {movie["top_cast"] or "unknown"}'
            f' | keywords: {movie["keywords"] or "unknown"}'
            f' | overview: {movie["overview"] or "unknown"}'
        )
        for movie in shortlist
    )

    return f"""You are a movie recommendation assistant.

User preferences:
"{preferences}"

Watch history (already watched, do not recommend directly):
{history_text}

Structured taste profile:
{profile_text}

Important avoid directions:
{avoid_text}

Pick exactly one movie from the shortlist below that best matches the user's taste and avoids the listed negatives.

Shortlist:
{shortlist_text}

Respond with ONLY valid JSON:
{{
  "tmdb_id": <integer from the shortlist>,
  "description": "<under 500 chars; explain the match clearly>"
}}"""


def _fallback_description(movie: dict[str, Any], preferences: str, user_profile: dict[str, Any]) -> str:
    hook = str(movie["overview"]).strip()
    if hook:
        hook = hook[:220]
        last_space = hook.rfind(" ")
        if len(hook) == 220 and last_space > 120:
            hook = hook[:last_space]
        hook = hook.rstrip(".") + "."
    else:
        hook = f'{movie["title"]} is a strong fit.'

    reasons = []
    if user_profile["target_genres"]:
        reasons.append("genres like " + ", ".join(user_profile["target_genres"][:2]))
    if user_profile["preferred_themes"]:
        reasons.append("themes like " + ", ".join(user_profile["preferred_themes"][:2]))
    if not reasons and movie["genres"]:
        reasons.append(movie["genres"])

    reason_text = "It matches your taste for " + " and ".join(reasons) + "."
    description = f"{hook} {reason_text}"
    return description[:500]


def _choose_with_llm(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    shortlist: list[dict[str, Any]],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    prompt = _build_selection_prompt(preferences, history, shortlist, user_profile)
    response = _get_client().chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    payload = json.loads(response.message.content)
    valid_ids = {movie["tmdb_id"] for movie in shortlist}
    tmdb_id = int(payload.get("tmdb_id", -1))
    if tmdb_id not in valid_ids:
        raise ValueError(f"Model selected tmdb_id {tmdb_id}, which is outside the shortlist")
    payload["description"] = str(payload.get("description", ""))[:500]
    return payload


@lru_cache(maxsize=256)
def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    shortlist, user_profile, _base_profile = _build_shortlist(preferences, history)
    if not shortlist:
        raise ValueError("No candidate movies available")

    try:
        return _choose_with_llm(preferences, history, shortlist, user_profile)
    except Exception as exc:
        logger.warning(
            "Falling back to heuristic recommendation: preferences_len=%d history_count=%d shortlist_count=%d error=%s",
            len(preferences),
            len(history),
            len(shortlist),
            exc,
        )
        best = shortlist[0]
        return {
            "tmdb_id": best["tmdb_id"],
            "description": _fallback_description(best, preferences, user_profile),
        }


def get_recommendation(preferences: str, history: list[Any]) -> dict[str, Any]:
    normalized_preferences = " ".join(preferences.split())
    normalized_history = _normalize_history(history)
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
