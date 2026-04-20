"""
Movie recommendation logic.

Runtime flow:
1. Local retrieval and reranking from `retrieval.py`
2. Single LLM call to pick from a compact shortlist
3. Deterministic fallback if the LLM step fails
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from typing import Any

import ollama

from retrieval import (
    MOVIES_BY_TMDB_ID,
    TOP_MOVIES,
    build_prompt_profile as _build_prompt_profile,
    build_retrieval_profile as _build_retrieval_profile,
    build_shortlist as _build_shortlist,
    history_rows as _history_rows,
    normalize_history as _normalize_history,
    normalize_history_item as _normalize_history_item,
    normalize_text as _normalize_text,
    title_root as _title_root,
    tokenize as _tokenize,
)


TOTAL_REQUEST_BUDGET_SECONDS = 20.0
FALLBACK_BUFFER_SECONDS = 1.0
LLM_TIMEOUT_SAFETY_MARGIN_SECONDS = 0.25
MODEL = "gemma4:31b-cloud"
logger = logging.getLogger(__name__)

try:
    from semantic_retrieval import warm_semantic_runtime as _warm_semantic_runtime
except Exception:
    _warm_semantic_runtime = None

if _warm_semantic_runtime is not None:
    try:
        _warm_semantic_runtime()
    except Exception as exc:
        logger.warning("Semantic runtime prewarm failed: %r", exc)


@lru_cache(maxsize=8)
def _get_client(timeout_seconds: float) -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=timeout_seconds,
    )


def _extract_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    # Parse the first JSON value that appears in the string.
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")

    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Could not parse JSON object from model response") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def _match_shortlist_title(value: Any, shortlist: list[dict[str, Any]]) -> int | None:
    target = _normalize_text(value)
    if not target:
        return None

    exact = { _normalize_text(movie["title"]): int(movie["tmdb_id"]) for movie in shortlist }
    if target in exact:
        return exact[target]

    partial_matches = [int(movie["tmdb_id"]) for movie in shortlist if _normalize_text(movie["title"]) in target or target in _normalize_text(movie["title"])]
    if len(partial_matches) == 1:
        return partial_matches[0]
    return None


def _resolve_selected_tmdb_id(selection_payload: dict[str, Any], shortlist: list[dict[str, Any]]) -> int:
    valid_ids = {int(movie["tmdb_id"]) for movie in shortlist}

    raw_tmdb_id = selection_payload.get("tmdb_id")
    if raw_tmdb_id not in (None, ""):
        try:
            tmdb_id = int(raw_tmdb_id)
        except (TypeError, ValueError):
            tmdb_id = -1
        if tmdb_id in valid_ids:
            return tmdb_id

    for key in ("title", "movie", "name", "choice"):
        matched = _match_shortlist_title(selection_payload.get(key), shortlist)
        if matched is not None:
            return matched

    description = str(selection_payload.get("description", "") or "")
    desc_matches = [int(movie["tmdb_id"]) for movie in shortlist if _normalize_text(movie["title"]) and _normalize_text(movie["title"]) in _normalize_text(description)]
    if len(desc_matches) == 1:
        return desc_matches[0]

    raise ValueError(f"Model did not return a usable shortlist selection: {selection_payload}")


def _build_selection_prompt(
    preferences: str,
    shortlist: list[dict[str, Any]],
    history_exclusion_text: str,
) -> str:
    history_hint = _build_history_exposure_hint(history_exclusion_text)

    def _trim_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        trimmed = text[:limit].rstrip()
        last_space = trimmed.rfind(" ")
        if len(text) > limit and last_space > max(20, limit // 2):
            trimmed = trimmed[:last_space]
        return trimmed.rstrip(" ,.;:")

    def _csv_head(value: Any, limit: int) -> str:
        items = [part.strip() for part in str(value or "").split(",") if part.strip()]
        return ", ".join(items[:limit])

    def _fmt(movie: dict[str, Any]) -> str:
        title = movie["title"]
        if movie.get("year"):
            title = f'{title} ({movie["year"]})'
        parts = [f'{movie["tmdb_id"]} | {title}']

        genres = _csv_head(movie["genres"], 3)
        if genres:
            parts.append(f"genres: {genres}")

        director = _trim_text(movie.get("director"), 48)
        if director:
            parts.append(f"director: {director}")

        cast = _csv_head(movie.get("top_cast"), 2)
        if cast:
            parts.append(f"cast: {cast}")

        keywords = _csv_head(movie.get("keywords"), 3)
        if keywords:
            parts.append(f"keywords: {keywords}")

        country = _csv_head(movie.get("production_countries"), 2)
        if country:
            parts.append(f"country: {country}")

        premise = _trim_text(movie.get("overview"), 72) or "clear premise"
        parts.append(f"premise: {premise}")

        return " | ".join(parts)

    shortlist_text = "\n".join(_fmt(movie) for movie in shortlist)
    sections = [
        "Choose the single best fit from this retrieved shortlist.",
        "Your job is to select first, then sell the choice.",
        f"Request: {preferences}",
    ]
    if history_hint != "none":
        sections.append(f"History: {history_hint}")
    sections.extend(
        [
            "",
            "Rules:",
            "- Do not retrieve or invent a different movie.",
            "- Use the user's request as the source of constraints and preferences.",
            "- Judge each movie only by the dataset-backed candidate details shown below.",
            "- If the user references a director or actor, treat matching director/cast details as an important clue.",
            "- If a year is shown, use it only when the request cares about recency or era.",
            "- Do not introduce qualities that are not supported by the request or the candidate details.",
            'Return JSON only: {"tmdb_id": <id>, "title": "<exact title>", "description": "<2-3 sentences, under 500 chars>"}',
            "",
            "Candidates:",
            shortlist_text,
            "",
            "Description: be personal and persuasive, but specific to this user.",
            "Sentence 1 gives a vivid hook.",
            "Sentence 2 and 3 explain why this fits the user's request right now.",
            "Address the user directly with personal persuasive tone. No generic hype or critic voice.",
        ]
    )
    return "\n".join(sections)


def _fallback_description(movie: dict[str, Any], prompt_profile: dict[str, Any]) -> str:
    keywords = [part.strip() for part in str(movie.get("keywords") or "").split(",") if part.strip()]
    lead_keywords = ", ".join(keywords[:4])
    if lead_keywords:
        hook = f'{movie["title"]} throws you into {lead_keywords}, with a premise that feels concrete right away.'
    else:
        overview = str(movie["overview"]).strip()
        if overview:
            trimmed = overview[:180]
            last_space = trimmed.rfind(" ")
            if len(overview) > 180 and last_space > 100:
                trimmed = trimmed[:last_space]
            hook = trimmed.rstrip(".") + "."
        else:
            hook = f'{movie["title"]} has a clear, story-first setup instead of a vague effects reel.'

    tones = prompt_profile.get("tone", [])
    if tones:
        reason = ", ".join(tones[:2])
        fit_line = f"If you want something {reason} right now, this lands better because the appeal comes from the story pressure and mood, not empty spectacle."
    elif prompt_profile["target_genres"]:
        target = ", ".join(prompt_profile["target_genres"][:2])
        fit_line = f"If you want something in the {target} lane, this is a strong bet because the hook is clear and the payoff is easy to picture."
    elif movie["genres"]:
        fit_line = f"If you want something in the {movie['genres']} lane, this is a strong bet because the hook is clear and the payoff is easy to picture."
    else:
        fit_line = "If you want something engaging without overthinking it, this gives you a clearer hook than a generic effects-first pick."

    return f"{hook} {fit_line}"[:500]


def _enrich_shortlist(shortlist_refs: list[dict[str, Any]], include_year: bool = False) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for ref in shortlist_refs:
        tmdb_id = int(ref["tmdb_id"])
        if tmdb_id not in MOVIES_BY_TMDB_ID.index:
            continue
        row = MOVIES_BY_TMDB_ID.loc[tmdb_id]
        if hasattr(row, "iloc") and not isinstance(row, dict) and getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        movie = {
            "tmdb_id": tmdb_id,
            "title": str(row["title"]),
            "genres": str(row["genres"]),
            "overview": str(row["overview"])[:220],
            "director": str(row.get("director", "")),
            "top_cast": str(row.get("top_cast", "")),
            "keywords": ", ".join(sorted(row["keywords_set"])[:5]),
            "production_countries": str(row.get("production_countries", "")),
        }
        if include_year and str(row.get("year", "")).strip():
            movie["year"] = int(row["year"])
        enriched.append(movie)
    return enriched


def _build_history_exposure_hint(history_exclusion_text: str) -> str:
    if history_exclusion_text == "none":
        return "none"
    return "avoid repeats and near-duplicates from watch history"


def _choose_with_llm(
    preferences: str,
    shortlist: list[dict[str, Any]],
    history_exclusion_text: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = _build_selection_prompt(preferences, shortlist, history_exclusion_text)
    response = _get_client(timeout_seconds).chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    payload = _extract_json_object(response.message.content)
    selection_payload = payload.get("selection") if isinstance(payload.get("selection"), dict) else payload

    tmdb_id = _resolve_selected_tmdb_id(selection_payload, shortlist)

    return {
        "tmdb_id": tmdb_id,
        "description": str(selection_payload.get("description", ""))[:500],
    }


def _remaining_llm_timeout_seconds(started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    remaining = TOTAL_REQUEST_BUDGET_SECONDS - elapsed - FALLBACK_BUFFER_SECONDS - LLM_TIMEOUT_SAFETY_MARGIN_SECONDS
    return max(0.0, remaining)


@lru_cache(maxsize=256)
def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    started_at = time.perf_counter()
    shortlist_refs, prompt_profile, retrieval_profile = _build_shortlist(preferences, history)
    shortlist = _enrich_shortlist(shortlist_refs, include_year=bool(prompt_profile.get("year_relevant")))
    if not shortlist:
        raise ValueError("No candidate movies available")

    try:
        llm_timeout_seconds = _remaining_llm_timeout_seconds(started_at)
        if llm_timeout_seconds <= 0.5:
            raise TimeoutError("Not enough request budget left for LLM selection")
        result = _choose_with_llm(
            preferences,
            shortlist,
            retrieval_profile["history_exclusion_text"],
            timeout_seconds=llm_timeout_seconds,
        )
        result["used_llm"] = True
        return result
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
            "description": _fallback_description(best, prompt_profile),
            "used_llm": False,
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
