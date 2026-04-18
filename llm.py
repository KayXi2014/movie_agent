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
from functools import lru_cache
from typing import Any

import ollama

from retrieval import (
    MOVIES,
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


LLM_STAGE_BUDGET_SECONDS = 18.0
MODEL = "gemma4:31b-cloud"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_client(timeout_seconds: float = LLM_STAGE_BUDGET_SECONDS) -> ollama.Client:
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


def _build_selection_prompt(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
) -> str:
    avoid_text = ", ".join(prompt_profile["avoid"][:4]) if prompt_profile["avoid"] else "none"
    themes_text = ", ".join(prompt_profile["preferred_themes"][:4]) if prompt_profile["preferred_themes"] else "none"
    history_hint = _build_history_exposure_hint(prompt_profile, history_exclusion_text)
    liked_director_tokens: set[str] = set()
    liked_cast_tokens: set[str] = set()

    def _short_hook(movie: dict[str, Any]) -> str:
        if movie["keywords"]:
            bits = [
                part.strip()
                for part in str(movie["keywords"]).split(",")
                if part.strip() and part.strip().lower() not in {"sequel", "prequel", "franchise", "part one", "part two"}
            ]
            if bits:
                return ", ".join(bits[:4])
        overview = str(movie["overview"] or "").strip()
        if not overview:
            return "strong fit"
        trimmed = overview[:72].rstrip()
        last_space = trimmed.rfind(" ")
        if len(overview) > 72 and last_space > 35:
            trimmed = trimmed[:last_space]
        return trimmed

    def _fmt(movie: dict[str, Any]) -> str:
        relevance_cues = []
        director = str(movie["director"] or "").strip()
        cast = str(movie["top_cast"] or "").strip()

        if director and any(d in director.lower() for d in liked_director_tokens):
            relevance_cues.append(f"same director lane: {director}")

        if cast and liked_cast_tokens:
            matched_cast = [name.strip() for name in cast.split(",") if name.strip().lower() in liked_cast_tokens]
            if matched_cast:
                relevance_cues.append("familiar cast: " + ", ".join(matched_cast[:2]))

        cue_text = f" | {'; '.join(relevance_cues)}" if relevance_cues else ""
        return f'{movie["tmdb_id"]}: "{movie["title"]}" [{movie["genres"]}] - {_short_hook(movie)}{cue_text}'

    shortlist_text = "\n".join(_fmt(movie) for movie in shortlist)

    return (
        "You are recommending one movie to a friend based on what they want right now.\n\n"
        f"User wants: {preferences}\n"
        f"Themes: {themes_text}\n"
        f"Avoid: {avoid_text}\n"
        f"Already watched: {history_hint}\n\n"
        "Candidates:\n"
        f"{shortlist_text}\n\n"
        "Decide fast. No reasoning. Write exactly two sentences. "
        "Sentence 1 should give one vivid, concrete hook about the movie. "
        "Sentence 2 should explain why it fits what the user wants right now. "
        "Sound warm, confident, and personal, like a thoughtful friend. "
        "Do not sound like a critic, trailer, or recommendation engine. "
        "Do not assume watched movies were liked; use history mainly to avoid repeats or stale picks. "
        "Avoid generic phrases like 'strong fit', 'lines up well', 'high-octane', 'stylish thriller', or 'visceral experience'.\n\n"
        'Return raw JSON only: {"tmdb_id": <candidate id>, "description": "<under 320 chars, warm, specific, and personal>"}'
    )


def _fallback_description(movie: dict[str, Any], prompt_profile: dict[str, Any]) -> str:
    hook = str(movie["overview"]).strip()
    if hook:
        hook = hook[:150]
        last_space = hook.rfind(" ")
        if len(hook) == 150 and last_space > 80:
            hook = hook[:last_space]
        hook = hook.rstrip(".") + "."
    else:
        hook = f'{movie["title"]} has a clear, easy-to-sell hook.'

    if prompt_profile["preferred_themes"]:
        reason = ", ".join(prompt_profile["preferred_themes"][:2])
        fit_line = f"If you want {reason} right now, this is an easy movie to sink into."
    elif movie["genres"]:
        fit_line = f"If you want something in the {movie['genres']} lane, this is a welcoming pick."
    else:
        fit_line = "If you want something engaging without overthinking it, this is a good pick."

    return f"{hook} {fit_line}"[:500]


def _build_history_exposure_hint(prompt_profile: dict[str, Any], history_exclusion_text: str) -> str:
    history_signals = prompt_profile.get("history_signals", {})
    bits = []
    liked_genres = history_signals.get("liked_genres", [])
    liked_directors = history_signals.get("liked_directors", [])
    liked_cast = history_signals.get("liked_cast", [])

    if liked_genres:
        bits.append("they have already seen some " + ", ".join(liked_genres[:2]) + " movies")
    if liked_directors:
        bits.append("they have watched work from " + ", ".join(liked_directors[:2]))
    if liked_cast:
        bits.append("they have already seen titles with " + ", ".join(liked_cast[:2]))

    if bits:
        return "; ".join(bits) + "; avoid repeating the same movie or something overly familiar unless the request clearly points there"
    return history_exclusion_text.replace("use history primarily to avoid re-recommending already watched titles", "avoid repeating the same movie or something too obviously repetitive")


def _choose_with_llm(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = _build_selection_prompt(preferences, shortlist, prompt_profile, history_exclusion_text)
    response = _get_client(timeout_seconds).chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    payload = _extract_json_object(response.message.content)
    selection_payload = payload.get("selection") if isinstance(payload.get("selection"), dict) else payload

    valid_ids = {movie["tmdb_id"] for movie in shortlist}
    tmdb_id = int(selection_payload.get("tmdb_id", -1))
    if tmdb_id not in valid_ids:
        raise ValueError(f"Model selected tmdb_id {tmdb_id}, which is outside the shortlist")

    return {
        "tmdb_id": tmdb_id,
        "description": str(selection_payload.get("description", ""))[:500],
    }


@lru_cache(maxsize=256)
def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    shortlist, prompt_profile, retrieval_profile = _build_shortlist(preferences, history)
    if not shortlist:
        raise ValueError("No candidate movies available")

    try:
        result = _choose_with_llm(
            preferences,
            shortlist,
            prompt_profile,
            retrieval_profile["history_exclusion_text"],
            timeout_seconds=LLM_STAGE_BUDGET_SECONDS,
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
