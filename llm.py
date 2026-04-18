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
INTENT_LLM_TIMEOUT_SECONDS = 6.0
SELECTION_LLM_TIMEOUT_SECONDS = 12.0
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


def _default_intent(preferences: str) -> dict[str, Any]:
    return {
        "query_text": preferences,
        "must_have": [],
        "avoid": [],
        "tone": [],
    }


def _build_intent_prompt(preferences: str, history: tuple[tuple[int | None, str], ...]) -> str:
    watched_titles = [name for _, name in history if name][:5]
    watched_text = ", ".join(f'"{name}"' for name in watched_titles) if watched_titles else "none"
    return (
        "Turn this movie request into a compact retrieval intent.\n\n"
        f"User request: {preferences}\n"
        f"Already watched: {watched_text}\n\n"
        "Use history only to avoid repeats or overly similar picks. Do not assume watched movies were liked.\n"
        "Return raw JSON only in this format:\n"
        '{"query_text":"<short retrieval query>","must_have":["..."],"avoid":["..."],"tone":["..."]}\n'
        "Rules: keep query_text short and literal; make query_text positive-only retrieval language; put exclusions only in avoid; keep each list to at most 5 short items; only include things clearly supported by the request."
    )


def _sanitize_intent(payload: dict[str, Any], preferences: str) -> dict[str, Any]:
    def _clean_list(value: Any) -> list[str]:
        if isinstance(value, list):
            items = value
        elif value:
            items = [value]
        else:
            items = []
        cleaned = []
        for item in items:
            text = " ".join(str(item or "").split()).strip()
            if text and text.lower() not in {"movie", "movies", "something"}:
                cleaned.append(text)
        return cleaned[:5]

    query_text = " ".join(str(payload.get("query_text") or "").split()).strip() or preferences
    return {
        "query_text": query_text,
        "must_have": _clean_list(payload.get("must_have")),
        "avoid": _clean_list(payload.get("avoid")),
        "tone": _clean_list(payload.get("tone")),
    }


@lru_cache(maxsize=256)
def _extract_intent_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    prompt = _build_intent_prompt(preferences, history)
    try:
        response = _get_client(INTENT_LLM_TIMEOUT_SECONDS).chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        payload = _extract_json_object(response.message.content)
        return _sanitize_intent(payload, preferences)
    except Exception as exc:
        logger.warning("Falling back to default intent extraction: preferences_len=%d error=%s", len(preferences), exc)
        return _default_intent(preferences)


def _build_selection_prompt(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
) -> str:
    avoid_text = ", ".join(prompt_profile["avoid"][:4]) if prompt_profile["avoid"] else "none"
    focus_text = ", ".join(prompt_profile["preferred_themes"][:4]) if prompt_profile["preferred_themes"] else "none"
    history_hint = _build_history_exposure_hint(prompt_profile, history_exclusion_text)

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
            return "clear premise"
        trimmed = overview[:64].rstrip()
        last_space = trimmed.rfind(" ")
        if len(overview) > 64 and last_space > 28:
            trimmed = trimmed[:last_space]
        return trimmed

    def _fmt(movie: dict[str, Any]) -> str:
        return f'{movie["tmdb_id"]}: "{movie["title"]}" [{movie["genres"]}] - {_short_hook(movie)}'

    shortlist_text = "\n".join(_fmt(movie) for movie in shortlist)

    return (
        "Recommend one movie from this shortlist.\n\n"
        f"User request: {preferences}\n"
        f"Focus: {focus_text}\n"
        f"Avoid: {avoid_text}\n"
        f"Already watched: {history_hint}\n\n"
        "Candidates:\n"
        f"{shortlist_text}\n\n"
        "Return raw JSON only.\n"
        'Format: {"tmdb_id": <candidate id>, "title": "<exact shortlisted title>", "description": "<two short sentences, under 280 chars>"}\n'
        "Description rules: sentence 1 gives a concrete hook from the premise; sentence 2 says why it matches this request. "
        "Sound warm and direct, like a thoughtful friend. No generic hype, no critic voice, no trailer voice."
    )


def _fallback_description(movie: dict[str, Any], prompt_profile: dict[str, Any]) -> str:
    keywords = [part.strip() for part in str(movie.get("keywords") or "").split(",") if part.strip()]
    lead_keywords = ", ".join(keywords[:3])
    if lead_keywords:
        hook = f'{movie["title"]} leans into {lead_keywords} rather than vague spectacle.'
    else:
        overview = str(movie["overview"]).strip()
        if overview:
            trimmed = overview[:140]
            last_space = trimmed.rfind(" ")
            if len(overview) > 140 and last_space > 80:
                trimmed = trimmed[:last_space]
            hook = trimmed.rstrip(".") + "."
        else:
            hook = f'{movie["title"]} has a concrete, story-first sci-fi setup.'

    if prompt_profile["preferred_themes"]:
        reason = ", ".join(prompt_profile["preferred_themes"][:2])
        fit_line = f"If you want {reason} right now, this gets there through a specific premise instead of generic blockbuster noise."
    elif movie["genres"]:
        fit_line = f"If you want something in the {movie['genres']} lane, this is an easier sell because the premise is clear from the start."
    else:
        fit_line = "If you want something engaging without overthinking it, this gives you a clearer hook than a generic effects-first pick."

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

    tmdb_id = _resolve_selected_tmdb_id(selection_payload, shortlist)

    return {
        "tmdb_id": tmdb_id,
        "description": str(selection_payload.get("description", ""))[:500],
    }


@lru_cache(maxsize=256)
def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    intent = _extract_intent_cached(preferences, history)
    shortlist, prompt_profile, retrieval_profile = _build_shortlist(preferences, history, intent=intent)
    if not shortlist:
        raise ValueError("No candidate movies available")

    try:
        result = _choose_with_llm(
            preferences,
            shortlist,
            prompt_profile,
            retrieval_profile["history_exclusion_text"],
            timeout_seconds=SELECTION_LLM_TIMEOUT_SECONDS,
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
