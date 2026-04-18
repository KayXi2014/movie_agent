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


def _build_selection_prompt(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
) -> str:
    avoid_text = ", ".join(prompt_profile["avoid"][:4]) if prompt_profile["avoid"] else "none"
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
        genres = ", ".join(part.strip() for part in str(movie["genres"]).split(",")[:2] if part.strip()) or "movie"
        return f'{movie["tmdb_id"]} | {movie["title"]} | {genres} | {_short_hook(movie)}'

    shortlist_text = "\n".join(_fmt(movie) for movie in shortlist)

    return (
        "Recommend one movie from this shortlist.\n\n"
        f"User request: {preferences}\n"
        f"Avoid: {avoid_text}\n"
        f"Already watched: {history_hint}\n\n"
        "Candidates:\n"
        f"{shortlist_text}\n\n"
        "Pick the best fit and write like a sharp, thoughtful friend.\n"
        "Be concrete, direct, and persuasive.\n"
        "Only use support that is visible in the candidate line. Do not invent soundtrack, score, acclaim, awards, or adaptation details beyond what is shown.\n"
        "Return raw JSON only.\n"
        'Format: {"tmdb_id": <candidate id>, "title": "<exact shortlisted title>", "description": "<2 or 3 sentences, under 500 chars>"}\n'
        "Description rules: sentence 1 gives a vivid premise hook; sentence 2 explains why it matches this request; sentence 3 is optional. "
        "Address the user directly when natural. No generic hype, critic voice, or trailer voice."
    )


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

    tones = set(prompt_profile.get("preferred_tones", []))
    if {"bad", "fun", "laugh"}.intersection(tones):
        fit_line = "If you want something loose, silly, and easy to laugh at with friends, this is the kind of pick that works better once everyone starts leaning into the chaos."
    elif prompt_profile["preferred_themes"]:
        reason = ", ".join(prompt_profile["preferred_themes"][:3])
        fit_line = f"If you want {reason} right now, this is easier to buy into because the appeal comes from the idea and the pressure of the situation, not empty spectacle."
    elif movie["genres"]:
        fit_line = f"If you want something in the {movie['genres']} lane, this is a strong bet because the hook is clear and the payoff is easy to picture."
    else:
        fit_line = "If you want something engaging without overthinking it, this gives you a clearer hook than a generic effects-first pick."

    return f"{hook} {fit_line}"[:500]


def _enrich_shortlist(shortlist_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for ref in shortlist_refs:
        tmdb_id = int(ref["tmdb_id"])
        if tmdb_id not in MOVIES_BY_TMDB_ID.index:
            continue
        row = MOVIES_BY_TMDB_ID.loc[tmdb_id]
        if hasattr(row, "iloc") and not isinstance(row, dict) and getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        enriched.append(
            {
                "tmdb_id": tmdb_id,
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "overview": str(row["overview"])[:220],
                "keywords": ", ".join(sorted(row["keywords_set"])[:5]),
                "score": ref.get("score", 0.0),
                "semantic_score": ref.get("semantic_score", 0.0),
                "fts_score": ref.get("fts_score", 0.0),
            }
        )
    return enriched


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
    shortlist_refs, prompt_profile, retrieval_profile = _build_shortlist(preferences, history)
    shortlist = _enrich_shortlist(shortlist_refs)
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
