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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from typing import Any

import ollama

from retrieval import (
    MOVIES_BY_TMDB_ID,
    TOP_MOVIES,
    build_prompt_profile as _build_prompt_profile,
    build_retrieval_profile as _build_retrieval_profile,
    build_shortlist as _build_shortlist,
    GENRE_ALIASES,
    history_rows as _history_rows,
    normalize_history as _normalize_history,
    normalize_history_item as _normalize_history_item,
    normalize_text as _normalize_text,
    title_root as _title_root,
    tokenize as _tokenize,
)


TOTAL_REQUEST_BUDGET_SECONDS = 20.0
FALLBACK_BUFFER_SECONDS = 1.0
LLM_TIMEOUT_SAFETY_MARGIN_SECONDS = 1.0
MODEL = "gemma4:31b-cloud"
ENABLE_LLM_INTENT = os.getenv("ENABLE_LLM_INTENT", "1") == "1"
INTENT_LLM_TIMEOUT_S = float(os.getenv("INTENT_LLM_TIMEOUT_S", "2.5"))  # Reduced from 3.5s to preserve budget
logger = logging.getLogger(__name__)
INTENT_EXECUTOR = ThreadPoolExecutor(max_workers=1)
KNOWN_GENRES = tuple(sorted({genre.strip() for value in TOP_MOVIES["genres"] for genre in str(value or "").split(",") if genre.strip()}))

# Track active requests for potential cancellation
_active_request_cancel = {"cancel": False}


@lru_cache(maxsize=8)
def _get_client(timeout_seconds: float) -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=timeout_seconds,
    )


def cancel_active_requests() -> None:
    """Signal active LLM requests to abort. Call when budget is exhausted."""
    _active_request_cancel["cancel"] = True


def reset_request_cancellation() -> None:
    """Reset cancellation flag for new request cycle."""
    _active_request_cancel["cancel"] = False


def _check_cancellation() -> None:
    """Raise if cancellation was requested."""
    if _active_request_cancel.get("cancel"):
        raise TimeoutError("Request cancelled due to budget exhaustion")


def _build_intent_prompt(preferences: str) -> str:
    genres_text = ", ".join(KNOWN_GENRES)
    return (
        "JSON only. Extract movie search hints.\n"
        'Schema: {"genres":[],"avoid":[],"avoid_tone":[],"keyword_hints":[],"release_year":null,"runtime":null,"tone":[],"tone_modifiers":[],"quality_preference":false,"language":[]}\n'
        f"Genres: {genres_text}\n"
        "avoid: negative requirements (e.g., 'no romance' → romance, 'minimal violence' → gore, 'not cheesy' → cliché)\n"
        "avoid_tone: moods to exclude (e.g., 'not depressing' → depressing, 'not slow-paced' → slow-burn)\n"
        "keyword_hints: subgenre/theme terms (courtroom→trial,lawyer; biopic→biography,true story; superhero→comic,powers)\n"
        "tone_modifiers: subtype qualifiers like psychological, supernatural, gore-heavy, suspense, thriller (for disambiguation)\n"
        "release_year: {min,max,label} for year constraints (e.g., '80s movie' → {min:1980,max:1989})\n"
        "runtime: {max,label} for length constraints\n"
        "language: target language/country (e.g., 'Spanish film' → Spanish, 'Korean thriller' → Korean)\n"
        'Ex: "courtroom drama not too dark" -> {"genres":["Drama"],"avoid_tone":["dark"],"keyword_hints":["courtroom","trial","legal"]}\n'
        f"Request: {preferences}"
    )


def _intent_llm_options() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 5,
        "top_p": 0.3,
        "num_predict": 120,  # Reduced from 180 - intent JSON is compact
    }


def _final_llm_options() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 5,
        "top_p": 0.4,
        "num_predict": 150,  # Reduced - selection JSON is compact
    }


def _description_llm_options() -> dict[str, Any]:
    return {
        "temperature": 0.3,
        "top_k": 15,
        "top_p": 0.7,
        "num_predict": 180,  # Reduced from 220 - description is capped at 500 chars anyway
    }


def _get_intent_override(preferences: str) -> dict[str, Any]:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key or not ENABLE_LLM_INTENT:
        return {}
    try:
        _check_cancellation()
        response = _get_client(INTENT_LLM_TIMEOUT_S).chat(
            model=MODEL,
            messages=[{"role": "user", "content": _build_intent_prompt(preferences)}],
            format="json",
            options=_intent_llm_options(),
        )
        payload = _extract_json_object(response.message.content)
    except Exception:
        return {}

    genres = [
        genre
        for item in payload.get("genres", [])
        if (genre := next((known for known in KNOWN_GENRES if _normalize_text(known) == _normalize_text(item)), None))
    ]
    avoid = [" ".join(str(item or "").split()).strip() for item in payload.get("avoid", []) if str(item or "").strip()]
    avoid_tone = [" ".join(str(item or "").split()).strip() for item in payload.get("avoid_tone", []) if str(item or "").strip()]
    named_people = [" ".join(str(item or "").split()).strip() for item in payload.get("named_people", []) if str(item or "").strip()]
    language = [" ".join(str(item or "").split()).strip() for item in payload.get("language", []) if str(item or "").strip()]
    tone_modifiers = [" ".join(str(item or "").split()).strip() for item in payload.get("tone_modifiers", []) if str(item or "").strip()]
    result: dict[str, Any] = {}
    if genres:
        result["genres"] = genres[:3]
    if avoid:
        result["avoid"] = avoid[:3]
    if avoid_tone:
        result["avoid_tone"] = avoid_tone[:3]
    if named_people:
        result["named_people"] = named_people[:3]
    if language:
        result["language"] = language[:3]
    if tone_modifiers:
        result["tone_modifiers"] = tone_modifiers[:4]
    release_year = payload.get("release_year")
    if isinstance(release_year, dict):
        try:
            min_year = int(release_year.get("min", release_year.get("min_year", 0)))
            max_year = int(release_year.get("max", release_year.get("max_year", 9999)))
        except (TypeError, ValueError):
            min_year = 0
            max_year = -1
        if min_year > 0 and max_year >= min_year:
            result["release_year"] = {
                "min": min_year,
                "max": max_year,
                "label": str(release_year.get("label") or f"{min_year}-{max_year}")[:40],
            }
    setting_period = " ".join(str(payload.get("setting_period", "") or "").split()).strip()
    if setting_period:
        result["setting_period"] = setting_period[:80]
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        parsed_runtime: dict[str, Any] = {}
        for source_key, target_key in (("min", "min_runtime"), ("min_runtime", "min_runtime"), ("max", "max_runtime"), ("max_runtime", "max_runtime")):
            if source_key not in runtime:
                continue
            try:
                parsed_runtime[target_key] = int(float(runtime[source_key]))
            except (TypeError, ValueError):
                continue
        short_requested = bool(runtime.get("short_requested", False))
    
    # Map tone_modifiers to tone for better disambiguation
    if tone_modifiers and not result.get("tone"):
        result["tone"] = tone_modifiers[:4]
    
        if short_requested and "max_runtime" not in parsed_runtime:
            parsed_runtime["max_runtime"] = 110
        if "min_runtime" in parsed_runtime or "max_runtime" in parsed_runtime:
            parsed_runtime["label"] = str(runtime.get("label") or "runtime request")[:40]
            parsed_runtime["short_requested"] = short_requested
            result["runtime"] = parsed_runtime
    tone = [" ".join(str(item or "").split()).strip() for item in payload.get("tone", []) if str(item or "").strip()]
    if tone:
        result["tone"] = tone[:4]
    keyword_hints = [
        " ".join(str(item or "").split()).strip()
        for item in payload.get("keyword_hints", [])
        if str(item or "").strip()
    ]
    if keyword_hints:
        result["keyword_hints"] = keyword_hints[:5]
    if isinstance(payload.get("quality_preference"), bool):
        result["quality_preference"] = bool(payload["quality_preference"])
    country_or_language = [
        " ".join(str(item or "").split()).strip()
        for item in payload.get("country_or_language", [])
        if str(item or "").strip()
    ]
    if country_or_language:
        result["country_or_language"] = country_or_language[:3]
    return result


def _extract_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Empty model response")

    # Log very short responses for debugging; they often indicate an upstream failure.
    if len(raw) < 50:
        logger.debug("Model returned short response (%d chars): %s", len(raw), raw)

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

    def _find_complete_object(text_value: str) -> str | None:
        start_index = -1
        depth = 0
        in_string = False
        escaped = False
        candidate: str | None = None

        for index, char in enumerate(text_value):
            if start_index == -1:
                if char == "{":
                    start_index = index
                    depth = 1
                continue

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text_value[start_index : index + 1]
                    start_index = -1

        return candidate

    start = raw.find("{")
    if start < 0:
        logger.warning("JSON parsing failed - no { found in response: %s", raw[:200])
        raise ValueError("No JSON object found in response")

    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        logger.warning("JSON parsing failed - invalid JSON at position %d: %s", start, raw[start:start + 200])
        # Try to recover the last complete JSON object in the text.
        complete_object = _find_complete_object(raw)
        if complete_object:
            logger.info("Retrying JSON extraction from recovered complete object")
            try:
                parsed = json.loads(complete_object)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
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
    force_include_director: bool = False,
    force_include_cast: bool = False,
    force_include_country: bool = False,
    constraint_note: str = "",
    setting_period: str = "",
) -> str:
    # history_exclusion_text is handled by retrieval filtering, not needed in prompt
    normalized_preferences = _normalize_text(preferences)
    preference_tokens = set(_tokenize(preferences))

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

    def _query_mentions_any(items: list[str]) -> bool:
        return any(_normalize_text(item) in normalized_preferences for item in items if item)

    def _quality_signal(movie: dict[str, Any]) -> str:
        try:
            rating = float(movie.get("vote_average", 0.0))
            votes = int(float(movie.get("vote_count", 0)))
        except (TypeError, ValueError):
            rating = 0.0
            votes = 0
        if rating >= 8.0 and votes >= 100_000:
            return "very strong"
        if rating >= 7.2 and votes >= 25_000:
            return "strong"
        if rating >= 6.2 and votes >= 1_000:
            return "acceptable"
        if rating > 0.0 and votes > 0:
            return "limited"
        return "unknown"

    include_director = force_include_director or "director" in preference_tokens or "filmmaker" in preference_tokens
    include_cast = force_include_cast or bool({"actor", "actors", "actress", "actresses", "cast", "star", "stars", "starring"} & preference_tokens)
    include_country = force_include_country
    include_quality = True

    if not include_director:
        include_director = _query_mentions_any([movie.get("director", "") for movie in shortlist])
    if not include_cast:
        cast_names: list[str] = []
        for movie in shortlist:
            cast_names.extend(part.strip() for part in str(movie.get("top_cast", "")).split(",") if part.strip())
        include_cast = _query_mentions_any(cast_names)
    if not include_country:
        countries: list[str] = []
        for movie in shortlist:
            countries.extend(part.strip() for part in str(movie.get("production_countries", "")).split(",") if part.strip())
        include_country = _query_mentions_any(countries)

    def _fmt(movie: dict[str, Any]) -> str:
        title = movie["title"]
        if movie.get("year"):
            title = f'{title} ({movie["year"]})'
        parts = [f'{movie["tmdb_id"]} | {title}']

        genres = _csv_head(movie["genres"], 2)
        if genres:
            parts.append(f"genres: {genres}")

        director = _trim_text(movie.get("director"), 40)
        if include_director and director:
            parts.append(f"director: {director}")

        cast = _csv_head(movie.get("top_cast"), 1)
        if include_cast and cast:
            parts.append(f"cast: {cast}")

        keywords = _csv_head(movie.get("keywords"), 5)
        if keywords:
            parts.append(f"keywords: {keywords}")

        country = _csv_head(movie.get("production_countries"), 2)
        if include_country and country:
            parts.append(f"country: {country}")

        if include_quality:
            parts.append(f"quality signal: {_quality_signal(movie)}")

        premise = _trim_text(movie.get("overview"), 56) or "clear premise"
        parts.append(f"premise: {premise}")

        return " | ".join(parts)

    shortlist_text = "\n".join(_fmt(movie) for movie in shortlist)
    sections = [
        f"Pick one movie for: {preferences}",
    ]
    if constraint_note:
        sections.append(f"Note: {constraint_note}")
    if setting_period:
        sections.append(f"Setting: {setting_period}")
    sections.extend(
        [
            "",
            "Rules: Pick from candidates only. Honor constraints. Quality signal is tie-breaker.",
            'Return ONLY JSON. No markdown, no commentary, no code fences.',
            'Return JSON: {"tmdb_id": <id>, "description": "<one short paragraph, under 220 chars>"}',
            "Description style: concise, direct, and specific. Use at most 1-2 sentences. Never explain your reasoning outside the JSON.",
            "",
            "Candidates:",
            shortlist_text,
        ]
    )
    return "\n".join(sections)


def _build_description_only_prompt(
    preferences: str,
    movie: dict[str, Any],
    history_exclusion_text: str,
    setting_period: str = "",
) -> str:
    # history_exclusion_text handled by retrieval filtering, not needed in description prompt

    def _csv_head(value: Any, limit: int) -> str:
        items = [part.strip() for part in str(value or "").split(",") if part.strip()]
        return ", ".join(items[:limit])

    title = movie["title"]
    if movie.get("year"):
        title = f'{title} ({movie["year"]})'

    details = [f'{movie["tmdb_id"]} | {title}']
    genres = _csv_head(movie.get("genres"), 2)
    if genres:
        details.append(f"genres: {genres}")
    director = str(movie.get("director", "")).strip()
    if director:
        details.append(f"director: {director}")
    cast = _csv_head(movie.get("top_cast"), 1)
    if cast:
        details.append(f"cast: {cast}")
    premise = str(movie.get("overview", "")).strip()
    if premise:
        details.append(f"premise: {premise[:120]}")

    sections = [
        "Write a compelling recommendation for this movie.",
        f"User wants: {preferences}",
    ]
    if setting_period:
        sections.append(f"Setting preference: {setting_period}")
    sections.extend(
        [
            f"Movie: {' | '.join(details)}",
            'Return JSON: {"description": "<2-3 persuasive sentences, under 500 chars>"}',
            "Style: Write like you're recommending this to a friend - warm, genuine, and specific. Use 'you' and 'your' to make it personal. Hook them with plot details, emotional beats, or the unique vibe. Vary your phrasing. Skip generic praise and rating numbers. If the movie doesn't perfectly match their request, acknowledge that honestly and explain why it's still worth watching.",
        ]
    )
    return "\n".join(sections)


def _route_shortlist(shortlist: list[dict[str, Any]], route: str) -> list[dict[str, Any]]:
    if route == "judge_5":
        return shortlist[:5]
    if route == "judge_8":
        return shortlist[:8]
    return shortlist[:1]


def _fallback_description(movie: dict[str, Any], prompt_profile: dict[str, Any]) -> str:
    raw_keywords = [part.strip() for part in str(movie.get("keywords") or "").split(",") if part.strip()]
    keywords = [kw for kw in raw_keywords if "based on" not in kw.lower()]
    if not keywords:
        keywords = raw_keywords
    overview = str(movie.get("overview") or "").strip()
    
    # Check if this is a no-preference request (very few signals in prompt_profile)
    is_no_preference = (
        not prompt_profile.get("target_genres")
        and not prompt_profile.get("tone")
        and not prompt_profile.get("avoid")
        and not prompt_profile.get("year_relevant")
        and not prompt_profile.get("setting_period")
        and not prompt_profile.get("quality_preference")
    )

    def _trim_sentence(text: str, limit: int = 180) -> str:
        trimmed = text[:limit].rstrip()
        abbreviations = ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "L.A.", "U.S.", "U.K.")
        sentence_break = -1
        for marker in (". ", "! ", "? "):
            search_from = 0
            while True:
                idx = trimmed.find(marker, search_from)
                if idx == -1:
                    break
                prefix = trimmed[max(0, idx - 5) : idx + 1]
                if not prefix.endswith(abbreviations):
                    sentence_break = idx
                    break
                search_from = idx + 1
            if sentence_break > 40:
                break
        if sentence_break > 40:
            trimmed = trimmed[: sentence_break + 1]
            return trimmed.rstrip(" ,.;:") + "."
        if trimmed.endswith((".", "!", "?")):
            return trimmed.rstrip(" ,.;:") + "."
        last_space = trimmed.rfind(" ")
        if len(text) > limit and last_space > max(20, limit // 2):
            trimmed = trimmed[:last_space]
        return trimmed.rstrip(" ,.;:") + "."

    def _natural_list(items: list[str]) -> str:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _appeal_phrase() -> str:
        if keywords:
            return _natural_list(keywords[:3])
        genres = [part.strip().lower() for part in str(movie.get("genres") or "").split(",") if part.strip()]
        return _natural_list(genres[:2])

    def _request_phrase() -> str:
        tones = prompt_profile.get("tone", [])
        if tones:
            return _natural_list([str(item) for item in tones[:2]])
        if prompt_profile.get("setting_period"):
            return str(prompt_profile["setting_period"])
        if prompt_profile.get("target_genres"):
            return _natural_list([str(item) for item in prompt_profile["target_genres"][:2]])
        return ""

    def _article_for(text: str) -> str:
        return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"

    def _quality_phrase() -> str:
        try:
            rating = float(movie.get("vote_average", 0.0) or 0.0)
            votes = int(float(movie.get("vote_count", 0) or 0))
        except (TypeError, ValueError):
            rating = 0.0
            votes = 0
        if rating >= 8.0 and votes >= 100_000:
            return "It’s one of those rare films that lives up to the hype."
        if rating >= 8.0 and votes >= 10_000:
            return "People genuinely love this one, and I think you will too."
        if rating >= 7.2 and votes >= 25_000:
            return "It’s a crowd favorite for good reason."
        if rating >= 7.2 and votes >= 1_000:
            return "Those who’ve seen it tend to really like it."
        if rating > 0.0 and votes > 0:
            return "It’s more of a hidden gem than a blockbuster, but sometimes that’s exactly what you want."
        return ""

    title = movie["title"]
    movie_genres = [part.strip().lower() for part in str(movie.get("genres", "")).split(",") if part.strip()]
    request_phrase = _request_phrase()
    appeal_phrase = _appeal_phrase()

    # Check for constraint mismatches to acknowledge
    year_unavailable = prompt_profile.get("year_constraint_unavailable", False)
    year_label = ""
    if year_unavailable and prompt_profile.get("year_constraint"):
        year_label = prompt_profile["year_constraint"].get("label", "")

    # Check if requested genres don’t match movie genres
    target_genres = prompt_profile.get("target_genres", [])
    target_genres_lower = {str(g).lower() for g in target_genres}
    genre_mismatch = bool(target_genres_lower and not (target_genres_lower & set(movie_genres)))

    # Build the hook - vary the structure based on what info we have
    trimmed_overview = _trim_sentence(overview) if overview else ""

    # Helper to explain what the movie DOES offer
    def _movie_strengths() -> str:
        strengths = []
        if movie_genres:
            strengths.append(_natural_list(movie_genres[:2]))
        if keywords:
            strengths.append(f"{_natural_list(keywords[:2])} themes")
        if strengths:
            return strengths[0]
        return "a fresh take"

    if year_unavailable and year_label:
        # Acknowledge we couldn’t find movies from the requested era
        if overview:
            hook = f"I don’t have {year_label} films in my catalog, but {title} captures a similar spirit. {trimmed_overview}"
        else:
            hook = f"I don’t have {year_label} films available, but {title} has that classic feel you’re looking for."
    elif genre_mismatch:
        # Acknowledge genre mismatch but explain why it’s still good
        requested = _natural_list(list(target_genres_lower)[:2])
        actual = _natural_list(movie_genres[:2])
        if overview:
            hook = f"Pure {requested} options are limited right now, but {title} blends {actual} in a way you might enjoy. {trimmed_overview}"
        else:
            hook = f"Not a straight {requested} pick, but {title} brings {_movie_strengths()} that could hit the same spot."
    elif overview:
        # Vary the hook structure using a simple hash of the title for determinism
        variant = sum(ord(c) for c in title) % 4
        if is_no_preference:
            # Special handling for open-ended requests
            appeal = appeal_phrase or "crowd favorite"
            hook = f"Since you're open to anything, here's {_article_for(appeal)} {appeal}: {title}. {trimmed_overview}"
        elif variant == 0 and request_phrase:
            hook = f"For your {request_phrase} mood, check out {title}. {trimmed_overview}"
        elif variant == 1 and appeal_phrase:
            hook = f"{title} brings the {appeal_phrase} vibes. {trimmed_overview}"
        elif variant == 2:
            hook = f"You might love {title}. {trimmed_overview}"
        else:
            hook = f"Here’s one: {title}. {trimmed_overview}"
    elif keywords:
        hook = f"{title} has that {appeal_phrase} energy."
    else:
        hook = f"Go with {title}."

    wants_quality_context = bool(prompt_profile.get("quality_preference") or prompt_profile.get("rating_constraint"))
    support_line = _quality_phrase() if wants_quality_context else ""

    # Build the fit line - explain why this matches, with variety
    tones = prompt_profile.get("tone", [])
    movie_genres = [part.strip().lower() for part in str(movie.get("genres", "")).split(",") if part.strip()]

    # Use title hash for variety here too
    fit_variant = sum(ord(c) for c in title) % 3

    if tones:
        reason = _natural_list([str(t) for t in tones[:2]])
        fit_phrases = [
            f"It nails that {reason} feeling.",
            f"Perfect when you want something {reason}.",
            f"This one really delivers on the {reason} front.",
        ]
        fit_line = fit_phrases[fit_variant]
    elif prompt_profile.get("setting_period"):
        setting = str(prompt_profile["setting_period"])
        fit_line = f"Great for that {setting} atmosphere."
    elif prompt_profile.get("target_genres"):
        target = _natural_list([str(item) for item in prompt_profile["target_genres"][:2]])
        fit_phrases = [
            f"A solid choice for {target}.",
            f"One of the better {target} options out there.",
            f"If you want {target}, this delivers.",
        ]
        fit_line = fit_phrases[fit_variant]
    elif movie_genres:
        genres = _natural_list(movie_genres[:2])
        fit_line = f"A great pick for {_article_for(genres)} {genres} night."
    else:
        fit_line = ""

    parts = [hook]
    if support_line:
        parts.append(support_line)
    parts.append(fit_line)
    description = " ".join(parts)
    if len(description) <= 500:
        return description
    clipped = description[:500].rstrip()
    sentence_end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if sentence_end >= 260:
        return clipped[: sentence_end + 1]
    return clipped.rsplit(" ", 1)[0].rstrip(" ,.;:") + "."


def _safe_fallback_movie(shortlist: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best movie from shortlist, balancing relevance (position) with quality.

    Strategy: Among top candidates, prefer higher-ranked (more relevant) movies
    unless a lower-ranked movie has significantly better quality.
    """
    def _quality(movie: dict[str, Any]) -> tuple[float, int]:
        try:
            rating = float(movie.get("vote_average", 0.0) or 0.0)
            votes = int(float(movie.get("vote_count", 0) or 0))
        except (TypeError, ValueError):
            return 0.0, 0
        return rating, votes

    def _is_good_quality(movie: dict[str, Any]) -> bool:
        rating, votes = _quality(movie)
        return rating >= 6.5 and votes >= 500

    viable = [
        movie
        for movie in shortlist
        if not bool(movie.get("avoid_hit", False))
        and bool(movie.get("genre_match", True))
        and bool(movie.get("person_match", True))
        and bool(movie.get("year_match", True))
        and bool(movie.get("runtime_match", True))
        and bool(movie.get("seed_match", True))
    ]
    if not viable:
        viable = [movie for movie in shortlist if not bool(movie.get("avoid_hit", False))] or shortlist

    # Prefer top 3 candidates (most relevant) if any have good quality
    # This respects retrieval ranking rather than always picking highest-rated
    top_candidates = viable[:3]
    good_top = [m for m in top_candidates if _is_good_quality(m)]
    if good_top:
        # Among good top candidates, pick best rated
        return max(good_top, key=lambda m: (_quality(m)[0], _quality(m)[1]))

    # If no good quality in top 3, expand to top 5
    top_5 = viable[:5]
    good_5 = [m for m in top_5 if _is_good_quality(m)]
    if good_5:
        return max(good_5, key=lambda m: (_quality(m)[0], _quality(m)[1]))

    # Fallback: first viable with any rating, or just first
    for m in viable:
        if _quality(m)[0] > 0:
            return m
    return shortlist[0]


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
            "keywords": str(row.get("keywords", "")),
            "production_countries": str(row.get("production_countries", "")),
            "vote_average": row.get("effective_rating", row.get("vote_average", "")),
            "vote_count": row.get("effective_votes", row.get("vote_count", "")),
        }
        for key in ("genre_match", "avoid_hit", "person_match", "year_match", "runtime_match", "seed_match"):
            if key in ref:
                movie[key] = ref[key]
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
    force_include_director: bool = False,
    force_include_cast: bool = False,
    force_include_country: bool = False,
    constraint_note: str = "",
    setting_period: str = "",
) -> tuple[dict[str, Any], int]:
    prompt = _build_selection_prompt(
        preferences,
        shortlist,
        history_exclusion_text,
        force_include_director=force_include_director,
        force_include_cast=force_include_cast,
        force_include_country=force_include_country,
        constraint_note=constraint_note,
        setting_period=setting_period,
    )
    _check_cancellation()
    response = _get_client(timeout_seconds).chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options=_final_llm_options(),
    )

    payload = _extract_json_object(response.message.content)
    selection_payload = payload.get("selection") if isinstance(payload.get("selection"), dict) else payload

    try:
        tmdb_id = _resolve_selected_tmdb_id(selection_payload, shortlist)
    except ValueError:
        tmdb_id = int(shortlist[0]["tmdb_id"])
        fallback_title = shortlist[0]["title"]
        raw_description = str(selection_payload.get("description", "") or "").strip()
        if raw_description:
            selection_payload["description"] = (
                f"{raw_description} The closest available pick from this shortlist is {fallback_title}."
            )
        else:
            selection_payload["description"] = (
                f"I do not see a perfect match for every constraint in the available shortlist, "
                f"so the closest available pick is {fallback_title}."
            )

    description = str(selection_payload.get("description", "")).strip()
    result = {
        "tmdb_id": tmdb_id,
        "description": description[:500],
    }
    return result, len(prompt)


def _describe_selected_movie(
    preferences: str,
    movie: dict[str, Any],
    history_exclusion_text: str,
    timeout_seconds: float,
    setting_period: str = "",
) -> tuple[dict[str, Any], int]:
    prompt = _build_description_only_prompt(preferences, movie, history_exclusion_text, setting_period=setting_period)
    _check_cancellation()
    response = _get_client(timeout_seconds).chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options=_description_llm_options(),
    )
    payload = _extract_json_object(response.message.content)
    description = str(payload.get("description", "")).strip()
    if not description:
        raise ValueError("Description-only response was empty")
    return {"tmdb_id": int(movie["tmdb_id"]), "description": description[:500]}, len(prompt)


def _remaining_llm_timeout_seconds(started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    remaining = TOTAL_REQUEST_BUDGET_SECONDS - elapsed - FALLBACK_BUFFER_SECONDS - LLM_TIMEOUT_SAFETY_MARGIN_SECONDS
    return max(0.0, remaining)


def _should_wait_for_intent(retrieval_profile: dict[str, Any]) -> bool:
    # Always wait for intent - the intent LLM can extract subgenre keywords,
    # tone, and other signals that BM25 lexical matching misses.
    # Previously we skipped intent when retrieval confidence was "high",
    # but BM25 confidence doesn't reflect semantic understanding.
    return True


def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    started_at = time.perf_counter()
    reset_request_cancellation()  # Clear any stale cancellation from previous requests

    intent_future = None
    intent_started = 0.0
    intent_elapsed = 0.0
    intent_used = False
    if ENABLE_LLM_INTENT:
        intent_started = time.perf_counter()
        intent_future = INTENT_EXECUTOR.submit(_get_intent_override, preferences)

    shortlist_refs, prompt_profile, retrieval_profile = _build_shortlist(preferences, history)

    # Check budget before waiting for intent
    # Intent runs in parallel, so if retrieval alone took >5s, something is wrong
    elapsed_so_far = time.perf_counter() - started_at
    retrieval_was_slow = elapsed_so_far > 3.0
    if elapsed_so_far > 5.0:
        # Retrieval was abnormally slow - skip intent to preserve LLM budget
        cancel_active_requests()
        logger.warning("Retrieval slow (%.2fs), skipping intent wait", elapsed_so_far)
        intent_future = None
    elif elapsed_so_far > 6.5:
        # If retrieval took > 6.5s, don't even wait for intent - go straight to LLM
        cancel_active_requests()
        logger.warning("Retrieval very slow (%.2fs), cancelling intent entirely", elapsed_so_far)
        intent_future = None

    if intent_future is not None and _should_wait_for_intent(retrieval_profile):
        # If first retrieval was already slow, be conservative with intent timeout
        # This prevents holding up LLM selection budget
        intent_timeout_budget = INTENT_LLM_TIMEOUT_S
        if retrieval_was_slow:
            intent_timeout_budget = min(2.0, INTENT_LLM_TIMEOUT_S - 1.5)
        timeout_left = max(0.0, intent_timeout_budget - (time.perf_counter() - intent_started))
        try:
            intent_override = intent_future.result(timeout=timeout_left)
        except FuturesTimeoutError:
            intent_override = {}
            logger.debug("Intent LLM timed out after %.2fs", time.perf_counter() - intent_started)
        except Exception:
            intent_override = {}
        intent_elapsed = time.perf_counter() - intent_started
        
        # Skip intent override processing if we're already near budget limit
        # Prevents second _build_shortlist from eating into LLM selection time
        elapsed_before_override = time.perf_counter() - started_at
        if elapsed_before_override > 7.0:
            logger.info("Already at %.1fs, skipping intent override re-retrieval", elapsed_before_override)
            intent_override = {}
        elif intent_override:
            shortlist_refs, prompt_profile, retrieval_profile = _build_shortlist(
                preferences,
                history,
                mode="lexical",
                retrieval_profile_override=intent_override,
            )
            intent_used = True
    elif intent_future is not None:
        intent_elapsed = time.perf_counter() - intent_started

    retrieval_elapsed = time.perf_counter() - started_at
    shortlist = _enrich_shortlist(shortlist_refs, include_year=bool(prompt_profile.get("year_relevant")))
    if not shortlist:
        raise ValueError("No candidate movies available")

    confidence_bundle = retrieval_profile.get("confidence_bundle", {})
    route = confidence_bundle.get("route", "judge_8")
    confidence = confidence_bundle.get("confidence", "low")
    convergence = confidence_bundle.get("convergence", "weak")
    semantic_elapsed = float(retrieval_profile.get("semantic_elapsed_s", 0.0) or 0.0)
    semantic_used = bool(retrieval_profile.get("semantic_active", False))
    llm_shortlist = _route_shortlist(shortlist, route)
    force_include_person_fields = bool(retrieval_profile.get("has_named_person_signal"))
    force_include_country = bool(prompt_profile.get("country_relevant"))
    setting_period = str(prompt_profile.get("setting_period", "") or "")
    prompt_chars = 0
    llm_elapsed = 0.0
    try:
        llm_timeout_seconds = _remaining_llm_timeout_seconds(started_at)
        if llm_timeout_seconds <= 0.5:
            raise TimeoutError("Not enough request budget left for LLM selection")
        llm_started = time.perf_counter()
        if route == "description_only":
            result, prompt_chars = _describe_selected_movie(
                preferences,
                shortlist[0],
                retrieval_profile["history_exclusion_text"],
                timeout_seconds=llm_timeout_seconds,
                setting_period=setting_period,
            )
        else:
            result, prompt_chars = _choose_with_llm(
                preferences,
                llm_shortlist,
                retrieval_profile["history_exclusion_text"],
                timeout_seconds=llm_timeout_seconds,
                force_include_director=force_include_person_fields,
                force_include_cast=force_include_person_fields,
                force_include_country=force_include_country,
                constraint_note=str(retrieval_profile.get("candidate_constraint_note", "")),
                setting_period=setting_period,
            )
        llm_elapsed = time.perf_counter() - llm_started
        result["used_llm"] = True
        logger.info(
            "Recommendation metrics: retrieval_s=%.3f intent_s=%.3f semantic_s=%.3f llm_s=%.3f fallback_used=%s route=%s confidence=%s intent_used=%s semantic_used=%s convergence=%s",
            retrieval_elapsed,
            intent_elapsed,
            semantic_elapsed,
            llm_elapsed,
            False,
            route,
            confidence,
            intent_used,
            semantic_used,
            convergence,
        )
        return result
    except Exception as exc:
        logger.warning(
            "Falling back to heuristic recommendation: preferences_len=%d history_count=%d shortlist_count=%d error=%s",
            len(preferences),
            len(history),
            len(shortlist),
            exc,
        )
        llm_elapsed = max(0.0, time.perf_counter() - started_at - retrieval_elapsed)
        if not prompt_chars:
            prompt_chars = len(
                _build_selection_prompt(
                    preferences,
                    shortlist,
                    retrieval_profile["history_exclusion_text"],
                    force_include_director=force_include_person_fields,
                    force_include_cast=force_include_person_fields,
                    force_include_country=force_include_country,
                    constraint_note=str(retrieval_profile.get("candidate_constraint_note", "")),
                    setting_period=setting_period,
                )
            )
        logger.info(
            "Recommendation metrics: retrieval_s=%.3f intent_s=%.3f semantic_s=%.3f llm_s=%.3f fallback_used=%s route=%s confidence=%s intent_used=%s semantic_used=%s convergence=%s",
            retrieval_elapsed,
            intent_elapsed,
            semantic_elapsed,
            llm_elapsed,
            True,
            route,
            confidence,
            intent_used,
            semantic_used,
            convergence,
        )
        best = _safe_fallback_movie(shortlist)
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
