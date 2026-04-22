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
INTENT_LLM_TIMEOUT_S = float(os.getenv("INTENT_LLM_TIMEOUT_S", "4.0"))
logger = logging.getLogger(__name__)
INTENT_EXECUTOR = ThreadPoolExecutor(max_workers=1)
KNOWN_GENRES = tuple(sorted({genre.strip() for value in TOP_MOVIES["genres"] for genre in str(value or "").split(",") if genre.strip()}))


@lru_cache(maxsize=8)
def _get_client(timeout_seconds: float) -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=timeout_seconds,
    )


def _build_intent_prompt(preferences: str) -> str:
    genres_text = ", ".join(KNOWN_GENRES)
    return (
        "Return JSON only. Extract retrieval hints from this movie request.\n"
        "Omit unclear fields.\n"
        'Schema: {"genres":[],"avoid":[],"named_people":[],"release_year":null,'
        '"setting_period":"","runtime":null,"tone":[],"keyword_hints":[],"quality_preference":false,'
        '"country_or_language":[]}\n'
        "Rules: release_year=release date only: recent, classic, from 1990s, after 2020; use an object with min/max/label. "
        "setting_period=story world: modern day, period piece, set in 1990s, dystopian setting. "
        "runtime=movie length only: short, quick, under 90 minutes, over 2 hours; use an object with min/max/label. "
        "keyword_hints=searchable themes implied by the request. "
        "quality_preference=true only for high-rated/popular/acclaimed/crowd-pleasing. "
        f"Use only these genres: {genres_text}.\n"
        'Examples: dystopian setting -> {"setting_period":"dystopian setting","keyword_hints":["dystopia","dystopian future","post-apocalyptic future"]}; '
        'modern story -> {"setting_period":"contemporary setting"}; '
        'after 2020 -> {"release_year":{"min":2021,"max":9999,"label":"after 2020"}}; '
        'quick under 90 minutes -> {"runtime":{"max":90,"label":"under 90 minutes"}}.\n'
        f"Request: {preferences}"
    )


def _intent_llm_options() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 10,
        "top_p": 0.5,
        "num_predict": 180,
    }


def _final_llm_options() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 10,
        "top_p": 0.5,
        "num_predict": 180,
    }


def _description_llm_options() -> dict[str, Any]:
    return {
        "temperature": 0.35,
        "top_k": 20,
        "top_p": 0.8,
        "num_predict": 220,
    }


def _get_intent_override(preferences: str) -> dict[str, Any]:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key or not ENABLE_LLM_INTENT:
        return {}
    try:
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
    named_people = [" ".join(str(item or "").split()).strip() for item in payload.get("named_people", []) if str(item or "").strip()]
    result: dict[str, Any] = {}
    if genres:
        result["genres"] = genres[:3]
    if avoid:
        result["avoid"] = avoid[:3]
    if named_people:
        result["named_people"] = named_people[:3]
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
    force_include_director: bool = False,
    force_include_cast: bool = False,
    force_include_country: bool = False,
    constraint_note: str = "",
    setting_period: str = "",
) -> str:
    history_hint = _build_history_exposure_hint(history_exclusion_text)
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
        "Pick one movie from the shortlist and write a persuasive recommendation.",
        f"Request: {preferences}",
    ]
    if history_hint != "none":
        sections.append(f"History: {history_hint}")
    if constraint_note:
        sections.append(f"Constraint note: {constraint_note}")
    if setting_period:
        sections.append(f"Story setting preference: {setting_period}")
    sections.extend(
        [
            "",
            "Rules:",
            "- Use only listed candidate details; do not invent another movie.",
            "- Honor explicit director, cast, year, story-setting, and avoid constraints.",
            "- If no candidate fully fits, pick the closest and briefly acknowledge the gap.",
            "- Use quality signal only as an internal tie-breaker when fit is similar.",
            "- Do not mention scores, ratings, vote counts, quality signals, or selection metadata in the description.",
            'Return JSON only: {"tmdb_id": <id>, "title": "<exact title>", "description": "<2-3 sentences, under 500 chars>"}',
            "",
            "Candidates:",
            shortlist_text,
            "",
            "Description: address the user directly; give a vivid hook, then why it fits. No generic hype.",
        ]
    )
    return "\n".join(sections)


def _build_description_only_prompt(
    preferences: str,
    movie: dict[str, Any],
    history_exclusion_text: str,
    setting_period: str = "",
) -> str:
    history_hint = _build_history_exposure_hint(history_exclusion_text)

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
        "Write only the recommendation blurb for this already-selected movie.",
        f"Request: {preferences}",
    ]
    if history_hint != "none":
        sections.append(f"History: {history_hint}")
    if setting_period:
        sections.append(f"Story setting preference: {setting_period}")
    sections.extend(
        [
            f"Selected movie: {' | '.join(details)}",
            'Return JSON only: {"description": "<2-3 sentences, under 500 chars>"}',
            "Be personal, persuasive, and specific to this user.",
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
        if rating >= 7.2 and votes >= 1000:
            return "There is enough audience love behind it that I would feel comfortable recommending it, not just naming it."
        if rating > 0.0 and votes > 0:
            return "This is more of a fit-first pick, so I would go in for the premise rather than expect a universal crowd-pleaser."
        return "The audience signal is thin, so I would frame this as a fit-first pick rather than a sure thing."

    title = f'"{movie["title"]}"'
    request_phrase = _request_phrase()
    appeal_phrase = _appeal_phrase()
    if overview:
        if request_phrase:
            hook = f"I’d point you to {title} because it gives your {request_phrase} request a concrete shape: {_trim_sentence(overview)}"
        elif appeal_phrase:
            hook = f"I’d point you to {title}; it has a clear {appeal_phrase} pull from the start: {_trim_sentence(overview)}"
        else:
            hook = f"I’d point you to {title}; the setup has an immediate pull: {_trim_sentence(overview)}"
    elif keywords:
        hook = f"I’d point you to {title} for its {appeal_phrase} energy; that gives the pick a sharper identity than a generic fallback."
    else:
        hook = f"I’d point you to {title} because it looks like the cleanest fit in the current shortlist, not just a random safe choice."

    wants_quality_context = bool(prompt_profile.get("quality_preference") or prompt_profile.get("rating_constraint"))
    support_line = _quality_phrase() if wants_quality_context else ""

    tones = prompt_profile.get("tone", [])
    if tones:
        reason = ", ".join(tones[:2])
        fit_line = f"If you want something {reason} right now, I’d choose this for the feeling it promises, not just the category it falls into."
    elif prompt_profile.get("setting_period"):
        setting = str(prompt_profile["setting_period"])
        fit_line = f"For a {setting} feel, it gives you a real atmosphere to step into instead of just matching a label."
    elif prompt_profile["target_genres"]:
        target = _natural_list([str(item) for item in prompt_profile["target_genres"][:2]])
        fit_line = f"For a {target} request, I’d rather send you toward something with a distinct angle than something that merely checks the genre box."
    elif movie["genres"]:
        genres = _natural_list([part.strip().lower() for part in str(movie["genres"]).split(",") if part.strip()][:2])
        fit_line = f"If you are browsing in {_article_for(genres)} {genres} mood, this gives you a clearer reason to press play than most filler picks."
    else:
        fit_line = "If you want something without overthinking it, this is the most defensible choice I can make."

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
    def _quality(movie: dict[str, Any]) -> tuple[float, int]:
        try:
            rating = float(movie.get("vote_average", 0.0) or 0.0)
            votes = int(float(movie.get("vote_count", 0) or 0))
        except (TypeError, ValueError):
            return 0.0, 0
        return rating, votes

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

    reasonable = [movie for movie in viable if (lambda q: q[0] >= 6.2 and q[1] >= 1000)(_quality(movie))]
    if reasonable:
        return max(reasonable, key=lambda movie: (_quality(movie)[0], _quality(movie)[1]))

    nonzero = [movie for movie in viable if (lambda q: q[0] > 0.0 and q[1] > 0)(_quality(movie))]
    if nonzero:
        return max(nonzero, key=lambda movie: (_quality(movie)[0], _quality(movie)[1]))

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

    result = {
        "tmdb_id": tmdb_id,
        "description": str(selection_payload.get("description", ""))[:500],
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
    confidence = retrieval_profile.get("confidence_bundle", {})
    return not (
        confidence.get("confidence") == "high"
        and confidence.get("top_fulfillment") == "high"
        and not confidence.get("contradictions")
    )


def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    started_at = time.perf_counter()
    intent_future = None
    intent_started = 0.0
    intent_elapsed = 0.0
    intent_used = False
    if ENABLE_LLM_INTENT:
        intent_started = time.perf_counter()
        intent_future = INTENT_EXECUTOR.submit(_get_intent_override, preferences)

    shortlist_refs, prompt_profile, retrieval_profile = _build_shortlist(preferences, history)
    if intent_future is not None and _should_wait_for_intent(retrieval_profile):
        timeout_left = max(0.0, INTENT_LLM_TIMEOUT_S - (time.perf_counter() - intent_started))
        try:
            intent_override = intent_future.result(timeout=timeout_left)
        except FuturesTimeoutError:
            intent_override = {}
        except Exception:
            intent_override = {}
        intent_elapsed = time.perf_counter() - intent_started
        if intent_override:
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
