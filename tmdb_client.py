"""
tmdb_client.py — TMDB API fallback for history items not found in the local dataset.

Usage:
    from tmdb_client import fetch_tmdb_history_row

    pseudo_row = fetch_tmdb_history_row(tmdb_id=299536, title="Avengers: Infinity War")
    # Returns a dict with genres_set, keywords_set, cast_set, director_set, etc.
    # Returns None if the API is unavailable or the title cannot be resolved.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any

import httpx
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

TMDB_API_KEY: str = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"
_TIMEOUT = float(os.environ.get("TMDB_TIMEOUT_SECONDS", "1.0"))
_MAX_CAST = 5


def _enabled() -> bool:
    return bool(TMDB_API_KEY)


def _get(path: str, **params: Any) -> dict | None:
    """Raw GET helper. Returns parsed JSON or None on any error."""
    if not _enabled():
        return None
    t0 = time.perf_counter()
    try:
        resp = httpx.get(
            f"{TMDB_BASE}{path}",
            params={"api_key": TMDB_API_KEY, **params},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0
        logger.warning("TIMING tmdb_http_seconds=%.3f path=%s", elapsed, path)
        return resp.json()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.warning("TIMING tmdb_http_seconds=%.3f path=%s failed=True", elapsed, path)
        logger.debug("TMDB request failed: %s", exc)
        return None


def _search_tmdb_id(title: str) -> int | None:
    """Search TMDB by title and return the best-match tmdb_id, or None."""
    data = _get("/search/movie", query=title, include_adult=False)
    if not data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return int(results[0]["id"])


@lru_cache(maxsize=256)
def _fetch_details(tmdb_id: int) -> dict | None:
    """Fetch full movie detail + credits + keywords in one append_to_response call."""
    return _get(
        f"/movie/{tmdb_id}",
        append_to_response="credits,keywords",
    )


def _parse_details(detail: dict) -> dict[str, Any]:
    """
    Convert a raw TMDB detail payload into the same field shapes
    that _prepare_movies() produces on the local CSV rows.

    Returned keys used downstream by _derive_history_signals():
        genres_set, keywords_set, cast_set, director_set,
        genres_tokens, keywords_tokens,
        title, title_root, normalized_title,
        tmdb_id, year, vote_average, vote_count,
        overview, tagline, director, top_cast
    """
    from llm import _normalize_text, _tokenize, _split_csvish, _title_root  # local import avoids circular at module level

    genres_raw = ", ".join(g["name"] for g in (detail.get("genres") or []))
    genres_set = _split_csvish(genres_raw)

    kw_list = (detail.get("keywords") or {}).get("keywords") or []
    keywords_raw = ", ".join(k["name"] for k in kw_list)
    keywords_set = _split_csvish(keywords_raw)

    credits = detail.get("credits") or {}
    cast_list = credits.get("cast") or []
    cast_names = [m["name"] for m in cast_list[:_MAX_CAST]]
    cast_raw = ", ".join(cast_names)
    cast_set = _split_csvish(cast_raw)

    crew_list = credits.get("crew") or []
    directors = [m["name"] for m in crew_list if m.get("job") == "Director"]
    director_raw = ", ".join(directors)
    director_set = _split_csvish(director_raw)

    title = str(detail.get("title") or "")
    release = str(detail.get("release_date") or "")
    year = int(release[:4]) if release and release[:4].isdigit() else 0

    return {
        # identity
        "tmdb_id": int(detail.get("id", 0)),
        "title": title,
        "normalized_title": _normalize_text(title),
        "title_root": _title_root(title),
        "year": year,
        # quality signals
        "vote_average": float(detail.get("vote_average") or 0.0),
        "vote_count": float(detail.get("vote_count") or 0.0),
        # text fields (raw strings, mirroring CSV columns)
        "genres": genres_raw,
        "keywords": keywords_raw,
        "overview": str(detail.get("overview") or ""),
        "tagline": str(detail.get("tagline") or ""),
        "director": director_raw,
        "top_cast": cast_raw,
        "original_language": str(detail.get("original_language") or ""),
        "production_countries": ", ".join(
            c["name"] for c in (detail.get("production_countries") or [])
        ),
        # set / token fields expected by _derive_history_signals()
        "genres_set": genres_set,
        "keywords_set": keywords_set,
        "cast_set": cast_set,
        "director_set": director_set,
        "genres_tokens": _tokenize(genres_raw),
        "keywords_tokens": _tokenize(keywords_raw),
    }


def _merge_row_with_detail(row: pd.Series, detail: dict[str, Any]) -> pd.Series:
    """Overlay TMDB data onto a local movie row, preferring richer non-empty values."""
    from llm import _normalize_text, _split_csvish, _tokenize, _title_root  # local import avoids circular at module level

    parsed = _parse_details(detail)
    updated = row.copy()

    text_fields = (
        "title",
        "genres",
        "keywords",
        "overview",
        "tagline",
        "director",
        "top_cast",
        "original_language",
        "production_countries",
    )

    for field in text_fields:
        current_value = str(updated.get(field) or "")
        incoming_value = str(parsed.get(field) or "")
        if not current_value.strip() and incoming_value.strip():
            updated[field] = incoming_value
        elif field == "overview" and len(incoming_value) > len(current_value):
            updated[field] = incoming_value

    updated["tmdb_id"] = parsed["tmdb_id"]
    updated["year"] = parsed["year"] or int(updated.get("year") or 0)
    updated["normalized_title"] = parsed["normalized_title"]
    updated["title_root"] = parsed["title_root"]
    updated["vote_average"] = float(updated.get("vote_average") or parsed["vote_average"] or 0.0)
    updated["vote_count"] = float(updated.get("vote_count") or parsed["vote_count"] or 0.0)

    updated["genres_set"] = _split_csvish(updated["genres"])
    updated["keywords_set"] = _split_csvish(updated["keywords"])
    updated["cast_set"] = _split_csvish(updated["top_cast"])
    updated["director_set"] = _split_csvish(updated["director"])
    updated["genres_tokens"] = _tokenize(updated["genres"])
    updated["keywords_tokens"] = _tokenize(updated["keywords"])
    updated["overview_tokens"] = _tokenize(updated["overview"])
    updated["tagline_tokens"] = _tokenize(updated["tagline"])
    updated["title_tokens"] = _tokenize(updated["title"])
    updated["search_blob"] = _normalize_text(
        " ".join(
            str(updated.get(col, ""))
            for col in [
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
        )
    )
    updated["search_blob_tokens"] = _tokenize(updated["search_blob"])
    return updated


def _needs_enrichment(row: pd.Series) -> bool:
    """Return True when local metadata is sparse enough to justify a TMDB fetch."""
    overview = str(row.get("overview") or "").strip()
    keywords = str(row.get("keywords") or "").strip()
    director = str(row.get("director") or "").strip()
    top_cast = str(row.get("top_cast") or "").strip()

    # Only enrich rows that are likely to benefit from external metadata.
    if len(overview) < 80:
        return True
    if not keywords:
        return True
    if not director:
        return True
    if not top_cast:
        return True
    return False


def enrich_movie_rows(rows: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Optionally enrich the top-N rows with TMDB data.

    This is a bounded, opt-in quality boost for retrieval ranking.
    If TMDB is disabled or the API fails, the input rows are returned unchanged.
    """
    if rows.empty or top_n <= 0 or not _enabled():
        return rows

    t0 = time.perf_counter()
    top_n = min(top_n, len(rows))
    enriched = rows.copy()
    candidate_indices = list(enriched.head(top_n).index)
    indices = [index for index in candidate_indices if _needs_enrichment(enriched.loc[index])]

    if not indices:
        return enriched

    def _fetch_row(index: Any) -> tuple[Any, dict[str, Any] | None]:
        row = enriched.loc[index]
        detail = _fetch_details(int(row["tmdb_id"]))
        return index, detail

    with ThreadPoolExecutor(max_workers=min(4, len(indices))) as executor:
        futures = {executor.submit(_fetch_row, index): index for index in indices}
        for future in as_completed(futures):
            index, detail = future.result()
            if detail:
                enriched.loc[index] = _merge_row_with_detail(enriched.loc[index], detail)

    elapsed = time.perf_counter() - t0
    logger.warning(
        "TIMING tmdb_enrich_rows_seconds=%.3f requested_top_n=%d attempted=%d",
        elapsed,
        top_n,
        len(indices),
    )

    return enriched


def fetch_tmdb_history_row(
    tmdb_id: int | None,
    title: str,
) -> dict[str, Any] | None:
    """
    Public entry point for llm._history_rows().

    Resolution order:
      1. If tmdb_id is provided, fetch by ID directly.
      2. Otherwise, search by title to find the ID, then fetch details.

    Returns a parsed row dict on success, None if TMDB is unavailable
    or the title cannot be resolved (so the caller silently skips it).
    """
    if not _enabled():
        logger.debug("TMDB_API_KEY not set — skipping external lookup for %r", title)
        return None

    t0 = time.perf_counter()

    resolved_id = tmdb_id
    if resolved_id is None:
        resolved_id = _search_tmdb_id(title)
        if resolved_id is None:
            logger.debug("TMDB search found no match for %r", title)
            elapsed = time.perf_counter() - t0
            logger.warning(
                "TIMING tmdb_history_lookup_seconds=%.3f resolved=False title=%r",
                elapsed,
                title,
            )
            return None

    detail = _fetch_details(resolved_id)
    if not detail:
        elapsed = time.perf_counter() - t0
        logger.warning(
            "TIMING tmdb_history_lookup_seconds=%.3f resolved=False tmdb_id=%s title=%r",
            elapsed,
            resolved_id,
            title,
        )
        return None

    elapsed = time.perf_counter() - t0
    logger.warning(
        "TIMING tmdb_history_lookup_seconds=%.3f resolved=True tmdb_id=%d title=%r",
        elapsed,
        int(resolved_id),
        title,
    )
    return _parse_details(detail)