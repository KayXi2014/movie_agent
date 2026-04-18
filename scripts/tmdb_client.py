"""
tmdb_client.py — optional offline TMDB enrichment for metadata not already present
in `tmdb_top1000_movies.csv`.

This module is only used by `build_enriched_dataset.py`.

Current enrichment goals:
- add franchise / collection metadata
- add similar-movie metadata

It intentionally does NOT overwrite the core fields already present in the CSV
such as overview, genres, keywords, director, top cast, ratings, or runtime.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import httpx
import pandas as pd


logger = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
DEFAULT_TIMEOUT_SECONDS = 5.0
SIMILAR_LIMIT = 8

ENRICHMENT_COLUMNS = [
    "collection_id",
    "collection_name",
    "similar_tmdb_ids",
    "similar_titles",
]


def _api_key() -> str:
    return os.environ.get("TMDB_API_KEY", "").strip()


def _bearer_token() -> str:
    return os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()


def _timeout_seconds() -> float:
    try:
        return float(os.environ.get("TMDB_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _enabled() -> bool:
    return bool(_api_key() or _bearer_token())


def _get(path: str, **params: Any) -> dict[str, Any] | None:
    if not _enabled():
        return None
    api_key = _api_key()
    bearer_token = _bearer_token()
    request_params = dict(params)
    headers: dict[str, str] = {}

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif api_key:
        request_params = {"api_key": api_key, **request_params}

    try:
        response = httpx.get(
            f"{TMDB_BASE}{path}",
            params=request_params,
            headers=headers,
            timeout=_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("TMDB request failed: path=%s status=%s body=%r", path, exc.response.status_code, exc.response.text[:200])
        return None
    except Exception as exc:
        logger.warning("TMDB request failed: path=%s error=%r", path, exc)
        return None


@lru_cache(maxsize=1024)
def _fetch_related_metadata(tmdb_id: int) -> dict[str, Any] | None:
    return _get(f"/movie/{tmdb_id}", append_to_response="similar")


def _parse_related_metadata(detail: dict[str, Any]) -> dict[str, Any]:
    collection = detail.get("belongs_to_collection") or {}
    similar_results = (detail.get("similar") or {}).get("results") or []

    similar_ids = [int(item["id"]) for item in similar_results[:SIMILAR_LIMIT] if item.get("id") is not None]
    similar_titles = [str(item.get("title") or "") for item in similar_results[:SIMILAR_LIMIT] if item.get("title")]

    return {
        "collection_id": str(collection["id"]) if collection.get("id") is not None else "",
        "collection_name": str(collection.get("name") or ""),
        "similar_tmdb_ids": json.dumps(similar_ids),
        "similar_titles": json.dumps(similar_titles),
    }


def _ensure_enrichment_columns(rows: pd.DataFrame) -> pd.DataFrame:
    enriched = rows.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.Series([""] * len(enriched), index=enriched.index, dtype="object")
        else:
            enriched[column] = enriched[column].fillna("").astype("object")
    return enriched


def _needs_enrichment(row: pd.Series) -> bool:
    for column in ENRICHMENT_COLUMNS:
        if str(row.get(column, "") or "").strip() == "":
            return True
    return False


def _merge_row_with_related_metadata(row: pd.Series, detail: dict[str, Any]) -> pd.Series:
    parsed = _parse_related_metadata(detail)
    updated = row.copy()
    for field, value in parsed.items():
        updated[field] = str(value or "")
    return updated


def enrich_movie_rows(rows: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Add collection/franchise metadata and similar-movie metadata to the dataset.

    This is an offline enrichment step only. Existing movie fields from the source
    CSV are preserved as-is.
    """
    if rows.empty or top_n <= 0 or not _enabled():
        return _ensure_enrichment_columns(rows)

    enriched = _ensure_enrichment_columns(rows)
    top_n = min(top_n, len(enriched))
    candidate_indices = [index for index in enriched.head(top_n).index if _needs_enrichment(enriched.loc[index])]
    if not candidate_indices:
        return enriched

    def _fetch_row(index: Any) -> tuple[Any, dict[str, Any] | None]:
        detail = _fetch_related_metadata(int(enriched.loc[index]["tmdb_id"]))
        return index, detail

    with ThreadPoolExecutor(max_workers=min(4, len(candidate_indices))) as executor:
        futures = {executor.submit(_fetch_row, index): index for index in candidate_indices}
        for future in as_completed(futures):
            index, detail = future.result()
            if detail:
                for field, value in _parse_related_metadata(detail).items():
                    enriched.at[index, field] = str(value or "")

    return enriched
