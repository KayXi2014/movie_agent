"""
Offline TMDB dataset enrichment for metadata not already present in
`data/tmdb_top1000_movies.csv`.

This module is not used by the deployed API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)

ROOT = PROJECT_ROOT
DATA_DIR = ROOT / "data"
SOURCE_PATH = DATA_DIR / "tmdb_top1000_movies.csv"
TARGET_PATH = DATA_DIR / "tmdb_top1000_movies_enriched.csv"

TMDB_BASE = "https://api.themoviedb.org/3"
DEFAULT_TIMEOUT_SECONDS = 5.0
SIMILAR_LIMIT = 8
RECOMMENDATION_LIMIT = 8
ALTERNATIVE_TITLE_LIMIT = 12
DEFAULT_ENRICHMENT_WORKERS = 8
ENRICHMENT_STATUS_COLUMN = "tmdb_enrichment_status"
ENRICHMENT_VERSION = "v2"

ENRICHMENT_COLUMNS = [
    "collection_id",
    "collection_name",
    "similar_tmdb_ids",
    "similar_titles",
    "recommended_tmdb_ids",
    "recommended_titles",
    "alternative_titles",
    "spoken_languages",
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


def _require_credentials() -> None:
    if not _enabled():
        raise SystemExit("TMDB_API_KEY or TMDB_READ_ACCESS_TOKEN is required to build the enriched dataset.")


def _enrichment_workers() -> int:
    try:
        value = int(os.environ.get("TMDB_ENRICHMENT_WORKERS", str(DEFAULT_ENRICHMENT_WORKERS)))
    except ValueError:
        value = DEFAULT_ENRICHMENT_WORKERS
    return max(1, value)


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
    return _get(
        f"/movie/{tmdb_id}",
        append_to_response="similar,recommendations,alternative_titles,release_dates",
        language="en-US",
    )


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _extract_us_certification(detail: dict[str, Any]) -> str:
    release_results = (detail.get("release_dates") or {}).get("results") or []
    for entry in release_results:
        if str(entry.get("iso_3166_1") or "").upper() != "US":
            continue
        for release_date in entry.get("release_dates") or []:
            certification = str(release_date.get("certification") or "").strip()
            if certification:
                return certification
    return ""


def _parse_related_metadata(detail: dict[str, Any]) -> dict[str, Any]:
    collection = detail.get("belongs_to_collection") or {}
    similar_results = (detail.get("similar") or {}).get("results") or []
    recommendation_results = (detail.get("recommendations") or {}).get("results") or []
    alternative_title_results = (detail.get("alternative_titles") or {}).get("titles") or []
    spoken_languages = detail.get("spoken_languages") or []

    similar_ids = [int(item["id"]) for item in similar_results[:SIMILAR_LIMIT] if item.get("id") is not None]
    similar_titles = [str(item.get("title") or "") for item in similar_results[:SIMILAR_LIMIT] if item.get("title")]
    recommended_ids = [
        int(item["id"]) for item in recommendation_results[:RECOMMENDATION_LIMIT] if item.get("id") is not None
    ]
    recommended_titles = [
        str(item.get("title") or item.get("original_title") or "")
        for item in recommendation_results[:RECOMMENDATION_LIMIT]
        if item.get("title") or item.get("original_title")
    ]
    alternative_titles = _unique_preserve_order(
        [
            str(item.get("title") or "")
            for item in alternative_title_results[:ALTERNATIVE_TITLE_LIMIT]
            if item.get("title")
        ]
    )
    spoken_language_names = _unique_preserve_order(
        [str(item.get("english_name") or item.get("name") or "") for item in spoken_languages]
    )

    return {
        "collection_id": str(collection["id"]) if collection.get("id") is not None else "",
        "collection_name": str(collection.get("name") or ""),
        "similar_tmdb_ids": json.dumps(similar_ids),
        "similar_titles": json.dumps(similar_titles),
        "recommended_tmdb_ids": json.dumps(recommended_ids),
        "recommended_titles": json.dumps(recommended_titles),
        "alternative_titles": json.dumps(alternative_titles),
        "spoken_languages": ", ".join(spoken_language_names),
        "us_rating": _extract_us_certification(detail),
    }


def _ensure_enrichment_columns(rows: pd.DataFrame) -> pd.DataFrame:
    enriched = rows.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.Series([""] * len(enriched), index=enriched.index, dtype="object")
        else:
            enriched[column] = enriched[column].fillna("").astype("object")
    if "us_rating" not in enriched.columns:
        enriched["us_rating"] = pd.Series([""] * len(enriched), index=enriched.index, dtype="object")
    else:
        enriched["us_rating"] = enriched["us_rating"].fillna("").astype("object")
    if ENRICHMENT_STATUS_COLUMN not in enriched.columns:
        enriched[ENRICHMENT_STATUS_COLUMN] = pd.Series([""] * len(enriched), index=enriched.index, dtype="object")
    else:
        enriched[ENRICHMENT_STATUS_COLUMN] = enriched[ENRICHMENT_STATUS_COLUMN].fillna("").astype("object")
    return enriched


def _needs_enrichment(row: pd.Series) -> bool:
    return str(row.get(ENRICHMENT_STATUS_COLUMN, "") or "").strip() != ENRICHMENT_VERSION


def _merge_row_with_related_metadata(row: pd.Series, detail: dict[str, Any]) -> pd.Series:
    parsed = _parse_related_metadata(detail)
    updated = row.copy()
    for field, value in parsed.items():
        if field == "us_rating" and str(updated.get("us_rating", "") or "").strip():
            continue
        updated[field] = str(value or "")
    updated[ENRICHMENT_STATUS_COLUMN] = ENRICHMENT_VERSION
    return updated


def enrich_movie_rows(rows: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
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

    with ThreadPoolExecutor(max_workers=min(_enrichment_workers(), len(candidate_indices))) as executor:
        futures = {executor.submit(_fetch_row, index): index for index in candidate_indices}
        for future in as_completed(futures):
            index, detail = future.result()
            if detail:
                updated = _merge_row_with_related_metadata(enriched.loc[index], detail)
                for field, value in updated.items():
                    enriched.at[index, field] = value

    return enriched


def build_enriched_dataset(source_path: Path = SOURCE_PATH, target_path: Path = TARGET_PATH, top_n: int | None = None) -> None:
    _require_credentials()
    df = pd.read_csv(source_path).fillna("")
    requested_top_n = len(df) if top_n is None else int(top_n)
    enriched = enrich_movie_rows(df.copy(), top_n=requested_top_n)
    enriched.to_csv(target_path, index=False)
    print(f"Wrote enriched dataset to {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TMDB-enriched movie dataset.")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="Source CSV to enrich.")
    parser.add_argument("--output", type=Path, default=TARGET_PATH, help="Output CSV path.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only enrich the first N rows. Defaults to the entire dataset.",
    )
    args = parser.parse_args()
    build_enriched_dataset(source_path=args.source, target_path=args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()
