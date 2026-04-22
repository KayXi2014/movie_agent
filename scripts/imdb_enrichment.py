"""
Offline IMDb rating overlay for locally curated quality metadata.

This module is not used by the deployed API. It applies IMDb title ratings
from `data/IMDB_ratings.tsv` into the active movie CSV by joining the movie
dataset's `imdb_id` column to IMDb's `tconst` column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = PROJECT_ROOT / "data"
BASE_SOURCE_PATH = DATA_DIR / "tmdb_top1000_movies.csv"
ENRICHED_PATH = DATA_DIR / "tmdb_top1000_movies_enriched.csv"
IMDB_RATINGS_TSV_PATH = DATA_DIR / "IMDB_ratings.tsv"

IMDB_COLUMNS = [
    "imdb_id",
    "imdb_rating",
    "imdb_votes",
    "imdb_source",
    "updated_at",
]
OVERLAY_COLUMNS = ["tmdb_id", *IMDB_COLUMNS]


def _active_data_path() -> Path:
    return ENRICHED_PATH if ENRICHED_PATH.exists() else BASE_SOURCE_PATH


def _read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator, dtype="object").fillna("")


def _ensure_imdb_columns(rows: pd.DataFrame) -> pd.DataFrame:
    prepared = rows.copy()
    for column in IMDB_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.Series([""] * len(prepared), index=prepared.index, dtype="object")
        else:
            prepared[column] = prepared[column].fillna("").astype("object")
    return prepared


def _load_imdb_ratings_tsv(ratings_path: Path = IMDB_RATINGS_TSV_PATH) -> pd.DataFrame:
    if not ratings_path.exists():
        return pd.DataFrame(columns=IMDB_COLUMNS)
    ratings = _read_table(ratings_path)
    required = {"tconst", "averageRating", "numVotes"}
    if not required.issubset(ratings.columns):
        return pd.DataFrame(columns=IMDB_COLUMNS)
    ratings = ratings[["tconst", "averageRating", "numVotes"]].copy()
    ratings["tconst"] = ratings["tconst"].astype(str).str.strip()
    ratings = ratings[ratings["tconst"] != ""]
    ratings = ratings.drop_duplicates(subset=["tconst"], keep="last")
    ratings = ratings.rename(
        columns={
            "tconst": "imdb_id",
            "averageRating": "imdb_rating",
            "numVotes": "imdb_votes",
        }
    )
    ratings["imdb_source"] = ratings_path.name
    ratings["updated_at"] = ""
    return ratings


def _build_overlay_from_ratings(
    movies: pd.DataFrame,
    ratings_path: Path = IMDB_RATINGS_TSV_PATH,
) -> pd.DataFrame:
    ratings = _load_imdb_ratings_tsv(ratings_path)
    if not ratings.empty and "imdb_id" in movies.columns:
        movie_ids = movies[["tmdb_id", "imdb_id"]].copy()
        movie_ids["imdb_id"] = movie_ids["imdb_id"].astype(str).str.strip()
        movie_ids = movie_ids[movie_ids["imdb_id"] != ""]
        joined = movie_ids.merge(ratings, on="imdb_id", how="inner")
        if not joined.empty:
            overlay = joined[OVERLAY_COLUMNS].copy()
        else:
            overlay = pd.DataFrame(columns=OVERLAY_COLUMNS)
    else:
        overlay = pd.DataFrame(columns=OVERLAY_COLUMNS)
    overlay["tmdb_id"] = overlay["tmdb_id"].astype(str).str.strip()
    overlay = overlay[overlay["tmdb_id"] != ""]
    return overlay.drop_duplicates(subset=["tmdb_id"], keep="last")


def _nonempty_overlay_fields(row: pd.Series) -> dict[str, str]:
    fields: dict[str, str] = {}
    for column in IMDB_COLUMNS:
        value = str(row.get(column, "") or "").strip()
        if value:
            fields[column] = value
    return fields


def imdb_status_summary(
    source_path: Path | None = None,
    ratings_path: Path = IMDB_RATINGS_TSV_PATH,
) -> dict[str, Any]:
    source_path = source_path or _active_data_path()
    if not source_path.exists():
        return {
            "rows_total": 0,
            "overlay_rows": 0,
            "rows_current": 0,
            "rows_needing_imdb": 0,
            "overlay_rows_unmatched": 0,
        }

    movies = _ensure_imdb_columns(pd.read_csv(source_path, dtype="object").fillna(""))
    overlay = _build_overlay_from_ratings(movies, ratings_path=ratings_path)
    if overlay.empty:
        return {
            "rows_total": int(len(movies)),
            "overlay_rows": 0,
            "rows_current": 0,
            "rows_needing_imdb": 0,
            "overlay_rows_unmatched": 0,
        }

    movie_by_id = movies.set_index(movies["tmdb_id"].astype(str).str.strip(), drop=False)
    rows_current = 0
    rows_needing = 0
    rows_unmatched = 0
    for _, row in overlay.iterrows():
        tmdb_id = str(row["tmdb_id"]).strip()
        fields = _nonempty_overlay_fields(row)
        if not fields:
            continue
        if tmdb_id not in movie_by_id.index:
            rows_unmatched += 1
            continue
        movie_row = movie_by_id.loc[tmdb_id]
        if getattr(movie_row, "ndim", 1) > 1:
            movie_row = movie_row.iloc[0]
        if all(str(movie_row.get(column, "") or "").strip() == value for column, value in fields.items()):
            rows_current += 1
        else:
            rows_needing += 1

    return {
        "rows_total": int(len(movies)),
        "overlay_rows": int(len(overlay)),
        "rows_current": int(rows_current),
        "rows_needing_imdb": int(rows_needing),
        "overlay_rows_unmatched": int(rows_unmatched),
    }


def apply_imdb_overlay(
    source_path: Path | None = None,
    ratings_path: Path = IMDB_RATINGS_TSV_PATH,
) -> bool:
    source_path = source_path or _active_data_path()
    if not source_path.exists():
        raise FileNotFoundError(f"Movie dataset not found: {source_path}")

    movies = _ensure_imdb_columns(pd.read_csv(source_path, dtype="object").fillna(""))
    overlay = _build_overlay_from_ratings(movies, ratings_path=ratings_path)
    if overlay.empty:
        print("IMDb ratings overlay is empty or missing. Skipping.")
        return False

    movie_id_index = movies["tmdb_id"].astype(str).str.strip()
    changed = False
    applied_rows = 0
    unmatched_rows = 0
    for _, row in overlay.iterrows():
        tmdb_id = str(row["tmdb_id"]).strip()
        fields = _nonempty_overlay_fields(row)
        if not tmdb_id or not fields:
            continue
        matches = movies.index[movie_id_index == tmdb_id].tolist()
        if not matches:
            unmatched_rows += 1
            continue
        index = matches[0]
        row_changed = False
        for column, value in fields.items():
            if str(movies.at[index, column] or "").strip() != value:
                movies.at[index, column] = value
                row_changed = True
        if row_changed:
            changed = True
            applied_rows += 1

    if changed:
        movies.to_csv(source_path, index=False)
        print(f"Applied IMDb ratings overlay to {applied_rows} rows in {source_path}.")
    else:
        print("IMDb ratings overlay already current. Skipping.")
    if unmatched_rows:
        print(f"Warning: {unmatched_rows} IMDb overlay rows did not match a tmdb_id in {source_path}.")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply data/IMDB_ratings.tsv to the active movie dataset.")
    parser.add_argument("--source", type=Path, default=None, help="Movie CSV to update; defaults to the active dataset.")
    parser.add_argument("--ratings", type=Path, default=IMDB_RATINGS_TSV_PATH, help="IMDb title.ratings TSV.")
    parser.add_argument("--status", action="store_true", help="Print status only without writing.")
    args = parser.parse_args()

    if args.status:
        print(imdb_status_summary(source_path=args.source, ratings_path=args.ratings))
    else:
        apply_imdb_overlay(source_path=args.source, ratings_path=args.ratings)


if __name__ == "__main__":
    main()
