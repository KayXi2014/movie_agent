from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from tmdb_client import enrich_movie_rows


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "tmdb_top1000_movies.csv"
TARGET_PATH = ROOT / "tmdb_top1000_movies_enriched.csv"


def build_enriched_dataset() -> None:
    if not os.environ.get("TMDB_API_KEY"):
        raise SystemExit("TMDB_API_KEY is required to build the enriched dataset.")

    df = pd.read_csv(SOURCE_PATH).fillna("")
    enriched = enrich_movie_rows(df.copy(), top_n=len(df))
    enriched.to_csv(TARGET_PATH, index=False)
    print(f"Wrote enriched dataset to {TARGET_PATH}")


def main() -> None:
    build_enriched_dataset()


if __name__ == "__main__":
    main()
