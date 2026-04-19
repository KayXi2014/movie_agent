from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.text_artifacts import refresh_retrieval_artifacts
from scripts.tmdb_enrichment import build_enriched_dataset


def prepare_local_runtime() -> None:
    if os.environ.get("TMDB_API_KEY") or os.environ.get("TMDB_READ_ACCESS_TOKEN"):
        print("TMDB credentials detected. Building enriched dataset first...")
        build_enriched_dataset()
    else:
        print("TMDB credentials not set. Skipping optional dataset enrichment.")

    print("Building retrieval artifacts...")
    try:
        refresh_retrieval_artifacts()
    except PermissionError as exc:
        raise SystemExit(
            "Could not rebuild retrieval artifacts because a data file is in use. "
            "Stop any running API, Streamlit, or SQLite process using data/movies.sqlite and try again."
        ) from exc
    print("Local runtime preparation complete.")


def main() -> None:
    prepare_local_runtime()


if __name__ == "__main__":
    main()
