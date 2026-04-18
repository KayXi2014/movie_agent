from __future__ import annotations

import os

from retrieval import refresh_retrieval_artifacts
from scripts.build_enriched_dataset import build_enriched_dataset


def prepare_local_runtime() -> None:
    if os.environ.get("TMDB_API_KEY"):
        print("TMDB_API_KEY detected. Building enriched dataset first...")
        build_enriched_dataset()
    else:
        print("TMDB_API_KEY not set. Skipping optional dataset enrichment.")

    print("Building retrieval artifacts...")
    refresh_retrieval_artifacts()
    print("Local runtime preparation complete.")


def main() -> None:
    prepare_local_runtime()


if __name__ == "__main__":
    main()
