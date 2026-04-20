from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import DATA_PATH, EMBEDDING_META_PATH, ENRICHED_DATA_PATH, RETRIEVAL_DB_META_PATH, load_json, metadata_matches
from scripts.llm_augment import augment_dataset, augmentation_status_summary
from scripts.text_artifacts import refresh_retrieval_artifacts
from scripts.tmdb_enrichment import build_enriched_dataset, enrichment_status_summary


def _tmdb_enabled() -> bool:
    return bool(os.environ.get("TMDB_API_KEY") or os.environ.get("TMDB_READ_ACCESS_TOKEN"))


def _augmentation_enabled() -> bool:
    return bool(os.environ.get("OLLAMA_API_KEY", "").strip())


def _active_data_path() -> Path:
    return ENRICHED_DATA_PATH if ENRICHED_DATA_PATH.exists() else DATA_PATH


def _artifacts_current() -> bool:
    active_data_path = _active_data_path()
    try:
        retrieval_meta = load_json(RETRIEVAL_DB_META_PATH)
        embedding_meta = load_json(EMBEDDING_META_PATH)
    except Exception:
        return False
    return metadata_matches(retrieval_meta, active_data_path) and metadata_matches(embedding_meta, active_data_path)


def prepare_local_runtime(
    *,
    refresh_tmdb: bool = False,
    refresh_augmentation: bool = False,
    skip_tmdb: bool = False,
    skip_augmentation: bool = False,
    skip_artifacts: bool = False,
    augment_model: str | None = None,
    augment_workers: int | None = None,
    augment_batch_size: int | None = None,
) -> None:
    dataset_changed = False

    if skip_tmdb:
        print("Skipping TMDB enrichment by request.")
    elif _tmdb_enabled():
        tmdb_summary = enrichment_status_summary()
        if refresh_tmdb or tmdb_summary["rows_needing_enrichment"] > 0:
            print(
                "Running TMDB enrichment..."
                if refresh_tmdb
                else f"TMDB enrichment pending for {tmdb_summary['rows_needing_enrichment']} rows. Refreshing now..."
            )
            build_enriched_dataset()
            dataset_changed = True
        else:
            print("TMDB enrichment already current. Skipping.")
    else:
        print("TMDB credentials not set. Skipping optional dataset enrichment.")

    if skip_augmentation:
        print("Skipping LLM augmentation by request.")
    elif _augmentation_enabled():
        augmentation_summary = augmentation_status_summary()
        if refresh_augmentation or augmentation_summary["rows_needing_llm"] > 0:
            print(
                "Running LLM augmentation..."
                if refresh_augmentation
                else f"LLM augmentation pending for {augmentation_summary['rows_needing_llm']} rows. Refreshing now..."
            )
            augment_dataset(
                model_name=augment_model,
                workers=augment_workers,
                batch_size=augment_batch_size,
                force=refresh_augmentation,
            )
            dataset_changed = True
        else:
            print("LLM augmentation already current. Skipping.")
    else:
        print("OLLAMA_API_KEY not set. Skipping optional LLM augmentation.")

    if skip_artifacts:
        print("Skipping retrieval artifact rebuild by request.")
        if dataset_changed:
            print("Note: dataset changed, so retrieval artifacts may now be stale until you rebuild them.")
    elif dataset_changed or not _artifacts_current():
        print("Building retrieval artifacts...")
        try:
            refresh_retrieval_artifacts()
        except PermissionError as exc:
            raise SystemExit(
                "Could not rebuild retrieval artifacts because a data file is in use. "
                "Stop any running API, Streamlit, or SQLite process using data/movies.sqlite and try again."
            ) from exc
    else:
        print("Retrieval artifacts already current. Skipping rebuild.")

    print("Local runtime preparation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local movie-recommender runtime assets.")
    parser.add_argument("--refresh-tmdb", action="store_true", help="Force a TMDB enrichment refresh when credentials are present.")
    parser.add_argument(
        "--refresh-augmentation",
        action="store_true",
        help="Force LLM augmentation even if fields are already populated.",
    )
    parser.add_argument("--skip-tmdb", action="store_true", help="Skip optional TMDB enrichment.")
    parser.add_argument("--skip-augmentation", action="store_true", help="Skip optional LLM augmentation.")
    parser.add_argument("--skip-artifacts", action="store_true", help="Skip rebuilding SQLite/embedding artifacts.")
    parser.add_argument("--augment-model", default=None, help="Override the Ollama model used for LLM augmentation.")
    parser.add_argument("--augment-workers", type=int, default=None, help="Override parallel workers for LLM augmentation.")
    parser.add_argument("--augment-batch-size", type=int, default=None, help="Override batch size for LLM augmentation saves.")
    args = parser.parse_args()

    prepare_local_runtime(
        refresh_tmdb=args.refresh_tmdb,
        refresh_augmentation=args.refresh_augmentation,
        skip_tmdb=args.skip_tmdb,
        skip_augmentation=args.skip_augmentation,
        skip_artifacts=args.skip_artifacts,
        augment_model=args.augment_model,
        augment_workers=args.augment_workers,
        augment_batch_size=args.augment_batch_size,
    )


if __name__ == "__main__":
    main()
