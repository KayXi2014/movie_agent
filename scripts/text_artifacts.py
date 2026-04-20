from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import (
    DATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_IDS_PATH,
    EMBEDDING_META_PATH,
    EMBEDDINGS_PATH,
    ENRICHED_DATA_PATH,
    RETRIEVAL_DB_META_PATH,
    RETRIEVAL_DB_PATH,
    dataset_fingerprint,
    ensure_dataset_columns,
    title_root,
    write_json,
)


FTS_TEXT_COLUMNS = [
    "title",
    "original_title",
    "alternative_titles",
    "genres",
    "keywords",
    "keywords_augmented",
    "tone_tags",
    "audience_tags",
    "source_tags",
    "director",
    "top_cast",
    "collection_name",
    "production_companies",
    "original_language",
    "production_countries",
    "spoken_languages",
    "us_rating",
]

EMBEDDING_TEXT_FIELDS = [
    ("Essence", "essence"),
    ("Tone Tags", "tone_tags_text"),
    ("Audience Tags", "audience_tags_text"),
    ("Source Tags", "source_tags_text"),
    ("Keywords", "keywords"),
    ("Keywords Augmented", "keywords_augmented_text"),
    ("Genres", "genres"),
    ("Overview", "overview"),
    ("Tagline", "tagline"),
]


def _normalized_text(value: object) -> str:
    return str(value or "").strip()


def _parse_text_list(value: object) -> list[str]:
    raw = _normalized_text(value)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(parsed, list):
        return []

    unique: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        text = _normalized_text(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            unique.append(text)
    return unique


def _list_text(value: object) -> str:
    return ", ".join(_parse_text_list(value))


def _prepare_text_artifact_columns(rows: pd.DataFrame) -> pd.DataFrame:
    prepared = rows.copy()
    def _column_or_empty(name: str) -> pd.Series:
        if name in prepared.columns:
            return prepared[name]
        return pd.Series([""] * len(prepared), index=prepared.index, dtype="object")

    prepared["keywords_augmented_text"] = _column_or_empty("keywords_augmented_json").map(_list_text)
    prepared["tone_tags_text"] = _column_or_empty("tone_tags_json").map(_list_text)
    prepared["audience_tags_text"] = _column_or_empty("audience_tags_json").map(_list_text)
    prepared["source_tags_text"] = _column_or_empty("source_tags_json").map(_list_text)
    return prepared


def _active_data_path() -> Path:
    return ENRICHED_DATA_PATH if ENRICHED_DATA_PATH.exists() else DATA_PATH


def build_movie_embedding_text(row: dict[str, object]) -> str:
    parts: list[str] = []
    for label, key in EMBEDDING_TEXT_FIELDS:
        value = _normalized_text(row.get(key, ""))
        if value:
            parts.append(f"{label}: {value}")
    if parts:
        return "\n".join(parts)
    return _normalized_text(row.get("title", ""))


def build_retrieval_index() -> None:
    active_data_path = _active_data_path()
    movies = ensure_dataset_columns(pd.read_csv(active_data_path).fillna(""))
    movies = _prepare_text_artifact_columns(movies)
    if RETRIEVAL_DB_PATH.exists():
        RETRIEVAL_DB_PATH.unlink()

    connection = sqlite3.connect(str(RETRIEVAL_DB_PATH))
    try:
        connection.executescript(
            """
            CREATE TABLE movies (
                tmdb_id INTEGER PRIMARY KEY,
                title TEXT,
                original_title TEXT,
                year INTEGER,
                genres TEXT,
                overview TEXT,
                tagline TEXT,
                keywords TEXT,
                director TEXT,
                top_cast TEXT,
                original_language TEXT,
                production_countries TEXT,
                vote_average REAL,
                vote_count INTEGER,
                title_root TEXT
            );

            CREATE VIRTUAL TABLE movies_fts USING fts5(
                tmdb_id UNINDEXED,
                title,
                original_title,
                alternative_titles,
                genres,
                keywords,
                keywords_augmented,
                tone_tags,
                audience_tags,
                source_tags,
                director,
                top_cast,
                collection_name,
                production_companies,
                original_language,
                production_countries,
                spoken_languages,
                us_rating
            );

            CREATE INDEX idx_movies_title ON movies(title);
            """
        )

        rows = []
        fts_rows = []
        for row in movies.itertuples(index=False):
            rows.append(
                (
                    int(row.tmdb_id),
                    _normalized_text(row.title),
                    _normalized_text(row.original_title),
                    int(row.year),
                    _normalized_text(row.genres),
                    _normalized_text(row.overview),
                    _normalized_text(row.tagline),
                    _normalized_text(row.keywords),
                    _normalized_text(row.director),
                    _normalized_text(row.top_cast),
                    _normalized_text(row.original_language),
                    _normalized_text(row.production_countries),
                    float(row.vote_average),
                    int(row.vote_count),
                    title_root(row.title),
                )
            )
            fts_rows.append(
                (
                    int(row.tmdb_id),
                    _normalized_text(row.title),
                    _normalized_text(row.original_title),
                    _normalized_text(row.alternative_titles),
                    _normalized_text(row.genres),
                    _normalized_text(row.keywords),
                    _normalized_text(row.keywords_augmented_text),
                    _normalized_text(row.tone_tags_text),
                    _normalized_text(row.audience_tags_text),
                    _normalized_text(row.source_tags_text),
                    _normalized_text(row.director),
                    _normalized_text(row.top_cast),
                    _normalized_text(row.collection_name),
                    _normalized_text(row.production_companies),
                    _normalized_text(row.original_language),
                    _normalized_text(row.production_countries),
                    _normalized_text(row.spoken_languages),
                    _normalized_text(row.us_rating),
                )
            )

        connection.executemany(
            """
            INSERT INTO movies (
                tmdb_id, title, original_title, year, genres, overview, tagline,
                keywords, director, top_cast, original_language, production_countries,
                vote_average, vote_count, title_root
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.executemany(
            """
            INSERT INTO movies_fts (
                tmdb_id, title, original_title, alternative_titles, genres, keywords, keywords_augmented,
                tone_tags, audience_tags, source_tags, director, top_cast, collection_name,
                production_companies, original_language, production_countries, spoken_languages, us_rating
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fts_rows,
        )
        connection.commit()
    finally:
        connection.close()

    write_json(
        RETRIEVAL_DB_META_PATH,
        {
            **dataset_fingerprint(active_data_path),
            "artifact_type": "sqlite_fts",
            "row_count": int(len(movies)),
        },
    )


def build_movie_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    from sentence_transformers import SentenceTransformer

    active_data_path = _active_data_path()
    movies = ensure_dataset_columns(pd.read_csv(active_data_path).fillna(""))
    movies = _prepare_text_artifact_columns(movies)
    documents = [build_movie_embedding_text(row._asdict()) for row in movies.itertuples(index=False)]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_PATH, embeddings)
    EMBEDDING_IDS_PATH.write_text(json.dumps(movies["tmdb_id"].astype(int).tolist(), indent=2) + "\n")
    write_json(
        EMBEDDING_META_PATH,
        {
            **dataset_fingerprint(active_data_path),
            "artifact_type": "semantic_embeddings",
            "embedding_model": model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "row_count": int(len(movies)),
        },
    )


def refresh_retrieval_artifacts() -> None:
    build_retrieval_index()
    build_movie_embeddings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local text retrieval artifacts.")
    parser.add_argument(
        "--target",
        choices=("all", "index", "embeddings"),
        default="all",
        help="Which text artifacts to rebuild.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name to use when rebuilding embeddings.",
    )
    args = parser.parse_args()

    if args.target in {"all", "index"}:
        build_retrieval_index()
        print(f"Built retrieval index at {RETRIEVAL_DB_PATH}")
    if args.target in {"all", "embeddings"}:
        build_movie_embeddings(model_name=args.embedding_model)
        print(f"Built movie embeddings at {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
