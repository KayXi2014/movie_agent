from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import httpx

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
EMBEDDING_PROVIDER = "huggingface"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_EMBEDDING_BATCH_SIZE = 16
HF_FEATURE_EXTRACTION_BASE_URL = "https://router.huggingface.co/hf-inference/models"
PLACEHOLDER_HF_KEYS = {"", "your_key_here", "your_key", "xxxxx", "xxxx", "replace_me"}


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


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def _hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if token.lower() in PLACEHOLDER_HF_KEYS:
        raise RuntimeError(
            "HF_TOKEN is missing or still set to a placeholder. "
            "Create a Hugging Face token with Inference Providers access and export HF_TOKEN."
        )
    return token


def _hf_embedding_url(model_name: str) -> str:
    base_url = os.getenv("HF_EMBEDDING_BASE_URL", HF_FEATURE_EXTRACTION_BASE_URL).rstrip("/")
    return f"{base_url}/{model_name}/pipeline/feature-extraction"


def _extract_hf_embeddings(payload: object, *, expected_count: int) -> list[list[float]]:
    if not isinstance(payload, list):
        raise ValueError(f"Hugging Face embedding response had unexpected type: {type(payload).__name__}")
    if not payload:
        raise ValueError("Hugging Face embedding response was empty")
    if all(isinstance(item, (int, float)) for item in payload):
        embeddings = [payload]
    else:
        embeddings = payload
    if len(embeddings) != expected_count:
        raise ValueError(f"Hugging Face returned {len(embeddings)} embeddings for {expected_count} texts")
    for vector in embeddings:
        if not isinstance(vector, list) or not all(isinstance(item, (int, float)) for item in vector):
            raise ValueError("Hugging Face embedding response had an unexpected nested shape")
    return embeddings


def _hf_embedding_error(exc: Exception, *, model_name: str, batch_start: int | None = None) -> RuntimeError:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text
        if status in {401, 403}:
            return RuntimeError(
                "Hugging Face rejected the embedding request. Check that HF_TOKEN is valid and has "
                f"Inference Providers access. model={model_name} batch_start={batch_start} status={status}"
            )
        if status == 429:
            return RuntimeError(
                "Hugging Face rate-limited the embedding request. Lower --embedding-batch-size or retry later. "
                f"model={model_name} batch_start={batch_start}"
            )
        return RuntimeError(
            f"Hugging Face embedding request failed. model={model_name} batch_start={batch_start} "
            f"status={status} detail={detail!r}"
        )
    return RuntimeError(f"Hugging Face embedding request failed: {exc!r}. model={model_name} batch_start={batch_start}")


def _post_hf_embeddings(client: httpx.Client, *, model_name: str, texts: list[str]) -> list[list[float]]:
    response = client.post(
        _hf_embedding_url(model_name),
        json={"inputs": texts, "options": {"wait_for_model": True}},
    )
    response.raise_for_status()
    return _extract_hf_embeddings(response.json(), expected_count=len(texts))


def check_hf_embedding_access(*, model_name: str = DEFAULT_EMBEDDING_MODEL, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
    token = _hf_token()
    timeout = float(os.getenv("HF_EMBED_TIMEOUT_SECONDS", "60"))
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=timeout) as client:
        try:
            vectors = _post_hf_embeddings(client, model_name=model_name, texts=["embedding preflight"])
        except Exception as exc:
            raise _hf_embedding_error(exc, model_name=model_name) from exc
    if not vectors or len(vectors[0]) != embedding_dim:
        raise RuntimeError(
            f"Hugging Face embedding preflight returned dimension {len(vectors[0]) if vectors else 0}; expected {embedding_dim}."
        )


def _build_hf_embeddings(
    documents: list[str],
    *,
    model_name: str,
    embedding_dim: int,
    batch_size: int,
) -> np.ndarray:
    token = _hf_token()
    timeout = float(os.getenv("HF_EMBED_TIMEOUT_SECONDS", "60"))
    vectors: list[list[float]] = []
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=timeout) as client:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            try:
                vectors.extend(_post_hf_embeddings(client, model_name=model_name, texts=batch))
            except Exception as exc:
                raise _hf_embedding_error(exc, model_name=model_name, batch_start=start) from exc

    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Expected embeddings with dimension {embedding_dim}, got shape {embeddings.shape}")
    return _normalize_embeddings(embeddings).astype(np.float32, copy=False)


def build_movie_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    embedding_dim: int | None = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> None:
    active_data_path = _active_data_path()
    movies = ensure_dataset_columns(pd.read_csv(active_data_path).fillna(""))
    movies = _prepare_text_artifact_columns(movies)
    documents = [build_movie_embedding_text(row._asdict()) for row in movies.itertuples(index=False)]

    resolved_dim = int(embedding_dim or DEFAULT_EMBEDDING_DIM)
    embeddings = _build_hf_embeddings(
        documents,
        model_name=model_name,
        embedding_dim=resolved_dim,
        batch_size=max(1, int(embedding_batch_size)),
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    EMBEDDING_IDS_PATH.write_text(json.dumps(movies["tmdb_id"].astype(int).tolist(), indent=2) + "\n")
    write_json(
        EMBEDDING_META_PATH,
        {
            **dataset_fingerprint(active_data_path),
            "artifact_type": "semantic_embeddings",
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": model_name,
            "embedding_dim": resolved_dim,
            "row_count": int(len(movies)),
        },
    )


def refresh_retrieval_artifacts(
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dim: int | None = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> None:
    build_retrieval_index()
    build_movie_embeddings(
        model_name=embedding_model,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
    )


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
        help="Embedding model name to use when rebuilding embeddings.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Embedding dimension for Hugging Face embedding rebuilds.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Batch size for Hugging Face embedding rebuilds.",
    )
    parser.add_argument(
        "--check-embeddings",
        action="store_true",
        help="Run a one-text Hugging Face embedding preflight and exit.",
    )
    args = parser.parse_args()

    if args.check_embeddings:
        try:
            check_hf_embedding_access(model_name=args.embedding_model, embedding_dim=args.embedding_dim)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Hugging Face embedding preflight succeeded for model={args.embedding_model}")
        return

    if args.target in {"all", "index"}:
        build_retrieval_index()
        print(f"Built retrieval index at {RETRIEVAL_DB_PATH}")
    if args.target in {"all", "embeddings"}:
        try:
            build_movie_embeddings(
                model_name=args.embedding_model,
                embedding_dim=args.embedding_dim,
                embedding_batch_size=args.embedding_batch_size,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Built movie embeddings at {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
