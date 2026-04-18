from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np

from retrieval import (
    ACTIVE_DATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_IDS_PATH,
    EMBEDDING_META_PATH,
    EMBEDDINGS_PATH,
    load_json,
    metadata_matches,
)


@lru_cache(maxsize=1)
def semantic_ready() -> bool:
    if not EMBEDDINGS_PATH.exists() or not EMBEDDING_IDS_PATH.exists() or not EMBEDDING_META_PATH.exists():
        return False
    try:
        metadata = load_json(EMBEDDING_META_PATH)
    except Exception:
        return False
    return metadata_matches(metadata, ACTIVE_DATA_PATH)


@lru_cache(maxsize=1)
def _get_encoder():
    from sentence_transformers import SentenceTransformer

    model_name = DEFAULT_EMBEDDING_MODEL
    if EMBEDDING_META_PATH.exists():
        try:
            model_name = str(load_json(EMBEDDING_META_PATH).get("embedding_model") or model_name)
        except Exception:
            model_name = DEFAULT_EMBEDDING_MODEL
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    tmdb_ids = np.array(json.loads(EMBEDDING_IDS_PATH.read_text()), dtype=np.int64)
    return embeddings, tmdb_ids


def _encode_query(query_text: str) -> np.ndarray:
    vector = _get_encoder().encode([query_text], normalize_embeddings=True)
    return np.asarray(vector[0], dtype=np.float32)


def search_semantic(query_text: str, exclude_ids: set[int] | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if not semantic_ready():
        return []

    query_text = str(query_text or "").strip()
    if not query_text:
        return []

    embeddings, tmdb_ids = _load_embeddings()
    query_vector = _encode_query(query_text)
    similarities = embeddings @ query_vector

    exclude_ids = exclude_ids or set()
    if exclude_ids:
        excluded = np.isin(tmdb_ids, np.array(sorted(exclude_ids), dtype=np.int64))
        similarities = similarities.copy()
        similarities[excluded] = -np.inf

    top_n = min(int(limit), len(similarities))
    if top_n <= 0:
        return []

    candidate_indices = np.argpartition(-similarities, range(top_n))[:top_n]
    candidate_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]

    results: list[dict[str, Any]] = []
    for idx in candidate_indices:
        score = float(similarities[idx])
        if not np.isfinite(score):
            continue
        normalized = max(0.0, min(1.0, (score + 1.0) / 2.0))
        results.append(
            {
                "tmdb_id": int(tmdb_ids[idx]),
                "semantic_score": normalized,
                "raw_semantic_score": score,
            }
        )
    return results
