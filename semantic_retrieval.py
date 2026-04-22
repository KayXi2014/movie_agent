from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import httpx
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

logger = logging.getLogger(__name__)
EMBEDDING_PROVIDER = "huggingface"
HF_FEATURE_EXTRACTION_BASE_URL = "https://router.huggingface.co/hf-inference/models"
PLACEHOLDER_HF_KEYS = {"", "your_key_here", "your_key", "xxxxx", "xxxx", "replace_me"}


@lru_cache(maxsize=1)
def semantic_ready() -> bool:
    if not EMBEDDINGS_PATH.exists() or not EMBEDDING_IDS_PATH.exists() or not EMBEDDING_META_PATH.exists():
        return False
    try:
        metadata = load_json(EMBEDDING_META_PATH)
    except Exception:
        return False
    return metadata_matches(metadata, ACTIVE_DATA_PATH) and metadata.get("embedding_provider") == EMBEDDING_PROVIDER


def semantic_runtime_status() -> dict[str, Any]:
    metadata = _embedding_metadata()
    if not semantic_ready():
        return {
            "ready": False,
            "reason": "artifacts_unavailable",
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": _model_name(metadata),
            "embedding_dim": _embedding_dim(metadata),
        }

    if os.getenv("HF_TOKEN", "").strip().lower() in PLACEHOLDER_HF_KEYS:
        return {
            "ready": False,
            "reason": "hf_token_missing",
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": _model_name(metadata),
            "embedding_dim": _embedding_dim(metadata),
        }

    return {
        "ready": True,
        "reason": "ok",
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": _model_name(metadata),
        "embedding_dim": _embedding_dim(metadata),
    }


def _embedding_metadata() -> dict[str, Any]:
    if EMBEDDING_META_PATH.exists():
        try:
            return load_json(EMBEDDING_META_PATH)
        except Exception:
            return {}
    return {}


def _model_name(metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata if metadata is not None else _embedding_metadata()
    if metadata.get("embedding_provider") != EMBEDDING_PROVIDER:
        return DEFAULT_EMBEDDING_MODEL
    return str(metadata.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)


def _embedding_dim(metadata: dict[str, Any] | None = None) -> int | None:
    metadata = metadata if metadata is not None else _embedding_metadata()
    raw_dim = metadata.get("embedding_dim")
    if raw_dim in (None, ""):
        return None
    try:
        return int(raw_dim)
    except (TypeError, ValueError):
        return None


def _hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if token.lower() in PLACEHOLDER_HF_KEYS:
        raise RuntimeError("HF_TOKEN is required for Hugging Face query embeddings")
    return token


@lru_cache(maxsize=1)
def _load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    tmdb_ids = np.array(json.loads(EMBEDDING_IDS_PATH.read_text()), dtype=np.int64)
    return embeddings, tmdb_ids


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Embedding vector has zero norm")
    return (vector / norm).astype(np.float32, copy=False)


def _hf_embedding_url(model_name: str) -> str:
    base_url = os.getenv("HF_EMBEDDING_BASE_URL", HF_FEATURE_EXTRACTION_BASE_URL).rstrip("/")
    return f"{base_url}/{model_name}/pipeline/feature-extraction"


def _extract_hf_embedding(payload: object) -> list[float]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Hugging Face embedding response was empty or malformed")
    if all(isinstance(item, (int, float)) for item in payload):
        return payload
    first = payload[0]
    if isinstance(first, list) and all(isinstance(item, (int, float)) for item in first):
        return first
    raise ValueError("Hugging Face embedding response had an unexpected nested shape")


def _encode_hf_query(query_text: str, model_name: str, expected_dim: int | None) -> np.ndarray:
    if expected_dim is None:
        raise RuntimeError("Hugging Face embedding artifacts must declare embedding_dim")
    response = httpx.post(
        _hf_embedding_url(model_name),
        headers={"Authorization": f"Bearer {_hf_token()}"},
        json={"inputs": [query_text], "options": {"wait_for_model": True}},
        timeout=float(os.getenv("HF_SEMANTIC_TIMEOUT_S", os.getenv("HF_EMBED_TIMEOUT_SECONDS", "2.0"))),
    )
    response.raise_for_status()
    query_vector = np.asarray(_extract_hf_embedding(response.json()), dtype=np.float32)
    if query_vector.shape[0] != expected_dim:
        raise ValueError(f"Hugging Face query embedding dimension {query_vector.shape[0]} does not match artifact dimension {expected_dim}")
    return _normalize_vector(query_vector)


def _encode_query(query_text: str) -> np.ndarray:
    metadata = _embedding_metadata()
    model_name = _model_name(metadata)
    expected_dim = _embedding_dim(metadata)
    return _encode_hf_query(query_text, model_name, expected_dim)


@lru_cache(maxsize=1)
def warm_semantic_runtime() -> bool:
    if not semantic_ready():
        return False
    try:
        _load_embeddings()
        return os.getenv("HF_TOKEN", "").strip().lower() not in PLACEHOLDER_HF_KEYS
    except Exception as exc:
        logger.warning("Semantic runtime warm-up failed: %r", exc)
        return False


def search_semantic(query_text: str, exclude_ids: set[int] | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if not semantic_ready():
        return []

    query_text = str(query_text or "").strip()
    if not query_text:
        return []

    embeddings, tmdb_ids = _load_embeddings()
    try:
        query_vector = _encode_query(query_text)
    except Exception:
        return []
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
