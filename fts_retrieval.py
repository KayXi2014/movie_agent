from __future__ import annotations

import re
import sqlite3
import threading
from functools import lru_cache
from typing import Any

from retrieval import ACTIVE_DATA_PATH, RETRIEVAL_DB_META_PATH, RETRIEVAL_DB_PATH, load_json, metadata_matches


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _fts_query_terms(query_text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(str(query_text or "").lower()) if len(token) > 1]


def _fts_query(query_text: str) -> str:
    terms = _fts_query_terms(query_text)
    if not terms:
        return ""
    return " OR ".join(f'"{term}"' for term in terms[:24])


@lru_cache(maxsize=1)
def fts_ready() -> bool:
    if not RETRIEVAL_DB_PATH.exists() or not RETRIEVAL_DB_META_PATH.exists():
        return False
    try:
        metadata = load_json(RETRIEVAL_DB_META_PATH)
    except Exception:
        return False
    return metadata_matches(metadata, ACTIVE_DATA_PATH)


@lru_cache(maxsize=8)
def _get_connection(thread_id: int) -> sqlite3.Connection:
    connection = sqlite3.connect(str(RETRIEVAL_DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def search_fts(query_text: str, exclude_ids: set[int] | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if not fts_ready():
        return []

    match_query = _fts_query(query_text)
    if not match_query:
        return []

    exclude_ids = exclude_ids or set()
    params: list[Any] = [match_query]
    where_parts = ["movies_fts MATCH ?"]

    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        where_parts.append(f"m.tmdb_id NOT IN ({placeholders})")
        params.extend(sorted(exclude_ids))

    params.append(int(limit))
    sql = f"""
        SELECT
            m.tmdb_id,
            -bm25(movies_fts) AS lexical_score
        FROM movies_fts
        JOIN movies m ON m.tmdb_id = movies_fts.tmdb_id
        WHERE {' AND '.join(where_parts)}
        ORDER BY bm25(movies_fts)
        LIMIT ?
    """

    rows = _get_connection(threading.get_ident()).execute(sql, params).fetchall()
    if not rows:
        return []

    raw_scores = [float(row["lexical_score"]) for row in rows]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    span = max(max_score - min_score, 1e-6)

    results: list[dict[str, Any]] = []
    for row in rows:
        raw_score = float(row["lexical_score"])
        normalized = (raw_score - min_score) / span if span > 0 else 1.0
        results.append(
            {
                "tmdb_id": int(row["tmdb_id"]),
                "lexical_score": float(normalized),
                "raw_lexical_score": raw_score,
            }
        )
    return results
