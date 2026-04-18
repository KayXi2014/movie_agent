from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHORTLIST_SIZE = 5
FTS_LIMIT = 20
SEMANTIC_LIMIT = 20
MERGED_POOL_SIZE = 24
SECOND_STAGE_POOL_SIZE = 16
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "tmdb_top1000_movies.csv"
ENRICHED_DATA_PATH = ROOT / "tmdb_top1000_movies_enriched.csv"
ACTIVE_DATA_PATH = ENRICHED_DATA_PATH if ENRICHED_DATA_PATH.exists() else DATA_PATH

RETRIEVAL_DB_PATH = ROOT / "movies.sqlite"
RETRIEVAL_DB_META_PATH = ROOT / "movies.sqlite.meta.json"
EMBEDDINGS_PATH = ROOT / "movie_embeddings.npy"
EMBEDDING_IDS_PATH = ROOT / "movie_embedding_ids.json"
EMBEDDING_META_PATH = ROOT / "movie_embedding_meta.json"

TEXT_COLUMNS = [
    "title",
    "original_title",
    "genres",
    "overview",
    "tagline",
    "keywords",
    "director",
    "top_cast",
    "original_language",
    "production_countries",
]
TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "could",
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "like",
    "love",
    "movie",
    "movies",
    "no",
    "of",
    "on",
    "or",
    "sound",
    "something",
    "stories",
    "story",
    "that",
    "the",
    "to",
    "want",
    "watch",
    "would",
    "with",
}

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(path: Path = ACTIVE_DATA_PATH) -> dict[str, Any]:
    try:
        source_relpath = str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        source_relpath = path.name
    return {
        "source_path": source_relpath,
        "source_name": path.name,
        "source_sha256": file_sha256(path),
        "source_size": path.stat().st_size,
    }


def metadata_matches(metadata: dict[str, Any] | None, path: Path = ACTIVE_DATA_PATH) -> bool:
    if not metadata:
        return False
    fingerprint = dataset_fingerprint(path)
    return (
        metadata.get("source_name") == fingerprint["source_name"]
        and metadata.get("source_sha256") == fingerprint["source_sha256"]
        and int(metadata.get("source_size", -1)) == int(fingerprint["source_size"])
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("sci-fi", "science fiction")
    text = text.replace("sci fi", "science fiction")
    text = text.replace("superpowers", "superpower")
    return text


def tokenize(value: Any) -> set[str]:
    return {token for token in TOKEN_RE.findall(normalize_text(value)) if token not in STOP_WORDS}


def split_csvish(value: Any) -> set[str]:
    return {part.strip().lower() for part in str(value or "").split(",") if part.strip()}


def _intent_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value:
        items = [value]
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]


def normalize_intent(intent: dict[str, Any] | None, preferences: str) -> dict[str, Any]:
    payload = dict(intent or {})
    query_text = str(payload.get("query_text") or "").strip() or str(preferences or "").strip()
    must_have = _intent_list(payload.get("must_have"))
    avoid = _intent_list(payload.get("avoid"))
    tone = _intent_list(payload.get("tone"))
    return {
        "query_text": query_text,
        "must_have": must_have,
        "avoid": avoid,
        "tone": tone,
    }


def title_root(title: Any) -> str:
    text = normalize_text(title)
    if not text:
        return ""
    for delimiter in (" - ", ":"):
        if delimiter in text:
            text = text.split(delimiter, 1)[0]
    text = re.sub(r"\b(part|chapter|episode|vol|volume)\b.*$", "", text).strip()
    text = re.sub(r"\b\d+\b$", "", text).strip()
    return text


def build_movie_embedding_text(row: dict[str, Any]) -> str:
    parts = [
        f"Title: {row.get('title', '')}",
        f"Original Title: {row.get('original_title', '')}",
        f"Genres: {row.get('genres', '')}",
        f"Keywords: {row.get('keywords', '')}",
        f"Overview: {row.get('overview', '')}",
        f"Tagline: {row.get('tagline', '')}",
        f"Director: {row.get('director', '')}",
        f"Top Cast: {row.get('top_cast', '')}",
        f"Original Language: {row.get('original_language', '')}",
        f"Production Countries: {row.get('production_countries', '')}",
    ]
    return "\n".join(part.strip() for part in parts if part.strip())


def prepare_movies(df: pd.DataFrame) -> pd.DataFrame:
    movies = df.copy()
    movies["title_root"] = movies["title"].map(title_root)
    movies["genres_set"] = movies["genres"].map(split_csvish)
    movies["keywords_set"] = movies["keywords"].map(split_csvish)
    movies["cast_set"] = movies["top_cast"].map(split_csvish)
    movies["director_set"] = movies["director"].map(split_csvish)
    movies["search_blob"] = movies[TEXT_COLUMNS].agg(" ".join, axis=1).map(normalize_text)
    movies["search_blob_tokens"] = movies["search_blob"].map(tokenize)
    return movies


TOP_MOVIES = pd.read_csv(ACTIVE_DATA_PATH).fillna("")
MOVIES = prepare_movies(TOP_MOVIES)
MOVIES_BY_EXACT_TITLE = MOVIES.groupby("title", sort=False)
MOVIES_BY_TMDB_ID = MOVIES.set_index("tmdb_id", drop=False)


def build_retrieval_index() -> None:
    movies = pd.read_csv(ACTIVE_DATA_PATH).fillna("")
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
                genres,
                overview,
                tagline,
                keywords,
                director,
                top_cast
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
                    str(row.title),
                    str(row.original_title),
                    int(row.year),
                    str(row.genres),
                    str(row.overview),
                    str(row.tagline),
                    str(row.keywords),
                    str(row.director),
                    str(row.top_cast),
                    str(row.original_language),
                    str(row.production_countries),
                    float(row.vote_average),
                    int(row.vote_count),
                    title_root(row.title),
                )
            )
            fts_rows.append(
                (
                    int(row.tmdb_id),
                    str(row.title),
                    str(row.genres),
                    str(row.overview),
                    str(row.tagline),
                    str(row.keywords),
                    str(row.director),
                    str(row.top_cast),
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
                tmdb_id, title, genres, overview, tagline, keywords, director, top_cast
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fts_rows,
        )
        connection.commit()
    finally:
        connection.close()

    write_json(
        RETRIEVAL_DB_META_PATH,
        {
            **dataset_fingerprint(ACTIVE_DATA_PATH),
            "artifact_type": "sqlite_fts",
            "row_count": int(len(movies)),
        },
    )


def build_movie_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    from sentence_transformers import SentenceTransformer

    movies = pd.read_csv(ACTIVE_DATA_PATH).fillna("")
    documents = [build_movie_embedding_text(row._asdict()) for row in movies.itertuples(index=False)]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_PATH, embeddings)
    EMBEDDING_IDS_PATH.write_text(json.dumps(movies["tmdb_id"].astype(int).tolist(), indent=2) + "\n")
    write_json(
        EMBEDDING_META_PATH,
        {
            **dataset_fingerprint(ACTIVE_DATA_PATH),
            "artifact_type": "semantic_embeddings",
            "embedding_model": model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "row_count": int(len(movies)),
        },
    )


def refresh_retrieval_artifacts() -> None:
    build_retrieval_index()
    build_movie_embeddings()


def normalize_history_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        tmdb_id = item.get("tmdb_id")
        name = item.get("name", "")
    else:
        tmdb_id = None
        name = item
    exact_name = str(name or "")
    if tmdb_id is None and not exact_name:
        return None
    result: dict[str, Any] = {"name": exact_name}
    if tmdb_id is not None:
        try:
            result["tmdb_id"] = int(tmdb_id)
        except (TypeError, ValueError):
            pass
    return result


def normalize_history(history: list[Any]) -> tuple[tuple[int | None, str], ...]:
    normalized = []
    for item in history:
        normalized_item = normalize_history_item(item)
        if normalized_item is not None:
            normalized.append((normalized_item.get("tmdb_id"), normalized_item["name"]))
    return tuple(sorted(set(normalized), key=lambda item: (item[0] is None, item[0], item[1])))


def history_rows(history: tuple[tuple[int | None, str], ...]) -> pd.DataFrame:
    rows = []
    seen_ids: set[int] = set()
    for tmdb_id, name in history:
        if tmdb_id is not None and tmdb_id in MOVIES_BY_TMDB_ID.index:
            row = MOVIES_BY_TMDB_ID.loc[tmdb_id]
            if int(row.tmdb_id) not in seen_ids:
                rows.append(row.to_frame().T)
                seen_ids.add(int(row.tmdb_id))
            continue
        if name in MOVIES_BY_EXACT_TITLE.groups:
            group = MOVIES_BY_EXACT_TITLE.get_group(name)
            unseen = group[~group["tmdb_id"].isin(seen_ids)]
            if not unseen.empty:
                rows.append(unseen)
                seen_ids.update(unseen["tmdb_id"].astype(int).tolist())
    if not rows:
        return MOVIES.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tmdb_id"])


def derive_history_signals(history_df: pd.DataFrame) -> dict[str, set[str]]:
    genre_counts: Counter[str] = Counter()
    cast_counts: Counter[str] = Counter()
    director_counts: Counter[str] = Counter()
    watched_roots: set[str] = set()
    for row in history_df.itertuples():
        genre_counts.update(item for item in row.genres_set if item)
        cast_counts.update(item for item in row.cast_set if item)
        director_counts.update(item for item in row.director_set if item)
        if row.title_root:
            watched_roots.add(row.title_root)
    return {
        "liked_genres": {name for name, _ in genre_counts.most_common(3)},
        "liked_cast": {name for name, count in cast_counts.most_common(6) if count >= 2},
        "liked_directors": {name for name, count in director_counts.most_common(4) if count >= 2},
        "watched_roots": watched_roots,
    }


def history_prompt_text(history_count: int) -> str:
    return "none" if history_count <= 0 else f"known_watch_history_count={history_count}; use history primarily to avoid re-recommending already watched titles"


def build_retrieval_profile(preferences: str, history: tuple[tuple[int | None, str], ...], intent: dict[str, Any] | None = None) -> dict[str, Any]:
    history_df = history_rows(history)
    normalized_intent = normalize_intent(intent, preferences)
    raw_preference_tokens = tokenize(preferences)
    history_signals = derive_history_signals(history_df)
    query_tokens = tokenize(normalized_intent["query_text"])
    intent_positive_tokens = tokenize(" ".join(normalized_intent["must_have"] + normalized_intent["tone"]))
    intent_negative_tokens = tokenize(" ".join(normalized_intent["avoid"]))
    return {
        "intent": normalized_intent,
        "history_df": history_df,
        "history_count": len(history),
        "history_exclusion_text": history_prompt_text(len(history)),
        "history_titles": {name for _, name in history if name},
        "history_tmdb_ids": {tmdb_id for tmdb_id, _ in history if tmdb_id is not None},
        "raw_preference_tokens": raw_preference_tokens,
        "query_tokens": query_tokens,
        "intent_positive_tokens": intent_positive_tokens,
        "intent_negative_tokens": intent_negative_tokens,
        **history_signals,
    }


def build_prompt_profile(preferences: str, retrieval_profile: dict[str, Any]) -> dict[str, Any]:
    intent = retrieval_profile["intent"]
    preferred_themes = intent["must_have"][:6] + [tone for tone in intent["tone"][:3] if tone not in intent["must_have"][:6]]
    if not preferred_themes:
        fallback_tokens = sorted(token for token in tokenize(preferences) if len(token) > 2 and token not in {"science", "fiction"})[:6]
        preferred_themes = fallback_tokens
    return {
        "target_genres": [],
        "preferred_tones": intent["tone"][:4],
        "preferred_themes": preferred_themes,
        "avoid": intent["avoid"][:6],
        "history_signals": {
            "liked_genres": sorted(retrieval_profile["liked_genres"])[:6],
            "liked_directors": sorted(retrieval_profile["liked_directors"])[:4],
            "liked_cast": sorted(retrieval_profile["liked_cast"])[:6],
        },
    }


def build_query_text(preferences: str, retrieval_profile: dict[str, Any]) -> str:
    intent = retrieval_profile["intent"]
    primary_query = ", ".join(intent["must_have"][:6] + intent["tone"][:4]).strip(", ")
    parts = [primary_query or intent["query_text"]]
    if intent["must_have"]:
        parts.append("must have: " + ", ".join(intent["must_have"][:6]))
    if intent["tone"]:
        parts.append("tone: " + ", ".join(intent["tone"][:4]))
    if intent["avoid"]:
        parts.append("avoid: " + ", ".join(intent["avoid"][:6]))
    return " | ".join(part for part in parts if part)


def appeal_score(row: pd.Series) -> float:
    score = min(float(row["vote_average"]) or 0.0, 10.0) * 0.35
    score += min(float(row["vote_count"]) or 0.0, 8000.0) / 4000.0
    if row["tagline"]:
        score += 0.8
    hook_terms = {"friendship", "heist", "mission", "rivalry", "secret", "love", "romance", "detective", "superhero", "haunted", "ghost", "survival", "future", "betrayal", "adventure", "family", "team"}
    score += 0.35 * len(hook_terms.intersection(row["search_blob_tokens"]))
    if row["director"]:
        score += 0.2
    if row["top_cast"]:
        score += 0.3
    return score


def history_affinity_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    score = 1.2 * len(retrieval_profile["liked_genres"].intersection(row["genres_set"]))
    score += 1.8 * len(retrieval_profile["liked_directors"].intersection(row["director_set"]))
    score += 1.0 * len(retrieval_profile["liked_cast"].intersection(row["cast_set"]))
    return score


def intent_alignment_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    positive_tokens = (
        retrieval_profile["raw_preference_tokens"]
        | retrieval_profile["query_tokens"]
        | retrieval_profile["intent_positive_tokens"]
    )
    negative_tokens = retrieval_profile["intent_negative_tokens"]
    tokens = row["search_blob_tokens"]

    positive_overlap = len(positive_tokens.intersection(tokens))
    negative_overlap = len(negative_tokens.intersection(tokens))

    score = 0.7 * positive_overlap
    score += 0.35 * len(retrieval_profile["intent_positive_tokens"].intersection(row["keywords_set"]))
    if retrieval_profile["intent"]["must_have"]:
        score += 0.25 * sum(
            1 for phrase in retrieval_profile["intent"]["must_have"] if normalize_text(phrase) in row["search_blob"]
        )
    score -= 2.4 * negative_overlap
    if retrieval_profile["intent"]["avoid"]:
        score -= 1.2 * sum(
            1 for phrase in retrieval_profile["intent"]["avoid"] if normalize_text(phrase) in row["search_blob"]
        )
    if negative_overlap and positive_overlap <= 1:
        score -= 1.5
    return score


def novelty_penalty(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    penalty = 4.0 if row["title_root"] and row["title_root"] in retrieval_profile["watched_roots"] else 0.0
    if retrieval_profile["intent_negative_tokens"] and retrieval_profile["intent_negative_tokens"].intersection(row["search_blob_tokens"]):
        penalty += 4.0
    if retrieval_profile["intent"]["avoid"]:
        penalty += 1.0 * sum(
            1 for phrase in retrieval_profile["intent"]["avoid"] if normalize_text(phrase) in row["search_blob"]
        )
    return penalty


def _normalize_component(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def hybrid_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    history_component = _normalize_component(history_affinity_score(row, retrieval_profile), 6.0)
    appeal_component = _normalize_component(appeal_score(row), 8.0)
    novelty_component = _normalize_component(novelty_penalty(row, retrieval_profile), 7.0)
    intent_component = _normalize_component(max(0.0, intent_alignment_score(row, retrieval_profile)), 8.0)
    intent_penalty = _normalize_component(max(0.0, -intent_alignment_score(row, retrieval_profile)), 5.0)
    return (
        0.42 * float(row.get("semantic_score", 0.0))
        + 0.23 * float(row.get("fts_score", 0.0))
        + 0.15 * intent_component
        + 0.15 * history_component
        + 0.10 * appeal_component
        - 0.05 * novelty_component
        - 0.08 * intent_penalty
    )


def local_fallback_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if int(row["tmdb_id"]) in retrieval_profile["history_tmdb_ids"] or row["title"] in retrieval_profile["history_titles"]:
        return -10_000.0

    query_terms = retrieval_profile["raw_preference_tokens"] | retrieval_profile["query_tokens"] | retrieval_profile["intent_positive_tokens"]
    overlap = len(query_terms.intersection(row["search_blob_tokens"]))
    negative_hits = len(retrieval_profile["intent_negative_tokens"].intersection(row["search_blob_tokens"]))
    alignment_score = intent_alignment_score(row, retrieval_profile)

    return (
        float(overlap)
        + 0.9 * alignment_score
        + 0.8 * history_affinity_score(row, retrieval_profile)
        + 0.4 * appeal_score(row)
        - 2.1 * negative_hits
        - novelty_penalty(row, retrieval_profile)
    )


def diversify_candidates(candidates: pd.DataFrame, limit: int) -> pd.DataFrame:
    selected: list[tuple[float, int]] = []
    selected_roots: set[str] = set()
    selected_directors: set[str] = set()

    for row in candidates.itertuples():
        diversity_penalty = 0.0
        if row.title_root and row.title_root in selected_roots:
            diversity_penalty += 0.25
        if row.director_set and row.director_set.intersection(selected_directors):
            diversity_penalty += 0.1
        adjusted_score = float(row.second_stage_score) - diversity_penalty
        if len(selected) < limit:
            selected.append((adjusted_score, row.Index))
            selected_roots.add(row.title_root)
            selected_directors.update(row.director_set)

    if not selected:
        return candidates.head(limit)

    indices = [idx for _, idx in sorted(selected, key=lambda item: item[0], reverse=True)]
    return candidates.loc[indices].sort_values(["second_stage_score", "vote_average", "vote_count"], ascending=False)


def _candidate_frame(
    lexical_hits: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    retrieval_profile: dict[str, Any],
    mode: str,
) -> pd.DataFrame:
    combined: dict[int, dict[str, float]] = {}

    if mode in {"hybrid", "lexical"}:
        for hit in lexical_hits:
            tmdb_id = int(hit["tmdb_id"])
            combined.setdefault(tmdb_id, {"fts_score": 0.0, "semantic_score": 0.0})
            combined[tmdb_id]["fts_score"] = max(combined[tmdb_id]["fts_score"], float(hit["lexical_score"]))

    if mode in {"hybrid", "semantic"}:
        for hit in semantic_hits:
            tmdb_id = int(hit["tmdb_id"])
            combined.setdefault(tmdb_id, {"fts_score": 0.0, "semantic_score": 0.0})
            combined[tmdb_id]["semantic_score"] = max(combined[tmdb_id]["semantic_score"], float(hit["semantic_score"]))

    if not combined:
        return MOVIES.iloc[0:0].copy()

    candidates = MOVIES[MOVIES["tmdb_id"].isin(list(combined))].copy()
    candidates = candidates[~candidates["tmdb_id"].isin(retrieval_profile["history_tmdb_ids"])]
    candidates = candidates[~candidates["title"].isin(retrieval_profile["history_titles"])]
    candidates["fts_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["fts_score"])
    candidates["semantic_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["semantic_score"])
    return candidates


def local_fallback_candidates(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    intent: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    retrieval_profile = build_retrieval_profile(preferences, history, intent=intent)
    candidates = MOVIES.copy()
    candidates["fts_score"] = 0.0
    candidates["semantic_score"] = 0.0
    candidates["second_stage_score"] = candidates.apply(local_fallback_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["second_stage_score", "vote_average", "vote_count"], ascending=False)
    return candidates.head(MERGED_POOL_SIZE).copy(), retrieval_profile


def build_candidate_pool(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    mode: str = "hybrid",
    intent: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    retrieval_profile = build_retrieval_profile(preferences, history, intent=intent)
    query_text = build_query_text(preferences, retrieval_profile)
    exclude_ids = set(retrieval_profile["history_tmdb_ids"])

    from fts_retrieval import fts_ready, search_fts
    from semantic_retrieval import search_semantic, semantic_ready

    lexical_hits = search_fts(query_text, exclude_ids=exclude_ids, limit=FTS_LIMIT) if mode in {"hybrid", "lexical"} and fts_ready() else []
    semantic_hits = search_semantic(query_text, exclude_ids=exclude_ids, limit=SEMANTIC_LIMIT) if mode in {"hybrid", "semantic"} and semantic_ready() else []

    candidates = _candidate_frame(lexical_hits, semantic_hits, retrieval_profile, mode)
    if candidates.empty:
        fallback, fallback_profile = local_fallback_candidates(preferences, history, intent=intent)
        return fallback, fallback_profile, "local_fallback"

    candidates["second_stage_score"] = candidates.apply(hybrid_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["second_stage_score", "semantic_score", "fts_score", "vote_average", "vote_count"], ascending=False)
    return candidates.head(MERGED_POOL_SIZE).copy(), retrieval_profile, mode


def build_shortlist(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    mode: str = "hybrid",
    intent: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates, retrieval_profile, resolved_mode = build_candidate_pool(preferences, history, mode=mode, intent=intent)
    prompt_profile = build_prompt_profile(preferences, retrieval_profile)

    reranked = candidates.head(SECOND_STAGE_POOL_SIZE).copy()
    reranked = reranked.sort_values(["second_stage_score", "semantic_score", "fts_score", "vote_average", "vote_count"], ascending=False)
    reranked = diversify_candidates(reranked, SHORTLIST_SIZE)

    retrieval_profile = dict(retrieval_profile)
    retrieval_profile["retrieval_mode"] = resolved_mode

    shortlist = []
    for row in reranked.head(SHORTLIST_SIZE).itertuples():
        shortlist.append(
            {
                "tmdb_id": int(row.tmdb_id),
                "title": row.title,
                "year": int(row.year),
                "genres": row.genres,
                "overview": str(row.overview)[:260],
                "keywords": ", ".join(sorted(row.keywords_set)[:6]),
                "director": row.director,
                "top_cast": ", ".join(sorted(row.cast_set)[:4]),
                "score": round(float(row.second_stage_score), 3),
                "semantic_score": round(float(getattr(row, "semantic_score", 0.0)), 3),
                "fts_score": round(float(getattr(row, "fts_score", 0.0)), 3),
            }
        )
    return shortlist, prompt_profile, retrieval_profile
