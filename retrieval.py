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


SHORTLIST_SIZE = 30
FTS_LIMIT = 40
SEMANTIC_LIMIT = 40
MERGED_POOL_SIZE = 60
SECOND_STAGE_POOL_SIZE = 36
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
NEGATION_RE = re.compile(r"\b(?:no|not|without|avoid)\s+([a-z0-9][a-z0-9\-\s]{0,32}?)(?=,|\.|;|\bbut\b|\bexcept\b|\binstead\b|$)", re.IGNORECASE)
SIMILARITY_RE = re.compile(r"\b(?:like|similar to|something like|in the vein of|along the lines of)\b", re.IGNORECASE)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "but",
    "could",
    "for",
    "find",
    "from",
    "great",
    "have",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "like",
    "love",
    "me",
    "movie",
    "movies",
    "must",
    "no",
    "of",
    "on",
    "or",
    "sound",
    "something",
    "stories",
    "story",
    "that",
    "that's",
    "thats",
    "the",
    "to",
    "want",
    "watch",
    "would",
    "with",
}
GENRE_ALIASES = {
    "action": "action",
    "adventure": "adventure",
    "animation": "animation",
    "animated": "animation",
    "comedy": "comedy",
    "crime": "crime",
    "drama": "drama",
    "family": "family",
    "fantasy": "fantasy",
    "history": "history",
    "horror": "horror",
    "mystery": "mystery",
    "romance": "romance",
    "sci fi": "science fiction",
    "sci-fi": "science fiction",
    "science fiction": "science fiction",
    "thriller": "thriller",
    "war": "war",
    "western": "western",
}
SUBJECTIVE_QUERY_TOKENS = {
    "awesome",
    "bad",
    "best",
    "cool",
    "enjoyable",
    "entertaining",
    "fun",
    "funny",
    "good",
    "great",
    "interesting",
    "nice",
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


def parse_json_int_list(value: Any) -> set[int]:
    raw = str(value or "").strip()
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    if not isinstance(parsed, list):
        return set()
    result: set[int] = set()
    for item in parsed:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    return result


def extract_negation_context(preferences: str) -> dict[str, Any]:
    normalized = normalize_text(preferences)
    negative_phrases: list[str] = []
    negative_tokens: set[str] = set()
    hard_block_genres: set[str] = set()

    for match in NEGATION_RE.finditer(normalized):
        phrase = " ".join(match.group(1).split()).strip()
        if not phrase:
            continue
        negative_phrases.append(phrase)
        tokens = tokenize(phrase)
        negative_tokens.update(tokens)
        phrase_variants = {phrase, phrase.replace("-", " ")}
        for variant in phrase_variants:
            if variant in GENRE_ALIASES:
                hard_block_genres.add(GENRE_ALIASES[variant])

    raw_positive_tokens = tokenize(preferences) - negative_tokens
    subjective_tokens = raw_positive_tokens.intersection(SUBJECTIVE_QUERY_TOKENS)
    positive_query_tokens = raw_positive_tokens - subjective_tokens
    if not positive_query_tokens:
        positive_query_tokens = set(raw_positive_tokens)
    lexical_query_text = " ".join(sorted(positive_query_tokens))
    return {
        "negative_phrases": negative_phrases[:6],
        "negative_tokens": negative_tokens,
        "hard_block_genres": hard_block_genres,
        "lexical_query_text": lexical_query_text,
        "semantic_query_text": lexical_query_text or str(preferences or "").strip(),
        "positive_query_tokens": positive_query_tokens,
        "subjective_tokens": subjective_tokens,
        "raw_positive_tokens": raw_positive_tokens,
    }


def extract_explicit_genre_targets(preferences: str) -> set[str]:
    normalized = normalize_text(preferences)
    targets: set[str] = set()
    for alias, canonical in GENRE_ALIASES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            targets.add(canonical)
    return targets


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
    movies["normalized_title"] = movies["title"].map(normalize_text)
    movies["title_root"] = movies["title"].map(title_root)
    movies["genres_set"] = movies["genres"].map(split_csvish)
    movies["keywords_set"] = movies["keywords"].map(split_csvish)
    movies["cast_set"] = movies["top_cast"].map(split_csvish)
    movies["director_set"] = movies["director"].map(split_csvish)
    if "similar_tmdb_ids" in movies.columns:
        movies["similar_ids_set"] = movies["similar_tmdb_ids"].map(parse_json_int_list)
    else:
        movies["similar_ids_set"] = [set() for _ in range(len(movies))]
    movies["search_blob"] = movies[TEXT_COLUMNS].agg(" ".join, axis=1).map(normalize_text)
    movies["search_blob_tokens"] = movies["search_blob"].map(tokenize)
    return movies


TOP_MOVIES = pd.read_csv(ACTIVE_DATA_PATH).fillna("")
MOVIES = prepare_movies(TOP_MOVIES)
MOVIES_BY_EXACT_TITLE = MOVIES.groupby("title", sort=False)
MOVIES_BY_TMDB_ID = MOVIES.set_index("tmdb_id", drop=False)


def preference_seed_rows(preferences: str, history_df: pd.DataFrame) -> pd.DataFrame:
    normalized_preferences = normalize_text(preferences)
    if not normalized_preferences:
        return MOVIES.iloc[0:0].copy()

    matched_tmdb_ids: set[int] = set()
    for row in MOVIES.itertuples():
        normalized_title = row.normalized_title
        if len(normalized_title) < 5:
            continue
        title_pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalized_title)}(?![a-z0-9])")
        if title_pattern.search(normalized_preferences):
            matched_tmdb_ids.add(int(row.tmdb_id))

    if not history_df.empty:
        for row in history_df.itertuples():
            normalized_title = row.normalized_title
            if normalized_title and re.search(rf"(?<![a-z0-9]){re.escape(normalized_title)}(?![a-z0-9])", normalized_preferences):
                matched_tmdb_ids.add(int(row.tmdb_id))

    if not matched_tmdb_ids:
        return MOVIES.iloc[0:0].copy()
    return MOVIES[MOVIES["tmdb_id"].isin(sorted(matched_tmdb_ids))].drop_duplicates(subset=["tmdb_id"]).copy()


def derive_seed_signals(seed_df: pd.DataFrame) -> dict[str, Any]:
    seed_genres: set[str] = set()
    seed_keywords: set[str] = set()
    seed_cast: set[str] = set()
    seed_directors: set[str] = set()
    seed_title_tokens: set[str] = set()
    seed_story_tokens: set[str] = set()
    seed_similar_ids: set[int] = set()
    seed_tmdb_ids: set[int] = set()
    seed_titles: list[str] = []

    for row in seed_df.itertuples():
        seed_genres.update(row.genres_set)
        seed_keywords.update(set(sorted(row.keywords_set)[:12]))
        seed_cast.update(set(sorted(row.cast_set)[:6]))
        seed_directors.update(set(sorted(row.director_set)[:4]))
        seed_title_tokens.update(tokenize(row.title))
        seed_story_tokens.update(tokenize(row.overview))
        seed_similar_ids.update(row.similar_ids_set)
        seed_tmdb_ids.add(int(row.tmdb_id))
        seed_titles.append(str(row.title))

    return {
        "seed_titles": seed_titles,
        "seed_tmdb_ids": seed_tmdb_ids,
        "seed_genres": seed_genres,
        "seed_keywords": seed_keywords,
        "seed_cast": seed_cast,
        "seed_directors": seed_directors,
        "seed_title_tokens": seed_title_tokens,
        "seed_story_tokens": seed_story_tokens,
        "seed_similar_ids": seed_similar_ids,
    }


def derive_seed_query_tokens(seed_signals: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for genre in seed_signals["seed_genres"]:
        tokens.update(tokenize(genre))
    for keyword in seed_signals["seed_keywords"]:
        tokens.update(tokenize(keyword))
    tokens.difference_update(seed_signals["seed_title_tokens"])
    return {token for token in tokens if len(token) > 2}


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


def build_retrieval_profile(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    history_df = history_rows(history)
    raw_preference_tokens = tokenize(preferences)
    history_signals = derive_history_signals(history_df)
    negation_context = extract_negation_context(preferences)
    explicit_genre_targets = extract_explicit_genre_targets(preferences)
    seed_df = preference_seed_rows(preferences, history_df)
    seed_signals = derive_seed_signals(seed_df)
    similarity_request = bool(seed_signals["seed_tmdb_ids"]) and bool(SIMILARITY_RE.search(str(preferences or "")))

    positive_query_tokens = set(negation_context["positive_query_tokens"])
    positive_query_tokens.difference_update(seed_signals["seed_title_tokens"])

    seed_query_tokens = derive_seed_query_tokens(seed_signals)
    if similarity_request:
        positive_query_tokens.update(seed_query_tokens)

    lexical_query_text = " ".join(sorted(positive_query_tokens))
    semantic_query_text = lexical_query_text or str(preferences or "").strip()

    return {
        "history_df": history_df,
        "history_count": len(history),
        "history_exclusion_text": history_prompt_text(len(history)),
        "history_titles": {name for _, name in history if name},
        "history_tmdb_ids": {tmdb_id for tmdb_id, _ in history if tmdb_id is not None},
        "raw_preference_tokens": raw_preference_tokens,
        "explicit_genre_targets": explicit_genre_targets,
        **negation_context,
        "positive_query_tokens": positive_query_tokens,
        "lexical_query_text": lexical_query_text,
        "semantic_query_text": semantic_query_text,
        "similarity_request": similarity_request,
        "exclude_seed_tmdb_ids": seed_signals["seed_tmdb_ids"] if similarity_request else set(),
        **seed_signals,
        "seed_query_tokens": seed_query_tokens,
        **history_signals,
    }


def build_prompt_profile(preferences: str, retrieval_profile: dict[str, Any]) -> dict[str, Any]:
    preferred_themes = sorted(retrieval_profile["positive_query_tokens"])[:6]
    if not preferred_themes:
        fallback_tokens = sorted(token for token in tokenize(preferences) if len(token) > 2 and token not in {"science", "fiction"})[:6]
        preferred_themes = fallback_tokens
    return {
        "target_genres": sorted(retrieval_profile["explicit_genre_targets"])[:4],
        "preferred_tones": sorted(retrieval_profile["subjective_tokens"])[:4],
        "preferred_themes": preferred_themes,
        "avoid": retrieval_profile["negative_phrases"][:6],
        "history_signals": {
            "liked_genres": sorted(retrieval_profile["liked_genres"])[:6],
            "liked_directors": sorted(retrieval_profile["liked_directors"])[:4],
            "liked_cast": sorted(retrieval_profile["liked_cast"])[:6],
        },
    }


def build_query_text(preferences: str, retrieval_profile: dict[str, Any]) -> str:
    return retrieval_profile["lexical_query_text"] or str(preferences or "").strip()


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


def seed_similarity_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if not retrieval_profile.get("similarity_request"):
        return 0.0
    score = 0.0
    score += 1.4 * len(retrieval_profile["seed_genres"].intersection(row["genres_set"]))
    score += 0.55 * len(retrieval_profile["seed_keywords"].intersection(row["keywords_set"]))
    score += 0.18 * len(retrieval_profile["seed_story_tokens"].intersection(row["search_blob_tokens"]))
    score += 0.8 * len(retrieval_profile["seed_directors"].intersection(row["director_set"]))
    score += 0.45 * len(retrieval_profile["seed_cast"].intersection(row["cast_set"]))
    if int(row["tmdb_id"]) in retrieval_profile["seed_similar_ids"]:
        score += 2.5
    return score


def preference_alignment_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    positive_tokens = retrieval_profile["positive_query_tokens"] or retrieval_profile["raw_preference_tokens"]
    negative_tokens = retrieval_profile["negative_tokens"]
    tokens = row["search_blob_tokens"]

    positive_overlap = len(positive_tokens.intersection(tokens))
    negative_overlap = len(negative_tokens.intersection(tokens))

    score = 0.7 * positive_overlap
    score += 0.35 * len(positive_tokens.intersection(row["keywords_set"]))
    if retrieval_profile["explicit_genre_targets"]:
        genre_overlap = len(retrieval_profile["explicit_genre_targets"].intersection(row["genres_set"]))
        score += 2.2 * genre_overlap
        if genre_overlap == 0:
            score -= 2.8
    score -= 2.4 * negative_overlap
    if retrieval_profile["negative_phrases"]:
        score -= 1.2 * sum(
            1 for phrase in retrieval_profile["negative_phrases"] if normalize_text(phrase) in row["search_blob"]
        )
    if negative_overlap and positive_overlap <= 1:
        score -= 1.5
    if retrieval_profile["hard_block_genres"].intersection(row["genres_set"]):
        score -= 3.0
    return score


def novelty_penalty(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    penalty = 4.0 if row["title_root"] and row["title_root"] in retrieval_profile["watched_roots"] else 0.0
    if retrieval_profile["negative_tokens"] and retrieval_profile["negative_tokens"].intersection(row["search_blob_tokens"]):
        penalty += 4.0
    if retrieval_profile["negative_phrases"]:
        penalty += 1.0 * sum(
            1 for phrase in retrieval_profile["negative_phrases"] if normalize_text(phrase) in row["search_blob"]
        )
    if retrieval_profile["hard_block_genres"].intersection(row["genres_set"]):
        penalty += 4.0
    return penalty


def _normalize_component(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def hybrid_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    history_component = _normalize_component(history_affinity_score(row, retrieval_profile), 6.0)
    appeal_component = _normalize_component(appeal_score(row), 8.0)
    novelty_component = _normalize_component(novelty_penalty(row, retrieval_profile), 7.0)
    preference_component = _normalize_component(max(0.0, preference_alignment_score(row, retrieval_profile)), 8.0)
    preference_penalty = _normalize_component(max(0.0, -preference_alignment_score(row, retrieval_profile)), 5.0)
    seed_component = _normalize_component(seed_similarity_score(row, retrieval_profile), 8.0)
    return (
        0.42 * float(row.get("semantic_score", 0.0))
        + 0.23 * float(row.get("fts_score", 0.0))
        + 0.18 * seed_component
        + 0.15 * preference_component
        + 0.12 * history_component
        + 0.10 * appeal_component
        - 0.05 * novelty_component
        - 0.08 * preference_penalty
    )


def local_fallback_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if int(row["tmdb_id"]) in retrieval_profile["history_tmdb_ids"] or row["title"] in retrieval_profile["history_titles"]:
        return -10_000.0
    if int(row["tmdb_id"]) in retrieval_profile.get("exclude_seed_tmdb_ids", set()):
        return -10_000.0

    query_terms = retrieval_profile["positive_query_tokens"] or retrieval_profile["raw_preference_tokens"]
    overlap = len(query_terms.intersection(row["search_blob_tokens"]))
    negative_hits = len(retrieval_profile["negative_tokens"].intersection(row["search_blob_tokens"]))
    alignment_score = preference_alignment_score(row, retrieval_profile)

    return (
        float(overlap)
        + 0.9 * alignment_score
        + 0.9 * seed_similarity_score(row, retrieval_profile)
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
    if retrieval_profile.get("exclude_seed_tmdb_ids"):
        candidates = candidates[~candidates["tmdb_id"].isin(retrieval_profile["exclude_seed_tmdb_ids"])]
    if retrieval_profile["hard_block_genres"]:
        candidates = candidates[~candidates["genres_set"].map(lambda genres: bool(genres.intersection(retrieval_profile["hard_block_genres"])))]
    candidates["fts_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["fts_score"])
    candidates["semantic_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["semantic_score"])
    return candidates


def local_fallback_candidates(preferences: str, history: tuple[tuple[int | None, str], ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    retrieval_profile = build_retrieval_profile(preferences, history)
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
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    retrieval_profile = build_retrieval_profile(preferences, history)
    lexical_query_text = build_query_text(preferences, retrieval_profile)
    semantic_query_text = retrieval_profile["semantic_query_text"]
    exclude_ids = set(retrieval_profile["history_tmdb_ids"])

    from fts_retrieval import fts_ready, search_fts
    from semantic_retrieval import search_semantic, semantic_ready

    lexical_hits = search_fts(lexical_query_text, exclude_ids=exclude_ids, limit=FTS_LIMIT) if mode in {"hybrid", "lexical"} and fts_ready() else []
    try:
        semantic_hits = search_semantic(semantic_query_text, exclude_ids=exclude_ids, limit=SEMANTIC_LIMIT) if mode in {"hybrid", "semantic"} and semantic_ready() else []
    except Exception:
        semantic_hits = []

    candidates = _candidate_frame(lexical_hits, semantic_hits, retrieval_profile, mode)
    if candidates.empty:
        fallback, fallback_profile = local_fallback_candidates(preferences, history)
        return fallback, fallback_profile, "local_fallback"

    candidates["second_stage_score"] = candidates.apply(hybrid_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["second_stage_score", "semantic_score", "fts_score", "vote_average", "vote_count"], ascending=False)
    return candidates.head(MERGED_POOL_SIZE).copy(), retrieval_profile, mode


def build_shortlist(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    mode: str = "hybrid",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates, retrieval_profile, resolved_mode = build_candidate_pool(preferences, history, mode=mode)
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
                "score": round(float(row.second_stage_score), 3),
                "semantic_score": round(float(getattr(row, "semantic_score", 0.0)), 3),
                "fts_score": round(float(getattr(row, "fts_score", 0.0)), 3),
            }
        )
    return shortlist, prompt_profile, retrieval_profile
