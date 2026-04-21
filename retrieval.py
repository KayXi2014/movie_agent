from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHORTLIST_SIZE = 10
FTS_LIMIT = 40
SEMANTIC_LIMIT = 40
MERGED_POOL_SIZE = 60
SECOND_STAGE_POOL_SIZE = 36
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "tmdb_top1000_movies.csv"
ENRICHED_DATA_PATH = DATA_DIR / "tmdb_top1000_movies_enriched.csv"
ACTIVE_DATA_PATH = ENRICHED_DATA_PATH if ENRICHED_DATA_PATH.exists() else DATA_PATH

RETRIEVAL_DB_PATH = DATA_DIR / "movies.sqlite"
RETRIEVAL_DB_META_PATH = DATA_DIR / "movies.sqlite.meta.json"
EMBEDDINGS_PATH = DATA_DIR / "movie_embeddings.npy"
EMBEDDING_IDS_PATH = DATA_DIR / "movie_embedding_ids.json"
EMBEDDING_META_PATH = DATA_DIR / "movie_embedding_meta.json"

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
    "collection_name",
    "spoken_languages",
    "alternative_titles",
    "us_rating",
]
OPTIONAL_TEXT_COLUMNS = [
    "collection_name",
    "spoken_languages",
    "alternative_titles",
    "us_rating",
]
OPTIONAL_DATA_COLUMNS = [
    "similar_tmdb_ids",
    "recommended_tmdb_ids",
    *OPTIONAL_TEXT_COLUMNS,
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
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "ive",
    "i've",
    "like",
    "me",
    "movie",
    "movies",
    "no",
    "of",
    "on",
    "or",
    "something",
    "that",
    "that's",
    "thats",
    "the",
    "to",
    "want",
    "watch",
    "with",
}
GENRE_ALIASES = {
    "action": "action",
    "adventure": "adventure",
    "anime": "animation",
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
TONE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgrounded\b", re.IGNORECASE), "grounded"),
    (re.compile(r"\bserious\b", re.IGNORECASE), "serious"),
    (re.compile(r"\bdark\b", re.IGNORECASE), "dark"),
    (re.compile(r"\bgritty\b", re.IGNORECASE), "gritty"),
    (re.compile(r"\btense\b", re.IGNORECASE), "tense"),
    (re.compile(r"\bintense\b", re.IGNORECASE), "intense"),
    (re.compile(r"\bwarm\b", re.IGNORECASE), "warm"),
    (re.compile(r"\bintimate\b", re.IGNORECASE), "intimate"),
    (re.compile(r"\bthoughtful\b", re.IGNORECASE), "thoughtful"),
    (re.compile(r"\bcerebral\b", re.IGNORECASE), "cerebral"),
    (re.compile(r"\bhaunting\b", re.IGNORECASE), "haunting"),
    (re.compile(r"\bbleak\b", re.IGNORECASE), "bleak"),
    (re.compile(r"\bmoody\b", re.IGNORECASE), "moody"),
    (re.compile(r"\blighthearted\b|\blight-hearted\b", re.IGNORECASE), "lighthearted"),
    (re.compile(r"\buplifting\b", re.IGNORECASE), "uplifting"),
    (re.compile(r"\bromantic\b", re.IGNORECASE), "romantic"),
    (re.compile(r"\bfunny\b", re.IGNORECASE), "funny"),
    (re.compile(r"\bcampy\b", re.IGNORECASE), "campy"),
)
YEAR_SIGNAL_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:19|20)\d0s\b|\b(?:80s|90s|2000s|2010s|2020s)\b", re.IGNORECASE)
YEAR_SIGNAL_PHRASES = (
    "recent",
    "newer",
    "latest",
    "modern",
    "older",
    "classic",
    "relatively new",
    "new movie",
    "older movie",
    "old movie",
    "this decade",
    "last decade",
)
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
        source_relpath = str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
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


def ensure_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    movies = df.copy()
    for column in OPTIONAL_DATA_COLUMNS:
        if column not in movies.columns:
            movies[column] = ""
    return movies


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("sci-fi", "science fiction")
    text = text.replace("sci fi", "science fiction")
    return text


def tokenize(value: Any) -> set[str]:
    return {token for token in TOKEN_RE.findall(normalize_text(value)) if token not in STOP_WORDS}


def normalize_match_token(token: str) -> str:
    text = normalize_text(token)
    if len(text) <= 3:
        return text
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith(("ches", "shes", "xes", "zes", "sses", "oes")) and len(text) > 4:
        return text[:-2]
    if text.endswith("s") and not text.endswith(("ss", "us", "is")) and len(text) > 3:
        return text[:-1]
    return text


def match_tokens(value: Any) -> set[str]:
    result: set[str] = set()
    for token in TOKEN_RE.findall(normalize_text(value)):
        if token in STOP_WORDS:
            continue
        normalized = normalize_match_token(token)
        if normalized and normalized not in STOP_WORDS:
            result.add(normalized)
    return result


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


def parse_json_string_list(value: Any) -> set[str]:
    raw = str(value or "").strip()
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return split_csvish(raw)
    if not isinstance(parsed, list):
        return set()
    result: set[str] = set()
    for item in parsed:
        text = str(item or "").strip()
        if text:
            result.add(text)
    return result


def _build_title_variants(row: pd.Series) -> set[str]:
    variants = {
        normalize_text(row.get("title")),
        normalize_text(row.get("original_title")),
    }
    variants.update(normalize_text(item) for item in row.get("alternative_titles_set", set()))
    return {variant for variant in variants if variant}


def _normalized_title_mentioned(text: str, title_variants: set[str]) -> bool:
    for variant in title_variants:
        if len(variant) < 5:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", text):
            return True
    return False


def extract_negation_context(preferences: str) -> dict[str, Any]:
    normalized = normalize_text(preferences)
    negative_phrases: list[str] = []
    negative_tokens: set[str] = set()
    negative_match_tokens: set[str] = set()
    hard_block_genres: set[str] = set()

    for match in NEGATION_RE.finditer(normalized):
        phrase = " ".join(match.group(1).split()).strip()
        if not phrase:
            continue
        negative_phrases.append(phrase)
        tokens = tokenize(phrase)
        negative_tokens.update(tokens)
        negative_match_tokens.update(match_tokens(phrase))
        phrase_variants = {phrase, phrase.replace("-", " ")}
        for variant in phrase_variants:
            if variant in GENRE_ALIASES:
                hard_block_genres.add(GENRE_ALIASES[variant])
        for alias, canonical in GENRE_ALIASES.items():
            for variant in phrase_variants:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", variant):
                    hard_block_genres.add(canonical)

    raw_positive_tokens = tokenize(preferences) - negative_tokens
    positive_query_tokens = set(raw_positive_tokens)
    lexical_query_text = " ".join(sorted(positive_query_tokens))
    return {
        "negative_phrases": negative_phrases[:6],
        "negative_tokens": negative_tokens,
        "negative_match_tokens": negative_match_tokens,
        "hard_block_genres": hard_block_genres,
        "lexical_query_text": lexical_query_text,
        "semantic_query_text": lexical_query_text or str(preferences or "").strip(),
        "positive_query_tokens": positive_query_tokens,
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


def prepare_movies(df: pd.DataFrame) -> pd.DataFrame:
    movies = ensure_dataset_columns(df)
    movies["normalized_title"] = movies["title"].map(normalize_text)
    movies["title_root"] = movies["title"].map(title_root)
    movies["genres_set"] = movies["genres"].map(split_csvish)
    movies["keywords_set"] = movies["keywords"].map(split_csvish)
    movies["cast_set"] = movies["top_cast"].map(split_csvish)
    movies["director_set"] = movies["director"].map(split_csvish)
    movies["similar_ids_set"] = movies["similar_tmdb_ids"].map(parse_json_int_list)
    movies["recommended_ids_set"] = movies["recommended_tmdb_ids"].map(parse_json_int_list)
    movies["alternative_titles_set"] = movies["alternative_titles"].map(parse_json_string_list)
    movies["title_variants"] = movies.apply(_build_title_variants, axis=1)
    movies["search_blob"] = movies[TEXT_COLUMNS].agg(" ".join, axis=1).map(normalize_text)
    movies["search_blob_tokens"] = movies["search_blob"].map(tokenize)
    movies["search_blob_match_tokens"] = movies["search_blob"].map(match_tokens)
    return movies


TOP_MOVIES = pd.read_csv(ACTIVE_DATA_PATH).fillna("")
MOVIES = prepare_movies(TOP_MOVIES)
MOVIES_BY_EXACT_TITLE = MOVIES.groupby("title", sort=False)
MOVIES_BY_TMDB_ID = MOVIES.set_index("tmdb_id", drop=False)
MOVIES_BY_TITLE_VARIANT: dict[str, tuple[int, ...]] = {}
KNOWN_PERSON_NAMES = tuple(
    sorted(
        {
            name
            for row in MOVIES.itertuples()
            for name in (*row.director_set, *row.cast_set)
            if len(tokenize(name)) >= 2
        },
        key=len,
        reverse=True,
    )
)
for row in MOVIES.itertuples():
    tmdb_id = int(row.tmdb_id)
    for variant in row.title_variants:
        existing = MOVIES_BY_TITLE_VARIANT.get(variant, ())
        if tmdb_id not in existing:
            MOVIES_BY_TITLE_VARIANT[variant] = (*existing, tmdb_id)


def preference_seed_rows(preferences: str, history_df: pd.DataFrame) -> pd.DataFrame:
    normalized_preferences = normalize_text(preferences)
    if not normalized_preferences:
        return MOVIES.iloc[0:0].copy()

    matched_tmdb_ids: set[int] = set()
    for row in MOVIES.itertuples():
        if _normalized_title_mentioned(normalized_preferences, row.title_variants):
            matched_tmdb_ids.add(int(row.tmdb_id))

    if not history_df.empty:
        for row in history_df.itertuples():
            if _normalized_title_mentioned(normalized_preferences, row.title_variants):
                matched_tmdb_ids.add(int(row.tmdb_id))

    if not matched_tmdb_ids:
        return MOVIES.iloc[0:0].copy()
    return MOVIES[MOVIES["tmdb_id"].isin(sorted(matched_tmdb_ids))].drop_duplicates(subset=["tmdb_id"]).copy()


def derive_seed_signals(seed_df: pd.DataFrame) -> dict[str, Any]:
    seed_genres: set[str] = set()
    seed_keywords: set[str] = set()
    seed_title_tokens: set[str] = set()
    seed_similar_ids: set[int] = set()
    seed_recommended_ids: set[int] = set()
    seed_tmdb_ids: set[int] = set()
    seed_titles: list[str] = []

    for row in seed_df.itertuples():
        seed_genres.update(row.genres_set)
        seed_keywords.update(set(sorted(row.keywords_set)[:10]))
        seed_title_tokens.update(tokenize(row.title))
        seed_similar_ids.update(row.similar_ids_set)
        seed_recommended_ids.update(row.recommended_ids_set)
        seed_tmdb_ids.add(int(row.tmdb_id))
        seed_titles.append(str(row.title))

    return {
        "seed_titles": seed_titles,
        "seed_tmdb_ids": seed_tmdb_ids,
        "seed_genres": seed_genres,
        "seed_keywords": seed_keywords,
        "seed_title_tokens": seed_title_tokens,
        "seed_similar_ids": seed_similar_ids,
        "seed_recommended_ids": seed_recommended_ids,
    }


def derive_seed_query_tokens(seed_signals: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for genre in seed_signals["seed_genres"]:
        tokens.update(tokenize(genre))
    for keyword in seed_signals["seed_keywords"]:
        tokens.update(tokenize(keyword))
    tokens.difference_update(seed_signals["seed_title_tokens"])
    return {token for token in tokens if len(token) > 2}


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
        matched_by_name = False
        if name in MOVIES_BY_EXACT_TITLE.groups:
            group = MOVIES_BY_EXACT_TITLE.get_group(name)
            unseen = group[~group["tmdb_id"].isin(seen_ids)]
            if not unseen.empty:
                rows.append(unseen)
                seen_ids.update(unseen["tmdb_id"].astype(int).tolist())
                matched_by_name = True
        if matched_by_name:
            continue

        normalized_name = normalize_text(name)
        if normalized_name in MOVIES_BY_TITLE_VARIANT:
            matched_ids = [candidate_id for candidate_id in MOVIES_BY_TITLE_VARIANT[normalized_name] if candidate_id not in seen_ids]
            if matched_ids:
                unseen = MOVIES[MOVIES["tmdb_id"].isin(matched_ids)]
                rows.append(unseen)
                seen_ids.update(unseen["tmdb_id"].astype(int).tolist())
                matched_by_name = True
        if matched_by_name:
            continue

        if tmdb_id is not None and tmdb_id in MOVIES_BY_TMDB_ID.index:
            row = MOVIES_BY_TMDB_ID.loc[tmdb_id]
            if int(row.tmdb_id) not in seen_ids:
                rows.append(row.to_frame().T)
                seen_ids.add(int(row.tmdb_id))
    if not rows:
        return MOVIES.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tmdb_id"])


def derive_history_context(history_df: pd.DataFrame) -> dict[str, set[str]]:
    genre_counts: Counter[str] = Counter()
    watched_roots: set[str] = set()
    for row in history_df.itertuples():
        genre_counts.update(item for item in row.genres_set if item)
        if row.title_root:
            watched_roots.add(row.title_root)
    return {
        "watched_genres": {name for name, _ in genre_counts.most_common(3)},
        "watched_roots": watched_roots,
    }


def history_prompt_text(history_count: int) -> str:
    return "none" if history_count <= 0 else f"known_watch_history_count={history_count}; use history primarily to avoid re-recommending already watched titles"


def _phrase_mentioned(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def extract_named_person_signals(preferences: str) -> list[str]:
    normalized = normalize_text(preferences)
    matches: list[str] = []
    for name in KNOWN_PERSON_NAMES:
        if _phrase_mentioned(normalized, name):
            matches.append(name)
    return matches[:6]


def build_retrieval_profile(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
) -> dict[str, Any]:
    history_df = history_rows(history)
    normalized_preferences = normalize_text(preferences)
    raw_preference_tokens = tokenize(preferences)
    history_context = derive_history_context(history_df)
    negation_context = extract_negation_context(preferences)
    explicit_genre_targets = extract_explicit_genre_targets(preferences).difference(negation_context["hard_block_genres"])
    named_person_signals = extract_named_person_signals(preferences)
    seed_df = preference_seed_rows(preferences, history_df)
    seed_signals = derive_seed_signals(seed_df)
    similarity_request = bool(seed_signals["seed_tmdb_ids"]) and bool(SIMILARITY_RE.search(str(preferences or "")))

    positive_query_tokens = set(negation_context["positive_query_tokens"])
    positive_query_tokens.difference_update(seed_signals["seed_title_tokens"])

    for genre in explicit_genre_targets:
        positive_query_tokens.update(tokenize(genre))

    negative_tokens = set(negation_context["negative_tokens"])
    seed_query_tokens = derive_seed_query_tokens(seed_signals)

    lexical_query_text = " ".join(sorted(positive_query_tokens))
    semantic_query_text = str(preferences or "").strip()
    if not semantic_query_text:
        semantic_query_text = lexical_query_text

    return {
        "history_df": history_df,
        "history_count": len(history),
        "history_exclusion_text": history_prompt_text(len(history)),
        "history_titles": {name for _, name in history if name},
        "history_tmdb_ids": {tmdb_id for tmdb_id, _ in history if tmdb_id is not None},
        "normalized_preferences": normalized_preferences,
        "raw_preference_tokens": raw_preference_tokens,
        "explicit_genre_targets": explicit_genre_targets,
        "named_person_signals": named_person_signals,
        "has_named_person_signal": bool(named_person_signals),
        **negation_context,
        "negative_tokens": negative_tokens,
        "positive_query_tokens": positive_query_tokens,
        "lexical_query_text": lexical_query_text,
        "semantic_query_text": semantic_query_text,
        "similarity_request": similarity_request,
        "exclude_seed_tmdb_ids": seed_signals["seed_tmdb_ids"] if similarity_request else set(),
        **seed_signals,
        "seed_query_tokens": seed_query_tokens,
        **history_context,
    }


def merge_intent_override(retrieval_profile: dict[str, Any], intent_override: dict[str, Any] | None) -> dict[str, Any]:
    if not intent_override:
        return retrieval_profile

    merged = dict(retrieval_profile)

    override_genres = {
        GENRE_ALIASES.get(normalize_text(item), normalize_text(item))
        for item in intent_override.get("genres", [])
        if normalize_text(item)
    }
    override_genres = {genre for genre in override_genres if genre}
    if override_genres:
        merged["explicit_genre_targets"] = set(merged["explicit_genre_targets"]).union(override_genres).difference(merged["hard_block_genres"])
        for genre in merged["explicit_genre_targets"]:
            merged["positive_query_tokens"].update(tokenize(genre))

    override_avoid = [" ".join(str(item or "").split()).strip() for item in intent_override.get("avoid", []) if str(item or "").strip()]
    if override_avoid:
        negative_phrases = list(merged["negative_phrases"])
        for phrase in override_avoid:
            normalized_phrase = normalize_text(phrase)
            if normalized_phrase and normalized_phrase not in negative_phrases:
                negative_phrases.append(normalized_phrase)
                merged["negative_tokens"].update(tokenize(normalized_phrase))
                merged["negative_match_tokens"].update(match_tokens(normalized_phrase))
        merged["negative_phrases"] = negative_phrases[:6]

    override_people = [normalize_text(item) for item in intent_override.get("named_people", []) if normalize_text(item)]
    if override_people:
        named_person_signals = list(merged.get("named_person_signals", []))
        for person in override_people:
            if person not in named_person_signals:
                named_person_signals.append(person)
        merged["named_person_signals"] = named_person_signals[:6]
        merged["has_named_person_signal"] = bool(merged["named_person_signals"])

    merged["lexical_query_text"] = " ".join(sorted(merged["positive_query_tokens"])) or merged["lexical_query_text"]
    return merged


def build_prompt_profile(preferences: str, retrieval_profile: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_text(preferences)
    tone: list[str] = []
    for pattern, canonical in TONE_PATTERNS:
        if pattern.search(normalized) and canonical not in tone:
            tone.append(canonical)
    year_relevant = bool(YEAR_SIGNAL_RE.search(normalized)) or any(phrase in normalized for phrase in YEAR_SIGNAL_PHRASES)
    country_relevant_tokens = {
        "foreign",
        "international",
        "country",
        "countries",
        "language",
        "languages",
        "korean",
        "japanese",
        "french",
        "spanish",
        "italian",
        "german",
        "british",
        "english",
    }
    return {
        "target_genres": sorted(retrieval_profile["explicit_genre_targets"])[:4],
        "tone": tone[:3],
        "avoid": retrieval_profile["negative_phrases"][:6],
        "year_relevant": year_relevant,
        "country_relevant": bool(set(tokenize(preferences)) & country_relevant_tokens),
    }


def _component_scores(row: pd.Series, retrieval_profile: dict[str, Any]) -> dict[str, float]:
    return {
        "keyword_alignment_score": round(keyword_alignment_score(row, retrieval_profile), 3),
        "genre_alignment_score": round(genre_alignment_score(row, retrieval_profile), 3),
        "avoid_penalty_score": round(avoid_penalty_score(row, retrieval_profile), 3),
        "person_anchor_score": round(person_anchor_score(row, retrieval_profile), 3),
        "seed_similarity_score": round(_normalize_component(seed_similarity_score(row, retrieval_profile), 8.0), 3),
    }


def build_confidence_bundle(
    shortlist: list[dict[str, Any]],
    retrieval_profile: dict[str, Any],
    prompt_profile: dict[str, Any],
) -> dict[str, Any]:
    if not shortlist:
        return {
            "confidence": "low",
            "route": "judge_10",
            "specificity": "low",
            "convergence": "weak",
            "top_fulfillment": "low",
            "intent_used": False,
            "gap1": 0.0,
            "gap5": 0.0,
            "contradictions": ["empty_shortlist"],
        }

    top = shortlist[0]
    second = shortlist[1] if len(shortlist) > 1 else None
    fifth = shortlist[4] if len(shortlist) > 4 else None
    gap1 = float(top["score"]) - float(second["score"]) if second else float(top["score"])
    gap5 = float(top["score"]) - float(fifth["score"]) if fifth else gap1

    specificity_points = 0
    specificity_points += 1 if retrieval_profile["explicit_genre_targets"] else 0
    specificity_points += 1 if retrieval_profile["negative_phrases"] else 0
    specificity_points += 1 if retrieval_profile.get("named_person_signals") else 0
    specificity_points += 1 if prompt_profile.get("year_relevant") else 0
    specificity_points += 1 if retrieval_profile.get("similarity_request") else 0
    if specificity_points >= 3:
        specificity = "high"
    elif specificity_points >= 2:
        specificity = "medium"
    else:
        specificity = "low"

    semantic_available = bool(retrieval_profile.get("semantic_available"))
    lexical_available = bool(retrieval_profile.get("lexical_available"))
    fts_present = float(top.get("fts_score", 0.0)) > 0.0
    semantic_present = float(top.get("semantic_score", 0.0)) > 0.0
    if semantic_available and fts_present and semantic_present:
        convergence = "strong"
    elif fts_present or semantic_present:
        convergence = "partial"
    else:
        convergence = "weak"

    contradictions: list[str] = []
    if retrieval_profile["explicit_genre_targets"] and float(top.get("genre_alignment_score", 0.0)) <= 0.0:
        contradictions.append("genre_miss")
    if retrieval_profile["negative_phrases"] and float(top.get("avoid_penalty_score", 0.0)) > 0.1:
        contradictions.append("avoid_hit")
    if retrieval_profile.get("named_person_signals") and float(top.get("person_anchor_score", 0.0)) <= 0.0:
        contradictions.append("person_miss")
    if retrieval_profile.get("similarity_request") and float(top.get("seed_similarity_score", 0.0)) < 0.2:
        contradictions.append("seed_miss")

    keyword_score = float(top.get("keyword_alignment_score", 0.0))
    genre_score = float(top.get("genre_alignment_score", 0.0))
    avoid_score = float(top.get("avoid_penalty_score", 0.0))
    person_score = float(top.get("person_anchor_score", 0.0))
    seed_score = float(top.get("seed_similarity_score", 0.0))

    high_match = keyword_score >= 0.3 and avoid_score <= 0.1
    medium_match = keyword_score >= 0.16 and avoid_score <= 0.25
    if retrieval_profile["explicit_genre_targets"]:
        high_match = high_match and genre_score > 0.0
        medium_match = medium_match and genre_score > -0.01
    if retrieval_profile.get("named_person_signals"):
        high_match = high_match and person_score > 0.0
        medium_match = medium_match and person_score > 0.0
    if retrieval_profile.get("similarity_request"):
        high_match = high_match and seed_score >= 0.2
        medium_match = medium_match and seed_score >= 0.1

    if contradictions:
        top_fulfillment = "low"
    elif high_match:
        top_fulfillment = "high"
    elif medium_match:
        top_fulfillment = "medium"
    else:
        top_fulfillment = "low"

    lexical_only_equivalent = (
        lexical_available
        and not semantic_available
        and fts_present
        and keyword_score >= 0.3
        and gap1 >= 0.08
        and not contradictions
    )

    strong_convergence = convergence == "strong" or lexical_only_equivalent
    very_strong_gap = gap1 >= 0.18 or gap5 >= 0.25
    named_person_high_confidence = (
        bool(retrieval_profile.get("named_person_signals"))
        and person_score > 0.0
        and strong_convergence
        and very_strong_gap
        and top_fulfillment in {"high", "medium"}
        and not contradictions
    )

    if (
        not contradictions
        and strong_convergence
        and (
            (specificity == "high" and top_fulfillment == "high" and gap1 >= 0.03)
            or (specificity == "medium" and top_fulfillment == "high" and gap1 >= 0.03)
            or (specificity == "low" and top_fulfillment == "high" and very_strong_gap)
            or named_person_high_confidence
        )
    ):
        confidence = "high"
        route = "description_only"
    elif (
        top_fulfillment in {"high", "medium"}
        and not contradictions
        and (
            convergence in {"strong", "partial"}
            or lexical_only_equivalent
            or (convergence == "weak" and top_fulfillment == "high" and gap1 >= 0.05)
        )
    ):
        confidence = "medium"
        route = "judge_5"
    else:
        confidence = "low"
        route = "judge_8"

    return {
        "confidence": confidence,
        "route": route,
        "specificity": specificity,
        "convergence": convergence,
        "top_fulfillment": top_fulfillment,
        "gap1": round(gap1, 3),
        "gap5": round(gap5, 3),
        "contradictions": contradictions,
        "semantic_available": semantic_available,
        "lexical_available": lexical_available,
        "top_candidate_tmdb_id": int(top["tmdb_id"]),
    }


def build_query_text(preferences: str, retrieval_profile: dict[str, Any]) -> str:
    return retrieval_profile["lexical_query_text"] or str(preferences or "").strip()


def seed_similarity_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if not retrieval_profile.get("similarity_request"):
        return 0.0
    score = 0.0
    score += 1.6 * len(retrieval_profile["seed_genres"].intersection(row["genres_set"]))
    score += 0.6 * len(retrieval_profile["seed_keywords"].intersection(row["keywords_set"]))
    if int(row["tmdb_id"]) in retrieval_profile["seed_similar_ids"]:
        score += 2.5
    if int(row["tmdb_id"]) in retrieval_profile.get("seed_recommended_ids", set()):
        score += 1.8
    return score


def keyword_alignment_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    positive_tokens = retrieval_profile["positive_query_tokens"] or retrieval_profile["raw_preference_tokens"]
    if not positive_tokens:
        return 0.0
    blob_overlap = len(positive_tokens.intersection(row["search_blob_tokens"]))
    keyword_overlap = len(positive_tokens.intersection(row["keywords_set"]))
    score = 0.6 * blob_overlap + 0.4 * keyword_overlap
    return _normalize_component(score, 6.0)


def genre_alignment_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    targets = retrieval_profile["explicit_genre_targets"]
    if not targets:
        return 0.0
    overlap = len(targets.intersection(row["genres_set"]))
    if overlap <= 0:
        return -1.0
    return _normalize_component(float(overlap), float(len(targets)))


def avoid_penalty_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    negative_overlap = len(retrieval_profile["negative_match_tokens"].intersection(row["search_blob_match_tokens"]))
    phrase_hits = sum(
        1 for phrase in retrieval_profile["negative_phrases"] if normalize_text(phrase) in row["search_blob"]
    )
    hard_block = 1.0 if retrieval_profile["hard_block_genres"].intersection(row["genres_set"]) else 0.0
    return _normalize_component(float(negative_overlap + phrase_hits) + hard_block * 2.0, 5.0)


def quality_prior_score(row: pd.Series) -> float:
    rating = max(0.0, min(float(row["vote_average"]) / 10.0, 1.0))
    votes = _normalize_component(float(np.log1p(max(float(row["vote_count"]), 0.0))), float(np.log1p(8000.0)))
    return 0.65 * rating + 0.35 * votes


def novelty_penalty(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    penalty = 4.0 if row["title_root"] and row["title_root"] in retrieval_profile["watched_roots"] else 0.0
    return penalty


def person_anchor_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    signals = retrieval_profile.get("named_person_signals", [])
    if not signals:
        return 0.0
    director_matches = len(set(signals).intersection(row["director_set"]))
    cast_matches = len(set(signals).intersection(row["cast_set"]))
    if director_matches > 0:
        return 1.0
    if cast_matches > 0:
        return 0.6
    return 0.0


def _normalize_component(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def hybrid_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    semantic_component = float(row.get("semantic_score", 0.0))
    fts_component = float(row.get("fts_score", 0.0))
    keyword_component = keyword_alignment_score(row, retrieval_profile)
    genre_component = genre_alignment_score(row, retrieval_profile)
    genre_reward = max(0.0, genre_component)
    genre_penalty = max(0.0, -genre_component)
    quality_component = quality_prior_score(row)
    seed_component = _normalize_component(seed_similarity_score(row, retrieval_profile), 8.0)
    avoid_component = avoid_penalty_score(row, retrieval_profile)
    novelty_component = _normalize_component(novelty_penalty(row, retrieval_profile), 4.0)
    person_component = person_anchor_score(row, retrieval_profile)
    if retrieval_profile.get("has_named_person_signal"):
        semantic_weight = 0.48
        fts_weight = 0.22
        person_weight = 0.18
    else:
        semantic_weight = 0.62
        fts_weight = 0.16
        person_weight = 0.0
    return (
        semantic_weight * semantic_component
        + fts_weight * fts_component
        + 0.10 * keyword_component
        + 0.08 * quality_component
        + 0.12 * seed_component
        + 0.10 * genre_reward
        + person_weight * person_component
        - 0.18 * avoid_component
        - 0.10 * genre_penalty
        - 0.05 * novelty_component
    )


def local_fallback_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if int(row["tmdb_id"]) in retrieval_profile["history_tmdb_ids"] or row["title"] in retrieval_profile["history_titles"]:
        return -10_000.0
    if int(row["tmdb_id"]) in retrieval_profile.get("exclude_seed_tmdb_ids", set()):
        return -10_000.0

    return (
        2.5 * keyword_alignment_score(row, retrieval_profile)
        + 0.9 * seed_similarity_score(row, retrieval_profile)
        + 0.8 * quality_prior_score(row)
        + 0.6 * max(0.0, genre_alignment_score(row, retrieval_profile))
        + 0.8 * person_anchor_score(row, retrieval_profile)
        - 1.8 * avoid_penalty_score(row, retrieval_profile)
        - _normalize_component(novelty_penalty(row, retrieval_profile), 4.0)
    )


def diversify_candidates(candidates: pd.DataFrame, limit: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.head(limit)

    primary_indices: list[int] = []
    deferred_indices: list[int] = []
    seen_roots: set[str] = set()

    for row in candidates.itertuples():
        if row.title_root and row.title_root in seen_roots:
            deferred_indices.append(row.Index)
            continue
        primary_indices.append(row.Index)
        if row.title_root:
            seen_roots.add(row.title_root)

    indices = primary_indices[:limit]
    if len(indices) < limit:
        indices.extend(deferred_indices[: max(0, limit - len(indices))])

    return candidates.loc[indices].sort_values(["second_stage_score", "vote_average", "vote_count"], ascending=False)


def filter_semantic_only_candidates(candidates: pd.DataFrame, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    semantic_only_mask = (candidates["semantic_score"] > 0.0) & (candidates["fts_score"] <= 0.0)
    if not semantic_only_mask.any():
        return candidates

    keep_mask = pd.Series(True, index=candidates.index)
    semantic_only = candidates[semantic_only_mask]

    targets = retrieval_profile["explicit_genre_targets"]
    if targets:
        minimum_overlap = 1 if len(targets) == 1 else 2
        genre_keep = semantic_only["genres_set"].map(lambda genres: len(targets.intersection(genres)) >= minimum_overlap)
        keep_mask.loc[semantic_only.index] &= genre_keep

    avoid_hits = semantic_only["search_blob_match_tokens"].map(
        lambda tokens: bool(tokens.intersection(retrieval_profile["negative_match_tokens"]))
    )
    keep_mask.loc[semantic_only.index] &= ~avoid_hits

    if retrieval_profile.get("has_named_person_signal"):
        keep_mask.loc[semantic_only.index] &= semantic_only["semantic_score"] >= 0.72

    return candidates[keep_mask].copy()


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
    retrieval_profile_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    retrieval_profile = build_retrieval_profile(preferences, history)
    retrieval_profile = merge_intent_override(retrieval_profile, retrieval_profile_override)
    lexical_query_text = build_query_text(preferences, retrieval_profile)
    semantic_query_text = retrieval_profile["semantic_query_text"]
    exclude_ids = set(retrieval_profile["history_tmdb_ids"])

    from fts_retrieval import fts_ready, search_fts
    from semantic_retrieval import search_semantic, semantic_ready

    lexical_available = mode in {"hybrid", "lexical"} and fts_ready()
    semantic_available = mode in {"hybrid", "semantic"} and semantic_ready()
    lexical_hits = search_fts(lexical_query_text, exclude_ids=exclude_ids, limit=FTS_LIMIT) if lexical_available else []
    try:
        semantic_hits = search_semantic(semantic_query_text, exclude_ids=exclude_ids, limit=SEMANTIC_LIMIT) if semantic_available else []
    except Exception:
        semantic_hits = []

    resolved_mode = mode
    if mode == "hybrid":
        has_lexical = bool(lexical_hits)
        has_semantic = bool(semantic_hits)
        if has_lexical and has_semantic:
            resolved_mode = "hybrid"
        elif has_semantic:
            resolved_mode = "semantic"
        elif has_lexical:
            resolved_mode = "lexical"

    candidates = _candidate_frame(lexical_hits, semantic_hits, retrieval_profile, resolved_mode)
    candidates = filter_semantic_only_candidates(candidates, retrieval_profile)
    if candidates.empty:
        fallback, fallback_profile = local_fallback_candidates(preferences, history)
        return fallback, fallback_profile, "local_fallback"

    candidates["second_stage_score"] = candidates.apply(hybrid_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["second_stage_score", "semantic_score", "fts_score", "vote_average", "vote_count"], ascending=False)
    retrieval_profile = dict(retrieval_profile)
    retrieval_profile["lexical_hit_count"] = len(lexical_hits)
    retrieval_profile["semantic_hit_count"] = len(semantic_hits)
    retrieval_profile["semantic_active"] = bool(semantic_hits)
    retrieval_profile["lexical_available"] = bool(lexical_available)
    retrieval_profile["semantic_available"] = bool(semantic_available)
    retrieval_profile["preserve_fts_tmdb_ids"] = [
        int(hit["tmdb_id"])
        for hit in lexical_hits[:3]
        if retrieval_profile.get("has_named_person_signal")
    ]
    return candidates.head(MERGED_POOL_SIZE).copy(), retrieval_profile, resolved_mode


def build_shortlist(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    mode: str = "hybrid",
    retrieval_profile_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates, retrieval_profile, resolved_mode = build_candidate_pool(preferences, history, mode=mode, retrieval_profile_override=retrieval_profile_override)
    prompt_profile = build_prompt_profile(preferences, retrieval_profile)

    reranked = candidates.head(SECOND_STAGE_POOL_SIZE).copy()
    preserve_fts_ids = retrieval_profile.get("preserve_fts_tmdb_ids", [])
    if preserve_fts_ids:
        preserved = candidates[candidates["tmdb_id"].isin(preserve_fts_ids)]
        reranked = pd.concat([reranked, preserved], ignore_index=False).drop_duplicates(subset=["tmdb_id"], keep="first")
    reranked = reranked.sort_values(["second_stage_score", "semantic_score", "fts_score", "vote_average", "vote_count"], ascending=False)
    reranked = diversify_candidates(reranked, SHORTLIST_SIZE)

    retrieval_profile = dict(retrieval_profile)
    retrieval_profile["retrieval_mode"] = resolved_mode

    shortlist = []
    for row in reranked.head(SHORTLIST_SIZE).itertuples():
        component_scores = _component_scores(MOVIES_BY_TMDB_ID.loc[int(row.tmdb_id)], retrieval_profile)
        shortlist.append(
            {
                "tmdb_id": int(row.tmdb_id),
                "title": row.title,
                "score": round(float(row.second_stage_score), 3),
                "semantic_score": round(float(getattr(row, "semantic_score", 0.0)), 3),
                "fts_score": round(float(getattr(row, "fts_score", 0.0)), 3),
                **component_scores,
            }
        )
    retrieval_profile["confidence_bundle"] = build_confidence_bundle(shortlist, retrieval_profile, prompt_profile)
    return shortlist, prompt_profile, retrieval_profile
