from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHORTLIST_SIZE = 10
LEXICAL_LIMIT = 80
SEMANTIC_LIMIT = 30
MERGED_POOL_SIZE = 60
SECOND_STAGE_POOL_SIZE = 36
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENABLE_HF_SEMANTIC_RETRIEVAL = os.getenv("ENABLE_HF_SEMANTIC_RETRIEVAL", "0") == "1"
RRF_K = 60.0
MAX_OVERVIEW_KEYWORDS = 18

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "tmdb_top1000_movies.csv"
ENRICHED_DATA_PATH = DATA_DIR / "tmdb_top1000_movies_enriched.csv"
ACTIVE_DATA_PATH = ENRICHED_DATA_PATH if ENRICHED_DATA_PATH.exists() else DATA_PATH

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
    "production_companies",
    "collection_name",
    "spoken_languages",
    "alternative_titles",
    "us_rating",
    "essence",
    "tone_tags_json",
    "audience_tags_json",
    "source_tags_json",
    "keywords_augmented_json",
]
OPTIONAL_TEXT_COLUMNS = [
    "collection_name",
    "spoken_languages",
    "alternative_titles",
    "us_rating",
    "production_companies",
]
LLM_AUGMENTATION_TEXT_COLUMNS = [
    "essence",
    "tone_tags_json",
    "audience_tags_json",
    "source_tags_json",
    "keywords_augmented_json",
]
OPTIONAL_DATA_COLUMNS = [
    "similar_tmdb_ids",
    "recommended_tmdb_ids",
    "imdb_id",
    "imdb_rating",
    "imdb_votes",
    "imdb_source",
    "updated_at",
    *OPTIONAL_TEXT_COLUMNS,
    *LLM_AUGMENTATION_TEXT_COLUMNS,
]
QUALITY_PREFERENCE_RE = re.compile(
    r"\b(?:good|great|best|top|high[-\s]?quality|high[-\s]?rated|highly rated|acclaimed|popular|crowd[-\s]?pleasing|well reviewed|must[-\s]?watch|greatest|masterpiece)\b",
    re.IGNORECASE,
)
QUALITY_PHRASE_RE = re.compile(
    r"\b(?:best|greatest|top)(?:\s+[a-z0-9]+){0,5}\s+(?:of\s+)?all\s+time\b|\ball[-\s]?time\s+(?:best|great|classic|favorite)\b",
    re.IGNORECASE,
)
QUALITY_QUERY_FILLER_TOKENS = {
    "all",
    "best",
    "ever",
    "favorite",
    "good",
    "great",
    "greatest",
    "high",
    "highly",
    "masterpiece",
    "must",
    "popular",
    "acclaimed",
    "rated",
    "rating",
    "time",
    "top",
    "watch",
}
REQUEST_FILLER_TOKENS = {
    "cast",
    "casting",
    "directed",
    "dislikes",
    "fan",
    "fans",
    "film",
    "films",
    "find",
    "give",
    "likes",
    "recommend",
    "recommendation",
    "please",
    "show",
    "someone",
    "starring",
    "suggest",
    "who",
}
MOOD_QUERY_HINTS = (
    (
        re.compile(r"\bfast[-\s]?paced\b|\bfast\s+movie\b|\bexciting\b|\bhigh[-\s]?energy\b", re.IGNORECASE),
        ("speed", "exciting", "action", "race", "chase", "suspenseful"),
    ),
)
FUZZY_SEMANTIC_RE = re.compile(
    r"\b(?:like|similar|vibe|feel|feels|style|mood|tone|atmospheric|thoughtful|weird|grounded|bleak|slow[-\s]?burn|dystopian|modern story|modern setting|epic|intense|tense|mind[-\s]?bending|light[-\s]?hearted|uplifting|heartwarming|charming|comfort|cozy)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9']+")
NEGATION_RE = re.compile(r"\b(?:no|not|without|avoid)\s+([a-z0-9][a-z0-9\-\s]{0,32}?)(?=,|\.|;|\bbut\b|\bexcept\b|\binstead\b|$)", re.IGNORECASE)
SIMILARITY_RE = re.compile(r"\b(?:like|similar to|something like|in the vein of|along the lines of)\b", re.IGNORECASE)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "avoid",
    "be",
    "but",
    "exclude",
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
    "not",
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
    "without",
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
    "romantic": "romance",
    "sci fi": "science fiction",
    "sci-fi": "science fiction",
    "science fiction": "science fiction",
    "thriller": "thriller",
    "war": "war",
    "western": "western",
}
YEAR_SIGNAL_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:19|20)\d0s\b|\b(?:80s|90s|2000s|2010s|2020s)\b", re.IGNORECASE)
YEAR_SIGNAL_PHRASES = (
    "recent",
    "newer",
    "latest",
    "older",
    "classic",
    "relatively new",
    "new movie",
    "older movie",
    "old movie",
    "this decade",
    "last decade",
)
RELEASE_YEAR_CONTEXT_RE = re.compile(
    r"\b(?:released|release|made|produced|came out|from|movie from|film from|after|before|since|newer than|older than|between|recent|newer|latest|classic)\b",
    re.IGNORECASE,
)
SETTING_PERIOD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:based in modern times|set in modern times|takes place in modern times|modern setting|modern day|modern-day|present day|present-day)\b", re.IGNORECASE), "contemporary setting"),
    (re.compile(r"\b(?:period piece|period drama|historical setting|set in the past)\b", re.IGNORECASE), "historical setting"),
    (re.compile(r"\bset (?:in|during) (?:the )?((?:19|20)\d0s|80s|90s|(?:19|20)\d{2})\b", re.IGNORECASE), "period setting"),
)
SHORT_RUNTIME_RE = re.compile(r"\b(?:short|quick|brief|light(?:\s+watch)?)\b", re.IGNORECASE)
LONG_RUNTIME_RE = re.compile(r"\b(?:long|epic|lengthy)\b", re.IGNORECASE)
MAX_RUNTIME_RE = re.compile(
    r"\b(?:under|less than|within|below|max(?:imum)?|up to)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)?\b",
    re.IGNORECASE,
)
MIN_RUNTIME_RE = re.compile(
    r"\b(?:over|more than|at least|min(?:imum)?)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)?\b",
    re.IGNORECASE,
)
MIN_RATING_RE = re.compile(
    r"\b(?:imdb\s+)?(?:rating|rated|score)\s*(?:above|over|at least|higher than|>=)\s*(\d(?:\.\d+)?)\b"
    r"|\b(?:above|over|at least|higher than|>=)\s*(\d(?:\.\d+)?)\s*(?:/10)?\s*(?:imdb\s+)?(?:rating|rated|score)\b",
    re.IGNORECASE,
)


def extract_year_constraint(preferences: str) -> dict[str, Any] | None:
    normalized = normalize_text(preferences)
    if not normalized:
        return None
    if re.search(r"\bset (?:in|during)\b", normalized) and not RELEASE_YEAR_CONTEXT_RE.search(normalized):
        return None

    between = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", normalized)
    if between:
        start, end = sorted((int(between.group(1)), int(between.group(2))))
        return {"min_year": start, "max_year": end, "label": f"between {start} and {end}"}

    decade = re.search(r"\b((?:19|20)\d0)s\b", normalized)
    if decade:
        start = int(decade.group(1))
        return {"min_year": start, "max_year": start + 9, "label": f"{start}s"}

    short_decade = re.search(r"\b(80|90)s\b", normalized)
    if short_decade:
        start = 1900 + int(short_decade.group(1))
        return {"min_year": start, "max_year": start + 9, "label": f"{start}s"}

    year_match = re.search(r"\b((?:19|20)\d{2})\b", normalized)
    if not year_match:
        return None

    year = int(year_match.group(1))
    prefix = normalized[max(0, year_match.start() - 24) : year_match.start()]
    if re.search(r"\b(after|post|newer than|since)\s+$", prefix):
        min_year = year + (1 if "after" in prefix or "newer than" in prefix or "post" in prefix else 0)
        return {"min_year": min_year, "max_year": 9999, "label": f"after {year}"}
    if re.search(r"\b(before|pre|older than|prior to)\s+$", prefix):
        max_year = year - (1 if "before" in prefix or "older than" in prefix or "pre" in prefix else 0)
        return {"min_year": 0, "max_year": max_year, "label": f"before {year}"}
    if re.search(r"\b(from|in|around)\s+(?:the\s+)?$", prefix):
        return {"min_year": year, "max_year": year, "label": str(year)}
    return None


def extract_setting_period(preferences: str) -> str:
    normalized = normalize_text(preferences)
    for pattern, label in SETTING_PERIOD_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        if label == "period setting" and match.groups():
            return f"set in {match.group(1)}"
        return label
    return ""


def _runtime_minutes(amount: str, unit: str | None) -> int | None:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    unit_text = str(unit or "").lower()
    if unit_text.startswith(("h", "hr")):
        return int(round(value * 60))
    return int(round(value))


def extract_runtime_constraint(preferences: str) -> dict[str, Any] | None:
    normalized = normalize_text(preferences)
    min_runtime: int | None = None
    max_runtime: int | None = None
    labels: list[str] = []
    short_requested = bool(SHORT_RUNTIME_RE.search(normalized))

    max_match = MAX_RUNTIME_RE.search(normalized)
    if max_match:
        parsed = _runtime_minutes(max_match.group(1), max_match.group(2))
        if parsed and parsed > 0:
            max_runtime = parsed
            labels.append(f"under {parsed} minutes")

    min_match = MIN_RUNTIME_RE.search(normalized)
    if min_match:
        parsed = _runtime_minutes(min_match.group(1), min_match.group(2))
        if parsed and parsed > 0:
            min_runtime = parsed
            labels.append(f"over {parsed} minutes")

    if max_runtime is None and short_requested:
        max_runtime = 110
        labels.append("short")
    if min_runtime is None and LONG_RUNTIME_RE.search(normalized):
        min_runtime = 130
        labels.append("long")

    if min_runtime is None and max_runtime is None and not short_requested:
        return None
    return {
        "min_runtime": min_runtime,
        "max_runtime": max_runtime,
        "short_requested": short_requested,
        "label": ", ".join(labels) or "runtime request",
    }


def extract_rating_constraint(preferences: str) -> dict[str, Any] | None:
    normalized = normalize_text(preferences)
    match = MIN_RATING_RE.search(normalized)
    if not match:
        return None
    raw_rating = next((group for group in match.groups() if group), "")
    try:
        min_rating = float(raw_rating)
    except (TypeError, ValueError):
        return None
    if min_rating <= 0.0 or min_rating > 10.0:
        return None
    return {"min_rating": min_rating, "label": f"rating at least {min_rating:g}/10"}


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


def _parse_numeric(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_vote_count(value: Any) -> float:
    text = str(value or "").strip().lower().replace(",", "")
    if not text:
        return float("nan")
    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def ensure_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    movies = df.copy()
    for column in OPTIONAL_DATA_COLUMNS:
        if column not in movies.columns:
            movies[column] = ""
    return movies


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("scifi", "science fiction")
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


def extract_quality_preference(preferences: str) -> bool:
    return bool(QUALITY_PREFERENCE_RE.search(str(preferences or "")) or QUALITY_PHRASE_RE.search(str(preferences or "")))


def extract_all_time_quality_preference(preferences: str) -> bool:
    return bool(QUALITY_PHRASE_RE.search(str(preferences or "")))


def clean_positive_query_tokens(tokens: set[str], preferences: str, *, quality_preference: bool) -> set[str]:
    cleaned = set(tokens)
    cleaned.difference_update(REQUEST_FILLER_TOKENS)
    if quality_preference:
        cleaned.difference_update(QUALITY_QUERY_FILLER_TOKENS)
    elif QUALITY_PHRASE_RE.search(str(preferences or "")):
        cleaned.difference_update({"all", "best", "greatest", "time"})
    return cleaned


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
        if len(variant) < 3:
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
        negative_match_tokens.update({normalize_match_token(token) for token in tokens})
        phrase_variants = {phrase, phrase.replace("-", " ")}
        for variant in phrase_variants:
            if variant in GENRE_ALIASES:
                hard_block_genres.add(GENRE_ALIASES[variant])
        for alias, canonical in GENRE_ALIASES.items():
            for variant in phrase_variants:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", variant):
                    hard_block_genres.add(canonical)

    for match in re.finditer(r"\bnon[-\s]+([a-z0-9]+)\b", normalized):
        phrase = normalize_text(match.group(1))
        if not phrase:
            continue
        negative_phrases.append(phrase)
        negative_tokens.update({"non", phrase})
        negative_match_tokens.update(match_tokens(phrase))
        negative_match_tokens.update({normalize_match_token(phrase)})
        if phrase in GENRE_ALIASES:
            hard_block_genres.add(GENRE_ALIASES[phrase])

    for match in re.finditer(r"\b(?:dislikes?|hates?|doesn'?t like|do not like)\s+([a-z0-9][a-z0-9\-\s]{0,24}?)(?=,|\.|;|\bbut\b|\bwith\b|\bfilms?\b|\bmovies?\b|$)", normalized):
        phrase = " ".join(match.group(1).split()).strip()
        if not phrase:
            continue
        negative_phrases.append(phrase)
        tokens = tokenize(phrase)
        negative_tokens.update(tokens)
        negative_match_tokens.update(match_tokens(phrase))
        negative_match_tokens.update({normalize_match_token(token) for token in tokens})
        if phrase in GENRE_ALIASES:
            hard_block_genres.add(GENRE_ALIASES[phrase])

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


def _lexical_terms(value: Any) -> list[str]:
    terms: list[str] = []
    for token in TOKEN_RE.findall(normalize_text(value)):
        if token in STOP_WORDS:
            continue
        normalized = normalize_match_token(token)
        if normalized and normalized not in STOP_WORDS:
            terms.append(normalized)
    return terms


def _json_list_text(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, list):
        return " ".join(str(item or "") for item in parsed)
    return raw


def _add_weighted_terms(counter: Counter[str], value: Any, weight: float) -> None:
    for term, count in Counter(_lexical_terms(value)).items():
        counter[term] += count * weight


def _normalize_component(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def add_quality_columns(movies: pd.DataFrame) -> pd.DataFrame:
    prepared = movies.copy()
    tmdb_rating = pd.to_numeric(prepared.get("vote_average", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=10.0)
    tmdb_votes = pd.to_numeric(prepared.get("vote_count", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    imdb_rating = prepared.get("imdb_rating", pd.Series([""] * len(prepared), index=prepared.index)).map(_parse_numeric)
    imdb_votes = prepared.get("imdb_votes", pd.Series([""] * len(prepared), index=prepared.index)).map(_parse_vote_count)

    prepared["effective_rating"] = imdb_rating.where(imdb_rating.notna() & (imdb_rating > 0), tmdb_rating)
    prepared["effective_votes"] = imdb_votes.where(imdb_votes.notna() & (imdb_votes > 0), tmdb_votes)

    global_vote_avg = float(prepared["effective_rating"].replace(0, np.nan).mean())
    if not np.isfinite(global_vote_avg):
        global_vote_avg = float(tmdb_rating.replace(0, np.nan).mean() or 6.0)
    bayes_m = max(300.0, float(prepared["effective_votes"].quantile(0.70)))
    votes = prepared["effective_votes"].astype(float).clip(lower=0.0)
    ratings = prepared["effective_rating"].astype(float).clip(lower=0.0, upper=10.0)
    prepared["bayesian_rating"] = ((votes / (votes + bayes_m)) * ratings) + ((bayes_m / (votes + bayes_m)) * global_vote_avg)

    max_vote_log = max(1.0, float(np.log1p(votes.max())))
    prepared["vote_reliability"] = votes.map(lambda value: _normalize_component(float(np.log1p(max(value, 0.0))), max_vote_log))
    prepared["consensus_quality_score"] = (
        0.72 * (prepared["bayesian_rating"] / 10.0).clip(lower=0.0, upper=1.0)
        + 0.28 * prepared["vote_reliability"].clip(lower=0.0, upper=1.0)
    )
    return prepared


def add_lexical_columns(movies: pd.DataFrame) -> pd.DataFrame:
    prepared = movies.copy()
    base_counters: list[Counter[str]] = []
    overview_counters: list[Counter[str]] = []
    doc_term_sets: list[set[str]] = []

    for row in prepared.itertuples(index=False):
        base: Counter[str] = Counter()
        _add_weighted_terms(base, getattr(row, "title", ""), 4.0)
        _add_weighted_terms(base, getattr(row, "original_title", ""), 3.0)
        _add_weighted_terms(base, getattr(row, "alternative_titles", ""), 3.0)
        _add_weighted_terms(base, getattr(row, "genres", ""), 3.4)
        _add_weighted_terms(base, getattr(row, "keywords", ""), 3.0)
        _add_weighted_terms(base, _json_list_text(getattr(row, "tone_tags_json", "")), 2.5)
        _add_weighted_terms(base, _json_list_text(getattr(row, "audience_tags_json", "")), 2.0)
        _add_weighted_terms(base, _json_list_text(getattr(row, "source_tags_json", "")), 1.8)
        _add_weighted_terms(base, getattr(row, "tagline", ""), 1.8)
        _add_weighted_terms(base, getattr(row, "director", ""), 2.2)
        _add_weighted_terms(base, getattr(row, "top_cast", ""), 2.0)
        _add_weighted_terms(base, getattr(row, "production_countries", ""), 1.4)
        _add_weighted_terms(base, getattr(row, "spoken_languages", ""), 1.4)
        _add_weighted_terms(base, getattr(row, "original_language", ""), 1.2)
        _add_weighted_terms(base, getattr(row, "essence", ""), 1.1)

        overview_counter = Counter(_lexical_terms(getattr(row, "overview", "")))
        base_counters.append(base)
        overview_counters.append(overview_counter)
        doc_term_sets.append(set(base) | set(overview_counter))

    doc_count = max(1, len(prepared))
    df_counts: Counter[str] = Counter()
    for terms in doc_term_sets:
        df_counts.update(terms)
    idf = {term: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0 for term, freq in df_counts.items()}

    lexical_counters: list[Counter[str]] = []
    lexical_terms: list[set[str]] = []
    doc_lengths: list[float] = []
    for base, overview_counter in zip(base_counters, overview_counters):
        combined = base.copy()
        overview_terms = sorted(
            overview_counter.items(),
            key=lambda item: (item[1] * idf.get(item[0], 1.0), item[1], item[0]),
            reverse=True,
        )
        for term, count in overview_terms[:MAX_OVERVIEW_KEYWORDS]:
            combined[term] += count * 1.0
        lexical_counters.append(combined)
        lexical_terms.append(set(combined))
        doc_lengths.append(float(sum(combined.values()) or 1.0))

    prepared["lexical_counter"] = lexical_counters
    prepared["lexical_terms"] = lexical_terms
    prepared["lexical_doc_len"] = doc_lengths
    prepared.attrs["lexical_idf"] = idf
    prepared.attrs["avg_lexical_doc_len"] = float(np.mean(doc_lengths) if doc_lengths else 1.0)
    return prepared


def prepare_movies(df: pd.DataFrame) -> pd.DataFrame:
    movies = ensure_dataset_columns(df)
    movies = add_quality_columns(movies)
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
    return add_lexical_columns(movies)


TOP_MOVIES = ensure_dataset_columns(pd.read_csv(ACTIVE_DATA_PATH).fillna(""))
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
KNOWN_COUNTRY_LANGUAGE_TERMS = tuple(
    sorted(
        {
            term
            for row in MOVIES.itertuples()
            for term in (*split_csvish(row.production_countries), *split_csvish(row.spoken_languages))
            if len(term) >= 4
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
    seed_title_roots: set[str] = set()
    seed_titles: list[str] = []

    for row in seed_df.itertuples():
        seed_genres.update(row.genres_set)
        seed_keywords.update(set(sorted(row.keywords_set)[:10]))
        seed_title_tokens.update(tokenize(row.title))
        seed_similar_ids.update(row.similar_ids_set)
        seed_recommended_ids.update(row.recommended_ids_set)
        seed_tmdb_ids.add(int(row.tmdb_id))
        if row.title_root:
            seed_title_roots.add(row.title_root)
        seed_titles.append(str(row.title))

    return {
        "seed_titles": seed_titles,
        "seed_tmdb_ids": seed_tmdb_ids,
        "seed_title_roots": seed_title_roots,
        "seed_genres": seed_genres,
        "seed_keywords": seed_keywords,
        "seed_title_tokens": seed_title_tokens,
        "seed_similar_ids": seed_similar_ids,
        "seed_recommended_ids": seed_recommended_ids,
    }


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


def extract_country_or_language_signals(preferences: str) -> list[str]:
    normalized = normalize_text(preferences)
    matches: list[str] = []
    for term in KNOWN_COUNTRY_LANGUAGE_TERMS:
        if _phrase_mentioned(normalized, term):
            matches.append(term)
    return matches[:3]


def extract_affinity_phrases(preferences: str) -> list[str]:
    normalized = normalize_text(preferences)
    phrases: list[str] = []
    patterns = (
        r"\b(?:likes?|loves?)\s+([a-z0-9][a-z0-9\-\s]{0,32}?)(?=\s+(?:films?|movies?)\b|,|\.|;|$)",
        r"\b([a-z0-9][a-z0-9\-\s]{0,24}?)\s+fans?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            phrase = " ".join(match.group(1).split()).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases[:3]


def derive_affinity_terms(phrases: list[str], negative_match_tokens: set[str]) -> set[str]:
    terms: Counter[str] = Counter()
    for phrase in phrases:
        normalized_phrase = normalize_text(phrase)
        if not normalized_phrase:
            continue
        phrase_tokens = match_tokens(normalized_phrase)
        matching_movies = MOVIES[MOVIES["search_blob"].map(lambda blob: normalized_phrase in blob)].head(40)
        if matching_movies.empty:
            continue
        for row in matching_movies.itertuples():
            for genre in row.genres_set:
                for token in match_tokens(genre):
                    terms[token] += 3
            for keyword in sorted(row.keywords_set)[:10]:
                for token in match_tokens(keyword):
                    terms[token] += 1
    for token in set(terms):
        if token in negative_match_tokens or token in STOP_WORDS:
            del terms[token]
    return {token for token, _ in terms.most_common(8)}


def build_retrieval_profile(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
) -> dict[str, Any]:
    history_df = history_rows(history)
    normalized_preferences = normalize_text(preferences)
    raw_preference_tokens = tokenize(preferences)
    negation_context = extract_negation_context(preferences)
    explicit_genre_targets = extract_explicit_genre_targets(preferences).difference(negation_context["hard_block_genres"])
    named_person_signals = extract_named_person_signals(preferences)
    seed_df = preference_seed_rows(preferences, history_df)
    seed_signals = derive_seed_signals(seed_df)
    similarity_request = bool(seed_signals["seed_tmdb_ids"]) and bool(SIMILARITY_RE.search(str(preferences or "")))
    year_constraint = extract_year_constraint(preferences)
    runtime_constraint = extract_runtime_constraint(preferences)
    rating_constraint = extract_rating_constraint(preferences)
    setting_period = extract_setting_period(preferences)
    country_or_language_signals = extract_country_or_language_signals(preferences)
    quality_preference = extract_quality_preference(preferences)
    all_time_quality_preference = extract_all_time_quality_preference(preferences)
    direct_mood_request = any(pattern.search(str(preferences or "")) for pattern, _ in MOOD_QUERY_HINTS)

    positive_query_tokens = set(negation_context["positive_query_tokens"])
    positive_query_tokens.difference_update(seed_signals["seed_title_tokens"])
    positive_query_tokens = clean_positive_query_tokens(
        positive_query_tokens,
        preferences,
        quality_preference=quality_preference,
    )

    for genre in explicit_genre_targets:
        positive_query_tokens.update(tokenize(genre))
    for signal in country_or_language_signals:
        positive_query_tokens.update(tokenize(signal))
    for pattern, hints in MOOD_QUERY_HINTS:
        if pattern.search(str(preferences or "")):
            positive_query_tokens.update(hints)
    if similarity_request:
        positive_query_tokens.update(token for genre in seed_signals["seed_genres"] for token in match_tokens(genre))
        positive_query_tokens.update(token for keyword in sorted(seed_signals["seed_keywords"])[:8] for token in match_tokens(keyword))
    affinity_phrases = extract_affinity_phrases(preferences)
    affinity_query_terms: set[str] = set()
    if affinity_phrases:
        affinity_query_terms = derive_affinity_terms(affinity_phrases, negation_context["negative_match_tokens"])
        positive_query_tokens.update(affinity_query_terms)
    affinity_genre_targets = {
        genre
        for genre in set(GENRE_ALIASES.values())
        if match_tokens(genre).intersection(affinity_query_terms)
    }
    negative_tokens = set(negation_context["negative_tokens"])

    lexical_query_text = " ".join(sorted(positive_query_tokens))
    semantic_query_text = str(preferences or "").strip()
    if not semantic_query_text:
        semantic_query_text = lexical_query_text

    return {
        "history_count": len(history),
        "history_exclusion_text": (
            "none"
            if len(history) <= 0
            else f"known_watch_history_count={len(history)}; use history primarily to avoid re-recommending already watched titles"
        ),
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
        "year_constraint": year_constraint,
        "runtime_constraint": runtime_constraint,
        "rating_constraint": rating_constraint,
        "setting_period": setting_period,
        "tone": [],
        "keyword_hints": [],
        "quality_preference": quality_preference,
        "all_time_quality_preference": all_time_quality_preference,
        "direct_mood_request": direct_mood_request,
        "affinity_phrases": affinity_phrases,
        "affinity_genre_targets": affinity_genre_targets,
        "country_or_language_signals": country_or_language_signals,
        "year_constraint_unavailable": False,
        "runtime_constraint_unavailable": False,
        "candidate_constraint_note": "",
        "exclude_seed_tmdb_ids": seed_signals["seed_tmdb_ids"] if similarity_request else set(),
        "exclude_seed_title_roots": seed_signals["seed_title_roots"] if similarity_request else set(),
        **seed_signals,
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

    release_year = intent_override.get("release_year")
    if isinstance(release_year, dict):
        try:
            min_year = int(release_year.get("min", release_year.get("min_year", 0)))
            max_year = int(release_year.get("max", release_year.get("max_year", 9999)))
            if min_year > 0 and max_year >= min_year:
                merged["year_constraint"] = {
                    "min_year": min_year,
                    "max_year": max_year,
                    "label": str(release_year.get("label") or f"{min_year}-{max_year}"),
                }
        except (TypeError, ValueError):
            pass

    runtime = intent_override.get("runtime")
    if isinstance(runtime, dict):
        try:
            min_runtime = runtime.get("min", runtime.get("min_runtime"))
            max_runtime = runtime.get("max", runtime.get("max_runtime"))
            parsed_runtime = {
                "min_runtime": int(min_runtime) if min_runtime not in (None, "") else None,
                "max_runtime": int(max_runtime) if max_runtime not in (None, "") else None,
                "short_requested": bool(runtime.get("short_requested", False)),
                "label": str(runtime.get("label") or "runtime request"),
            }
            if parsed_runtime["short_requested"] and parsed_runtime["max_runtime"] is None:
                parsed_runtime["max_runtime"] = 110
            if parsed_runtime["min_runtime"] or parsed_runtime["max_runtime"] or parsed_runtime["short_requested"]:
                merged["runtime_constraint"] = parsed_runtime
        except (TypeError, ValueError):
            pass

    setting_period = " ".join(str(intent_override.get("setting_period", "") or "").split()).strip()
    if setting_period:
        merged["setting_period"] = setting_period[:80]
        merged["positive_query_tokens"].update(tokenize(setting_period))

    tone = [" ".join(str(item or "").split()).strip() for item in intent_override.get("tone", []) if str(item or "").strip()]
    if tone:
        merged["tone"] = tone[:4]
        for item in merged["tone"]:
            merged["positive_query_tokens"].update(tokenize(item))

    keyword_hints = [
        " ".join(str(item or "").split()).strip()
        for item in intent_override.get("keyword_hints", [])
        if str(item or "").strip()
    ]
    if keyword_hints:
        merged["keyword_hints"] = keyword_hints[:5]
        for item in merged["keyword_hints"]:
            merged["positive_query_tokens"].update(tokenize(item))

    if isinstance(intent_override.get("quality_preference"), bool):
        merged["quality_preference"] = bool(merged.get("quality_preference")) or bool(intent_override["quality_preference"])

    country_or_language = [
        " ".join(str(item or "").split()).strip()
        for item in intent_override.get("country_or_language", [])
        if str(item or "").strip()
    ]
    if country_or_language:
        merged["country_or_language_signals"] = country_or_language[:3]
        for item in merged["country_or_language_signals"]:
            merged["positive_query_tokens"].update(tokenize(item))

    merged["positive_query_tokens"] = clean_positive_query_tokens(
        set(merged["positive_query_tokens"]),
        str(merged.get("normalized_preferences", "")),
        quality_preference=bool(merged.get("quality_preference")),
    )
    merged["lexical_query_text"] = " ".join(sorted(merged["positive_query_tokens"])) or merged["lexical_query_text"]
    semantic_hints = [
        str(merged.get("setting_period", "") or ""),
        " ".join(merged.get("tone", [])),
        " ".join(merged.get("keyword_hints", [])),
        " ".join(merged.get("country_or_language_signals", [])),
    ]
    semantic_hint_text = " ".join(item for item in semantic_hints if item).strip()
    if semantic_hint_text:
        merged["semantic_query_text"] = f"{merged['semantic_query_text']} {semantic_hint_text}".strip()
    return merged


def build_prompt_profile(preferences: str, retrieval_profile: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_text(preferences)
    year_relevant = bool(retrieval_profile.get("year_constraint")) or any(phrase in normalized for phrase in YEAR_SIGNAL_PHRASES)
    keyword_hints = list(dict.fromkeys([
        *list(retrieval_profile.get("keyword_hints", [])),
    ]))
    return {
        "target_genres": sorted(retrieval_profile["explicit_genre_targets"])[:4],
        "tone": list(retrieval_profile.get("tone", []))[:3],
        "keyword_hints": keyword_hints[:5],
        "avoid": retrieval_profile["negative_phrases"][:6],
        "year_relevant": year_relevant,
        "year_constraint": retrieval_profile.get("year_constraint"),
        "year_constraint_unavailable": bool(retrieval_profile.get("year_constraint_unavailable")),
        "runtime_constraint": retrieval_profile.get("runtime_constraint"),
        "runtime_constraint_unavailable": bool(retrieval_profile.get("runtime_constraint_unavailable")),
        "rating_constraint": retrieval_profile.get("rating_constraint"),
        "setting_period": retrieval_profile.get("setting_period", ""),
        "country_relevant": bool(retrieval_profile.get("country_or_language_signals")),
    }


def candidate_evidence(row: pd.Series, retrieval_profile: dict[str, Any]) -> dict[str, Any]:
    genre_match = not retrieval_profile["explicit_genre_targets"] or bool(
        retrieval_profile["explicit_genre_targets"].intersection(row["genres_set"])
    )
    constraint_source = str(row.get("constraint_source", "") or "")
    return {
        "genre_match": bool(genre_match),
        "avoid_hit": bool(has_avoid_hit(row, retrieval_profile)),
        "person_match": bool(has_person_match(row, retrieval_profile)),
        "year_match": bool(constraint_source == "year" or has_year_match(row, retrieval_profile)),
        "runtime_match": bool(constraint_source == "runtime" or has_runtime_match(row, retrieval_profile)),
        "seed_match": bool(has_seed_match(row, retrieval_profile)),
    }


def build_confidence_bundle(
    shortlist: list[dict[str, Any]],
    retrieval_profile: dict[str, Any],
    prompt_profile: dict[str, Any],
) -> dict[str, Any]:
    if not shortlist:
        return {
            "confidence": "low",
            "route": "judge_8",
            "specificity": "low",
            "convergence": "weak",
            "top_fulfillment": "low",
            "gap1": 0.0,
            "gap5": 0.0,
            "contradictions": ["empty_shortlist"],
        }

    top = shortlist[0]
    second = shortlist[1] if len(shortlist) > 1 else None
    fifth = shortlist[4] if len(shortlist) > 4 else None
    gap1 = float(top["score"]) - float(second["score"]) if second else float(top["score"])
    gap5 = float(top["score"]) - float(fifth["score"]) if fifth else gap1
    top_rating = float(top.get("effective_rating", top.get("vote_average", 0.0)) or 0.0)
    top_votes = int(float(top.get("effective_votes", top.get("vote_count", 0)) or 0))

    specificity_points = 0
    specificity_points += 1 if retrieval_profile["explicit_genre_targets"] else 0
    specificity_points += 1 if retrieval_profile["negative_phrases"] else 0
    specificity_points += 1 if retrieval_profile.get("named_person_signals") else 0
    specificity_points += 1 if prompt_profile.get("year_relevant") else 0
    specificity_points += 1 if retrieval_profile.get("runtime_constraint") else 0
    specificity_points += 1 if retrieval_profile.get("rating_constraint") else 0
    specificity_points += 1 if prompt_profile.get("setting_period") else 0
    specificity_points += 1 if retrieval_profile.get("similarity_request") else 0
    if specificity_points >= 3:
        specificity = "high"
    elif specificity_points >= 2:
        specificity = "medium"
    else:
        specificity = "low"

    semantic_available = bool(retrieval_profile.get("semantic_available"))
    lexical_available = bool(retrieval_profile.get("lexical_available"))
    fuzzy_or_mood_request = bool(
        retrieval_profile.get("similarity_request")
        or prompt_profile.get("setting_period")
        or retrieval_profile.get("tone")
        or retrieval_profile.get("keyword_hints")
        or FUZZY_SEMANTIC_RE.search(str(retrieval_profile.get("normalized_preferences", "")))
    )
    bm25_present = float(top.get("bm25_score", 0.0)) > 0.0
    semantic_present = float(top.get("semantic_score", 0.0)) > 0.0
    if semantic_available and bm25_present and semantic_present:
        convergence = "strong"
    elif bm25_present or semantic_present:
        convergence = "partial"
    else:
        convergence = "weak"

    contradictions: list[str] = []
    if retrieval_profile["explicit_genre_targets"] and not bool(top.get("genre_match", False)):
        contradictions.append("genre_miss")
    if retrieval_profile["negative_phrases"] and bool(top.get("avoid_hit", False)):
        contradictions.append("avoid_hit")
    if retrieval_profile.get("named_person_signals") and not bool(top.get("person_match", False)):
        contradictions.append("person_miss")
    if (
        retrieval_profile.get("year_constraint")
        and not retrieval_profile.get("year_constraint_unavailable")
        and not bool(top.get("year_match", False))
    ):
        contradictions.append("year_miss")
    if (
        retrieval_profile.get("runtime_constraint")
        and not retrieval_profile.get("runtime_constraint_unavailable")
        and not bool(top.get("runtime_match", False))
    ):
        contradictions.append("runtime_miss")
    if retrieval_profile.get("similarity_request") and not bool(top.get("seed_match", False)):
        contradictions.append("seed_miss")
    rating_constraint_satisfied = True
    rating_constraint = retrieval_profile.get("rating_constraint")
    if rating_constraint:
        try:
            min_rating = float(rating_constraint.get("min_rating", 0.0))
        except (TypeError, ValueError):
            min_rating = 0.0
        if min_rating > 0.0 and top_rating < min_rating:
            rating_constraint_satisfied = False

    top_source = float(top.get("source_score", top.get("score", 0.0)) or 0.0)
    top_quality = float(top.get("consensus_quality_score", 0.0) or 0.0)
    quality_safe = top_rating >= 6.4 and top_votes >= 120 and top_quality >= 0.55
    if retrieval_profile.get("quality_preference"):
        quality_safe = top_rating >= 6.8 and top_votes >= 500 and top_quality >= 0.60

    if contradictions:
        top_fulfillment = "low"
    elif top_source >= 0.50:
        top_fulfillment = "high"
    elif top_source >= 0.28:
        top_fulfillment = "medium"
    else:
        top_fulfillment = "low"

    quality_genre_request = bool(retrieval_profile.get("quality_preference") and retrieval_profile["explicit_genre_targets"])
    if (
        not contradictions
        and top_source >= 0.50
        and gap1 >= 0.05
        and quality_safe
        and not fuzzy_or_mood_request
        and not quality_genre_request
        and rating_constraint_satisfied
    ):
        confidence = "high"
        route = "description_only"
    elif not contradictions and top_source >= 0.28:
        confidence = "medium"
        route = "judge_8" if fuzzy_or_mood_request or retrieval_profile.get("similarity_request") else "judge_5"
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
        "fuzzy_or_mood_request": fuzzy_or_mood_request,
        "rating_constraint_satisfied": rating_constraint_satisfied,
        "quality_safe": quality_safe,
        "top_candidate_tmdb_id": int(top["tmdb_id"]),
    }


def build_query_text(preferences: str, retrieval_profile: dict[str, Any]) -> str:
    lexical_query_text = str(retrieval_profile.get("lexical_query_text") or "").strip()
    if lexical_query_text:
        return lexical_query_text
    if retrieval_profile.get("quality_preference") or is_broad_quality_request(retrieval_profile):
        return ""
    return str(preferences or "").strip()


def _bm25(tf: float, doc_len: float, avg_doc_len: float, idf: float, k1: float = 1.45, b: float = 0.72) -> float:
    if tf <= 0.0:
        return 0.0
    denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
    return idf * ((tf * (k1 + 1.0)) / denominator)


def _rank_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    span = max(high - low, 1e-6)
    return [(score - low) / span for score in scores]


def search_weighted_bm25(query_text: str, exclude_ids: set[int] | None = None, limit: int = LEXICAL_LIMIT) -> list[dict[str, Any]]:
    query_terms = _lexical_terms(query_text)
    if not query_terms:
        return []
    query_weights = Counter(query_terms)
    idf: dict[str, float] = MOVIES.attrs.get("lexical_idf", {})
    avg_doc_len = float(MOVIES.attrs.get("avg_lexical_doc_len", 1.0) or 1.0)
    exclude_ids = exclude_ids or set()

    scored: list[tuple[int, float]] = []
    for row in MOVIES.itertuples():
        tmdb_id = int(row.tmdb_id)
        if tmdb_id in exclude_ids:
            continue
        counter: Counter[str] = row.lexical_counter
        score = 0.0
        for term, query_weight in query_weights.items():
            tf = float(counter.get(term, 0.0))
            if tf <= 0.0:
                continue
            score += float(query_weight) * _bm25(tf, float(row.lexical_doc_len), avg_doc_len, float(idf.get(term, 1.0)))
        if score > 0.0:
            scored.append((tmdb_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[: max(0, int(limit))]
    normalized_scores = _rank_normalize([score for _, score in top])
    return [
        {
            "tmdb_id": tmdb_id,
            "lexical_score": normalized,
            "raw_lexical_score": raw_score,
            "lexical_rank": rank,
        }
        for rank, ((tmdb_id, raw_score), normalized) in enumerate(zip(top, normalized_scores), start=1)
    ]


def should_use_semantic_recall(retrieval_profile: dict[str, Any], lexical_hits: list[dict[str, Any]]) -> bool:
    if not ENABLE_HF_SEMANTIC_RETRIEVAL:
        return False
    normalized = str(retrieval_profile.get("normalized_preferences", ""))
    if retrieval_profile.get("has_named_person_signal") and not (
        retrieval_profile.get("similarity_request")
        or retrieval_profile.get("setting_period")
        or retrieval_profile.get("tone")
        or retrieval_profile.get("keyword_hints")
        or FUZZY_SEMANTIC_RE.search(normalized)
    ):
        return False
    if retrieval_profile.get("similarity_request") or retrieval_profile.get("setting_period"):
        return True
    if retrieval_profile.get("tone") or retrieval_profile.get("keyword_hints"):
        return True
    if FUZZY_SEMANTIC_RE.search(normalized):
        return True
    if len(lexical_hits) < 8 and not retrieval_profile.get("has_named_person_signal"):
        return True
    return False


def search_optional_semantic(
    retrieval_profile: dict[str, Any],
    exclude_ids: set[int],
    lexical_hits: list[dict[str, Any]],
    allow_semantic: bool = True,
) -> tuple[list[dict[str, Any]], bool, float]:
    if not allow_semantic:
        return [], False, 0.0
    if not should_use_semantic_recall(retrieval_profile, lexical_hits):
        return [], False, 0.0
    import time

    started = time.perf_counter()
    try:
        from semantic_retrieval import search_semantic, semantic_runtime_status

        if not semantic_runtime_status().get("ready"):
            return [], False, time.perf_counter() - started
        hits = search_semantic(
            str(retrieval_profile.get("semantic_query_text") or retrieval_profile.get("lexical_query_text") or ""),
            exclude_ids=exclude_ids,
            limit=SEMANTIC_LIMIT,
        )
        return hits, True, time.perf_counter() - started
    except Exception:
        return [], False, time.perf_counter() - started


def quality_prior_score(row: pd.Series) -> float:
    if "consensus_quality_score" in row:
        try:
            return max(0.0, min(float(row["consensus_quality_score"]), 1.0))
        except (TypeError, ValueError):
            pass
    rating = max(0.0, min(float(row["vote_average"]) / 10.0, 1.0))
    votes = _normalize_component(float(np.log1p(max(float(row["vote_count"]), 0.0))), float(np.log1p(8000.0)))
    return 0.65 * rating + 0.35 * votes


def is_broad_quality_request(retrieval_profile: dict[str, Any]) -> bool:
    if retrieval_profile.get("quality_preference"):
        return True
    if retrieval_profile.get("similarity_request") or retrieval_profile.get("direct_mood_request"):
        return False
    specificity = 0
    specificity += 1 if retrieval_profile.get("explicit_genre_targets") else 0
    specificity += 1 if retrieval_profile.get("negative_phrases") else 0
    specificity += 1 if retrieval_profile.get("named_person_signals") else 0
    specificity += 1 if retrieval_profile.get("year_constraint") else 0
    specificity += 1 if retrieval_profile.get("rating_constraint") else 0
    specificity += 1 if retrieval_profile.get("setting_period") else 0
    specificity += 1 if retrieval_profile.get("similarity_request") else 0
    specificity += 1 if retrieval_profile.get("tone") or retrieval_profile.get("keyword_hints") else 0
    return specificity <= 1


def needs_quality_backup(retrieval_profile: dict[str, Any]) -> bool:
    if retrieval_profile.get("quality_preference") or is_broad_quality_request(retrieval_profile):
        return True
    if retrieval_profile.get("similarity_request"):
        return True
    genre_or_affinity_request = bool(
        retrieval_profile.get("explicit_genre_targets")
        or retrieval_profile.get("affinity_genre_targets")
    )
    return bool(
        genre_or_affinity_request
        and not retrieval_profile.get("named_person_signals")
        and not retrieval_profile.get("year_constraint")
        and not retrieval_profile.get("runtime_constraint")
    )


def excluded_by_profile(row: pd.Series, retrieval_profile: dict[str, Any]) -> bool:
    tmdb_id = int(row["tmdb_id"])
    if tmdb_id in retrieval_profile["history_tmdb_ids"] or row["title"] in retrieval_profile["history_titles"]:
        return True
    if tmdb_id in retrieval_profile.get("exclude_seed_tmdb_ids", set()):
        return True
    return False


def seed_relation_for_id(tmdb_id: int, retrieval_profile: dict[str, Any]) -> str:
    if tmdb_id in retrieval_profile.get("seed_similar_ids", set()):
        return "similar"
    if tmdb_id in retrieval_profile.get("seed_recommended_ids", set()):
        return "recommended"
    return ""


def has_avoid_hit(row: pd.Series, retrieval_profile: dict[str, Any]) -> bool:
    return bool(
        retrieval_profile["negative_match_tokens"].intersection(row["search_blob_match_tokens"])
        or any(normalize_text(phrase) in row["search_blob"] for phrase in retrieval_profile["negative_phrases"])
        or retrieval_profile["hard_block_genres"].intersection(row["genres_set"])
    )


def has_person_match(row: pd.Series, retrieval_profile: dict[str, Any]) -> bool:
    signals = set(retrieval_profile.get("named_person_signals", []))
    return not signals or bool(signals.intersection(row["director_set"]) or signals.intersection(row["cast_set"]))


def has_year_match(row: pd.Series, retrieval_profile: dict[str, Any]) -> bool:
    constraint = retrieval_profile.get("year_constraint")
    if not constraint or retrieval_profile.get("year_constraint_unavailable"):
        return True
    try:
        year = int(row["year"])
    except (TypeError, ValueError):
        return False
    return int(constraint["min_year"]) <= year <= int(constraint["max_year"])


def has_runtime_match(row: pd.Series, retrieval_profile: dict[str, Any]) -> bool:
    constraint = retrieval_profile.get("runtime_constraint")
    if not constraint or retrieval_profile.get("runtime_constraint_unavailable"):
        return True
    runtime = pd.to_numeric(pd.Series([row.get("runtime_min")]), errors="coerce").iloc[0]
    if pd.isna(runtime):
        return False
    max_runtime = constraint.get("max_runtime")
    min_runtime = constraint.get("min_runtime")
    if max_runtime is not None and float(runtime) > float(max_runtime):
        return False
    if min_runtime is not None and float(runtime) < float(min_runtime):
        return False
    return True


def has_seed_match(row: pd.Series, retrieval_profile: dict[str, Any]) -> bool:
    if not retrieval_profile.get("similarity_request"):
        return True
    return bool(
        seed_relation_for_id(int(row["tmdb_id"]), retrieval_profile)
        or retrieval_profile["seed_genres"].intersection(row["genres_set"])
        or retrieval_profile["seed_keywords"].intersection(row["keywords_set"])
    )


def _rank_source_component(rank: Any) -> float:
    try:
        rank_value = float(rank)
    except (TypeError, ValueError):
        return 0.0
    if rank_value <= 0.0:
        return 0.0
    return 30.0 / (RRF_K + rank_value)


def candidate_source_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if excluded_by_profile(row, retrieval_profile):
        return -10_000.0

    score = _rank_source_component(row.get("bm25_rank", 0.0)) + _rank_source_component(row.get("semantic_rank", 0.0))

    seed_relation = str(row.get("seed_relation", "")) or seed_relation_for_id(int(row["tmdb_id"]), retrieval_profile)
    if seed_relation == "similar":
        score += 0.65
    elif seed_relation == "recommended":
        score += 0.50
    elif retrieval_profile.get("similarity_request") and has_seed_match(row, retrieval_profile):
        score += 0.12

    constraint_source = str(row.get("constraint_source", "") or "")
    if constraint_source == "quality":
        score += 0.45 if needs_quality_backup(retrieval_profile) else 0.35
    elif constraint_source:
        score += 0.18
    if retrieval_profile["explicit_genre_targets"] and retrieval_profile["explicit_genre_targets"].intersection(row["genres_set"]):
        score += 0.10
    if retrieval_profile.get("named_person_signals") and has_person_match(row, retrieval_profile):
        score += 0.22
    if retrieval_profile.get("year_constraint") and has_year_match(row, retrieval_profile):
        score += 0.08
    if retrieval_profile.get("runtime_constraint") and has_runtime_match(row, retrieval_profile):
        score += 0.06

    if has_avoid_hit(row, retrieval_profile):
        score -= 0.35

    quality_component = quality_prior_score(row)
    if retrieval_profile.get("quality_preference"):
        quality_weight = 0.35
    elif is_broad_quality_request(retrieval_profile):
        quality_weight = 0.25
    elif needs_quality_backup(retrieval_profile):
        quality_weight = 0.18
    else:
        quality_weight = 0.08
    score += quality_weight * quality_component

    return max(score, -10_000.0)


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

    return candidates.loc[indices].sort_values(["source_score", "consensus_quality_score", "effective_rating", "effective_votes"], ascending=False)


def filter_semantic_only_candidates(candidates: pd.DataFrame, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    semantic_only_mask = (candidates["semantic_score"] > 0.0) & (candidates["bm25_score"] <= 0.0)
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


def apply_quality_floor(candidates: pd.DataFrame, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    if retrieval_profile.get("quality_preference"):
        rating_floor = 8.0 if retrieval_profile.get("all_time_quality_preference") else 7.4
        vote_floor = 1000
    elif is_broad_quality_request(retrieval_profile):
        rating_floor = 6.5
        vote_floor = 1000
    else:
        return candidates

    stable = candidates[
        (pd.to_numeric(candidates["effective_rating"], errors="coerce").fillna(0.0) >= rating_floor)
        & (pd.to_numeric(candidates["effective_votes"], errors="coerce").fillna(0.0) >= vote_floor)
    ].copy()
    return stable if len(stable) >= min(8, len(candidates)) else candidates


def preserve_seed_candidates(candidates: pd.DataFrame, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    if not retrieval_profile.get("similarity_request"):
        return candidates

    seed_ids = set(retrieval_profile.get("seed_similar_ids", set())) | set(retrieval_profile.get("seed_recommended_ids", set()))
    seed_ids.difference_update(retrieval_profile.get("exclude_seed_tmdb_ids", set()))
    seed_ids.difference_update(retrieval_profile.get("history_tmdb_ids", set()))
    if not seed_ids:
        return candidates

    existing_ids = set(candidates["tmdb_id"].astype(int)) if not candidates.empty else set()
    missing_ids = seed_ids.difference(existing_ids)
    if not missing_ids:
        return candidates

    preserved = MOVIES[MOVIES["tmdb_id"].isin(missing_ids)].copy()
    if preserved.empty:
        return candidates
    if retrieval_profile.get("hard_block_genres"):
        preserved = preserved[
            ~preserved["genres_set"].map(lambda genres: bool(genres.intersection(retrieval_profile["hard_block_genres"])))
        ].copy()
    if preserved.empty:
        return candidates

    preserved = _with_empty_recall_scores(preserved)
    preserved["seed_relation"] = preserved["tmdb_id"].map(lambda value: seed_relation_for_id(int(value), retrieval_profile))
    preserved["constraint_source"] = "seed"
    return pd.concat([candidates, preserved], ignore_index=False).drop_duplicates(subset=["tmdb_id"], keep="first")


def _candidate_frame(
    lexical_hits: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    retrieval_profile: dict[str, Any],
    mode: str,
) -> pd.DataFrame:
    combined: dict[int, dict[str, float]] = {}

    if mode in {"hybrid", "lexical"}:
        for rank, hit in enumerate(lexical_hits, start=1):
            tmdb_id = int(hit["tmdb_id"])
            combined.setdefault(
                tmdb_id,
                {"bm25_score": 0.0, "semantic_score": 0.0, "retrieval_vote_score": 0.0, "bm25_rank": 0.0, "semantic_rank": 0.0},
            )
            combined[tmdb_id]["bm25_score"] = max(combined[tmdb_id]["bm25_score"], float(hit["lexical_score"]))
            lexical_rank = float(hit.get("lexical_rank", rank))
            existing_rank = combined[tmdb_id]["bm25_rank"]
            combined[tmdb_id]["bm25_rank"] = lexical_rank if existing_rank <= 0.0 else min(existing_rank, lexical_rank)
            combined[tmdb_id]["retrieval_vote_score"] += 1.0 / (RRF_K + lexical_rank)

    if mode in {"hybrid", "semantic"}:
        for rank, hit in enumerate(semantic_hits, start=1):
            tmdb_id = int(hit["tmdb_id"])
            combined.setdefault(
                tmdb_id,
                {"bm25_score": 0.0, "semantic_score": 0.0, "retrieval_vote_score": 0.0, "bm25_rank": 0.0, "semantic_rank": 0.0},
            )
            combined[tmdb_id]["semantic_score"] = max(combined[tmdb_id]["semantic_score"], float(hit["semantic_score"]))
            semantic_rank = float(hit.get("semantic_rank", rank))
            existing_rank = combined[tmdb_id]["semantic_rank"]
            combined[tmdb_id]["semantic_rank"] = semantic_rank if existing_rank <= 0.0 else min(existing_rank, semantic_rank)
            combined[tmdb_id]["retrieval_vote_score"] += 1.0 / (RRF_K + semantic_rank)

    if not combined:
        return MOVIES.iloc[0:0].copy()

    candidates = MOVIES[MOVIES["tmdb_id"].isin(list(combined))].copy()
    candidates = candidates[~candidates["tmdb_id"].isin(retrieval_profile["history_tmdb_ids"])]
    candidates = candidates[~candidates["title"].isin(retrieval_profile["history_titles"])]
    if retrieval_profile.get("exclude_seed_tmdb_ids"):
        candidates = candidates[~candidates["tmdb_id"].isin(retrieval_profile["exclude_seed_tmdb_ids"])]
    if retrieval_profile["hard_block_genres"]:
        candidates = candidates[~candidates["genres_set"].map(lambda genres: bool(genres.intersection(retrieval_profile["hard_block_genres"])))]
    candidates["bm25_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["bm25_score"])
    candidates["semantic_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["semantic_score"])
    candidates["retrieval_vote_score"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["retrieval_vote_score"])
    candidates["bm25_rank"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["bm25_rank"])
    candidates["semantic_rank"] = candidates["tmdb_id"].map(lambda value: combined[int(value)]["semantic_rank"])
    candidates["seed_relation"] = candidates["tmdb_id"].map(lambda value: seed_relation_for_id(int(value), retrieval_profile))
    candidates["constraint_source"] = ""
    max_vote = float(candidates["retrieval_vote_score"].max() or 0.0)
    if max_vote > 0.0:
        candidates["retrieval_vote_score"] = candidates["retrieval_vote_score"] / max_vote
    return candidates


def _year_mask(frame: pd.DataFrame, constraint: dict[str, Any]) -> pd.Series:
    years = pd.to_numeric(frame["year"], errors="coerce")
    return years.between(int(constraint["min_year"]), int(constraint["max_year"]), inclusive="both")


def _runtime_mask(frame: pd.DataFrame, constraint: dict[str, Any]) -> pd.Series:
    runtimes = pd.to_numeric(frame["runtime_min"], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if constraint.get("max_runtime") is not None:
        mask &= runtimes <= float(constraint["max_runtime"])
    if constraint.get("min_runtime") is not None:
        mask &= runtimes >= float(constraint["min_runtime"])
    return mask.fillna(False)


def _eligible_movies(mask: pd.Series, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    matches = MOVIES[mask].copy()
    matches = matches[~matches["tmdb_id"].isin(retrieval_profile["history_tmdb_ids"])]
    matches = matches[~matches["title"].isin(retrieval_profile["history_titles"])]
    if retrieval_profile.get("exclude_seed_tmdb_ids"):
        matches = matches[~matches["tmdb_id"].isin(retrieval_profile["exclude_seed_tmdb_ids"])]
    if retrieval_profile["hard_block_genres"]:
        matches = matches[~matches["genres_set"].map(lambda genres: bool(genres.intersection(retrieval_profile["hard_block_genres"])))]
    return matches


def _with_empty_recall_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["bm25_score"] = 0.0
    scored["semantic_score"] = 0.0
    scored["retrieval_vote_score"] = 0.0
    scored["bm25_rank"] = 0.0
    scored["semantic_rank"] = 0.0
    scored["seed_relation"] = ""
    scored["constraint_source"] = ""
    scored["source_score"] = 0.0
    return scored


def _append_constraint_matches(
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    retrieval_profile: dict[str, Any],
    limit: int = 20,
    source: str = "constraint",
) -> pd.DataFrame:
    match_ids = set(matches["tmdb_id"].astype(int)) if not matches.empty else set()
    if candidates.empty:
        missing = _with_empty_recall_scores(matches)
    else:
        existing_mask = candidates["tmdb_id"].isin(match_ids)
        if existing_mask.any():
            candidates = candidates.copy()
            candidates.loc[existing_mask & (candidates["constraint_source"].astype(str) == ""), "constraint_source"] = source
            candidates.loc[existing_mask, "source_score"] = candidates[existing_mask].apply(
                candidate_source_score,
                axis=1,
                retrieval_profile=retrieval_profile,
            )
        missing = matches[~matches["tmdb_id"].isin(set(candidates["tmdb_id"].astype(int)))].copy()
        if missing.empty:
            return candidates
        missing = _with_empty_recall_scores(missing)
    missing["constraint_source"] = source
    missing["seed_relation"] = missing["tmdb_id"].map(lambda value: seed_relation_for_id(int(value), retrieval_profile))
    missing["source_score"] = missing.apply(candidate_source_score, axis=1, retrieval_profile=retrieval_profile)
    missing = missing.sort_values(
        ["source_score", "consensus_quality_score", "effective_rating", "effective_votes"],
        ascending=False,
    ).head(limit)
    if candidates.empty:
        return missing
    return pd.concat([candidates, missing], ignore_index=False).drop_duplicates(subset=["tmdb_id"], keep="first")


def apply_year_constraint(
    candidates: pd.DataFrame,
    retrieval_profile: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    constraint = retrieval_profile.get("year_constraint")
    if not constraint:
        return candidates, retrieval_profile

    profile = dict(retrieval_profile)
    global_matches = _eligible_movies(_year_mask(MOVIES, constraint), profile)

    if global_matches.empty:
        label = constraint.get("label", "requested year range")
        profile["year_constraint_unavailable"] = True
        profile["candidate_constraint_note"] = (
            f"No movies in the local dataset match the requested era ({label}). "
            "Choose the closest available candidate and briefly acknowledge that limitation."
        )
        return candidates, profile

    candidates = _append_constraint_matches(candidates, global_matches, profile, source="year")

    constrained = candidates[_year_mask(candidates, constraint)].copy()
    if constrained.empty:
        return candidates, profile
    return constrained, profile


def apply_runtime_constraint(
    candidates: pd.DataFrame,
    retrieval_profile: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    constraint = retrieval_profile.get("runtime_constraint")
    if not constraint or candidates.empty:
        return candidates, retrieval_profile

    profile = dict(retrieval_profile)
    global_matches = _eligible_movies(_runtime_mask(MOVIES, constraint), profile)

    if global_matches.empty:
        profile["runtime_constraint_unavailable"] = True
        return candidates, profile

    candidates = _append_constraint_matches(candidates, global_matches, profile, source="runtime")

    hard_short_mask = pd.Series(True, index=candidates.index)
    if constraint.get("short_requested"):
        runtimes = pd.to_numeric(candidates["runtime_min"], errors="coerce")
        hard_short_mask = runtimes <= 180
        hard_short_filtered = candidates[hard_short_mask].copy()
        if len(hard_short_filtered) >= min(8, len(candidates)):
            candidates = hard_short_filtered

    constrained = candidates[_runtime_mask(candidates, constraint)].copy()
    if len(constrained) >= min(8, len(candidates)):
        return constrained, profile
    return candidates, profile


def preserve_person_candidates(candidates: pd.DataFrame, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    signals = set(retrieval_profile.get("named_person_signals", []))
    if not signals:
        return candidates
    person_mask = MOVIES["director_set"].map(lambda people: bool(signals.intersection(people))) | MOVIES["cast_set"].map(
        lambda people: bool(signals.intersection(people))
    )
    matches = _eligible_movies(person_mask, retrieval_profile)
    if matches.empty:
        return candidates
    return _append_constraint_matches(candidates, matches, retrieval_profile, limit=20, source="person")


def preserve_quality_candidates(candidates: pd.DataFrame, retrieval_profile: dict[str, Any]) -> pd.DataFrame:
    if not needs_quality_backup(retrieval_profile):
        return candidates
    rating_floor = 8.0 if retrieval_profile.get("all_time_quality_preference") else 7.4 if retrieval_profile.get("quality_preference") else 7.2
    quality_mask = (
        (pd.to_numeric(MOVIES["effective_rating"], errors="coerce").fillna(0.0) >= rating_floor)
        & (pd.to_numeric(MOVIES["effective_votes"], errors="coerce").fillna(0.0) >= 1000)
    )
    matches = _eligible_movies(quality_mask, retrieval_profile)
    genre_targets = set(retrieval_profile.get("explicit_genre_targets", set()))
    if retrieval_profile.get("similarity_request") and not genre_targets:
        genre_targets = set(retrieval_profile.get("seed_genres", set()))
    if retrieval_profile.get("affinity_genre_targets") and not genre_targets:
        genre_targets = set(retrieval_profile.get("affinity_genre_targets", set()))
    if genre_targets and not matches.empty:
        genre_matches = matches[matches["genres_set"].map(lambda genres: bool(genre_targets.intersection(genres)))].copy()
        if len(genre_matches) >= min(5, len(matches)):
            matches = genre_matches
    if matches.empty:
        return candidates
    return _append_constraint_matches(candidates, matches, retrieval_profile, limit=20, source="quality")


def local_fallback_candidates(preferences: str, history: tuple[tuple[int | None, str], ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    retrieval_profile = build_retrieval_profile(preferences, history)
    candidates = _with_empty_recall_scores(MOVIES)
    candidates["seed_relation"] = candidates["tmdb_id"].map(lambda value: seed_relation_for_id(int(value), retrieval_profile))
    candidates["source_score"] = candidates.apply(candidate_source_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["source_score", "consensus_quality_score", "effective_rating", "effective_votes"], ascending=False)
    return candidates.head(MERGED_POOL_SIZE).copy(), retrieval_profile


def build_candidate_pool(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    mode: str = "auto",
    retrieval_profile_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    retrieval_profile = build_retrieval_profile(preferences, history)
    retrieval_profile = merge_intent_override(retrieval_profile, retrieval_profile_override)
    lexical_query_text = build_query_text(preferences, retrieval_profile)
    exclude_ids = set(retrieval_profile["history_tmdb_ids"])

    lexical_available = True
    lexical_hits = search_weighted_bm25(lexical_query_text, exclude_ids=exclude_ids, limit=LEXICAL_LIMIT)
    semantic_hits, semantic_available, semantic_elapsed_s = search_optional_semantic(
        retrieval_profile,
        exclude_ids=exclude_ids,
        lexical_hits=lexical_hits,
        allow_semantic=mode in {"auto", "hybrid", "semantic"},
    )
    resolved_mode = "hybrid" if semantic_hits else "lexical" if lexical_hits else "local_fallback"

    candidates = _candidate_frame(lexical_hits, semantic_hits, retrieval_profile, resolved_mode)
    candidates = preserve_seed_candidates(candidates, retrieval_profile)
    candidates = preserve_person_candidates(candidates, retrieval_profile)
    candidates = preserve_quality_candidates(candidates, retrieval_profile)
    candidates = filter_semantic_only_candidates(candidates, retrieval_profile)
    candidates = apply_quality_floor(candidates, retrieval_profile)
    candidates, retrieval_profile = apply_year_constraint(candidates, retrieval_profile)
    candidates, retrieval_profile = apply_runtime_constraint(candidates, retrieval_profile)
    if candidates.empty:
        fallback, fallback_profile = local_fallback_candidates(preferences, history)
        fallback, fallback_profile = apply_year_constraint(fallback, fallback_profile)
        fallback, fallback_profile = apply_runtime_constraint(fallback, fallback_profile)
        fallback_profile = dict(fallback_profile)
        fallback_profile["lexical_hit_count"] = 0
        fallback_profile["semantic_hit_count"] = 0
        fallback_profile["semantic_active"] = False
        fallback_profile["lexical_available"] = True
        fallback_profile["semantic_available"] = False
        fallback_profile["semantic_elapsed_s"] = 0.0
        fallback_profile["semantic_enabled"] = ENABLE_HF_SEMANTIC_RETRIEVAL
        return fallback, fallback_profile, "local_fallback"

    candidates["seed_relation"] = candidates["tmdb_id"].map(lambda value: seed_relation_for_id(int(value), retrieval_profile))
    candidates["source_score"] = candidates.apply(candidate_source_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["source_score", "retrieval_vote_score", "consensus_quality_score", "effective_rating", "effective_votes"], ascending=False)
    retrieval_profile = dict(retrieval_profile)
    retrieval_profile["lexical_hit_count"] = len(lexical_hits)
    retrieval_profile["semantic_hit_count"] = len(semantic_hits)
    retrieval_profile["semantic_active"] = bool(semantic_hits)
    retrieval_profile["lexical_available"] = bool(lexical_available)
    retrieval_profile["semantic_available"] = bool(semantic_available)
    retrieval_profile["semantic_elapsed_s"] = round(float(semantic_elapsed_s), 3)
    retrieval_profile["semantic_enabled"] = ENABLE_HF_SEMANTIC_RETRIEVAL
    retrieval_profile["preserve_bm25_tmdb_ids"] = [
        int(hit["tmdb_id"])
        for hit in lexical_hits[:3]
        if retrieval_profile.get("has_named_person_signal")
    ]
    return candidates.head(MERGED_POOL_SIZE).copy(), retrieval_profile, resolved_mode


def build_shortlist(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
    mode: str = "auto",
    retrieval_profile_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates, retrieval_profile, resolved_mode = build_candidate_pool(preferences, history, mode=mode, retrieval_profile_override=retrieval_profile_override)
    prompt_profile = build_prompt_profile(preferences, retrieval_profile)

    reranked = candidates.head(SECOND_STAGE_POOL_SIZE).copy()
    preserve_bm25_ids = retrieval_profile.get("preserve_bm25_tmdb_ids", [])
    if preserve_bm25_ids:
        preserved = candidates[candidates["tmdb_id"].isin(preserve_bm25_ids)]
        reranked = pd.concat([reranked, preserved], ignore_index=False).drop_duplicates(subset=["tmdb_id"], keep="first")
    reranked = reranked.sort_values(["source_score", "retrieval_vote_score", "consensus_quality_score", "effective_rating", "effective_votes"], ascending=False)
    reranked = diversify_candidates(reranked, SHORTLIST_SIZE)

    retrieval_profile = dict(retrieval_profile)
    retrieval_profile["retrieval_mode"] = resolved_mode

    shortlist = []
    for row in reranked.head(SHORTLIST_SIZE).itertuples():
        source_row = reranked.loc[row.Index]
        evidence = candidate_evidence(source_row, retrieval_profile)
        shortlist.append(
            {
                "tmdb_id": int(row.tmdb_id),
                "title": row.title,
                "score": round(float(getattr(row, "source_score", 0.0)), 3),
                "source_score": round(float(getattr(row, "source_score", 0.0)), 3),
                "semantic_score": round(float(getattr(row, "semantic_score", 0.0)), 3),
                "bm25_score": round(float(getattr(row, "bm25_score", 0.0)), 3),
                "bm25_rank": int(float(getattr(row, "bm25_rank", 0.0))),
                "semantic_rank": int(float(getattr(row, "semantic_rank", 0.0))),
                "seed_relation": str(getattr(row, "seed_relation", "") or ""),
                "constraint_source": str(getattr(row, "constraint_source", "") or ""),
                "retrieval_vote_score": round(float(getattr(row, "retrieval_vote_score", 0.0)), 3),
                "vote_average": round(float(row.vote_average), 3),
                "vote_count": int(row.vote_count),
                "effective_rating": round(float(getattr(row, "effective_rating", row.vote_average)), 3),
                "effective_votes": int(float(getattr(row, "effective_votes", row.vote_count))),
                "consensus_quality_score": round(float(getattr(row, "consensus_quality_score", 0.0)), 3),
                **evidence,
            }
        )
    retrieval_profile["confidence_bundle"] = build_confidence_bundle(shortlist, retrieval_profile, prompt_profile)
    return shortlist, prompt_profile, retrieval_profile
