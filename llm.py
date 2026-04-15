"""
Movie recommendation logic.

Pipeline:
1. Broad heuristic prefilter over the full dataset
2. Heuristic-first reranking (PhraseRules + BM25 heavy-lifting)
3. Single LLM call returns both taste analysis and final selection
4. Deterministic fallback if the LLM step fails
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any

import ollama
import pandas as pd
from tmdb_client import enrich_movie_rows, fetch_tmdb_history_row
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

LLM_CLIENT_TIMEOUT_SECONDS = 18.0
PREFILTER_SIZE = 80
SHORTLIST_SIZE = 5
logger = logging.getLogger(__name__)

MODEL = "gemini-3-flash-preview"
TMDB_ENRICH_TOP_N = 1
SELECTION_LLM_TIMEOUT_SECONDS = 10.0
LLM_STAGE_BUDGET_SECONDS = 18.0
SELECTION_RETRY_TIMEOUT_SECONDS = 4.0
SELECTION_RETRY_RESERVED_SECONDS = 2.0
FAST_FAIL_RETRY_TIMEOUT_SECONDS = 8.0
FAST_FAIL_PRIMARY_SECONDS = 1.0
# Keep primary bounded so retry gets a real chance when the model stalls.
PRIMARY_SELECTION_TIMEOUT_SECONDS = 8.0
MIN_RETRY_WINDOW_SECONDS = 4.0
HISTORY_INFLUENCE = 0.15
HISTORY_NOVELTY_PENALTY = 1.2
BM25_K1 = 1.2
BM25_B = 0.75

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_MOVIES = pd.read_csv(DATA_PATH).fillna("")

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
    "of",
    "on",
    "or",
    "something",
    "stories",
    "story",
    "that",
    "the",
    "to",
    "want",
    "watch",
    "with",
}

# ── Phrase & pattern rules ────────────────────────────────────────────────────
# Each rule carries a compiled regex and the hint tokens it contributes.
# Positive rules add to the candidate score; negative rules subtract.
# Using regex instead of plain substring matching lets us:
#   - handle morphological variants ("feel-good", "feel good", "feelgood")
#   - scope negation windows ("not another marvel" vs "loves marvel")
#   - avoid false substring hits ("action" inside "reaction")
@dataclass
class PhraseRule:
    pattern: re.Pattern
    hints: list[str]

@dataclass
class NegativeRule:
    pattern: re.Pattern
    hints: list[str]          # tokens to penalise in candidate scoring

PHRASE_RULES: list[PhraseRule] = [
    PhraseRule(
        pattern=re.compile(r"\b(superhero|super.?hero|comic.?book|cape|vigilante|masked)\b"),
        hints=["superhero", "hero", "powers", "vigilante", "marvel", "dc"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(buddy.?cop|partner|detective|police.?comedy|cop.?comedy)\b"),
        hints=["buddy cop", "cop", "detective", "police", "partners"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(rom.?com|romantic.?comedy|love.?story|romance)\b"),
        hints=["romance", "comedy", "romantic", "chemistry", "funny"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(feel.?good|uplifting|heartwarming|wholesome|life.?affirming)\b"),
        hints=["uplifting", "heartwarming", "friendship", "hopeful", "fun"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(coming.?of.?age|grow.?up|teen|youth|adolescen)\b"),
        hints=["coming of age", "teen", "growing up", "youth"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(dark\s+fantasy|grimdark|dark.?fairy)\b"),
        hints=["dark", "ominous", "fantasy", "mature", "adult"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(mind.?bend|mindf\w+|twist|psychological|cerebral|complex.?plot|non.?linear)\b"),
        hints=["psychological", "twist", "cerebral", "ambitious", "complex"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(sci.?fi|science.?fiction|space\s+opera|dystop)\b"),
        hints=["science fiction", "space", "dystopia", "futuristic", "sci-fi"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(slow.?burn|atmospheric|arthouse|art.?film|indie)\b"),
        hints=["atmospheric", "slow burn", "indie", "arthouse"],
    ),
    PhraseRule(
        pattern=re.compile(r"\b(action.?pack|high.?octane|adrenaline|explosive|thrill)\b"),
        hints=["action", "thriller", "chase", "explosion", "fight"],
    ),
]

NEGATIVE_RULES: list[NegativeRule] = [
    # Marvel / superhero fatigue — catches "tired of marvel", "no more capes", "superhero fatigue"
    NegativeRule(
        pattern=re.compile(
            r"(\b(no more|tired of|not another|sick of|done with|enough)\b.{0,40}"
            r"\b(marvel|superhero|hero|avengers|cape|comic)\b)"
            r"|\b(superhero.?fatigue|no.?capes?)\b",
            re.IGNORECASE,
        ),
        hints=["superhero", "marvel", "avengers", "dc", "hero"],
    ),
    # Generic action avoidance — "not an action movie", "no explosions", "avoid war films"
    NegativeRule(
        pattern=re.compile(
            r"(\b(not?|avoid|no)\b.{0,30}\b(action|explosion|war|combat|fight)\b)"
            r"|\bno.?action\b",
            re.IGNORECASE,
        ),
        hints=["action", "explosion", "battle", "war", "combat"],
    ),
    # Kid / family avoidance — "not kid-friendly", "no animations", "adults only"
    NegativeRule(
        pattern=re.compile(
            r"\b(not?.?kid.?friendly|no.?kids?|adult.?only|not?.?for.?children|no.?animat|not?.?family)\b",
            re.IGNORECASE,
        ),
        hints=["family", "animation", "children", "kid", "animated"],
    ),
    # Horror avoidance
    NegativeRule(
        pattern=re.compile(
            r"\b(not?.?\bhorror\b|no.?horror|avoid.?horror|no.?gore|not?.?scary)\b",
            re.IGNORECASE,
        ),
        hints=["horror", "gore", "slasher", "scary"],
    ),
]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _tokenize(value: Any) -> set[str]:
    return {token for token in TOKEN_RE.findall(_normalize_text(value)) if token not in STOP_WORDS}


def _split_csvish(value: Any) -> set[str]:
    items = set()
    for part in str(value or "").split(","):
        cleaned = part.strip().lower()
        if cleaned:
            items.add(cleaned)
    return items


def _title_root(title: Any) -> str:
    text = _normalize_text(title)
    if not text:
        return ""
    for delimiter in (" - ", ":"):
        if delimiter in text:
            text = text.split(delimiter, 1)[0]
    text = re.sub(r"\b(part|chapter|episode|vol|volume)\b.*$", "", text).strip()
    text = re.sub(r"\b\d+\b$", "", text).strip()
    return text


def _prepare_movies(df: pd.DataFrame) -> pd.DataFrame:
    movies = df.copy()
    movies["normalized_title"] = movies["title"].map(_normalize_text)
    movies["title_root"] = movies["title"].map(_title_root)
    movies["genres_set"] = movies["genres"].map(_split_csvish)
    movies["keywords_set"] = movies["keywords"].map(_split_csvish)
    movies["genres_tokens"] = movies["genres"].map(_tokenize)
    movies["keywords_tokens"] = movies["keywords"].map(_tokenize)
    movies["cast_set"] = movies["top_cast"].map(_split_csvish)
    movies["director_set"] = movies["director"].map(_split_csvish)
    movies["overview_tokens"] = movies["overview"].map(_tokenize)
    movies["tagline_tokens"] = movies["tagline"].map(_tokenize)
    movies["title_tokens"] = movies["title"].map(_tokenize)
    movies["search_blob"] = movies[TEXT_COLUMNS].agg(" ".join, axis=1).map(_normalize_text)
    movies["search_blob_tokens"] = movies["search_blob"].map(_tokenize)
    return movies


MOVIES = _prepare_movies(TOP_MOVIES)
MOVIES_BY_TITLE = MOVIES.groupby("normalized_title", sort=False)
MOVIES_BY_TMDB_ID = MOVIES.set_index("tmdb_id", drop=False)


def _build_token_stats(movies: pd.DataFrame) -> tuple[Counter[str], float]:
    doc_freq: Counter[str] = Counter()
    total_length = 0
    for tokens in movies["search_blob_tokens"]:
        unique_tokens = set(tokens)
        doc_freq.update(unique_tokens)
        total_length += len(unique_tokens)
    avg_doc_len = total_length / max(len(movies), 1)
    return doc_freq, avg_doc_len


TOKEN_DOC_FREQ, AVG_DOC_LEN = _build_token_stats(MOVIES)


@lru_cache(maxsize=8)
def _get_client(timeout_seconds: float = LLM_CLIENT_TIMEOUT_SECONDS) -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=timeout_seconds,
    )


def _normalize_history_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        tmdb_id = item.get("tmdb_id")
        name = item.get("name", "")
    else:
        tmdb_id = None
        name = item

    normalized_name = " ".join(str(name or "").split())
    if tmdb_id is None and not normalized_name:
        return None

    normalized_item: dict[str, Any] = {"name": normalized_name}
    if tmdb_id is not None:
        try:
            normalized_item["tmdb_id"] = int(tmdb_id)
        except (TypeError, ValueError):
            pass
    return normalized_item


def _normalize_history(history: list[Any]) -> tuple[tuple[int | None, str], ...]:
    normalized = []
    for item in history:
        normalized_item = _normalize_history_item(item)
        if normalized_item is None:
            continue
        normalized.append((normalized_item.get("tmdb_id"), normalized_item["name"]))
    return tuple(sorted(set(normalized), key=lambda item: (item[0] is None, item[0], item[1].lower())))


def _history_rows(history: tuple[tuple[int | None, str], ...]) -> pd.DataFrame:
    history_t0 = time.perf_counter()
    rows = []
    seen_ids: set[int] = set()
    api_needed = []  # (tmdb_id, name) pairs that missed local lookup

    # Resolve local paths first (fast, no change)
    for tmdb_id, name in history:
        if tmdb_id is not None and tmdb_id in MOVIES_BY_TMDB_ID.index:
            row = MOVIES_BY_TMDB_ID.loc[tmdb_id]
            if int(row.tmdb_id) not in seen_ids:
                rows.append(row.to_frame().T)
                seen_ids.add(int(row.tmdb_id))
            continue
        title = _normalize_text(name)
        if title in MOVIES_BY_TITLE.groups:
            group = MOVIES_BY_TITLE.get_group(title)
            unseen = group[~group["tmdb_id"].isin(seen_ids)]
            if not unseen.empty:
                rows.append(unseen)
                seen_ids.update(unseen["tmdb_id"].astype(int).tolist())
            continue
        api_needed.append((tmdb_id, name))

    MAX_TMDB_FALLBACKS = 1
    api_needed = api_needed[:MAX_TMDB_FALLBACKS]
    # Fan out all TMDB calls in parallel
    if api_needed:
        tmdb_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(api_needed)) as executor:
            futures = {
                executor.submit(fetch_tmdb_history_row, tmdb_id, name): (tmdb_id, name)
                for tmdb_id, name in api_needed
            }
            for future in as_completed(futures):
                pseudo_row = future.result()
                if pseudo_row is not None:
                    resolved_id = int(pseudo_row["tmdb_id"])
                    if resolved_id not in seen_ids:
                        rows.append(pd.DataFrame([pseudo_row]))
                        seen_ids.add(resolved_id)
        tmdb_elapsed = time.perf_counter() - tmdb_t0

    if not rows:
        return MOVIES.iloc[0:0].copy()
    history_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tmdb_id"])
    return history_df


def _extract_phrase_hints(preferences: str) -> tuple[set[str], set[str]]:
    """
    Run all PHRASE_RULES and NEGATIVE_RULES against the preference string.
    Returns (positive_hint_tokens, negative_hint_tokens).
    Regex matching handles variants and negation scoping that plain substring
    matching misses ("superhero fatigue", "no more capes", "not kid-friendly").
    """
    positive: set[str] = set()
    negative: set[str] = set()

    for rule in PHRASE_RULES:
        if rule.pattern.search(preferences):
            positive.update(rule.hints)

    for rule in NEGATIVE_RULES:
        if rule.pattern.search(preferences):
            negative.update(rule.hints)
            # Remove any positive signals that contradict this negative rule
            positive -= set(rule.hints)

    return positive, negative


def _bm25_like_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    norm = BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / max(AVG_DOC_LEN, 1e-9)))
    score = 0.0

    for token in query_tokens:
        if token not in doc_tokens:
            continue
        df = TOKEN_DOC_FREQ.get(token, 0)
        idf = max(0.0, math.log((len(MOVIES) - df + 0.5) / (df + 0.5) + 1.0))
        tf = 1.0
        score += idf * ((tf * (BM25_K1 + 1)) / (tf + norm))

    return score


def _matches_explicit_preference(row: pd.Series, explicit_tokens: set[str]) -> bool:
    if not explicit_tokens:
        return False
    explicit_overlap = explicit_tokens.intersection(row["title_tokens"])
    explicit_overlap |= explicit_tokens.intersection(row["genres_tokens"])
    explicit_overlap |= explicit_tokens.intersection(row["keywords_tokens"])
    explicit_overlap |= explicit_tokens.intersection(row["search_blob_tokens"])
    return bool(explicit_overlap)


def _derive_history_signals(history_df: pd.DataFrame) -> dict[str, set[str]]:
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

    # Use recurring history patterns rather than every one-off signal.
    liked_genres = {name for name, count in genre_counts.most_common(6) if count >= 2}
    if not liked_genres:
        liked_genres = {name for name, _ in genre_counts.most_common(3)}

    liked_directors = {name for name, count in director_counts.most_common(4) if count >= 2}
    liked_cast = {name for name, count in cast_counts.most_common(6) if count >= 2}

    return {
        "liked_genres": liked_genres,
        "liked_cast": liked_cast,
        "liked_directors": liked_directors,
        "watched_roots": watched_roots,
    }


def _history_prompt_text(history_count: int) -> str:
    if history_count <= 0:
        return "none"
    return f"known_watch_history_count={history_count}; use history primarily to avoid re-recommending already watched titles"


def _prefilter_score(row: pd.Series, retrieval_profile: dict[str, Any]) -> float:
    if int(row["tmdb_id"]) in retrieval_profile["history_tmdb_ids"]:
        return -10_000.0
    if row["normalized_title"] in retrieval_profile["history_titles"]:
        return -10_000.0

    score = 0.0
    positive_tokens = retrieval_profile["retrieval_query_tokens"]
    negative_tokens = retrieval_profile["phrase_negative_tokens"]

    score += 2.5 * _bm25_like_score(positive_tokens, row["search_blob_tokens"])

    token_matches = positive_tokens.intersection(row["title_tokens"])
    token_matches |= positive_tokens.intersection(row["genres_tokens"])
    token_matches |= positive_tokens.intersection(row["keywords_tokens"])
    score += 1.0 * len(token_matches)

    genre_matches = retrieval_profile["liked_genres"].intersection(row["genres_set"])
    score += (2.0 * HISTORY_INFLUENCE) * len(genre_matches)

    # Favor novelty: penalize movies that look too similar to recurring history patterns.
    history_director_overlap = len(retrieval_profile["liked_directors"].intersection(row["director_set"]))
    history_cast_overlap = len(retrieval_profile["liked_cast"].intersection(row["cast_set"]))
    similarity_hits = len(genre_matches) + history_director_overlap + history_cast_overlap
    explicit_tokens = retrieval_profile["raw_preference_tokens"] | retrieval_profile["phrase_positive_tokens"]
    if similarity_hits >= 3 and not _matches_explicit_preference(row, explicit_tokens):
        score -= HISTORY_NOVELTY_PENALTY * float(similarity_hits - 2)

    if row["title_root"] and row["title_root"] in retrieval_profile["watched_roots"]:
        score -= 8.0

    if negative_tokens:
        negative_hits = negative_tokens.intersection(row["search_blob_tokens"])
        score -= 3.0 * len(negative_hits)

    score += min(float(row["vote_average"]) or 0.0, 10.0) * 0.2
    score += min(float(row["vote_count"]) or 0.0, 5000.0) / 8000.0
    return score


def _build_prompt_profile(preferences: str, retrieval_profile: dict[str, Any]) -> dict[str, Any]:
    preference_tokens = _tokenize(preferences)
    phrase_positive_tokens, phrase_negative_tokens = _extract_phrase_hints(preferences)
    preferred_genres: list[str] = []
    avoid = sorted(phrase_negative_tokens)[:8]
    themes = sorted((preference_tokens | phrase_positive_tokens) - phrase_negative_tokens)[:10]

    return {
        "target_genres": preferred_genres,
        "preferred_tones": [],
        "preferred_themes": themes,
        "avoid": avoid,
        "history_signals": {
            "liked_genres": sorted(retrieval_profile["liked_genres"])[:6],
            "liked_directors": sorted(retrieval_profile["liked_directors"])[:4],
            "liked_cast": sorted(retrieval_profile["liked_cast"])[:6],
        },
    }


def _extract_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ValueError("Could not parse JSON object from model response")


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "status code: 429" in message
        or "session usage limit" in message
        or "rate limit" in message
        or "too many requests" in message
    )


def _build_retrieval_profile(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    history_df = _history_rows(history)
    raw_preference_tokens = _tokenize(preferences)
    phrase_positive_tokens, phrase_negative_tokens = _extract_phrase_hints(preferences)
    history_signals = _derive_history_signals(history_df)
    history_signal_tokens: set[str] = set()
    for key in ("liked_genres", "liked_cast", "liked_directors"):
        for value in history_signals[key]:
            history_signal_tokens.update(_tokenize(value))

    preference_signal_tokens = raw_preference_tokens | phrase_positive_tokens
    retrieval_query_tokens = set(preference_signal_tokens)
    if len(retrieval_query_tokens) < 4:
        retrieval_query_tokens |= set(sorted(history_signal_tokens)[:6])

    return {
        "history_df": history_df,
        "history_count": len(history),
        "history_exclusion_text": _history_prompt_text(len(history)),
        "history_titles": {_normalize_text(name) for _, name in history if _normalize_text(name)},
        "history_tmdb_ids": {tmdb_id for tmdb_id, _ in history if tmdb_id is not None},
        "raw_preference_tokens": raw_preference_tokens,
        "phrase_positive_tokens": phrase_positive_tokens,
        "phrase_negative_tokens": phrase_negative_tokens,
        "retrieval_query_tokens": retrieval_query_tokens,
        **history_signals,
    }


def _prefilter_candidates(preferences: str, history: tuple[tuple[int | None, str], ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    retrieval_profile = _build_retrieval_profile(preferences, history)
    candidates = MOVIES.copy()
    candidates["prefilter_score"] = candidates.apply(_prefilter_score, axis=1, retrieval_profile=retrieval_profile)
    candidates = candidates.sort_values(["prefilter_score", "vote_average", "vote_count"], ascending=False)

    if TMDB_ENRICH_TOP_N > 0:
        top_n = min(TMDB_ENRICH_TOP_N, len(candidates))
        if top_n > 0:
            enriched_top = enrich_movie_rows(candidates.head(top_n).copy(), top_n=top_n)
            candidates.loc[enriched_top.index, enriched_top.columns] = enriched_top
            candidates.loc[enriched_top.index, "prefilter_score"] = candidates.loc[enriched_top.index].apply(
                _prefilter_score, axis=1, retrieval_profile=retrieval_profile
            )
            candidates = candidates.sort_values(["prefilter_score", "vote_average", "vote_count"], ascending=False)

    return candidates.head(PREFILTER_SIZE).copy(), retrieval_profile


def _build_shortlist(
    preferences: str,
    history: tuple[tuple[int | None, str], ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prefiltered, retrieval_profile = _prefilter_candidates(preferences, history)
    # Single-call architecture: rely on heuristic ranking before LLM selection.
    reranked = prefiltered.copy()
    reranked["score"] = reranked["prefilter_score"]
    reranked = reranked.sort_values(["score", "vote_average", "vote_count"], ascending=False)

    # Use deterministic profile as prompt context only; no separate profile LLM call.
    prompt_profile = _build_prompt_profile(preferences, retrieval_profile)

    # FIX: the old loop scanned head(40) and re-checked history exclusion.
    # History items already score -10_000 so they're at the tail, not head(40).
    # Just take the top SHORTLIST_SIZE directly.
    shortlist = []
    for row in reranked.head(SHORTLIST_SIZE).itertuples():
        shortlist.append({
            "tmdb_id": int(row.tmdb_id),
            "title": row.title,
            "year": int(row.year),
            "genres": row.genres,
            "overview": str(row.overview)[:260],
            "keywords": ", ".join(row.keywords_set),
            "director": row.director,
            "top_cast": ", ".join(sorted(row.cast_set)[:4]),
            "score": round(float(row.score), 2),
        })

    return shortlist, prompt_profile, retrieval_profile


def _build_selection_prompt(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
) -> str:
    avoid_text = ", ".join(prompt_profile["avoid"]) if prompt_profile["avoid"] else "none"
    themes_text = ", ".join(prompt_profile["preferred_themes"][:5]) if prompt_profile["preferred_themes"] else "none"

    def _fmt(m: dict[str, Any]) -> str:
        # Keep director + cast (same-actor/director matching).
        # Drop score (opaque to model), trim overview to 80 chars.
        overview = (m["overview"] or "")[:80].rstrip()
        return (
            f'{m["tmdb_id"]}: "{m["title"]}" ({m["year"]}) '
            f'[{m["genres"]}] '
            f'dir:{m["director"] or "?"} '
            f'cast:{m["top_cast"] or "?"} — {overview}'
        )

    shortlist_text = "\n".join(_fmt(m) for m in shortlist)

    # Schema-only few-shot: shows output shape without a verbose writing sample.
    few_shot = (
        'Example: user wants "rom-com not action"\n'
        'Candidates: 455207:"Crazy Rich Asians"[Romance,Comedy] dir:Jon M. Chu cast:Constance Wu...\n'
        'Output: {"tmdb_id":455207,"description":"Crazy Rich Asians delivers the rom-com energy...'
        ' No explosions, just wedding chaos and swoony romance."}'
    )

    return f"""You are a movie recommendation assistant.

{few_shot}

User wants: {preferences}
Themes: {themes_text}
Avoid: {avoid_text}
Already watched: {history_exclusion_text}

Candidates (prefer same director/cast as watch history when relevant):
{shortlist_text}

Reply with raw JSON only, no markdown:
{{"tmdb_id": <id from candidates>, "description": "<2-3 sentences, strictly under 500 chars>"}}"""


def _build_micro_prompt(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
) -> str:
    """Minimal prompt for ultra-fast LLM selection when time is critical."""
    titles = ", ".join(f'{m["title"]} ({m["year"]})' for m in shortlist[:3])
    avoid = ", ".join(prompt_profile["avoid"]) if prompt_profile["avoid"] else "none"
    return f"""Pick one movie from: {titles}. User wants: {preferences}. History: {history_exclusion_text}. Avoid: {avoid}.
Decision order: 1) core mood 2) filter by avoid 3) choose winner.
Return JSON: {{"selection": {{"tmdb_id": <id>, "description": "<under 500 chars>"}}}}.
Do not use markdown or code fences. Output raw JSON only."""


def _fallback_description(movie: dict[str, Any], preferences: str, prompt_profile: dict[str, Any]) -> str:
    hook = str(movie["overview"]).strip()
    if hook:
        hook = hook[:220]
        last_space = hook.rfind(" ")
        if len(hook) == 220 and last_space > 120:
            hook = hook[:last_space]
        hook = hook.rstrip(".") + "."
    else:
        hook = f'{movie["title"]} is a strong fit.'

    reasons = []
    if prompt_profile["target_genres"]:
        reasons.append("genres like " + ", ".join(prompt_profile["target_genres"][:2]))
    if prompt_profile["preferred_themes"]:
        reasons.append("themes like " + ", ".join(prompt_profile["preferred_themes"][:2]))
    if not reasons and movie["genres"]:
        reasons.append(movie["genres"])

    reason_text = "If you're in the mood for " + " and ".join(reasons) + ", this is a strong pick."
    description = f"{hook} {reason_text}"
    return description[:500]


def _choose_with_llm(
    preferences: str,
    shortlist: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    history_exclusion_text: str,
    timeout_seconds: float,
    stage_deadline: float,
) -> dict[str, Any]:
    def _attempt(attempt_name: str, call_timeout_seconds: float) -> dict[str, Any]:
        prompt = _build_selection_prompt(
            preferences,
            shortlist,
            prompt_profile,
            history_exclusion_text,
        )
        t0 = time.perf_counter()
        logger.warning("TIMING prompt_chars=%d", len(prompt))
        response = _get_client(call_timeout_seconds).chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        elapsed = time.perf_counter() - t0
        logger.warning("TIMING final_selection_seconds=%.3f attempt=%s", elapsed, attempt_name)
        payload = _extract_json_object(response.message.content)
        valid_ids = {movie["tmdb_id"] for movie in shortlist}

        selection_payload = payload.get("selection") if isinstance(payload.get("selection"), dict) else payload

        tmdb_id = int(selection_payload.get("tmdb_id", -1))
        if tmdb_id not in valid_ids:
            raise ValueError(f"Model selected tmdb_id {tmdb_id}, which is outside the shortlist")
        description = str(selection_payload.get("description", ""))[:500]
        return {
            "tmdb_id": tmdb_id,
            "description": description,
        }

    remaining_total = max(0.0, stage_deadline - time.perf_counter())
    first_timeout = min(
        timeout_seconds,
        PRIMARY_SELECTION_TIMEOUT_SECONDS,
        max(1.0, remaining_total - MIN_RETRY_WINDOW_SECONDS),
    )
    first_attempt_t0 = time.perf_counter()
    try:
        return _attempt(attempt_name="primary", call_timeout_seconds=first_timeout)
    except Exception as exc:
        elapsed = time.perf_counter() - first_attempt_t0
        logger.warning(
            "TIMING final_selection_failed_seconds=%.3f attempt=primary error_type=%s",
            elapsed,
            type(exc).__name__,
        )
        if _is_quota_or_rate_limit_error(exc):
            logger.warning("Skipping retry: LLM quota/rate-limit detected on primary attempt")
            raise
        remaining = max(0.0, stage_deadline - time.perf_counter())
        # If the primary call failed very quickly (often server-side response errors),
        # give the retry more headroom than the normal 5s cap.
        desired_retry_timeout = (
            FAST_FAIL_RETRY_TIMEOUT_SECONDS
            if elapsed <= FAST_FAIL_PRIMARY_SECONDS
            else SELECTION_RETRY_TIMEOUT_SECONDS
        )
        retry_timeout = min(desired_retry_timeout, remaining)
        if retry_timeout < 1.0:
            raise
        logger.warning(
            "Retrying final selection with micro prompt timeout=%.3f fast_fail=%s",
            retry_timeout,
            elapsed <= FAST_FAIL_PRIMARY_SECONDS,
        )
        retry_t0 = time.perf_counter()
        try:
            micro_prompt = _build_micro_prompt(
                preferences,
                shortlist,
                prompt_profile,
                history_exclusion_text,
            )

            response = _get_client(retry_timeout).chat(
                model=MODEL,
                messages=[{"role": "user", "content": micro_prompt}],
                format="json",
            )
            retry_elapsed = time.perf_counter() - retry_t0
            logger.warning("TIMING final_selection_seconds=%.3f attempt=retry_micro", retry_elapsed)
            payload = _extract_json_object(response.message.content)
            valid_ids = {movie["tmdb_id"] for movie in shortlist}
            selection_payload = payload.get("selection") if isinstance(payload.get("selection"), dict) else payload
            tmdb_id = int(selection_payload.get("tmdb_id", -1))
            if tmdb_id not in valid_ids:
                raise ValueError(f"Model selected tmdb_id {tmdb_id}, which is outside the shortlist")
            description = str(selection_payload.get("description", ""))[:500]
            return {
                "tmdb_id": tmdb_id,
                "description": description,
            }
        except Exception as retry_exc:
            retry_elapsed = time.perf_counter() - retry_t0
            logger.warning(
                "TIMING final_selection_failed_seconds=%.3f attempt=retry_micro error_type=%s",
                retry_elapsed,
                type(retry_exc).__name__,
            )
            raise


@lru_cache(maxsize=256)
def _get_recommendation_cached(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    llm_stage_t0 = time.perf_counter()
    stage_deadline = llm_stage_t0 + LLM_STAGE_BUDGET_SECONDS
    shortlist, prompt_profile, retrieval_profile = _build_shortlist(
        preferences,
        history,
    )
    if not shortlist:
        raise ValueError("No candidate movies available")

    selection_timeout_seconds = min(
        SELECTION_LLM_TIMEOUT_SECONDS,
        max(1.0, stage_deadline - time.perf_counter() - SELECTION_RETRY_RESERVED_SECONDS),
    )

    try:
        result = _choose_with_llm(
            preferences,
            shortlist,
            prompt_profile,
            retrieval_profile["history_exclusion_text"],
            timeout_seconds=selection_timeout_seconds,
            stage_deadline=stage_deadline,
        )
        result["used_llm"] = True
        return result
    except Exception as exc:
        logger.warning(
            "Falling back to heuristic recommendation: preferences_len=%d history_count=%d shortlist_count=%d error=%s",
            len(preferences),
            len(history),
            len(shortlist),
            exc,
        )
        best = shortlist[0]
        return {
            "tmdb_id": best["tmdb_id"],
            "description": _fallback_description(best, preferences, prompt_profile),
            "used_llm": False,
        }


def get_recommendation(preferences: str, history: list[Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    normalized_preferences = " ".join(preferences.split())
    normalized_history = _normalize_history(history)
    result = _get_recommendation_cached(normalized_preferences, normalized_history)
    total_elapsed = time.perf_counter() - t0
    logger.warning("TIMING total_recommendation_seconds=%.3f", total_elapsed)
    # Log LLM usage for benchmarking
    used_llm = result.get("used_llm", True)
    logger.warning("Recommendation completed: used_llm=%s tmdb_id=%d", used_llm, result["tmdb_id"])
    return result


if __name__ == "__main__":
    print("Movie recommender – type your preferences and press Enter.")
    print("For watch history, enter comma-separated movie titles (or leave blank).")
    print()

    preferences = input("Preferences: ").strip()
    history_raw = input("Watch history (optional): ").strip()
    history = [t.strip() for t in history_raw.split(",")] if history_raw else []

    print("\nThinking...\n")
    result = get_recommendation(preferences, history)

    match = TOP_MOVIES[TOP_MOVIES["tmdb_id"] == result["tmdb_id"]]
    title = match.iloc[0]["title"] if not match.empty else "unknown"

    print(f"Recommendation: {title} (tmdb_id={result['tmdb_id']})")
    print(f"\n{result['description']}")
