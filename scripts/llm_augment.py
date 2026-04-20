"""
Offline LLM augmentation for semantic retrieval metadata.

This module is not used by the deployed API. It enriches the local movie CSV
with compact semantic fields that improve retrieval quality without adding
runtime LLM cost.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)

ROOT = PROJECT_ROOT
DATA_DIR = ROOT / "data"
BASE_SOURCE_PATH = DATA_DIR / "tmdb_top1000_movies.csv"
ENRICHED_PATH = DATA_DIR / "tmdb_top1000_movies_enriched.csv"
SOURCE_PATH = ENRICHED_PATH if ENRICHED_PATH.exists() else BASE_SOURCE_PATH
TARGET_PATH = ENRICHED_PATH

DEFAULT_AUGMENT_MODEL = "gemma4:31b-cloud"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_AUGMENT_WORKERS = 4
DEFAULT_BATCH_SIZE = 10

STATUS_COLUMN = "llm_augmentation_status"
MODEL_COLUMN = "llm_augmentation_model"
ERROR_COLUMN = "llm_augmentation_error"

AUGMENTATION_COLUMNS = [
    "essence",
    "tone_tags_json",
    "audience_tags_json",
    "source_tags_json",
    "keywords_augmented_json",
    STATUS_COLUMN,
    MODEL_COLUMN,
    ERROR_COLUMN,
]

REQUESTED_FIELDS: tuple[str, ...] = (
    "essence",
    "tone_tags_json",
    "source_tags_json",
    "keywords_augmented_json",
    "audience_tags_json",
)

TONE_TAGS = [
    "grounded",
    "serious",
    "dark",
    "gritty",
    "tense",
    "intense",
    "warm",
    "intimate",
    "thoughtful",
    "cerebral",
    "haunting",
    "bleak",
    "moody",
    "lighthearted",
    "uplifting",
    "romantic",
    "funny",
    "campy",
]

AUDIENCE_TAGS = [
    "date_night",
    "group_watch",
    "solo_watch",
    "late_night_watch",
    "family_watch",
    "easy_watch",
    "requires_attention",
    "rewatchable",
    "conversation_starter",
    "comfort_watch",
]

SOURCE_TAGS = [
    "novel_adaptation",
    "comic_adaptation",
    "video_game_adaptation",
    "true_story",
    "myth_folklore",
    "original",
]


def _enabled() -> bool:
    return bool(os.environ.get("OLLAMA_API_KEY", "").strip())


def _require_credentials() -> None:
    if not _enabled():
        raise SystemExit("OLLAMA_API_KEY is required to run offline LLM augmentation.")


def _timeout_seconds() -> float:
    try:
        return float(os.environ.get("AUGMENT_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _augment_workers() -> int:
    try:
        value = int(os.environ.get("AUGMENT_WORKERS", str(DEFAULT_AUGMENT_WORKERS)))
    except ValueError:
        value = DEFAULT_AUGMENT_WORKERS
    return max(1, value)


def _batch_size() -> int:
    try:
        value = int(os.environ.get("AUGMENT_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    except ValueError:
        value = DEFAULT_BATCH_SIZE
    return max(1, value)


def _model_name(cli_value: str | None = None) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    return os.environ.get("AUGMENT_MODEL", DEFAULT_AUGMENT_MODEL).strip() or DEFAULT_AUGMENT_MODEL


@lru_cache(maxsize=4)
def _get_client(timeout_seconds: float) -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=timeout_seconds,
    )


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

    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")

    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(raw[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def _ensure_columns(rows: pd.DataFrame) -> pd.DataFrame:
    augmented = rows.copy()
    for column in AUGMENTATION_COLUMNS:
        if column not in augmented.columns:
            augmented[column] = pd.Series([""] * len(augmented), index=augmented.index, dtype="object")
        else:
            augmented[column] = augmented[column].fillna("").astype("object")
    return augmented


def _parse_json_string_list(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _csv_list(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _keywords_are_sparse(row: pd.Series) -> bool:
    return len(_csv_list(row.get("keywords", ""))) < 3


def _requested_fields_for_row(row: pd.Series) -> tuple[str, ...]:
    requested = list(REQUESTED_FIELDS)
    if "keywords_augmented_json" in requested and not _keywords_are_sparse(row):
        requested.remove("keywords_augmented_json")
    return tuple(requested)


def _field_missing(row: pd.Series, field: str) -> bool:
    return not str(row.get(field, "") or "").strip()


def _fields_to_generate(row: pd.Series, force: bool) -> tuple[str, ...]:
    requested_fields = _requested_fields_for_row(row)
    if not requested_fields:
        return ()
    if force:
        return requested_fields
    return tuple(field for field in requested_fields if _field_missing(row, field))


def _needs_augmentation(row: pd.Series, force: bool) -> bool:
    return bool(_fields_to_generate(row, force))


def augmentation_status_summary(
    source_path: Path = SOURCE_PATH,
    *,
    top_n: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not source_path.exists():
        return {
            "rows_total": 0,
            "rows_in_scope": 0,
            "rows_complete": 0,
            "rows_needing_llm": 0,
            "rows_error": 0,
        }

    df = _ensure_columns(pd.read_csv(source_path).fillna(""))
    scoped = df if top_n is None else df.head(min(int(top_n), len(df)))

    rows_complete = 0
    rows_needing_llm = 0
    rows_error = 0
    for _, row in scoped.iterrows():
        requested_fields = _requested_fields_for_row(row)
        if not requested_fields:
            continue
        if str(row.get(STATUS_COLUMN, "") or "").strip() == "error":
            rows_error += 1
        if _needs_augmentation(row, force):
            rows_needing_llm += 1
        else:
            rows_complete += 1

    return {
        "rows_total": int(len(df)),
        "rows_in_scope": int(len(scoped)),
        "rows_complete": int(rows_complete),
        "rows_needing_llm": int(rows_needing_llm),
        "rows_error": int(rows_error),
    }


def _chunked(values: list[Any], chunk_size: int) -> list[list[Any]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _limited_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rstrip()
    last_space = trimmed.rfind(" ")
    if last_space > max(20, limit // 2):
        trimmed = trimmed[:last_space]
    return trimmed.rstrip()


def _movie_context(row: pd.Series) -> dict[str, str]:
    recommended_titles = _parse_json_string_list(row.get("recommended_titles", ""))
    payload = {
        "title": str(row.get("title", "")).strip(),
        "year": str(row.get("year", "")).strip(),
        "genres": str(row.get("genres", "")).strip(),
        "overview": _limited_text(row.get("overview", ""), 900),
        "tagline": _limited_text(row.get("tagline", ""), 220),
        "keywords": str(row.get("keywords", "")).strip(),
        "director": str(row.get("director", "")).strip(),
        "top_cast": str(row.get("top_cast", "")).strip(),
        "us_rating": str(row.get("us_rating", "")).strip(),
        "runtime_min": str(row.get("runtime_min", "")).strip(),
        "recommended_titles": ", ".join(recommended_titles[:6]),
        "collection_name": str(row.get("collection_name", "")).strip(),
        "production_countries": str(row.get("production_countries", "")).strip(),
        "spoken_languages": str(row.get("spoken_languages", "")).strip(),
    }
    return payload


def _schema_text(fields: tuple[str, ...]) -> str:
    lines = ["{"]
    for index, field in enumerate(fields):
        suffix = "," if index < len(fields) - 1 else ""
        if field == "essence":
            lines.append(f'  "essence": "string"{suffix}')
        else:
            lines.append(f'  "{field}": ["string"]{suffix}')
    lines.append("}")
    return "\n".join(lines)


def _prompt_for_row(row: pd.Series, fields: tuple[str, ...]) -> str:
    sparse_keywords = _keywords_are_sparse(row)
    context = _movie_context(row)

    requested_rules: list[str] = []
    if "essence" in fields:
        requested_rules.extend(
            [
                "- essence must be 90-140 words.",
                "- essence should describe what the film feels like, its tone, pacing, emotional register, thematic identity, and who would love it.",
                "- essence must be spoiler-free.",
                "- essence must not be a plot recap.",
            ]
        )
    if "tone_tags_json" in fields:
        requested_rules.append(f"- tone_tags_json: choose up to 4 only from: {json.dumps(TONE_TAGS)}")
    if "audience_tags_json" in fields:
        requested_rules.append(f"- audience_tags_json: choose up to 4 only from: {json.dumps(AUDIENCE_TAGS)}")
    if "source_tags_json" in fields:
        requested_rules.append(f"- source_tags_json: choose up to 3 only from: {json.dumps(SOURCE_TAGS)}")
    if "keywords_augmented_json" in fields:
        if sparse_keywords:
            requested_rules.append(
                "- keywords_augmented_json: output 8-12 short thematic keywords because the existing keywords are sparse."
            )
        else:
            requested_rules.append("- keywords_augmented_json: return [] because the existing keywords are already rich enough.")

    prompt = (
        "You are generating compact semantic metadata for a movie recommendation system.\n\n"
        "Use only the provided movie fields and broad high-level knowledge about viewing experience.\n"
        "Do not invent awards, acclaim, soundtrack quality, hidden facts, or detailed plot points not supported by the fields.\n"
        "Do not include spoilers.\n"
        "Do not summarize the plot beyond premise-level framing.\n"
        "Return JSON only.\n\n"
        f"Movie fields:\n"
        f"title: {context['title']}\n"
        f"year: {context['year']}\n"
        f"genres: {context['genres']}\n"
        f"overview: {context['overview']}\n"
        f"tagline: {context['tagline']}\n"
        f"keywords: {context['keywords']}\n"
        f"director: {context['director']}\n"
        f"top_cast: {context['top_cast']}\n"
        f"us_rating: {context['us_rating']}\n"
        f"runtime_min: {context['runtime_min']}\n"
        f"recommended_titles: {context['recommended_titles']}\n"
        f"collection_name: {context['collection_name']}\n"
        f"production_countries: {context['production_countries']}\n"
        f"spoken_languages: {context['spoken_languages']}\n\n"
        "Return exactly:\n"
        f"{_schema_text(fields)}\n\n"
        "Rules:\n"
        + "\n".join(requested_rules)
        + "\n- Keep all tags normalized, deduplicated, and factual."
    )
    return prompt


def _normalize_tag_list(values: Any, allowed: list[str], limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    allowed_map = {value.casefold(): value for value in allowed}
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = str(item or "").strip().casefold()
        if not key or key not in allowed_map or key in seen:
            continue
        seen.add(key)
        normalized.append(allowed_map[key])
        if len(normalized) >= limit:
            break
    return normalized


def _normalize_keywords(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "").strip().lower().replace("_", " "))
        text = text.strip(" ,.;:")
        if not text or len(text) > 48 or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if len(normalized) >= 12:
            break
    return normalized


def _normalize_essence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        raise ValueError("essence is empty")
    words = text.split()
    if len(words) > 170:
        text = " ".join(words[:170]).strip()
    return text


def _normalize_payload(raw_payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if "essence" in fields:
        normalized["essence"] = _normalize_essence(raw_payload.get("essence", ""))
    if "tone_tags_json" in fields:
        normalized["tone_tags_json"] = json.dumps(
            _normalize_tag_list(raw_payload.get("tone_tags_json", []), TONE_TAGS, 4),
            ensure_ascii=False,
        )
    if "audience_tags_json" in fields:
        normalized["audience_tags_json"] = json.dumps(
            _normalize_tag_list(raw_payload.get("audience_tags_json", []), AUDIENCE_TAGS, 4),
            ensure_ascii=False,
        )
    if "source_tags_json" in fields:
        normalized["source_tags_json"] = json.dumps(
            _normalize_tag_list(raw_payload.get("source_tags_json", []), SOURCE_TAGS, 3),
            ensure_ascii=False,
        )
    if "keywords_augmented_json" in fields:
        normalized["keywords_augmented_json"] = json.dumps(
            _normalize_keywords(raw_payload.get("keywords_augmented_json", [])),
            ensure_ascii=False,
        )
    return normalized


def _apply_success_metadata(enriched: pd.DataFrame, index: Any, model_name: str) -> None:
    enriched.at[index, STATUS_COLUMN] = "ok"
    enriched.at[index, MODEL_COLUMN] = model_name
    enriched.at[index, ERROR_COLUMN] = ""


def _complete_without_llm_if_applicable(
    enriched: pd.DataFrame,
    index: Any,
    row: pd.Series,
    model_name: str,
    force: bool,
) -> bool:
    requested_fields = _requested_fields_for_row(row)
    if not requested_fields:
        return False
    if force:
        return False
    if _fields_to_generate(row, force=False):
        return False
    _apply_success_metadata(enriched, index, model_name)
    return True


def _augment_row(row: pd.Series, fields: tuple[str, ...], model_name: str) -> dict[str, str]:
    prompt = _prompt_for_row(row, fields)
    response = _get_client(_timeout_seconds()).chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    payload = _extract_json_object(response.message.content)
    return _normalize_payload(payload, fields)


def augment_dataset(
    source_path: Path = SOURCE_PATH,
    target_path: Path = TARGET_PATH,
    *,
    top_n: int | None = None,
    model_name: str | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
    force: bool = False,
) -> None:
    _require_credentials()

    model_name = _model_name(model_name)
    workers = max(1, workers or _augment_workers())
    batch_size = max(1, batch_size or _batch_size())

    df = pd.read_csv(source_path).fillna("")
    enriched = _ensure_columns(df)
    requested_top_n = len(enriched) if top_n is None else min(int(top_n), len(enriched))

    candidate_indices = []
    metadata_only_indices = []
    for index in enriched.head(requested_top_n).index:
        row = enriched.loc[index]
        if _needs_augmentation(row, force):
            candidate_indices.append(index)
        elif _complete_without_llm_if_applicable(enriched, index, row, model_name, force):
            metadata_only_indices.append(index)

    if not candidate_indices:
        if metadata_only_indices:
            enriched.to_csv(target_path, index=False)
            print(f"Updated augmentation metadata for {len(metadata_only_indices)} completed rows -> {target_path}")
        print("No rows need LLM augmentation.")
        return

    total = len(candidate_indices)
    print(f"Running LLM augmentation: model={model_name} rows={total}")

    completed = 0
    for batch in _chunked(candidate_indices, batch_size):
        field_map = {
            index: _fields_to_generate(enriched.loc[index], force)
            for index in batch
        }
        remaining = [index for index, fields in field_map.items() if fields]
        if remaining:
            with ThreadPoolExecutor(max_workers=min(workers, len(remaining))) as executor:
                futures = {
                    executor.submit(_augment_row, enriched.loc[index].copy(), field_map[index], model_name): index
                    for index in remaining
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        updates = future.result()
                        for field, value in updates.items():
                            enriched.at[index, field] = value
                        _apply_success_metadata(enriched, index, model_name)
                    except Exception as exc:
                        logger.warning("LLM augmentation failed for index=%s tmdb_id=%s error=%r", index, enriched.at[index, "tmdb_id"], exc)
                        enriched.at[index, STATUS_COLUMN] = "error"
                        enriched.at[index, ERROR_COLUMN] = str(exc)[:500]
                    finally:
                        completed += 1

        enriched.to_csv(target_path, index=False)
        print(f"Saved progress: {completed}/{total} rows -> {target_path}")

    print(f"LLM augmentation complete: {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LLM augmentation for movie retrieval metadata.")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="Source CSV to augment.")
    parser.add_argument("--output", type=Path, default=TARGET_PATH, help="Output CSV path.")
    parser.add_argument("--top-n", type=int, default=None, help="Only augment the first N rows.")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name to use.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel augmentation workers.")
    parser.add_argument("--batch-size", type=int, default=None, help="Rows to save per batch.")
    parser.add_argument("--force", action="store_true", help="Re-run augmentation even if version/status look current.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    augment_dataset(
        source_path=args.source,
        target_path=args.output,
        top_n=args.top_n,
        model_name=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
