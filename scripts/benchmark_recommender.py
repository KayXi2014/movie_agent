"""Run the main recommender against the prompt CSV and save inspection results.

This benchmark intentionally calls ``llm.get_recommendation()`` directly instead
of going through the FastAPI layer. The output CSV is meant for manual review:
recommended id/title, message quality, LLM usage, runtime, and repeat safety.

Evaluation metrics:
- genre_match: Does the movie's genre align with the requested genre?
- person_match: If a specific actor/director was requested, are they in the movie?
- year_match: Does the movie fall within the requested time period?
- constraint_match: Overall score (0-1) based on how many constraints were satisfied
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm
from retrieval import GENRE_ALIASES, LANGUAGE_FILTER_MAP

DATA_DIR = ROOT / "data"
DEFAULT_INPUT_CSV = DATA_DIR / "movie_recommender_test_prompts2.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "movie_recommender_test_outputs_direct2.csv"
logger = logging.getLogger(__name__)


def _empty_trace() -> dict[str, Any]:
    return {
        "intent_enabled": bool(llm.ENABLE_LLM_INTENT),
        "intent_used": False,
        "intent_override_keys": "",
        "parse_failures": [],
        "semantic_enabled": False,
        "semantic_available": False,
        "semantic_used": False,
        "semantic_hit_count": 0,
        "semantic_elapsed_s": 0.0,
        "retrieval_mode": "",
        "route": "",
        "confidence": "",
        "convergence": "",
    }


def _capture_profile(trace: dict[str, Any], retrieval_profile: dict[str, Any]) -> None:
    confidence_bundle = retrieval_profile.get("confidence_bundle", {})
    trace.update(
        {
            "semantic_enabled": bool(retrieval_profile.get("semantic_enabled", False)),
            "semantic_available": bool(retrieval_profile.get("semantic_available", False)),
            "semantic_used": bool(retrieval_profile.get("semantic_active", False)),
            "semantic_hit_count": int(retrieval_profile.get("semantic_hit_count", 0) or 0),
            "semantic_elapsed_s": float(retrieval_profile.get("semantic_elapsed_s", 0.0) or 0.0),
            "retrieval_mode": str(retrieval_profile.get("retrieval_mode", "")),
            "route": str(confidence_bundle.get("route", "")),
            "confidence": str(confidence_bundle.get("confidence", "")),
            "convergence": str(confidence_bundle.get("convergence", "")),
        }
    )


def _record_parse_failure(trace: dict[str, Any], raw_response: Any, exc: Exception) -> None:
    raw_text = str(raw_response or "")
    trace["parse_failures"].append(
        {
            "error": repr(exc),
            "raw_response": raw_text,
        }
    )
    logger.warning("JSON parse failure during benchmark: %s", repr(exc))
    logger.warning("Raw model response follows:\n%s", raw_text)


def _get_recommendation_with_trace(prompt: str, history: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the recommender and capture internal routing without changing API output."""
    trace = _empty_trace()
    original_build_shortlist = llm._build_shortlist
    original_extract_json_object = llm._extract_json_object

    def traced_build_shortlist(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        override = kwargs.get("retrieval_profile_override")
        if override:
            trace["intent_used"] = True
            trace["intent_override_keys"] = ",".join(sorted(str(key) for key in override.keys()))
        shortlist_refs, prompt_profile, retrieval_profile = original_build_shortlist(*args, **kwargs)
        _capture_profile(trace, retrieval_profile)
        return shortlist_refs, prompt_profile, retrieval_profile

    def traced_extract_json_object(text: Any) -> dict[str, Any]:
        try:
            return original_extract_json_object(text)
        except Exception as exc:
            _record_parse_failure(trace, text, exc)
            raise

    llm._build_shortlist = traced_build_shortlist
    llm._extract_json_object = traced_extract_json_object
    try:
        result = llm.get_recommendation(prompt, history)
    finally:
        llm._build_shortlist = original_build_shortlist
        llm._extract_json_object = original_extract_json_object
    return result, trace


def _lookup_title(tmdb_id: Any) -> str:
    try:
        movie_id = int(tmdb_id)
    except (TypeError, ValueError):
        return ""
    if movie_id not in llm.MOVIES_BY_TMDB_ID.index:
        return ""
    row = llm.MOVIES_BY_TMDB_ID.loc[movie_id]
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    return str(row.get("title", ""))


def _parse_history(raw_history: Any) -> list[Any]:
    """Accept optional history cells as JSON, Python lists, or comma text."""
    if raw_history is None or pd.isna(raw_history):
        return []
    if isinstance(raw_history, list):
        return raw_history
    text = str(raw_history).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]

    return [item.strip() for item in text.split(",") if item.strip()]


def _history_repeat(result: dict[str, Any], title: str, history: list[Any]) -> bool:
    normalized = [
        llm._normalize_history_item(item)
        for item in history
    ]
    history_ids = {
        int(item["tmdb_id"])
        for item in normalized
        if item and item.get("tmdb_id") is not None
    }
    history_titles = {
        llm._normalize_text(str(item["name"]))
        for item in normalized
        if item and item.get("name")
    }
    try:
        movie_id = int(result.get("tmdb_id"))
    except (TypeError, ValueError):
        movie_id = None
    return (movie_id is not None and movie_id in history_ids) or llm._normalize_text(title) in history_titles


def _get_movie_metadata(tmdb_id: Any) -> dict[str, Any]:
    """Get full movie metadata for evaluation."""
    try:
        movie_id = int(tmdb_id)
    except (TypeError, ValueError):
        return {}
    if movie_id not in llm.MOVIES_BY_TMDB_ID.index:
        return {}
    row = llm.MOVIES_BY_TMDB_ID.loc[movie_id]
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    return {
        "title": str(row.get("title", "")),
        "genres": str(row.get("genres", "")),
        "year": row.get("year"),
        "director": str(row.get("director", "")),
        "top_cast": str(row.get("top_cast", "")),
        "production_countries": str(row.get("production_countries", "")),
        "spoken_languages": str(row.get("spoken_languages", "")),
        "runtime": row.get("runtime"),
        "keywords": str(row.get("keywords", "")),
    }


def _normalize(text: str) -> str:
    """Normalize text for matching."""
    return " ".join(str(text or "").lower().split())


def _extract_requested_genres(preference: str) -> set[str]:
    """Extract genre requests from preference text."""
    pref_lower = _normalize(preference)
    found = set()
    for alias, canonical in GENRE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", pref_lower):
            found.add(canonical.lower())
    return found


def _extract_requested_people(preference: str) -> list[str]:
    """Extract actor/director names from preference text."""
    pref_lower = _normalize(preference)
    people = []

    # Check against known names in database
    all_directors = set()
    all_cast = set()
    for _, row in llm.TOP_MOVIES.iterrows():
        director = str(row.get("director", "")).strip()
        if director:
            all_directors.add(_normalize(director))
        cast = str(row.get("top_cast", ""))
        for name in cast.split(","):
            name = name.strip()
            if name:
                all_cast.add(_normalize(name))

    # Find mentioned names
    for name in all_directors | all_cast:
        if name and len(name) > 3 and name in pref_lower:
            people.append(name)

    return people


def _extract_year_constraint(preference: str) -> dict[str, int] | None:
    """Extract year/era constraints from preference text."""
    pref_lower = _normalize(preference)

    # Decade patterns (80s, 90s, 2000s, etc.)
    decade_match = re.search(r"\b(19[5-9]0|20[0-2]0)s\b", pref_lower)
    if decade_match:
        decade = int(decade_match.group(1))
        return {"min": decade, "max": decade + 9}

    # Short decade format (80s, 90s)
    short_decade = re.search(r"\b([5-9]0)s\b", pref_lower)
    if short_decade:
        decade = 1900 + int(short_decade.group(1))
        return {"min": decade, "max": decade + 9}

    # "after YYYY" pattern
    after_match = re.search(r"\bafter\s+(19\d{2}|20\d{2})\b", pref_lower)
    if after_match:
        return {"min": int(after_match.group(1)) + 1, "max": 2030}

    # "before YYYY" pattern
    before_match = re.search(r"\bbefore\s+(19\d{2}|20\d{2})\b", pref_lower)
    if before_match:
        return {"min": 1900, "max": int(before_match.group(1)) - 1}

    # "from YYYY" or "released YYYY"
    year_match = re.search(r"\b(?:from|released in|in)\s+(19\d{2}|20\d{2})\b", pref_lower)
    if year_match:
        year = int(year_match.group(1))
        return {"min": year, "max": year}

    # Recent/classic (only if no specific decade found)
    if "recent" in pref_lower or "new" in pref_lower:
        return {"min": 2020, "max": 2030}
    if "classic" in pref_lower or "old" in pref_lower:
        return {"min": 1900, "max": 2000}

    return None


def _extract_language_constraint(preference: str) -> str | None:
    """Extract language/country constraint from preference text."""
    pref_lower = _normalize(preference)
    for term in LANGUAGE_FILTER_MAP:
        if term in pref_lower:
            return term
    return None


def _evaluate_recommendation(preference: str, movie_meta: dict[str, Any]) -> dict[str, Any]:
    """Evaluate how well the recommendation matches the preference."""
    if not movie_meta:
        return {
            "genre_match": None,
            "person_match": None,
            "year_match": None,
            "language_match": None,
            "constraints_checked": 0,
            "constraints_passed": 0,
            "match_score": 0.0,
            "match_details": "no_movie_data",
        }

    checks = []
    passed = []
    details = []

    # Genre check
    requested_genres = _extract_requested_genres(preference)
    if requested_genres:
        movie_genres = {g.strip().lower() for g in movie_meta["genres"].split(",") if g.strip()}
        genre_match = bool(requested_genres & movie_genres)
        checks.append("genre")
        if genre_match:
            passed.append("genre")
        details.append(f"genre:{'Y' if genre_match else 'N'}")
    else:
        genre_match = None

    # Person check
    requested_people = _extract_requested_people(preference)
    if requested_people:
        movie_people = _normalize(movie_meta["director"] + " " + movie_meta["top_cast"])
        person_match = any(p in movie_people for p in requested_people)
        checks.append("person")
        if person_match:
            passed.append("person")
        details.append(f"person:{'Y' if person_match else 'N'}")
    else:
        person_match = None

    # Year check
    year_constraint = _extract_year_constraint(preference)
    if year_constraint and movie_meta.get("year"):
        try:
            movie_year = int(movie_meta["year"])
            year_match = year_constraint["min"] <= movie_year <= year_constraint["max"]
        except (TypeError, ValueError):
            year_match = None
        if year_match is not None:
            checks.append("year")
            if year_match:
                passed.append("year")
            details.append(f"year:{'Y' if year_match else 'N'}")
    else:
        year_match = None

    # Language/country check
    language_term = _extract_language_constraint(preference)
    if language_term:
        lang_codes, countries = LANGUAGE_FILTER_MAP.get(language_term, (set(), set()))
        movie_langs = _normalize(movie_meta.get("spoken_languages", ""))
        movie_countries = _normalize(movie_meta.get("production_countries", ""))

        lang_match = any(code in movie_langs for code in lang_codes) if lang_codes else False
        country_match = any(c in movie_countries for c in countries) if countries else False
        language_match = lang_match or country_match

        checks.append("language")
        if language_match:
            passed.append("language")
        details.append(f"lang:{'Y' if language_match else 'N'}")
    else:
        language_match = None

    # Calculate overall score
    if checks:
        match_score = len(passed) / len(checks)
    else:
        match_score = 1.0  # No constraints to check = automatic pass

    return {
        "genre_match": genre_match,
        "person_match": person_match,
        "year_match": year_match,
        "language_match": language_match,
        "constraints_checked": len(checks),
        "constraints_passed": len(passed),
        "match_score": round(match_score, 2),
        "match_details": "|".join(details) if details else "no_constraints",
    }


def run_case(case_index: int, row: pd.Series) -> dict[str, Any]:
    prompt = str(row.get("preference") or row.get("prompt") or "").strip()
    history = _parse_history(row.get("watch_history", ""))
    started_at = time.perf_counter()
    error = ""
    result: dict[str, Any] = {}

    try:
        result, trace = _get_recommendation_with_trace(prompt, history)
    except Exception as exc:
        error = repr(exc)
        trace = _empty_trace()

    runtime_s = round(time.perf_counter() - started_at, 3)
    output_tmdb_id = result.get("tmdb_id", "") if isinstance(result, dict) else ""
    movie_title = _lookup_title(output_tmdb_id)

    # Evaluate recommendation quality
    movie_meta = _get_movie_metadata(output_tmdb_id)
    evaluation = _evaluate_recommendation(prompt, movie_meta)

    return {
        "case_index": case_index,
        "category": str(row.get("category", "")),
        "prompt": prompt,
        "history": json.dumps(history, ensure_ascii=False),
        "output_tmdb_id": output_tmdb_id,
        "movie_title": movie_title,
        "movie_genres": movie_meta.get("genres", ""),
        "movie_year": movie_meta.get("year", ""),
        "recommendation_message": result.get("description", "") if isinstance(result, dict) else "",
        "used_llm": result.get("used_llm", "") if isinstance(result, dict) else "",
        "intent_enabled": trace["intent_enabled"],
        "intent_used": trace["intent_used"],
        "intent_override_keys": trace["intent_override_keys"],
        "parse_failures": json.dumps(trace["parse_failures"], ensure_ascii=False),
        "semantic_enabled": trace["semantic_enabled"],
        "semantic_available": trace["semantic_available"],
        "semantic_used": trace["semantic_used"],
        "semantic_hit_count": trace["semantic_hit_count"],
        "semantic_elapsed_s": round(float(trace["semantic_elapsed_s"]), 3),
        "retrieval_mode": trace["retrieval_mode"],
        "route": trace["route"],
        "confidence": trace["confidence"],
        "convergence": trace["convergence"],
        "runtime_s": runtime_s,
        "history_repeat": _history_repeat(result, movie_title, history) if isinstance(result, dict) else False,
        "match_score": evaluation["match_score"],
        "constraints_checked": evaluation["constraints_checked"],
        "constraints_passed": evaluation["constraints_passed"],
        "match_details": evaluation["match_details"],
        "genre_match": evaluation["genre_match"],
        "person_match": evaluation["person_match"],
        "year_match": evaluation["year_match"],
        "language_match": evaluation["language_match"],
        "error": error,
    }


def run_benchmark(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    cases = pd.read_csv(input_csv, comment="#").fillna("")
    if "preference" not in cases.columns and "prompt" not in cases.columns:
        raise ValueError(f"{input_csv} must contain a 'preference' or 'prompt' column")

    rows = []
    for case_index, row in cases.iterrows():
        output_row = run_case(int(case_index), row)
        rows.append(output_row)
        match_indicator = f"match={output_row['match_score']:.0%}" if output_row["constraints_checked"] > 0 else "match=N/A"
        print(
            f"[{case_index + 1}/{len(cases)}] "
            f"{output_row['runtime_s']:.2f}s "
            f"{match_indicator} "
            f"llm={output_row['used_llm']} "
            f"title={output_row['movie_title'][:25] or '<none>'} "
            f"({output_row['match_details']}) "
            f"{'REPEAT!' if output_row['history_repeat'] else ''}"
            f"{' ERR:' + str(output_row['error'])[:40] if output_row['error'] else ''}"
            f"{' PF:' + str(len(json.loads(output_row['parse_failures']))) if output_row['parse_failures'] else ''}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return results


def print_summary(results: pd.DataFrame) -> None:
    """Print evaluation summary statistics."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    total = len(results)
    errors = results["error"].astype(bool).sum()
    successful = total - errors

    print(f"\nTotal cases: {total}")
    print(f"Successful: {successful} ({successful/total:.1%})")
    print(f"Errors: {errors} ({errors/total:.1%})")

    # Runtime stats
    runtimes = results[results["error"] == ""]["runtime_s"]
    if len(runtimes) > 0:
        print(f"\nRuntime (successful cases):")
        print(f"  Mean: {runtimes.mean():.2f}s")
        print(f"  Median: {runtimes.median():.2f}s")
        print(f"  Min: {runtimes.min():.2f}s")
        print(f"  Max: {runtimes.max():.2f}s")
        print(f"  Under 20s: {(runtimes < 20).sum()}/{len(runtimes)} ({(runtimes < 20).mean():.1%})")

    # LLM usage
    if successful > 0:
        llm_used = results["used_llm"].astype(bool).sum()
        print(f"\nLLM usage: {llm_used}/{successful} ({llm_used/successful:.1%})")

    # History repeat check
    repeats = results["history_repeat"].astype(bool).sum()
    with_history = results[results["history"].apply(lambda x: x != "[]" and x != "")].shape[0]
    print(f"History repeats: {repeats}/{with_history} cases with history")

    # Match score stats (only for cases with constraints)
    constrained = results[results["constraints_checked"] > 0]
    if len(constrained) > 0:
        print(f"\nConstraint matching ({len(constrained)} cases with extractable constraints):")
        print(f"  Mean match score: {constrained['match_score'].mean():.1%}")
        print(f"  Perfect matches (100%): {(constrained['match_score'] == 1.0).sum()}/{len(constrained)}")
        print(f"  Partial matches (>0%): {((constrained['match_score'] > 0) & (constrained['match_score'] < 1)).sum()}/{len(constrained)}")
        print(f"  No matches (0%): {(constrained['match_score'] == 0).sum()}/{len(constrained)}")

        # Per-constraint breakdown
        for constraint in ["genre_match", "person_match", "year_match", "language_match"]:
            checked = constrained[constrained[constraint].notna()]
            if len(checked) > 0:
                passed = checked[constraint].astype(bool).sum()
                print(f"  {constraint}: {passed}/{len(checked)} ({passed/len(checked):.1%})")

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llm.get_recommendation() against a prompt CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="CSV with at least a preference or prompt column.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Where to write benchmark results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(args.input, args.output)
    print(f"\nBenchmark results saved to: {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
