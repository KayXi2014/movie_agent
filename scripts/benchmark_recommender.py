"""Run the main recommender against the prompt CSV and save inspection results.

This benchmark intentionally calls ``llm.get_recommendation()`` directly instead
of going through the FastAPI layer. The output CSV is meant for manual review:
recommended id/title, message quality, LLM usage, runtime, and repeat safety.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm

DATA_DIR = ROOT / "data"
DEFAULT_INPUT_CSV = DATA_DIR / "movie_recommender_test_prompts2.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "movie_recommender_test_outputs_direct2.csv"


def _empty_trace() -> dict[str, Any]:
    return {
        "intent_enabled": bool(llm.ENABLE_LLM_INTENT),
        "intent_used": False,
        "intent_override_keys": "",
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


def _get_recommendation_with_trace(prompt: str, history: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the recommender and capture internal routing without changing API output."""
    trace = _empty_trace()
    original_build_shortlist = llm._build_shortlist

    def traced_build_shortlist(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        override = kwargs.get("retrieval_profile_override")
        if override:
            trace["intent_used"] = True
            trace["intent_override_keys"] = ",".join(sorted(str(key) for key in override.keys()))
        shortlist_refs, prompt_profile, retrieval_profile = original_build_shortlist(*args, **kwargs)
        _capture_profile(trace, retrieval_profile)
        return shortlist_refs, prompt_profile, retrieval_profile

    llm._build_shortlist = traced_build_shortlist
    try:
        result = llm.get_recommendation(prompt, history)
    finally:
        llm._build_shortlist = original_build_shortlist
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

    return {
        "case_index": case_index,
        "category": str(row.get("category", "")),
        "prompt": prompt,
        "history": json.dumps(history, ensure_ascii=False),
        "output_tmdb_id": output_tmdb_id,
        "movie_title": movie_title,
        "recommendation_message": result.get("description", "") if isinstance(result, dict) else "",
        "used_llm": result.get("used_llm", "") if isinstance(result, dict) else "",
        "intent_enabled": trace["intent_enabled"],
        "intent_used": trace["intent_used"],
        "intent_override_keys": trace["intent_override_keys"],
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
        "error": error,
    }


def run_benchmark(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    cases = pd.read_csv(input_csv).fillna("")
    if "preference" not in cases.columns and "prompt" not in cases.columns:
        raise ValueError(f"{input_csv} must contain a 'preference' or 'prompt' column")

    rows = []
    for case_index, row in cases.iterrows():
        output_row = run_case(int(case_index), row)
        rows.append(output_row)
        print(
            f"[{case_index + 1}/{len(cases)}] "
            f"{output_row['runtime_s']:.3f}s "
            f"used_llm={output_row['used_llm']} "
            f"intent_used={output_row['intent_used']} "
            f"semantic_used={output_row['semantic_used']} "
            f"route={output_row['route']} "
            f"title={output_row['movie_title'] or '<none>'} "
            f"history_repeat={output_row['history_repeat']} "
            f"error={str(output_row['error'])[:80]}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llm.get_recommendation() against a prompt CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="CSV with at least a preference or prompt column.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Where to write benchmark results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(args.input, args.output)
    print(f"Benchmark results saved to: {args.output}")
    print(
        results[
            [
                "case_index",
                "movie_title",
                "used_llm",
                "intent_used",
                "semantic_used",
                "route",
                "confidence",
                "runtime_s",
                "history_repeat",
                "error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
