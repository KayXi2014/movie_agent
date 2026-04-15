import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

import llm
import llm_baseline


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation_cases.json"
HARD_TIMEOUT_SECONDS = 20.0


def normalize_history(history: list[dict[str, Any]]) -> tuple[tuple[int | None, str], ...]:
    normalized = []
    for item in history:
        normalized_item = llm._normalize_history_item(item)
        if normalized_item is None:
            continue
        normalized.append((normalized_item.get("tmdb_id"), normalized_item["name"]))
    return tuple(sorted(set(normalized), key=lambda item: (item[0] is None, item[0], item[1].lower())))


def _enrich_result(result: dict[str, Any]) -> dict[str, Any]:
    tmdb_id = int(result["tmdb_id"])
    description = str(result.get("description", ""))
    used_llm = bool(result.get("used_llm", True))
    match = llm.MOVIES[llm.MOVIES["tmdb_id"] == tmdb_id]
    if not match.empty:
        row = match.iloc[0]
        return {
            "tmdb_id": tmdb_id,
            "title": str(row.title),
            "genres": str(row.genres),
            "description": description,
            "used_llm": used_llm,
        }
    return {
        "tmdb_id": tmdb_id,
        "title": "unknown",
        "genres": "unknown",
        "description": description,
        "used_llm": used_llm,
    }


def baseline_pick(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    history_list = [name for _, name in history if name]
    return _enrich_result(llm_baseline.get_recommendation(preferences, history_list))


def improved_pick(preferences: str, history_payload: list[dict[str, Any]]) -> dict[str, Any]:
    return _enrich_result(llm.get_recommendation(preferences, history_payload))


def _validate_payload(payload: Any) -> tuple[bool, bool, str]:
    if not isinstance(payload, dict):
        return False, True, "non-dict payload"
    if "tmdb_id" not in payload or "description" not in payload:
        return False, True, "missing required keys"
    try:
        int(payload["tmdb_id"])
    except (TypeError, ValueError):
        return False, True, "tmdb_id is not an integer"
    return True, False, ""


def _seen_recommendation(result: dict[str, Any], history: tuple[tuple[int | None, str], ...]) -> bool:
    history_ids = {tmdb_id for tmdb_id, _ in history if tmdb_id is not None}
    history_titles = {llm._normalize_text(name) for _, name in history if name}
    return int(result["tmdb_id"]) in history_ids or llm._normalize_text(result["title"]) in history_titles


def run_system(
    system_name: str,
    picker,
    preferences: str,
    history_payload: list[dict[str, Any]],
    normalized_history: tuple[tuple[int | None, str], ...],
) -> dict[str, Any]:
    start = time.perf_counter()
    payload: Any = None
    error = ""
    invalid_json = False

    try:
        if system_name == "improved":
            payload = picker(preferences, history_payload)
        else:
            payload = picker(preferences, normalized_history)
    except json.JSONDecodeError as exc:
        error = f"invalid json: {exc}"
        invalid_json = True
    except Exception as exc:
        error = str(exc)

    elapsed_seconds = time.perf_counter() - start
    timed_out = elapsed_seconds > HARD_TIMEOUT_SECONDS

    payload_ok, schema_invalid, schema_error = _validate_payload(payload)
    invalid_json = invalid_json or schema_invalid
    if not error and schema_error:
        error = schema_error

    tmdb_on_list = bool(payload_ok and int(payload["tmdb_id"]) in llm.MOVIES_BY_TMDB_ID.index)
    seen_repeat = bool(payload_ok and _seen_recommendation(payload, normalized_history))

    hard_constraint_pass = bool(
        payload_ok
        and not timed_out
        and tmdb_on_list
        and not seen_repeat
        and not invalid_json
    )

    return {
        "payload": payload if payload_ok else None,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timed_out": timed_out,
        "invalid_json": invalid_json,
        "tmdb_on_list": tmdb_on_list,
        "seen_repeat": seen_repeat,
        "hard_constraint_pass": hard_constraint_pass,
        "used_llm": bool(payload_ok and bool(payload.get("used_llm", True))),
        "has_error": bool(error),
        "error": error,
    }


def lookup_movie(result: dict[str, Any]) -> pd.Series:
    return llm.MOVIES_BY_TMDB_ID.loc[int(result["tmdb_id"])]


def movie_blob(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.title),
            str(row.genres),
            str(row.keywords),
            str(row.overview),
            str(row.tagline),
            str(row.director),
            str(row.top_cast),
            str(row.original_language),
            str(row.production_countries),
        ]
    ).lower()


def history_signal_count(row: pd.Series, history_df: pd.DataFrame) -> int:
    if history_df.empty:
        return 0

    shared = 0
    if any(not row.director_set.isdisjoint(h.director_set) for h in history_df.itertuples()):
        shared += 1
    if any(not row.cast_set.isdisjoint(h.cast_set) for h in history_df.itertuples()):
        shared += 1
    if any(not row.genres_set.isdisjoint(h.genres_set) for h in history_df.itertuples()):
        shared += 1
    return shared


def _description_persuadability(result: dict[str, Any], case: dict[str, Any], row: pd.Series) -> dict[str, float]:
    description_raw = str(result.get("description", "")).strip()
    if not description_raw:
        return {
            "persuadability_score": -4.0,
            "hook_signal": 0.0,
            "preference_language": 0.0,
            "movie_specificity": 0.0,
            "expected_term_hits": 0.0,
            "generic_penalty": 2.0,
        }

    description = description_raw.lower()
    description_tokens = llm._tokenize(description_raw)
    preference_tokens = llm._tokenize(case.get("preferences", ""))
    movie_tokens = llm._tokenize(
        " ".join(
            [
                str(row.title),
                str(row.genres),
                str(row.keywords),
                str(row.director),
                str(row.top_cast),
            ]
        )
    )

    expected_terms = {t.lower() for t in case.get("expected_positive_keywords", [])}
    expected_term_hits = sum(1 for term in expected_terms if term in description)
    preference_language = min(len(description_tokens.intersection(preference_tokens)), 6)
    movie_specificity = min(len(description_tokens.intersection(movie_tokens)), 6)

    hook_phrases = ("imagine", "tonight", "right away", "instantly", "you'll", "you will", "if you're")
    hook_signal = 1.0 if any(phrase in description[:120] for phrase in hook_phrases) or "!" in description[:120] else 0.0

    generic_phrases = (
        "great movie",
        "good movie",
        "strong pick",
        "good choice",
        "worth watching",
    )
    generic_penalty = 1.0 if any(phrase in description for phrase in generic_phrases) else 0.0
    if len(description_raw) < 70:
        generic_penalty += 1.0

    persuadability_score = (
        1.2 * hook_signal
        + 0.8 * min(expected_term_hits, 4)
        + 0.35 * preference_language
        + 0.35 * movie_specificity
        - generic_penalty
    )

    return {
        "persuadability_score": float(persuadability_score),
        "hook_signal": float(hook_signal),
        "preference_language": float(preference_language),
        "movie_specificity": float(movie_specificity),
        "expected_term_hits": float(expected_term_hits),
        "generic_penalty": float(generic_penalty),
    }


def _movie_distinction_score(row: pd.Series, history_df: pd.DataFrame) -> dict[str, float]:
    if history_df.empty:
        return {"movie_distinction_score": 0.0, "history_max_similarity": 0.0}

    max_similarity = 0.0
    for hist in history_df.itertuples():
        genre_union = row.genres_set.union(hist.genres_set)
        genre_jaccard = len(row.genres_set.intersection(hist.genres_set)) / max(len(genre_union), 1)
        director_overlap = 1.0 if not row.director_set.isdisjoint(hist.director_set) else 0.0
        cast_overlap = min(len(row.cast_set.intersection(hist.cast_set)), 2) / 2.0
        root_overlap = 1.0 if row.title_root and row.title_root == hist.title_root else 0.0

        similarity = 0.45 * genre_jaccard + 0.30 * director_overlap + 0.20 * cast_overlap + 0.05 * root_overlap
        max_similarity = max(max_similarity, similarity)

    distinction_score = (1.0 - max_similarity) * 4.0
    return {
        "movie_distinction_score": float(distinction_score),
        "history_max_similarity": float(max_similarity),
    }


def evaluate_result(
    result: dict[str, Any],
    case: dict[str, Any],
    history: tuple[tuple[int | None, str], ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    row = lookup_movie(result)
    blob = movie_blob(row)
    genres = set(row.genres_set)
    title = str(row.title).strip().lower()
    title_root = llm._title_root(row.title)

    expected_genres = {g.lower() for g in case.get("expected_positive_genres", [])}
    expected_terms = {t.lower() for t in case.get("expected_positive_keywords", [])}
    discourage_genres = {g.lower() for g in case.get("discourage_genres", [])}
    discourage_terms = {t.lower() for t in case.get("discourage_terms", [])}
    avoid_titles = {t.strip().lower() for t in case.get("avoid_titles", [])}
    avoid_title_roots = {t.strip().lower() for t in case.get("avoid_title_roots", [])}

    history_ids = {tmdb_id for tmdb_id, _ in history if tmdb_id is not None}
    history_titles_normalized = {llm._normalize_text(name) for _, name in history if name}
    history_df = llm._history_rows(history)

    genre_hits = sorted(expected_genres.intersection(genres))
    term_hits = sorted(term for term in expected_terms if term in blob)
    discourage_genre_hits = sorted(discourage_genres.intersection(genres))
    discourage_term_hits = sorted(term for term in discourage_terms if term in blob)

    exact_repeat = int(row.tmdb_id) in history_ids or llm._normalize_text(row.title) in history_titles_normalized
    avoided_title = title in avoid_titles
    avoided_root = title_root in avoid_title_roots if title_root else False
    non_english_hit = bool(case.get("prefer_non_english")) and str(row.original_language).lower() != "en"
    history_signal_hits = history_signal_count(row, history_df)
    needs_history_signal = bool(case.get("requires_history_signal"))
    history_signal_ok = (history_signal_hits > 0) if needs_history_signal else True

    constraint_pass = not exact_repeat and not avoided_title and not avoided_root

    total_score = 0
    total_score += 2 * len(genre_hits)
    total_score += 1 * len(term_hits)
    total_score -= 2 * len(discourage_genre_hits)
    total_score -= 1 * len(discourage_term_hits)
    total_score += 2 if non_english_hit else 0
    total_score += history_signal_hits if needs_history_signal else 0
    total_score -= 5 if exact_repeat else 0
    total_score -= 4 if avoided_title or avoided_root else 0
    total_score -= 2 if needs_history_signal and not history_signal_ok else 0

    # DESCRIPTION + DISTINCTION METRICS
    persuadability = _description_persuadability(result, case, row)
    distinction = _movie_distinction_score(row, history_df)

    # 2. MOVIE QUALITY (per-case)
    vote_penalty = 0
    if row.vote_average < 7.0:  # Low quality movie
        vote_penalty = -2
    elif row.vote_average >= 8.0:  # High quality bonus
        vote_penalty = 2

    # 3. ADD TO SCORE (new scope emphasis)
    total_score += persuadability["persuadability_score"] * 2.0
    total_score += distinction["movie_distinction_score"] * 1.5
    total_score += vote_penalty  # Movie quality matters

    # 4. Runtime is part of scoring.
    # Faster responses receive a small bonus; slower responses get increasing penalties.
    # Hard >20s disqualification is enforced separately in hard_constraint_pass.
    runtime_bonus = 0.0
    runtime_penalty = 0.0
    if elapsed_seconds <= 5:
        runtime_bonus = 2.0
    elif elapsed_seconds <= 10:
        runtime_bonus = 1.0
    elif elapsed_seconds <= 15:
        runtime_penalty = 1.0
    elif elapsed_seconds <= HARD_TIMEOUT_SECONDS:
        runtime_penalty = 3.0
    else:
        runtime_penalty = 6.0

    total_score += runtime_bonus
    total_score -= runtime_penalty

    return {
        "tmdb_id": int(row.tmdb_id),
        "title": str(row.title),
        "genres": str(row.genres),
        "vote_average": float(row.vote_average),
        "constraint_pass": constraint_pass,
        "genre_hits": genre_hits,
        "term_hits": term_hits,
        "discourage_genre_hits": discourage_genre_hits,
        "discourage_term_hits": discourage_term_hits,
        "history_signal_hits": history_signal_hits,
        "history_signal_ok": history_signal_ok,
        "non_english_hit": non_english_hit,
        "exact_repeat": exact_repeat,
        "avoided_title_hit": avoided_title or avoided_root,
        "total_score": total_score,
        "persuadability_score": persuadability["persuadability_score"],
        "hook_signal": persuadability["hook_signal"],
        "preference_language": persuadability["preference_language"],
        "movie_specificity": persuadability["movie_specificity"],
        "expected_term_hits": persuadability["expected_term_hits"],
        "generic_penalty": persuadability["generic_penalty"],
        "movie_distinction_score": distinction["movie_distinction_score"],
        "history_max_similarity": distinction["history_max_similarity"],
        "movie_quality_score": vote_penalty,
        "runtime_bonus": runtime_bonus,
        "runtime_penalty": runtime_penalty,
    }


def summarize(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    return {
        "avg_score": round(float(df[f"{prefix}_total_score"].mean()), 2),
        "avg_latency_seconds": round(float(df[f"{prefix}_elapsed_seconds"].mean()), 3),
        "hard_constraint_pass_rate": round(float(df[f"{prefix}_hard_constraint_pass"].mean()), 3),
        "disqualification_rate": round(float(df[f"{prefix}_disqualified"].mean()), 3),
        "timeout_rate": round(float(df[f"{prefix}_timed_out"].mean()), 3),
        "seen_repeat_rate": round(float(df[f"{prefix}_seen_repeat"].mean()), 3),
        "tmdb_on_list_rate": round(float(df[f"{prefix}_tmdb_on_list"].mean()), 3),
        "invalid_json_rate": round(float(df[f"{prefix}_invalid_json"].mean()), 3),
        "constraint_pass_rate": round(float(df[f"{prefix}_constraint_pass"].mean()), 3),
        "avg_vote_average": round(float(df[f"{prefix}_vote_average"].mean()), 2),
        "avg_history_signal_hits": round(float(df[f"{prefix}_history_signal_hits"].mean()), 2),
        "avg_persuadability_score": round(float(df[f"{prefix}_persuadability_score"].mean()), 2),
        "avg_movie_distinction_score": round(float(df[f"{prefix}_movie_distinction_score"].mean()), 2),
        "avg_history_max_similarity": round(float(df[f"{prefix}_history_max_similarity"].mean()), 3),
        "used_llm_rate": round(float(df[f"{prefix}_used_llm"].mean()), 3),
        "error_rate": round(float(df[f"{prefix}_has_error"].mean()), 3),
        "unique_titles": int(df[f"{prefix}_title"].nunique()),
    }


def _failed_eval() -> dict[str, Any]:
    return {
        "tmdb_id": -1,
        "title": "invalid-response",
        "genres": "",
        "vote_average": 0.0,
        "constraint_pass": False,
        "genre_hits": [],
        "term_hits": [],
        "discourage_genre_hits": [],
        "discourage_term_hits": [],
        "history_signal_hits": 0,
        "history_signal_ok": False,
        "non_english_hit": False,
        "exact_repeat": False,
        "avoided_title_hit": False,
        "total_score": -100,
        "persuadability_score": -4.0,
        "hook_signal": 0.0,
        "preference_language": 0.0,
        "movie_specificity": 0.0,
        "expected_term_hits": 0.0,
        "generic_penalty": 2.0,
        "movie_distinction_score": 0.0,
        "history_max_similarity": 1.0,
        "movie_quality_score": -2,
        "runtime_bonus": 0.0,
        "runtime_penalty": 6.0,
    }


def main() -> None:
    if not os.environ.get("OLLAMA_API_KEY"):
        print("Error: OLLAMA_API_KEY environment variable not set.")
        print("Set it with: $env:OLLAMA_API_KEY = 'your_key_here' (PowerShell)")
        print("or: export OLLAMA_API_KEY=your_key_here (Bash)")
        return

    cases = json.loads(CASES_PATH.read_text())
    rows = []

    for case in cases:
        request_payload = {
            "user_id": case["user_id"],
            "preferences": case["preferences"],
            "history": case.get("history", []),
        }
        normalized_history = normalize_history(request_payload["history"])

        baseline_run = run_system(
            "baseline",
            baseline_pick,
            request_payload["preferences"],
            request_payload["history"],
            normalized_history,
        )
        improved_run = run_system(
            "improved",
            improved_pick,
            request_payload["preferences"],
            request_payload["history"],
            normalized_history,
        )

        baseline_eval = (
            evaluate_result(
                baseline_run["payload"],
                case,
                normalized_history,
                baseline_run["elapsed_seconds"],
            )
            if baseline_run["payload"] is not None
            else _failed_eval()
        )
        improved_eval = (
            evaluate_result(
                improved_run["payload"],
                case,
                normalized_history,
                improved_run["elapsed_seconds"],
            )
            if improved_run["payload"] is not None
            else _failed_eval()
        )

        baseline_disqualified = not baseline_run["hard_constraint_pass"]
        improved_disqualified = not improved_run["hard_constraint_pass"]

        if baseline_disqualified and not improved_disqualified:
            winner = "improved"
        elif improved_disqualified and not baseline_disqualified:
            winner = "baseline"
        elif baseline_disqualified and improved_disqualified:
            winner = "tie_disqualified"
        else:
            winner = (
                "improved"
                if improved_eval["total_score"] > baseline_eval["total_score"]
                else "baseline"
                if baseline_eval["total_score"] > improved_eval["total_score"]
                else "tie"
            )

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "history_count": len(request_payload["history"]),
                "baseline_title": baseline_eval["title"],
                "baseline_total_score": baseline_eval["total_score"],
                "baseline_constraint_pass": baseline_eval["constraint_pass"],
                "baseline_vote_average": baseline_eval["vote_average"],
                "baseline_genre_hits": ", ".join(baseline_eval["genre_hits"]),
                "baseline_term_hits": ", ".join(baseline_eval["term_hits"]),
                "baseline_discourage_hits": ", ".join(
                    baseline_eval["discourage_genre_hits"] + baseline_eval["discourage_term_hits"]
                ),
                "baseline_history_signal_hits": baseline_eval["history_signal_hits"],
                "baseline_persuadability_score": baseline_eval["persuadability_score"],
                "baseline_movie_distinction_score": baseline_eval["movie_distinction_score"],
                "baseline_history_max_similarity": baseline_eval["history_max_similarity"],
                "baseline_runtime_bonus": baseline_eval["runtime_bonus"],
                "baseline_runtime_penalty": baseline_eval["runtime_penalty"],
                "baseline_elapsed_seconds": baseline_run["elapsed_seconds"],
                "baseline_timed_out": baseline_run["timed_out"],
                "baseline_seen_repeat": baseline_run["seen_repeat"],
                "baseline_tmdb_on_list": baseline_run["tmdb_on_list"],
                "baseline_invalid_json": baseline_run["invalid_json"],
                "baseline_hard_constraint_pass": baseline_run["hard_constraint_pass"],
                "baseline_disqualified": baseline_disqualified,
                "baseline_used_llm": baseline_run["used_llm"],
                "baseline_has_error": baseline_run["has_error"],
                "baseline_error": baseline_run["error"],
                "improved_title": improved_eval["title"],
                "improved_total_score": improved_eval["total_score"],
                "improved_constraint_pass": improved_eval["constraint_pass"],
                "improved_vote_average": improved_eval["vote_average"],
                "improved_genre_hits": ", ".join(improved_eval["genre_hits"]),
                "improved_term_hits": ", ".join(improved_eval["term_hits"]),
                "improved_discourage_hits": ", ".join(
                    improved_eval["discourage_genre_hits"] + improved_eval["discourage_term_hits"]
                ),
                "improved_history_signal_hits": improved_eval["history_signal_hits"],
                "improved_persuadability_score": improved_eval["persuadability_score"],
                "improved_movie_distinction_score": improved_eval["movie_distinction_score"],
                "improved_history_max_similarity": improved_eval["history_max_similarity"],
                "improved_runtime_bonus": improved_eval["runtime_bonus"],
                "improved_runtime_penalty": improved_eval["runtime_penalty"],
                "improved_elapsed_seconds": improved_run["elapsed_seconds"],
                "improved_timed_out": improved_run["timed_out"],
                "improved_seen_repeat": improved_run["seen_repeat"],
                "improved_tmdb_on_list": improved_run["tmdb_on_list"],
                "improved_invalid_json": improved_run["invalid_json"],
                "improved_hard_constraint_pass": improved_run["hard_constraint_pass"],
                "improved_disqualified": improved_disqualified,
                "improved_used_llm": improved_run["used_llm"],
                "improved_has_error": improved_run["has_error"],
                "improved_error": improved_run["error"],
                "winner": winner,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    baseline_summary = summarize(df, "baseline")
    improved_summary = summarize(df, "improved")

    print("Baseline summary:", baseline_summary)
    print("Improved summary:", improved_summary)
    print(
        "Error rates:",
        {
            "baseline_error_rate": baseline_summary["error_rate"],
            "improved_error_rate": improved_summary["error_rate"],
        },
    )
    print(
        "LLM usage rates:",
        {
            "baseline_used_llm_rate": baseline_summary["used_llm_rate"],
            "improved_used_llm_rate": improved_summary["used_llm_rate"],
        },
    )
    print(
        "Average runtime (seconds):",
        {
            "baseline": baseline_summary["avg_latency_seconds"],
            "improved": improved_summary["avg_latency_seconds"],
        },
    )
    print("Pairwise wins:", df["winner"].value_counts().to_dict())

    results_csv = ROOT / "benchmark_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")

    summary_data = {
        "metric": [
            "avg_score",
            "avg_latency_seconds",
            "hard_constraint_pass_rate",
            "disqualification_rate",
            "timeout_rate",
            "seen_repeat_rate",
            "tmdb_on_list_rate",
            "invalid_json_rate",
            "constraint_pass_rate",
            "avg_vote_average",
            "avg_history_signal_hits",
            "avg_persuadability_score",
            "avg_movie_distinction_score",
            "avg_history_max_similarity",
            "used_llm_rate",
            "error_rate",
            "unique_titles",
        ],
        "baseline": [
            baseline_summary["avg_score"],
            baseline_summary["avg_latency_seconds"],
            baseline_summary["hard_constraint_pass_rate"],
            baseline_summary["disqualification_rate"],
            baseline_summary["timeout_rate"],
            baseline_summary["seen_repeat_rate"],
            baseline_summary["tmdb_on_list_rate"],
            baseline_summary["invalid_json_rate"],
            baseline_summary["constraint_pass_rate"],
            baseline_summary["avg_vote_average"],
            baseline_summary["avg_history_signal_hits"],
            baseline_summary["avg_persuadability_score"],
            baseline_summary["avg_movie_distinction_score"],
            baseline_summary["avg_history_max_similarity"],
            baseline_summary["used_llm_rate"],
            baseline_summary["error_rate"],
            baseline_summary["unique_titles"],
        ],
        "improved": [
            improved_summary["avg_score"],
            improved_summary["avg_latency_seconds"],
            improved_summary["hard_constraint_pass_rate"],
            improved_summary["disqualification_rate"],
            improved_summary["timeout_rate"],
            improved_summary["seen_repeat_rate"],
            improved_summary["tmdb_on_list_rate"],
            improved_summary["invalid_json_rate"],
            improved_summary["constraint_pass_rate"],
            improved_summary["avg_vote_average"],
            improved_summary["avg_history_signal_hits"],
            improved_summary["avg_persuadability_score"],
            improved_summary["avg_movie_distinction_score"],
            improved_summary["avg_history_max_similarity"],
            improved_summary["used_llm_rate"],
            improved_summary["error_rate"],
            improved_summary["unique_titles"],
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = ROOT / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to: {summary_csv}")


if __name__ == "__main__":
    main()
