import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

import llm
import llm_baseline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CASES_PATH = DATA_DIR / "evaluation_cases.json"
RESULTS_CSV = DATA_DIR / "benchmark_results.csv"
SUMMARY_CSV = DATA_DIR / "benchmark_summary.csv"
HARD_TIMEOUT_SECONDS = 20.0


def normalize_history(history: list[dict[str, Any]]) -> tuple[tuple[int | None, str], ...]:
    normalized = []
    for item in history:
        normalized_item = llm._normalize_history_item(item)
        if normalized_item is None:
            continue
        normalized.append((normalized_item.get("tmdb_id"), normalized_item["name"]))
    return tuple(sorted(set(normalized), key=lambda item: (item[0] is None, item[0], item[1].lower())))


def _lookup_movie_row(tmdb_id: int) -> pd.Series | None:
    if tmdb_id not in llm.MOVIES_BY_TMDB_ID.index:
        return None
    row = llm.MOVIES_BY_TMDB_ID.loc[tmdb_id]
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    return row


def _enrich_result(payload: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    tmdb_id = int(payload["tmdb_id"])
    row = _lookup_movie_row(tmdb_id)
    description = str(payload.get("description", ""))
    used_llm = bool(payload.get("used_llm", True))

    if row is None:
        return {
            "tmdb_id": tmdb_id,
            "title": "unknown",
            "genres": "",
            "description": description,
            "used_llm": used_llm,
            "retrieval_mode": retrieval_mode,
        }

    return {
        "tmdb_id": tmdb_id,
        "title": str(row.title),
        "genres": str(row.genres),
        "description": description,
        "used_llm": used_llm,
        "retrieval_mode": retrieval_mode,
    }


def baseline_pick(preferences: str, history_payload: list[dict[str, Any]]) -> dict[str, Any]:
    history_names = [str(item.get("name", "")).strip() for item in history_payload if str(item.get("name", "")).strip()]
    history_ids = [int(item["tmdb_id"]) for item in history_payload if item.get("tmdb_id") is not None]
    payload = llm_baseline.get_recommendation(preferences, history_names, history_ids)
    return _enrich_result(payload, retrieval_mode="baseline_top5")


def _infer_improved_retrieval_mode(preferences: str, normalized_history: tuple[tuple[int | None, str], ...]) -> str:
    _, _, retrieval_profile = llm._build_shortlist(preferences, normalized_history)
    return str(retrieval_profile.get("retrieval_mode", "unknown"))


def improved_pick(
    preferences: str,
    history_payload: list[dict[str, Any]],
    normalized_history: tuple[tuple[int | None, str], ...],
) -> dict[str, Any]:
    payload = llm.get_recommendation(preferences, history_payload)
    retrieval_mode = _infer_improved_retrieval_mode(preferences, normalized_history)
    return _enrich_result(payload, retrieval_mode=retrieval_mode)


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
    agent_name: str,
    preferences: str,
    history_payload: list[dict[str, Any]],
    normalized_history: tuple[tuple[int | None, str], ...],
) -> dict[str, Any]:
    start = time.perf_counter()
    payload: Any = None
    error = ""
    invalid_json = False

    try:
        if agent_name == "improved":
            payload = improved_pick(preferences, history_payload, normalized_history)
        elif agent_name == "baseline":
            payload = baseline_pick(preferences, history_payload)
        else:
            raise ValueError(f"unknown agent: {agent_name}")
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
        "retrieval_mode": str(payload.get("retrieval_mode", "")) if payload_ok else "",
        "has_error": bool(error),
        "error": error,
    }


def lookup_movie(result: dict[str, Any]) -> pd.Series:
    row = _lookup_movie_row(int(result["tmdb_id"]))
    if row is None:
        raise KeyError(f"tmdb_id {result['tmdb_id']} not found")
    return row


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

    total_score = 0.0
    total_score += 2 * len(genre_hits)
    total_score += 1 * len(term_hits)
    total_score -= 2 * len(discourage_genre_hits)
    total_score -= 1 * len(discourage_term_hits)
    total_score += 2 if non_english_hit else 0
    total_score += history_signal_hits if needs_history_signal else 0
    total_score -= 5 if exact_repeat else 0
    total_score -= 4 if avoided_title or avoided_root else 0
    total_score -= 2 if needs_history_signal and not history_signal_ok else 0

    persuadability = _description_persuadability(result, case, row)
    distinction = _movie_distinction_score(row, history_df)

    vote_penalty = 0
    if row.vote_average < 7.0:
        vote_penalty = -2
    elif row.vote_average >= 8.0:
        vote_penalty = 2

    total_score += persuadability["persuadability_score"] * 2.0
    total_score += distinction["movie_distinction_score"] * 1.5
    total_score += vote_penalty

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
        "total_score": float(total_score),
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
        "total_score": -100.0,
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


def _winner_for_case(case_df: pd.DataFrame) -> str:
    baseline_row = case_df[case_df["agent"] == "baseline"].iloc[0]
    improved_row = case_df[case_df["agent"] == "improved"].iloc[0]

    baseline_disqualified = not bool(baseline_row["hard_constraint_pass"])
    improved_disqualified = not bool(improved_row["hard_constraint_pass"])

    if baseline_disqualified and not improved_disqualified:
        return "improved"
    if improved_disqualified and not baseline_disqualified:
        return "baseline"
    if baseline_disqualified and improved_disqualified:
        return "tie_disqualified"

    baseline_score = float(baseline_row["total_score"])
    improved_score = float(improved_row["total_score"])
    if improved_score > baseline_score:
        return "improved"
    if baseline_score > improved_score:
        return "baseline"
    return "tie"


def summarize_agent(agent_df: pd.DataFrame, wins: int, losses: int, ties: int) -> dict[str, Any]:
    return {
        "agent": agent_df["agent"].iloc[0],
        "cases": int(len(agent_df)),
        "avg_score": round(float(agent_df["total_score"].mean()), 2),
        "avg_runtime": round(float(agent_df["runtime"].mean()), 3),
        "hard_constraint_pass_rate": round(float(agent_df["hard_constraint_pass"].mean()), 3),
        "timeout_rate": round(float(agent_df["timed_out"].mean()), 3),
        "seen_repeat_rate": round(float(agent_df["seen_repeat"].mean()), 3),
        "tmdb_on_list_rate": round(float(agent_df["tmdb_on_list"].mean()), 3),
        "invalid_json_rate": round(float(agent_df["invalid_json"].mean()), 3),
        "constraint_pass_rate": round(float(agent_df["constraint_pass"].mean()), 3),
        "avg_vote_average": round(float(agent_df["vote_average"].mean()), 2),
        "avg_history_signal_hits": round(float(agent_df["history_signal_hits"].mean()), 2),
        "avg_persuadability_score": round(float(agent_df["persuadability_score"].mean()), 2),
        "avg_movie_distinction_score": round(float(agent_df["movie_distinction_score"].mean()), 2),
        "avg_history_max_similarity": round(float(agent_df["history_max_similarity"].mean()), 3),
        "used_llm_rate": round(float(agent_df["used_llm"].mean()), 3),
        "error_rate": round(float(agent_df["has_error"].mean()), 3),
        "unique_titles": int(agent_df["movie_title"].nunique()),
        "pairwise_wins": wins,
        "pairwise_losses": losses,
        "pairwise_ties": ties,
    }


def main() -> None:
    if not os.environ.get("OLLAMA_API_KEY"):
        print("Error: OLLAMA_API_KEY environment variable not set.")
        print("Set it with: export OLLAMA_API_KEY=your_key_here")
        return

    cases = json.loads(CASES_PATH.read_text())
    result_rows: list[dict[str, Any]] = []

    for case in cases:
        history_payload = case.get("history", [])
        normalized_history = normalize_history(history_payload)

        for agent_name in ("baseline", "improved"):
            run = run_system(
                agent_name,
                case["preferences"],
                history_payload,
                normalized_history,
            )

            evaluation = (
                evaluate_result(
                    run["payload"],
                    case,
                    normalized_history,
                    run["elapsed_seconds"],
                )
                if run["payload"] is not None
                else _failed_eval()
            )

            payload = run["payload"] or {}
            result_rows.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "agent": agent_name,
                    "input_prompt": case["preferences"],
                    "history_count": len(history_payload),
                    "movie_id": int(payload.get("tmdb_id", -1)) if payload else -1,
                    "movie_title": evaluation["title"],
                    "description": str(payload.get("description", "")) if payload else "",
                    "used_llm": run["used_llm"],
                    "retrieval_mode": run["retrieval_mode"] or ("baseline_top5" if agent_name == "baseline" else ""),
                    "runtime": run["elapsed_seconds"],
                    "hard_constraint_pass": run["hard_constraint_pass"],
                    "timed_out": run["timed_out"],
                    "invalid_json": run["invalid_json"],
                    "tmdb_on_list": run["tmdb_on_list"],
                    "seen_repeat": run["seen_repeat"],
                    "has_error": run["has_error"],
                    "error": run["error"],
                    "constraint_pass": evaluation["constraint_pass"],
                    "vote_average": evaluation["vote_average"],
                    "history_signal_hits": evaluation["history_signal_hits"],
                    "persuadability_score": evaluation["persuadability_score"],
                    "movie_distinction_score": evaluation["movie_distinction_score"],
                    "history_max_similarity": evaluation["history_max_similarity"],
                    "total_score": evaluation["total_score"],
                    "genre_hits": ", ".join(evaluation["genre_hits"]),
                    "term_hits": ", ".join(evaluation["term_hits"]),
                    "discourage_hits": ", ".join(evaluation["discourage_genre_hits"] + evaluation["discourage_term_hits"]),
                }
            )

    results_df = pd.DataFrame(result_rows)

    winner_map: dict[str, str] = {}
    for case_id, case_df in results_df.groupby("case_id"):
        winner = _winner_for_case(case_df)
        winner_map[case_id] = winner
    results_df["winner"] = results_df["case_id"].map(winner_map)

    summary_rows = []
    for agent_name, agent_df in results_df.groupby("agent"):
        wins = sum(1 for winner in winner_map.values() if winner == agent_name)
        losses = sum(1 for winner in winner_map.values() if winner not in {agent_name, "tie", "tie_disqualified"})
        ties = sum(1 for winner in winner_map.values() if winner in {"tie", "tie_disqualified"})
        summary_rows.append(summarize_agent(agent_df, wins=wins, losses=losses, ties=ties))

    summary_df = pd.DataFrame(summary_rows).sort_values("agent")

    results_df.to_csv(RESULTS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"Benchmark results saved to: {RESULTS_CSV}")
    print(f"Benchmark summary saved to: {SUMMARY_CSV}")
    print()
    print(results_df[["case_id", "agent", "movie_title", "runtime", "used_llm", "retrieval_mode", "total_score", "winner"]].to_string(index=False))
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
