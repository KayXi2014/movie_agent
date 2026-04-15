import json
from pathlib import Path
from typing import Any

import pandas as pd

import llm


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation_cases.json"


def normalize_history(history: list[dict[str, Any]]) -> tuple[tuple[int | None, str], ...]:
    normalized = []
    for item in history:
        normalized_item = llm._normalize_history_item(item)
        if normalized_item is None:
            continue
        normalized.append((normalized_item.get("tmdb_id"), normalized_item["name"]))
    return tuple(sorted(set(normalized), key=lambda item: (item[0] is None, item[0], item[1].lower())))


def baseline_pick(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    baseline_movies = llm.MOVIES.nlargest(40, "vote_count").copy()
    profile = llm._derive_user_profile(preferences, history)
    baseline_movies["score"] = baseline_movies.apply(llm._score_movie, axis=1, profile=profile)
    ranked = baseline_movies.sort_values(["score", "vote_average", "vote_count"], ascending=False)
    top = ranked.iloc[0]
    return {
        "tmdb_id": int(top.tmdb_id),
        "title": str(top.title),
        "genres": str(top.genres),
        "score": round(float(top.score), 2),
    }


def improved_pick(preferences: str, history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
    shortlist = llm._build_shortlist(preferences, history)
    top = shortlist[0]
    return {
        "tmdb_id": int(top["tmdb_id"]),
        "title": str(top["title"]),
        "genres": str(top["genres"]),
        "score": round(float(top["score"]), 2),
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


def evaluate_result(result: dict[str, Any], case: dict[str, Any], history: tuple[tuple[int | None, str], ...]) -> dict[str, Any]:
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
    }


def summarize(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    return {
        "avg_score": round(float(df[f"{prefix}_total_score"].mean()), 2),
        "constraint_pass_rate": round(float(df[f"{prefix}_constraint_pass"].mean()), 3),
        "avg_vote_average": round(float(df[f"{prefix}_vote_average"].mean()), 2),
        "avg_history_signal_hits": round(float(df[f"{prefix}_history_signal_hits"].mean()), 2),
        "unique_titles": int(df[f"{prefix}_title"].nunique()),
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())
    rows = []

    for case in cases:
        request_payload = {
            "user_id": case["user_id"],
            "preferences": case["preferences"],
            "history": case.get("history", []),
        }
        normalized_history = normalize_history(request_payload["history"])

        baseline = baseline_pick(request_payload["preferences"], normalized_history)
        improved = improved_pick(request_payload["preferences"], normalized_history)

        baseline_eval = evaluate_result(baseline, case, normalized_history)
        improved_eval = evaluate_result(improved, case, normalized_history)

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
                "winner": (
                    "improved"
                    if improved_eval["total_score"] > baseline_eval["total_score"]
                    else "baseline"
                    if baseline_eval["total_score"] > improved_eval["total_score"]
                    else "tie"
                ),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    baseline_summary = summarize(df, "baseline")
    improved_summary = summarize(df, "improved")

    print("Baseline summary:", baseline_summary)
    print("Improved summary:", improved_summary)
    print("Pairwise wins:", df["winner"].value_counts().to_dict())


if __name__ == "__main__":
    main()
