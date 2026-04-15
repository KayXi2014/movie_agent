import json
from pathlib import Path

import pandas as pd

import llm


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "evaluation_cases.json"


def baseline_pick(preferences: str, history: list[str]) -> dict:
    baseline_movies = llm.MOVIES.nlargest(40, "vote_count").copy()
    profile = llm._derive_user_profile(preferences, tuple(history))
    baseline_movies["score"] = baseline_movies.apply(llm._score_movie, axis=1, profile=profile)
    ranked = baseline_movies.sort_values(["score", "vote_average", "vote_count"], ascending=False)
    top = ranked.iloc[0]
    return {
        "tmdb_id": int(top.tmdb_id),
        "title": top.title,
        "genres": top.genres,
        "score": round(float(top.score), 2),
    }


def improved_pick(preferences: str, history: list[str]) -> dict:
    shortlist = llm._build_shortlist(preferences, tuple(history))
    top = shortlist[0]
    return {
        "tmdb_id": int(top["tmdb_id"]),
        "title": top["title"],
        "genres": top["genres"],
        "score": round(float(top["score"]), 2),
    }


def score_result(result: dict, case: dict) -> dict:
    genres = {part.strip().lower() for part in str(result["genres"]).split(",") if part.strip()}
    title = str(result["title"]).strip().lower()

    expected = set(case.get("expected_positive_genres", []))
    discouraged = set(case.get("discourage_genres", []))
    avoided = {name.strip().lower() for name in case.get("avoid_titles", [])}

    expected_hits = sorted(expected.intersection(genres))
    discouraged_hits = sorted(discouraged.intersection(genres))
    avoided_hit = title in avoided

    score = len(expected_hits) - len(discouraged_hits) - (2 if avoided_hit else 0)
    return {
        "score": score,
        "expected_hits": expected_hits,
        "discouraged_hits": discouraged_hits,
        "avoided_hit": avoided_hit,
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
        history_names = [item["name"] for item in request_payload["history"]]
        baseline = baseline_pick(request_payload["preferences"], history_names)
        improved = improved_pick(request_payload["preferences"], history_names)

        baseline_eval = score_result(baseline, case)
        improved_eval = score_result(improved, case)

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "baseline_title": baseline["title"],
                "baseline_genres": baseline["genres"],
                "baseline_score": baseline_eval["score"],
                "baseline_expected_hits": ", ".join(baseline_eval["expected_hits"]),
                "baseline_discouraged_hits": ", ".join(baseline_eval["discouraged_hits"]),
                "baseline_avoided_hit": baseline_eval["avoided_hit"],
                "improved_title": improved["title"],
                "improved_genres": improved["genres"],
                "improved_score": improved_eval["score"],
                "improved_expected_hits": ", ".join(improved_eval["expected_hits"]),
                "improved_discouraged_hits": ", ".join(improved_eval["discouraged_hits"]),
                "improved_avoided_hit": improved_eval["avoided_hit"],
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print("Average baseline score:", round(df["baseline_score"].mean(), 2))
    print("Average improved score:", round(df["improved_score"].mean(), 2))
    print("Cases improved:", int((df["improved_score"] > df["baseline_score"]).sum()), "of", len(df))


if __name__ == "__main__":
    main()
