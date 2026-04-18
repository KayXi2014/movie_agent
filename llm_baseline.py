"""
TODO: This is the file you should edit.

get_recommendation() is called once per request with the user's input.
It should return a dict with keys "tmdb_id" and "description".

build_prompt() and call_llm() are broken out as separate functions so they are
easy to swap or extend individually, but you are free to restructure this file
however you like.
"""

import json
import os
import time
import argparse

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# TODO: Edit these to improve your recommendations
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_MOVIES = pd.read_csv(DATA_PATH).nlargest(5, "vote_count")


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    movie_list = "\n".join(
        f'- tmdb_id={row.tmdb_id} | "{row.title}" ({row.year}) | genres: {row.genres} | overview: {row.overview[:200]}'
        for row in TOP_MOVIES.itertuples()
    )
    history_text = (
        ", ".join(
            f'"{name}" (tmdb_id={tid})' for name, tid in zip(history, history_ids)
        ) if history else "none"
    )
    prompt = f"""You are a movie recommendation assistant.

A user is looking for a movie to watch. Here are their preferences:
"{preferences}"

Movies they have already watched (do NOT recommend these):
{history_text}

Below is the list of candidate movies you may recommend. You MUST pick exactly one.

{movie_list}

Respond with ONLY a JSON object — no markdown, no extra text — in this exact format:
{{
  "tmdb_id": <integer>,
  "description": "<a compelling blurb ≤500 chars that tells the user why this movie matches their preferences>"
}}"""
    print(prompt)

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(response.message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history",
        type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    print(
        "For watch history, enter comma-separated movie titles (or leave blank)."
    )

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = (
        [t.strip() for t in history_raw.split(",") if t.strip()]
        if history_raw
        else []
    )

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    elapsed = time.perf_counter() - start

    print(f"\nServed in {elapsed:.2f}s")

