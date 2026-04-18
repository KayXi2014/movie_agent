import json
import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

from llm import get_recommendation


@st.cache_data
def load_movies() -> pd.DataFrame:
    data_path = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
    return pd.read_csv(data_path)


MOVIES_DF = load_movies()

st.set_page_config(page_title="Movie Recommender UI", page_icon="🎬", layout="centered")


def lookup_title(tmdb_id: Any) -> str:
    movie_match = MOVIES_DF[MOVIES_DF["tmdb_id"] == tmdb_id]
    return str(movie_match.iloc[0]["title"]) if not movie_match.empty else "Unknown"


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history", [])
    normalized_history = []
    for item in history:
        if not isinstance(item, dict):
            continue
        tmdb_id = item.get("tmdb_id")
        name = str(item.get("name", "")).strip()
        if tmdb_id in (None, "") or not name:
            continue
        normalized_history.append({"tmdb_id": int(tmdb_id), "name": name})

    return {
        "user_id": int(payload.get("user_id", 1)),
        "preferences": str(payload.get("preferences", "")).strip(),
        "history": normalized_history,
    }


def parse_json_payload(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"

    if not isinstance(payload, dict):
        return None, "JSON payload must be an object."

    try:
        normalized = normalize_payload(payload)
    except (TypeError, ValueError) as exc:
        return None, f"Invalid payload values: {exc}"

    if not normalized["preferences"]:
        return None, "Preferences must not be empty."

    return normalized, None


def render_result(data: dict[str, Any], elapsed: float, source_label: str) -> None:
    st.success("Recommendation generated")
    st.caption(f"{source_label} response time: {elapsed:.2f}s")

    tmdb_id = data.get("tmdb_id")
    movie_title = lookup_title(tmdb_id)
    st.markdown(f"### 🎬 {movie_title}")

    used_llm = data.get("used_llm")
    if used_llm is None:
        st.info("LLM path: not exposed by the API response")
    else:
        indicator = "LLM used" if bool(used_llm) else "Fallback used"
        st.caption(f"LLM path: {indicator}")

    st.write(data.get("description", ""))


st.title("Movie Recommender")
st.caption("Frontend for the FastAPI movie recommendation endpoint, with optional local debug info.")

with st.sidebar:
    st.header("API Settings")
    api_base_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
    request_timeout = st.number_input("Request timeout (seconds)", min_value=1, max_value=120, value=25)
    show_local_debug = st.checkbox("Show local used_llm indicator", value=True)

st.subheader("User Inputs")

input_mode = st.radio("Input mode", options=["Form", "JSON"], horizontal=True)

payload: dict[str, Any]

if input_mode == "Form":
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    preferences = st.text_area(
        "Preferences",
        placeholder="Example: I enjoy emotional sci-fi, mind-bending plots, and strong character arcs.",
        height=140,
    )

    st.markdown("### Watch History")
    st.caption("Add movies already watched so the recommender can avoid repeats.")

    history_count = st.number_input("How many watched movies?", min_value=0, max_value=20, value=0, step=1)

    history = []
    for i in range(history_count):
        col1, col2 = st.columns([1, 2])
        with col1:
            tmdb_id = st.number_input(
                f"TMDB ID #{i + 1}",
                min_value=1,
                value=1,
                step=1,
                key=f"history_tmdb_{i}",
            )
        with col2:
            name = st.text_input(
                f"Movie Name #{i + 1}",
                placeholder="Movie title",
                key=f"history_name_{i}",
            )
        if name.strip():
            history.append({"tmdb_id": int(tmdb_id), "name": name.strip()})

    payload = {
        "user_id": int(user_id),
        "preferences": preferences.strip(),
        "history": history,
    }
else:
    default_payload = {
        "user_id": 1,
        "preferences": "I want to watch a good sci-fi movie, with no superpowers, but grounded in scientific theory.",
        "history": [],
    }
    raw_json = st.text_area(
        "Request JSON",
        value=json.dumps(default_payload, indent=2),
        height=260,
    )
    parsed_payload, parse_error = parse_json_payload(raw_json)
    if parse_error:
        st.error(parse_error)
        payload = {"user_id": 1, "preferences": "", "history": []}
    else:
        payload = parsed_payload

st.markdown("### Request Preview")
st.code(json.dumps(payload, indent=2), language="json")

if st.button("Get Recommendation", type="primary"):
    if not payload["preferences"]:
        st.error("Please enter your preferences before submitting.")
    else:
        endpoint = api_base_url.rstrip("/") + "/recommend"
        start_time = time.perf_counter()
        try:
            response = requests.post(endpoint, json=payload, timeout=int(request_timeout))
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - start_time
            st.error(f"Request failed: {exc}")
            st.caption(f"Elapsed time: {elapsed:.2f}s")
        else:
            elapsed = time.perf_counter() - start_time
            if response.ok:
                api_data = response.json()
                render_result(api_data, elapsed, "API")

                if show_local_debug:
                    debug_start = time.perf_counter()
                    try:
                        local_result = get_recommendation(payload["preferences"], payload["history"])
                    except Exception as exc:
                        debug_elapsed = time.perf_counter() - debug_start
                        st.warning(f"Local debug run failed: {exc}")
                        st.caption(f"Local debug attempt time: {debug_elapsed:.2f}s")
                    else:
                        debug_elapsed = time.perf_counter() - debug_start
                        st.markdown("### Local Debug")
                        render_result(local_result, debug_elapsed, "Local")
            else:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = response.text
                st.error(f"Backend error ({response.status_code})")
                st.caption(f"Response time: {elapsed:.2f}s")
                st.code(json.dumps(error_body, indent=2) if isinstance(error_body, dict) else str(error_body))
