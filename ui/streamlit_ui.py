import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import build_shortlist, normalize_history


@st.cache_data
def load_movies() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "tmdb_top1000_movies.csv"
    return pd.read_csv(data_path)


MOVIES_DF = load_movies()

st.set_page_config(page_title="Movie Recommender UI", page_icon="🎬", layout="centered")

DEBUG_MODE_OPTIONS = {
    "Auto (recommended)": "auto",
    "Lexical only": "lexical",
    "Full local pipeline": "hybrid",
}


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

    st.write(data.get("description", ""))


def render_shortlist_debug(payload: dict[str, Any], debug_mode: str) -> None:
    import semantic_retrieval as sr

    normalized_history = normalize_history(payload["history"])
    semantic_status_fn = getattr(sr, "semantic_runtime_status", None)
    if callable(semantic_status_fn):
        semantic_status = semantic_status_fn()
    else:
        semantic_status = {
            "ready": bool(getattr(sr, "semantic_ready", lambda: False)()),
            "reason": "legacy_module_without_status",
            "model_name": None,
        }
    requested_mode = DEBUG_MODE_OPTIONS.get(debug_mode, "auto")
    if requested_mode == "auto":
        requested_mode = "hybrid" if semantic_status["ready"] else "lexical"

    try:
        shortlist_refs, prompt_profile, retrieval_profile = build_shortlist(
            payload["preferences"],
            normalized_history,
            mode=requested_mode,
        )
        effective_mode = requested_mode
    except ModuleNotFoundError as exc:
        if requested_mode != "hybrid":
            raise
        shortlist_refs, prompt_profile, retrieval_profile = build_shortlist(
            payload["preferences"],
            normalized_history,
            mode="lexical",
        )
        effective_mode = "lexical"
        st.info(
            "Local semantic debug is unavailable in this environment "
            f"({exc}). Showing lexical-only debug instead."
        )

    st.markdown("### Retrieval Debug")
    st.caption(
        f"Debug mode: {effective_mode} | "
        f"Retrieval mode: {retrieval_profile.get('retrieval_mode', 'unknown')} | "
        f"Candidates shown: {len(shortlist_refs)} | "
        f"Semantic hits: {retrieval_profile.get('semantic_hit_count', 0)}"
    )

    if effective_mode == "lexical":
        if semantic_status["ready"]:
            st.info("Semantic retrieval is available in this environment, but debug is currently running in lexical-only mode.")
        else:
            st.info(
                "Semantic retrieval is unavailable in this Streamlit process, so semantic scores will stay at 0. "
                f"Reason: {semantic_status['reason']}."
            )

    if prompt_profile.get("preferred_themes") or prompt_profile.get("avoid") or prompt_profile.get("target_genres"):
        debug_summary = {
            "target_genres": prompt_profile.get("target_genres", []),
            "preferred_themes": prompt_profile.get("preferred_themes", []),
            "avoid": prompt_profile.get("avoid", []),
        }
        st.code(json.dumps(debug_summary, indent=2), language="json")

    shortlist_rows = []
    for rank, movie in enumerate(shortlist_refs, start=1):
        shortlist_rows.append(
            {
                "rank": rank,
                "tmdb_id": movie["tmdb_id"],
                "title": movie["title"],
                "score": movie.get("score"),
                "semantic": movie.get("semantic_score"),
                "fts": movie.get("fts_score"),
            }
        )

    st.dataframe(pd.DataFrame(shortlist_rows), use_container_width=True, hide_index=True)


st.title("Movie Recommender")
st.caption("Frontend for the FastAPI movie recommendation endpoint, with optional retrieval debug info.")

with st.sidebar:
    st.header("API Settings")
    api_base_url = st.text_input("Backend URL", value="http://127.0.0.1:8080")
    request_timeout = st.number_input("Request timeout (seconds)", min_value=1, max_value=120, value=25)
    show_local_debug = st.checkbox("Show retrieval debug", value=True)
    debug_mode = st.selectbox(
        "Debug mode",
        options=list(DEBUG_MODE_OPTIONS),
        index=0,
        disabled=not show_local_debug,
        help="Auto uses semantic retrieval when this Streamlit process can load the local embedding model.",
    )

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
                st.markdown("## API Result")
                st.caption("This is the backend's actual HTTP response.")
                render_result(api_data, elapsed, "API")

                if show_local_debug:
                    st.markdown("## Retrieval Debug")
                    st.caption(
                        "This debug panel is generated locally from the retrieval pipeline only. "
                        "It does not trigger a second recommendation call."
                    )
                    try:
                        render_shortlist_debug(payload, debug_mode)
                    except Exception as exc:
                        st.warning(f"Retrieval debug failed: {exc}")
            else:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = response.text
                st.error(f"Backend error ({response.status_code})")
                st.caption(f"Response time: {elapsed:.2f}s")
                st.code(json.dumps(error_body, indent=2) if isinstance(error_body, dict) else str(error_body))
