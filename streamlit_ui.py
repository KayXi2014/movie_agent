import json
import time
from typing import List, Dict

import pandas as pd
import requests
import streamlit as st

# Load movie data for title lookup
@st.cache_data
def load_movies():
    import os
    data_path = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
    return pd.read_csv(data_path)

MOVIES_DF = load_movies()

st.set_page_config(page_title="Movie Recommender UI", page_icon="🎬", layout="centered")

st.title("Movie Recommender")
st.caption("Frontend for the existing FastAPI movie recommendation endpoint")

with st.sidebar:
    st.header("API Settings")
    api_base_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
    request_timeout = st.number_input("Request timeout (seconds)", min_value=1, max_value=120, value=25)

st.subheader("User Inputs")

user_id = st.number_input("User ID", min_value=1, value=1, step=1)
preferences = st.text_area(
    "Preferences",
    placeholder="Example: I enjoy emotional sci-fi, mind-bending plots, and strong character arcs.",
    height=140,
)

st.markdown("### Watch History")
st.caption("Add movies already watched so the recommender can avoid repeats.")

history_count = st.number_input("How many watched movies?", min_value=0, max_value=20, value=0, step=1)

history: List[Dict[str, object]] = []
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

st.markdown("### Request Preview")
st.code(json.dumps(payload, indent=2), language="json")

if st.button("Get Recommendation", type="primary"):
    if not preferences.strip():
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
                data = response.json()
                st.success("Recommendation generated")
                st.caption(f"Response time: {elapsed:.2f}s")
                # Look up movie title
                tmdb_id = data.get('tmdb_id')
                movie_match = MOVIES_DF[MOVIES_DF['tmdb_id'] == tmdb_id]
                movie_title = movie_match.iloc[0]['title'] if not movie_match.empty else "Unknown"
                st.markdown(f"### 🎬 {movie_title}")
                st.write(data.get("description", ""))
            else:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = response.text
                st.error(f"Backend error ({response.status_code})")
                st.caption(f"Response time: {elapsed:.2f}s")
                st.code(json.dumps(error_body, indent=2) if isinstance(error_body, dict) else str(error_body))
