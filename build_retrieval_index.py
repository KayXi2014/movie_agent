from __future__ import annotations

from retrieval import RETRIEVAL_DB_PATH, build_retrieval_index


if __name__ == "__main__":
    build_retrieval_index()
    print(f"Built retrieval index at {RETRIEVAL_DB_PATH}")
