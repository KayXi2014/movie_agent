from __future__ import annotations

from retrieval import DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_PATH, build_movie_embeddings


if __name__ == "__main__":
    build_movie_embeddings()
    print(f"Built movie embeddings at {EMBEDDINGS_PATH}")
