from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    with _lock:
        if _model is None:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        return _model


def embed_queries(texts: Sequence[str]) -> np.ndarray:
    m = get_model()
    prefixed = [f"query: {t}" if not t.strip().lower().startswith("query:") else t for t in texts]
    return m.encode(
        list(prefixed),
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_passages(texts: Sequence[str]) -> np.ndarray:
    m = get_model()
    out: list[str] = []
    for t in texts:
        t = t.strip()
        if t.lower().startswith("passage:"):
            out.append(t)
        else:
            out.append(f"passage: {t}")
    return m.encode(
        out,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_profile(profile: str) -> list[float]:
    v = embed_queries([profile])[0]
    return np.asarray(v, dtype=np.float32).tolist()


def embed_chunks(chunks: Sequence[str]) -> list[list[float]]:
    mat = embed_passages(chunks)
    return [np.asarray(row, dtype=np.float32).tolist() for row in mat]
