"""Embeddings. Uses sentence-transformers if installed; otherwise a dependency-free
hashing bag-of-words embedding so retrieval works out of the box."""
from __future__ import annotations

import re

import numpy as np

_DIM = 384
_WORD = re.compile(r"[a-z0-9]+")

try:  # optional, better quality if present
    from sentence_transformers import SentenceTransformer  # type: ignore

    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _st_embed(texts: list[str]) -> np.ndarray:
        return np.asarray(_st_model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    HAS_ST = True
except Exception:  # pragma: no cover - fallback path
    HAS_ST = False


def _hash_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in _WORD.findall(t.lower()):
            h = hash(tok)
            out[i, h % _DIM] += 1.0
            out[i, (h // _DIM) % _DIM] += 0.5  # a second bucket reduces collisions
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _DIM), dtype=np.float32)
    return _st_embed(texts) if HAS_ST else _hash_embed(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def backend_name() -> str:
    return "sentence-transformers/all-MiniLM-L6-v2" if HAS_ST else f"hash-bow-{_DIM}d"
