"""Evidence retrieval.

Two interchangeable backends behind one `Retriever` interface:
  * LexicalRetriever - TF-IDF cosine (scikit-learn only; no download; used offline).
  * DenseRetriever   - sentence-transformers + FAISS (meaning-based; used on GPU runs).

`retrieve` returns (passage, score) pairs; the top score doubles as the retrieval
confidence signal fed to the router.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: list[str]) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]: ...


class LexicalRetriever(Retriever):
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._corpus: list[str] = []

    def index(self, corpus: list[str]) -> None:
        self._corpus = list(corpus)
        if not self._corpus:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(self._corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._matrix is None or not self._corpus:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        order = np.argsort(-sims)[:top_k]
        return [(self._corpus[i], float(sims[i])) for i in order]


class DenseRetriever(Retriever):
    """Meaning-based retrieval. Uses FAISS if available; otherwise falls back to a numpy
    cosine search (per-example contexts are tiny, so brute force is more than fast enough).
    This means `faiss` is optional — handy for local CPU runs where it isn't installed."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self._index = None
        self._emb = None
        self._corpus: list[str] = []

    def index(self, corpus: list[str]) -> None:
        self._corpus = list(corpus)
        self._index = None
        self._emb = None
        if not self._corpus:
            return
        emb = self._model.encode(self._corpus, normalize_embeddings=True)
        self._emb = np.asarray(emb, dtype="float32")
        try:
            import faiss  # lazy, optional

            self._index = faiss.IndexFlatIP(self._emb.shape[1])
            self._index.add(self._emb)
        except Exception:
            self._index = None  # fall back to numpy at query time

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._emb is None or not self._corpus:
            return []
        q = self._model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        if self._index is not None:
            scores, idx = self._index.search(q, min(top_k, len(self._corpus)))
            return [(self._corpus[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        # numpy cosine (embeddings are normalized, so dot product == cosine)
        sims = self._emb @ q[0]
        order = np.argsort(-sims)[: top_k]
        return [(self._corpus[i], float(sims[i])) for i in order]


_RETRIEVER_CACHE: dict = {}


def build_retriever(cfg) -> Retriever:
    # Cached so the embedding model loads once per process. Runs are sequential, so sharing
    # one retriever (its index is re-set per example) is safe.
    key = (cfg.backend, cfg.model_name)
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]
    retriever: Retriever = DenseRetriever(cfg.model_name) if cfg.backend == "dense" else LexicalRetriever()
    _RETRIEVER_CACHE[key] = retriever
    return retriever
