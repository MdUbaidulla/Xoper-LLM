"""Minimal Retrieval-Augmented Generation: a TF-IDF document store that
retrieves the top-k passages for a query and formats them into the
prompt, grounding the model's answer in auditable source text instead
of letting it answer from parametric memory alone.

Swap `TfidfStore` for a real vector database (FAISS, pgvector, a
hosted service) at production scale; the `retrieve` / `build_prompt`
interface stays the same.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TfidfStore:
    documents: list[str] = field(default_factory=list)
    _vectorizer: object | None = field(default=None, init=False, repr=False)
    _matrix: object | None = field(default=None, init=False, repr=False)

    def add_documents(self, docs: list[str]) -> None:
        self.documents.extend(docs)
        self._reindex()

    def load_jsonl(self, path: str, text_field: str = "text") -> None:
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)[text_field])
        self.add_documents(docs)

    def _reindex(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.documents:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(self.documents)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.documents[i], float(sims[i])) for i in ranked if sims[i] > 0]


def build_prompt(query: str, retrieved: list[tuple[str, float]]) -> str:
    if not retrieved:
        return query
    context = "\n\n".join(f"[Source {i + 1}]\n{doc}" for i, (doc, _score) in enumerate(retrieved))
    return (
        f"Use the sources below to answer the question. If the sources don't contain the "
        f"answer, say so.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    )
