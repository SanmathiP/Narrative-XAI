"""
Lightweight local knowledge base for the chat agent (no external services).
- Loads Markdown/JSON/CSV files from a /docs (or any) folder.
- Provides simple retrieval over a small corpus using TF-IDF cosine similarity.

Usage:
    kb = KnowledgeBase(paths=["docs", "prompts"])  # folders to crawl
    ctx = kb.query("where are we using vectordb here again", k=3)
    # returns joined text snippets (<= ~1k chars) for LLM grounding

Notes:
- This is intentionally tiny; replace with your real RAG stack later if desired.
"""
from __future__ import annotations
import os
import glob
from typing import List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False


class KnowledgeBase:
    def __init__(self, paths: List[str] | None = None, char_limit: int = 1200):
        self.paths = paths or []
        self.char_limit = char_limit
        self.docs: List[Tuple[str, str]] = []  # (path, text)
        if self.paths:
            self._load()

    def _load(self):
        patterns = []
        for p in self.paths:
            patterns += [
                os.path.join(p, "**", "*.md"),
                os.path.join(p, "**", "*.txt"),
                os.path.join(p, "**", "*.json"),
                os.path.join(p, "**", "*.csv"),
                os.path.join(p, "**", "*.py"),
            ]
        files = []
        for pat in patterns:
            files += glob.glob(pat, recursive=True)
        for fp in sorted(set(files)):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                # Cheap collapses
                txt = txt.replace("\r", " ").replace("\n", " ")
                self.docs.append((fp, txt))
            except Exception:
                continue

    def query(self, question: str, k: int = 3) -> str:
        if not self.docs or not _HAS_SK:
            return ""
        corpus = [d[1] for d in self.docs]
        vec = TfidfVectorizer(stop_words="english", max_df=0.9)
        try:
            X = vec.fit_transform(corpus + [question])
        except Exception:
            return ""
        sim = cosine_similarity(X[-1], X[:-1]).flatten()
        idx = sim.argsort()[::-1][:k]
        chunks: List[str] = []
        size = 0
        for i in idx:
            t = corpus[i]
            take = t[: min(len(t), self.char_limit - size)]
            if take:
                chunks.append(take)
                size += len(take)
            if size >= self.char_limit:
                break
        return "\n---\n".join(chunks)
