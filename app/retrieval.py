"""Hybrid retrieval: ChromaDB vector search + BM25 keyword search, fused with RRF and diversified with MMR."""
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .store import Store

_TOKEN = re.compile(r"\w+", re.UNICODE)

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
CANDIDATES = 40
MMR_LAMBDA = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 or t.isdigit()]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_name: str
    kind: str
    page: int
    text: str
    score: float  # cosine similarity to the query (for display)


class _Snapshot:
    """Immutable in-memory BM25 view of all chunks; swapped atomically on reload."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.pos = {r["id"]: i for i, r in enumerate(rows)}
        self.doc_ids = np.array([r["doc_id"] for r in rows]) if rows else np.array([], dtype=str)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(len(rows), dtype=np.float32)
        for i, r in enumerate(rows):
            counts = Counter(tokenize(f"{r['doc_name']} {r['text']}"))
            self.lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.avg_len = float(self.lengths.mean()) if rows else 1.0

    def bm25(self, query: str, allowed: np.ndarray | None) -> np.ndarray:
        n = len(self.rows)
        scores = np.zeros(n, dtype=np.float32)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[i] / self.avg_len)
                scores[i] += idf * tf * (BM25_K1 + 1) / denom
        if allowed is not None:
            scores[~allowed] = 0
        return scores


class Index:
    def __init__(self, store: Store):
        self._store = store
        self._lock = threading.Lock()
        self._snap = _Snapshot([])
        self.reload()

    def reload(self) -> None:
        """Rebuild the keyword index from Chroma (vectors stay in Chroma)."""
        snap = _Snapshot(self._store.iter_chunks())
        with self._lock:
            self._snap = snap

    @property
    def size(self) -> int:
        return len(self._snap.rows)

    def search(self, query_vec: np.ndarray, query_text: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        snap = self._snap
        n = len(snap.rows)
        if n == 0:
            return []
        q = query_vec.astype(np.float32)

        allowed = None
        where = None
        if doc_ids is not None:
            allowed = np.isin(snap.doc_ids, doc_ids)
            if not allowed.any():
                return []
            where = {"doc_id": {"$in": list(doc_ids)}}

        # --- dense ranking from ChromaDB (cosine space)
        res = self._store.collection.query(
            query_embeddings=[q], n_results=min(CANDIDATES, n), where=where, include=["distances"]
        )
        dense_ids: list[str] = res["ids"][0]

        # --- sparse ranking from BM25
        sparse = snap.bm25(query_text, allowed)
        sparse_ids = [snap.rows[int(i)]["id"] for i in np.argsort(-sparse)[:CANDIDATES] if sparse[i] > 0]

        # --- Reciprocal Rank Fusion of the two rankings
        fused: dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(dense_ids):
            fused[cid] += 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            fused[cid] += 1.0 / (RRF_K + rank + 1)
        if not fused:
            return []

        candidates = sorted(fused, key=fused.get, reverse=True)[: CANDIDATES // 2]

        # --- fetch candidate vectors from Chroma for similarity scores and MMR
        got = self._store.collection.get(ids=candidates, include=["embeddings"])
        vectors = {cid: np.asarray(v, dtype=np.float32) for cid, v in zip(got["ids"], got["embeddings"])}
        candidates = [c for c in candidates if c in vectors]
        if not candidates:
            return []
        sim = {c: float(vectors[c] @ q) for c in candidates}

        # --- Maximal Marginal Relevance: relevant but not redundant
        top = max(fused[c] for c in candidates)
        selected: list[str] = []
        remaining = list(candidates)
        while remaining and len(selected) < k:
            best, best_score = None, -math.inf
            for c in remaining:
                relevance = fused[c] / top
                redundancy = max((float(vectors[c] @ vectors[s]) for s in selected), default=0.0)
                score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            remaining.remove(best)

        hits = []
        for cid in selected:
            r = snap.rows[snap.pos[cid]]
            hits.append(Hit(cid, r["doc_id"], r["doc_name"], r["kind"], r["page"], r["text"], sim[cid]))
        return hits
