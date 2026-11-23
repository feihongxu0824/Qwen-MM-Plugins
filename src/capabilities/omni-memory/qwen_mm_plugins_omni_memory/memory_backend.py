"""The query contract; JSON is a first-class implementation, not an error fallback."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

try:
    from .memory_query import Selection, project
    from .retrieval_text import kw
except ImportError:
    from memory_query import Selection, project
    from retrieval_text import kw


class MemoryBackend(ABC):
    @abstractmethod
    def scan(self, pool, selection=Selection()):
        """Return all matching projected fields, ordered by original ordinal."""

    @abstractmethod
    def sparse(self, pool, query, selection=Selection(), *, k1=1.5, b=0.75, boost=2.0, limit=500):
        """Return ranked original ordinals, bounded by limit."""

    @abstractmethod
    def dense(self, pool, vector, selection=Selection(), *, limit=500):
        """Return ranked original ordinals, bounded by limit."""

    def close(self):
        pass

    def validate(self):
        pass


class JsonBackend(MemoryBackend):
    def __init__(self, snapshot=None, *, rows=None):
        self.rows = project(snapshot) if rows is None else rows
        # Decode once, as the legacy JSON store does; repeated queries reuse float32 arrays.
        for records in self.rows.values():
            for row in records:
                if row["vector"] is not None:
                    row["vector"] = np.asarray(row["vector"], dtype=np.float32)

    def _selected(self, pool, selection):
        return [r for r in self.rows[pool] if selection.matches(r["fields"])]

    def scan(self, pool, selection=Selection()):
        return [r["fields"] for r in self._selected(pool, selection)]

    def sparse(self, pool, query, selection=Selection(), *, k1=1.5, b=0.75, boost=2.0, limit=500):
        records = self._selected(pool, selection)
        terms = kw(query)
        if not terms or not records:
            return []
        scores = [0.0] * len(records)
        for field, weight in (("text", 1.0), ("names", boost)):
            if field == "names" and pool != "episodes":
                continue
            tfs = [Counter(r["fields"][field].split()) for r in records]
            lengths = [sum(tf.values()) for tf in tfs]
            avg = sum(lengths) / len(lengths) or 1.0
            df = Counter(term for tf in tfs for term in terms & tf.keys())
            for i, tf in enumerate(tfs):
                for term, count in df.items():
                    f = tf.get(term, 0)
                    if f:
                        idf = math.log(1 + (len(records) - count + 0.5) / (count + 0.5))
                        scores[i] += weight * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lengths[i] / avg))
        ranked = sorted(
            ((r["fields"]["ordinal"], s) for r, s in zip(records, scores) if s > 0),
            key=lambda p: (-p[1], p[0]),
        )
        return [i for i, _ in ranked[:limit]]

    def dense(self, pool, vector, selection=Selection(), *, limit=500):
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or not query.size or not np.isfinite(query).all() or not np.isfinite(np.linalg.norm(query)):
            raise ValueError("query embedding must be a finite nonempty vector")
        scored = []
        for row in self._selected(pool, selection):
            if row["vector"] is None:
                continue
            v = np.asarray(row["vector"], dtype=np.float32)
            if v.shape != query.shape:
                raise ValueError("query embedding dimension does not match stored vectors")
            norm = np.linalg.norm(v) * np.linalg.norm(query)
            score = float(v @ query / norm) if norm else 0.0
            scored.append((row["fields"]["ordinal"], score))
        return [i for i, _ in sorted(scored, key=lambda p: (-p[1], p[0]))[:limit]]


def open_backend(directory, snapshot):
    backend = (snapshot.get("storage") or {}).get("backend", "json")
    if backend == "json":
        return JsonBackend(snapshot)
    if backend != "zvec":
        raise ValueError(f"unsupported memory backend: {backend}")
    try:
        from .zvec_backend import ZvecBackend
    except ImportError:
        from zvec_backend import ZvecBackend
    return ZvecBackend(directory, snapshot["storage"])
