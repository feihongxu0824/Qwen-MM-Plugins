"""The store as the build WRITES it: the setters, the vector codec on the way out, and the index build.

Retrieval is not here. It lives on the query side, in the server package's mem_core.MemoryStore. Both
classes carry the name MemoryStore and both extend omni_core.StoreBase, and StoreBase is what makes a
store written by this one readable by that one: the fields, and the from_dict that reads them back,
are defined once for both.

Nothing on this side ranks, searches or answers — a build only ever appends and re-indexes.
"""

import copy

import numpy as np
from env_config import embed_config
from memory_query import SOURCE_KEYS, text_hash
from omni_core import EMB_DTYPE, StoreBase, emb_encode, embed_texts, get_embed_client


def _emb_out(d):
    """Shallow copy of a record with its vector encoded for storage.

    Always copies, even when there is no vector: snapshot_for_async relies on the per-dict copy to
    isolate the top-level in-place mutations that happen while the snapshot is being serialized off
    the main thread."""
    if d.get("emb") is None:
        return dict(d)
    return {**d, "emb": emb_encode(d["emb"])}


class MemoryStore(StoreBase):
    """StoreBase plus everything a build needs to fill it in and index it."""

    def add_episodic(self, rec):
        self.episodic.append(rec)

    def set_semantic(self, triples):
        self.semantic = [t for t in (triples or []) if t.get("status") != "superseded"]

    def set_clips(self, clips):
        self.clips = [
            {
                "idx": c["idx"],
                "path": c.get("path"),
                "win_start": c["win_start"],
                "win_end": c["win_end"],
                "oss_key": c.get("oss_key"),
            }
            for c in (clips or [])
        ]

    def add_clips(self, clips):
        """Append clip pointers (incremental multi-upload); keeps existing ones."""
        self.clips = list(self.clips) + [
            {
                "idx": c["idx"],
                "path": c.get("path"),
                "win_start": c["win_start"],
                "win_end": c["win_end"],
                "oss_key": c.get("oss_key"),
            }
            for c in (clips or [])
        ]

    def clip_idx_next(self):
        """Next global clip index (0 if empty)."""
        return max((c.get("idx", -1) for c in self.clips), default=-1) + 1

    def time_offset(self):
        """Global time offset for the next segment = video seconds already ingested."""
        if self.processed_sec:
            return float(self.processed_sec)
        return float(max((e.get("win_end", 0) or 0 for e in self.episodic), default=0.0))

    def clear(self):
        self.__init__()

    def snapshot_for_async(self):
        """Freeze a serializable snapshot, including nested text that name alignment can change."""
        return copy.deepcopy(
            {
                **self.extra,
                "version": 4,
                "storage": dict(self.storage),
                "embedding_model": self.embedding_model,
                "scene_env": list(self.scene_env),
                "entities": [dict(e) for e in self.entities],
                "episodic": [_emb_out(r) for r in self.episodic],
                "semantic": [_emb_out(t) for t in self.semantic],
                "clips": [dict(c) for c in self.clips],
                "driver_state": self.driver_state,
                "processed_sec": self.processed_sec,
                "dense_ok": self.dense_ok,
                "name_ledger": dict(self.name_ledger),
            }
        )

    # ---------- dense index ----------
    @classmethod
    def from_dict(cls, data):
        store = super().from_dict(data)
        store._embedding_cache = {}
        for pool, source in SOURCE_KEYS.items():
            for record in getattr(store, source):
                if record.get("emb") is not None:
                    fingerprint = record.get("emb_text_hash") or text_hash(pool, record)
                    record["emb_text_hash"] = fingerprint
                    identity = record.get("idx" if pool == "episodes" else "key")
                    store._embedding_cache[(pool, identity)] = (fingerprint, record["emb"])
        return store

    def build_index(self, client):
        """Reuse unchanged vectors; never retain an embedding after its input text changes."""
        model = embed_config()[1]
        retained = [r for source in SOURCE_KEYS.values() for r in getattr(self, source) if r.get("emb") is not None]
        if retained and self.embedding_model is not None and self.embedding_model != model:
            self.dense_ok = False
            raise ValueError("embedding model changed; rebuild explicitly before mixing vectors")
        known_model = self.embedding_model
        cache = getattr(self, "_embedding_cache", {})
        needs = []
        for pool, source in SOURCE_KEYS.items():
            for record in getattr(self, source):
                fingerprint = text_hash(pool, record)
                identity = record.get("idx" if pool == "episodes" else "key")
                cached = cache.get((pool, identity))
                old_hash = record.get("emb_text_hash")
                if record.get("emb") is not None and old_hash is not None and old_hash != fingerprint:
                    record["emb"] = None
                if record.get("emb") is None and cached and cached[0] == fingerprint:
                    record["emb"] = cached[1]
                if record.get("emb") is None:
                    text = self._ep_text(record) if pool == "episodes" else record.get("statement") or ""
                    needs.append((pool, identity, record, fingerprint, text))
                else:
                    record["emb_text_hash"] = fingerprint
        vectors = embed_texts(get_embed_client(), [entry[4] for entry in needs]) if needs else []
        if vectors is None or len(vectors) != len(needs):
            self.dense_ok = False
            return False
        for (pool, identity, record, fingerprint, _), vector in zip(needs, vectors):
            record["emb"] = np.asarray(vector, dtype=EMB_DTYPE)
            record["emb_text_hash"] = fingerprint
            cache[(pool, identity)] = (fingerprint, record["emb"])
        self._embedding_cache = cache
        if not retained or known_model == model:
            self.embedding_model = model
        self.dense_ok = True
        return True
