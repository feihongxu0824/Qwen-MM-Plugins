"""Native adapter: three collections, filtered ranking, and bounded complete scans."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

try:
    from .memory_backend import JsonBackend, MemoryBackend
    from .memory_query import POOLS, Predicate, Selection
    from .retrieval_text import kw
except ImportError:
    from memory_backend import JsonBackend, MemoryBackend
    from memory_query import POOLS, Predicate, Selection
    from retrieval_text import kw

SCHEMA_VERSION = 1
BATCH = 256


def require_zvec():
    try:
        import zvec
    except ImportError as exc:
        raise RuntimeError("install qwen-mm-plugins[omni-memory-zvec] to open this memory") from exc
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", zvec.__version__)
    if not match or not (0, 8, 1) <= tuple(map(int, match.groups())) < (1, 0, 0):
        raise RuntimeError("this backend requires zvec>=0.8.1,<1")
    return zvec


def expression(selection):
    def atom(p):
        field, op, value = p.field, p.op, p.value
        # Field names come exclusively from the typed projection, never from raw user expressions.
        if not field.replace("_", "").isalnum():
            raise ValueError("invalid filter field")
        if op in {"in", "any"}:
            if not value:
                return "ordinal < 0"
            values = ", ".join(json.dumps(v, ensure_ascii=False) for v in value)
            return f"{field} {'CONTAIN_ANY' if op == 'any' else 'IN'} ({values})"
        if op == "prefix":
            # Prefix is a literal string, not a LIKE expression. Lexical range also handles %/_.
            if not value:
                return "ordinal >= 0"
            chars = list(value)
            while chars and ord(chars[-1]) == 0x10FFFF:
                chars.pop()
            lo = json.dumps(value, ensure_ascii=False)
            if not chars:
                return f"{field} >= {lo}"
            chars[-1] = chr(ord(chars[-1]) + 1)
            hi = json.dumps("".join(chars), ensure_ascii=False)
            return f"({field} >= {lo} AND {field} < {hi})"
        operator = {"eq": "=", "ne": "!=", "ge": ">=", "le": "<="}[op]
        return f"{field} {operator} {json.dumps(value, ensure_ascii=False)}"

    return " AND ".join("(" + " OR ".join(atom(p) for p in clause) + ")" for clause in selection.clauses) or None


def schema(zvec, pool, dimension):
    types = {"ordinal": "INT64", "fingerprint": "STRING"}
    if pool == "utterances":
        types.update(
            clip_idx="INT64",
            episode_ordinal="INT64",
            utterance_index="INT64",
            speaker_id="STRING",
            start_sec="DOUBLE",
            end_sec="DOUBLE",
        )
    else:
        types.update(text="STRING", text_hash="STRING", has_embedding="BOOL", nonzero="BOOL")
        if pool == "episodes":
            types.update(
                idx="INT64",
                win_start="DOUBLE",
                win_end="DOUBLE",
                person_ids="ARRAY_STRING",
                speaker_ids="ARRAY_STRING",
                names="STRING",
            )
        else:
            types.update(
                key="STRING", subject_id="STRING", subject="STRING", status="STRING", type="STRING", confidence="STRING"
            )
    fields = []
    for name, kind in types.items():
        if name in {"text", "names"}:
            index = zvec.FtsIndexParam(tokenizer_name="whitespace", filters=[])
        elif name in {"fingerprint", "text_hash"}:
            index = None
        else:
            index = zvec.InvertIndexParam(enable_range_optimization=kind in {"INT64", "DOUBLE", "STRING"})
        fields.append(zvec.FieldSchema(name, getattr(zvec.DataType, kind), index_param=index))
    vectors = (
        []
        if pool == "utterances"
        else [
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension or 1,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE),
            )
        ]
    )
    return zvec.CollectionSchema(name=pool, fields=fields, vectors=vectors)


def checked(statuses):
    statuses = statuses if isinstance(statuses, list) else [statuses]
    for status in statuses:
        if not status.ok():
            raise RuntimeError(f"zvec write failed: {status}")


def prepare_changes(directory, rows, metadata, old):
    """Compute row deltas without modifying the live collections."""
    zvec = require_zvec()
    directory = Path(directory)
    dimension = metadata["dimension"]
    changes = {}
    for pool in POOLS:
        path = directory / "zvec" / pool
        if path.is_symlink() or (directory / "zvec").is_symlink():
            raise ValueError("zvec paths must not be symlinks")
        reset = (
            not path.exists()
            or old.get("schema_version") != SCHEMA_VERSION
            or old.get("dimension", 0) != dimension
            or old.get("backend") != "zvec"
        )
        existing = {}
        if not reset:
            collection = zvec.open(str(path), zvec.CollectionOption(read_only=True))
            try:
                with collection.iter_docs(output_fields=["fingerprint"], include_vector=False) as docs:
                    existing = {d.id: d.fields["fingerprint"] for d in docs}
            finally:
                collection.close()
        desired = {r["id"]: r for r in rows[pool]}
        changes[pool] = {
            "reset": reset,
            "delete": sorted(existing.keys() - desired.keys()),
            "upsert": [r for r in rows[pool] if existing.get(r["id"]) != r["fields"]["fingerprint"]],
        }
    return changes


def apply_changes(directory, journal):
    """Replay one durable commit; all native handles close before JSON is published."""
    zvec = require_zvec()
    metadata = journal["snapshot"]["storage"]
    root = Path(directory) / "zvec"
    if root.is_symlink():
        raise ValueError("zvec directory must not be a symlink")
    root.mkdir(exist_ok=True)
    for pool in POOLS:
        change = journal["changes"][pool]
        path = root / pool
        if path.is_symlink():
            raise ValueError("zvec collection must not be a symlink")
        if change["reset"] and path.exists():
            shutil.rmtree(path)
        collection = (
            zvec.open(str(path))
            if path.exists()
            else zvec.create_and_open(str(path), schema(zvec, pool, metadata["dimension"]))
        )
        try:
            # Delete is idempotent: only attempt IDs still present after a partial previous replay.
            for offset in range(0, len(change["delete"]), BATCH):
                ids = change["delete"][offset : offset + BATCH]
                present = list(collection.fetch(ids, output_fields=[], include_vector=False))
                if present:
                    checked(collection.delete(present))
            for offset in range(0, len(change["upsert"]), BATCH):
                rows = change["upsert"][offset : offset + BATCH]
                docs = []
                for row in rows:
                    vectors = None
                    if pool != "utterances":
                        # Internal padding only. has_embedding=false excludes it from ranking/export.
                        vectors = {
                            "embedding": row["vector"]
                            if row["vector"] is not None
                            else [0.0] * (metadata["dimension"] or 1)
                        }
                    docs.append(zvec.Doc(id=row["id"], fields=row["fields"], vectors=vectors))
                checked(collection.upsert(docs))
            collection.flush()
            if collection.stats.doc_count != metadata["counts"][pool]:
                raise RuntimeError(f"incomplete zvec commit in {pool}")
        finally:
            collection.close()


class ZvecBackend(MemoryBackend):
    def __init__(self, directory, metadata):
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported zvec memory schema; migrate with a compatible version")
        self.zvec = require_zvec()
        self.root = Path(directory) / "zvec"
        self.metadata = metadata
        self.collections = {}

    def validate(self):
        for pool in ("episodes", "facts", "utterances"):
            self._collection(pool)

    def _collection(self, pool):
        if pool not in self.collections:
            path = self.root / pool
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"missing or unsafe zvec collection: {pool}")
            self.collections[pool] = self.zvec.open(str(path), self.zvec.CollectionOption(read_only=True))
            if self.collections[pool].stats.doc_count != self.metadata["counts"][pool]:
                raise RuntimeError(f"zvec collection count mismatch: {pool}")
        return self.collections[pool]

    def scan(self, pool, selection=Selection()):
        collection = self._collection(pool)
        result = []
        # Ordinals are unique and contiguous in each committed snapshot. Each range contains at
        # most BATCH rows, so native topk cannot truncate it, regardless of native result order.
        for offset in range(0, self.metadata["counts"][pool], BATCH):
            batch = selection.where(Predicate("ordinal", "ge", offset)).where(
                Predicate("ordinal", "le", offset + BATCH - 1)
            )
            result.extend(
                d.fields for d in collection.query(filter=expression(batch), topk=BATCH, include_vector=False)
            )
        return sorted(result, key=lambda f: f["ordinal"])

    def sparse(self, pool, query, selection=Selection(), *, k1=1.5, b=0.75, boost=2.0, limit=500):
        if k1 != 1.5 or b != 0.75:
            # Native FTS does not expose these legacy tuning knobs. Honor explicit overrides
            # using the shared scorer on the natively filtered fields instead of ignoring them.
            rows = {pool: [{"fields": f, "vector": None} for f in self.scan(pool, selection)]}
            return JsonBackend(rows=rows).sparse(pool, query, k1=k1, b=b, boost=boost, limit=limit)
        terms = sorted(kw(query))
        if not terms:
            return []
        collection = self._collection(pool)
        lanes = ("text", "names") if pool == "episodes" and boost else ("text",)
        scores = {}
        for field in lanes:
            for doc in collection.query(
                self.zvec.Query(field_name=field, fts=self.zvec.Fts(match_string=" ".join(terms))),
                filter=expression(selection),
                topk=limit,
                output_fields=["ordinal"],
                include_vector=False,
            ):
                scores[doc.fields["ordinal"]] = 0.0
        if not scores:
            return []
        # Complete both field scores for the bounded candidate union before applying name boost.
        keys = sorted(scores)
        for offset in range(0, len(keys), BATCH):
            chunk = keys[offset : offset + BATCH]
            restricted = selection.where(Predicate("ordinal", "in", tuple(chunk)))
            for field in lanes:
                weight = boost if field == "names" else 1.0
                for doc in collection.query(
                    self.zvec.Query(field_name=field, fts=self.zvec.Fts(match_string=" ".join(terms))),
                    filter=expression(restricted),
                    topk=len(chunk),
                    output_fields=["ordinal"],
                    include_vector=False,
                ):
                    scores[doc.fields["ordinal"]] += float(doc.score) * weight
        return sorted(scores, key=lambda i: (-scores[i], i))[:limit]

    def dense(self, pool, vector, selection=Selection(), *, limit=500):
        query = np.asarray(vector, dtype=np.float32)
        dimension = self.metadata.get("dimension", 0)
        if query.ndim != 1 or not query.size or not np.isfinite(query).all() or not np.isfinite(np.linalg.norm(query)):
            raise ValueError("query embedding must be a finite nonempty vector")
        if not dimension:
            return []
        if query.size != dimension:
            raise ValueError("query embedding dimension does not match stored vectors")
        ready = selection.where(Predicate("has_embedding", "eq", True))
        if not np.linalg.norm(query):
            return [f["ordinal"] for f in self.scan(pool, ready)][:limit]
        collection = self._collection(pool)
        scored = [
            (doc.fields["ordinal"], 1.0 - float(doc.score))
            for doc in collection.query(
                self.zvec.Query(field_name="embedding", vector=query.tolist()),
                filter=expression(ready.where(Predicate("nonzero", "eq", True))),
                topk=limit,
                output_fields=["ordinal"],
                include_vector=False,
            )
        ]
        # True zero vectors have cosine 0 in the JSON backend. Missing embeddings are never ranked.
        zeros = (
            self.scan(pool, ready.where(Predicate("nonzero", "eq", False)))
            if self.metadata["zero_counts"][pool]
            else []
        )
        scored.extend((f["ordinal"], 0.0) for f in zeros[:limit])
        return [i for i, _ in sorted(scored, key=lambda p: (-p[1], p[0]))[:limit]]

    def embeddings(self, pool):
        collection = self._collection(pool)
        ready = self.scan(pool, Selection().where(Predicate("has_embedding", "eq", True)))
        by_ordinal = {f["ordinal"]: f for f in ready}
        result = {}
        # IDs are opaque; scan only scalar fields first, fetching vectors only for genuine values.
        with collection.iter_docs(output_fields=["ordinal", "has_embedding"], include_vector=False) as docs:
            ids = [(d.id, d.fields["ordinal"]) for d in docs if d.fields["has_embedding"]]
        for offset in range(0, len(ids), BATCH):
            chunk = ids[offset : offset + BATCH]
            found = collection.fetch([i for i, _ in chunk], output_fields=[], include_vector=True)
            for doc_id, ordinal in chunk:
                if doc_id not in found or ordinal not in by_ordinal:
                    raise RuntimeError("missing persisted embedding")
                result[ordinal] = found[doc_id].vector("embedding")
        return result

    def close(self):
        for collection in self.collections.values():
            collection.close()
        self.collections.clear()
