"""Backend-independent records, predicates and text projection.

Mirrored in the standalone builder; neither tools nor predicates depend on zvec.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

try:
    from .omni_core import StoreBase, emb_decode
    from .retrieval_text import tok_list
except ImportError:  # standalone skill
    from omni_core import StoreBase, emb_decode
    from retrieval_text import tok_list

POOLS = ("episodes", "facts", "utterances")
SOURCE_KEYS = {"episodes": "episodic", "facts": "semantic"}
TEXT_VERSION = 1


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def record_id(pool, identity):
    return digest([pool, identity])


def episode_text(item):
    return StoreBase._ep_text(item)


def embedding_text(pool, item):
    return episode_text(item) if pool == "episodes" else item.get("statement") or ""


def text_hash(pool, item):
    return digest([TEXT_VERSION, embedding_text(pool, item)])


def finite_time(value, default=0.0):
    value = default if value is None else float(value)
    if not math.isfinite(value):
        raise ValueError("timestamps must be finite")
    return value


def utterance_start(rec, utterance):
    base = finite_time(rec.get("win_start"))
    start = utterance.get("start_sec")
    return base if start is None else (finite_time(start) if start >= base else base + finite_time(start))


@dataclass(frozen=True)
class Predicate:
    field: str
    op: str
    value: object

    def matches(self, fields):
        current = fields[self.field]
        if self.op == "eq":
            return current == self.value
        if self.op == "ne":
            return current != self.value
        if self.op == "in":
            return current in self.value
        if self.op == "any":
            return bool(set(current) & set(self.value))
        if self.op == "prefix":
            return current.startswith(self.value)
        if self.op == "ge":
            return current >= self.value
        if self.op == "le":
            return current <= self.value
        raise ValueError(f"unsupported predicate operator: {self.op}")


@dataclass(frozen=True)
class Selection:
    """AND of clauses; each clause is an OR of predicates. No database syntax leaks out."""

    clauses: tuple = ()

    def where(self, *alternatives):
        return Selection(self.clauses + (tuple(alternatives),))

    def matches(self, fields):
        return all(any(p.matches(fields) for p in clause) for clause in self.clauses)


def select(filters=None, pool="episodes"):
    if hasattr(filters, "model_dump"):
        filters = filters.model_dump(exclude_none=True)
    filters = dict(filters or {})
    allowed = {
        "episodes": {"person_ids", "speaker_ids", "start_sec", "end_sec"},
        "facts": {"subject_ids", "key_prefix", "status", "type", "confidence"},
        "utterances": {"speaker_ids", "start_sec", "end_sec"},
    }[pool]
    unknown = filters.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported {pool} filters: {sorted(unknown)}")
    start, end = filters.get("start_sec"), filters.get("end_sec")
    if start is not None and end is not None and finite_time(start) > finite_time(end):
        raise ValueError("start_sec must not exceed end_sec")
    out = Selection()
    for key, value in filters.items():
        if value is None:
            continue
        if key in {"person_ids", "speaker_ids", "subject_ids"}:
            if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
                raise ValueError(f"{key} must be a list of strings")
            field = "speaker_id" if pool == "utterances" else "subject_id" if key == "subject_ids" else key
            out = out.where(Predicate(field, "any" if field.endswith("_ids") else "in", tuple(value)))
        elif key in {"start_sec", "end_sec"}:
            field = ("win_end" if key == "start_sec" else "win_start") if pool == "episodes" else "start_sec"
            out = out.where(Predicate(field, "ge" if key == "start_sec" else "le", finite_time(value)))
        else:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            out = out.where(
                Predicate("key" if key == "key_prefix" else key, "prefix" if key == "key_prefix" else "eq", value)
            )
    return out


def project(snapshot):
    """Create identical query records for both backends, retaining JSON positions for hydration."""
    result = {pool: [] for pool in POOLS}
    for pool, source in SOURCE_KEYS.items():
        seen = set()
        for ordinal, item in enumerate(snapshot.get(source) or []):
            identity = item.get("idx" if pool == "episodes" else "key")
            if pool == "episodes":
                if type(identity) is not int or identity < 0:
                    raise ValueError("episode idx must be a nonnegative integer")
            elif not isinstance(identity, str) or not identity:
                raise ValueError("fact key must be a nonempty string")
            if identity in seen:
                raise ValueError(f"duplicate {pool} identity: {identity!r}")
            seen.add(identity)
            fields = {"ordinal": ordinal}
            if pool == "episodes":
                parsed = item.get("parsed") or {}
                entities = (parsed.get("visual") or {}).get("key_entities") or []
                utterances = (parsed.get("audio") or {}).get("utterances") or []
                speakers = sorted({u["speaker_id"] for u in utterances if u.get("speaker_id")})
                people = sorted({e["person_id"] for e in entities if e.get("person_id")} | set(speakers))
                names = " ".join(" ".join(str(e[k]) for k in ("name", "ref") if e.get(k)) for e in entities)
                fields.update(
                    idx=identity,
                    win_start=finite_time(item.get("win_start"), 1e18),
                    win_end=finite_time(item.get("win_end")),
                    person_ids=people,
                    speaker_ids=speakers,
                    names=" ".join(tok_list(names)),
                )
                for index, utterance in enumerate(utterances):
                    result["utterances"].append(
                        {
                            "id": record_id("utterances", [identity, index]),
                            "fields": {
                                "ordinal": len(result["utterances"]),
                                "clip_idx": identity,
                                "episode_ordinal": ordinal,
                                "utterance_index": index,
                                "speaker_id": utterance.get("speaker_id") or "",
                                "start_sec": round(utterance_start(item, utterance), 1),
                                "end_sec": finite_time(utterance.get("end_sec"), utterance_start(item, utterance)),
                            },
                            "vector": None,
                        }
                    )
            else:
                fields.update(
                    key=identity,
                    subject_id=item.get("subject_id") or "",
                    subject=(item.get("subject") or "").lower(),
                    status=item.get("status") or "active",
                    type=item.get("type") or "",
                    confidence=item.get("confidence") or "",
                )
            vector = emb_decode(item.get("emb"))
            if item.get("emb_text_hash") and item["emb_text_hash"] != text_hash(pool, item):
                vector = None
            if vector is not None:
                vector = np.asarray(vector, dtype=np.float32)
                if (
                    vector.ndim != 1
                    or not vector.size
                    or not np.isfinite(vector).all()
                    or not np.isfinite(np.linalg.norm(vector))
                ):
                    raise ValueError(f"invalid embedding for {pool} {identity!r}")
            fields.update(
                text=" ".join(tok_list(embedding_text(pool, item))),
                text_hash=text_hash(pool, item),
                has_embedding=vector is not None,
                nonzero=bool(vector is not None and np.linalg.norm(vector) > 0),
            )
            result[pool].append(
                {
                    "id": record_id(pool, identity),
                    "fields": fields,
                    "vector": vector.tolist() if vector is not None else None,
                }
            )
    for records in result.values():
        for row in records:
            row["fields"]["fingerprint"] = digest(row)
    return result
