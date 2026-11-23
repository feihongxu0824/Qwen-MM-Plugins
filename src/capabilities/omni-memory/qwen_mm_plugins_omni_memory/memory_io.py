"""Atomic memory snapshots and recoverable, incremental native commits.

The journal spans JSON and the three collections. Queries and commits hold the same OS lock;
native handles are closed before releasing it. Builder ownership is a separate, longer-lived lock.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    from .memory_query import POOLS, SOURCE_KEYS, TEXT_VERSION, project, text_hash
    from .omni_core import emb_encode
except ImportError:
    from memory_query import POOLS, SOURCE_KEYS, TEXT_VERSION, project, text_hash
    from omni_core import emb_encode

_locks = {}
_guard = threading.Lock()
_held = threading.local()
JOURNAL = ".memory-commit.json"


@contextmanager
def memory_lock(directory, name=".memory.lock", *, blocking=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = str((directory / name).resolve())
    with _guard:
        mutex = _locks.setdefault(path, threading.RLock())
    if not mutex.acquire(blocking=blocking):
        raise RuntimeError("another writer is using this memory")
    held = getattr(_held, "paths", None)
    if held is None:
        held = _held.paths = set()
    try:
        if path in held:
            yield
            return
        with open(path, "a+b") as stream:
            if os.name == "nt":
                import msvcrt

                if not stream.tell():
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                except BlockingIOError as exc:
                    raise RuntimeError("another writer is using this memory") from exc
            held.add(path)
            try:
                yield
            finally:
                held.remove(path)
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        mutex.release()


def sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def _native():
    try:
        from . import zvec_backend
    except ImportError:
        import zvec_backend
    return zvec_backend


def _apply(directory, journal):
    _native().apply_changes(directory, journal)
    atomic_json(Path(directory) / "store.json", journal["snapshot"])


def recover(directory):
    """Caller holds memory_lock. Replaying the persisted changes is safe after any write boundary."""
    path = Path(directory) / JOURNAL
    if path.exists():
        journal = json.loads(path.read_text(encoding="utf-8"))
        _apply(directory, journal)
        path.unlink()
        sync_directory(directory)


def read_snapshot(directory, *, include_vectors=False):
    directory = Path(directory)
    if not (directory / "store.json").is_file() and not (directory / JOURNAL).is_file():
        raise FileNotFoundError(f"no memory at {directory} — build it first")
    with memory_lock(directory):
        recover(directory)
        snapshot = json.loads((directory / "store.json").read_text(encoding="utf-8"))
        metadata = snapshot.get("storage") or {}
        if metadata.get("backend", "json") not in {"json", "zvec"}:
            raise ValueError("unsupported memory backend")
        if include_vectors and metadata.get("backend") == "zvec":
            reader = _native().ZvecBackend(directory, metadata)
            try:
                for pool, source in SOURCE_KEYS.items():
                    for ordinal, vector in reader.embeddings(pool).items():
                        snapshot[source][ordinal]["emb"] = emb_encode(vector)
            finally:
                reader.close()
        return snapshot


def commit_snapshot(directory, snapshot, backend=None):
    """Commit a caller-owned snapshot. Vectors enter as lists/base64, never as native SDK objects."""
    directory = Path(directory)
    with memory_lock(directory):
        recover(directory)
        previous = json.loads((directory / "store.json").read_text()) if (directory / "store.json").exists() else {}
        selected = backend or (snapshot.get("storage") or previous.get("storage") or {}).get("backend", "json")
        if selected not in {"json", "zvec"}:
            raise ValueError(f"unsupported memory backend: {selected}")
        disk = copy.deepcopy(snapshot)
        for pool, source in SOURCE_KEYS.items():
            for record in disk.get(source) or []:
                fingerprint = text_hash(pool, record)
                if record.get("emb_text_hash") and record["emb_text_hash"] != fingerprint:
                    record.pop("emb", None)
                    disk["dense_ok"] = False
                elif record.get("emb") is not None:
                    record["emb_text_hash"] = fingerprint
        # The former optional-index PR used retrieval; records remain usable during migration.
        disk.pop("retrieval", None)
        old = previous.get("storage") or {}
        metadata = {
            "backend": selected,
            "schema_version": 1,
            "revision": old.get("revision", 0) + 1,
            "text_version": TEXT_VERSION,
            "embedding_model": snapshot.get("embedding_model"),
        }
        disk.update(version=4, storage=metadata)
        rows = project(disk)
        dimensions = {len(row["vector"]) for pool in POOLS for row in rows[pool] if row["vector"] is not None}
        if len(dimensions) > 1:
            raise ValueError("mixed embedding dimensions in memory")
        dimension = next(iter(dimensions), old.get("dimension", 0))
        metadata.update(
            dimension=dimension,
            counts={p: len(rows[p]) for p in POOLS},
            zero_counts={
                p: sum(r["fields"]["has_embedding"] and not r["fields"]["nonzero"] for r in rows[p])
                for p in SOURCE_KEYS
            },
        )
        if selected == "json":
            atomic_json(directory / "store.json", disk)
            return metadata
        native = _native()
        metadata["schema_version"] = native.SCHEMA_VERSION
        changes = native.prepare_changes(directory, rows, metadata, old)
        for source in SOURCE_KEYS.values():
            for record in disk.get(source) or []:
                record.pop("emb", None)
        journal = {"snapshot": disk, "changes": changes}
        atomic_json(directory / JOURNAL, journal)
        _apply(directory, journal)
        (directory / JOURNAL).unlink()
        sync_directory(directory)
        return metadata


def migrate(directory, backend):
    """Switch explicitly; existing vectors round-trip without an embedding API call."""
    with memory_lock(directory, ".writer.lock", blocking=False):
        with memory_lock(directory):
            snapshot = read_snapshot(directory, include_vectors=True)
            return commit_snapshot(directory, snapshot, backend)
