"""Read-only service boundary over the memory core — what MORE THAN ONE tool needs.

Locating a memory, loading it, and the few helpers several tools share. A helper only one tool uses
lives in that tool's own module instead, so this file stays the answer to "what do the tools have in
common" rather than a drawer for everything that is not a schema.

Building is NOT here — it is a long, serial, stateful job and lives in skill/script/build_memory/,
driven by the agent through Bash. This module only reads memories that already exist, so the MCP
server stays stateless and cacheable.

Memory location, either form:
  · video_path  → <video_path>.memory/store.json        (per-video, preferred; matches video-memory)
  · namespace   → $MEM_LOCAL_DIR/<namespace>/store.json when configured, otherwise
                  <video-dir>/<namespace>/store.json (several videos may stream into one memory)
"""

from __future__ import annotations

import inspect
import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any

from . import config, mem_core, omni_core
from .memory_backend import open_backend
from .memory_io import memory_lock, read_snapshot


def _environment_tuning_names(environment) -> tuple[str, ...]:
    """Return configured MEM_* names without retaining values that may contain credentials."""
    return tuple(sorted(k for k in environment if k.startswith("MEM_")))


# Uncatalogued MEM_* knobs the environment set; on_start reports their names because they change
# retrieval or determinism without surfacing anywhere else. Values are deliberately never retained
# for logging: historical or third-party variables under this broad prefix may contain credentials.
ENV_TUNING = _environment_tuning_names(os.environ)

MEMORY_SUFFIX = ".memory"
DEFAULT_OMNI_MODEL = omni_core.MODEL  # what an unspecified `model` resolves to, for tool descriptions

_stores: dict[str, tuple[tuple[int, int, int], Any]] = {}  # path → (file signature, MemoryStore)
_lock = threading.Lock()
_tl = threading.local()


# ─────────────────────────────────────────────────────────── locating a memory


def memory_dir(video_path: str | None = None, namespace: str | None = None) -> Path:
    if namespace:
        ns = str(namespace).strip().strip("/")
        if not ns or ns in {".", ".."} or "/" in ns or "\\" in ns:
            raise ValueError("namespace must be a simple non-empty name")
        root = config.local_dir()
        if root:
            return Path(root) / ns
        if video_path:
            return Path(video_path).expanduser().resolve().parent / ns
        raise ValueError("namespace requires video_path when MEM_LOCAL_DIR is not configured")
    if video_path:
        return Path(str(Path(video_path).expanduser().resolve()) + MEMORY_SUFFIX)
    raise ValueError("pass either video_path or namespace")


def memory_label(video_path: str | None, namespace: str | None) -> str:
    return namespace or (Path(video_path).name if video_path else "?")


def library_namespaces() -> list[str]:
    """Namespaces under the shared library root. Only used to make a not-found status actionable —
    per-video memories are located by video_path, so there is no listing tool."""
    root = Path(config.local_dir())
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "store.json").is_file())


# ─────────────────────────────────────────────────────────── loading


def load_store(video_path: str | None = None, namespace: str | None = None):
    """Cache business records; backend handles live only within a tool's read transaction."""
    directory = memory_dir(video_path, namespace)
    path = directory / "store.json"
    if not path.is_file() and not (directory / ".memory-commit.json").is_file():
        raise FileNotFoundError(f"no memory at {directory} — build it first")
    with memory_lock(directory):
        if path.is_file() and not (directory / ".memory-commit.json").is_file():
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            with _lock:
                hit = _stores.get(str(path))
                if hit and hit[0] == signature:
                    return hit[1]
        snapshot = read_snapshot(directory)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        key = str(path)
        with _lock:
            hit = _stores.get(key)
            if hit and hit[0] == signature:
                return hit[1]
            store = mem_core.MemoryStore.from_dict(snapshot)
            store._backend = open_backend(directory, snapshot)
            _stores[key] = (signature, store)
            return store


def memory_read(function):
    """Keep JSON and all native reads at one committed revision for the entire public operation."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        directory = memory_dir(bound.arguments.get("video_path"), bound.arguments.get("namespace"))
        if not (directory / "store.json").is_file() and not (directory / ".memory-commit.json").is_file():
            return function(*args, **kwargs)
        with memory_lock(directory):
            store = load_store(bound.arguments.get("video_path"), bound.arguments.get("namespace"))
            try:
                store.backend.validate()
                return function(*args, **kwargs)
            finally:
                store.backend.close()

    return wrapped


def preload() -> None:
    """Warm the cache for a fixed library root (called from on_start in a daemon thread)."""
    for ns in library_namespaces()[:8]:
        try:
            load_store(namespace=ns)
        except Exception:
            pass


def embed_client():
    """Client used ONLY to embed the query for dense retrieval. None if no key is configured."""
    settings = config.embed_config()
    cached = getattr(_tl, "embed", None)
    if cached is None or cached[0] != settings:
        client = omni_core.get_embed_client()
        if cached is not None and cached[1] is not None:
            cached[1].close()
        cached = _tl.embed = (settings, client)
    return cached[1]


# ───────────────────────────────── dialogue: read by get_people and get_person_dialogue


def clip_utterances(store, rec: dict) -> list[dict]:
    """Expand ONE clip's utterances: absolute time + resolved speaker name.

    Per-clip `start_sec` may be either absolute or relative to the window, so it is normalised
    against the window start. Speaker attribution is the omni model's own (prompt Part B6: each line
    is bound to a canonical person_id using lip movement + who is visibly speaking) — a pure read,
    no ASR involved.
    """
    aud = (rec.get("parsed") or {}).get("audio") or {}
    base = rec.get("win_start", 0) or 0
    out = []
    for u in aud.get("utterances") or []:
        sid = u.get("speaker_id")
        t0 = u.get("start_sec")
        t0 = base if t0 is None else (t0 if t0 >= base else base + t0)
        out.append(
            {
                "clip_idx": rec.get("idx"),
                "start_sec": round(t0, 1),
                "speaker_id": sid,
                "speaker_name": store.name_of(sid) if sid else None,
                "text": u.get("text") or "",
                "paralinguistic": u.get("paralinguistic") or None,
            }
        )
    return out


def utterances(
    store, person_id: str | None = None, start_sec: float | None = None, end_sec: float | None = None
) -> list[dict]:
    """Every utterance in the video, optionally filtered by speaker and/or time window."""
    filters = {}
    if person_id:
        filters["speaker_ids"] = [person_id]
    if start_sec is not None:
        filters["start_sec"] = start_sec
    if end_sec is not None:
        filters["end_sec"] = end_sec
    out = dialogue_from_rows(store, store.dialogue_rows(filters))
    out.sort(key=lambda x: (x["start_sec"], x["clip_idx"] or 0))
    return out


# ──────────── one clip in brief: read by get_timeline, search_memory and plan_and_search


def moment_brief(store, rec: dict) -> dict[str, Any]:
    vis = (rec.get("parsed") or {}).get("visual") or {}
    aud = (rec.get("parsed") or {}).get("audio") or {}
    utt = aud.get("utterances") or []
    return {
        "idx": rec.get("idx"),
        "win_start": rec.get("win_start"),
        "win_end": rec.get("win_end"),
        "visual_caption": (vis.get("visual_caption") or "")[:400],
        "people": [e.get("person_id") for e in (vis.get("key_entities") or [])],
        "utterance_count": len(utt),
        "first_line": (utt[0].get("text") if utt else None),
    }


def dialogue_from_rows(store, rows):
    out = []
    for row in rows:
        rec = store.episodic[row["episode_ordinal"]]
        u = rec["parsed"]["audio"]["utterances"][row["utterance_index"]]
        sid = u.get("speaker_id")
        out.append(
            {
                "clip_idx": rec.get("idx"),
                "start_sec": round(row["start_sec"], 1),
                "speaker_id": sid,
                "speaker_name": store.name_of(sid) if sid else None,
                "text": u.get("text") or "",
                "paralinguistic": u.get("paralinguistic") or None,
            }
        )
    return out
