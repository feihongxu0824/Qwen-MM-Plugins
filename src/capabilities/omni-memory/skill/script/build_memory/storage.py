"""Storage backend: local filesystem.

A memory library is a namespace laid out as:
    <root>/<ns>/store.json   (business records and backend/version metadata)
    <root>/<ns>/zvec/        (optional episodes/facts/utterances collections, including vectors)
    <root>/<ns>/meta.json    (light preview for listing / ns_info)
    <root>/<ns>/clips/win_NNN.mp4   (original-resolution clips for replay)

Interface used by pipeline: available, status_line, save, save_async, load,
list_namespaces, meta, delete, clip_key, upload_clip_async, download_by_key.

Memories are stored on local disk, next to the video by default.
"""

import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import env_config as config
from memory_io import atomic_json, commit_snapshot, memory_lock, read_snapshot
from store_writer import MemoryStore


def select_retrieval_backend(requested, descriptor=None):
    selected = requested or (descriptor or {}).get("backend", "json")
    if selected not in {"json", "zvec"}:
        raise ValueError(f"unsupported memory backend: {selected}")
    return selected


def preflight_retrieval_backend(backend):
    if backend == "zvec":
        from zvec_backend import require_zvec

        require_zvec()


class _LocalBackend:
    """Memory libraries on local disk, rooted at config.local_dir()."""

    def __init__(self, root):
        self.root = root
        self._exec = None
        self._pending = []

    def _submit(self, function, *args):
        future = self._ex().submit(function, *args)
        self._pending.append(future)
        return future

    def drain(self, *, raise_errors=True):
        if self._exec is not None:
            self._exec.shutdown(wait=True)
            self._exec = None
        pending, self._pending = self._pending, []
        errors = []
        for future in pending:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        if errors and raise_errors:
            raise errors[0]

    @contextmanager
    def writer(self, ns):
        with memory_lock(self._ns_dir(ns), ".writer.lock", blocking=False):
            try:
                yield
            finally:
                self.drain(raise_errors=sys.exc_info()[0] is None)

    def _ex(self):
        if self._exec is None:
            self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-save")
        return self._exec

    def _ns_dir(self, ns):
        """Namespace must stay a single directory name — a namespace like "../../x" would otherwise
        write outside the library root. The service layer validates too, but the build path does not
        come through it."""
        name = str(ns).strip().strip("/")
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"invalid namespace: {ns!r}")
        return os.path.join(self.root, name)

    def available(self):
        return True

    def status_line(self):
        return f"💾 本机存储：{self.root}"

    def _meta_dict(self, ns, sd, extra=None):
        m = {
            "namespace": ns,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_clips": len(sd.get("episodic", [])),
            "n_entities": len(sd.get("entities", [])),
            "n_semantic": len(sd.get("semantic", [])),
            "dense_ok": bool(sd.get("dense_ok")),
            "processed_sec": sd.get("processed_sec")
            or float(max((e.get("win_end", 0) or 0 for e in (sd.get("episodic") or [])), default=0)),
        }
        if extra:
            m.update(extra)
        return m

    def _write(self, ns, sd, meta):
        d = self._ns_dir(ns)
        os.makedirs(os.path.join(d, "clips"), exist_ok=True)
        descriptor = commit_snapshot(d, sd)
        atomic_json(os.path.join(d, "meta.json"), self._meta_dict(ns, sd, meta))
        return descriptor

    def save_async(self, ns, store, meta=None):
        snap = store.snapshot_for_async()  # freeze before submitting to the worker
        return self._submit(self._write, ns, snap, meta)

    save_final_async = save_async

    def load(self, ns):
        return MemoryStore.from_dict(read_snapshot(self._ns_dir(ns), include_vectors=True))

    def list_namespaces(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(n for n in os.listdir(self.root) if os.path.exists(os.path.join(self.root, n, "store.json")))

    def clip_key(self, ns, idx):
        return os.path.join(self._ns_dir(ns), "clips", f"win_{int(idx):03d}.mp4")

    def upload_clip_async(self, ns, idx, path):
        dst = self.clip_key(ns, idx)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        return self._submit(shutil.copyfile, path, dst)

    def download_by_key(self, key):
        with open(key, "rb") as f:
            return f.read()


_LOCAL = None


def get_backend(root=None):
    """Return the local-filesystem backend, rebuilt when config.local_dir() changes."""
    global _LOCAL
    if root is not None:
        return _LocalBackend(str(root))
    if _LOCAL is None or _LOCAL.root != config.local_dir():
        _LOCAL = _LocalBackend(config.local_dir())
    return _LOCAL
