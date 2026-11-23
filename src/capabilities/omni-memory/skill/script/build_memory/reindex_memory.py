#!/usr/bin/env python3
"""Migrate an existing memory between JSON and zvec without calling a model."""

import argparse
import json

from memory_backend import open_backend
from memory_io import memory_lock, migrate, read_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory_dir")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--backend", choices=("json", "zvec"))
    action.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    if args.backend:
        result = migrate(args.memory_dir, args.backend)
    else:
        with memory_lock(args.memory_dir):
            snapshot = read_snapshot(args.memory_dir)
            backend = open_backend(args.memory_dir, snapshot)
            try:
                result = {
                    **snapshot.get("storage", {"backend": "json"}),
                    "counts": {pool: len(backend.scan(pool)) for pool in ("episodes", "facts", "utterances")},
                }
            finally:
                backend.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
