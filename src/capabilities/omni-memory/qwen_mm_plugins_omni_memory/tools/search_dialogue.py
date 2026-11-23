from __future__ import annotations

from typing import Any

from pydantic import Field

from shared.content import json_text

from ..service import dialogue_from_rows, embed_client, load_store, memory_label, memory_read
from . import DialogueFilter, MemoryRef


class SearchDialogueArgs(MemoryRef):
    query: str
    filters: DialogueFilter | None = None
    top_k: int = Field(default=8, ge=1, le=20)


TOOL = {"name": "search_dialogue", "args": SearchDialogueArgs}


@memory_read
def search_dialogue(
    video_path: str | None = None,
    namespace: str | None = None,
    query: str = "",
    top_k: int = 8,
    filters: dict | None = None,
) -> dict[str, Any]:
    """Dialogue search that keeps speaker attribution — the thing a separate ASR index cannot do."""
    store = load_store(video_path, namespace)
    eps, rows = store.search_dialogue(query, client=embed_client(), k=max(1, min(int(top_k), 20)), filters=filters)
    lines = dialogue_from_rows(store, rows)
    lines.sort(key=lambda x: (x["start_sec"], x["clip_idx"] or 0))
    return {"label": memory_label(video_path, namespace), "matched_clips": len(eps), "utterances": lines}


def handle(arguments: dict[str, Any]) -> list[dict[str, str]]:
    """Search spoken lines and get them back WITH the speaker attached (person_id + resolved name). Use
    it when the question is about who said what, or to find the moment a topic was discussed. Unlike
    a standalone ASR index, every line here is already bound to a person that persists across the
    whole video.

    Args:
        video_path: Absolute path to the source video; memory is read from <video_path>.memory/.
        namespace: Memory name. Pass video_path too when MEM_LOCAL_DIR is unset; otherwise the
            memory is read from the configured shared root.
        query: What was said, phrased as a statement (e.g. 'someone offers to help with the
            dishes'), not as a question.
        top_k: Maximum number of matching results to return.
        filters: Optional hard constraints, applied before retrieval and to all expanded results.
    """
    return [json_text(search_dialogue(**arguments))]
