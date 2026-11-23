"""MCP tool modules. Each module exports TOOL and handle; discovered at startup by build_registry.

MemoryRef lives here rather than in a helper module because every tool but watch_and_answer starts
from it: the two fields are how a memory is addressed, and they mirror service.memory_dir's rules.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryRef(BaseModel):
    """Every tool locates a memory by video path, optionally with a namespace."""

    video_path: str | None = None
    namespace: str | None = None


class TimeFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.start_sec is not None and self.end_sec is not None and self.start_sec > self.end_sec:
            raise ValueError("start_sec must not exceed end_sec")
        return self


class DialogueFilter(TimeFilter):
    speaker_ids: list[str] | None = None


class EpisodeFilter(DialogueFilter):
    person_ids: list[str] | None = None


class FactFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_ids: list[str] | None = None
    key_prefix: str | None = None
    status: str | None = None
    type: str | None = None
    confidence: str | None = None


class MemoryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodes: EpisodeFilter | None = None
    facts: FactFilter | None = None
