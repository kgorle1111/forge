"""LangGraph state shared by all Forge agents."""
import operator
from typing import Annotated, TypedDict


class Concept(TypedDict):
    name: str
    summary: str


class Question(TypedDict):
    prompt: str
    key_points: list[str]


class ForgeState(TypedDict, total=False):
    current_topic: str
    concepts: list[Concept]        # first-principles breakdown, foundational -> advanced
    concept_idx: int               # concept currently being drilled
    angle: int                     # teaching angle; bumps every failure
    lesson: str                    # explanation shown to the user this cycle
    visual_artifact: str           # optional local SVG diagram for visual lessons
    questions: list[Question]
    answers: list[str]
    score: float                   # last quiz score, 0..1
    mastery_level: float           # rolling mean of passing scores
    mastery: bool                  # True only when every concept is passed
    missed: list[str]              # key points flunked last quiz -> targeted re-teach
    source_material: str           # optional user document the curriculum is built from
    language: str                  # WS-LANG: target teaching language; None/absent = existing behavior
    source_manifest: dict          # source docs split into citeable local chunks
    failed_attempts: int
    max_failed_attempts: int       # 0 = never give up (the mandate)
    spaced_repetition_log: Annotated[list[dict], operator.add]
    done: bool
