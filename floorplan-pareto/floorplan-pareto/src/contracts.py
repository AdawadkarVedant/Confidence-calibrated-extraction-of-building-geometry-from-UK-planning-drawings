"""Shared data types for the extraction pipeline.

Rule: every stage must return a confidence score. No stage can just
"fail" silently. This confidence is what the optimiser tunes later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Stage(str, Enum):
    """The four pipeline stages, in the order they run."""

    TRIAGE = "triage"          # is this page a floor plan?
    SCALE = "scale"            # what is the drawing scale?
    GEOMETRY = "geometry"      # ground-floor footprint, storeys, room count
    LINK = "link"              # match to UPRN / EPC record

    @classmethod
    def ordered(cls) -> list["Stage"]:
        return [cls.TRIAGE, cls.SCALE, cls.GEOMETRY, cls.LINK]


@dataclass
class StageOutput:
    """One stage's result for one record."""

    stage: Stage
    confidence: float                     # must be 0-1
    payload: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"{self.stage.value} emitted confidence {self.confidence}, "
                "expected [0, 1]"
            )


@dataclass
class PropertyRecord:
    """One planning application as it moves through the pipeline."""

    record_id: str
    application_ref: str = ""            # e.g. "25/1223/FUL"
    source_pdf: str = ""
    stage_outputs: dict[Stage, StageOutput] = field(default_factory=dict)
    truth: dict[str, Any] = field(default_factory=dict)
    truth_source: str = ""               # "epc" | "hand" | ""

    def confidence(self, stage: Stage) -> float:
        """Confidence for one stage. 0.0 if that stage never ran (= abstained)."""
        out = self.stage_outputs.get(stage)
        return out.confidence if out else 0.0

    def confidence_vector(self) -> list[float]:
        """All four confidences, in stage order."""
        return [self.confidence(s) for s in Stage.ordered()]

    def prediction(self, key: str, default: Any = None) -> Any:
        """Look up one value (e.g. "scale_denominator") from any stage's output."""
        for out in self.stage_outputs.values():
            if key in out.payload:
                return out.payload[key]
        return default

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict, ready for json.dump."""
        return {
            "record_id": self.record_id,
            "application_ref": self.application_ref,
            "source_pdf": self.source_pdf,
            "truth": self.truth,
            "truth_source": self.truth_source,
            "stage_outputs": {
                s.value: asdict(o) | {"stage": s.value}
                for s, o in self.stage_outputs.items()
            },
        }


# How expensive it is for a human reviewer to check an abstention at each
# stage. Bigger number = more reviewer time.
REVIEW_COST: dict[Stage, float] = {
    Stage.TRIAGE: 1.0,
    Stage.SCALE: 2.0,
    Stage.GEOMETRY: 5.0,
    Stage.LINK: 3.0,
}

# Cost of one manual survey (used when a record abstains completely).
MANUAL_SURVEY_COST = 1.0

# Price of one REVIEW_COST unit, same currency as MANUAL_SURVEY_COST.
REVIEW_EFFORT_PRICE = 0.05
