"""The contract every judge meets: one summary in, one Pass/Fail verdict out, with the reason in words."""

from dataclasses import dataclass
from typing import Protocol

from llm_judge_vs_expert_labels.data.qags import Summary


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str
    """Why, in words: empty for a plain Pass."""
    score: float | None = None
    """A graded score, higher for more consistent, when the judge has one. Correlation uses it in place of 0 or 1."""


class Judge(Protocol):
    name: str

    def judge(self, summary: Summary) -> Verdict: ...
