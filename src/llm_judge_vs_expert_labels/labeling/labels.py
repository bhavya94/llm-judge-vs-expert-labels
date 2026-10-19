"""A person's Pass/Fail labels, kept in an append-only JSON lines file: relabeling an answer adds a row, and the
newest row wins."""

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_judge_vs_expert_labels.labeling.items import Item

VERDICTS = ("pass", "fail")


@dataclass(frozen=True)
class Label:
    candidate_id: str
    verdict: str
    note: str
    """What went wrong, in the labeler's words: the first problem a reader would hit. Free text, so failure modes come
    from the answers rather than from a list made up beforehand."""
    question_id: str
    answer: str
    """The labeled answer, so the file stands on its own."""
    labeled_at: str

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, not {self.verdict!r}")
        if self.verdict == "fail" and not self.note:
            raise ValueError("a Fail needs a note saying what went wrong")

    @classmethod
    def of(cls, item: Item, verdict: str, note: str) -> "Label":
        return cls(item.id, verdict, note.strip(), item.question_id, item.answer,
                   datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"))

    def to_row(self) -> dict:
        return asdict(self)


class LabelStore:
    def __init__(self, path: Path):
        self.path = path

    def save(self, label: Label) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(label.to_row(), ensure_ascii=False) + "\n")

    def latest(self) -> dict[str, Label]:
        """The newest label of each answer."""
        if not self.path.exists():
            return {}
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return {row["candidate_id"]: Label(**row) for row in rows}
