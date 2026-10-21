"""The summaries where a judge's Pass/Fail differs from the human majority label, and a page to say who was right in
each. The page reuses the labeling server: a review is a label whose value is who was right."""

import datetime
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_judge_vs_expert_labels.data.qags import Summary
from llm_judge_vs_expert_labels.judges.base import Judge, Verdict
from llm_judge_vs_expert_labels.labeling.labels import LabelStore
from llm_judge_vs_expert_labels.labeling.server import CONFIG, BadLabel, LabelingServer, render
from llm_judge_vs_expert_labels.llm.client import NotCached

TEMPLATE = Path(__file__).with_name("template.html")
RIGHT = ("human", "judge", "unclear")


@dataclass(frozen=True)
class Disagreement:
    summary: Summary
    verdict: Verdict

    @property
    def id(self) -> str:
        return self.summary.id

    def to_row(self) -> dict:
        """What the page shows. Nothing in it gives the summary's position in the split."""
        s = self.summary
        return {"id": s.id, "dataset": s.dataset, "article": s.article,
                "sentences": [{"text": t.text, "votes": t.votes} for t in s.sentences],
                "human_label": "pass" if s.consistent else "fail",
                "judge_verdict": "pass" if self.verdict.passed else "fail", "reason": self.verdict.reason}


def find_disagreements(summaries: list[Summary], judge: Judge) -> tuple[list[Disagreement], int]:
    """The summaries whose verdict differs from the human majority label, in order, and how many were skipped because
    the judge has no cached verdict. A prompted judge with no client reads its cache only, so this spends nothing."""
    kept, skipped = [], 0
    for summary in summaries:
        try:
            verdict = judge.judge(summary)
        except NotCached:
            skipped += 1
            continue
        if verdict.passed != summary.consistent:
            kept.append(Disagreement(summary, verdict))
    return kept, skipped


@dataclass(frozen=True)
class Review:
    summary_id: str
    dataset: str
    human_label: str
    judge_verdict: str
    right: str
    """human, judge or unclear."""
    note: str
    """Why, in the reviewer's words. Free text, so failure modes come from the summaries, not a list made up first."""
    reviewed_at: str

    def __post_init__(self):
        if self.right not in RIGHT:
            raise ValueError(f"right must be one of {RIGHT}, not {self.right!r}")
        if not self.note:
            raise ValueError("every review needs a note saying why")

    @classmethod
    def of(cls, item: Disagreement, right: str, note: str) -> "Review":
        row = item.to_row()
        return cls(item.id, row["dataset"], row["human_label"], row["judge_verdict"], right, note.strip(),
                   datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"))

    def to_row(self) -> dict:
        return asdict(self)


class ReviewStore(LabelStore):
    """One judge's reviews. Append-only like the labels: the newest review of a summary wins."""

    @classmethod
    def from_config(cls, judge: str, path: Path = CONFIG) -> "ReviewStore":
        return cls((path.parent / tomllib.loads(path.read_text())["review"].format(judge=judge)).resolve())

    def latest(self) -> dict[str, Review]:
        if not self.path.exists():
            return {}
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return {row["summary_id"]: Review(**row) for row in rows}


class ReviewServer(LabelingServer):
    """The labeling server with the review page. It serves the same way: one save at a time, on 127.0.0.1 only."""

    def __init__(self, items: list[Disagreement], reviews: ReviewStore):
        super().__init__(items, reviews, blind=False)

    def page(self) -> str:
        """Only the disagreements with no saved review, in order. The page saves them in order, so a reopened page
        resumes after the last one saved, and a review saved in a one-dataset run never hides an earlier summary."""
        saved = self.labels.latest()
        pending = [self.items[i].to_row() for i in self.order if i not in saved]
        data = {"items": pending, "done": len(self.order) - len(pending), "total": len(self.order)}
        return render(data, TEMPLATE.read_text())

    def label(self, body: dict) -> Review:
        """Saves one review from the page's {"summary_id", "right", "note"}."""
        sid = body.get("summary_id") if isinstance(body, dict) else None
        if not isinstance(sid, str) or (item := self.items.get(sid)) is None:
            raise BadLabel("unknown summary_id")
        try:
            review = Review.of(item, body.get("right"), str(body.get("note") or ""))
        except ValueError as error:
            raise BadLabel(str(error)) from None
        self.labels.save(review)
        return review


def summarize_reviews(reviews) -> dict[str, dict]:
    """Per dataset: how often each side was right, and every note with its side, in the order saved."""
    by_dataset: dict[str, dict] = {}
    for r in sorted(reviews, key=lambda r: r.dataset):
        group = by_dataset.setdefault(r.dataset, {"counts": dict.fromkeys(RIGHT, 0), "notes": []})
        group["counts"][r.right] += 1
        group["notes"].append((r.right, r.note))
    return by_dataset
