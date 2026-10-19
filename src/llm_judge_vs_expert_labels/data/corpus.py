"""The exported answers, questions and paragraphs in `data/`, written once by `scripts/export_candidates.py`."""

import json
from pathlib import Path

from llm_judge_vs_expert_labels.data.records import Candidate, Paragraph, Question, Reference, normalize

DATA = Path(__file__).resolve().parents[3] / "data"


def _paper_order(paragraph_id: str) -> tuple[int, int]:
    """`s10p1` comes after `s2p0`: section, then paragraph, as numbers."""
    section, paragraph = paragraph_id[1:].split("p")
    return int(section), int(paragraph)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Corpus:
    """Everything a person or a judge needs to decide whether an answer is right: the question, the references and
    the paper's paragraphs."""

    def __init__(self, questions: list[Question], paragraphs: dict[tuple[str, str], Paragraph], answers: list[dict]):
        self.questions = {q.id: q for q in questions}
        self.paragraphs = paragraphs
        self.answers = answers

    @classmethod
    def load(cls, root: Path = DATA) -> "Corpus":
        questions = [Question(r["question_id"], r["paper_id"], r["paper_title"], r["question"],
                              tuple(Reference(a["text"], a["type"], tuple(a["evidence_ids"])) for a in r["references"]))
                     for r in _rows(root / "questions.jsonl")]
        paragraphs = {(r["paper_id"], r["id"]): Paragraph(r["id"], r["section"], r["text"])
                      for r in _rows(root / "paragraphs.jsonl")}
        return cls(questions, paragraphs, _rows(root / "candidates.jsonl"))

    def candidates(self) -> list[Candidate]:
        """One candidate per distinct (question, answer). The export lists first answers before revisions, so the first
        row of a repeated answer decides its group and the paragraphs its writer saw: a revision that repeats a first
        answer counts as that first answer, and only revisions that changed the answer form the revision group. Every
        later row adds its source."""
        merged: dict[tuple[str, str], dict] = {}
        for row in self.answers:
            key = (row["question_id"], normalize(row["answer"]))
            if key in merged:
                merged[key]["sources"].append(row["source"])
            else:
                merged[key] = row | {"sources": [row["source"]]}
        return [Candidate(r["question_id"], r["answer"], tuple(r["cited"]), tuple(r["seen"]), tuple(r["sources"]),
                          r["kind"] == "revision", r["answer_f1"]) for r in merged.values()]

    def evidence(self, question_id: str) -> list[Paragraph]:
        """Every paragraph any writer saw or cited for the question, plus the annotators' evidence, in paper order. All
        answers to a question show this same set, so its size never hints at which writer answered."""
        ids = {pid for row in self.answers if row["question_id"] == question_id for pid in row["seen"] + row["cited"]}
        ids |= {pid for r in self.questions[question_id].references for pid in r.evidence_ids}
        return [self.paragraph(question_id, pid) for pid in sorted(ids, key=_paper_order)]

    def paragraph(self, question_id: str, paragraph_id: str) -> Paragraph:
        return self.paragraphs[self.questions[question_id].paper_id, paragraph_id]
