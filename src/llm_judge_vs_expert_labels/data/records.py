"""Typed records for judging: QASPER questions with the annotators' answers, and the candidate answers to judge."""

import hashlib
from dataclasses import dataclass

UNANSWERABLE = "Unanswerable"


@dataclass(frozen=True)
class Paragraph:
    id: str
    """Unique within its paper only, as in QASPER: `s{section}p{paragraph}`."""
    section: str
    text: str


@dataclass(frozen=True)
class Reference:
    """One annotator's answer. QASPER has one to three per question, and they can disagree."""

    text: str
    type: str
    """extractive, abstractive, boolean or none (unanswerable)."""
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    id: str
    paper_id: str
    paper_title: str
    text: str
    references: tuple[Reference, ...]


@dataclass(frozen=True)
class Candidate:
    """One distinct answer to a question. Several writers can give the same answer; it is judged once."""

    question_id: str
    answer: str
    cited: tuple[str, ...]
    seen: tuple[str, ...]
    """Paragraphs the first writer to give this answer was shown, in retrieval order. Closed-book writers see none."""
    sources: tuple[str, ...]
    """Every writer run or reflection setup that gave this answer, first one first."""
    revision: bool
    """The first source revised an earlier draft, rather than answering once."""
    answer_f1: float
    """Best token F1 against the references, as QASPER's evaluator scores it."""

    @property
    def id(self) -> str:
        """Stable while the question and the answer text are: labels keep pointing at the same candidate."""
        return hashlib.sha1(f"{self.question_id}\n{normalize(self.answer)}".encode()).hexdigest()[:12]

    @property
    def unanswerable(self) -> bool:
        return normalize(self.answer) == UNANSWERABLE.lower()


def normalize(answer: str) -> str:
    """Case and spacing differences don't make a different answer."""
    return " ".join(answer.split()).lower()
