"""Judges written as code: free, instant and deterministic. They are the baseline every LLM judge must beat.

They see what the labeler sees: the question, the references and the answer's citations. Answer-F1 is the metric
QASPER reports; the checks are rules a reviewer might write before reading any answers.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.records import UNANSWERABLE, Candidate
from llm_judge_vs_expert_labels.judges.base import Verdict

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "judges.toml"


@dataclass(frozen=True)
class F1Threshold:
    """Pass when answer-F1 against the best reference reaches the threshold."""

    threshold: float

    @property
    def name(self) -> str:
        return f"answer-F1 >= {self.threshold:g}"

    def judge(self, candidate: Candidate) -> Verdict:
        if candidate.answer_f1 >= self.threshold:
            return Verdict(True, "")
        return Verdict(False, f"answer-F1 {candidate.answer_f1:.2f} is below {self.threshold:g}")


class CodeChecks:
    """Three checks, each a rule about form rather than meaning. Fails at the first check that fails."""

    name = "code checks"

    def __init__(self, corpus: Corpus, max_words: int):
        self.corpus = corpus
        self.max_words = max_words

    @classmethod
    def from_config(cls, corpus: Corpus, path: Path = CONFIG) -> "CodeChecks":
        return cls(corpus, tomllib.loads(path.read_text())["code_checks"]["max_answer_words"])

    def judge(self, candidate: Candidate) -> Verdict:
        references = self.corpus.questions[candidate.question_id].references
        unanswerable = [r.text == UNANSWERABLE for r in references]
        if candidate.unanswerable and not any(unanswerable):
            return Verdict(False, "says Unanswerable, but every reference answers the question")
        if not candidate.unanswerable and all(unanswerable):
            return Verdict(False, "answers, but every reference says the paper doesn't")
        if not candidate.unanswerable and not candidate.cited:
            return Verdict(False, "gives an answer but cites no paragraph")
        if (words := len(candidate.answer.split())) > self.max_words:
            return Verdict(False, f"{words} words, more than {self.max_words}")
        return Verdict(True, "")
