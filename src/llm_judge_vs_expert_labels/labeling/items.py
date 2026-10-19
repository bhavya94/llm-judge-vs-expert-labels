"""What a person sees to label one answer: the question, the references, and the paragraphs that decide it. Nothing
on it says which writer answered or how answer-F1 scored it, so neither can sway the label."""

from dataclasses import asdict, dataclass

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.records import Candidate, Reference


@dataclass(frozen=True)
class Passage:
    id: str
    section: str
    text: str
    cited: bool
    """This answer cites the paragraph."""
    expert: bool
    """Some annotator cited this paragraph as evidence."""


@dataclass(frozen=True)
class Item:
    id: str
    question_id: str
    paper_title: str
    question: str
    answer: str
    passages: tuple[Passage, ...]
    """The question's evidence set (`Corpus.evidence`), the same for every answer to it, in paper order."""
    references: tuple[Reference, ...]

    @classmethod
    def of(cls, candidate: Candidate, corpus: Corpus) -> "Item":
        question = corpus.questions[candidate.question_id]
        expert = {pid for r in question.references for pid in r.evidence_ids}
        passages = tuple(Passage(p.id, p.section, p.text, p.id in candidate.cited, p.id in expert)
                         for p in corpus.evidence(question.id))
        return cls(candidate.id, question.id, question.paper_title, question.text, candidate.answer, passages,
                   question.references)

    def to_row(self) -> dict:
        return asdict(self)
