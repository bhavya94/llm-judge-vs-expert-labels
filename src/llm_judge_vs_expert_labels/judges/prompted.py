"""A prompted LLM judge. It sees exactly what the labeler sees, the labeling page's `Item`, written out as text.

The prompt asks for the reason before the verdict, so the verdict follows from the reason. This is the general prompt,
written before the failure modes exist; per-mode judges come after axial coding.
"""

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.records import Candidate
from llm_judge_vs_expert_labels.judges.base import Verdict
from llm_judge_vs_expert_labels.judges.code import CONFIG
from llm_judge_vs_expert_labels.labeling.items import Item
from llm_judge_vs_expert_labels.llm.client import CachedLLM, Meter

CACHE = CONFIG.parents[1] / ".cache" / "llm"
SYSTEM = """You review answers to questions about NLP research papers, as a domain expert would.

An answer passes when it is correct according to the paper and answers what was asked. Its wording can differ from the
reference answers if its meaning agrees. The annotators who wrote the references can disagree; an answer passes if it
agrees with a reference that the paragraphs support. "Unanswerable" passes only if the paper does not answer the
question. An answer fails if it is wrong, answers a different question, misses the main point, or adds claims the
paper does not make.

Reply with one JSON object and nothing else, with the reason first:
{"reason": "<one or two sentences>", "verdict": "pass" or "fail"}"""


def render(item: Item) -> str:
    """The labeling page as text: question, references, paragraphs with their marks, then the answer."""
    references = "\n".join(f"- [{r.type}] {r.text}" for r in item.references)
    passages = "\n\n".join(
        f"[{p.id}, section: {p.section}]"
        + "".join(f" ({mark})" for mark, on in [("cited by the answer", p.cited), ("expert evidence", p.expert)] if on)
        + f"\n{p.text}" for p in item.passages)
    return (f"Paper: {item.paper_title}\nQuestion: {item.question}\n\nReference answers:\n{references}\n\n"
            f"Paragraphs, in paper order:\n\n{passages}\n\nAnswer to judge:\n{item.answer}")


class PromptedJudge:
    def __init__(self, name: str, llm: CachedLLM, corpus: Corpus):
        self.name = name
        self.llm = llm
        self.corpus = corpus
        self.unreadable = 0
        """Replies with no readable verdict. Each counts as a Fail, so a high count drags TPR down."""

    @classmethod
    def from_config(cls, model: str, corpus: Corpus, client: Any, path: Path = CONFIG) -> "PromptedJudge":
        """`model` is a key under [llm.models]. A None client reads the cache only."""
        config = tomllib.loads(path.read_text())["llm"]
        settings = config["models"][model]
        meter = Meter(settings["input_price"], settings["output_price"], config["cap"])
        llm = CachedLLM(client, settings["id"], meter, cache=CACHE, max_tokens=config["max_tokens"])
        return cls(model, llm, corpus)

    def estimate(self, candidates: list[Candidate]) -> float:
        """Worst-case dollars to judge `candidates` with an empty cache: 1 token per 3 characters of input, and every
        reply at full length."""
        meter = self.llm.meter
        return sum((len(SYSTEM) + len(self.prompt(c))) / 3 * meter.input_price + self.llm.max_tokens * meter.output_price
                   for c in candidates) / 1e6

    def prompt(self, candidate: Candidate) -> str:
        return render(Item.of(candidate, self.corpus))

    def judge(self, candidate: Candidate) -> Verdict:
        reply = self.llm.ask(SYSTEM, [{"role": "user", "content": self.prompt(candidate)}])
        if (verdict := parse(reply)) is None:
            self.unreadable += 1
            return Verdict(False, f"unreadable reply: {reply[:80]!r}")
        return verdict


def parse(reply: str) -> Verdict | None:
    """The first JSON object in the reply, if it holds a pass or fail verdict."""
    if not (match := re.search(r"\{.*\}", reply, re.DOTALL)):
        return None
    try:
        row = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict) or row.get("verdict") not in ("pass", "fail"):
        return None
    return Verdict(row["verdict"] == "pass", str(row.get("reason", "")))
