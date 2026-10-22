"""QAGS (Wang, Cho and Lewis, 2020), with an LLM in place of the paper's question-generation and QA models.

1. Ask questions about the summary: one per answer span (a name, number, date or noun phrase), up to `max_questions`.
2. Answer every question from the summary, then from the article, in two separate calls.
3. Drop questions the summary can't answer, as the paper drops questions with no predicted answer, and duplicates.
4. Score each question by token F1 between the two answers, so an article that can't answer scores 0. The
   summary's score is the mean. A consistent summary gets the same answers from both texts.

The paper writes 100 candidate questions and keeps the 20 most probable. One LLM call writes fewer, better ones.
"""

import json
import re
import string
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_judge_vs_expert_labels.data.qags import Summary
from llm_judge_vs_expert_labels.judges.base import Verdict
from llm_judge_vs_expert_labels.judges.code import CONFIG
from llm_judge_vs_expert_labels.judges.prompted import CACHE
from llm_judge_vs_expert_labels.llm.client import CachedLLM, Meter

UNANSWERABLE = "unanswerable"
ASK = """You write questions to check a summary against its source.

Pick answer spans from the summary: names, numbers, dates and short noun phrases. For each span, write one question
that the span answers, using only the summary. Each question must be at least three words long.

Reply with one JSON object and nothing else:
{"questions": [{"answer": "<span>", "question": "<question>"}, ...]}"""
ANSWER = f"""You answer questions using only the given text.

Answer each question with the shortest span from the text that answers it. If the text does not say, answer
"{UNANSWERABLE}". Keep the questions' order.

Reply with one JSON object and nothing else:
{{"answers": ["<answer to question 1>", "<answer to question 2>", ...]}}"""


def _normalize(text: str) -> list[str]:
    """SQuAD's answer normalization: lowercase, no punctuation, no articles."""
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return [w for w in text.split() if w not in ("a", "an", "the")]


def token_f1(predicted: str, gold: str) -> float:
    p, g = _normalize(predicted), _normalize(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return float(p == g)
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class Probe:
    question: str
    from_summary: str
    from_article: str

    @property
    def f1(self) -> float:
        """An article answer of "unanswerable" shares no word with a real answer, so it scores 0."""
        return token_f1(self.from_article, self.from_summary)


def _json(reply: str, key: str) -> list | None:
    if not (match := re.search(r"\{.*\}", reply, re.DOTALL)):
        return None
    try:
        row = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    value = row.get(key) if isinstance(row, dict) else None
    return value if isinstance(value, list) else None


class QagsJudge:
    def __init__(self, model: str, llm: CachedLLM, threshold: float, max_questions: int = 10,
                 retry_max_tokens: int = 1000):
        self.model = model
        self.llm = llm
        self.threshold = threshold
        self.max_questions = max_questions
        self.retry_max_tokens = retry_max_tokens
        self.unreadable = 0
        """Summaries whose questions or answers couldn't be read even after a retry. Each counts as a Fail."""
        self.no_questions = 0
        """Summaries left with no usable question. Nothing contradicts them, so each passes with score 1."""

    @classmethod
    def from_config(cls, client: Any, path: Path = CONFIG) -> list["QagsJudge"]:
        """One judge per threshold. They share one client and cache, so extra thresholds cost nothing."""
        config = tomllib.loads(path.read_text())
        qags, models = config["qags"], config["llm"]["models"]
        settings = models[qags["model"]]
        meter = Meter(settings["input_price"], settings["output_price"], config["llm"]["cap"])
        llm = CachedLLM(client, settings["id"], meter, cache=CACHE, max_tokens=qags["max_tokens"])
        return [cls(qags["model"], llm, t, qags["max_questions"], config["llm"]["retry_max_tokens"])
                for t in qags["thresholds"]]

    @property
    def name(self) -> str:
        return f"qags ({self.model}) >= {self.threshold:g}"

    def estimate(self, summaries: list[Summary]) -> float:
        """Worst-case dollars with an empty cache: three calls per summary, 1 token per 3 characters of input, every
        reply at full length. The questions add about 80 characters each to both answer calls."""
        meter, questions = self.llm.meter, 80 * self.max_questions
        characters = sum(len(ASK) + len(s.text) + 2 * (len(ANSWER) + questions) + len(s.text) + len(s.article)
                         for s in summaries)
        return (characters / 3 * meter.input_price + 3 * len(summaries) * self.llm.max_tokens * meter.output_price) / 1e6

    def judge(self, summary: Summary) -> Verdict:
        probes = self.probes(summary)
        if probes is None:
            self.unreadable += 1
            return Verdict(False, "unreadable questions or answers", 0.0)
        if not probes:
            self.no_questions += 1
            return Verdict(True, "no usable question", 1.0)
        score = sum(p.f1 for p in probes) / len(probes)
        worst = min(probes, key=lambda p: p.f1)
        reason = (f"mean answer F1 {score:.2f} over {len(probes)} questions; worst: {worst.question!r} -> summary "
                  f"{worst.from_summary!r}, article {worst.from_article!r}")
        return Verdict(score >= self.threshold, reason, score)

    def probes(self, summary: Summary) -> list[Probe] | None:
        """The questions that survive filtering, each with its two answers. None when a reply can't be read."""
        asked = self._ask(ASK, summary.text, "questions")
        if asked is None:
            return None
        questions = list(dict.fromkeys(q["question"].strip() for q in asked if isinstance(q, dict)
                                       and isinstance(q.get("question"), str) and len(q["question"].split()) >= 3))
        questions = questions[:self.max_questions]
        if not questions:
            return []
        from_summary = self._answers(summary.text, questions)
        from_article = self._answers(summary.article, questions)
        if from_summary is None or from_article is None:
            return None
        return [Probe(q, s, a) for q, s, a in zip(questions, from_summary, from_article)
                if s.strip().lower() != UNANSWERABLE]

    def _answers(self, text: str, questions: list[str]) -> list[str] | None:
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        answers = self._ask(ANSWER, f"Text:\n{text}\n\nQuestions:\n{numbered}", "answers")
        if answers is None or len(answers) != len(questions):
            return None
        return [str(a) for a in answers]

    def _ask(self, system: str, content: str, key: str) -> list | None:
        """One call, and one retry with a higher token limit if the reply can't be read."""
        messages = [{"role": "user", "content": content}]
        if (value := _json(self.llm.ask(system, messages), key)) is not None:
            return value
        return _json(self.llm.ask(system, messages, max_tokens=self.retry_max_tokens), key)
