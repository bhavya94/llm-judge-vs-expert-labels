"""A prompted LLM judge. It sees what the QAGS crowd workers saw: the article and the summary, sentence by sentence.

The prompt asks for the reason before the verdict, so the verdict follows from the reason. A summary passes only when
every sentence is consistent, which is how its human label is built.
"""

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from llm_judge_vs_expert_labels.data.qags import Summary
from llm_judge_vs_expert_labels.judges.base import Verdict
from llm_judge_vs_expert_labels.judges.code import CONFIG
from llm_judge_vs_expert_labels.llm.client import CachedLLM, Meter

CACHE = CONFIG.parents[1] / ".cache" / "llm"
SYSTEM = """You check summaries of news articles for factual consistency.

A sentence is consistent when everything it states is supported by the article. It is inconsistent when it
contradicts the article or adds facts the article does not state, even if those facts might be true. Wording can
differ from the article. A summary passes only when every one of its sentences is consistent.

Reply with one JSON object and nothing else, with the reason first:
{"reason": "<one or two sentences; name any inconsistent sentence>", "verdict": "pass" or "fail"}"""


def render(summary: Summary) -> str:
    sentences = "\n".join(f"{i}. {s.text}" for i, s in enumerate(summary.sentences, 1))
    return f"Article:\n{summary.article}\n\nSummary, one sentence per line:\n{sentences}"


class PromptedJudge:
    def __init__(self, name: str, llm: CachedLLM):
        self.name = name
        self.llm = llm
        self.unreadable = 0
        """Replies with no readable verdict. Each counts as a Fail, so a high count drags TPR down."""

    @classmethod
    def from_config(cls, model: str, client: Any, path: Path = CONFIG) -> "PromptedJudge":
        """`model` is a key under [llm.models]. A None client reads the cache only."""
        config = tomllib.loads(path.read_text())["llm"]
        settings = config["models"][model]
        meter = Meter(settings["input_price"], settings["output_price"], config["cap"])
        return cls(model, CachedLLM(client, settings["id"], meter, cache=CACHE, max_tokens=config["max_tokens"]))

    def estimate(self, summaries: list[Summary]) -> float:
        """Worst-case dollars to judge `summaries` with an empty cache: 1 token per 3 characters of input, and every
        reply at full length."""
        meter = self.llm.meter
        return sum((len(SYSTEM) + len(render(s))) / 3 * meter.input_price + self.llm.max_tokens * meter.output_price
                   for s in summaries) / 1e6

    def judge(self, summary: Summary) -> Verdict:
        reply = self.llm.ask(SYSTEM, [{"role": "user", "content": render(summary)}])
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
