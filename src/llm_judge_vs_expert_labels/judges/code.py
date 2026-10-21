"""Judges written as code: free, instant and deterministic. They are the baseline every LLM judge must beat.

A summary that says something the article doesn't often uses words the article doesn't. So the baseline passes a
summary when enough of its words appear in the article. It reads only the surface: a summary can reuse the article's
words and still get the facts wrong.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from llm_judge_vs_expert_labels.data.qags import Summary
from llm_judge_vs_expert_labels.judges.base import Verdict

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "judges.toml"


def words(text: str) -> list[str]:
    """Lowercase runs of letters and digits. The QAGS texts are mostly lowercased already."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class WordOverlap:
    """Pass when the share of the summary's words found in the article reaches the threshold."""

    threshold: float

    @classmethod
    def from_config(cls, path: Path = CONFIG) -> list["WordOverlap"]:
        return [cls(t) for t in tomllib.loads(path.read_text())["overlap"]["thresholds"]]

    @property
    def name(self) -> str:
        return f"words in article >= {self.threshold:g}"

    def judge(self, summary: Summary) -> Verdict:
        article = set(words(summary.article))
        summary_words = words(summary.text)
        missing = [w for w in summary_words if w not in article]
        share = 1 - len(missing) / len(summary_words) if summary_words else 1.0
        if share >= self.threshold:
            return Verdict(True, "")
        return Verdict(False, f"{share:.0%} of words in the article; missing: {', '.join(missing[:5])}")
