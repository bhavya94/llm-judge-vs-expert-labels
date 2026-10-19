"""The answers to label: a seeded sample spread over writers and kinds of answer, split by question into dev and test.

The sample is drawn once and frozen in `data/sample.json`. Labels point at its answers, so a re-export or a code
change must never move them silently: `tests/test_sample.py` fails if a fresh draw differs from the frozen file."""

import json
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

from llm_judge_vs_expert_labels.data.records import Candidate

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "sample.toml"
FROZEN = ROOT / "data" / "sample.json"
F1_BANDS = ("zero", "below_half", "half_to_one", "exact")


def f1_band(f1: float) -> str:
    if f1 == 0:
        return "zero"
    if f1 == 1:
        return "exact"
    return "below_half" if f1 < 0.5 else "half_to_one"


def group(candidate: Candidate) -> str:
    """The writer run for a first answer; every reflection revision shares one group."""
    return "revision" if candidate.revision else candidate.sources[0]


def cell(candidate: Candidate) -> tuple[bool, str]:
    """Answered or not, and how far answer-F1 says it is from the references. High-F1 answers can still be wrong and
    low-F1 answers right, which is why the sample needs some of each."""
    return candidate.unanswerable, f1_band(candidate.answer_f1)


@dataclass(frozen=True)
class Split:
    dev: list[Candidate]
    test: list[Candidate]


@dataclass(frozen=True)
class Sampler:
    size: int
    test_size: int
    """Test gets whole questions until it holds at least this many answers, so it can run a little over."""
    seed: int

    @classmethod
    def from_config(cls, path: Path = CONFIG) -> "Sampler":
        config = tomllib.loads(path.read_text())
        return cls(config["size"], config["test_size"], config["seed"])

    def sample(self, candidates: list[Candidate]) -> list[Candidate]:
        """An equal share from each group. Within a group, cells take turns, so rare cells (unanswerable answers,
        exact matches) are all taken before common ones fill the rest. Each group draws from its own seeded generator,
        so a change in one group leaves the others' picks alone. The result is shuffled."""
        groups: dict[str, dict[tuple[bool, str], list[Candidate]]] = {}
        for c in sorted(candidates, key=lambda c: c.id):
            groups.setdefault(group(c), {}).setdefault(cell(c), []).append(c)
        share, picked = self.size // len(groups), []
        for name, cells in sorted(groups.items()):
            if sum(map(len, cells.values())) < share:
                raise ValueError(f"group {name} has fewer than {share} answers")
            queues, rng = [cells[key] for key in sorted(cells)], random.Random(f"{self.seed}:{name}")
            for queue in queues:
                rng.shuffle(queue)
            taken: list[Candidate] = []
            while len(taken) < share:
                taken += [queue.pop() for queue in queues if queue][: share - len(taken)]
            picked += taken
        random.Random(self.seed).shuffle(picked)
        return picked

    def split(self, sample: list[Candidate]) -> Split:
        """Test takes whole questions, so no question has answers on both sides: a judge tuned on dev never sees a
        test question."""
        questions = sorted({c.question_id for c in sample})
        random.Random(self.seed).shuffle(questions)
        test_questions: set[str] = set()
        for qid in questions:
            if sum(c.question_id in test_questions for c in sample) >= self.test_size:
                break
            test_questions.add(qid)
        return Split(dev=[c for c in sample if c.question_id not in test_questions],
                     test=[c for c in sample if c.question_id in test_questions])


def freeze(sample: list[Candidate], split: Split, path: Path = FROZEN) -> None:
    """Write the answer ids in labeling order, each with its side of the split."""
    test = {c.id for c in split.test}
    rows = [{"id": c.id, "split": "test" if c.id in test else "dev"} for c in sample]
    path.write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")


def load(candidates: list[Candidate], path: Path = FROZEN) -> tuple[list[Candidate], Split]:
    """The frozen sample, in labeling order, and its split."""
    by_id = {c.id: c for c in candidates}
    rows = json.loads(path.read_text())
    if missing := [r["id"] for r in rows if r["id"] not in by_id]:
        raise ValueError(f"{len(missing)} frozen answers are not in data/candidates.jsonl")
    sample = [by_id[r["id"]] for r in rows]
    return sample, Split(dev=[by_id[r["id"]] for r in rows if r["split"] == "dev"],
                         test=[by_id[r["id"]] for r in rows if r["split"] == "test"])
