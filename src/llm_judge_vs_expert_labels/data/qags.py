"""The human labels released with QAGS (Wang, Cho and Lewis, 2020): summaries of CNN/DailyMail and XSum articles, with
three crowd workers' yes/no answers to "is this sentence consistent with the article?" for every summary sentence.

The QAGS repository has no license file, so the labels are downloaded at a pinned commit and checked against their
SHA-256, never committed. The dev/test split is frozen in `data/qags_split.json` as summary ids only.
"""

import hashlib
import json
import random
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "qags.toml"
DATA = ROOT / "data" / "qags"
FROZEN = ROOT / "data" / "qags_split.json"


@dataclass(frozen=True)
class Sentence:
    text: str
    votes: tuple[bool, ...]
    """Each worker's answer, True for "yes, consistent", in the release's order."""

    @property
    def consistent(self) -> bool:
        """The majority of workers say yes."""
        return 2 * sum(self.votes) > len(self.votes)


@dataclass(frozen=True)
class Summary:
    dataset: str
    """cnndm or xsum."""
    article: str
    sentences: tuple[Sentence, ...]

    @property
    def id(self) -> str:
        """Stable while the article and the summary text are."""
        return hashlib.sha1(f"{self.dataset}\n{self.article}\n{self.text}".encode()).hexdigest()[:12]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def consistent(self) -> bool:
        """The label every judge is scored against: the majority says yes for every sentence."""
        return all(s.consistent for s in self.sentences)


@dataclass(frozen=True)
class Split:
    dev: list[Summary]
    test: list[Summary]


class Release:
    """The pinned files: where they come from and what they must hash to."""

    def __init__(self, base_url: str, files: dict[str, str], directory: Path = DATA):
        self.base_url = base_url
        self.files = files
        """Dataset name -> SHA-256 of `mturk_{name}.jsonl`."""
        self.directory = directory

    @classmethod
    def from_config(cls, path: Path = CONFIG, directory: Path = DATA) -> "Release":
        config = tomllib.loads(path.read_text())["release"]
        return cls(f"{config['url']}/{config['commit']}/data", config["sha256"], directory)

    def path(self, dataset: str) -> Path:
        return self.directory / f"mturk_{dataset}.jsonl"

    def download(self) -> list[Path]:
        """Fetch any missing file and check every file's hash. A file that doesn't match is deleted, not used."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for dataset, digest in self.files.items():
            path = self.path(dataset)
            if not path.exists():
                with urllib.request.urlopen(f"{self.base_url}/{path.name}") as response:
                    path.write_bytes(response.read())
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                path.unlink()
                raise ValueError(f"{path.name} does not match its pinned SHA-256; deleted it")
        return [self.path(d) for d in self.files]

    def summaries(self) -> list[Summary]:
        """Every summary, CNN/DM first, in file order."""
        if missing := [d for d in self.files if not self.path(d).exists()]:
            raise FileNotFoundError(f"missing {', '.join(missing)}: run `ljvel download` first")
        return [_summary(dataset, json.loads(line)) for dataset in self.files
                for line in self.path(dataset).read_text().splitlines() if line.strip()]


def _summary(dataset: str, row: dict) -> Summary:
    sentences = tuple(Sentence(s["sentence"], tuple(r["response"] == "yes" for r in s["responses"]))
                      for s in row["summary_sentences"])
    return Summary(dataset, row["article"], sentences)


def split(summaries: list[Summary], test_share: float, seed: int) -> Split:
    """A seeded split, stratified by dataset and label, so dev and test have the same mix. Every article has one
    summary, so splitting summaries also splits articles."""
    test = set()
    for stratum in sorted({(s.dataset, s.consistent) for s in summaries}):
        ids = sorted(s.id for s in summaries if (s.dataset, s.consistent) == stratum)
        random.Random(f"{seed}:{stratum}").shuffle(ids)
        test |= set(ids[:round(len(ids) * test_share)])
    return Split(dev=[s for s in summaries if s.id not in test], test=[s for s in summaries if s.id in test])


def split_from_config(summaries: list[Summary], path: Path = CONFIG) -> Split:
    config = tomllib.loads(path.read_text())["split"]
    return split(summaries, config["test_share"], config["seed"])


def freeze(split: Split, path: Path = FROZEN) -> None:
    rows = [{"id": s.id, "split": "dev"} for s in split.dev] + [{"id": s.id, "split": "test"} for s in split.test]
    path.write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")


def load(summaries: list[Summary], path: Path = FROZEN) -> Split:
    """The frozen split. Fails if a frozen id has no summary, which means the data changed."""
    by_id = {s.id: s for s in summaries}
    rows = json.loads(path.read_text())
    if missing := [r["id"] for r in rows if r["id"] not in by_id]:
        raise ValueError(f"{len(missing)} frozen summaries are not in the downloaded data")
    return Split(dev=[by_id[r["id"]] for r in rows if r["split"] == "dev"],
                 test=[by_id[r["id"]] for r in rows if r["split"] == "test"])
