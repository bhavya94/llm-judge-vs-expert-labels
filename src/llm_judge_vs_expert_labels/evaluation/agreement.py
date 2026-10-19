"""How well a judge's verdicts match the person's labels on the same answers, each number with a bootstrap interval.

TPR is the share of the person's Passes the judge passes; TNR is the share of their Fails it fails. Cohen's kappa is
agreement beyond what the two pass rates would give by chance. The same measure, run on the person's first labels
against their blind relabels, gives the ceiling: no judge should be expected to agree with them better than they
agree with themselves.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from metrics_from_scratch.agreement.kappa import cohen_kappa
from metrics_from_scratch.stats.bootstrap import Bootstrap, Interval

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "judges.toml"


def _tpr(human: np.ndarray, judged: np.ndarray) -> float:
    return float(judged[human].mean()) if human.any() else float("nan")


def _tnr(human: np.ndarray, judged: np.ndarray) -> float:
    return float((~judged[~human]).mean()) if (~human).any() else float("nan")


def _kappa(human: np.ndarray, judged: np.ndarray) -> float:
    return cohen_kappa(human, judged, labels=[False, True])


@dataclass(frozen=True)
class Agreement:
    n: int
    human_pass_rate: float
    judge_pass_rate: float
    tpr: Interval
    tnr: Interval
    kappa: Interval

    @classmethod
    def of(cls, human: list[bool], judged: list[bool], bootstrap: Bootstrap) -> "Agreement":
        """`human[i]` and `judged[i]` are verdicts on the same answer: True is Pass."""
        if len(human) != len(judged) or not human:
            raise ValueError("need one judge verdict per labeled answer, and at least one")
        h, j = np.array(human, dtype=bool), np.array(judged, dtype=bool)
        return cls(len(h), float(h.mean()), float(j.mean()), bootstrap.interval(_tpr, h, j),
                   bootstrap.interval(_tnr, h, j), bootstrap.interval(_kappa, h, j))


def bootstrap_from_config(path: Path = CONFIG) -> Bootstrap:
    config = tomllib.loads(path.read_text())["agreement"]
    return Bootstrap(config["bootstrap_resamples"], config["confidence"], config["seed"])
