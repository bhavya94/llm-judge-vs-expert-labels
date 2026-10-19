"""The `ljvel` command: sample the answers to label, and print what is in the sample."""

import argparse
from collections import Counter

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.sample import F1_BANDS, cell, group, load


def _candidates(args: argparse.Namespace) -> None:
    candidates = Corpus.load().candidates()
    sample, split = load(candidates)
    print(f"{len(candidates)} distinct answers to {len({c.question_id for c in candidates})} questions; "
          f"sampled {len(sample)}")
    columns = [(True, "zero"), (True, "exact")] + [(False, band) for band in F1_BANDS]
    names = ["unans., F1 0", "unans., F1 1"] + [f"F1 {band.replace('_', ' ')}" for band in F1_BANDS]
    print(f"  {'group':<20}" + "".join(f"{name:>15}" for name in names) + f"{'total':>8}")
    for rows, label in [(candidates, "all"), (sample, "sample")]:
        counts = Counter((group(c), cell(c)) for c in rows)
        print(f"  {label}")
        for name in sorted({group(c) for c in rows}):
            print(f"  {name:<20}" + "".join(f"{counts[name, key]:>15}" for key in columns)
                  + f"{sum(counts[name, key] for key in columns):>8}")
    for name, part in [("dev", split.dev), ("test", split.test)]:
        print(f"{name}: {len(part)} answers to {len({c.question_id for c in part})} questions")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ljvel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("candidates", help="the sample to label, by writer and kind of answer").set_defaults(
        run=_candidates)
    args = parser.parse_args(argv)
    args.run(args)
