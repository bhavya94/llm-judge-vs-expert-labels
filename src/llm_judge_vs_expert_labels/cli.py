"""The `ljvel` command: sample the answers to label, label them, and score judges against the labels."""

import argparse
import tomllib
from collections import Counter

from llm_judge_vs_expert_labels.data import qags
from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.sample import F1_BANDS, cell, group, load
from llm_judge_vs_expert_labels.evaluation.agreement import Agreement, bootstrap_from_config
from llm_judge_vs_expert_labels.judges.code import CONFIG as JUDGES_CONFIG
from llm_judge_vs_expert_labels.judges.code import CodeChecks, F1Threshold
from llm_judge_vs_expert_labels.labeling.items import Item
from llm_judge_vs_expert_labels.labeling.server import Rounds
from llm_judge_vs_expert_labels.labeling.summary import summarize


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


def _download(args: argparse.Namespace) -> None:
    release = qags.Release.from_config()
    try:
        paths = release.download()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(f"QAGS human labels at commit {release.base_url.split('/')[-2][:7]}: "
          + ", ".join(p.name for p in paths) + " match their SHA-256")
    split = qags.load(release.summaries())
    print(f"  {'':<6}{'dataset':<8}{'summaries':>10}{'consistent':>12}{'sentences':>11}{'unanimous':>11}")
    for name, part in [("dev", split.dev), ("test", split.test)]:
        for dataset in release.files:
            rows = [s for s in part if s.dataset == dataset]
            sentences = [t for s in rows for t in s.sentences]
            print(f"  {name:<6}{dataset:<8}{len(rows):>10}{sum(s.consistent for s in rows):>12}{len(sentences):>11}"
                  f"{sum(len(set(t.votes)) == 1 for t in sentences) / len(sentences):>11.0%}")


def _label(args: argparse.Namespace) -> None:
    corpus, rounds = Corpus.load(), Rounds.from_config()
    items = [Item.of(c, corpus) for c in load(corpus.candidates())[0]]
    try:
        app = rounds.relabel_round(items) if args.relabel else rounds.first_round(items)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    port = args.port or rounds.port
    print(f"{'Relabel' if args.relabel else 'First pass'}: {len(app.order)} answers, labels saved to "
          f"{app.labels.path}\nOpen http://127.0.0.1:{port}/ and press Ctrl-C to stop.")
    try:
        app.server(port).serve_forever()
    except KeyboardInterrupt:
        pass


def _labels(args: argparse.Namespace) -> None:
    sample = load(Corpus.load().candidates())[0]
    rounds = Rounds.from_config()
    summary = summarize(sample, rounds.first.latest())
    print(f"first pass: {summary['labeled']} of {summary['sample']} labeled; "
          f"relabel: {len(rounds.relabel.latest().keys() & {c.id for c in sample})} of {rounds.relabel_size}")
    for note in summary["notes"]:
        print(f"  {note}")
    if summary["labeled"] < summary["sample"]:
        print("Pass/Fail by writer and by answer-F1 show once the first pass is complete, so they can't sway it.")
        return
    for name, counts in sorted(summary["by_group"].items()):
        print(f"  {name:<20} pass {counts['pass']:>3}  fail {counts['fail']:>3}")
    print(f"pass with answer-F1 below 0.5: {summary['pass_low_f1']}; "
          f"fail with answer-F1 0.5 or more: {summary['fail_high_f1']}")


def _interval(i) -> str:
    return f"{i.estimate:.2f} [{i.low:.2f}, {i.high:.2f}]"


def _agree(args: argparse.Namespace) -> None:
    corpus = Corpus.load()
    _, split = load(corpus.candidates())
    rounds, bootstrap = Rounds.from_config(), bootstrap_from_config()
    first, relabel = rounds.first.latest(), rounds.relabel.latest()
    labeled = [c for c in split.dev if c.id in first]
    judges = [CodeChecks.from_config(corpus)] + [
        F1Threshold(t) for t in tomllib.loads(JUDGES_CONFIG.read_text())["f1"]["thresholds"]]
    print(f"dev: {len(labeled)} of {len(split.dev)} answers labeled. Test stays unscored until the final run.")
    for judge in judges:
        rate = sum(judge.judge(c).passed for c in split.dev) / len(split.dev)
        print(f"  {judge.name:<18} passes {rate:.0%} of dev")
    if not labeled:
        return
    print(f"  {'judge':<18}{'n':>5}{'TPR':>20}{'TNR':>20}{'kappa':>20}")
    human = [first[c.id].verdict == "pass" for c in labeled]
    rows = [(j.name, Agreement.of(human, [j.judge(c).passed for c in labeled], bootstrap)) for j in judges]
    if both := [i for i in relabel if i in first]:
        rows.insert(0, ("self (relabel)", Agreement.of([first[i].verdict == "pass" for i in both],
                                                       [relabel[i].verdict == "pass" for i in both], bootstrap)))
    for name, a in rows:
        print(f"  {name:<18}{a.n:>5}{_interval(a.tpr):>20}{_interval(a.tnr):>20}{_interval(a.kappa):>20}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ljvel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch the QAGS human labels at the pinned commit; print the split").set_defaults(
        run=_download)
    commands.add_parser("candidates", help="the sample to label, by writer and kind of answer").set_defaults(
        run=_candidates)
    label = commands.add_parser("label", help="serve the labeling page on localhost")
    label.add_argument("--relabel", action="store_true", help="the blind second pass, after the first is complete")
    label.add_argument("--port", type=int, help="default from configs/labeling.toml")
    label.set_defaults(run=_label)
    commands.add_parser("labels", help="labeling progress and the Fail notes; Pass/Fail by writer once the first pass is done").set_defaults(
        run=_labels)
    commands.add_parser("agree", help="each judge's agreement with the labels on dev, with 95% intervals").set_defaults(
        run=_agree)
    args = parser.parse_args(argv)
    args.run(args)
