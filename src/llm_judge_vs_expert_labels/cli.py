"""The `ljvel` command: sample the answers to label, label them, and score judges against the labels."""

import argparse
import tomllib
from collections import Counter

from llm_judge_vs_expert_labels.data import qags
from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.sample import F1_BANDS, cell, group, load
from llm_judge_vs_expert_labels.evaluation.agreement import Agreement, bootstrap_from_config, correlation
from llm_judge_vs_expert_labels.judges.code import CONFIG as JUDGES_CONFIG
from llm_judge_vs_expert_labels.judges.code import WordOverlap
from llm_judge_vs_expert_labels.judges.prompted import PromptedJudge
from llm_judge_vs_expert_labels.judges.qa_based import QagsJudge
from llm_judge_vs_expert_labels.labeling.items import Item
from llm_judge_vs_expert_labels.labeling.server import Rounds
from llm_judge_vs_expert_labels.labeling.summary import summarize
from llm_judge_vs_expert_labels.llm.client import BudgetExceeded, NotCached


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


def _judge(args: argparse.Namespace) -> None:
    dev = qags.load(qags.Release.from_config().summaries()).dev
    client = None
    if args.paid:
        import anthropic
        client = anthropic.Anthropic()
    judge = QagsJudge.from_config(client)[0] if args.model == "qags" else PromptedJudge.from_config(args.model, client)
    print(f"{judge.name}: {judge.llm.model} on {len(dev)} dev summaries: at most ${judge.estimate(dev):.2f} with an empty cache; "
          f"cap ${judge.llm.meter.cap:.2f}")
    if not args.paid:
        print("Nothing was spent. Add --run to judge.")
        return
    try:
        passed = sum(judge.judge(s).passed for s in dev)
    except BudgetExceeded as error:
        raise SystemExit(f"stopped: {error}") from None
    finally:
        print(f"{judge.llm.calls} API calls, ${judge.llm.meter.spent:.2f} spent "
              f"(${judge.llm.dollars_uncached():.2f} with an empty cache)")
    print(f"passes {passed / len(dev):.0%} of dev; {judge.unreadable} unreadable replies counted as Fail"
          + (f"; {judge.no_questions} with no usable question" if isinstance(judge, QagsJudge) else ""))


def _cached_judges(dev: list[qags.Summary]) -> list:
    """LLM judges whose verdicts on all of dev are cached. `agree` never calls the API."""
    candidates = [(model, [PromptedJudge.from_config(model, client=None)])
                  for model in tomllib.loads(JUDGES_CONFIG.read_text())["llm"]["models"]]
    candidates.append(("qags", QagsJudge.from_config(client=None)))
    judges = []
    for key, same_cache in candidates:
        try:
            for summary in dev:
                same_cache[0].judge(summary)
        except NotCached:
            print(f"  {key}: not judged yet; run `ljvel judge {key} --run`")
            continue
        judges += same_cache
    return judges


def _interval(i) -> str:
    return f"{i.estimate:.2f} [{i.low:.2f}, {i.high:.2f}]"


def _agree(args: argparse.Namespace) -> None:
    dev = qags.load(qags.Release.from_config().summaries()).dev
    bootstrap = bootstrap_from_config()
    judges: list = WordOverlap.from_config()
    print(f"dev: {len(dev)} summaries. Pass = most workers say every sentence is consistent. "
          "Test stays unscored until the final run.")
    judges += _cached_judges(dev)
    for dataset in ["cnndm", "xsum", None]:
        part = [s for s in dev if dataset in (None, s.dataset)]
        human = [s.consistent for s in part]
        print(f"\n{dataset or 'both'}: {len(part)} summaries, {sum(human) / len(part):.0%} pass")
        print(f"  {'judge':<26}{'passes':>7}{'TPR':>20}{'TNR':>20}{'kappa':>20}{'r with human score':>22}")
        # The ceiling: one crowd worker's label against another's, for each pair of answer positions. r is for judges
        # only: a worker's own votes are part of the human score.
        rows = [(f"worker {a + 1} vs worker {b + 1}", [s.by_worker(a) for s in part], [s.by_worker(b) for s in part],
                 None) for a, b in [(0, 1), (0, 2), (1, 2)]]
        for j in judges:
            verdicts = [j.judge(s) for s in part]
            scores = [float(v.passed) if v.score is None else v.score for v in verdicts]
            rows.append((j.name, human, [v.passed for v in verdicts], scores))
        for name, reference, judged, scores in rows:
            a = Agreement.of(reference, judged, bootstrap)
            r = "-" if scores is None else _interval(correlation([s.human_score for s in part], scores, bootstrap))
            print(f"  {name:<26}{a.judge_pass_rate:>7.0%}{_interval(a.tpr):>20}{_interval(a.tnr):>20}"
                  f"{_interval(a.kappa):>20}{r:>22}")
        _against_one_worker(part, judges, bootstrap)


def _against_one_worker(part: list[qags.Summary], judges: list, bootstrap) -> None:
    """The fair comparison with the ceiling. A worker pair compares one person with one person, while the table above
    scores judges against the majority of three, which is less noisy. Here every row is scored against one worker at
    a time, and a worker is never scored against itself."""
    workers = [[s.by_worker(k) for s in part] for k in range(3)]
    rows = [(f"worker {k + 1}", workers[k], k) for k in range(3)]
    rows += [(j.name, [j.judge(s).passed for s in part], None) for j in judges]
    print(f"  kappa against one worker at a time {'vs worker 1':>13}{'vs worker 2':>13}{'vs worker 3':>13}{'mean':>8}")
    for name, judged, own in rows:
        kappas = [None if k == own else Agreement.of(workers[k], judged, bootstrap).kappa.estimate for k in range(3)]
        mean = sum(k for k in kappas if k is not None) / sum(k is not None for k in kappas)
        print(f"  {name:<35}" + "".join(f"{'-' if k is None else f'{k:.2f}':>13}" for k in kappas) + f"{mean:>8.2f}")


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
    commands.add_parser("agree", help="each judge's agreement with the QAGS human labels on dev, with 95%% intervals").set_defaults(
        run=_agree)
    judge = commands.add_parser("judge", help="estimate the cost of a prompted judge on dev; --run judges")
    judge.add_argument("model", help="a key under [llm.models] in configs/judges.toml, or qags")
    judge.add_argument("--run", dest="paid", action="store_true", help="call the API, within the cap; replies are cached")
    judge.set_defaults(run=_judge)
    args = parser.parse_args(argv)
    args.run(args)
