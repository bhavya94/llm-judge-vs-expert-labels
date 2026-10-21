# llm-judge-vs-expert-labels

How well an automated judge reproduces human labels of factual consistency, how few labels it needs, and whether a
small trained judge beats a prompted one. The labels are the ones released with QAGS (Wang, Cho and Lewis, 2020).
Every judge, from word overlap to QAGS itself to a prompted LLM, is scored against them.

## Evidence

## What is here
- **Human labels** (`data/qags.py`, `configs/qags.toml`): 474 summaries, 235 of CNN/DailyMail articles and 239 of
  XSum articles. For every summary sentence, three crowd workers answered whether it is consistent with the
  article. A summary passes when most workers say yes for every sentence. About half pass. The QAGS repository has
  no license file, so `uv run ljvel download` fetches the files at a pinned commit and checks their SHA-256. Nothing
  from them is committed. Dev holds 315 summaries and test 159, with the same mix of dataset and label. The split
  is frozen in `data/qags_split.json` as ids only.

The QASPER sample and labeling page below came first. Labeling 300 answers by hand cost more time than it gave, so
they are parked. The page stays for a small check of one person's labels.

- **Data** (`data/`, `src/llm_judge_vs_expert_labels/data/`): 680 distinct answers to 200 QASPER questions, saved by
  [reflection-agent-for-paper-qa](https://github.com/bhavya94/reflection-agent-for-paper-qa). Three writer runs gave
  first answers: Sonnet with 20 retrieved paragraphs, Haiku with 5, and Sonnet closed-book. Six reflection setups
  gave revisions. Each answer keeps the paragraphs its writer saw and cited. `scripts/export_candidates.py` wrote
  these files once, from that repo's saved runs. When writers gave the same answer, it is labeled once and credited
  to the first writer run that gave it. So the revision group holds only revisions that changed the answer.
- **The sample** (`configs/sample.toml`): 300 answers, 75 from each writer run and 75 revisions. Within each group,
  answers spread over answered or "Unanswerable" and over answer-F1 bands, because a high F1 can hide a wrong answer.
  Dev holds 200 answers and test 100, split by question, so no question appears in both. `scripts/freeze_sample.py`
  drew the sample once into `data/sample.json`, and a test fails if a fresh draw would differ. `uv run ljvel
  candidates` prints the counts.

- **Labeling** (`labeling/`, `configs/labeling.toml`): a page on localhost shows one answer at a time, with the
  question, the reference answers and the paragraphs that decide it. Every answer to a question shows the same
  paragraphs, so their number doesn't reveal the writer. The page never shows the writer or answer-F1. The labeler
  marks Pass or Fail and writes what went wrong in every Fail. `uv run ljvel label` starts the first pass.
  `uv run ljvel label --relabel` starts a blind second pass over 60 of the answers, which measures the labeler's
  self-agreement. `uv run ljvel labels` prints progress and the Fail notes. Labels live in `labels/`.
- **Judges** (`judges/`, `configs/judges.toml`): every judge takes one answer and returns Pass or Fail with a
  reason. The first judges are code, free and instant: answer-F1 at three thresholds, and three form checks. Those
  checks fail an "Unanswerable" that every reference answers, an answer where every reference says Unanswerable, an
  answer with no citation, and an answer over 35 words.
- **Agreement** (`evaluation/`): TPR is the share of the person's Passes a judge passes. TNR is the share of their
  Fails it fails. Cohen's kappa is agreement beyond chance. Each comes with a 95% bootstrap interval. The person's
  blind relabels against their first labels give the ceiling. `uv run ljvel agree` scores every judge on dev; test
  stays unscored until the final run.

QASPER (Dasigi et al., 2021) is licensed CC BY 4.0. `data/` holds its questions, answers and the paragraphs the
labels need.

## Run it
```console
uv sync
uv run pytest
uv run ljvel --help
```

## Charts
`docs/views/index.html` draws the results above from the same seeded runs. Rebuild it with
`uv run python scripts/build_views.py`.

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py` enforces both.

## Limits
