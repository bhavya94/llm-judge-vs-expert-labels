# llm-judge-vs-expert-labels

How well an automated judge reproduces one expert's Pass/Fail labels on grounded-QA answers, how few labels it needs, and whether a small trained judge beats a prompted one.

## Evidence

## What is here
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
