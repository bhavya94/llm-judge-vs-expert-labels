PR: [bhavya94/llm-judge-vs-expert-labels#1](https://github.com/bhavya94/llm-judge-vs-expert-labels/pull/1)

# Stack 1/3: candidate answers, a frozen sample, and the dev/test split

## Summary
This project asks how well an automated judge reproduces one person's Pass/Fail labels. This PR picks the answers that
person will label. It reuses 805 answers that the reflection project already paid for, so it costs nothing. They
merge into 680 distinct answers to 200 QASPER questions. A seeded draw takes 300, and a file freezes the draw, so labels
never lose their answers.

## How it works
```mermaid
flowchart LR
  R["reflection repo: saved runs<br/>and QASPER dev"] -->|"export once"| D["data/: candidates,<br/>questions, paragraphs"]
  D --> C["Corpus.candidates:<br/>merge duplicates -> 680"]
  C --> S["Sampler: 75 per group,<br/>rare cells first"]
  S --> Q["split by question:<br/>dev 200, test 100"]
  Q --> F["data/sample.json<br/>(frozen)"]
```
- [`scripts/export_candidates.py:26`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/scripts/export_candidates.py#L26) runs once, inside the reflection repo's environment. It writes each answer with the
  paragraphs its writer saw and cited. Only the current run files count.
- [`scripts/export_candidates.py:30`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/scripts/export_candidates.py#L30) recomputes what each writer saw with the same BM25 retriever. A check confirmed
  the counts: every short list comes from a paper with fewer paragraphs than the writer's k.
- [`src/llm_judge_vs_expert_labels/data/corpus.py:33`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/src/llm_judge_vs_expert_labels/data/corpus.py#L33) merges answers that differ only in case or spacing. The first
  row decides the group, so a revision that repeats a first answer counts as that first answer.
- [`src/llm_judge_vs_expert_labels/data/records.py:53`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/src/llm_judge_vs_expert_labels/data/records.py#L53) gives each answer an id from its question and normalized text.
  Labels point at this id, and a test pins it.
- [`src/llm_judge_vs_expert_labels/data/sample.py:57`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/src/llm_judge_vs_expert_labels/data/sample.py#L57) takes 75 answers per group. Cells take turns, so rare answers
  (Unanswerable, exact matches) all enter before common ones fill the rest.
- [`src/llm_judge_vs_expert_labels/data/sample.py:68`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/src/llm_judge_vs_expert_labels/data/sample.py#L68) gives each group its own seed. A change in one group leaves the
  others' picks alone.
- [`src/llm_judge_vs_expert_labels/data/sample.py:78`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/src/llm_judge_vs_expert_labels/data/sample.py#L78) moves whole questions into test until it holds 100 answers.
- [`src/llm_judge_vs_expert_labels/data/sample.py:99`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/src/llm_judge_vs_expert_labels/data/sample.py#L99) loads the frozen sample. [`tests/test_sample.py:57`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/0c89cb9df3a3a1782a15ffc3c960866c1ee853db/tests/test_sample.py#L57) fails if a
  fresh draw would differ from it.

## Results
| Group | All answers | Sampled |
|---|---|---|
| sonnet-k20 (Sonnet, 20 paragraphs) | 200 | 75 |
| sonnet-closed-book | 178 | 75 |
| haiku-k5 (Haiku, 5 paragraphs) | 148 | 75 |
| revision (changed by reflection) | 154 | 75 |

Dev holds 200 answers to 115 questions; test holds 100 answers to 57 questions.

## Decision log
| Decision | Options considered | Chosen | Why |
|---|---|---|---|
| Source of answers | Generate new answers; reuse saved runs | Saved runs | No API spend, and four writers give many kinds of failure |
| Stratify by | Writer only; writer, answered or not, and F1 band | All three | High-F1 answers can be wrong and low-F1 answers right; judges must see both |
| Split unit | Answer; question | Question | A judge tuned on dev must never see a test question |
| Stability | Recompute the sample; freeze it in a file | Freeze | A re-export with another row order swapped about 100 of 300 answers |
| Seeding | One generator; one per group | Per group | A change in one group can't ripple into the others |

## Cost & risk
No API cost. The data holds QASPER text, which is CC BY 4.0, and the README credits it. Risk: the export cannot run
outside one machine, so the committed files are the record.

## Open questions
- The merged answer is credited to the first writer that gave it. Another credit rule would move a few answers
  between groups.
