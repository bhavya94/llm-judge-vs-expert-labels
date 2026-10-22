PR: [bhavya94/llm-judge-vs-expert-labels#8](https://github.com/bhavya94/llm-judge-vs-expert-labels/pull/8)

# Prompted LLM judges on QAGS summaries

## Summary
This PR ports the prompted judges from #5 to QAGS. Haiku 4.5 and Sonnet 5.5 read the article and the numbered
summary sentences, as the workers did. Each writes a reason, then Pass or Fail. A dry run prints the worst-case
cost. A paid run stays under a hard cap, and every reply is cached.

## How it works
```mermaid
flowchart LR
  S["Summary"] --> R["render: article +<br/>numbered sentences"]
  R --> L["CachedLLM: cache, then cap, then API"]
  L --> P["parse: reason + verdict"]
  P --> A["ljvel agree:<br/>cached replies only"]
```
- [`judges/prompted.py:19`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/62efd6cc9f35eef0b9abb4b401cc08b4d081c0ed/src/llm_judge_vs_expert_labels/judges/prompted.py#L19) is the prompt. A summary passes only when every
  sentence is consistent.
- [`judges/prompted.py:29`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/62efd6cc9f35eef0b9abb4b401cc08b4d081c0ed/src/llm_judge_vs_expert_labels/judges/prompted.py#L29) writes the article and the sentences. It never
  writes the votes.
- [`cli.py:111`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/62efd6cc9f35eef0b9abb4b401cc08b4d081c0ed/src/llm_judge_vs_expert_labels/cli.py#L111) adds a prompted judge to `agree` once all of dev is cached.
- [`tests/test_llm_judges.py:34`](https://github.com/bhavya94/llm-judge-vs-expert-labels/blob/62efd6cc9f35eef0b9abb4b401cc08b4d081c0ed/tests/test_llm_judges.py#L34) checks that the votes and the dataset stay
  out of the prompt.

## Decision log
| Decision | Options considered | Chosen | Why |
|---|---|---|---|
| Unit judged | Each sentence; whole summary | Whole summary, sentences numbered | One call per summary; the label is per summary |
| Reason or verdict first | Verdict first; reason first | Reason first | The verdict follows from the reason |
| #5 | Rebase it; new PR | New PR, stacked | #5 was built on QASPER items |

## Cost & risk
Worst case on 315 dev summaries: $0.76 for Haiku, $2.28 for Sonnet. Nothing was spent. Check the Sonnet 5.5
price before a paid run.

## Open questions
- A per-sentence judge would match the label's structure more closely, at up to four calls per CNN/DM summary.
