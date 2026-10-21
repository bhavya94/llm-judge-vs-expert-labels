"""Draw the QAGS dev/test split once and write it to data/qags_split.json. Run after `ljvel download`."""

from llm_judge_vs_expert_labels.data.qags import FROZEN, Release, freeze, split_from_config

split = split_from_config(Release.from_config().summaries())
freeze(split)
print(f"dev {len(split.dev)}, test {len(split.test)} -> {FROZEN}")
