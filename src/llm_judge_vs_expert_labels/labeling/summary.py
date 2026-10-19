"""Progress and a first look at the labels: Pass/Fail per writer group, where answer-F1 disagrees, and the notes that
become failure modes."""

from llm_judge_vs_expert_labels.data.records import Candidate
from llm_judge_vs_expert_labels.data.sample import group
from llm_judge_vs_expert_labels.labeling.labels import Label


def summarize(sample: list[Candidate], labels: dict[str, Label]) -> dict:
    """Labels of answers outside the sample are ignored."""
    labeled = [(c, labels[c.id]) for c in sample if c.id in labels]
    by_group: dict[str, dict[str, int]] = {}
    for c, label in labeled:
        counts = by_group.setdefault(group(c), {"pass": 0, "fail": 0})
        counts[label.verdict] += 1
    return {"sample": len(sample), "labeled": len(labeled), "by_group": by_group,
            "pass_low_f1": sum(label.verdict == "pass" and c.answer_f1 < 0.5 for c, label in labeled),
            "fail_high_f1": sum(label.verdict == "fail" and c.answer_f1 >= 0.5 for c, label in labeled),
            "notes": [label.note for c, label in labeled if label.verdict == "fail"]}
