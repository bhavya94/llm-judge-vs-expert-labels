"""Export every saved answer from reflection-agent-for-paper-qa as a judging candidate. Run once, in that repo's
environment, because its answers and the QASPER files live only on that machine:

    uv run --project ~/reflective-qa python scripts/export_candidates.py data

It writes three JSON lines files: candidates (one per saved answer, with the paragraphs its writer saw and cited),
questions (with the annotators' reference answers) and paragraphs (only those some candidate or reference uses).
Only the current run files count: a run whose config changed since has a different fingerprint and is skipped.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from reflective_qa.agent.reflection import ReflectionAgent
from reflective_qa.agent.writer import Writer
from reflective_qa.data.qasper import QasperSplit
from reflective_qa.eval.metrics import answer_f1
from reflective_qa.experiments.base import Experiment
from reflective_qa.paths import DEV_SPLIT
from reflective_qa.retrieval.bm25 import Retriever
from reflective_qa.runs.store import RunStore


def main(out: Path) -> None:
    split, store = QasperSplit.load(DEV_SPLIT), RunStore()
    questions = split.sample(*Experiment.sample_config(None))
    retriever = Retriever(split)
    seen = {(name, q.id): [p.id for p in retriever.retrieve(q, writer.k)]
            for name, writer in Writer.load_all().items() for q in questions}

    def candidate(source: str, kind: str, writer: Writer, row: dict, question) -> dict:
        return {"source": source, "kind": kind, "model": writer.model, "question_id": question.id,
                "answer": row["predicted_answer"], "cited": list(row["predicted_evidence"]),
                "seen": seen[writer.name, question.id], "answer_f1": answer_f1(question, row["predicted_answer"])[0]}

    rows = []
    for name, writer in Writer.load_all().items():
        answers = {r["question_id"]: r for r in store.answers(writer)}
        rows += [candidate(name, "first", writer, answers[q.id], q) for q in questions
                 if q.id in answers and answers[q.id]["status"] == "ok"]
    for name, agent in ReflectionAgent.load_all().items():
        revised = store.stage(agent, "revise")
        rows += [candidate(name, "revision", agent.writer, revised[q.id], q) for q in questions
                 if q.id in revised and revised[q.id]["status"] == "ok"]

    paper_of = {q.id: q.paper_id for q in questions}
    used = {(paper_of[r["question_id"]], pid) for r in rows for pid in r["seen"] + r["cited"]}
    used |= {(q.paper_id, pid) for q in questions for a in q.answers for pid in a.evidence_ids}
    paragraphs = [{"paper_id": paper.id, **asdict(p)} for paper in split.papers for p in paper.paragraphs
                  if (paper.id, p.id) in used]
    question_rows = [{"question_id": q.id, "paper_id": q.paper_id, "paper_title": split.paper(q.paper_id).title,
                      "question": q.text, "references": [{"text": a.text, "type": a.type.value,
                                                          "evidence_ids": list(a.evidence_ids)} for a in q.answers]}
                     for q in questions]

    out.mkdir(parents=True, exist_ok=True)
    for name, items in [("candidates", rows), ("questions", question_rows), ("paragraphs", paragraphs)]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
        sys.stderr.write(f"{name}: {len(items)}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
