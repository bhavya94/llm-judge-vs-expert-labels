import json
import threading
import urllib.request

import pytest

from llm_judge_vs_expert_labels.data.qags import Sentence, Summary
from llm_judge_vs_expert_labels.judges.base import Verdict
from llm_judge_vs_expert_labels.labeling.server import BadLabel
from llm_judge_vs_expert_labels.llm.client import NotCached
from llm_judge_vs_expert_labels.review.disagreements import (ReviewServer, ReviewStore, find_disagreements,
                                                             summarize_reviews)

VAN = Summary("cnndm", "Police said three armed men robbed a security van in Glasgow on Monday. No one was hurt.",
              (Sentence("Three armed men robbed a security van in Glasgow.", (True, True, True)),
               Sentence("Two guards were taken to hospital.", (False, False, True))))
ROAD = Summary("xsum", "Heavy rain closed the A1 near Durham for six hours on Tuesday.",
               (Sentence("Flooding shut the A1 near Durham on Tuesday.", (True, True, False)),))
MUSEUM = Summary("xsum", "The city museum in Leeds reopens on 4 May after a £2m refurbishment.",
                 (Sentence("A Leeds museum reopens in May after a £2m revamp.", (True, True, True)),))
FERRY = Summary("cnndm", "Ferry crossings to Arran were cancelled on Friday because of high winds.",
                (Sentence("Ferries to Arran stopped for a week.", (False, True, False)),))


class FakeJudge:
    """Verdicts from a table. A summary not in it has no cached reply, as with a PromptedJudge that has no client."""

    verdicts = {VAN.id: Verdict(True, "Both sentences match the article."),
                ROAD.id: Verdict(False, "Sentence 1 says flooding, but the article says heavy rain."),
                MUSEUM.id: Verdict(True, "")}

    def judge(self, summary: Summary) -> Verdict:
        if summary.id not in self.verdicts:
            raise NotCached("sonnet: no cached reply for this request")
        return self.verdicts[summary.id]


@pytest.fixture
def app(tmp_path) -> ReviewServer:
    kept, _ = find_disagreements([VAN, MUSEUM, FERRY, ROAD], FakeJudge())
    return ReviewServer(kept, ReviewStore(tmp_path / "review-sonnet.jsonl"))


def _data(app: ReviewServer) -> dict:
    return json.loads(app.page().split("const DATA = ", 1)[1].split(";\n", 1)[0])


def test_only_disagreements_are_kept_and_a_summary_with_no_cached_verdict_is_counted():
    kept, skipped = find_disagreements([VAN, MUSEUM, FERRY, ROAD], FakeJudge())
    assert [(d.summary, d.verdict.passed) for d in kept] == [(VAN, True), (ROAD, False)] and skipped == 1


def test_the_page_shows_the_votes_both_labels_and_the_reason_but_no_position(app):
    row = _data(app)["items"][0]
    assert set(row) == {"id", "dataset", "article", "sentences", "human_label", "judge_verdict", "reason"}
    assert row["sentences"][1] == {"text": "Two guards were taken to hospital.", "votes": [False, False, True]}
    assert [row["human_label"], row["judge_verdict"]] == ["fail", "pass"]
    assert row["reason"] == "Both sentences match the article."


def test_a_review_posted_to_the_page_is_saved_and_a_reopened_page_resumes_after_it(app):
    server = app.server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    body = json.dumps({"summary_id": VAN.id, "right": "human", "note": "The article says no one was hurt."}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{server.server_port}/label", body,
                                                      {"Content-Type": "application/json"}))
    finally:
        server.shutdown()
    saved = json.loads(app.labels.path.read_text())
    assert saved | {"reviewed_at": ""} == {"summary_id": VAN.id, "dataset": "cnndm", "human_label": "fail",
                                           "judge_verdict": "pass", "right": "human", "reviewed_at": "",
                                           "note": "The article says no one was hurt."}
    reopened = _data(ReviewServer(list(app.items.values()), ReviewStore(app.labels.path)))
    assert [r["id"] for r in reopened["items"]] == [ROAD.id] and (reopened["done"], reopened["total"]) == (1, 2)


def test_a_review_saved_in_a_one_dataset_run_hides_no_earlier_summary(app):
    app.label({"summary_id": ROAD.id, "right": "judge", "note": "Heavy rain is not flooding."})
    assert [r["id"] for r in _data(app)["items"]] == [VAN.id]


def test_a_review_needs_a_shown_summary_who_was_right_and_a_note(app):
    for body in [{"summary_id": MUSEUM.id, "right": "human", "note": "Both agree, so it is never shown."},
                 {"summary_id": [VAN.id], "right": "human", "note": "The article says no one was hurt."},
                 {"summary_id": VAN.id, "right": "both", "note": "Hard to say."},
                 {"summary_id": VAN.id, "right": "judge", "note": "   "}]:
        with pytest.raises(BadLabel):
            app.label(body)
    assert app.labels.latest() == {}


def test_the_summary_counts_who_was_right_and_groups_the_notes_by_dataset(app):
    assert summarize_reviews(app.labels.latest().values()) == {}
    app.label({"summary_id": ROAD.id, "right": "unclear", "note": "Is heavy rain the same as flooding?"})
    app.label({"summary_id": VAN.id, "right": "judge", "note": "Misread sentence 2."})
    app.label({"summary_id": VAN.id, "right": "human", "note": "The judge missed that no one was hurt."})
    assert summarize_reviews(app.labels.latest().values()) == {
        "cnndm": {"counts": {"human": 1, "judge": 0, "unclear": 0},
                  "notes": [("human", "The judge missed that no one was hurt.")]},
        "xsum": {"counts": {"human": 0, "judge": 0, "unclear": 1},
                 "notes": [("unclear", "Is heavy rain the same as flooding?")]}}
