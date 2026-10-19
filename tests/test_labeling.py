import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.sample import group, load
from llm_judge_vs_expert_labels.labeling.items import Item
from llm_judge_vs_expert_labels.labeling.labels import Label, LabelStore
from llm_judge_vs_expert_labels.labeling.server import BadLabel, Rounds
from llm_judge_vs_expert_labels.labeling.summary import summarize


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load()


@pytest.fixture(scope="module")
def sample(corpus):
    return load(corpus.candidates())[0]


@pytest.fixture(scope="module")
def items(corpus, sample) -> list[Item]:
    return [Item.of(c, corpus) for c in sample]


@pytest.fixture
def rounds(tmp_path) -> Rounds:
    return Rounds(LabelStore(tmp_path / "first.jsonl"), LabelStore(tmp_path / "relabel.jsonl"), 60, 1, 0)


def test_every_answer_to_a_question_shows_the_same_paragraphs_in_paper_order(corpus):
    """Otherwise the number of paragraphs would tell the labeler which writer answered: none closed-book, 5 or 20."""
    by_question: dict[str, list] = {}
    for c in corpus.candidates():
        by_question.setdefault(c.question_id, []).append(c)
    answers = next(cs for _, cs in by_question.items() if len({group(c) for c in cs}) >= 3)
    shown = [[p.id for p in Item.of(c, corpus).passages] for c in answers]
    assert all(ids == shown[0] for ids in shown)
    order = [(int(i[1:].split("p")[0]), int(i.split("p")[1])) for i in shown[0]]
    assert order == sorted(order)


def test_passages_flag_this_answers_citations_and_the_expert_evidence(corpus, sample):
    candidate = next(c for c in sample if c.cited)
    item = Item.of(candidate, corpus)
    expert = {pid for r in corpus.questions[candidate.question_id].references for pid in r.evidence_ids}
    assert {p.id for p in item.passages if p.expert} == expert
    assert {p.id for p in item.passages if p.cited} == set(candidate.cited)


def test_the_page_carries_no_writer_or_answer_f1(rounds, items):
    page = rounds.first_round(items).page()
    for hint in ["sonnet-k20", "haiku-k5", "closed-book", "answer_f1", '"seen"', '"sources"']:
        assert hint not in page


def test_a_fail_needs_a_note_and_the_verdict_must_be_pass_or_fail(items):
    with pytest.raises(ValueError, match="note"):
        Label.of(items[0], "fail", "   ")
    with pytest.raises(ValueError, match="verdict"):
        Label.of(items[0], "maybe", "")
    assert Label.of(items[0], "pass", "").verdict == "pass"


def test_the_newest_label_of_an_answer_wins(tmp_path, items):
    store = LabelStore(tmp_path / "first.jsonl")
    store.save(Label.of(items[0], "pass", ""))
    store.save(Label.of(items[0], "fail", "Says Unanswerable, but s3p1 names the dataset"))
    store.save(Label.of(items[1], "pass", ""))
    latest = store.latest()
    assert latest[items[0].id].verdict == "fail" and latest[items[1].id].verdict == "pass"
    assert len((tmp_path / "first.jsonl").read_text().splitlines()) == 3


def test_an_unknown_answer_or_a_bad_label_is_refused_and_not_saved(rounds, items):
    app = rounds.first_round(items)
    with pytest.raises(BadLabel):
        app.label({"candidate_id": "000000000000", "verdict": "pass"})
    with pytest.raises(BadLabel):
        app.label({"candidate_id": items[0].id, "verdict": "fail", "note": ""})
    assert rounds.first.latest() == {}


def test_answer_text_cannot_end_the_page_script(rounds, items):
    hostile = replace(items[0], answer="</script><script>alert(1)</script>")
    page = rounds.first_round([hostile]).page()
    assert "</script><script>alert(1)" not in page


def test_relabel_waits_for_the_first_pass_then_is_blind_and_reordered(rounds, items):
    with pytest.raises(ValueError, match="no first label"):
        rounds.relabel_round(items)
    for item in items:
        rounds.first.save(Label.of(item, "fail", "Names BERT, but the paper fine-tunes RoBERTa"))
    app = rounds.relabel_round(items)
    assert len(app.order) == 60 and app.order != [i.id for i in items[:60]]
    assert app.order == rounds.relabel_round(items).order, "the same seed gives the same relabel set"
    data = json.loads(app.page().split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert data["blind"] and data["labels"] == {}, "first labels never reach the relabel page"


def test_a_label_posted_to_the_server_is_saved_and_shown_on_reload(rounds, items):
    server = rounds.first_round(items[:3]).server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        body = json.dumps({"candidate_id": items[1].id, "verdict": "pass", "note": ""}).encode()
        request = urllib.request.Request(url + "/label", body, {"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(request).read())["verdict"] == "pass"
        assert items[1].id in rounds.first.latest()
        assert f'"{items[1].id}": {{"candidate_id"' in urllib.request.urlopen(url + "/").read().decode()
    finally:
        server.shutdown()


def test_a_bad_request_is_refused_without_hanging_the_server(rounds, items):
    server = rounds.first_round(items[:3]).server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/label"
    try:
        for body in [b"{\"note\": \"\x80\x81\"}", b"[1, 2]", json.dumps({"candidate_id": items[0].id, "verdict": "fail",
                                                                       "note": None}).encode()]:
            request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            assert error.value.code == 400
        assert rounds.first.latest() == {}
    finally:
        server.shutdown()


def test_summary_counts_labels_in_the_sample_only(sample):
    labels = {sample[0].id: Label(sample[0].id, "pass", "", sample[0].question_id, sample[0].answer, "2026-10-19"),
              "000000000000": Label("000000000000", "fail", "Wrong dataset", "q", "SQuAD", "2026-10-19")}
    summary = summarize(sample, labels)
    assert summary["labeled"] == 1 and summary["notes"] == []
    assert summary["by_group"] == {group(sample[0]): {"pass": 1, "fail": 0}}
