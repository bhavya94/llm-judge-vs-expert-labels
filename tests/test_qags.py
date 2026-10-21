import hashlib
import json

import pytest

from llm_judge_vs_expert_labels.data import qags
from llm_judge_vs_expert_labels.data.qags import Release, Sentence, Summary, split


def row(article: str, *sentences: tuple[str, list[str]]) -> dict:
    return {"article": article, "summary_sentences": [
        {"sentence": text, "responses": [{"worker_id": i, "response": r} for i, r in enumerate(votes)]}
        for text, votes in sentences]}


ROWS = {"cnndm": [row("A van was robbed.", ("Robbers took a van.", ["yes", "yes", "no"]),
                      ("Two guards were hurt.", ["no", "no", "yes"]))],
        "xsum": [row("Rain fell in Leeds.", ("It rained in Leeds.", ["yes", "yes", "yes"]))]}


@pytest.fixture
def remote(tmp_path):
    """A stand-in for the pinned GitHub files, served from disk over file://."""
    (tmp_path / "remote").mkdir()
    digests = {}
    for name, rows in ROWS.items():
        data = "".join(json.dumps(r) + "\n" for r in rows).encode()
        (tmp_path / "remote" / f"mturk_{name}.jsonl").write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    return Release((tmp_path / "remote").as_uri(), digests, tmp_path / "local")


def test_download_checks_the_hash_and_loads_every_summary(remote):
    remote.download()
    summaries = remote.summaries()
    assert [s.dataset for s in summaries] == ["cnndm", "xsum"]
    assert summaries[0].text == "Robbers took a van. Two guards were hurt."


def test_a_file_that_does_not_match_its_hash_is_deleted(remote):
    remote.files["xsum"] = "0" * 64
    with pytest.raises(ValueError, match="mturk_xsum.jsonl does not match"):
        remote.download()
    assert not remote.path("xsum").exists()


def test_summaries_before_download_say_what_to_run(remote):
    with pytest.raises(FileNotFoundError, match="ljvel download"):
        remote.summaries()


def test_a_sentence_is_consistent_when_most_workers_say_yes():
    assert Sentence("s", (True, True, False)).consistent
    assert not Sentence("s", (True, False, False)).consistent
    assert not Sentence("s", (True, False)).consistent  # a tie is not a majority


def test_a_summary_is_consistent_only_when_every_sentence_is(remote):
    remote.download()
    cnndm, xsum = remote.summaries()
    assert not cnndm.consistent and xsum.consistent


def _summaries(n: int) -> list[Summary]:
    return [Summary(dataset, f"article {i}", (Sentence(f"summary {i}", (i % 2 == 0,) * 3),))
            for dataset in ["cnndm", "xsum"] for i in range(n)]


def test_the_split_keeps_each_dataset_and_label_mix_in_both_sides():
    s = split(_summaries(60), test_share=1 / 3, seed=0)
    assert not {x.id for x in s.dev} & {x.id for x in s.test} and len(s.dev) + len(s.test) == 120
    for dataset in ["cnndm", "xsum"]:
        for label in [True, False]:
            assert sum(x.dataset == dataset and x.consistent == label for x in s.test) == 10


def test_the_split_is_the_same_for_the_same_seed_and_differs_for_another():
    first, again, other = (split(_summaries(60), 1 / 3, seed) for seed in [0, 0, 1])
    assert [x.id for x in first.test] == [x.id for x in again.test] != [x.id for x in other.test]


def test_load_refuses_a_frozen_id_with_no_summary(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps([{"id": "000000000000", "split": "dev"}]))
    with pytest.raises(ValueError, match="1 frozen summaries"):
        qags.load(_summaries(2), path)


@pytest.mark.skipif(not qags.Release.from_config().path("xsum").exists(), reason="run `ljvel download` first")
def test_the_frozen_split_matches_a_fresh_draw():
    summaries = qags.Release.from_config().summaries()
    frozen, fresh = qags.load(summaries), qags.split_from_config(summaries)
    assert [s.id for s in frozen.test] == [s.id for s in fresh.test] and len(frozen.dev) + len(frozen.test) == 474
