from collections import Counter

import pytest

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.records import Candidate
from llm_judge_vs_expert_labels.data.sample import Sampler, cell, freeze, group, load

QID = "a4a1fcef760b133e9aa876ac28145ad98a609927"


def answer(answer: str, source: str = "sonnet-k20", kind: str = "first", question_id: str = QID, f1: float = 0.4):
    return {"source": source, "kind": kind, "model": "claude-sonnet-5-5", "question_id": question_id,
            "answer": answer, "cited": ["s3p1"], "seen": ["s3p1", "s2p0"], "answer_f1": f1}


def candidate(question_id: str, answer: str, source: str, f1: float, revision: bool = False) -> Candidate:
    return Candidate(question_id, answer, (), (), (source,), revision, f1)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load()


def test_case_and_spacing_differences_are_one_candidate_with_every_source():
    corpus = Corpus([], {}, [answer("Selection of word vectors"),
                             answer("selection of  word vectors", source="haiku-self-critique", kind="revision"),
                             answer("The number of epochs", source="haiku-k5")])
    merged, other = corpus.candidates()
    assert merged.sources == ("sonnet-k20", "haiku-self-critique")
    assert not merged.revision, "a revision that repeats a first answer counts as that first answer"
    assert other.answer == "The number of epochs"


def test_candidate_ids_do_not_change():
    """Labels point at candidates by id; a changed id would orphan every saved label."""
    assert candidate(QID, "Selection of word vectors", "sonnet-k20", 1.0).id == "d3c4df78ea95"
    assert candidate(QID, " selection of word  VECTORS", "haiku-k5", 1.0).id == "d3c4df78ea95"


def test_sample_takes_an_equal_share_from_each_group(corpus):
    sample = Sampler(300, 100, seed=0).sample(corpus.candidates())
    assert len({c.id for c in sample}) == 300
    assert Counter(group(c) for c in sample) == {
        "haiku-k5": 75, "revision": 75, "sonnet-closed-book": 75, "sonnet-k20": 75}


def test_sample_takes_rare_cells_before_common_ones():
    # 4 unanswerable answers, 40 answers with F1 below 0.5: a share of 10 must hold all 4 rare ones.
    rare = [candidate(f"q{i}", "Unanswerable", "haiku-k5", 0.0) for i in range(4)]
    common = [candidate(f"q{i}", f"BiLSTM with {i} layers", "haiku-k5", 0.3) for i in range(40)]
    sample = Sampler(10, 3, seed=0).sample(rare + common)
    assert Counter(cell(c) for c in sample) == {(True, "zero"): 4, (False, "below_half"): 6}


def test_the_frozen_sample_is_what_the_config_draws(corpus):
    """Labels point at the frozen answers. If a re-export or a code change would draw different ones, this fails
    before any label is orphaned."""
    sampler = Sampler.from_config()
    drawn = sampler.sample(corpus.candidates())
    sample, split = load(corpus.candidates())
    assert [c.id for c in sample] == [c.id for c in drawn]
    assert [c.id for c in split.test] == [c.id for c in sampler.split(drawn).test]


def test_a_change_in_one_group_leaves_the_other_groups_picks_alone(corpus):
    candidates = corpus.candidates()
    without_one = [c for c in candidates if not (group(c) == "haiku-k5" and c.id < "4")]
    picks = lambda cs: {c.id for c in Sampler(300, 100, seed=0).sample(cs) if group(c) != "haiku-k5"}
    assert picks(without_one) == picks(candidates)


def test_freeze_then_load_gives_the_same_order_and_split(tmp_path, corpus):
    sampler = Sampler(300, 100, seed=7)
    sample = sampler.sample(corpus.candidates())
    split = sampler.split(sample)
    freeze(sample, split, tmp_path / "sample.json")
    loaded, loaded_split = load(corpus.candidates(), tmp_path / "sample.json")
    assert loaded == sample and loaded_split == split


def test_a_group_too_small_for_its_share_is_an_error():
    small = [candidate(f"q{i}", "SVM", "haiku-k5", 0.5) for i in range(3)]
    large = [candidate(f"q{i}", "SVM", "sonnet-k20", 0.5) for i in range(30)]
    with pytest.raises(ValueError, match="haiku-k5"):
        Sampler(20, 5, seed=0).sample(small + large)


def test_no_question_has_answers_in_both_dev_and_test(corpus):
    sample, split = load(corpus.candidates())
    assert {c.question_id for c in split.dev}.isdisjoint({c.question_id for c in split.test})
    assert sorted(c.id for c in split.dev + split.test) == sorted(c.id for c in sample)
    assert len(split.test) >= Sampler.from_config().test_size


def test_every_paragraph_a_labeler_needs_was_exported(corpus):
    for c in corpus.candidates():
        expert = [pid for r in corpus.questions[c.question_id].references for pid in r.evidence_ids]
        for pid in c.seen + c.cited + tuple(expert):
            assert corpus.paragraph(c.question_id, pid).text
