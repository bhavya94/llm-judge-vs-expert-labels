import math
import warnings

import pytest
from metrics_from_scratch.stats.bootstrap import Bootstrap

from llm_judge_vs_expert_labels.data.qags import Sentence, Summary
from llm_judge_vs_expert_labels.evaluation.agreement import Agreement, correlation
from llm_judge_vs_expert_labels.judges.code import WordOverlap

ARTICLE = "Police said three armed men took a five-figure sum from the van in Glasgow on Monday."
BOOTSTRAP = Bootstrap(n_resamples=500, seed=0)


def summary(*sentences: tuple[str, tuple[bool, ...]], article: str = ARTICLE) -> Summary:
    return Summary("xsum", article, tuple(Sentence(text, votes) for text, votes in sentences))


def test_a_summary_made_of_article_words_passes_the_strictest_overlap():
    assert WordOverlap(1.0).judge(summary(("Three armed men took a sum from the van.", (True,) * 3))).passed


def test_a_new_word_fails_the_strict_cut_and_is_named_in_the_reason():
    verdict = WordOverlap(1.0).judge(summary(("Three armed men took a sum in Edinburgh.", (True,) * 3)))
    assert not verdict.passed and "edinburgh" in verdict.reason


def test_the_overlap_threshold_is_inclusive():
    # 9 of 10 words are in the article.
    text = "Three armed men took a sum from the van Tuesday"
    assert WordOverlap(0.9).judge(summary((text, (True,) * 3))).passed
    assert not WordOverlap(0.91).judge(summary((text, (True,) * 3))).passed


def test_overlap_ignores_case_and_punctuation():
    assert WordOverlap(1.0).judge(summary(("POLICE: three armed men -- Glasgow!", (True,) * 3))).passed


def test_one_workers_label_needs_a_yes_on_every_sentence():
    s = summary(("First.", (True, True, False)), ("Second.", (True, False, True)))
    assert s.consistent and s.by_worker(0) and not s.by_worker(1) and not s.by_worker(2)


def test_agreement_on_a_worked_example():
    # Person: 6 Pass, 4 Fail. Judge passes 5 of the 6 Passes and 1 of the 4 Fails.
    human = [True] * 6 + [False] * 4
    judged = [True] * 5 + [False] + [True] + [False] * 3
    a = Agreement.of(human, judged, BOOTSTRAP)
    assert a.tpr.estimate == pytest.approx(5 / 6) and a.tnr.estimate == pytest.approx(3 / 4)
    # Observed 0.8; chance 0.6 * 0.6 + 0.4 * 0.4 = 0.52; kappa (0.8 - 0.52) / 0.48.
    assert a.kappa.estimate == pytest.approx(0.28 / 0.48)
    assert a.tpr.low <= a.tpr.estimate <= a.tpr.high


def test_a_judge_that_passes_everything_catches_no_fail_and_has_zero_kappa():
    a = Agreement.of([True, True, False, True, False], [True] * 5, BOOTSTRAP)
    assert a.tpr.estimate == 1 and a.tnr.estimate == 0 and a.kappa.estimate == 0


def test_tnr_is_undefined_without_any_fail_label():
    a = Agreement.of([True, True, True], [True, False, True], BOOTSTRAP)
    assert math.isnan(a.tnr.estimate)


def test_verdicts_must_pair_with_labels():
    with pytest.raises(ValueError):
        Agreement.of([True, False], [True], BOOTSTRAP)


def test_the_overlap_score_is_the_share_of_words_in_the_article():
    # 9 of 10 words are in the article.
    text = "Three armed men took a sum from the van Tuesday"
    assert WordOverlap(1.0).judge(summary((text, (True,) * 3))).score == 0.9  # a Fail
    assert WordOverlap(0.9).judge(summary((text, (True,) * 3))).score == 0.9  # a Pass


def test_the_human_score_is_the_share_of_consistent_sentences():
    s = summary(("First.", (True, True, False)), ("Second.", (True, False, False)))
    assert s.human_score == 0.5


def test_correlation_on_a_worked_example():
    # y = 2x + 1 exactly, then one point moved: r is 1, then below 1.
    assert correlation([0, 1, 2, 3], [1, 3, 5, 7], BOOTSTRAP).estimate == pytest.approx(1.0)
    assert correlation([0, 1, 2, 3], [1, 3, 5, 0], BOOTSTRAP).estimate < 0


def test_correlation_with_a_constant_judge_is_undefined_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(correlation([0.0, 0.5, 1.0], [1.0, 1.0, 1.0], BOOTSTRAP).estimate)
