import math

import pytest
from metrics_from_scratch.stats.bootstrap import Bootstrap

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.records import Candidate, Question, Reference
from llm_judge_vs_expert_labels.evaluation.agreement import Agreement
from llm_judge_vs_expert_labels.judges.code import CodeChecks, F1Threshold

ANSWERED = Question("q-datasets", "1705.01265", "Feature set augmentation", "Which datasets do they evaluate on?",
                    (Reference("SemEval 2016 Task 4", "extractive", ("s3p1",)),))
UNANSWERED = Question("q-baseline", "1705.01265", "Feature set augmentation", "What baseline do they compare to?",
                      (Reference("Unanswerable", "none", ()), Reference("Unanswerable", "none", ())))
BOOTSTRAP = Bootstrap(n_resamples=500, seed=0)


@pytest.fixture
def checks() -> CodeChecks:
    return CodeChecks(Corpus([ANSWERED, UNANSWERED], {}, []), max_words=35)


def answer(question: Question, text: str, cited: tuple[str, ...] = ("s3p1",), f1: float = 0.0) -> Candidate:
    return Candidate(question.id, text, cited, cited, ("sonnet-k20",), False, f1)


def test_a_short_cited_answer_passes_the_code_checks(checks):
    assert checks.judge(answer(ANSWERED, "SemEval 2016 Task 4")).passed


@pytest.mark.parametrize("candidate, reason", [
    (answer(ANSWERED, "Unanswerable", cited=()), "every reference answers"),
    (answer(UNANSWERED, "A bag-of-words SVM"), "every reference says the paper doesn't"),
    (answer(ANSWERED, "SemEval 2016 Task 4", cited=()), "cites no paragraph"),
    (answer(ANSWERED, " ".join(["They evaluate on the SemEval benchmark"] * 6)), "36 words"),
])
def test_each_code_check_fails_with_its_reason(checks, candidate, reason):
    verdict = checks.judge(candidate)
    assert not verdict.passed and reason in verdict.reason


def test_unanswerable_passes_when_every_reference_agrees(checks):
    assert checks.judge(answer(UNANSWERED, "Unanswerable", cited=())).passed


def test_the_f1_threshold_is_inclusive():
    judge = F1Threshold(0.5)
    assert judge.judge(answer(ANSWERED, "SemEval", f1=0.5)).passed
    assert not judge.judge(answer(ANSWERED, "SemEval", f1=0.49)).passed


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
