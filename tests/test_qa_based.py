import json

import pytest
from conftest import FakeClient

from llm_judge_vs_expert_labels.data.qags import Sentence, Summary
from llm_judge_vs_expert_labels.judges.qa_based import QagsJudge, token_f1
from llm_judge_vs_expert_labels.llm.client import CachedLLM, Meter

SUMMARY = Summary("xsum", "Police said three armed men robbed a van in Glasgow on Monday.",
                  (Sentence("Three armed men robbed a van in Edinburgh.", (False, False, True)),))


def questions(*pairs: tuple[str, str]) -> str:
    return json.dumps({"questions": [{"answer": a, "question": q} for a, q in pairs]})


def answers(*texts: str) -> str:
    return json.dumps({"answers": list(texts)})


def _judge(client, tmp_path, threshold=0.5) -> QagsJudge:
    llm = CachedLLM(client, "test-model", Meter(input_price=1.0, output_price=5.0, cap=1.0), cache=tmp_path)
    return QagsJudge("test", llm, threshold)


@pytest.mark.parametrize("predicted, gold, f1", [
    ("Glasgow", "glasgow", 1.0),
    ("the van", "a van!", 1.0),
    ("three armed men", "three men", 0.8),
    ("Edinburgh", "Glasgow", 0.0),
])
def test_token_f1_compares_normalized_answers(predicted, gold, f1):
    assert token_f1(predicted, gold) == pytest.approx(f1)


def test_matching_answers_pass_with_score_one(tmp_path):
    client = FakeClient(questions(("three", "How many men robbed the van?")), answers("three"), answers("Three"))
    verdict = _judge(client, tmp_path).judge(SUMMARY)
    assert verdict.passed and verdict.score == 1.0


def test_an_answer_the_article_does_not_give_scores_zero_and_is_named(tmp_path):
    client = FakeClient(questions(("three", "How many men robbed the van?"), ("Edinburgh", "Where was the van robbed?")),
                        answers("three", "Edinburgh"), answers("three", "unanswerable"))
    verdict = _judge(client, tmp_path).judge(SUMMARY)
    assert verdict.score == 0.5 and "Where was the van robbed?" in verdict.reason and "unanswerable" in verdict.reason


def test_the_threshold_is_inclusive(tmp_path):
    replies = (questions(("three", "How many men robbed it?"), ("Edinburgh", "Where was it robbed?")),
               answers("three", "Edinburgh"), answers("three", "Glasgow"))
    assert _judge(FakeClient(*replies), tmp_path / "a", threshold=0.5).judge(SUMMARY).passed
    assert not _judge(FakeClient(*replies), tmp_path / "b", threshold=0.51).judge(SUMMARY).passed


def test_short_duplicate_and_summary_unanswerable_questions_are_dropped(tmp_path):
    client = FakeClient(questions(("x", "Who?"), ("three", "How many men robbed the van?"),
                                  ("three", "How many men robbed the van?"), ("Monday", "On which day was it?")),
                        answers("three", "unanswerable"), answers("three", "Monday"))
    judge = _judge(client, tmp_path)
    verdict = judge.judge(SUMMARY)
    assert "over 1 questions" in verdict.reason and verdict.score == 1.0
    assert "1. How many men robbed the van?\n2. On which day was it?" in client.requests[1]["messages"][0]["content"]


def test_each_text_is_asked_separately_so_the_article_call_never_sees_the_summary(tmp_path):
    client = FakeClient(questions(("Edinburgh", "Where was the van robbed?")), answers("Edinburgh"),
                        answers("Glasgow"))
    _judge(client, tmp_path).judge(SUMMARY)
    to_summary, to_article = (r["messages"][0]["content"] for r in client.requests[1:])
    assert SUMMARY.text in to_summary and SUMMARY.article not in to_summary
    assert SUMMARY.article in to_article and SUMMARY.text not in to_article


def test_a_summary_with_no_usable_question_passes_and_is_counted(tmp_path):
    judge = _judge(FakeClient(questions(("x", "Who?"))), tmp_path)
    verdict = judge.judge(SUMMARY)
    assert verdict.passed and verdict.score == 1.0 and judge.no_questions == 1


def test_answers_that_do_not_line_up_with_the_questions_count_as_unreadable(tmp_path):
    client = FakeClient(questions(("three", "How many men robbed the van?")), answers("three", "extra"),
                        answers("three"))
    judge = _judge(client, tmp_path)
    assert not judge.judge(SUMMARY).passed and judge.unreadable == 1


def test_a_cut_off_reply_is_asked_again_once_with_a_higher_limit(tmp_path):
    client = FakeClient('{"questions": [{"answer": "three", "question": "How many',
                        questions(("three", "How many men robbed the van?")), answers("three"), answers("three"))
    judge = _judge(client, tmp_path)
    assert judge.judge(SUMMARY).passed
    assert [r["max_tokens"] for r in client.requests[:2]] == [judge.llm.max_tokens, judge.retry_max_tokens]


def test_the_estimate_grows_with_the_summaries_and_spends_nothing(tmp_path):
    client = FakeClient()
    judge = _judge(client, tmp_path)
    one = judge.estimate([SUMMARY])
    assert one > 3 * judge.llm.max_tokens * 5.0 / 1e6
    assert judge.estimate([SUMMARY] * 4) == pytest.approx(4 * one) and client.requests == []
