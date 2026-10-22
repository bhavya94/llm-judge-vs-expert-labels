import json

import pytest
from conftest import FakeClient

from llm_judge_vs_expert_labels.data.qags import Sentence, Summary
from llm_judge_vs_expert_labels.judges.prompted import PromptedJudge, parse, render
from llm_judge_vs_expert_labels.llm.client import BudgetExceeded, CachedLLM, Meter, NotCached

ANSWER = Summary("cnndm", "Police said three armed men robbed a van in Glasgow on Monday.",
                 (Sentence("Three men robbed a van.", (True, True, True)),
                  Sentence("It happened in Glasgow.", (True, True, False))))
PASS = json.dumps({"reason": "Both sentences match the article.", "verdict": "pass"})


def _judge(client, tmp_path, cap=1.0) -> PromptedJudge:
    llm = CachedLLM(client, "test-model", Meter(input_price=1.0, output_price=5.0, cap=cap), cache=tmp_path)
    return PromptedJudge("test", llm)


def test_the_prompt_shows_the_article_and_each_sentence_but_not_the_votes():
    prompt = render(ANSWER)
    assert prompt.startswith("Article:\nPolice said three armed men")
    assert "1. Three men robbed a van.\n2. It happened in Glasgow." in prompt
    assert "True" not in prompt and "cnndm" not in prompt


def test_a_reply_becomes_a_verdict_with_its_reason(tmp_path):
    verdict = _judge(FakeClient(PASS), tmp_path).judge(ANSWER)
    assert verdict.passed and verdict.reason == "Both sentences match the article."


@pytest.mark.parametrize("reply, passed", [
    ('Here it is: {"reason": "Wrong dataset.", "verdict": "fail"} Hope that helps.', False),
    ('{"reason": "a\\nb", "verdict": "pass"}', True),
])
def test_parse_finds_the_verdict_in_surrounding_text(reply, passed):
    assert parse(reply).passed is passed


@pytest.mark.parametrize("reply", ["Pass.", '{"verdict": "maybe"}', '{"verdict": "pass"', "[1, 2]"])
def test_an_unreadable_reply_counts_as_fail_and_is_counted(tmp_path, reply):
    judge = _judge(FakeClient(reply, reply), tmp_path)
    verdict = judge.judge(ANSWER)
    assert not verdict.passed and "unreadable" in verdict.reason and judge.unreadable == 1


def test_a_cut_off_reply_is_asked_again_once_with_a_higher_limit(tmp_path):
    client = FakeClient('{"reason": "Sentence 2 adds a date the', PASS)
    judge = _judge(client, tmp_path)
    assert judge.judge(ANSWER).passed and judge.unreadable == 0
    assert [r["max_tokens"] for r in client.requests] == [judge.llm.max_tokens, judge.retry_max_tokens]


def test_the_cache_makes_a_rerun_free_and_offline(tmp_path):
    _judge(FakeClient(PASS), tmp_path).judge(ANSWER)
    offline = _judge(None, tmp_path)
    assert offline.judge(ANSWER).passed and offline.llm.calls == 0


def test_an_offline_judge_refuses_an_uncached_answer(tmp_path):
    with pytest.raises(NotCached):
        _judge(None, tmp_path).judge(ANSWER)


def test_the_budget_refuses_a_call_that_could_pass_the_cap(tmp_path):
    client = FakeClient(PASS)
    with pytest.raises(BudgetExceeded, match="cap"):
        _judge(client, tmp_path, cap=0.0001).judge(ANSWER)
    assert client.requests == []


def test_the_estimate_grows_with_the_answers_and_spends_nothing(tmp_path):
    client = FakeClient()
    judge = _judge(client, tmp_path)
    one = judge.estimate([ANSWER])
    assert one > judge.llm.max_tokens * 5.0 / 1e6  # every reply at full length, plus the input
    assert judge.estimate([ANSWER] * 3) == pytest.approx(3 * one) and client.requests == []
