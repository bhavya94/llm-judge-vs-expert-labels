import json
from types import SimpleNamespace

import pytest

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.records import (
    Candidate,
    Paragraph,
    Question,
    Reference,
)
from llm_judge_vs_expert_labels.judges.prompted import PromptedJudge, parse
from llm_judge_vs_expert_labels.llm.client import (
    BudgetExceeded,
    CachedLLM,
    Meter,
    NotCached,
)

QUESTION = Question("q-datasets", "1705.01265", "Feature set augmentation", "Which datasets do they evaluate on?",
                    (Reference("SemEval 2016 Task 4", "extractive", ("s3p1",)),))
PARAGRAPHS = {("1705.01265", "s3p1"): Paragraph("s3p1", "Data", "We evaluate on SemEval 2016 Task 4."),
              ("1705.01265", "s0p0"): Paragraph("s0p0", "Introduction", "Sentiment analysis is hard.")}
ROWS = [{"question_id": QUESTION.id, "seen": ["s0p0", "s3p1"], "cited": ["s3p1"]}]
ANSWER = Candidate(QUESTION.id, "SemEval 2016 Task 4", ("s3p1",), ("s0p0", "s3p1"), ("sonnet-k20",), False, 1.0)
PASS = json.dumps({"reason": "Matches the reference.", "verdict": "pass"})


class FakeClient:
    """Replies with the queued texts in order and records every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


def _judge(client, tmp_path, cap=1.0) -> PromptedJudge:
    llm = CachedLLM(client, "test-model", Meter(input_price=1.0, output_price=5.0, cap=cap), cache=tmp_path)
    return PromptedJudge("test", llm, Corpus([QUESTION], PARAGRAPHS, ROWS))


def test_the_prompt_shows_what_the_labeling_page_shows(tmp_path):
    prompt = _judge(FakeClient(), tmp_path).prompt(ANSWER)
    for shown in ["Which datasets do they evaluate on?", "[extractive] SemEval 2016 Task 4",
                  "[s3p1, section: Data] (cited by the answer) (expert evidence)", "Sentiment analysis is hard.",
                  "Answer to judge:\nSemEval 2016 Task 4"]:
        assert shown in prompt
    assert prompt.index("s0p0") < prompt.index("s3p1")


def test_the_prompt_never_names_the_writer_or_answer_f1(tmp_path):
    prompt = _judge(FakeClient(), tmp_path).prompt(ANSWER)
    assert "sonnet" not in prompt.lower() and "f1" not in prompt.lower()


def test_a_reply_becomes_a_verdict_with_its_reason(tmp_path):
    verdict = _judge(FakeClient(PASS), tmp_path).judge(ANSWER)
    assert verdict.passed and verdict.reason == "Matches the reference."


@pytest.mark.parametrize("reply, passed", [
    ('Here it is: {"reason": "Wrong dataset.", "verdict": "fail"} Hope that helps.', False),
    ('{"reason": "a\\nb", "verdict": "pass"}', True),
])
def test_parse_finds_the_verdict_in_surrounding_text(reply, passed):
    assert parse(reply).passed is passed


@pytest.mark.parametrize("reply", ["Pass.", '{"verdict": "maybe"}', '{"verdict": "pass"', "[1, 2]"])
def test_an_unreadable_reply_counts_as_fail_and_is_counted(tmp_path, reply):
    judge = _judge(FakeClient(reply), tmp_path)
    verdict = judge.judge(ANSWER)
    assert not verdict.passed and "unreadable" in verdict.reason and judge.unreadable == 1


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
