"""Eval harness tests — no model needed, uses fake scorers and a fake answer_fn."""
from src.eval.dataset import EvalCase
from src.eval.scorer import Score
from src.eval.harness import run_eval, summarize


class FixedScorer:
    def __init__(self, name, value):
        self.name, self.value = name, value
    def score(self, case, answer, context):
        return Score(self.name, self.value, "fixed")


def _answer_fn(q):
    return f"answer to {q}", "some context"


def test_run_eval_shape():
    cases = [EvalCase(question="q1"), EvalCase(question="q2")]
    scorers = [FixedScorer("a", 1.0), FixedScorer("b", 0.5)]
    results = run_eval(cases, _answer_fn, scorers)
    assert len(results) == 2
    assert len(results[0].scores) == 2


def test_summarize_averages():
    cases = [EvalCase(question="q1"), EvalCase(question="q2")]
    scorers = [FixedScorer("a", 1.0), FixedScorer("b", 0.0)]
    summary = summarize(run_eval(cases, _answer_fn, scorers))
    assert summary["a"] == 1.0
    assert summary["b"] == 0.0
