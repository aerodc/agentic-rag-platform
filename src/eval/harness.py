"""Eval harness: run the RAG pipeline over the eval set, score each answer
with every scorer, report aggregate results.

This is what you run after ANY change (chunking, retrieval, prompt) to check
you didn't regress. It's the ML-platform discipline applied to LLM output:
no shipping without measuring.
"""
from dataclasses import dataclass
from statistics import mean
from src.eval.dataset import EvalCase, load_cases
from src.eval.scorer import Scorer, Score


@dataclass
class CaseResult:
    question: str
    answer: str
    scores: list[Score]


def run_eval(cases: list[EvalCase], answer_fn, scorers: list[Scorer]) -> list[CaseResult]:
    """answer_fn(question) -> (answer, context). scorers each score the pair.
      for each case:
        - answer, context = answer_fn(case.question)
        - run every scorer -> collect Scores
        - append a CaseResult
      return the list.
    """
    results = []
    for case in cases:
        answer, context = answer_fn(case.question)
        scores = [scorer.score(case, answer, context) for scorer in scorers]
        results.append(CaseResult(question=case.question, answer=answer, scores=scores))
    return results


def summarize(results: list[CaseResult]) -> dict[str, float]:
    buckets = {}
    for r in results:
        for s in r.scores:
            buckets.setdefault(s.name, []).append(s.value)
    return {name: mean(values) for name, values in buckets.items()}
