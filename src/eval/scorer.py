"""Scorers: given a question, the generated answer, and the retrieved context,
return a score in [0,1] plus a short reason.

Both scorers implement the same Scorer protocol, so the harness can run any
mix of them over the same eval set — same interface discipline as serving.
"""
from dataclasses import dataclass
from typing import Protocol
from src.eval.dataset import EvalCase
from src.serving.client import GenerationRequest

@dataclass
class Score:
    name: str          # which scorer produced this
    value: float       # 0.0 - 1.0
    reason: str = ""


class Scorer(Protocol):
    def score(self, case: EvalCase, answer: str, context: str) -> Score: ...


class ReferenceScorer:
    """Reference-based: 'is the answer correct?'

    Simplest useful version: check the must_include facts appear in the answer.
    This is more robust than string-matching a full reference, which breaks on
    valid rephrasing.
    """
    def score(self, case: EvalCase, answer: str, context: str) -> Score:
        """
          - If case.must_include is empty, return Score('reference', 1.0,
            'no reference facts to check') — nothing to measure.
          - Otherwise: count how many of case.must_include appear in `answer`
            (case-insensitive substring). value = hits / total.
          - reason: note which facts were missing.
        Why must_include over exact-match: a correct answer worded differently
        should still pass — you're checking facts are present, not phrasing.
        """
        if not case.must_include:
            return Score("reference", 1.0, "no reference facts to check")
        answer_low = answer.lower()
        hits = [f for f in case.must_include if f.lower() in answer_low]
        value = len(hits) / len(case.must_include)
        missing = [f for f in case.must_include if f.lower() not in answer_low]
        reason = "all facts present" if not missing else f"missing: {', '.join(missing)}"
        return Score("reference", value, reason)


class GroundingScorer:
    """RAG-specific: 'did the answer stick to the retrieved context, or make
    things up?' The core hallucination check for a RAG system.

    Uses an LLM as judge: ask a model whether every claim in the answer is
    supported by the context.
    """
    def __init__(self, client):
        self.client = client   # any InferenceClient

    def score(self, case: EvalCase, answer: str, context: str) -> Score:
        judge_prompt = f"""You are evaluating whether an answer is faithful to the provided context.

        Score ONLY faithfulness: is every claim in the ANSWER supported by the CONTEXT?
        Do NOT reward answers that are generally true but not grounded in the context.
        An answer that sounds correct but goes beyond the context should score LOW.

        CONTEXT:
        {context}

        ANSWER:
        {answer}

        Respond in exactly this format:
        SCORE: <number between 0 and 1>
        REASON: <one sentence>"""
        raw = self.client.generate(GenerationRequest(prompt=judge_prompt)).text
        return self._parse(raw)

    def _parse(self, raw):
        value, reason = 0.0, raw.strip()
        for line in raw.splitlines():
            if line.upper().startswith("SCORE:"):
                try:
                    value = float(line.split(":", 1)[1].strip())
                except ValueError:
                    value = 0.0
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return Score("grounding", max(0.0, min(1.0, value)), reason)

class RelevanceScorer:
    """LLM-as-judge: 'does the answer actually address the question?'
    Catches answers that are grounded but off-topic.
    """
    def __init__(self, client):
        self.client = client

    def score(self, case: EvalCase, answer: str, context: str) -> Score:
        if not case.must_include:
            return Score("reference", 1.0, "no reference facts to check")
        answer_low = answer.lower()
        hits = [f for f in case.must_include if f.lower() in answer_low]
        value = len(hits) / len(case.must_include)
        missing = [f for f in case.must_include if f.lower() not in answer_low]
        reason = "all facts present" if not missing else f"missing: {', '.join(missing)}"
        return Score("reference", value, reason)
