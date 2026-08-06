"""Evaluation dataset: questions with optional reference answers.

A small, hand-curated set is enough to catch regressions when you change
chunking, retrieval, or prompts. Quality over quantity — 10 good cases beat
100 noisy ones.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class EvalCase:
    question: str
    reference: str = ""              # known-good answer (optional; enables reference scorer)
    must_include: list[str] = field(default_factory=list)  # facts a correct answer must contain


def load_cases(path: str = "data/eval/cases.jsonl") -> list[EvalCase]:
    p = Path(path)
    if not p.exists():
        return []
    cases = []
    for line in p.read_text().splitlines():
        if line.strip():
            cases.append(EvalCase(**json.loads(line)))
    return cases
