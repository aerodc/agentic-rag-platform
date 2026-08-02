"""Client interface to the inference server.

Design goal: the RAG/agent layers depend on THIS interface, not on vLLM
directly. That means you can develop against a CPU stub locally and swap in
real vLLM serving without touching any calling code. Same reason you'd put an
interface in front of any external dependency.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2


@dataclass
class GenerationResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    # latency_ms lets you measure and report p50/p99 — the numbers that make
    # this a portfolio piece rather than a toy
    latency_ms: float


class InferenceClient(Protocol):
    def generate(self, req: GenerationRequest) -> GenerationResponse: ...
