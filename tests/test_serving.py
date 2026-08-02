"""Serving-interface tests. Run against the stub — no GPU needed.
These lock the contract so the vLLM swap can't silently break callers.
"""
from src.serving.client import GenerationRequest
from src.serving.stub import StubClient


def test_stub_returns_response():
    client = StubClient()
    resp = client.generate(GenerationRequest(prompt="what is RAG?"))
    assert resp.text
    assert resp.completion_tokens > 0
    assert resp.latency_ms >= 0


def test_interface_shape():
    # any backend must expose .generate(GenerationRequest) -> GenerationResponse
    client = StubClient()
    resp = client.generate(GenerationRequest(prompt="x", max_tokens=10))
    for field in ("text", "prompt_tokens", "completion_tokens", "latency_ms"):
        assert hasattr(resp, field)
