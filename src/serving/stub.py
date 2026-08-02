"""CPU stub backend — lets you build and test the whole serving surface with
no GPU. Swap for VLLMClient in phase 2. Returns a canned completion so the
pipeline is exercisable end to end on a MacBook.
"""
import time
from src.serving.client import GenerationRequest, GenerationResponse


class StubClient:
    def generate(self, req: GenerationRequest) -> GenerationResponse:
        t0 = time.perf_counter()
        # trivial deterministic 'completion' — enough to wire up the agent layer
        text = f"[stub answer to: {req.prompt[:60]}...]"
        return GenerationResponse(
            text=text,
            prompt_tokens=len(req.prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
