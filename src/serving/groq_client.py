"""Real LLM backend via Groq (OpenAI-compatible, free tier, fast).

Same InferenceClient interface as StubClient/ScriptedClient — so swapping this
in makes the agent genuinely reason, with zero changes to run_agent or tools.

Setup:
  1. Get a free key at console.groq.com  ->  export GROQ_API_KEY=...
  2. pip install openai   (Groq speaks the OpenAI API)
"""
import os
import time
from openai import OpenAI
from src.serving.client import GenerationRequest, GenerationResponse


class GroqClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        self.model = model

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=["\nObservation:"],   # <- stop before hallucinating tool results
        )
        text = resp.choices[0].message.content
        return GenerationResponse(
            text=text,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
