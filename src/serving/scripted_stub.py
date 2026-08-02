"""Scripted stub for developing the agent loop with NO real model.

A plain stub can't exercise the ReAct loop because the loop needs the
'model' to emit tool-calls THEN an answer across successive steps. This
scripted version returns a preset sequence of decisions, so you can build and
test the loop mechanics deterministically before wiring a real LLM.

Example: hand it ["TOOL: search | INPUT: hybrid retrieval",
                  "ANSWER: Hybrid retrieval fuses dense and lexical search."]
and it returns them one per call.
"""
import time
from src.serving.client import GenerationRequest, GenerationResponse


class ScriptedClient:
    def __init__(self, script: list[str]):
        self.script = script
        self.i = 0

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        text = self.script[self.i] if self.i < len(self.script) else "ANSWER: (script exhausted)"
        self.i += 1
        return GenerationResponse(
            text=text,
            prompt_tokens=len(req.prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=0.0,
        )
