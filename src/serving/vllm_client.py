"""vLLM-backed inference client (phase 2 — GPU).

vLLM exposes an OpenAI-compatible HTTP server. You start it separately:

    python -m vllm.entrypoints.openai.api_server \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --quantization awq \
        --max-model-len 4096

Then this client POSTs to it. The point of the exercise is to understand WHY
vLLM serves far more throughput than a naive HF pipeline:
  - PagedAttention: manages the KV-cache in pages, so many requests share GPU
    memory efficiently instead of each reserving a worst-case block.
  - Continuous batching: new requests join the running batch every step,
    instead of waiting for the whole batch to finish.
Be ready to explain both — they are the interview payload of this component.
"""
import time
import requests
from src.serving.client import GenerationRequest, GenerationResponse


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.base_url = base_url
        self.model = model

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        """TODO (phase 2, on GPU):
          - POST to f"{self.base_url}/completions" with model, prompt,
            max_tokens, temperature.
          - Time the call; read usage.prompt_tokens / completion_tokens
            from the response.
          - Return a GenerationResponse.
        Keep the signature identical to StubClient so nothing downstream changes.
        """
        raise NotImplementedError
