"""Embedding model wrapper. CPU-friendly."""
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast on CPU


class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True -> cosine similarity via dot product downstream
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
