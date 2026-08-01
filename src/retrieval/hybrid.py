"""Hybrid retrieval: dense (embeddings) + lexical (BM25), then rerank.

Why hybrid, not pure vector (your README + interview answer):
  Dense retrieval captures semantic similarity but misses exact terms
  (names, codes, rare tokens). BM25 nails exact terms but misses paraphrase.
  Combining recovers both. Reranking with a cross-encoder then fixes the
  ordering, because the initial scores come from two different scales.
"""
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from src.ingest.embedder import Embedder
from src.ingest.store import VectorStore


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float


class HybridRetriever:
    def __init__(self, collection: str = "main"):
        self.store = VectorStore(collection=collection)
        self.embedder = Embedder()
        corpus = self.store.all_documents()  # [(id, text), ...]
        self.ids = [c[0] for c in corpus]
        self.texts = [c[1] for c in corpus]
        # TODO: build a BM25 index over self.texts (tokenize simply on whitespace/lowercase)
        self.bm25 = None  # <- you implement

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        """Return top-k hits by fused dense + lexical score.

        TODO (you implement):
          1. Dense: embed query, query the vector store, get ids+distances.
          2. Lexical: score all docs with BM25 for the query tokens.
          3. Fuse: normalize each score list to [0,1] and combine
             (start with simple weighted sum; note the weight in README).
             Reciprocal Rank Fusion is a cleaner alternative — try it as a stretch.
          4. Return top-k as Hit objects.
        """
        raise NotImplementedError


class Reranker:
    """Cross-encoder reranking of the fused candidate set."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int = 5) -> list[Hit]:
        """TODO: score each (query, hit.text) pair, sort desc, return top_n.
        This is the step that most improves answer quality — measure it."""
        raise NotImplementedError
