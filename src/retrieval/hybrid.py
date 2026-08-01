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
        self._tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(self._tokenized)

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        """Return top-k hits by fused dense + lexical score.

          1. Dense: embed query, query the vector store, get ids+distances.
          2. Lexical: score all docs with BM25 for the query tokens.
          3. Fuse: normalize each score list to [0,1] and combine
             (start with simple weighted sum; note the weight in README).
             Reciprocal Rank Fusion is a cleaner alternative — try it as a stretch.
          4. Return top-k as Hit objects.
        """
        q_emb = self.embedder.embed([query])[0]
        dense = self.store.query(q_emb, k=len(self.ids))
        dense_ids = dense["ids"][0]
        dense_dist = dense["distances"][0]
        dense_scores = {cid:1.0 - dist for cid, dist in zip(dense_ids, dense_dist)}

        q_tokens = query.lower().split()
        bm25_raw = self.bm25.get_scores(q_tokens)
        bm25_scores = {cid:score for cid, score in zip(self.ids, bm25_raw)}
        
        rrf_k = 60
        dense_rank = {cid: r for r, cid in enumerate(
            sorted(dense_scores, key=dense_scores.get, reverse=True)
        )}
        bm25_rank = {cid: r for r, cid in enumerate(
            sorted(bm25_scores, key=bm25_scores.get, reverse=True)
        )}
        
        fused = {}
        for cid in self.ids:
            score = 0.0
            if cid in dense_rank:
                score += 1.0 / (rrf_k + dense_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (rrf_k + bm25_rank[cid])
            fused[cid] = score

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        text_by_id = dict(zip(self.ids, self.texts))

        return [Hit(chunk_id=cid, text=text_by_id[cid], score=fused[cid]) for cid in top]


class Reranker:
    """Cross-encoder reranking of the fused candidate set."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int = 5) -> list[Hit]:
        pairs = [[query, h.text] for h in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [Hit(chunk_id=h.chunk_id, text=h.text, score=float(s))
                for h, s in ranked[:top_n]]
