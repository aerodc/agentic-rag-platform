# src/agent/rag.py  — minimal RAG chain
from src.retrieval.hybrid import HybridRetriever, Reranker
from src.serving.client import GenerationRequest
from src.serving.stub import StubClient


def answer(query: str, collection: str = "main", client=None) -> str:
    client = client or StubClient()

    retriever = HybridRetriever(collection=collection)
    hits = retriever.retrieve(query, k=10)
    context = "\n\n".join(h.text for h in Reranker().rerank(query, hits, top_n=3))

    prompt = (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    return client.generate(GenerationRequest(prompt=prompt)).text