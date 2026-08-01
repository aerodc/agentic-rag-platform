import typer
from src.retrieval.hybrid import HybridRetriever, Reranker

app = typer.Typer()

@app.command()
def main(collection: str = "main", q: str = ""):
    retriever = HybridRetriever(collection=collection)
    hits = retriever.retrieve(q, k=10)
    reranked = Reranker().rerank(q, hits, top_n=5)
    for i, h in enumerate(reranked, 1):
        typer.echo(f"{i}. [{h.score:.4f}] {h.text[:120]}")

if __name__ == "__main__":
    app()