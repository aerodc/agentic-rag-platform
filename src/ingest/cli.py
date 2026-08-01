"""Ingest pipeline: docs -> chunks -> embeddings -> vector store.

Run: python -m src.ingest.cli --source data/docs --collection main
"""
from pathlib import Path
import typer
from pypdf import PdfReader

from src.ingest.chunker import chunk_text
from src.ingest.embedder import Embedder
from src.ingest.store import VectorStore

app = typer.Typer()


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


@app.command()
def main(source: str = "data/docs", collection: str = "main"):
    embedder = Embedder()
    store = VectorStore(collection=collection)

    files = [p for p in Path(source).rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    typer.echo(f"found {len(files)} documents")

    total = 0
    for path in files:
        doc_id = str(path.relative_to(source))
        chunks = chunk_text(doc_id, load_text(path))
        if not chunks:
            continue
        embeddings = embedder.embed([c.text for c in chunks])
        store.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "start": c.start, "end": c.end} for c in chunks],
        )
        total += len(chunks)
        typer.echo(f"  {doc_id}: {len(chunks)} chunks")

    typer.echo(f"ingested {total} chunks into '{collection}'")


if __name__ == "__main__":
    app()
