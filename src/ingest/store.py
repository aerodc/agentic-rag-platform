"""Vector store wrapper over Chroma (persistent, local).

Production-relevant requirement: incremental indexing. Re-ingesting new docs
must NOT rebuild the whole index. Chroma upserts by id, which gives you this
for free IF you make chunk ids stable and deterministic.
"""
import chromadb


class VectorStore:
    def __init__(self, path: str = "data/chroma", collection: str = "main"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, embeddings, documents, metadatas):
        # upsert (not add) so re-ingesting the same doc replaces, not duplicates
        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], k: int = 10):
        return self.collection.query(query_embeddings=[embedding], n_results=k)

    def all_documents(self):
        """Return all (id, document) pairs — needed to build the BM25 index."""
        got = self.collection.get()
        return list(zip(got["ids"], got["documents"]))
