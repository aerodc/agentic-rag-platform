"""Document chunking.

Design decision to make and defend: chunk size and strategy.
Interview question you must be able to answer:
  "How did you choose your chunk size, and what breaks if it's wrong?"
Too large -> retrieval returns imprecise context, wastes tokens.
Too small -> loses surrounding meaning, retrieval recall drops.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    # position lets you cite/trace where an answer came from
    start: int
    end: int


def chunk_text(doc_id: str, text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split text into overlapping chunks.

    TODO (you implement):
      - Split `text` into windows of ~`size` characters with `overlap` between
        consecutive windows. Overlap matters: it keeps a sentence that straddles
        a boundary retrievable from both chunks.
      - Assign chunk_id as f"{doc_id}:{index}".
      - Return a list[Chunk].

    Stretch (do after the naive version works, and note it in the README):
      - Prefer splitting on sentence/paragraph boundaries rather than raw chars,
        so you don't cut mid-sentence. Compare retrieval quality before/after.
    """
    raise NotImplementedError
