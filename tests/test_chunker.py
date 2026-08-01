"""Acceptance tests for chunking. Make these pass — that's Brief 1 done."""
from src.ingest.chunker import chunk_text


def test_covers_full_text():
    text = "a" * 2000
    chunks = chunk_text("d", text, size=512, overlap=64)
    # every character of the source must appear in at least one chunk
    assert chunks[0].start == 0
    assert chunks[-1].end >= len(text) - 1


def test_overlap_present():
    text = "".join(f"word{i} " for i in range(400))
    chunks = chunk_text("d", text, size=512, overlap=64)
    assert len(chunks) >= 2
    # consecutive chunks should share some text due to overlap
    assert chunks[0].end > chunks[1].start


def test_stable_ids():
    text = "x" * 3000
    a = chunk_text("d", text)
    b = chunk_text("d", text)
    # deterministic ids -> re-ingest upserts instead of duplicating
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
