# CLAUDE.md

Context for AI assistants working in this repo.

## What this is

A production-oriented retrieval-augmented generation and agent platform. The
goal is to implement the infrastructure layer of an LLM system directly —
retrieval, serving, orchestration, evaluation — rather than wrapping a framework,
so the engineering tradeoffs are explicit and measurable.

## Stack

- Python 3.10+
- sentence-transformers (embeddings, reranking) — CPU-friendly models
- Chroma (persistent local vector store)
- rank-bm25 (lexical retrieval)
- vLLM (LLM serving — GPU, added in the serving layer)
- pytest

## Layout

- `src/ingest/`    chunking, embedding, vector store, ingest CLI
- `src/retrieval/` hybrid dense + BM25 retrieval, cross-encoder reranking
- `src/serving/`   vLLM-backed inference endpoint
- `src/agent/`     multi-step orchestration, tool calls (MCP), guardrails
- `src/eval/`      LLM-as-judge, reference sets, guardrail checks
- `tests/`         acceptance tests per component

## Conventions

- Chunk ids are deterministic (`{doc_id}:{index}`) so re-ingest upserts,
  never duplicates.
- Embeddings are normalized; cosine similarity via dot product.
- Every design decision (chunk size, fusion weights, model choices) is
  documented in README under "Design decisions" with the reasoning, not just
  the value.
- Small, testable commits — one component concern per commit.

## How to help here

- Explain concepts and review code; do not author the core logic
  (chunking, fusion, reranking, agent orchestration) — those are implemented
  by hand on purpose, to stay defensible in technical interviews.
- Plumbing and test scaffolding are fine to generate.
- Always leave changes runnable: if you touch a component, run or update its test.
