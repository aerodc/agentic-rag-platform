# agentic-rag-platform

A production-oriented retrieval-augmented generation and agent platform. Built to
explore the infrastructure problems that matter when LLM systems move from a
notebook to something a team depends on: efficient model serving, retrieval
quality under drift, agent orchestration with guardrails, and evaluation without
ground-truth labels.

Not a framework wrapper — the retrieval, serving, and evaluation layers are
implemented directly so the tradeoffs are explicit and measurable.

## Architecture

```
                        ┌────────────────────┐
   documents ─ ingest ─▶│  vector store       │
                        │  (hybrid: dense+BM25)│
                        └─────────┬───────────┘
                                  │ retrieve + rerank
                                  ▼
   query ──────────────▶  agent orchestrator  ──▶ tool calls (MCP)
                                  │
                                  ▼
                          vLLM inference server
                                  │
                                  ▼
                          response + trace
                                  │
                                  ▼
                    eval harness (LLM-as-judge + guardrails)
```

## Components

| Layer | What it does | Key concerns |
|-------|--------------|--------------|
| `ingest` | Chunking, embedding, index build | chunk strategy, embedding choice, incremental reindex |
| `retrieval` | Hybrid dense + lexical retrieval, reranking | recall vs precision, retrieval drift, latency |
| `serving` | vLLM-backed inference endpoint | KV-cache, continuous batching, quantization, cost/token |
| `agent` | Multi-step orchestration, tool calls via MCP | grounding, guardrails, failure handling |
| `eval` | LLM-as-judge, reference sets, guardrail checks | hallucination, grounding gaps, non-determinism |

## Design decisions

_(document these as you build — this section is what makes the repo defensible)_

- **Chunking:** …
- **Retrieval:** why hybrid over pure vector …
- **Serving:** vLLM over naive HF pipeline — throughput/latency numbers …
- **Evaluation:** why LLM-as-judge, and its failure modes …

## Benchmarks

_(fill in as you measure — real numbers, your hardware)_

| Metric | Value |
|--------|-------|
| Retrieval recall@10 | |
| p50 / p99 serving latency | |
| Throughput (req/s) | |
| Cost per 1k queries | |

## Running

```bash
make ingest      # build the index
make serve       # start vLLM endpoint
make query       # run a query through the agent
make eval        # run the evaluation harness
```
