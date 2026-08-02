"""Tools an agent can choose to invoke.

A tool = a name + a description the model reads to decide when to use it +
a function that runs it. The agent doesn't hardcode which tool to call;
it reads the descriptions and decides.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str            # the model reads THIS to decide when to use the tool
    func: Callable[[str], str]

    def run(self, arg: str) -> str:
        return self.func(arg)


def make_search_tool(collection: str = "main") -> Tool:
    from src.retrieval.hybrid import HybridRetriever, Reranker
    retriever = HybridRetriever(collection=collection)
    reranker = Reranker()

    def _search(query: str) -> str:
        hits = reranker.rerank(query, retriever.retrieve(query, k=10), top_n=3)
        return "\n\n".join(h.text for h in hits) or "(no results)"

    return Tool(
        name="search",
        description="Search the knowledge base for relevant context. Input: a search query string.",
        func=_search,
    )


def make_calculator_tool() -> Tool:
    def _calc(expr: str) -> str:
        """TODO (you implement):
          Safely evaluate a simple arithmetic expression and return the result.
          DO NOT use bare eval() on untrusted input — that's an RCE hole and a
          great interview talking point about why. Restrict to digits, spaces,
          and the operators + - * / ( ) . — reject anything else, then evaluate.
          (A clean approach: whitelist-check the characters, then use Python's
          `ast` module to parse and evaluate only arithmetic nodes.)
        """
        raise NotImplementedError

    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression. Input: a math expression like '3 * (4 + 2)'.",
        func=_calc,
    )


def registry(collection: str = "main") -> dict[str, Tool]:
    tools = [make_search_tool(collection), make_calculator_tool()]
    return {t.name: t for t in tools}
