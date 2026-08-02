"""ReAct agent loop: the model Reasons about what to do, Acts via a tool,
observes the result, and repeats until it can answer.

This is the core of an agentic system — the intelligence is in the model
CHOOSING actions, not in a hardcoded pipeline. Build it by hand; it's ~40
lines and understanding it beats importing a framework.
"""
from dataclasses import dataclass
from src.agent.tools import Tool
from src.serving.client import GenerationRequest


AGENT_PROMPT = """You are an assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

Respond in EXACTLY one of these two formats:
  TOOL: <tool_name> | INPUT: <argument>
  ANSWER: <your final answer>

Conversation so far:
{history}

Your next step:"""


@dataclass
class Step:
    kind: str          # "tool" or "answer"
    name: str = ""     # tool name if kind == "tool"
    content: str = ""  # tool input, or the final answer


def parse_decision(text: str) -> Step:
    """
      Parse the model's output into a Step.
      - If it starts with 'ANSWER:' -> Step(kind='answer', content=<rest>)
      - If it starts with 'TOOL:'   -> Step(kind='tool', name=<tool>, content=<input>)
        (format: 'TOOL: search | INPUT: what is X')
      - If it matches neither -> treat as an answer (graceful fallback), or
        raise — decide, and note WHY in the README. Robust parsing of model
        output is a real production concern (models don't always follow format).
    """
    text = text.strip()

    if text.startswith("ANSWER:"):
        return Step(kind="answer", content=text[len("ANSWER:"):].strip())

    elif text.startswith("TOOL:"):
        body = text[len("TOOL:"):]
        tool_part, _, input_part = body.partition("|")
        name = tool_part.strip()
        arg = input_part.replace("INPUT:", "").strip()
        return Step(kind="tool", name=name, content=arg)
    else:
        return Step(kind="answer", content=text)


def run_agent(query: str, tools: dict[str, Tool], client, max_steps: int = 5) -> str:
    # ---- guardrail: input validation ----
    if not query or not query.strip():
        return "Error: empty query."
    if len(query) > 2000:
        return "Error: query too long."

    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
    history = f"Question: {query}"

    # ---- guardrail: bounded loop (never let an agent run unbounded) ----
    for step in range(max_steps):
        prompt = AGENT_PROMPT.format(tool_descriptions=tool_descriptions, history=history)
        raw = client.generate(GenerationRequest(prompt=prompt)).text
        decision = parse_decision(raw)

        if decision.kind == "answer":
            return decision.content

        tool = tools.get(decision.name)
        if tool is None:
            history += f"\nObservation: unknown tool '{decision.name}'. Available: {', '.join(tools)}"
            continue
        result = tool.run(decision.content)
        history += f"\nAction: {decision.name}({decision.content})\nObservation: {result}"


    # ---- guardrail fired: loop exhausted without an answer ----
    return "Stopped: reached max reasoning steps without a final answer."
