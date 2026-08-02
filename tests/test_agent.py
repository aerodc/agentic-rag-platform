"""Agent loop tests — run against the ScriptedClient, no model, no GPU.
These lock the loop mechanics: parsing, tool execution, and the guardrails.
"""
from src.agent.tools import Tool
from src.agent.react import run_agent, parse_decision
from src.serving.scripted_stub import ScriptedClient


def _echo_tool():
    return {"search": Tool("search", "test search", lambda q: f"found: {q}")}


def test_direct_answer():
    client = ScriptedClient(["ANSWER: 42"])
    assert run_agent("what is the answer?", _echo_tool(), client) == "42"


def test_tool_then_answer():
    client = ScriptedClient([
        "TOOL: search | INPUT: meaning of life",
        "ANSWER: it is 42",
    ])
    assert run_agent("q", _echo_tool(), client) == "it is 42"


def test_max_steps_guardrail():
    # never emits ANSWER -> must stop at the cap, not loop forever
    client = ScriptedClient(["TOOL: search | INPUT: x"] * 20)
    out = run_agent("q", _echo_tool(), client, max_steps=3)
    assert "max reasoning steps" in out


def test_empty_input_guardrail():
    assert "empty" in run_agent("  ", _echo_tool(), ScriptedClient([])).lower()


def test_parse_answer():
    step = parse_decision("ANSWER: hello world")
    assert step.kind == "answer" and step.content == "hello world"


def test_parse_tool():
    step = parse_decision("TOOL: search | INPUT: cats")
    assert step.kind == "tool" and step.name == "search" and step.content == "cats"
