"""Run the agent with a real model.

    export GROQ_API_KEY=...
    python -m src.agent.cli --q "What is hybrid retrieval? Also what is 40 * 15?"
"""
import typer
from src.agent.tools import registry
from src.agent.react import run_agent
from src.serving.groq_client import GroqClient
from typing_extensions import Annotated

app = typer.Typer()

@app.command()
def main(q: Annotated[str, typer.Option(help="the question")], collection: str = "main", max_steps: int = 5):
    tools = registry(collection=collection)
    client = GroqClient()
    answer = run_agent(q, tools, client, max_steps=max_steps)
    typer.echo("\n=== ANSWER ===")
    typer.echo(answer)

if __name__ == "__main__":
    app()
