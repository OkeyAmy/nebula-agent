import logging

from dotenv import load_dotenv
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.text import Text

from src.graph import build_graph

load_dotenv()
set_llm_cache(InMemoryCache())

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
console = Console()


def main():
    """Runs the interactive agent session."""
    graph = build_graph()

    console.print(
        Panel.fit(
            "Insight Chat - Type 'exit' to quit",
            title="Welcome",
            style="bold blue",
        )
    )

    while True:
        try:
            user_input = console.input(
                Text("You: ", style=Style(color="green", bold=True))
            )
            if user_input.lower() in ("exit", "quit"):
                console.print("[bold red]Exiting...[/bold red]")
                break

            inputs = {"messages": [user_input], "wallets": {}}
            console.print("\nInsight:", style="bold magenta")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=None)

                config = {"configurable": {"thread_id": "1"}}
                message = ""
                for step in graph.stream(inputs, stream_mode="values", config=config):
                    message = step["messages"][-1]
                    console.clear()
                    console.print("\n")
                    console.print(message, style="dim")

                tools_used = [
                    {
                        "name": message.model_dump()["name"],
                        "status": message.model_dump()["status"],
                    }
                    for message in step["messages"]
                    if isinstance(message, ToolMessage)
                ]

                console.print(f"\nTools used: {tools_used}")
                progress.remove_task(task)

            console.print(Panel.fit(message.content, border_style="magenta"))
            print()
        except KeyboardInterrupt:
            console.print("[bold yellow]Session interrupted. Exiting...[/bold yellow]")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
