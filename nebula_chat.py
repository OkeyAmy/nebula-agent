import argparse
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.text import Text
from thirdweb_ai import Insight, Nebula
from thirdweb_ai.adapters.langchain import get_langchain_tools

from tools import count_json_list, extract_json_value

load_dotenv()
console = Console()


def define_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="thirdweb AI Chat - Interactive chat with Blockchain LLM"
    )
    parser.add_argument(
        "--thread",
        type=str,
        default="default-thread",
        help="Thread ID for conversation history",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (0.0-1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--chain_id",
        type=int,
        default=1,
        help="Blockchain chain ID for Insight model",
    )
    parser.add_argument("--provider", type=str, default="anthropic")
    return parser.parse_args()


def build_graph(args):
    """Build the Nebula chat agent graph."""

    if args.model == "nebula":
        model = Nebula(secret_key=os.getenv("THIRDWEB_SECRET_KEY"))
        prompt = (
            "You have acces to Nebula, a language model trained on the blockchain, with access to real-time data. "
            "When the user asks a blockchain-related question, you shall use Nebula's 'chat' tool to answer it. "
            "When you answer, state the key points, and do not say 'In summary'. "
            "If the query has a single answer, provide it only. "
            "If the user's query does not relate to blockchain, you do not have to call this tool. "
            "If you Nebula cannot answer a blockchain question state this; do not attempt to answer it yourself."
        )
    elif args.model == "insight":
        model = Insight(
            secret_key=os.getenv("THIRDWEB_SECRET_KEY"), chain_id=args.chain_id
        )
        prompt = (
            "You have access to 'thirdweb' tools which allow you to retrieve real-time "
            "Blockchain data. Never attempt to guess at the answer to Blockchain-related questions. "
            "Whenever you are presented with a JSON structure, you **must** use the provided tools."
        )

    tools = get_langchain_tools(model.get_tools())
    tools += [extract_json_value, count_json_list]
    if args.provider == "anthropic":
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307", temperature=args.temperature
        )
    elif args.provider == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=args.temperature)
    else:
        raise ValueError("Invalid provider")

    return create_react_agent(
        llm, tools=tools, checkpointer=MemorySaver(), prompt=prompt
    )


console = Console()


def chat_loop(graph, args):
    """Start the interactive chat loop with step progress indication."""
    console.print(
        Panel.fit(
            f"{args.model} Chat - Type 'exit' to quit",
            title="Welcome",
            style="bold blue",
        )
    )
    config = {"configurable": {"thread_id": args.thread}}

    messages = []
    while True:
        try:
            user_input = console.input(
                Text("You: ", style=Style(color="green", bold=True))
            )
            if user_input.lower() in ("exit", "quit"):
                break

            inputs = {"messages": messages + [user_input]}
            console.print(f"\n{args.model}:".title(), style="bold magenta")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                out = graph.invoke(inputs, config=config)
                out["messages"][-2]

                message = ""
                for step in graph.stream(inputs, stream_mode="values", config=config):
                    message = step["messages"][-1]
                    console.clear()
                    console.print("\n")
                    console.print(message, style="dim")

                progress.remove_task(task)

            console.print(Panel.fit(message.content, border_style="magenta"))
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    """Main function to initialise chat."""
    args = define_args()
    graph = build_graph(args)
    chat_loop(graph, args)


if __name__ == "__main__":
    main()
