import logging
import json
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.text import Text
from rich.table import Table
from rich.tree import Tree

from src.graph import build_graph

load_dotenv()
set_llm_cache(InMemoryCache())

# Configure logging to show warnings - helps with debugging
logging.basicConfig(
    level=logging.WARNING, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = Console()

def log_process_step(step_name, details=None):
    """Log a processing step with optional details."""
    log_msg = f"PROCESS: {step_name}"
    if details:
        log_msg += f" - {details}"
    logging.warning(log_msg)
    return log_msg


def display_tools_used(tools_used: List[Dict[str, Any]]) -> None:
    """Display the tools used in the graph execution in a structured format."""
    if not tools_used:
        console.print("[yellow]No tool usage data available[/yellow]")
        return
    
    # Create a tree view for tool usage
    tree = Tree("[bold blue]🔧 Tools Used[/bold blue]")
    
    for i, tool_info in enumerate(tools_used, 1):
        tool_name = tool_info.get("tool", "unknown_tool")
        
        # Create different styling based on tool type
        if "nebula" in tool_name.lower():
            tool_node = tree.add(f"[bold magenta]{i}. {tool_name}[/bold magenta]")
        elif "extract_wallets" == tool_name:
            tool_node = tree.add(f"[bold green]{i}. {tool_name}[/bold green]")
        elif "agent" == tool_name:
            tool_node = tree.add(f"[bold cyan]{i}. {tool_name}[/bold cyan]")
        elif "intent_router" == tool_name:
            tool_node = tree.add(f"[bold yellow]{i}. {tool_name}[/bold yellow]")
        else:
            tool_node = tree.add(f"[bold]{i}. {tool_name}[/bold]")
        
        # Add details for each tool
        for key, value in tool_info.items():
            if key != "tool":
                # Format lists and dictionaries nicely
                if isinstance(value, (list, dict)):
                    formatted_value = json.dumps(value, indent=2)
                    tool_node.add(f"[dim]{key}:[/dim] {formatted_value}")
                else:
                    tool_node.add(f"[dim]{key}:[/dim] {value}")
    
    console.print(tree)


def display_processing_log(steps):
    """Display a summary of the processing steps that occurred."""
    if not steps:
        return
    
    log_table = Table(title="Processing Log")
    log_table.add_column("Step", style="cyan")
    log_table.add_column("Action", style="magenta")
    log_table.add_column("Result", style="green")
    
    for i, step in enumerate(steps, 1):
        # Try to determine the node that was executed
        node = "Unknown"
        action = "Unknown"
        result = "Unknown"
        
        if "tools_used" in step and step["tools_used"]:
            last_tool = step["tools_used"][-1]
            node = last_tool.get("tool", "Unknown")
            
            if "decision" in last_tool:
                action = f"Decision: {last_tool['decision']}"
            elif "action" in last_tool:
                action = f"Action: {last_tool['action']}"
            elif "response_type" in last_tool:
                action = f"Response: {last_tool['response_type']}"
            
            if "trigger" in last_tool:
                result = last_tool["trigger"]
            elif "success" in last_tool:
                result = f"Success: {last_tool['success']}"
            elif "wallets_found" in last_tool:
                result = f"Wallets: {last_tool['wallets_found']}"
            
        log_table.add_row(f"{i}. {node}", action, result)
    
    console.print(log_table)


def main():
    """Runs the interactive agent session."""
    log_process_step("Starting Insight Chat")
    graph = build_graph()

    console.print(
        Panel.fit(
            "Insight Chat - Type 'exit' to quit",
            title="Welcome",
            style="bold blue",
        )
    )

    # Just display a welcome prompt
    console.print("I'm an AI assistant specialized in blockchain and web3. Please ask a blockchain-related question!", style="green")
    
    while True:
        try:
            user_input = console.input(
                Text("You: ", style=Style(color="green", bold=True))
            )
            if user_input.lower() in ("exit", "quit"):
                console.print("[bold red]Exiting...[/bold red]")
                break
            
            # Check for debug commands
            if user_input.lower() == "debug on":
                logging.getLogger().setLevel(logging.DEBUG)
                console.print("[bold yellow]Debug mode enabled[/bold yellow]")
                continue
            elif user_input.lower() == "debug off":
                logging.getLogger().setLevel(logging.WARNING)
                console.print("[bold yellow]Debug mode disabled[/bold yellow]")
                continue

            # Format the user input as a HumanMessage object before passing to the graph
            inputs = {"messages": [HumanMessage(content=user_input)], "wallets": {}, "tools_used": []}
            console.print("\nInsight:", style="bold magenta")
            
            log_process_step("Processing user input", f"Message: '{user_input}'")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=None)

                config = {"configurable": {"thread_id": "1"}}
                message = None
                steps = []
                tool_usage_log = []
                try:
                    # Process the graph and collect all steps
                    steps = list(graph.stream(inputs, stream_mode="values", config=config))
                    log_process_step("Graph execution completed", f"Steps: {len(steps)}")
                    
                    if steps:
                        step = steps[-1]  # Get the final step
                        if step.get("messages") and len(step["messages"]) > 0:
                            message = step["messages"][-1]
                        
                        # Collect all tools used across all steps
                        for s in steps:
                            if "tools_used" in s and s["tools_used"]:
                                tool_usage_log.extend(s["tools_used"])
                except Exception as e:
                    error_msg = f"Error in graph stream: {e}"
                    logging.error(error_msg)
                    console.print(f"\nError: {e}", style="bold red")

                # Collect tool calls from messages
                tool_messages = []
                if steps and steps[-1] and "messages" in steps[-1]:
                    tool_messages = [
                        {
                            "name": msg.model_dump().get("name", "unknown"),
                            "status": msg.model_dump().get("status", "unknown"),
                        }
                        for msg in steps[-1].get("messages", [])
                        if isinstance(msg, ToolMessage)
                    ]

                progress.remove_task(task)
                
                # Display processing log
                if steps:
                    display_processing_log(steps)

                # Display tool usage details
                if tool_usage_log:
                    display_tools_used(tool_usage_log)
                elif tool_messages:
                    console.print("\nTool calls detected:", style="bold yellow")
                    for tool in tool_messages:
                        console.print(f"- {tool['name']}: {tool['status']}")

            # Display the final message
            if message:
                if hasattr(message, "content"):
                    console.print(Panel.fit(message.content, border_style="magenta"))
                else:
                    console.print(Panel.fit(str(message), border_style="magenta"))
            else:
                # Fallback if we didn't get a proper message response
                console.print(Panel.fit(f"I couldn't process: {user_input}", border_style="red"))
            print()
        except KeyboardInterrupt:
            console.print("[bold yellow]Session interrupted. Exiting...[/bold yellow]")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            console.print(Panel.fit(f"An error occurred: {str(e)}", border_style="red"))


if __name__ == "__main__":
    main()
