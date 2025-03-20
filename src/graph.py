import re
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.chains.intent_chain import intent_chain
from src.chains.react_chain import react_llm, react_template, react_tools
from src.common.utils import ETH_REGEX


class State(MessagesState):
    wallets: dict = {}


def intent_router(state: State) -> Literal["extract_wallets", "tools"]:
    out = intent_chain.invoke({"message": state["messages"][0]})
    intent = out.intent.name
    return "extract_wallets" if intent in ["transaction", "query"] else "tools"


def extract_wallets(state: State) -> State:
    """Extracts wallet addresses from messages."""
    message = state["messages"][-1]
    wallets = re.findall(ETH_REGEX, message.content)

    for idx, wallet in enumerate(wallets):
        message.content = re.sub(wallet, f"{{wallet_{idx}}}", message.content)

    wallet_dict = {f"wallet_{idx}": wallet for idx, wallet in enumerate(wallets)}
    state["wallets"] = wallet_dict
    return state


def _inject_wallets_tool(out: AIMessage, wallets: dict):
    tool_name = out.tool_calls[0]["name"]

    if tool_name == "get_erc20_tokens":
        # assumes only 1 wallet
        out.tool_calls[0]["args"]["owner_address"] = list(wallets.values())[0]
    return out


def agent(state: State):
    """Agent function to process messages using LLM."""
    # inject prompt before final message
    messages = (
        state["messages"][:-1]
        + [SystemMessage(content=react_template)]
        + [state["messages"][-1]]
    )
    wallets = state["wallets"]

    out = react_llm.invoke(messages)

    if out.tool_calls:
        out = _inject_wallets_tool(out, wallets)

    return {"messages": out}


def should_continue(state: State) -> Literal["tools", "inject_params"]:
    """Determines whether to continue processing or end the graph."""
    messages = state["messages"]
    last_message = messages[-1]
    return "tools" if last_message.tool_calls else "inject_params"


def inject_params(state: State) -> State:
    messages = state["messages"]
    last_message = messages[-1]
    last_message.content = last_message.content.format_map(state["wallets"])
    return state


def build_graph():
    """Constructs and compiles the agent execution graph."""
    builder = StateGraph(State)
    builder.add_node("extract_wallets", extract_wallets)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(react_tools))
    builder.add_node("inject_params", inject_params)

    builder.add_conditional_edges(START, intent_router)
    builder.add_edge("extract_wallets", "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")
    builder.add_edge("inject_params", END)

    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    inputs = {
        "messages": [
            "How many ERC20 tokens does 0xF977814e90dA44bFA03b6295A0616a897441aceC have?"
        ],
        "wallets": {},
    }
    graph = build_graph()
    graph.get_graph().draw_mermaid_png(output_file_path="diagram.png")
    out = graph.invoke(inputs, config=config)
    out
