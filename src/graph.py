import re
from typing import Literal

from langchain_core.messages import SystemMessage
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
    if intent == "transaction":
        return "extract_wallets"
    elif intent == "query":
        return "extract_wallets"
    else:
        return "tools"


def extract_wallets(state: State) -> State:
    """Extracts wallet addresses from messages."""
    message = state["messages"][-1]
    wallets = re.findall(ETH_REGEX, message.content)

    for idx, wallet in enumerate(wallets):
        message.content = re.sub(wallet, f"{{wallet_{idx}}}", message.content)

    wallet_dict = {}
    for idx, wallet in enumerate(wallets):
        wallet_dict[f"wallet_{idx}"] = wallet

    state["wallets"] = wallet_dict
    return state


def agent(state: State):
    """Agent function to process messages using LLM."""
    messages = [SystemMessage(content=react_template)] + state["messages"]
    out = react_llm.invoke(messages)
    # tool = [t for t in react_tools if t.name == out.content[1]["name"]][0]

    # if state["wallets"]:
    #     for wallet in state["wallets"].values():
    #         tool.func(owner_address=wallet)

    return {"messages": out}


def should_continue(state: State) -> Literal["tools", END]:
    """Determines whether to continue processing or end the graph."""
    messages = state["messages"]
    last_message = messages[-1]
    return "tools" if last_message.tool_calls else END


def build_graph():
    """Constructs and compiles the agent execution graph."""
    builder = StateGraph(State)
    # builder.add_node("extract_wallets", extract_wallets)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(react_tools))

    builder.add_edge(START, "agent")
    # builder.add_conditional_edges(START, intent_router)
    # builder.add_edge("extract_wallets", "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    inputs = {
        "messages": [
            "How much ETH does 0xdAC17F958D2ee523a2206206994597C13D831ec7 have?"
        ],
        "wallets": {},
    }
    graph = build_graph()
    graph.get_graph().draw_mermaid_png(output_file_path="diagram.png")
    out = graph.invoke(inputs, config=config)
    out
