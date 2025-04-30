import re
import logging
import json
from typing import Literal, List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.chains.intent_chain import intent_chain, Intent
from src.chains.react_chain import react_llm, react_template, react_tools
from src.common.utils import ETH_REGEX

# Import MongoDB database tools if they exist in src/tools.py
# Update imports based on your actual implementation
try:
    from src.tools import get_financial_data, analyze_expenses, get_customer_insights
except ImportError:
    # If tools aren't imported, create stub functions to avoid errors
    def get_financial_data(user_id: str): return {"error": "Tool not implemented"}
    def analyze_expenses(user_id: str): return {"error": "Tool not implemented"}
    def get_customer_insights(user_id: str): return {"error": "Tool not implemented"}


class State(MessagesState):
    wallets: dict = {}
    tools_used: List[Dict[str, Any]] = []
    user_id: str = "default_user"


def get_latest_human_message(state: State) -> str:
    """
    Extracts the most recent human message content from the state.
    Returns an empty string if no human message is found.
    """
    # Find the most recent HumanMessage in the state
    message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
            message_content = msg.content
            break
    
    # Fall back to the first message if no HumanMessage is found
    if not message_content and state["messages"]:
        message = state["messages"][0]
        message_content = message.content if hasattr(message, "content") else str(message)
        
    return message_content


def intent_router(state: State):
    """Routes messages based on the detected intent."""
    message_content = get_latest_human_message(state)
    
    try:
        # Call the intent chain to classify the message
        result = intent_chain.invoke({"message": message_content})
        intent = result.intent
        
        logging.warning(f"LLM intent detected: {intent} for message: '{message_content}'")
        
        # Check for direct wallet detection first
        wallet_matches = re.findall(ETH_REGEX, message_content)
        if wallet_matches:
            state["tools_used"].append({
                "tool": "intent_router",
                "decision": "extract_wallets",
                "trigger": f"wallets detected: {wallet_matches}"
            })
            return "extract_wallets"
        
        # Route based on the detected intent
        if intent == Intent.nebula_query:
            state["tools_used"].append({
                "tool": "intent_router",
                "decision": "nebula_handler",
                "trigger": "nebula query intent"
            })
            return "nebula_handler"
        elif intent == Intent.financial_query:
            state["tools_used"].append({
                "tool": "intent_router",
                "decision": "financial_handler",
                "trigger": "financial query intent"
            })
            return "financial_handler"
        else:
            state["tools_used"].append({
                "tool": "intent_router",
                "decision": "general_handler",
                "trigger": "general query intent or fallback"
            })
            return "general_handler"
            
    except Exception as e:
        logging.error(f"Error in intent classification: {e}")
        # Fall back to general handler on error
        state["tools_used"].append({
            "tool": "intent_router",
            "decision": "general_handler",
            "trigger": f"error in intent classification: {str(e)}"
        })
        return "general_handler"


def extract_wallets(state: State) -> State:
    """Extracts wallet addresses from messages."""
    message = state["messages"][-1]
    
    if not hasattr(message, "content") or not message.content:
        logging.warning("No content found in message for wallet extraction")
        state["tools_used"].append({
            "tool": "extract_wallets",
            "wallets_found": 0,
            "error": "No content in message"
        })
        return state
        
    logging.warning(f"Extracting wallets from: {message.content[:50]}...")
    
    if wallets := re.findall(ETH_REGEX, message.content):
        # for now skip this as it confuses the LLM
        # for idx, wallet in enumerate(wallet
        #     message.content = re.sub(wallet, f"{{wallet_{idx}}}", message.content)

        wallet_dict = {f"wallet_{idx}": wallet for idx, wallet in enumerate(wallets)}
        state["wallets"] = wallet_dict
        state["tools_used"].append({
            "tool": "extract_wallets",
            "wallets_found": len(wallets),
            "wallets": list(wallet_dict.values())
        })
        logging.warning(f"Found {len(wallets)} wallet(s): {list(wallet_dict.values())}")
    else:
        state["tools_used"].append({
            "tool": "extract_wallets",
            "wallets_found": 0
        })
        logging.warning("No wallets found in message")
    return state


def _inject_wallets_tool(out: AIMessage, wallets: dict):
    # for now just assume a single wallet
    tool_name = out.tool_calls[0]["name"]
    if tool_name == "get_erc20_tokens":
        out.tool_calls[0]["args"]["owner_address"] = list(wallets.values())[0]
    elif tool_name == "resolve":
        out.tool_calls[0]["args"]["input_data"] = list(wallets.values())[0]
    elif tool_name == "get_token_prices":
        out.tool_calls[0]["args"]["token_addresses"] = list(wallets.values())
    return out


def agent(state: State):
    """Agent function to process messages using LLM."""
    messages = state["messages"]
    wallets = state.get("wallets", {})
    user_id = state.get("user_id", "default_user") # Ensure user_id is available
    
    # Check if the financial system prompt needs to be added (if financial_handler didn't add it)
    # Or ensure the standard react prompt is added if not financial
    is_financial_flow = any(item.get("tool") == "financial_handler" for item in state.get("tools_used", []))
    has_system_prompt = any(isinstance(msg, SystemMessage) for msg in messages)

    if not has_system_prompt:
        if is_financial_flow:
            # Add financial prompt if not already present (should be added by handler, but as fallback)
            system_prompt = """
            You are a financial analyst AI assistant with access to business financial data in MongoDB.
            Use these tools to analyze the data and answer financial questions:
            - For customer data queries, use get_customer_insights
            - For expense analysis, use analyze_expenses
            - For overall financial data, use get_financial_data
            
            Always analyze the data before responding, and provide specific insights based on what you find.
            """
            messages.insert(0, SystemMessage(content=system_prompt))
            logging.warning("Agent added Financial System Prompt (fallback)")
        else:
             # Add the standard React template if no system prompt exists
            messages.insert(0, SystemMessage(content=react_template))
            logging.warning("Agent added React System Prompt")

    # Ensure the last message is a valid type for LLM invocation
    if not isinstance(messages[-1], (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
        logging.warning(f"Converting message of type {type(messages[-1])} to HumanMessage")
        messages[-1] = HumanMessage(content=str(messages[-1]))

    # Log what we're about to do
    logging.warning(f"Invoking React LLM with {len(messages)} messages, last message type: {type(messages[-1])}")
    
    # Invoke the LLM with the current messages
    try:
        out = react_llm.invoke(messages)
        
        # Check if we have tool calls and wallets to inject
        if hasattr(out, "tool_calls") and out.tool_calls and wallets:
            logging.warning(f"Injecting wallet information into tool calls: {list(wallets.values())}")
            out = _inject_wallets_tool(out, wallets)
            state["tools_used"].append({
                "tool": "agent",
                "action": "inject_wallets",
                "tools_called": [call["name"] for call in out.tool_calls],
                "wallets": list(wallets.values())
            })
        # Check if we have financial data tool calls that need user_id
        elif hasattr(out, "tool_calls") and out.tool_calls and any(call["name"] in ["get_financial_data", "analyze_expenses", "get_customer_insights"] 
                                   for call in out.tool_calls):
            # Add user_id to tool calls if needed
            logging.warning(f"Injecting user_id ({user_id}) into financial tool calls")
            for call in out.tool_calls:
                if call["name"] in ["get_financial_data", "analyze_expenses", "get_customer_insights"]:
                    # Ensure args is a dict
                    call_args = call.get("args", {})
                    if isinstance(call_args, str):
                        try:
                            call_args = json.loads(call_args)
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse args for tool {call['name']}: {call_args}")
                            call_args = {}
                            
                    if "user_id" not in call_args:
                        call_args["user_id"] = user_id
                        call["args"] = call_args # Update args in the call
            
            state["tools_used"].append({
                "tool": "agent",
                "action": "financial_tool_calls",
                "tools_called": [call["name"] for call in out.tool_calls],
                "user_id": user_id
            })
        elif hasattr(out, "tool_calls") and out.tool_calls:
            tool_names = [call["name"] for call in out.tool_calls]
            logging.warning(f"LLM wants to call tools: {tool_names}")
            state["tools_used"].append({
                "tool": "agent",
                "action": "tool_calls",
                "tools_called": tool_names
            })
        else:
            response_length = len(out.content) if hasattr(out, "content") else 0
            logging.warning(f"LLM generated direct response of length {response_length}")
            state["tools_used"].append({
                "tool": "agent",
                "action": "direct_response",
                "response_length": response_length
            })
        
        # Instead of replacing the entire message list, append the new message
        state["messages"].append(out)
    except Exception as e:
        # Handle any errors that might occur during LLM invocation
        error_msg = f"Error in React LLM invocation: {e}"
        logging.error(error_msg)
        state["tools_used"].append({
            "tool": "agent",
            "action": "error",
            "error": str(e)
        })
        # Create a fallback AI message
        fallback_message = AIMessage(content="I'm sorry, I encountered an error while processing your request. Could you try rephrasing your question?")
        state["messages"].append(fallback_message)
    
    return state


def nebula_handler(state: State):
    """Handles queries that should be directed to the Nebula API."""
    from src.tools import call_nebula_api
    import json
    
    message_content = get_latest_human_message(state)
    logging.warning(f"Nebula handler processing: {message_content[:50]}...")
    wallets = state.get("wallets", {})
    user_id = f"user-{hash(json.dumps(wallets)) % 10000}" if wallets else "insight-user"
    execute_tx = "sign" in message_content.lower() or "execute" in message_content.lower() or "send" in message_content.lower()
    
    try:
        nebula_response = call_nebula_api.invoke({
            "message": message_content,
            "execute": execute_tx,
            "user_id": user_id
        })
        
        logging.warning(f"Nebula API response: {str(nebula_response)[:100]}...")
        
        response_content = nebula_response.get("message", "Error processing Nebula response.")
        if "error" in nebula_response:
            response_content = f"Error from Nebula API: {nebula_response.get('error')}"
        
        state["tools_used"].append({
            "tool": "nebula_handler",
            "execute_tx": execute_tx,
            "success": "error" not in nebula_response,
            "has_actions": bool(nebula_response.get("actions", []))
        })
        
        if actions := nebula_response.get("actions", []):
            # Simple formatting for actions
            response_content += "\n\n**Actions:**\n" + json.dumps(actions, indent=2)
        
        ai_message = AIMessage(content=response_content)
        
    except Exception as e:
        logging.error(f"Error in Nebula handler: {e}")
        ai_message = AIMessage(content=f"I encountered an error when trying to get blockchain data: {str(e)}")
        state["tools_used"].append({"tool": "nebula_handler", "success": False, "error": str(e)})
    
    state["messages"].append(ai_message)
    return state


def should_continue(state: State) -> Literal["tools", "inject_params", END]:
    """Determines whether to continue processing or end the graph."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message is not an AIMessage, end.
    if not isinstance(last_message, AIMessage):
        state["tools_used"].append({
            "tool": "should_continue",
            "decision": "END",
            "reason": f"Last message is not AIMessage: {type(last_message)}"
        })
        logging.warning(f"should_continue: ending because last message is not AIMessage: {type(last_message)}")
        return END

    # If the AIMessage has tool calls, route to the tools node.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [call["name"] for call in last_message.tool_calls]
        state["tools_used"].append({
            "tool": "should_continue",
            "decision": "tools",
            "reason": "AIMessage has tool calls",
            "tools": tool_names
        })
        logging.warning(f"should_continue: routing to tools node with tool calls: {tool_names}")
        return "tools"

    # Otherwise, route to inject_params to finalize the response.
    state["tools_used"].append({
        "tool": "should_continue",
        "decision": "inject_params",
        "reason": "AIMessage has no tool calls"
    })
    logging.warning("should_continue: proceeding to inject_params")
    return "inject_params"


def inject_params(state: State) -> State:
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, "content"):
        try:
            # Attempt to format with wallets, handle missing keys gracefully
            if state.get("wallets"):
                last_message.content = last_message.content.format_map(state["wallets"])
                state["tools_used"].append({
                    "tool": "inject_params",
                    "action": "format_with_wallets",
                    "wallets_count": len(state["wallets"])
                })
            else:
                 state["tools_used"].append({
                    "tool": "inject_params",
                    "action": "no_wallets_to_format"
                 })
        except KeyError as e:
            # Log specific key error but don't crash
            logging.warning(f"Inject_params formatting error (KeyError): {e}. Content: {last_message.content}")
            state["tools_used"].append({"tool": "inject_params", "action": "format_key_error", "error": str(e)})
        except Exception as e:
             # Catch other potential formatting errors
            logging.error(f"Inject_params formatting error: {e}. Content: {last_message.content}")
            state["tools_used"].append({"tool": "inject_params", "action": "format_error", "error": str(e)})
    else:
        state["tools_used"].append({
            "tool": "inject_params",
            "action": "no_formatting_needed",
            "reason": f"Last message type: {type(last_message)}"
        })
    
    # Always return state, even if formatting failed
    return state


def financial_handler(state: State):
    """
    Handles financial queries by adding a specific system prompt to guide the main agent.
    Relies on the main agent node to select and call the appropriate financial tools.
    """
    message_content = get_latest_human_message(state)
    user_id = state.get("user_id", "default_user")
    
    logging.warning(f"Financial handler processing message: '{message_content}' for user: {user_id}")
    
    # Add financial system prompt to guide the agent
    system_prompt = """
You are a financial analyst AI assistant with access to business financial data in MongoDB.
Use these tools to analyze the data and answer financial questions:
- For customer data queries, use get_customer_insights
- For expense analysis, use analyze_expenses
- For overall financial data, use get_financial_data

Always analyze the data before responding, and provide specific insights based on what you find.
    """
    
    system_message = SystemMessage(content=system_prompt)
    
    # Prepend the system message if it's not already there
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        state["messages"].insert(0, system_message)
        state["tools_used"].append({
            "tool": "financial_handler",
            "action": "added_financial_prompt"
        })
        logging.info("Financial system prompt added.")
    else:
        logging.warning("System prompt already exists, not adding financial prompt again.")

    # No direct tool calls here - always delegate to the main agent node
    logging.warning(f"Financial handler delegating to agent node for: '{message_content}'")
    
    # The graph structure will now direct this state to the 'agent' node
    return state 


def build_graph():
    """Constructs and compiles the agent execution graph."""
    builder = StateGraph(State)
    builder.add_node("extract_wallets", extract_wallets)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(react_tools))
    builder.add_node("inject_params", inject_params)
    builder.add_node("nebula_handler", nebula_handler)
    builder.add_node("general_handler", general_handler) 
    builder.add_node("financial_handler", financial_handler)

    # Intent routing at the start
    builder.add_conditional_edges(
        START,
        intent_router,
        {
            "extract_wallets": "extract_wallets", 
            "nebula_handler": "nebula_handler",
            "general_handler": "general_handler",
            "financial_handler": "financial_handler" 
        }
    )
    
    # Flow after handlers
    builder.add_edge("extract_wallets", "agent")
    builder.add_edge("general_handler", "agent") # General handler also goes to agent
    builder.add_edge("financial_handler", "agent") # Financial handler goes to agent
    builder.add_edge("nebula_handler", END) # Nebula is terminal
    
    # Main agent loop
    builder.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "inject_params": "inject_params", # Route to inject_params if no tools
        END: END # Allow ending directly from agent if needed (e.g., error)
    })
    builder.add_edge("tools", "agent") # Loop back to agent after tools run
    builder.add_edge("inject_params", END) # Final step before ending

    return builder.compile(checkpointer=MemorySaver())


def general_handler(state: State):
    """
    Handles general queries that aren't specifically blockchain or financial.
    Adds the standard react prompt if needed and passes to the agent.
    """
    message_content = get_latest_human_message(state)
    logging.warning(f"General handler processing message: '{message_content}'")
    
    # Ensure the standard react prompt is present if no system prompt exists
    if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
        state["messages"].insert(0, SystemMessage(content=react_template))
        state["tools_used"].append({
            "tool": "general_handler",
            "action": "added_react_prompt"
        })
        logging.info("Standard React system prompt added.")
    else:
         logging.warning("System prompt already exists, not adding react prompt again.")

    # Delegate to the main agent node
    logging.warning(f"General handler delegating to agent node for: '{message_content}'")
    return state


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    # Example Financial Query
    inputs = {
        "messages": [HumanMessage(content="how many customers do i have?")],
        "wallets": {},
        "tools_used": [],
        "user_id": "test_user_123" # Example user ID
    }
    
    # Example Blockchain Query
    # inputs = {
    #     "messages": [HumanMessage(content="How much USDT does 0xC22166664e820cdA6bf4cedBdbb4fa1E6A84C440 own?")],
    #     "wallets": {},
    #     "tools_used": []
    # }

    graph = build_graph()
    # Draw the graph (optional)
    try:
        graph.get_graph().draw_mermaid_png(output_file_path="diagram.png")
        print("Graph diagram saved to diagram.png")
    except Exception as draw_error:
        print(f"Could not draw graph: {draw_error}")

    # Stream the execution
    print("\n--- Running Graph ---")
    final_state = None
    for step in graph.stream(inputs, config=config, stream_mode="values"):
        final_state = step
        print("\n--- Step Output ---")
        # Pretty print step keys and message types/content for clarity
        for key, value in step.items():
            if key == "messages":
                print(f"  {key}:")
                for msg in value:
                    msg_type = type(msg).__name__
                    content_preview = str(getattr(msg, 'content', 'N/A'))[:80] + ("..." if len(str(getattr(msg, 'content', 'N/A'))) > 80 else "")
                    tool_calls = getattr(msg, 'tool_calls', None)
                    print(f"    - {msg_type}: {content_preview}")
                    if tool_calls:
                        print(f"      Tool Calls: {tool_calls}")
            else:
                print(f"  {key}: {value}")
    
    print("\n--- Final State Messages ---")
    if final_state and final_state.get("messages"):
        for msg in final_state["messages"]:
             msg.pretty_print()
    else:
        print("No final state or messages found.")
    print("--- Graph Execution Complete ---")
