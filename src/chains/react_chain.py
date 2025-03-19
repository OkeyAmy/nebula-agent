import os

from dotenv import load_dotenv

from src.llm import LLM
from src.tools import (
    count_json_list,
    extract_json_value,
    insight_tools,
    retrieve_web_content,
)

load_dotenv()

json_tools = [extract_json_value, count_json_list]
web_tools = [retrieve_web_content]

react_tools = insight_tools + json_tools + web_tools
react_llm = LLM.bind_tools(react_tools)

react_template = """
You have access to 'thirdweb' tools, which allow you to retrieve real-time Blockchain data. Never attempt to guess answers to Blockchain-related questions—always use the provided tools.  

When handling a JSON structure, you **must** use the appropriate tool to extract relevant information rather than relying on assumptions.  

You are capable of performing **multi-step processes** to answer complex queries. For example, you may:  
- Retrieve a token by ID using 'thirdweb' tools, then perform a web search to find its name. Or vice-versa, if a user only gives you a token name.
- Fetch a contract's details, then look up documentation to explain its functionality.  
- Query Blockchain transactions, then enrich results with external market data.  

If the 'thirdweb' tools do not provide sufficient information, you **must** use the web search tool to find additional context. For example Etherscan may provide the name of tokens if you construct a suitable web search.

Always aim for accuracy, combining Blockchain data with external sources when necessary. 
"""
