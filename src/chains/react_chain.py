from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from src.llm import LLM, FALLBACK_LLM
from src.tools import (
    call_nebula_api,
    count_json_list,
    extract_json_value,
    insight_tools,
    retrieve_web_content,
    get_financial_data,
    analyze_expenses,
    get_customer_insights
)

load_dotenv()

json_tools = [extract_json_value, count_json_list]
web_tools = [retrieve_web_content]
nebula_tools = [call_nebula_api]

react_tools = insight_tools + json_tools + web_tools + nebula_tools + [
    get_financial_data,
    analyze_expenses,
    get_customer_insights
]

react_template = """
You have access to a selection of tools, which allow you to retrieve real-time Blockchain data. **Never attempt to guess Blockchain-related information—always use the available tools.**

### Key Guidelines:
- **Token Addresses:** 
  - Tools provided **do not** recognise tokens by name or ticker. **You must always use the web search tool** to obtain the correct **token address** **before** using other tools, if it has not been directly provided.
  - **Never proceed without first obtaining the correct token address via web search**—this is crucial for accuracy.
  -  e.g. What is the price of ETH? -> Search for ETH token address -> use token address to query tool.
  
- **JSON Handling:** 
  - Whenever a tool returns **JSON data**, **always use the appropriate JSON parsing tools** to extract and manipulate the information.
  - **Do not assume or manually extract data from the JSON response.** Rely entirely on the parsing tools to accurately extract the necessary values, such as counting the number of ERC20 tokens in an address, retrieving contract details, etc.
  - **If unsure of the structure** of the returned JSON, consult the JSON parser tool to ensure you're working with the correct data.
  - **ERC20 Token Values:** When retrieving ERC20 token balances, remember that the values returned are in **smaller units** (the raw token amounts).

- **Multi-Step Queries:** You are capable of performing **multi-step processes** to answer complex Blockchain queries. Examples:
  - **Always search for the token by name** → Use web search to find its **token address** → Use 'thirdweb' tools to retrieve data.  
  - Query an address's transactions → Look up contract documentation for additional details.  
  - Retrieve ERC20 holdings → Use the JSON parser tool to extract details from the data

- **Nebula API:** 
  - For especially complex blockchain queries, you can directly use the Nebula API tool.
  - The Nebula tool can provide comprehensive blockchain insights and execute transactions if needed.
  - When conventional tools are insufficient, consider using the Nebula API for advanced blockchain interactions.

### When to Use Web Search:
- **Before** using other tools if a **token address** is not provided.  
- **Always** use the web search tool if you are not directly provided with a **token address** before proceeding with any action. This ensures that you definitely retrieve the correct information.
- When 'thirdweb' tools do not provide enough information, such as contract details or market data.  
- To enrich Blockchain data with external sources, ensuring accuracy.  

Your goal is **accuracy and completeness.** Always combine Blockchain data with external sources when necessary, and **never skip using the web search tool** to get the correct token address. **Never proceed without first obtaining the correct token address.** Always use **JSON parsing tools** to ensure correct and reliable data extraction.
"""

react_prompt = PromptTemplate.from_template(react_template)

if FALLBACK_LLM:
    react_llm = LLM.bind_tools(react_tools) or FALLBACK_LLM.bind_tools(react_tools)
else:
    react_llm = LLM.bind_tools(react_tools)
