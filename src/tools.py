import logging
import os
import json
import requests
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from thirdweb_ai import Insight, Nebula
from thirdweb_ai.adapters.langchain import get_langchain_tools

# Initialize Insight for blockchain data retrieval
insight = Insight(secret_key=os.getenv("THIRDWEB_SECRET_KEY"), chain_id=1)
insight_tools = get_langchain_tools(insight.get_tools())
# subset to tools I have tested
insight_tools = [
    tool
    for tool in insight_tools
    if tool.name
    in [
        "get_erc20_tokens",
        "get_contract_metadata",
        "get_erc721_tokens",
        "get_token_prices",
        "resolve",
    ]
]

# Initialize Nebula for direct API access
NEBULA_URL = "https://nebula-api.thirdweb.com/chat"
SECRET_KEY = os.getenv("THIRDWEB_SECRET_KEY")

@tool
def call_nebula_api(message: str, execute: bool = False, user_id: str = "tool-user"):
    """
    Send a message to Thirdweb's Nebula API and get blockchain-specific responses.
    
    This tool allows direct interaction with the Nebula API for blockchain queries 
    and can optionally execute transactions when the execute parameter is set to True.
    
    Example usage:
        - "What is the current price of Ethereum?" → Returns information about ETH price
        - "Show me the transaction history for address 0x123..." → Returns transaction data
    
    :param message: The user query about blockchain data
    :param execute: Whether to execute transactions (default: False)
    :param user_id: User identifier for the Nebula API
    :return: JSON response from the Nebula API containing assistant message and actions
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "x-secret-key": SECRET_KEY
        }
        payload = {
            "message": message,
            "user_id": user_id,
            "stream": False,
            "execute": execute
        }
        resp = requests.post(NEBULA_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Error calling Nebula API: {e}")
        return {"error": str(e)}


@tool
def extract_json_value(json_data, key_path):
    """Searches the web for relevant documents and extracts key highlights.

    This function uses Exa Search to retrieve up to 3 relevant web pages and extracts useful metadata,
    including key highlights and URLs. It is designed to help an AI assistant fetch real-time information.

    Example usage:
        - "What is the contract address of Ethereum?" → Returns a list of sources with relevant information.
        - "Find recent news about AI in finance." → Retrieves web content with AI-related financial news.

    :param query: The search query describing the information needed.
    :type query: str
    :return: A list of formatted strings containing URLs and highlighted excerpts.
    :rtype: list[str]
    """
    try:
        keys = key_path.split(".")
        value = json_data
        for key in keys:
            value = value.get(key)
            if value is None:
                raise KeyError(f"Key '{key}' not found in JSON structure.")
        return value
    except Exception as e:
        logging.error(f"Error extracting JSON value: {e}")
        return None


@tool
def count_json_list(json_data, key_path):
    """
    Counts the number of items in a list at a given key path in a JSON object.

    Example:
    json_data = {"data": [{"id": 1}, {"id": 2}]}
    count_json_list(json_data, "data")  # Returns 2

    :param json_data: Dictionary representing JSON data.
    :param key_path: String representing the nested key path (e.g., "data.items").
    :return: Integer count of list items or None if the path is invalid.
    """
    try:
        # Split key path into list of keys
        keys = key_path.split(".")
        value = json_data

        # Traverse the JSON using the keys
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None  # Return None if the key path doesn't exist or is invalid

        # Check if the final value is a list and return the count
        return len(value) if isinstance(value, list) else None
    except Exception:
        # In case of any error (though it should be rare), return None
        return None


@tool
def retrieve_web_content(query: str) -> list[str]:
    """Function to retrieve usable documents for AI assistant

    You can for example find the address of a token by its ticker or name:

    What is the token contract address of Ethereum? -> Returns the contract address of Ethereum token
    """
    # Initialize the Exa Search retriever
    retriever = ExaSearchRetriever(
        k=3, highlights=True, exa_api_key=os.getenv("EXA_API_KEY"), use_autoprompt=True
    )

    # Define how to extract relevant metadata from the search results
    document_prompt = PromptTemplate.from_template(
        """
    <source>
        <url>{url}</url>
        <highlights>{highlights}</highlights>
    </source>
    """
    )

    # Create a chain to process the retrieved documents
    document_chain = (
        RunnableLambda(
            lambda document: {
                "highlights": document.metadata.get("highlights", "No highlights"),
                "url": document.metadata["url"],
            }
        )
        | document_prompt
    )

    # Execute the retrieval and processing chain
    retrieval_chain = retriever | document_chain.map()

    return retrieval_chain.invoke(query)

@tool
def get_financial_data(user_id: str) -> Dict[str, Any]:
    """
    Retrieve financial data for a specific user including contacts, expenses, products, and transactions.
    
    This tool queries the MongoDB database to get comprehensive financial information for a user.
    Use this when you need to provide an overview of financial status or conduct a broad financial analysis.
    
    Args:
        user_id: The unique identifier of the user
        
    Returns:
        A dictionary containing all available financial data
    """
    # Handle both direct call and invoke with dict
    if isinstance(user_id, dict):
        user_id = user_id.get("user_id", "default_user")
    
    import asyncio
    from api.database import MongoDB
    
    async def _get_data():
        # Initialize MongoDB connection if not already initialized
        try:
            if MongoDB.db is None:
                await MongoDB.connect()
            
            # Retrieve financial data from MongoDB
            contacts = await MongoDB.get_contacts(user_id)
            expenses = await MongoDB.get_expenses(user_id)
            products = await MongoDB.get_products(user_id)
            transactions = await MongoDB.get_transactions(user_id)
            
            return {
                "contacts": contacts,
                "expenses": expenses,
                "products": products,
                "transactions": transactions,
                "success": True
            }
        except Exception as e:
            logging.error(f"Error retrieving financial data: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    return asyncio.run(_get_data())

@tool
def analyze_expenses(user_id: str) -> Dict[str, Any]:
    """
    Analyze a user's expenses to identify patterns, categories, and trends.
    
    This tool queries expense data and performs analysis on spending patterns.
    Use this when asked about expense management, budget planning, or spending habits.
    
    Args:
        user_id: The unique identifier of the user
        
    Returns:
        A dictionary containing expense analysis results
    """
    # Handle both direct call and invoke with dict
    if isinstance(user_id, dict):
        user_id = user_id.get("user_id", "default_user")
    
    import asyncio
    from api.database import MongoDB
    
    async def _analyze():
        # Initialize MongoDB connection if not already initialized
        try:
            if MongoDB.db is None:
                await MongoDB.connect()
            
            # Retrieve expense data from MongoDB
            expenses = await MongoDB.get_expenses(user_id)
            
            if not expenses:
                return {
                    "error": "No expense data found for this user",
                    "success": False
                }
            
            # Perform basic analysis
            total_expenses = sum(expense.get("amount", 0) for expense in expenses)
            expense_categories = {}
            
            for expense in expenses:
                category = expense.get("category", "Uncategorized")
                amount = expense.get("amount", 0)
                if category in expense_categories:
                    expense_categories[category] += amount
                else:
                    expense_categories[category] = amount
            
            # Sort categories by amount
            sorted_categories = sorted(
                expense_categories.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                "total_expenses": total_expenses,
                "expense_categories": dict(sorted_categories),
                "expense_count": len(expenses),
                "success": True
            }
        except Exception as e:
            logging.error(f"Error analyzing expenses: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    return asyncio.run(_analyze())

@tool
def get_customer_insights(user_id: str) -> Dict[str, Any]:
    """
    Analyze customer relationships and transaction patterns.
    
    This tool combines contact and transaction data to provide insights about customer relationships.
    Use this when asked about customer management, top customers, or transaction patterns.
    
    Args:
        user_id: The unique identifier of the user
        
    Returns:
        A dictionary containing customer analysis results
    """
    # Handle both direct call and invoke with dict
    if isinstance(user_id, dict):
        user_id = user_id.get("user_id", "default_user")
    
    import asyncio
    from api.database import MongoDB
    
    async def _analyze():
        # Initialize MongoDB connection if not already initialized
        try:
            if MongoDB.db is None:
                await MongoDB.connect()
            
            # Retrieve customer and transaction data from MongoDB
            contacts = await MongoDB.get_contacts(user_id)
            transactions = await MongoDB.get_transactions(user_id)
            
            if not contacts:
                return {
                    "error": "No contact data found for this user",
                    "success": False,
                    "customer_count": 0
                }
            
            # Map transactions to contacts
            customer_transactions = {}
            
            for transaction in transactions:
                contact_id = transaction.get("contactId")
                if not contact_id:
                    continue
                
                amount = transaction.get("amount", 0)
                if contact_id in customer_transactions:
                    customer_transactions[contact_id]["total"] += amount
                    customer_transactions[contact_id]["transactions"].append(transaction)
                else:
                    customer_transactions[contact_id] = {
                        "total": amount,
                        "transactions": [transaction]
                    }
            
            # Add contact details
            for contact in contacts:
                contact_id = contact.get("_id") or contact.get("id")
                if contact_id in customer_transactions:
                    customer_transactions[contact_id]["contact"] = contact
            
            # Sort customers by transaction total
            top_customers = sorted(
                [
                    {
                        "contact": data.get("contact", {}),
                        "total_value": data["total"],
                        "transaction_count": len(data["transactions"])
                    }
                    for contact_id, data in customer_transactions.items()
                    if "contact" in data
                ],
                key=lambda x: x["total_value"],
                reverse=True
            )[:10]  # Get top 10
            
            return {
                "top_customers": top_customers,
                "customer_count": len(contacts),
                "transaction_count": len(transactions),
                "success": True
            }
        except Exception as e:
            logging.error(f"Error analyzing customer data: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "customer_count": 0
            }
    
    return asyncio.run(_analyze())
