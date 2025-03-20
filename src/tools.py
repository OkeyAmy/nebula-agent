import logging
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from thirdweb_ai import Insight
from thirdweb_ai.adapters.langchain import get_langchain_tools

insight = Insight(secret_key=os.getenv("THIRDWEB_SECRET_KEY"), chain_id=1)
insight_tools = get_langchain_tools(insight.get_tools())


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
