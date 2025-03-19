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
    """
    Extracts a value from a JSON object based on a key path.
    Example: For key_path 'data.0.balance', it extracts the balance value from the first item in the "data" list.
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
    Example: For key_path 'data', it counts how many items are in the "data" list.
    """
    try:
        keys = key_path.split(".")
        value = json_data
        for key in keys:
            value = value.get(key)
            if value is None:
                raise KeyError(f"Key '{key}' not found in JSON structure.")

        if isinstance(value, list):
            return len(value)
        else:
            logging.warning(f"Value at key_path '{key_path}' is not a list.")
            return None
    except Exception as e:
        logging.error(f"Error counting JSON list: {e}")
        return None


@tool
def retrieve_web_content(query: str) -> list[str]:
    """Function to retrieve usable documents for AI assistant"""
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

    # Retrieve and return the documents
    documents = retrieval_chain.invoke(query)
    return documents
