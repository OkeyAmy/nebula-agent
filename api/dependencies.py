from typing import Callable, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv
import os

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from src.graph import build_graph

# Load environment variables
load_dotenv()

# Set up LLM cache
set_llm_cache(InMemoryCache())

@lru_cache()
def get_graph():
    """
    Returns a cached instance of the LangGraph.
    Uses lru_cache to ensure only one instance is created.
    """
    return build_graph()

def get_config() -> Dict[str, Dict[str, Any]]:
    """
    Returns the configuration for the LangGraph.
    """
    return {"configurable": {"thread_id": "1"}}

def check_api_keys() -> Dict[str, bool]:
    """
    Checks if necessary API keys are set in environment variables.
    Returns a dictionary with API statuses.
    """
    keys = {
        "thirdweb": os.getenv("THIRDWEB_SECRET_KEY") is not None,
        "google": os.getenv("GOOGLE_API_KEY") is not None,
        "exa": os.getenv("EXA_API_KEY") is not None,
        "openai": os.getenv("OPENAI_API_KEY") is not None,
    }
    
    return keys 