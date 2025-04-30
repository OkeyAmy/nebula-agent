import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from thirdweb_ai import Insight
from thirdweb_ai.adapters.pydantic_ai import get_pydantic_ai_tools

_ = load_dotenv()

insight = Insight(secret_key=os.getenv("THIRDWEB_SECRET_KEY"), chain_id=1)
tools = get_pydantic_ai_tools(insight.get_tools())
agent = Agent("openai:gpt-4o-mini", tools=tools)

agent.run_sync(
    "What is the most recent transaction hash and timestamp of thirdweb.eth?"
)
