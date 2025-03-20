from enum import Enum

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class Intent(Enum):
    blockchain_query = "blockchain_query"
    general_query = "general_query"


class IntentChecker(BaseModel):
    intent: Intent = Field(..., description="The user's intent.")


llm_intent = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
).with_structured_output(IntentChecker)

intent_template = """
You have to determine the user's intent. The user may want to query the blockchain, or have a different intent. General queries may be simply asking questions regarding the blockchain, rather than directly querying it.

Message: {message}
"""
intent_prompt = ChatPromptTemplate.from_template(intent_template)
intent_chain = intent_prompt | llm_intent

if __name__ == "__main__":
    test_message = "What is the blockchain?"
    print(f"Testing with message: {test_message}")
    result = intent_chain.invoke({"message": test_message})
    print(f"Result: {result}")
