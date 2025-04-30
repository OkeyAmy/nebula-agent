from enum import Enum

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class Intent(Enum):
    blockchain_query = "blockchain_query"
    nebula_query = "nebula_query"
    general_query = "general_query"
    financial_query = "financial_query"


class IntentChecker(BaseModel):
    intent: Intent = Field(..., description="The user's intent.")


llm_intent = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp', temperature=0.0  # Zero temperature for more consistent classifications
).with_structured_output(IntentChecker)

intent_template = """
You are an intent classifier for a multi-purpose AI assistant. Your job is to classify user messages into one of FOUR categories.
Be very specific in your classification and default to the most appropriate category based on the exact query content.

1. nebula_query (MUST USE for blockchain transaction queries):
   - ANY query about cryptocurrency prices (e.g. "What is the price of ETH?")
   - ANY blockchain transaction operations (sending, transferring crypto funds)
   - ANY mention of "transfer", "send", "pay" in the context of blockchain
   - Blockchain wallet balance checks
   - Chain switching or network operations
   - ANY query that involves reading or writing to the blockchain

2. blockchain_query (Use for information-only blockchain queries):
   - Contract metadata lookups
   - Token ownership information (e.g. "How many ETH does wallet X own?")
   - NFT collections data
   - Historical blockchain data
   - Any query mentioning tokens, NFTs, crypto wallets, or blockchain addresses

3. financial_query (MUST USE for ANY business database queries):
   - ANY query about business customers (e.g., "How many customers do I have?")
   - ANY query about business expenses, costs, or spending
   - ANY query about sales, revenue, or profits
   - ANY query about products or inventory
   - ANY mention of business transactions or payments NOT related to blockchain
   - Requests for financial analysis or business insights
   - Questions specifically about business database content
   - Queries containing terms like: customer, expense, product, sale, inventory, invoice, payment (not crypto)
   
4. general_query (Use ONLY for these very specific cases):
   - Simple greetings like "hey", "hello", "hi", "what's up", "good morning"
   - Questions about capabilities like "what can you do?", "help", "features"
   - Identity questions like "who are you?", "what are you?", "your name"
   - Completely non-blockchain and non-financial queries
   - Single words or very short phrases without sufficient context

IMPORTANT: Financial database queries should be classified as financial_query, NOT general_query or blockchain_query.

Examples:
- "hey there" → general_query
- "what is ethereum" → blockchain_query
- "eth price" → nebula_query
- "send 1 eth" → nebula_query
- "how many customers do i have" → financial_query
- "show my expenses" → financial_query
- "what are my recent sales" → financial_query
- "analyze my product inventory" → financial_query
- "how much did I sell last month" → financial_query
- "what was my last transaction" → financial_query (unless specifically about blockchain)

Message: {message}
"""
intent_prompt = ChatPromptTemplate.from_template(intent_template)
intent_chain = intent_prompt | llm_intent

if __name__ == "__main__":
    test_messages = [
        "What is the blockchain?",
        "how many customers do i have",
        "show my expenses",
        "hello",
        "what is the price of eth",
        "list my top customers",
        "what products are selling well"
    ]

    print("\n--- Intent Chain Test ---")
    for test_message in test_messages:
        print(f"Testing with message: \"{test_message}\"")
        try:
            result = intent_chain.invoke({"message": test_message})
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error invoking chain: {e}")
        print("-" * 20)
    print("--- Test Complete ---\n")
