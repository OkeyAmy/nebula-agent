import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Primary LLM using Google's Gemini
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

# Fallback LLM using OpenAI only if API key is available
FALLBACK_LLM = None
if os.getenv("OPENAI_API_KEY"):
    FALLBACK_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
