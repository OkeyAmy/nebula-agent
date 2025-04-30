# Insight Chat - Blockchain AI Assistant

Insight Chat is an AI-powered assistant for interacting with blockchain data using natural language. Built with `thirdweb-ai` using [thirdweb Insight](https://thirdweb.com/insight), LangChain and Google's Gemini, it provides real-time blockchain insights through an interactive chat interface.

## Features

- **Natural Language Queries**: Ask questions about blockchain data in plain English
- **Multi-step Reasoning**: Handles complex queries requiring multiple data lookups
- **Tool Integration**: 
  - Thirdweb blockchain tools
  - JSON data extraction
  - Web content retrieval
- **Rich Interface**: Beautiful terminal interface with progress tracking
- **Caching**: In-memory caching for faster responses
- **RESTful API**: FastAPI-based API for integrating with web applications

## Installation

### Prerequisites
- Python 3.12 or higher
- Thirdweb API key
- Google API key (for Gemini)
- OpenAI API key (for fallback)
- Exa API key (for web search)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/cjber/thirdweb-ai-langgraph.git
cd thirdweb-ai-langgraph
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# THIRDWEB_SECRET_KEY=your_thirdweb_key
# GOOGLE_API_KEY=your_google_key
# EXA_API_KEY=your_exa_api_key
# OPENAI_API_KEY=your_openai_api_key # used as fallback
```

## Usage

### Terminal Chat Interface

Run the chat interface:
```bash
python insight_chat.py
```

Example queries:

Note that the model supports memory:

* How much USDT does 0xC22166664e820cdA6bf4cedBdbb4fa1E6A84C440 own?
* How many different ERC20 tokens does this address own?

* What is the ENS name of 0xEb0effdFB4dC5b3d5d3aC6ce29F3ED213E95d675?
* How many different ERC20 tokens does this wallet own?
* How many ERC721 NFTs does this address own?

### RESTful API

Run the API server:
```bash
python run_api.py
```

For development with auto-reload:
```bash
python run_api.py --reload
```

API documentation is available at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

#### Example API Endpoints

- POST `/api/conversation/{user_id}` - Process a conversation with the AI assistant
- GET `/api/conversation/{user_id}/history` - Get conversation history
- GET `/api/health` - Health check endpoint

## Configuration

Environment Variables:
- `THIRDWEB_SECRET_KEY`: Your Thirdweb API secret key
- `GOOGLE_API_KEY`: Google API key (for Gemini)
- `OPENAI_API_KEY`: OpenAI API key (for fallback)
- `EXA_API_KEY`: Exa API key for web search

## Development

### Key Components

- **Graph Processing**: `src/graph.py`
- **LLM Configuration**: `src/llm.py`
- **Tools Integration**: `src/tools.py`
- **Intent Detection**: `src/chains/intent_chain.py`
- **Reasoning Chain**: `src/chains/react_chain.py`
- **API Layer**: `api/` directory
