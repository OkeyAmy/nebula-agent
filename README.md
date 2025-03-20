# Insight Chat - Blockchain AI Assistant

Insight Chat is an AI-powered assistant for interacting with blockchain data using natural language. Built with `thirdweb-ai`, LangChain and Claude, it provides real-time blockchain insights through an interactive chat interface.

## Features

- **Natural Language Queries**: Ask questions about blockchain data in plain English
- **Multi-step Reasoning**: Handles complex queries requiring multiple data lookups
- **Tool Integration**: 
  - Thirdweb blockchain tools
  - JSON data extraction
  - Web content retrieval
- **Rich Interface**: Beautiful terminal interface with progress tracking
- **Caching**: In-memory caching for faster responses

## Installation

### Prerequisites
- Python 3.12 or higher
- Thirdweb API key
- Anthropic API key (for Claude 3)

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
# ANTHROPIC_API_KEY=your_anthropic_key
# EXA_API_KEY=your_exa_api_key
# OPENAI_API_KEY=your_openai_api_key # not required for insight_chat.py
```

## Usage

Run the chat interface:
```bash
python insight_chat.py
```

Example queries:

Note that the model supports memory:

* How many different ERC20 tokens does 0xF977814e90dA44bFA03b6295A0616a897441aceC own?
* What's the value of the first token?

* How much USDT does 0xC22166664e820cdA6bf4cedBdbb4fa1E6A84C440 own?
* What is the ENS name of 0xEb0effdFB4dC5b3d5d3aC6ce29F3ED213E95d675?

## Configuration

Environment Variables:
- `THIRDWEB_SECRET_KEY`: Your Thirdweb API secret key
- `ANTHROPIC_API_KEY`: Anthropic API key (for Claude 3)
- `EXA_API_KEY`: Exa API key for web search

## Development

### Key Components

- **Graph Processing**: `src/graph.py`
- **LLM Configuration**: `src/llm.py`
- **Tools Integration**: `src/tools.py`
- **Intent Detection**: `src/chains/intent_chain.py`
- **Reasoning Chain**: `src/chains/react_chain.py`
