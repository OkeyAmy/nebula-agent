# Insight Chat - Blockchain AI Assistant

Insight Chat is an AI-powered assistant for interacting with blockchain data using natural language. Built with LangChain and Claude 3 Haiku, it provides real-time blockchain insights through an interactive chat interface.

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
git clone https://github.com/yourusername/insight-chat.git
cd insight-chat
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
```

## Usage

Run the chat interface:
```bash
python insight_chat.py
```

Example queries:
- "Show me details for wallet 0x123..."
- "What's the balance of this NFT contract: 0x456..."
- "Explain this transaction: 0x789..."

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

### Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Please follow the existing code style and include tests for new features.

## License

MIT License - See LICENSE file for details
