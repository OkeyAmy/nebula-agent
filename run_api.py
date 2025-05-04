import uvicorn
import argparse

def main():
    """
    Run the FastAPI application.
    
    Command line arguments:
    --host: Host to bind the server to (default: 127.0.0.1)
    --port: Port to bind the server to (default: 8000)
    --reload: Enable auto-reload for development (default: False)
    """
    parser = argparse.ArgumentParser(description="Run the Thirdweb AI LangGraph API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 
