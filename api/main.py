from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from api.dependencies import get_graph
from api.routers import conversations, health

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Thirdweb AI LangGraph API",
    description="RESTful API for blockchain AI assistant using LangGraph",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(conversations.router, prefix="/api", tags=["Conversations"])
app.include_router(health.router, prefix="/api", tags=["Health"])

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint that returns API information."""
    return {
        "message": "Welcome to Thirdweb AI LangGraph API",
        "docs": "/docs",
        "health": "/api/health"
    } 