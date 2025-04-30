from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any
from uuid import uuid4

from api.models import ConversationRequest, ConversationResponse, Message
from api.dependencies import get_graph, get_config
from api.services import ConversationService

router = APIRouter()

@router.post("/conversation/{user_id}", response_model=ConversationResponse)
async def process_conversation(
    request: Request,
    user_id: str,
    conversation: ConversationRequest = None
):
    """
    Process a conversation with the AI assistant.
    This endpoint handles multiple operations:
    - Send a message and get a response (default)
    - Retrieve conversation history
    - Clear conversation history

    The conversation history is maintained persistently until explicitly cleared.
    """
    # Initialize graph
    graph = get_graph()
    
    # Check if this is a new session (page refresh)
    # We can use this to track new sessions for analytics
    session_id = str(request.headers.get('sec-fetch-site', ''))
    is_new_session = session_id == 'none'
    
    # Determine operation from request
    operation = conversation.operation if conversation and conversation.operation else "message"
    
    # OPERATION: Clear conversation history
    if operation == "clear":
        try:
            success = await ConversationService.clear_conversation(user_id)
            return ConversationResponse(
                success=success,
                messages=[],
                response="Conversation history cleared"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear conversation history: {str(e)}")
    
    # OPERATION: Get conversation history
    if operation == "history":
        try:
            messages = await ConversationService.get_conversation_history(user_id)
            return ConversationResponse(
                messages=messages,
                success=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")
    
    # OPERATION: Process new message (default)
    if not conversation or not conversation.message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        # Initialize input state with user_id for MongoDB queries
        inputs = {
            "messages": [{"type": "human", "content": conversation.message}],
            "wallets": {},
            "tools_used": [],
            "user_id": user_id  # Add user_id to the state
        }
        
        # Configure the graph with user thread ID
        config = {"configurable": {"thread_id": user_id}}
        
        # Process the conversation
        result = await ConversationService.process_message(
            graph=graph,
            message=conversation.message,
            user_id=user_id
        )
        
        if not result["success"]:
            return ConversationResponse(
                response="I'm having trouble processing your request right now.",
                messages=result.get("messages", []),
                success=False,
                error=result.get("error", "Unknown error")
            )
        
        return ConversationResponse(
            response=result["response"],
            messages=result["messages"],
            success=True
        )
    except Exception as e:
        # Return a user-friendly error response
        return ConversationResponse(
            response="I'm having trouble processing your request right now. Our team has been notified of the issue.",
            messages=[{"role": "user", "content": conversation.message}],
            success=False,
            error=f"Internal server error: {str(e)}"
        )

@router.get("/conversation/{user_id}/history", response_model=ConversationResponse)
async def get_history(user_id: str):
    """Get conversation history for a user."""
    try:
        messages = await ConversationService.get_conversation_history(user_id)
        return ConversationResponse(
            messages=messages,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history") 