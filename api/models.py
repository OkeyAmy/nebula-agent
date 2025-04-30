from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Model for a single message in a conversation."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the message")

class ConversationRequest(BaseModel):
    """Model for a conversation request."""
    message: str = Field(..., description="User message to process")
    operation: Optional[str] = Field(None, description="Operation to perform (message, history, clear)")
    
class ConversationResponse(BaseModel):
    """Model for a conversation response."""
    success: bool = Field(..., description="Whether the operation was successful")
    messages: List[Dict[str, Any]] = Field(default=[], description="List of messages in the conversation")
    response: Optional[str] = Field(None, description="Response from the assistant")
    error: Optional[str] = Field(None, description="Error message if the operation failed")
    
class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="Status of the service (healthy, degraded)")
    api_keys: Dict[str, bool] = Field(..., description="Status of required API keys")
    graph: str = Field(..., description="Status of the LangGraph")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the health check") 