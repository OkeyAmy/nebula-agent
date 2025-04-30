from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ConversationService:
    """Service for handling conversation operations."""
    
    @staticmethod
    async def process_message(graph, message: str, user_id: str) -> Dict[str, Any]:
        """
        Process a user message through the LangGraph.
        
        Args:
            graph: The LangGraph instance
            message: User message to process
            user_id: Unique identifier for the user
            
        Returns:
            Dict containing the response and messages
        """
        try:
            # Format input for the graph
            inputs = {
                "messages": [HumanMessage(content=message)],
                "wallets": {},
                "tools_used": []
            }
            
            # Configure the graph with user thread ID
            config = {"configurable": {"thread_id": user_id}}
            
            # Process the graph and collect all steps
            steps = list(graph.stream(inputs, stream_mode="values", config=config))
            
            if not steps:
                return {
                    "success": False,
                    "error": "No response from AI assistant",
                    "messages": [{"role": "user", "content": message, "timestamp": datetime.now()}]
                }
            
            final_step = steps[-1]
            
            # Get the last AI message as the response
            response = ""
            if final_step.get("messages"):
                for msg in reversed(final_step["messages"]):
                    if isinstance(msg, AIMessage) and hasattr(msg, "content"):
                        response = msg.content
                        break
            
            # Format messages for response
            formatted_messages = []
            if final_step.get("messages"):
                for msg in final_step["messages"]:
                    if hasattr(msg, "content"):
                        role = "assistant" if isinstance(msg, AIMessage) else "user"
                        formatted_messages.append({
                            "role": role, 
                            "content": msg.content,
                            "timestamp": datetime.now()
                        })
            
            # Collect tools used for debugging
            tools_used = []
            for step in steps:
                if step.get("tools_used"):
                    tools_used.extend(step["tools_used"])
            
            return {
                "success": True,
                "response": response,
                "messages": formatted_messages,
                "tools_used": tools_used
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing message: {str(e)}",
                "messages": [{"role": "user", "content": message, "timestamp": datetime.now()}]
            }
    
    @staticmethod
    async def get_conversation_history(user_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        In a real application, this would retrieve messages from a database.
        For this example, we'll return an empty list as we don't persist conversations.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of message dictionaries
        """
        # In a real application, retrieve messages from database
        # For now, just return an empty list
        return []
    
    @staticmethod
    async def clear_conversation(user_id: str) -> bool:
        """
        Clear conversation history for a user.
        
        In a real application, this would delete messages from a database.
        For this example, we'll just return success as we don't persist conversations.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            True if successful
        """
        # In a real application, delete messages from database
        return True 