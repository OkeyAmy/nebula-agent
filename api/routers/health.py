from fastapi import APIRouter, Depends
from typing import Dict

from api.models import HealthCheckResponse
from api.dependencies import check_api_keys, get_graph

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
@router.head("/health")
async def health_check():
    """
    Check the health status of the service and its dependencies.
    Returns the status of API keys and the LangGraph availability.
    Supports both GET and HEAD requests.
    """
    # Check API keys
    api_keys = check_api_keys()
    
    # Determine overall status
    status = "healthy"
    graph_status = "available"
    
    # Check if any key is missing
    if not all(api_keys.values()):
        status = "degraded"
    
    # Try to initialize graph to check if it's working
    try:
        # Just initialize the graph (won't run it)
        get_graph()
    except Exception as e:
        status = "degraded"
        graph_status = f"unavailable: {str(e)}"
    
    return HealthCheckResponse(
        status=status,
        api_keys=api_keys,
        graph=graph_status
    ) 