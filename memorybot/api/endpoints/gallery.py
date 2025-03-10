from fastapi import APIRouter, HTTPException, Depends, Query, status, Request
from memorybot.services import QdrantService
from memorybot.core.database import get_qdrant_client
from memorybot.services.clip_service import ClipService
from memorybot.services.gemini_service import GeminiService
from typing import List, Dict, Any
import logging
from qdrant_client.http import models
from memorybot.core.cache import SearchCache
from memorybot.core.middleware import RateLimiter
import hashlib
import time
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services with dependency injection
def get_services():
    qdrant_client = get_qdrant_client()
    return {
        "qdrant_service": QdrantService(qdrant_client),
        "clip_service": ClipService(),
        "gemini_service": GeminiService()
    }

# Initialize cache and rate limiter
search_cache = SearchCache()
rate_limiter = RateLimiter()

class SearchError(Exception):
    """Custom exception for search-related errors"""
    def __init__(self, message: str, details: Dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)



@router.get("/", response_model=List[Dict[str, Any]])
async def get_gallery_images(
    refresh: bool = Query(False, description="Force refresh the cache"),
    services: Dict = Depends(get_services)
):
    """Get all images from the vector database"""
    try:
        qdrant_service = services["qdrant_service"]
        
        # If refresh is requested, invalidate the cache
        if refresh:
            qdrant_service.cache.delete("all_gallery_images")
            logger.info("Gallery cache invalidated due to refresh request")
        
        results = await qdrant_service.get_all_images()
        
        # Format the results to match the expected structure
        formatted_results = []
        for result in results:
            formatted_results.append({
                "filename": result.get("filename", ""),
                "metadata": {
                    "description": result.get("analysis", {}).get("description", ""),
                    "place": result.get("place", "Unknown"),
                    "event": result.get("event", "Unknown"),
                    "who": result.get("who", "Unknown"),
                    "year": result.get("year", "Unknown")
                }
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
