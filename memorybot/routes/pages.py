from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from datetime import datetime
import logging
from pathlib import Path
from fastapi.responses import RedirectResponse

from memorybot.services import QdrantService, get_qdrant_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize templates
templates = Jinja2Templates(directory="memorybot/templates")

# Setup template filters
def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp)
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            if diff.seconds < 3600:
                return f"{diff.seconds // 60}m ago"
            return f"{diff.seconds // 3600}h ago"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        elif diff.days < 30:
            return f"{diff.days // 7}w ago"
        elif diff.days < 365:
            return f"{diff.days // 30}mo ago"
        return f"{diff.days // 365}y ago"
    except:
        return "Unknown"

templates.env.filters["format_timestamp"] = format_timestamp

@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "premium-homepage.html",
        {"request": request}
    )

@router.get("/gallery")
async def gallery_page(request: Request):
    try:
        qdrant_service = QdrantService(get_qdrant_client())
        images = await qdrant_service.get_all_images()
        
        formatted_images = []
        for image in images:
            formatted_images.append({
                "filename": image.get("filename", ""),
                "metadata": {
                    "description": image.get("analysis", {}).get("description", ""),
                    "place": image.get("place", "Unknown"),
                    "event": image.get("event", "Unknown"),
                    "who": image.get("who", "Unknown"),
                    "year": image.get("year", "Unknown")
                }
            })

        return templates.TemplateResponse(
            "premium-gallery.html",
            {
                "request": request,
                "images": formatted_images
            }
        )
    except Exception as e:
        logger.error(f"Error loading gallery: {e}")
        return templates.TemplateResponse(
            "premium-gallery.html",
            {
                "request": request,
                "images": [],
                "error": str(e)
            }
        )

@router.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )

@router.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse(
        "chat-interface.html",
        {
            "request": request,
            "page_title": "Chat with MemoryBot",
            "initial_message": "Hello! I'm your MemoryBot assistant. How can I help you with your photos today?"
        }
    )

@router.get("/image/{filename}")
async def image_detail(request: Request, filename: str):
    try:
        qdrant_service = QdrantService(get_qdrant_client())
        image_data = await qdrant_service.get_image_by_filename(filename)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        formatted_image = {
            "filename": image_data.get("filename", ""),
            "metadata": {
                "description": image_data.get("analysis", {}).get("description", ""),
                "place": image_data.get("place", "Unknown"),
                "event": image_data.get("event", "Unknown"),
                "who": image_data.get("who", "Unknown"),
                "year": image_data.get("year", "Unknown"),
                "keywords": image_data.get("analysis", {}).get("keywords", [])
            }
        }
        
        return templates.TemplateResponse(
            "image-detail.html",
            {
                "request": request,
                "image": formatted_image
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading image detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/{filename}/delete")
async def delete_image(request: Request, filename: str):
    """Delete an image"""
    try:
        qdrant_service = QdrantService(get_qdrant_client())
        success = await qdrant_service.delete_image(filename)
        
        if success:
            return RedirectResponse(url="/gallery", status_code=303)
        else:
            raise HTTPException(status_code=404, detail="Image not found or could not be deleted")
            
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 