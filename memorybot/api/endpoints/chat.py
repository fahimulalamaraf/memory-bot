from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Request, Body
from memorybot.services.chat_service import ChatService
from typing import Dict, Any, List, Optional
import base64
from io import BytesIO
import logging
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import os
import uuid
from PIL import Image, ImageDraw
from fastapi.responses import Response
from memorybot.core.constants import API_MESSAGES, CHAT_MESSAGES

logger = logging.getLogger(__name__)
router = APIRouter()

# Define request model for JSON requests
class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None

@router.post("/message")
async def chat_message(
    request: Request,
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    try:
        # Use existing session_id or create new one
        current_session_id = session_id or str(uuid.uuid4())
        
        if not message.strip() and image:
            message = API_MESSAGES["DEFAULT_IMAGE_MESSAGE"]
        elif not message.strip() and not image:
            return {
                "text": API_MESSAGES["EMPTY_MESSAGE"],
                "type": "error",
                "results": [],
                "session_id": current_session_id
            }
        
        # Get image data if provided
        image_data = None
        if image:
            contents = await image.read()
            image_data = f"data:{image.content_type};base64,{base64.b64encode(contents).decode()}"
        
        chat_service = ChatService()
        response = await chat_service.handle_message(
            message=message,
            session_id=current_session_id,
            image_data=image_data
        )
        
        # Include session_id in response
        response["session_id"] = current_session_id
        return response

    except Exception as e:
        logger.error(f"Error in chat message endpoint: {e}")
        return {
            "text": API_MESSAGES["ERROR_PROCESSING"],
            "type": "error",
            "results": [],
            "session_id": current_session_id
        }

@router.get("/history")
async def get_chat_history() -> List[Dict[str, Any]]:
    """Get chat conversation history"""
    try:
        chat_service = ChatService()
        return chat_service.get_conversation_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/initial")
async def get_initial_message() -> Dict[str, str]:
    """Get the initial bot message"""
    return {
        "message": CHAT_MESSAGES["INITIAL_GREETING"]
    }

@router.get("/placeholder/{width}/{height}")
async def get_placeholder_image(width: int, height: int):
    """Serve placeholder images"""
    try:
        # Define default placeholder path
        default_placeholder = "memorybot/static/images/placeholder.jpg"
        
        # Check if the file exists
        if not os.path.exists(default_placeholder):
            # Create a simple placeholder image using PIL
            img = Image.new('RGB', (width, height), color='#f0f0f0')
            d = ImageDraw.Draw(img)
            text = f"{width}x{height}"
            
            # Save the generated placeholder
            img.save(default_placeholder)
        
        return FileResponse(default_placeholder)
    except Exception as e:
        logger.error(f"Error serving placeholder image: {e}")
        # Return a 1x1 transparent pixel as fallback
        return Response(content=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82', 
                      media_type="image/png")
