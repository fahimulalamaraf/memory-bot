from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List
from memorybot.models.image import ImageMetadata, ImageUploadResponse
from memorybot.services.qdrant_service import QdrantService
from memorybot.core.database import get_qdrant_client
import aiofiles
import os
import logging
from pathlib import Path
from memorybot.core.config import settings
from datetime import datetime
import imghdr
import asyncio
from memorybot.core.constants import UPLOAD_SETTINGS

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize templates
templates = Jinja2Templates(directory="memorybot/templates")

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

async def process_upload(
    file_content: bytes,
    filename: str,
    metadata: dict,
    qdrant_service: QdrantService
) -> bool:
    """Process a single file upload in the background"""
    try:
        # Create images directory if it doesn't exist
        os.makedirs(str(settings.IMAGES_DIR), exist_ok=True)
        
        # Generate file path
        file_path = settings.IMAGES_DIR / filename
        
        # Save file asynchronously
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(file_content)
        
        # Store in Qdrant with embedding
        success = await qdrant_service.store_image_with_metadata(
            str(file_path), 
            metadata
        )
        
        if success:
            # Invalidate gallery cache after successful upload
            qdrant_service.cache.delete("all_gallery_images")
            logger.info(f"Gallery cache invalidated after uploading {filename}")
        
        logger.info(f"Background upload completed for {filename}")
        return success
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return False

@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render upload page"""
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )

def validate_image(file_content: bytes, filename: str) -> bool:
    """Validate image file type and content"""
    # Check file extension
    ext = filename.lower().split('.')[-1]
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Verify file content is actually an image
    img_type = imghdr.what(None, file_content)
    if not img_type or img_type not in ALLOWED_EXTENSIONS:
        return False
        
    return True

@router.post("/file", response_model=ImageUploadResponse)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    who: Optional[str] = Form(None),
    place: Optional[str] = Form(None),
    event: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """Upload single image in background"""
    try:
        # Read file content once
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds maximum limit of 10MB"
            )
        
        # Validate image
        if not validate_image(file_content, file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload JPG, PNG, GIF, or WebP files only."
            )

        # Get Qdrant client and service
        client = get_qdrant_client()
        qdrant_service = QdrantService(client)

        # Prepare metadata
        metadata = {
            "who": who.lower() if who else None,
            "place": place.lower() if place else None,
            "event": event.lower() if event else None,
            "year": year.lower() if year else None,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        # Modified background task to include delay
        async def process_with_delay():
            await process_upload(file_content, file.filename, metadata, qdrant_service)
            # Add delay after processing
            await asyncio.sleep(UPLOAD_SETTINGS["DELAY_SECONDS"])
            
        # Add modified task to background tasks
        background_tasks.add_task(process_with_delay)
        
        logger.info(f"Upload task scheduled for {file.filename} with {UPLOAD_SETTINGS['DELAY_SECONDS']} second delay")
        
        # Return immediate response while processing continues in background
        return ImageUploadResponse(
            success=True,
            filename=file.filename,
            message=f"Upload started and processing in background. Please wait {UPLOAD_SETTINGS['DELAY_SECONDS']} seconds before next upload."
        )
            
    except HTTPException as e:
        logger.error(f"Error in upload_image: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_upload(
    files_data: List[tuple[bytes, str, dict]],
    qdrant_service: QdrantService
) -> List[bool]:
    """Process multiple files with delay between each upload"""
    results = []
    for file_content, filename, metadata in files_data:
        try:
            # Process single file
            success = await process_upload(file_content, filename, metadata, qdrant_service)
            results.append({"filename": filename, "success": success})
            
            # Add delay after each file (except the last one)
            if files_data[-1] != (file_content, filename, metadata):
                logger.info(f"Waiting {UPLOAD_SETTINGS['DELAY_SECONDS']} seconds before next file...")
                await asyncio.sleep(UPLOAD_SETTINGS["DELAY_SECONDS"])
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({"filename": filename, "success": False})
    
    # Invalidate gallery cache after all uploads are complete
    qdrant_service.cache.delete("all_gallery_images")
    logger.info("Gallery cache invalidated after batch upload")
    
    return results

@router.post("/files", response_model=List[ImageUploadResponse])
async def upload_multiple_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    who: Optional[str] = Form(None),
    place: Optional[str] = Form(None),
    event: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """Upload multiple images in background with delay between each"""
    try:
        # Prepare all files data
        files_data = []
        for file in files:
            # Read file content
            file_content = await file.read()
            
            # Validate file size
            if len(file_content) > UPLOAD_SETTINGS["MAX_FILE_SIZE"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds maximum limit of 10MB"
                )
            
            # Validate image
            if not validate_image(file_content, file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file: {file.filename}. Please upload JPG, PNG, GIF, or WebP files only."
                )
            
            # Prepare metadata
            metadata = {
                "who": who.lower() if who else None,
                "place": place.lower() if place else None,
                "event": event.lower() if event else None,
                "year": year.lower() if year else None,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "filename": file.filename,
                "content_type": file.content_type
            }
            
            files_data.append((file_content, file.filename, metadata))

        # Get Qdrant client and service
        client = get_qdrant_client()
        qdrant_service = QdrantService(client)
        
        # Process files in background with delay
        background_tasks.add_task(process_batch_upload, files_data, qdrant_service)
        
        # Return immediate response
        return [
            ImageUploadResponse(
                success=True,
                filename=filename,
                message=f"Upload queued. Files will be processed with {UPLOAD_SETTINGS['DELAY_SECONDS']} second delay between each."
            )
            for _, filename, _ in files_data
        ]
            
    except HTTPException as e:
        logger.error(f"Error in batch upload: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
