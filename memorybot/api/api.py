from fastapi import APIRouter
from memorybot.api.endpoints import upload, chat, gallery

api_router = APIRouter()

api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(gallery.router, prefix="/gallery", tags=["gallery"])
