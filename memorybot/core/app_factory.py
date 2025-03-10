import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from memorybot.core.config import settings
from memorybot.core.middleware import RateLimiter
from memorybot.api.api import api_router
from memorybot.routes.pages import router as pages_router
from memorybot.services import QdrantService, get_qdrant_client

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    try:
        # Startup
        qdrant_client = get_qdrant_client()
        app.state.qdrant_service = QdrantService(qdrant_client)
        new_images = await app.state.qdrant_service.sync_images_with_db()
        logger.info(f"Added {new_images} new images to vector database")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        
    yield
    
    try:
        # Shutdown
        from memorybot.core.database import cleanup_client
        cleanup_client()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="AI-powered photo gallery management system",
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="memorybot/static"), name="static")
    app.mount("/images", StaticFiles(directory=str(settings.IMAGES_DIR)), name="images")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/v1/gallery/search"):
            await RateLimiter().check_rate_limit(request)
        response = await call_next(request)
        return response
    
    # Include routers
    app.include_router(api_router, prefix=settings.API_V1_STR)
    app.include_router(pages_router)
    
    return app 