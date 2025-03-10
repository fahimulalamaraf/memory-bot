"""
MemoryBot - AI-Powered Photo Gallery Assistant
Main application entry point
"""

import uvicorn
import logging
from memorybot.core.logging_config import setup_logging
from memorybot.core.app_factory import create_app

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()

# Create FastAPI app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        workers=4
    )
