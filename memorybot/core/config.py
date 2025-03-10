from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load .env file
load_dotenv()

# Add these logging configurations
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/memorybot.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'memorybot': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

class Settings(BaseSettings):
    PROJECT_NAME: str = "Conversational Memory Bot– AI-Powered PhotoGallery Assistant"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    IMAGES_DIR: Path = BASE_DIR / "images"
    STATIC_DIR: Path = BASE_DIR / "memorybot" / "static"
    
    # Qdrant settings
    QDRANT_PATH: str = "qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "image_embeddings"
    VECTOR_SIZE: int = 512

    # Gemini settings
    GEMINI_API_KEY: str

    # Add logging config
    LOGGING: dict = LOGGING_CONFIG

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
