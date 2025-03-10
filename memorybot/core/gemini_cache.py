from typing import Dict, Any
import time
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GeminiCache:
    _instance = None
    CACHE_FILE = "gemini_cache.json"
    CACHE_DIR = Path("cache")
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiCache, cls).__new__(cls)
            cls._instance.cache = {}
            cls._instance.CACHE_DIR.mkdir(exist_ok=True)
            cls._instance._load_cache()
        return cls._instance
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            cache_file = self.CACHE_DIR / self.CACHE_FILE
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached Gemini responses")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            cache_file = self.CACHE_DIR / self.CACHE_FILE
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get(self, key: str) -> Dict[str, Any]:
        """Get cached response if exists and not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < 86400:  # 24 hour TTL
                return entry['data']
            else:
                del self.cache[key]
                self._save_cache()
        return None
    
    def set(self, key: str, value: Dict[str, Any]):
        """Cache a response"""
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        }
        self._save_cache() 