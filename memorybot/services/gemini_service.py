import google.generativeai as genai
from PIL import Image
import os
from typing import Dict, Any, List
from memorybot.core.config import settings
import logging
from fastapi import Request
from memorybot.core.middleware import RateLimiter
from google.generativeai.types import FunctionDeclaration
import json
from memorybot.core.constants import PROMPTS, LOG_MESSAGES

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        try:
            # Initialize Gemini with API key from settings
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info(LOG_MESSAGES["GEMINI_INIT"])
        except Exception as e:
            logger.error(LOG_MESSAGES["GEMINI_INIT_ERROR"].format(e))
            raise
        
    def store_analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Generate description and keywords for an image"""
        try:
            img = Image.open(image_path)
            
            # Generate description
            try:
                description = self._generate_description(img)
            except Exception as e:
                logger.error(LOG_MESSAGES["ERROR_GENERATING_DESCRIPTION"].format(e))
                description = ""
            
            # Generate keywords
            try:
                keywords = self._generate_keywords(img)
            except Exception as e:
                logger.error(LOG_MESSAGES["ERROR_GENERATING_KEYWORDS"].format(e))
                keywords = []
            
            return {
                "description": description,
                "keywords": keywords
            }
        except Exception as e:
            logger.error(LOG_MESSAGES["ERROR_ANALYZING_IMAGE"].format(e))
            return {
                "description": "",
                "keywords": []
            }

    def _generate_description(self, image: Image) -> str:
        """Generate a natural description of the image"""
        try:
            response = self.model.generate_content([PROMPTS["IMAGE_DESCRIPTION"], image])
            return response.text.strip() if response.text else ""
        except Exception as e:
            logger.error(LOG_MESSAGES["ERROR_GENERATING_DESCRIPTION"].format(e))
            return ""

    def _generate_keywords(self, image: Image) -> List[str]:
        """Generate relevant keywords/tags for the image"""
        try:
            response = self.model.generate_content([PROMPTS["IMAGE_KEYWORDS"], image])
            if response.text:
                keywords = [k.strip() for k in response.text.split(',')]
                return keywords
            return []
        except Exception as e:
            logger.error(LOG_MESSAGES["ERROR_GENERATING_KEYWORDS"].format(e))
            return []

    async def generate_content(self, prompt: str) -> str:
        """Generate content with rate limiting"""
        try:
            response = self.model.generate_content(prompt)
            
            # If response contains JSON, extract it
            if "```json" in response.text:
                try:
                    # Extract JSON between code blocks
                    json_str = response.text.split("```json")[1].split("```")[0].strip()
                    return json_str
                except Exception as e:
                    logger.error(LOG_MESSAGES["ERROR_EXTRACTING_JSON"].format(e))
                    return response.text
            
            # Return regular text response
            return response.text.strip()
            
        except Exception as e:
            logger.error(LOG_MESSAGES["ERROR_GENERATING_CONTENT"].format(e))
            return LOG_MESSAGES["GENERATE_ERROR_RESPONSE"]

    async def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyze an image using Gemini"""
        try:
            # Read the image
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
            
            # Create image parts for Gemini
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            ]

            # Get analysis from Gemini
            response = await self.model.generate_content([prompt, *image_parts])
            
            # Clean and return the response
            if response and hasattr(response, 'text'):
                return response.text.strip()
            return LOG_MESSAGES["UNABLE_TO_ANALYZE"]

        except Exception as e:
            logger.error(LOG_MESSAGES["ERROR_ANALYZING_IMAGE_GEMINI"].format(e))
            raise

    # Add new function for describing images with context
    async def generate_content_describe(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Generate content with conversation context for image descriptions"""
        try:
            # Format conversation history for context
            context = ""
            if conversation_history:
                context = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history[-5:]  # Last 5 messages
                ])
            
            # Combine context with prompt
            full_prompt = f"""
            Previous conversation:
            {context}

            Current request:
            {prompt}
            """
            
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating content with context: {e}")
            return "I encountered an error processing your request."
