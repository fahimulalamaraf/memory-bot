from PIL import Image
from io import BytesIO
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.max_size = (800, 800)  # Maximum dimensions
        self.quality = 85  # JPEG quality (0-100)
        self.format = 'JPEG'  # Output format
    
    def process_image(self, image_bytes: bytes, filename: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Process image: resize, compress, and convert to JPEG"""
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize if larger than max_size while maintaining aspect ratio
            if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes with compression
            output = BytesIO()
            img.save(output, 
                    format=self.format, 
                    quality=self.quality, 
                    optimize=True)
            
            processed_bytes = output.getvalue()
            
            # Generate new filename with .jpg extension
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            
            return processed_bytes, new_filename
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, None
    
    def save_image(self, image_bytes: bytes, save_path: str) -> bool:
        """Save processed image to disk"""
        try:
            processed_bytes, _ = self.process_image(image_bytes, os.path.basename(save_path))
            if processed_bytes:
                with open(save_path, 'wb') as f:
                    f.write(processed_bytes)
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            return False 