# API Messages
API_MESSAGES = {
    "ERROR_PROCESSING": "Sorry, I encountered an error processing your request.",
    "EMPTY_MESSAGE": "Please provide a message or an image to chat.",
    "DEFAULT_IMAGE_MESSAGE": "Please find similar images to this one.",
    "IMAGE_PROCESSING_ERROR": "I had trouble processing the image. Please try again.",
    "NO_SIMILAR_IMAGES": "I couldn't find any similar images.",
    "NO_RELEVANT_IMAGES": "I couldn't find any relevant images based on your query.",
    "NO_IMAGES_TO_DESCRIBE": "I don't see any recent images to describe. Could you show me which images you'd like me to describe?",
    "ERROR_DESCRIBING_IMAGES": "I encountered an error while trying to describe the images.",
}

# Chat Messages
CHAT_MESSAGES = {
    "INITIAL_GREETING": "Hello! I'm your MemoryBot assistant. How can I help you with your photos today?",
    "DEFAULT_RESPONSE": "I don't have specific information about that, but I'm here to help you find and manage your photos. Would you like to search your gallery?",
    "SEARCH_ERROR": "I couldn't understand how to search for that.",
}

# Gallery Messages
GALLERY_MESSAGES = {
    "GALLERY_ERROR": "I'm having trouble analyzing your gallery right now.",
    "UNKNOWN_LOCATION": "Unknown",
    "UNKNOWN_EVENT": "Unknown",
    "UNKNOWN_PERSON": "Unknown",
    "UNKNOWN_YEAR": "Unknown",
}

# Logging Messages
LOG_MESSAGES = {
    "CLIP_INIT": "CLIP model loaded on {}",
    "GEMINI_INIT": "Gemini service initialized successfully",
    "GEMINI_INIT_ERROR": "Failed to initialize Gemini service: {}",
    "QDRANT_INIT": "Created collection: {}",
    "QDRANT_EXISTS": "Collection {} already exists",
    "CACHE_LOADED": "Loaded {} cached Gemini responses",
    "CACHE_ERROR": "Error loading cache: {}",
    "GALLERY_SYNC": "Found {} images in directory",
    "NEW_IMAGES": "Found {} new images to process",
    "IMAGE_PROCESSED": "Successfully saved processed image to {}",
    "IMAGE_PROCESS_ERROR": "Failed to process and save image",
    "ERROR_ANALYZING_IMAGE": "Error analyzing image: {}",
    "ERROR_GENERATING_DESCRIPTION": "Error generating description: {}",
    "ERROR_GENERATING_KEYWORDS": "Error generating keywords: {}",
    "ERROR_GENERATING_CONTENT": "Error generating content: {}",
    "ERROR_EXTRACTING_JSON": "Error extracting JSON from response: {}",
    "ERROR_ANALYZING_IMAGE_GEMINI": "Error analyzing image with Gemini: {}",
    "UNABLE_TO_ANALYZE": "I'm unable to analyze this image at the moment.",
    "GENERATE_ERROR_RESPONSE": "I apologize, but I encountered an error generating a response.",
}

# Prompts
PROMPTS = {
    "IMAGE_DESCRIPTION": """
    Analyze this image and provide a natural, detailed description of what you see.
    Focus on:
    - The main subjects/people
    - The setting/location
    - The activity or event
    - Notable details or emotions
    Keep the description concise but informative (2-3 sentences).
    """,
    
    "IMAGE_KEYWORDS": """
    Analyze this image and generate relevant keywords/tags.
    Include:
    - Objects/subjects
    - Actions/activities
    - Emotions/mood
    - Setting/location
    - Style/composition
    Return only the keywords as a comma-separated list.
    """,
}

# Add this to your constants.py file
UPLOAD_SETTINGS = {
    "DELAY_SECONDS": 6,  # Delay between uploads in seconds
    "MAX_FILE_SIZE": 10 * 1024 * 1024,  # 10MB in bytes
    "ALLOWED_EXTENSIONS": {'jpg', 'jpeg', 'png', 'gif', 'webp'}
} 