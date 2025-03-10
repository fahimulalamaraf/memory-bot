from typing import Dict, Any, List, Optional
from memorybot.services.gemini_service import GeminiService
from memorybot.services.qdrant_service import QdrantService
from memorybot.core.database import get_qdrant_client
import json
import logging
from PIL import Image
import base64
from io import BytesIO
import os
from memorybot.services.clip_service import ClipService
from memorybot.services.image_processor import ImageProcessor
from pathlib import Path
import uuid
import tempfile

logger = logging.getLogger(__name__)

class ChatService:
    _instance = None
    CONTEXT_WINDOW_SIZE = 10  # Define constant for context window
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatService, cls).__new__(cls)
            cls._instance.gemini = GeminiService()
            cls._instance.qdrant = QdrantService(get_qdrant_client())
            cls._instance.clip_service = ClipService()
            cls._instance.conversations = {}
            cls._instance.image_processor = ImageProcessor()
        return cls._instance

    def __init__(self):
        """Initialize instance attributes"""
        # Ensure all required attributes are initialized
        if not hasattr(self, 'gemini'):
            self.gemini = GeminiService()
        if not hasattr(self, 'qdrant'):
            self.qdrant = QdrantService(get_qdrant_client())
        if not hasattr(self, 'clip_service'):
            self.clip_service = ClipService()
        if not hasattr(self, 'conversations'):
            self.conversations = {}
        if not hasattr(self, 'image_processor'):
            self.image_processor = ImageProcessor()

    async def handle_message(self, message: str, session_id: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Handle image-only requests
            if not message.strip() and image_data:
                message = "Please find similar images to this one"
            
            # Initialize or get conversation history
            if session_id not in self.conversations:
                self.conversations[session_id] = []
            
            conversation = self.conversations[session_id]
            conversation.append({"role": "user", "content": message})

            # Use consistent window size
            self.conversations[session_id] = conversation[-self.CONTEXT_WINDOW_SIZE:]

            # Get intent with conversation context
            intent = await self._determine_intent(
                message, 
                image_data,
                conversation_history=conversation
            )

            results = []
            
            if intent.get("type") == "function_call":
                function_name = intent.get("function")
                parameters = intent.get("parameters", {})
                
                logger.info(f"Executing function: {function_name} with parameters: {parameters}")
                
                if function_name == "handle_gallery_query":
                    response = await self._handle_gallery_query(
                        parameters["query_type"],
                        parameters["query"]
                    )
                    conversation.append({"role": "assistant", "content": response["text"]})
                    return {
                        "text": response["text"],
                        "type": "chat_response",
                        "results": []  # Maintain consistent response format
                    }

                if function_name == "search_by_image" and image_data:
                    # Save and process the image
                    image_path = await self._save_image_data(image_data)
                    if image_path:
                        results = await self.qdrant.search_similar_images(image_path)
                        if not results or not results.get("results"):
                            return {
                                "text": "I couldn't find any similar images.",
                                "type": "chat_response",
                                "results": []
                            }
                    else:
                        return {
                            "text": "I had trouble processing the image. Please try again.",
                            "type": "error",
                            "results": []
                        }
                
                elif function_name == "contextual_search" and image_data:
                    # Save and process the image
                    image_path = await self._save_image_data(image_data)
                    if image_path:
                        results = await self.qdrant.contextual_search(
                            image_path=image_path,
                            query=message
                        )
                        if not results:
                            return {
                                "text": "I couldn't find any relevant images based on your query.",
                                "type": "chat_response",
                                "results": []
                            }
                    else:
                        return {
                            "text": "I had trouble processing the image. Please try again.",
                            "type": "error",
                            "results": []
                        }
                
                elif function_name == "search_by_metadata":
                    cleaned_params = self._validate_metadata_parameters(parameters)
                    logger.info(f"Cleaned metadata parameters: {cleaned_params}")
                    results = await self.qdrant.search_by_metadata(cleaned_params)
                    
                # Inside the handle_message method of ChatService

                elif function_name == "search_by_text":
                    # Extract negations from the parameters
                    query = parameters.get("query", message)
                    negations = parameters.get("negations", [])

                    # Call search_by_text with the query and negations
                    results = await self.qdrant.search_by_text(query, negations)
                

                elif function_name == "handle_gallery_query":
                    results = await self.handle_gallery_query(parameters["query_type"], parameters["query"])

                
                
                elif function_name == "describe_images":
                    try:
                        if parameters.get("describe_last_results"):
                            # Get the last message with results from conversation history
                            last_message_with_results = next(
                                (msg for msg in reversed(conversation) 
                                 if msg["role"] == "assistant" and "results" in msg and msg.get("results")),
                                None
                            )
                            
                            if last_message_with_results and last_message_with_results.get("results"):
                                results = last_message_with_results["results"]
                                
                                # Create conversation history context
                                conversation_context = "\n".join([
                                    f"{msg['role']}: {msg['content']}"
                                    for msg in self.conversations[session_id][-5:]
                                ])
                                
                                # Create a detailed prompt with conversation context
                                prompt = f"""
                                Based on our conversation, analyze these images in detail:

                                Images to describe:
                                {json.dumps([{
                                    "filename": r["filename"],
                                    "metadata": r.get("metadata", {}),
                                    "description": r.get("metadata", {}).get("description", ""),
                                    "image_path": f"/images/{r['filename']}"
                                } for r in results], indent=2)}

                                Provide a detailed, engaging description that:
                                1. References any relevant context from our conversation
                                2. Describes the main subject of each image
                                3. Points out unique features and details
                                4. Describes the mood and atmosphere
                                5. Mentions any patterns or similarities between images
                                6. Uses natural, conversational language
                                7. Relates the images to any previous topics we discussed

                                Format as a friendly conversation, avoiding technical terms.
                                """
                                
                                # Use the new function for generating content with context
                                response_text = await self.gemini.generate_content_describe(
                                    prompt,
                                    conversation_history=self.conversations[session_id]
                                )
                                
                                # Store the response in conversation history
                                self.conversations[session_id].append({
                                    "role": "assistant",
                                    "content": response_text,
                                    "results": results
                                })
                                
                                return {
                                    "text": response_text,
                                    "type": "chat_response",
                                    "results": results
                                }
                            else:
                                return {
                                    "text": "I don't see any recent images to describe. Could you show me which images you'd like me to describe?",
                                    "type": "error",
                                    "results": []
                                }
                    except Exception as e:
                        logger.error(f"Error describing images: {e}")
                        return {
                            "text": "I encountered an error while trying to describe the images.",
                            "type": "error",
                            "results": []
                        }
                

                
                # Generate response with results
                response = await self._generate_response_with_results(
                    message, results, function_name
                )

            elif intent.get("type") == "greeting":
                response = {
                    "text": intent.get("content", "Hello! How can I help you with your photos today?"),
                    "type": "chat_response",
                    "results": []
                }
                
                
            elif intent.get("type") == "conversation":
                response = {
                    "text": intent.get("content", "I don't have specific information about that, but I'm here to help you find and manage your photos. Would you like to search your gallery?"),
                    "type": "chat_response",
                    "results": []
                }
            else:
                response = {
                    "text": intent.get("content", "I couldn't understand how to search for that."),
                    "type": "chat_response",
                    "results": []
                }
            
            # Before sending the response, store it with all details including results
            conversation.append({
                "role": "assistant", 
                "content": response["text"],
                "results": response.get("results", []),  # Store the results with the message
                "type": response["type"]
            })
            self.conversations[session_id] = conversation[-self.CONTEXT_WINDOW_SIZE:]  # Keep last 10 messages

            # After getting the response, store it in conversation history
            if "text" in response:
                self.conversations[session_id].append({
                    "role": "assistant",
                    "content": response["text"]
                })
            
            return response

        except Exception as e:
            logger.error(f"Error in handle_message: {e}")
            return {
                "text": "I encountered an error processing your request.",
                "type": "error",
                "results": []
            }
            

    async def _determine_intent(
        self, 
        message: str, 
        image_data: Optional[str] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Determine the intent of a user message and return appropriate response format.
        Returns Dict with format:
        {
            "type": str ("greeting"|"function_call"|"conversation"),
            "function": str (optional),
            "parameters": Dict (optional),
            "content": str (optional),
            "requires_search": bool
        }
        """
        try:
            # Create a prompt for Gemini to classify the query
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in conversation_history[-self.CONTEXT_WINDOW_SIZE:]
                ])

            prompt = f"""
            SYSTEM: You are an AI assistant that analyzes user messages and determines their intent.
            You need to analyze the user's message and determine the best way to search for the images.
            You need to understand the user's intent and return the appropriate function call.
            You are a photo gallery assistant and you need to help the user find the images they are looking for.
            

            You must return ONLY a JSON response matching one of the templates below.

            Context:
            Previous Messages: {history_text}
            Current Message: "{message}"
            Image Present: {"Yes" if image_data else "No"}

            ANALYSIS RULES (Check in order):
            
            1. GREETING
            IF message is ONLY a greeting (e.g., "hi", "hello", "how are you")
            THEN RETURN:
            {{
                "type": "greeting",
                "function": "respond_to_greeting",
                "content": "Hello! I'm your MemoryBot, your personalized photo-gallery assistant. How can I help you with your photos today?",
                "requires_search": false
            }}
            
            2. METADATA SEARCH
            IF message contains ANY of the following specific metadata:
            - **Person names**: Look for specific names (e.g., "John", "Sarah", "Ratul"). Avoid generic terms like "people" or "someone".
            - **Place names**: Identify specific locations (e.g., "Paris", "New York", "Dhaka"). Avoid generic terms like "outdoors" or "inside".
            - **Dates/Years**: Recognize specific dates or years (e.g., "2023", "last summer", "January"). Avoid vague time references.
            - **Events**: Detect specific events (e.g., "wedding", "birthday", "graduation"). Avoid generic activities.

            THEN RETURN:
            {{
                "type": "function_call",
                "function": "search_by_metadata",
                "parameters": {{
                    "who": ["<exact names mentioned>"],
                    "place": ["<exact places mentioned>"],
                    "year": ["<exact years/dates mentioned>"],
                    "event": ["<exact events mentioned>"]
                }},
                "requires_search": true
            }}

            3. IMAGE SIMILARITY SEARCH
            IF message indicates wanting similar images AND image is provided
            THEN RETURN:
            {{
                "type": "function_call",
                "function": "search_by_image",
                "parameters": {{}},
                "requires_search": true
            }}

            4. TEXT SEARCH
            IF the message contains a search intent but does NOT include specific metadata (like names, places, dates, or events), AND it includes negations (e.g., "except", "no", "not") to exclude certain terms or objects,
            THEN RETURN:
            {{
                "type": "function_call",
                "function": "search_by_text",
                "parameters": {{
                    "query": "<original message with negated terms removed>",
                    "negations": ["<list of terms to exclude>"]
                }},
                "requires_search": true
            }}

            IMPORTANT:
            - Identify negation words such as "except", "no", "not", and extract the terms following these words.
            - Remove these negated terms from the main query to ensure they are not included in the search.
            - Example: For the query "Show me animal images except lions", the query should be "Show me animal images" and negations should include "lions".
            - Ensure the final query is clear and concise, focusing only on the desired search terms.

            5. IMAGE DESCRIPTION REQUEST
            IF message asks about:
            - Description of an image
            - Details about a specific image
            - Questions about image content
            - "What do you see in this image?"
            - "Tell me about this image"
            - "Describe this image"
            - "Describe these images"
            - "Tell me about these images"
            - "What do you see in these images"
            OR if the message refers to previously shown images

            THEN RETURN:
            {{
                "type": "function_call",
                "function": "describe_images",
                "parameters": {{
                    "describe_last_results": true
                }},
                "requires_search": false
            }}

            6. CONTEXTUAL SEARCH
            IF message combines image understanding with a text query
            If the user searches for images based on a description of an image, then we will use the contextual search
            If the user sends both an image and a query, then we will use the contextual search
            If the user want sends a query and an image and the query is about the image, then we will use the contextual search
            THEN RETURN:
            {{
                "type": "function_call",
                "function": "contextual_search",
                "parameters": {{
                    "query": "<original message>"
                }},
                "requires_search": true
            }}

            7. GALLERY OVERVIEW
            IF message asks for an overview of the gallery or user preferences
            IF the user asks for a summary of the gallery or user preferences
            IF the user asks for a list of images in the gallery
            If the user asks the number of images in the gallery
            If the user asks for favorite anything about the gallery or the user
            THEN RETURN:
            {{
                "type": "function_call",
                "function": "handle_gallery_query",
                "parameters": {{
                    "query_type": "overview",
                    "query": "<original message>"
                }},
                "requires_search": false
            }}

            8. CONVERSATION - Lowest Priority
            IF none of the above match or the user is asking a general knowledge question
            THEN RETURN:
            {{
                "type": "conversation",
                "content": "<provide a helpful, accurate response to the user's question based on your knowledge>\\n\\nI'm primarily your photo gallery assistant - if you'd like help finding or organizing your images, just let me know!",
                "requires_search": false
            }}

            For general knowledge questions, provide a complete, accurate answer first, then add a gentle reminder about being a gallery assistant.
            For questions that might be sensitive or controversial, provide a balanced, factual response without political bias.
            Remind the user that you are a photo gallery assistant and you are here to help them find the images they are looking for.
            Always be helpful and informative while maintaining your primary identity as a gallery assistant.

            IMPORTANT:
            - Check rules in exact order (1 -> 8)
            - For metadata search, extract EXACT mentions (don't infer or expand)
            - Return ONLY the JSON object, no explanation
            - For image descriptions, always use describe_image function
            - When user asks about a specific image, use describe_image, not conversation
            """

            # Get response from Gemini
            response = await self.gemini.generate_content(prompt)
            
            # Parse the response
            intent_data = json.loads(response)
            return intent_data

        except Exception as e:
            logger.error(f"Error determining intent: {e}")
            return {
                "type": "error",
                "error": str(e)
            }


    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a specific session"""
        return self.conversations.get(session_id, [])



    async def _generate_response_with_results(self, message: str, results: List[Dict[str, Any]], function_name: str) -> Dict[str, Any]:
        """Generate a response incorporating search results using RAG approach"""
        try:
            # Format results for Gemini
            formatted_results = []
            # Check if results is a list or dict
            result_list = results if isinstance(results, list) else results.get("results", [])
            
            for result in result_list[:5]:  # Limit to top 5 results
                # Handle both direct dictionary and payload objects
                if hasattr(result, 'payload'):
                    metadata = result.payload
                else:
                    metadata = result.get("metadata", {})
                    
                formatted_result = {
                    "filename": result.get("filename") if isinstance(result, dict) else result.payload.get("filename"),
                    "description": metadata.get("description", ""),
                    "who": metadata.get("who", "Unknown"),
                    "place": metadata.get("place", "Unknown"),
                    "event": metadata.get("event", "Unknown"),
                    "year": metadata.get("year", "Unknown")
                }
                formatted_results.append(formatted_result)

            # Create a RAG-style prompt
            prompt = f"""
            User Query: {message}

            Retrieved Images: {json.dumps(formatted_results, indent=2)}

            Generate a natural, conversational response that:
            1. Directly answers the user's query
            2. Uses a friendly, helpful tone
            3. Gives a general summary of the retrieved images
            4. Makes it feel personal and like you are talking to a friend

            Do not include any JSON formatting in the response.
            Format the response as natural conversation.
            """

            # Get response from Gemini
            response_text = await self.gemini.generate_content(prompt)

            # Return formatted response with both text and results
            return {
                "text": response_text,
                "type": "chat_response",
                "results": result_list  # Use the original result list for frontend display
            }

        except Exception as e:
            logger.error(f"Error generating response with results: {e}")
            return {
                "text": "I found some relevant images based on your query.",
                "type": "chat_response",
                "results": results if isinstance(results, list) else []
            }

    def _validate_metadata_parameters(self, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate and format metadata parameters"""
        valid_fields = {"who", "place", "year", "event"}
        cleaned_params = {}
        
        for field in valid_fields:
            value = parameters.get(field, [])
            if isinstance(value, str):
                value = [value] if value else []
            elif not isinstance(value, list):
                value = []
            cleaned_params[field] = [str(v).lower() for v in value if v]
        
        return cleaned_params

    async def _save_image_data(self, image_data: str) -> Optional[str]:
        """Save image data to temporary file with processing"""
        try:
            if not image_data:
                return None
            
            # Create temp directory if it doesn't exist
            temp_dir = tempfile.gettempdir()
            
            try:
                # Extract image bytes from base64
                if 'base64,' in image_data:
                    image_bytes = base64.b64decode(image_data.split('base64,')[1])
                elif ',' in image_data:
                    image_bytes = base64.b64decode(image_data.split(',')[1])
                else:
                    image_bytes = base64.b64decode(image_data)
                
                # Generate temporary file path
                temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
                
                # Process and save image using ImageProcessor
                if self.image_processor.save_image(image_bytes, temp_file):
                    logger.info(f"Successfully saved processed image to {temp_file}")
                    return temp_file
                else:
                    logger.error("Failed to process and save image")
                    return None
                
            except Exception as e:
                logger.error(f"Error saving image data: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error in _save_image_data: {e}")
            return None

    async def _handle_gallery_query(self, query_type: str, query: str) -> Dict[str, Any]:
        """Handle gallery-related queries using the stored metadata"""
        try:
            # Get gallery content analysis
            gallery_content = await self.qdrant.analyze_gallery_content()
            
            # Create a prompt based on the query type
            prompt = f"""
            Based on this gallery content analysis, {query}

            Gallery Overview:
            - Total Images: {len(gallery_content["descriptions"])}
            - Keywords: {", ".join(list(gallery_content["keywords"])[:20])}
            
            Content Summary:
            {" ".join(gallery_content["descriptions"][:])}
            
            Metadata Statistics:
            - Places: {gallery_content["metadata"]["places"]}
            - Events: {gallery_content["metadata"]["events"]}
            - People: {gallery_content["metadata"]["people"]}
            
            Provide a natural, conversational response that:
            1. Directly answers the user's question
            2. Uses specific examples from the gallery
            3. Provides relevant statistics or patterns
            4. Maintains a friendly, personal tone
            """
            
            # Get response from Gemini
            response = await self.gemini.generate_content(prompt)
            
            return {
                "text": response,
                "type": "chat_response",
                "metadata": gallery_content["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Error handling gallery query: {e}")
            return {
                "text": "I'm having trouble analyzing your gallery right now.",
                "type": "error"
            }