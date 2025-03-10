from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from memorybot.core.config import settings
from memorybot.services.clip_service import ClipService
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid
from pathlib import Path
import os
from memorybot.services.gemini_service import GeminiService
from qdrant_client.http import models
from memorybot.core.cache import SearchCache

logger = logging.getLogger(__name__)

class QdrantService:
    _instance = None
    _initialized = False

    def __new__(cls, client: QdrantClient):
        if cls._instance is None:
            cls._instance = super(QdrantService, cls).__new__(cls)
            cls._instance.client = client
            cls._instance._clip_service = None
            cls._instance._gemini_service = None
            cls._instance.collection_name = "image_embeddings"
            cls._instance._ensure_collection_exists()
            cls._instance._processed_files = set()  # Track processed files
            cls._instance.cache = SearchCache()  # Initialize cache
            cls._instance.cache_ttl = 300  # Cache TTL in seconds (5 minutes)
        return cls._instance

    def __init__(self, client: QdrantClient):
        # Initialize is handled in __new__
        pass

    @property
    def clip_service(self):
        """Lazy load CLIP service only when needed"""
        if self._clip_service is None:
            self._clip_service = ClipService()
        return self._clip_service

    @property
    def gemini_service(self):
        """Lazy load Gemini service only when needed"""
        if self._gemini_service is None:
            self._gemini_service = GeminiService()
        return self._gemini_service

    """ From Here the Vector Database Functions Start """

    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper schema"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def sync_images_with_db(self):
        """Sync images from directory with vector database"""
        try:
            # Get all images from directory
            image_dir = Path("images")
            image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
            
            logger.info(f"Found {len(image_files)} images in directory")
            
            # Get existing images from Qdrant
            existing = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000
            )[0]
            existing_paths = {str(Path(p.payload.get('path', '')).absolute()) for p in existing if p.payload}
            self._processed_files.update(existing_paths)
            
            # Find new images
            new_images = [f for f in image_files if str(f.absolute()) not in self._processed_files]
            logger.info(f"Found {len(new_images)} new images to process")

            # Store new images
            success_count = 0
            for image_file in new_images:
                try:
                    success = await self.store_image_with_metadata(
                        str(image_file.absolute()),
                        metadata=None
                    )
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {e}")

            # Invalidate cache if new images were added
            if success_count > 0:
                self.invalidate_gallery_cache()
            
            return success_count

        except Exception as e:
            logger.error(f"Error syncing images: {e}")
            return 0

    async def store_image_with_metadata(
        self, 
        image_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store image with its embedding and metadata"""
        try:
            # Check if file was already processed
            abs_path = str(Path(image_path).absolute())
            if abs_path in self._processed_files:
                logger.info(f"Image already processed: {image_path}")
                return True

            # Generate embedding
            embedding = self.clip_service.get_image_embedding(image_path)
            
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
            
            # Generate image analysis
            analysis = self.gemini_service.store_analyze_image(image_path)
            
            # Get original filename without any _resized suffix
            original_filename = Path(image_path).name
            if "_resized" in original_filename:
                original_filename = original_filename.replace("_resized", "")
            
            # Prepare metadata with defaults and analysis
            image_data = {
                "id": str(uuid.uuid4()),
                "filename": original_filename,  # Store original filename
                "path": str(abs_path),
                "who": metadata.get('who', 'Unknown'),
                "place": metadata.get('place', 'Unknown'),
                "year": metadata.get('year', 'Unknown'),
                "event": metadata.get('event', 'Unknown'),
                "timestamp": datetime.now().isoformat(),
                "description": metadata.get('description') or analysis.get('description', ''),
                "keywords": analysis.get('keywords', []),
                "analysis": {
                    **analysis,
                    "content_summary": analysis.get('description', ''),
                    "detected_objects": analysis.get('objects', []),
                    "detected_activities": analysis.get('activities', []),
                    "mood": analysis.get('mood', ''),
                    "setting": analysis.get('setting', '')
                }
            }
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=image_data["id"],
                    vector=embedding.tolist(),
                    payload=image_data
                )]
            )
            
            # Mark file as processed
            self._processed_files.add(abs_path)
            return True
            
        except Exception as e:
            logger.error(f"Error storing image: {e}")
            return False

    async def get_all_images(self) -> List[Dict[str, Any]]:
        """Get all images from the collection with caching"""
        try:
            # Check cache first
            cache_key = "all_gallery_images"
            cached_images = self.cache.get(cache_key)
            
            if cached_images:
                logger.info("Returning cached gallery images")
                return cached_images
                
            # If not cached, fetch from database
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )[0]
            
            # Filter out images that don't exist on disk and deduplicate
            seen_filenames = set()
            valid_images = []
            for result in results:
                filename = result.payload.get("filename")
                # Check for both original and resized versions
                original_path = os.path.join("images", filename)
                if filename and filename not in seen_filenames and os.path.exists(original_path):
                    seen_filenames.add(filename)
                    valid_images.append(result.payload)
                else:
                    logger.debug(f"Checking image path: {original_path}")
            
            # Sort images by timestamp (newest first)
            valid_images.sort(
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            # Cache the results - remove ttl parameter
            self.cache.set(cache_key, valid_images)
            logger.info(f"Cached {len(valid_images)} gallery images")
            
            return valid_images
        except Exception as e:
            logger.error(f"Error getting images: {e}")
            raise


    def _format_result(self, result) -> Dict[str, Any]:
        """Format a single result"""
        try:
            return {
                "filename": result.payload.get("filename", ""),
                "metadata": {
                    "description": result.payload.get("description", ""),
                    "place": result.payload.get("place", "Unknown"),
                    "event": result.payload.get("event", "Unknown"),
                    "who": result.payload.get("who", "Unknown"),
                    "year": result.payload.get("year", "Unknown")
                }
            }
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
            return None

    """ From Here the Search Functions Start """


    async def search_by_text(self, query: str, negations: List[str] = [], max_limit: int = 10) -> Dict[str, Any]:
        """Search images using text query with dynamic result handling and keyword enhancement"""
        try:
            # Process negations
            processed_query = query
            for negation in negations:
                processed_query = processed_query.replace(negation, "").strip()
                
            # Clean up any double spaces created by removing terms
            processed_query = " ".join(processed_query.split())
            
            logger.info(f"Original query: {query}")
            logger.info(f"Processed query after removing negations: {processed_query}")
            logger.info(f"Negated terms: {negations}")

            # Extract keywords from the query for direct matching
            # Simple tokenization - split on spaces and remove common stop words
            stop_words = {"a", "an", "the", "in", "on", "at", "with", "by", "and", "or", "for", "to", "of", "Show", "show", "images", "image", "I", "i", "me", "my", "mine", "myself", "your", "yours", "yourself", "yourselves", "he", "she", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "we", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves"}
            query_keywords = [word.lower() for word in processed_query.split() 
                             if word.lower() not in stop_words and len(word) > 2]
            
            logger.info(f"Extracted query keywords for matching: {query_keywords}")

            # Generate text embedding for the processed query
            text_embedding = self.clip_service.get_text_embedding(processed_query)
            
            # Search in Qdrant with higher limit to get more candidates
            vector_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=text_embedding.tolist(),
                limit=max_limit * 2,  # Fetch more to re-rank and filter later
                with_payload=True,
                score_threshold=0.25  # Lower threshold for more recall
            )
            
            # Re-rank results based on combination of vector similarity and keyword matches
            enriched_results = []
            
            for result in vector_results:
                if not result or not hasattr(result, 'payload'):
                    continue
                    
                # Skip results matching negations
                should_exclude = False
                for negation in negations:
                    description = result.payload.get("description", "").lower()
                    keywords = " ".join(result.payload.get("keywords", [])).lower()
                    if (negation.lower() in description or negation.lower() in keywords):
                        should_exclude = True
                        break
                        
                if should_exclude:
                    continue
                    
                # Calculate keyword match score
                keyword_score = 0
                result_keywords = [k.lower() for k in result.payload.get("keywords", [])]
                result_description = result.payload.get("description", "").lower()
                
                # Score exact keyword matches
                for keyword in query_keywords:
                    # Check for exact matches in keywords
                    if keyword in result_keywords:
                        keyword_score += 0.15  # Strong bonus for exact keyword match
                        
                    # Check for partial matches in keywords
                    partial_matches = sum(1 for k in result_keywords if keyword in k)
                    keyword_score += 0.05 * partial_matches
                    
                    # Check for matches in description
                    if keyword in result_description:
                        keyword_score += 0.10
                
                # Combine vector similarity with keyword matching score
                # Base score is the vector similarity (scaled to 0-1)
                combined_score = result.score + keyword_score
                
                # Create enriched result with combined score
                enriched_result = self._format_result(result)
                if enriched_result:
                    enriched_result["score"] = combined_score
                    enriched_results.append(enriched_result)
            
            # Sort by combined score (highest first)
            enriched_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Apply limit after sorting by combined score
            total_relevant = len(enriched_results)
            limited_results = enriched_results[:max_limit]
            
            # Prepare response
            response = {
                "results": limited_results,
                "total": total_relevant,
                "has_more": total_relevant > max_limit,
                "message": (
                    f"Found {total_relevant} relevant images. "
                    f"Showing {len(limited_results)} of {total_relevant}."
                    if total_relevant > max_limit else None
                )
            }
            
            return response

        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return {"results": [], "total": 0, "has_more": False, "message": None}

            
    async def search_similar_images(self, image_path: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar images using CLIP embeddings"""
        try:
            # Generate embedding for the query image
            image_embedding = self.clip_service.get_image_embedding(image_path)
            
            # Search in Qdrant with higher limit to get more candidates
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=image_embedding.tolist(),
                limit=limit * 2,  # Fetch more to filter later
                with_payload=True,
                score_threshold=0.55  # Minimum similarity threshold
            )
            
            # Format results and filter by relevance
            formatted_results = []
            total_relevant = 0
            for result in results:
                if result.payload and result.score >= 0.6:  # Higher threshold for actual inclusion
                    total_relevant += 1
                    formatted_results.append({
                        "filename": result.payload.get("filename", ""),
                        "metadata": {
                            "description": result.payload.get("description", ""),
                            "place": result.payload.get("place", "Unknown"),
                            "event": result.payload.get("event", "Unknown"),
                            "who": result.payload.get("who", "Unknown"),
                            "year": result.payload.get("year", "Unknown")
                        },
                        "score": float(result.score) if hasattr(result, 'score') else 0.0
                    })
            
            # Prepare response similar to search_by_text
            response = {
                "results": formatted_results,
                "total": total_relevant,
                "has_more": total_relevant > limit,
                "message": (
                    f"Found {total_relevant} similar images. "
                    f"Showing {len(formatted_results)} most relevant matches."
                    if total_relevant > limit else None
                )
            }
            
            return response

        except Exception as e:
            logger.error(f"Error in similar image search: {e}")
            return {"results": [], "total": 0, "has_more": False, "message": None}


    async def get_image_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get image data by filename"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )[0]
            
            if not results:
                return None
            
            # Get the raw payload
            payload = results[0].payload
            
            # Return formatted data
            return {
                "filename": payload.get("filename", ""),
                "place": payload.get("place", "Unknown"),
                "event": payload.get("event", "Unknown"),
                "who": payload.get("who", "Unknown"),
                "year": payload.get("year", "Unknown"),
                "analysis": {  # Keep analysis for description and keywords
                    "description": payload.get("analysis", {}).get("description", ""),
                    "keywords": payload.get("analysis", {}).get("keywords", [])
                }
            }
        except Exception as e:
            logger.error(f"Error getting image by filename: {e}")
            raise


    async def search_by_metadata(self, parameters: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Search images by metadata with extracted parameters
        Handles: who, place, year, and event parameters
        Supports multiple parameters and combinations
        """
        try:
            should = []
            
            # Handle who parameter (people)
            if parameters.get("who"):
                for person in parameters["who"]:
                    should.append(
                        models.FieldCondition(
                            key="who",
                            match=models.MatchText(
                                text=person.lower()
                            )
                        )
                    )

            # Handle place parameter
            if parameters.get("place"):
                for place in parameters["place"]:
                    should.append(
                        models.FieldCondition(
                            key="place",
                            match=models.MatchText(
                                text=place.lower()
                            )
                        )
                    )

            # Handle year parameter
            if parameters.get("year"):
                for year in parameters["year"]:
                    should.append(
                        models.FieldCondition(
                            key="year",
                            match=models.MatchText(
                                text=str(year).lower()
                            )
                        )
                    )

            # Handle event parameter
            if parameters.get("event"):
                for event in parameters["event"]:
                    should.append(
                        models.FieldCondition(
                            key="event",
                            match=models.MatchText(
                                text=event.lower()
                            )
                        )
                    )

            # Execute search with combined conditions
            if should:
                # If we have multiple conditions, use must to ensure all conditions are met
                filter_conditions = models.Filter(
                    should=should if len(should) == 1 else None,
                    must=should if len(should) > 1 else None
                )

                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_conditions,
                    limit=20,
                    with_payload=True
                )[0]

                logger.info(f"Found {len(results)} metadata matches")
                
                # Enhanced result scoring for multiple parameters
                scored_results = []
                for result in results:
                    score = 0
                    payload = result.payload
                    
                    # Score matches for each parameter type
                    if parameters.get("who"):
                        who_text = payload.get("who", "").lower()
                        for person in parameters["who"]:
                            if person.lower() in who_text:
                                score += 10  # Higher weight for person matches
                    
                    if parameters.get("place"):
                        place_text = payload.get("place", "").lower()
                        for place in parameters["place"]:
                            if place.lower() in place_text:
                                score += 8  # Weight for place matches
                    
                    if parameters.get("year"):
                        year_text = str(payload.get("year", "")).lower()
                        for year in parameters["year"]:
                            if str(year).lower() in year_text:
                                score += 6  # Weight for year matches
                    
                    if parameters.get("event"):
                        event_text = payload.get("event", "").lower()
                        for event in parameters["event"]:
                            if event.lower() in event_text:
                                score += 7  # Weight for event matches
                    
                    # Format and add scored result
                    formatted_result = self._format_result(result)
                    if formatted_result:
                        formatted_result["score"] = score
                        scored_results.append(formatted_result)
                
                # Sort results by score (highest first)
                scored_results.sort(key=lambda x: x["score"], reverse=True)
                
                return scored_results

            return []

        except Exception as e:
            logger.error(f"Metadata search error: {str(e)}")
            return []


    async def contextual_search(self, image_path: str, query: str, max_limit: int = 5) -> Dict[str, Any]:
        """
        Perform contextual search combining image understanding and text query
        
        Args:
            image_path: Path to the query image
            query: User's text query
            max_limit: Maximum number of results to return (default: 5)
        
        Returns:
            Dictionary containing results and metadata
        """
        try:
            # First, analyze the image using Gemini with a specific prompt
            analysis_prompt = """
            Analyze this image and describe:
            1. The main subjects and objects
            2. The setting or environment
            3. Any notable features or characteristics
            4. The overall context and composition
            
            Provide a clear, detailed description that can be used for image comparison.
            """
            
            # Helper function to safely await coroutines
            async def safe_analyze_image():
                try:
                    # Get the analyze_image coroutine
                    analyze_coroutine = self.gemini_service.analyze_image(
                        image_path=image_path, 
                        prompt=analysis_prompt
                    )
                    
                    # Check if it's a coroutine and await it properly
                    if hasattr(analyze_coroutine, '__await__'):
                        return await analyze_coroutine
                    # If not a coroutine, return directly
                    return analyze_coroutine
                except Exception as e:
                    logger.error(f"Error in analyze_image: {e}")
                    return {"description": "Image analysis failed"}
            
            # Safely get image analysis
            image_analysis = await safe_analyze_image()
            
            # Safely extract description
            image_description = image_analysis.get("description", "") if isinstance(image_analysis, dict) else "Unknown image"
            
            # Create a prompt for understanding search context
            context_prompt = f"""
            Image Description: {image_description}
            User Query: {query}

            Based on the image content and user's query, create a search query that:
            1. Combines relevant aspects from both the image and query
            2. Handles negations (e.g., "find images like this but without people")
            3. Extracts key visual elements that should be matched
            4. Considers the user's specific requirements
            5. Maintains important context from the image

            Format: Return ONLY the refined search query text, no explanations.
            """
            
            # Helper function to safely generate content
            async def safe_generate_content():
                try:
                    # Get the generate_content coroutine
                    content_coroutine = self.gemini_service.generate_content(context_prompt)
                    
                    # Check if it's a coroutine and await it properly
                    if hasattr(content_coroutine, '__await__'):
                        return await content_coroutine
                    # If not a coroutine, return directly
                    return content_coroutine
                except Exception as e:
                    logger.error(f"Error in generate_content: {e}")
                    return "Find similar images"
            
            # Safely get refined query
            refined_query = await safe_generate_content()
            logger.info(f"Refined contextual query: {refined_query}")
            
            # Use CLIP to get embedding for refined query
            query_embedding = self.clip_service.get_text_embedding(refined_query)
            
            # Search using the refined query embedding with higher limit to filter later
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=max_limit * 2,  # Fetch more to filter later
                with_payload=True,
                score_threshold=0.25  # Lowered threshold for more permissive matching
            )
            
            # Filter and format results based on relevance
            formatted_results = []
            total_relevant = 0
            
            for result in results:
                if result and hasattr(result, 'score') and result.score >= 0.2:
                    total_relevant += 1
                    formatted_result = self._format_result(result)
                    if formatted_result:
                        formatted_results.append(formatted_result)
            
            # Prepare response similar to search_by_text
            response = {
                "results": formatted_results[:max_limit],  # Return only up to max_limit
                "total": total_relevant,
                "has_more": total_relevant > max_limit,
                "message": (
                    f"Found {total_relevant} relevant images based on your query. "
                    f"Showing {min(len(formatted_results), max_limit)} of {total_relevant}."
                    if total_relevant > max_limit else None
                )
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in contextual search: {e}")
            return {"results": [], "total": 0, "has_more": False, "message": None}
     

    async def analyze_gallery_content(self) -> Dict[str, Any]:
        """Analyze the content of the gallery and return metadata statistics"""
        try:
            # Retrieve all image metadata from the collection
            all_metadata, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )

            # Initialize counters and aggregators
            descriptions = []
            keywords = set()
            metadata_stats = {
                "places": {},
                "events": {},
                "people": {}
            }

            # Define key mapping for metadata fields
            key_mapping = {
                "place": "places",
                "event": "events",
                "who": "people"
            }

            # Process each metadata entry
            for item in all_metadata:
                if hasattr(item, 'payload') and item.payload:
                    payload = item.payload
                    descriptions.append(payload.get("description", ""))
                    keywords.update(payload.get("keywords", []))

                    # Aggregate metadata statistics using the mapping
                    for key, stat_key in key_mapping.items():
                        value = payload.get(key)
                        if value:
                            if isinstance(value, str):
                                values = [value]
                            elif isinstance(value, list):
                                values = value
                            else:
                                continue

                            for v in values:
                                if v not in metadata_stats[stat_key]:
                                    metadata_stats[stat_key][v] = 0
                                metadata_stats[stat_key][v] += 1

            return {
                "descriptions": descriptions,
                "keywords": list(keywords),
                "metadata": metadata_stats
            }

        except Exception as e:
            logger.error(f"Error analyzing gallery content: {e}")
            return {
                "descriptions": [],
                "keywords": [],
                "metadata": {
                    "places": {},
                    "events": {},
                    "people": {}
                }
            }

    def invalidate_gallery_cache(self):
        """Invalidate the gallery images cache"""
        try:
            self.cache.delete("all_gallery_images")
            logger.info("Gallery cache invalidated")
        except Exception as e:
            logger.error(f"Error invalidating gallery cache: {e}")

    async def delete_image(self, filename: str) -> bool:
        """Delete image from Qdrant and filesystem"""
        try:
            # Find the point ID by filename
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1
            )[0]

            if not results:
                logger.error(f"Image {filename} not found in database")
                return False

            point_id = results[0].id

            # Delete from Qdrant
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )

            # Delete file from filesystem
            file_path = settings.IMAGES_DIR / filename
            if file_path.exists():
                file_path.unlink()

            # Invalidate cache
            self.cache.delete("all_gallery_images")
            
            logger.info(f"Successfully deleted image {filename}")
            return True

        except Exception as e:
            logger.error(f"Error deleting image {filename}: {e}")
            return False
