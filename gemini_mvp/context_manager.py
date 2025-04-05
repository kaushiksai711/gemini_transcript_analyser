# gemini_mvp/context_manager.py - ACTIVELY USED
# Manages context selection and organization for queries

"""
Context Manager module for organizing and selecting context.

This module determines relevant context for queries, manages context
windows, and optimizes context usage.
"""

import time
import heapq
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from .caching import CacheManager
from .chunking import Chunk

logger = logging.getLogger(__name__)

class ContextItem:
    """Represents a question-answer pair in conversation history."""
    
    def __init__(
        self,
        question: str,
        answer: str,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a context item.
        
        Args:
            question: The user's question
            answer: The system's answer
            timestamp: When this exchange occurred
            metadata: Additional information about this exchange
        """
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary representation."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


class ContextManager:
    """
    Manages conversation context and relevant chunks.
    
    This class is responsible for:
    1. Maintaining conversation history
    2. Prioritizing chunks based on relevance to current query
    3. Implementing sliding context window
    4. Compressing history to reduce token usage
    """
    
    def __init__(self, config: Dict[str, Any], cache_manager: CacheManager):
        """
        Initialize the context manager.
        
        Args:
            config: Context manager configuration
            cache_manager: Cache manager instance
        """
        self.config = config
        self.cache = cache_manager
        self.conversation_history: List[ContextItem] = []
        self.last_update_time = time.time()
        
        # Initialize sentence transformer for semantic similarity
        self._load_embedding_model()
        
        # Current chunks being used for context
        self.current_context_chunks: List[Chunk] = []
    
    def _load_embedding_model(self):
        """Load the embedding model for semantic similarity."""
        logger.info("Loading embedding model")
        model_name = "all-MiniLM-L6-v2"  # Small, fast model good for semantic similarity
        try:
            # Check if model is in cache
            model_key = f"embedding_model_{model_name}"
            model = self.cache.get(model_key)
            if model is None:
                # Load and cache the model
                model = SentenceTransformer(model_name)
                self.cache.set(model_key, model)
            self.embedding_model = model
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to None - will use keyword-based matching instead
            self.embedding_model = None
    
    def get_context(
        self, 
        query: str, 
        query_info: Dict[str, Any],
        max_chunks: Optional[int] = None
    ) -> List[Chunk]:
        """
        Get the most relevant context chunks for a query.
        
        Args:
            query: The current query
            query_info: Information about the query
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of relevant chunks
        """
        if max_chunks is None:
            # Use max_context_chunks from config if available, otherwise default to 10
            max_chunks = self.config.get("max_context_chunks", 10)
        
        # Check if we have chunks from chunk manager
        from_chunk_manager = query_info.get("available_chunks", [])
        if not from_chunk_manager:
            logger.warning("No chunks available from chunk manager")
            return []
        
        # Analyze query complexity to determine how many chunks to include
        query_complexity = self._analyze_query_complexity(query)
        
        # For complex queries, include more chunks
        if query_complexity > 0.7:
            max_chunks = min(len(from_chunk_manager), max(max_chunks, 8))
        
        # For very complex queries, include even more chunks
        if query_complexity > 0.9:
            max_chunks = min(len(from_chunk_manager), max(max_chunks, 12))
            
        # Extract key concepts from query for diversity
        query_concepts = self._extract_query_concepts(query)
            
        # Score and rank chunks
        scored_chunks = self._score_chunks_for_query(query, from_chunk_manager, query_complexity)
        
        # Get initial top chunks
        top_chunks = [chunk for _, chunk in scored_chunks[:max_chunks]]
        
        # Check for concept coverage and diversity
        if query_concepts and len(from_chunk_manager) > max_chunks:
            # We have more chunks than we're using, check if we're missing important concepts
            used_chunks_set = set(top_chunks)
            candidate_chunks = [chunk for _, chunk in scored_chunks[max_chunks:max_chunks*2]]
            
            # Try to add chunks that cover missing concepts
            for concept in query_concepts:
                # Skip if we've already hit max chunks
                if len(top_chunks) >= max_chunks:
                    break
                    
                # Find chunks that contain this concept but aren't already included
                for chunk in candidate_chunks:
                    if chunk in used_chunks_set:
                        continue
                        
                    # Check if chunk contains this concept
                    if concept.lower() in chunk.text.lower():
                        top_chunks.append(chunk)
                        used_chunks_set.add(chunk)
                        break
        
        # Update current context chunks
        self.current_context_chunks = top_chunks
        
        return top_chunks
        
    def _analyze_query_complexity(self, query: str) -> float:
        """
        Analyze the complexity of a query to determine context needs.
        
        Args:
            query: The query to analyze
            
        Returns:
            Complexity score between 0 and 1
        """
        # Count words, question marks, commas
        word_count = len(query.split())
        question_count = query.count('?')
        comma_count = query.count(',')
        
        # Check for complexity indicators
        complexity_indicators = [
            "compare", "contrast", "analyze", "evaluate", "explain", 
            "detail", "comprehensive", "thorough", "elaborate", "discuss",
            "implications", "consequences", "benefits", "drawbacks", "challenges",
            "advantages", "disadvantages", "relationship", "connection", "difference"
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator.lower() in query.lower())
        
        # Multi-part questions (e.g., "X and Y" or "X, Y, Z")
        and_count = query.lower().count(" and ")
        or_count = query.lower().count(" or ")
        
        # Calculate base complexity score
        base_score = min(1.0, (
            (word_count / 30) * 0.5 +  # Longer questions are more complex
            (question_count / 2) * 0.15 +  # Multiple question marks indicate multiple questions
            (comma_count / 3) * 0.1 +  # Commas often separate parts of complex questions
            (indicator_count / 3) * 0.2 +  # Complexity indicators
            (and_count + or_count) * 0.15  # Multiple parts
        ))
        
        return base_score
        
    def _extract_query_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from a query for ensuring diverse context coverage.
        
        Args:
            query: The query to analyze
            
        Returns:
            List of key concepts
        """
        import re
        from collections import Counter
        
        # Remove common words and punctuation
        stop_words = set([
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "who", "whom", "whose", "where", "when", "why", "how", "is", 
            "are", "was", "were", "be", "been", "being", "have", "has", "had", 
            "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "in", "on", "at", "to", "for", "with", "about",
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "from", "up", "down", "of", "by", "this", "that"
        ])
        
        # Tokenize and clean
        tokens = re.findall(r'\b\w+\b', query.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
        
        # Count token frequencies
        token_counts = Counter(filtered_tokens)
        
        # Return most common concepts (max 5)
        return [concept for concept, _ in token_counts.most_common(5)]
    
    def _score_chunks_for_query(
        self, 
        query: str, 
        chunks: List[Chunk],
        query_complexity: float = None
    ) -> List[Tuple[float, Chunk]]:
        """
        Score chunks based on relevance to query.
        
        Args:
            query: The query text
            chunks: List of chunks to score
            query_complexity: Optional query complexity score (0-1)
            
        Returns:
            List of (score, chunk) tuples sorted by score
        """
        if not chunks:
            return []
            
        # Get current weights from config or use defaults
        weights = self.config.get("context_weights", {})
        recency_weight = weights.get("recency", 0.3)
        relevance_weight = weights.get("relevance", 0.5)
        entity_weight = weights.get("entity", 0.2)
        
        # Apply dynamic weight adjustments based on query complexity
        if query_complexity is not None:
            # For complex queries, prioritize entity matching and relevance
            if query_complexity > 0.7:
                entity_weight *= 1.2
                relevance_weight *= 1.1
                recency_weight *= 0.9
        
        # Extract entities from query for matching
        query_entities = self._extract_entities(query)
        
        # Calculate various scores for each chunk
        scored_chunks = []
        for chunk in chunks:
            # 1. Calculate semantic similarity score
            semantic_score = self._calculate_semantic_similarity(query, chunk.text)
            
            # 2. Calculate entity match score
            entity_score = self._calculate_entity_match(query_entities, chunk.entities)
            
            # 3. Calculate timestamp proximity score (if available)
            timestamp_score = self._calculate_timestamp_score(query, chunk)
            
            # 4. Calculate recency score (more recent chunks preferred)
            # This assumes chunks are in chronological order
            recency_score = 1.0 - (chunk.chunk_id / max(1, len(chunks)))
            
            # Calculate weighted final score
            final_score = (
                (relevance_weight * semantic_score) +
                (entity_weight * entity_score) +
                (recency_weight * recency_score)
            )
            
            # Add bonus for timestamp matches
            if timestamp_score > 0:
                final_score += 0.2  # Significant boost for timestamp matches
            
            scored_chunks.append((final_score, chunk))
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return scored_chunks
        
    def optimize_weights(self, training_data: List[Dict[str, Any]], iterations: int = 50) -> Dict[str, float]:
        """
        Tune context selection weights using training data.
        
        Args:
            training_data: List of training examples with queries and relevant chunks
            iterations: Number of iterations for optimization
            
        Returns:
            Optimized weights
        """
        import random
        
        logger.info(f"Optimizing context weights with {len(training_data)} examples")
        
        # Start with current weights or defaults
        weights = self.config.get("context_weights", {})
        best_weights = {
            "recency": weights.get("recency", 0.3),
            "relevance": weights.get("relevance", 0.5),
            "entity": weights.get("entity", 0.2)
        }
        
        # Evaluate current weights
        best_score = self._evaluate_weights(best_weights, training_data)
        logger.info(f"Initial score: {best_score:.4f} with weights {best_weights}")
        
        # Simple grid search with random restarts
        for i in range(iterations):
            # Randomly modify weights while keeping sum = 1.0
            if i < iterations / 2:
                # First half: random sampling
                w1 = random.uniform(0.1, 0.7)  # recency
                w2 = random.uniform(0.1, 0.7)  # relevance
                w3 = max(0.1, 1.0 - w1 - w2)   # entity
                
                # Normalize to sum to 1.0
                total = w1 + w2 + w3
                weights = {
                    "recency": w1 / total,
                    "relevance": w2 / total,
                    "entity": w3 / total
                }
            else:
                # Second half: local optimization
                noise = 0.05 * (1.0 - (i / iterations))  # Decreasing noise
                
                # Add noise to best weights
                w1 = max(0.1, min(0.7, best_weights["recency"] + random.uniform(-noise, noise)))
                w2 = max(0.1, min(0.7, best_weights["relevance"] + random.uniform(-noise, noise)))
                w3 = max(0.1, 1.0 - w1 - w2)
                
                # Normalize to sum to 1.0
                total = w1 + w2 + w3
                weights = {
                    "recency": w1 / total,
                    "relevance": w2 / total,
                    "entity": w3 / total
                }
            
            # Evaluate new weights
            score = self._evaluate_weights(weights, training_data)
            
            # Update best weights if better
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                logger.info(f"New best score: {best_score:.4f} with weights {best_weights}")
        
        # Update config with optimized weights
        self.config["context_weights"] = best_weights
        logger.info(f"Final optimized weights: {best_weights}")
        
        return best_weights
    
    def _evaluate_weights(self, weights: Dict[str, float], training_data: List[Dict[str, Any]]) -> float:
        """
        Evaluate a set of weights against training data.
        
        Args:
            weights: Weights dictionary
            training_data: Training examples
            
        Returns:
            Score (higher is better)
        """
        # Store original weights
        original_weights = self.config.get("context_weights", {}).copy()
        
        # Set new weights for evaluation
        self.config["context_weights"] = weights
        
        total_score = 0.0
        
        # Evaluate on each training example
        for example in training_data:
            query = example["query"]
            relevant_chunks = set(example["relevant_chunk_ids"])
            available_chunks = example["available_chunks"]
            
            # Score chunks with current weights
            scored_chunks = self._score_chunks_for_query(query, available_chunks)
            
            # Take top N chunks
            max_chunks = self.config.get("max_context_chunks", 10)
            top_chunks = [chunk for _, chunk in scored_chunks[:max_chunks]]
            top_chunk_ids = {chunk.chunk_id for chunk in top_chunks}
            
            # Calculate precision, recall, F1
            if not relevant_chunks or not top_chunk_ids:
                continue
                
            tp = len(relevant_chunks.intersection(top_chunk_ids))
            precision = tp / len(top_chunk_ids) if top_chunk_ids else 0
            recall = tp / len(relevant_chunks) if relevant_chunks else 0
            
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                total_score += f1
        
        # Restore original weights
        self.config["context_weights"] = original_weights
        
        # Return average F1 score
        return total_score / len(training_data) if training_data else 0.0
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from a query.
        
        Args:
            query: The query text
            
        Returns:
            List of entities
        """
        # For now, use a simple approach: extract capitalized words
        # In a production system, this would use a proper NER system
        words = query.split()
        entities = [
            word.strip(".,?!\"'()[]{}") 
            for word in words 
            if word and word[0].isupper()
        ]
        return entities
    
    def _calculate_semantic_similarity(self, query: str, text: str) -> float:
        """
        Calculate semantic similarity between query and text.
        
        Args:
            query: The query text
            text: The text to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use embedding model if available
        if self.embedding_model is not None:
            query_embedding = self.embedding_model.encode(query)
            text_embedding = self.embedding_model.encode(text)
            return self._cosine_similarity(query_embedding, text_embedding)
        else:
            # Fallback to keyword matching
            return self._keyword_similarity(query, text)
    
    def _calculate_entity_match(self, query_entities: List[str], chunk_entities: List[str]) -> float:
        """
        Calculate entity match score between query and chunk.
        
        Args:
            query_entities: Entities from query
            chunk_entities: Entities from chunk
            
        Returns:
            Match score between 0 and 1
        """
        if not query_entities or not chunk_entities:
            return 0.0
        
        # Calculate overlap
        overlap = len(set(query_entities).intersection(set(chunk_entities)))
        
        # Calculate match score
        match_score = overlap / max(len(query_entities), len(chunk_entities))
        
        return match_score
    
    def _calculate_timestamp_score(self, query: str, chunk: Chunk) -> float:
        """
        Calculate timestamp proximity score between query and chunk.
        
        Args:
            query: The query text
            chunk: The chunk to compare
            
        Returns:
            Proximity score between 0 and 1
        """
        # Extract timestamps from query
        query_timestamps = self._extract_timestamp_mentions(query)
        
        # Check if chunk has timestamp
        if not chunk.start_time:
            return 0.0
        
        # Calculate proximity score
        proximity_score = 0.0
        for query_ts in query_timestamps:
            # Convert timestamps to seconds for comparison
            from .utils import timestamp_to_seconds
            query_ts_seconds = timestamp_to_seconds(query_ts)
            chunk_start_seconds = timestamp_to_seconds(chunk.start_time)
            chunk_end_seconds = timestamp_to_seconds(chunk.end_time) if chunk.end_time else chunk_start_seconds + 60
            
            # Check if timestamp falls within chunk
            if chunk_start_seconds <= query_ts_seconds <= chunk_end_seconds:
                proximity_score = 1.0
                break
            else:
                # Calculate distance to closest chunk boundary
                distance_to_start = abs(query_ts_seconds - chunk_start_seconds)
                distance_to_end = abs(query_ts_seconds - chunk_end_seconds)
                min_distance = min(distance_to_start, distance_to_end)
                
                # Decay score based on distance (close timestamps get higher scores)
                proximity_score = max(proximity_score, 1.0 / (1.0 + 0.1 * min_distance))
        
        return proximity_score
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate keyword overlap similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation using token overlap
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                      "in", "on", "at", "to", "for", "with", "by", "about", "of"}
        tokens1 = tokens1.difference(stop_words)
        tokens2 = tokens2.difference(stop_words)
        
        overlap = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return overlap / max(1, union)
    
    def _extract_timestamp_mentions(self, text: str) -> List[str]:
        """
        Extract timestamps mentioned in a query.
        
        Args:
            text: Query text
            
        Returns:
            List of extracted timestamps
        """
        import re
        # Match timestamp patterns like [00:15:30] or timestamps without brackets
        timestamp_pattern = r'\[?(\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]?'
        
        timestamps = []
        for match in re.finditer(timestamp_pattern, text):
            timestamp = match.group(1)
            # Ensure HH:MM:SS format by adding seconds if needed
            if timestamp.count(':') == 1:
                timestamp += ':00'
            timestamps.append(timestamp)
            
        return timestamps
    
    def update_history(
        self, 
        question: str, 
        response: Dict[str, Any], 
        query_info: Dict[str, Any]
    ) -> None:
        """
        Update conversation history with a new Q&A pair.
        
        Args:
            question: The user's question
            response: The system's response
            query_info: Information about the query
        """
        # Create a new context item
        timestamp = time.time()
        answer = response.get("answer", "")
        metadata = {
            "query_type": query_info.get("query_type", "unknown"),
            "used_chunks": [chunk.chunk_id for chunk in self.current_context_chunks],
            "confidence": response.get("confidence", 0.0)
        }
        
        new_item = ContextItem(
            question=question,
            answer=answer,
            timestamp=timestamp,
            metadata=metadata
        )
        
        # Add to history
        self.conversation_history.append(new_item)
        
        # Limit history size
        history_limit = self.config["history_limit"]
        if len(self.conversation_history) > history_limit:
            self.conversation_history = self.conversation_history[-history_limit:]
        
        self.last_update_time = timestamp
    
    def reset_history(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        self.last_update_time = time.time()
    
    def get_conversation_context(self, max_pairs: Optional[int] = None) -> str:
        """
        Get formatted conversation history for context.
        
        Args:
            max_pairs: Maximum number of Q&A pairs to include
            
        Returns:
            Formatted conversation history
        """
        if max_pairs is None:
            max_pairs = self.config["history_limit"]
            
        # Get the most recent conversation pairs
        recent_items = self.conversation_history[-max_pairs:]
        
        # Format the conversation
        formatted = []
        for item in recent_items:
            formatted.append(f"User: {item.question}")
            formatted.append(f"Assistant: {item.answer}")
        
        return "\n".join(formatted)
    
    def get_compressed_history(self, max_tokens: int = 500) -> str:
        """
        Get a compressed version of conversation history.
        
        Args:
            max_tokens: Approximate maximum tokens in result
            
        Returns:
            Compressed conversation history
        """
        if not self.conversation_history:
            return ""
            
        # Estimate tokens (rough approximation)
        def estimate_tokens(text: str) -> int:
            return len(text.split())
        
        # Start with most recent history
        history = list(reversed(self.conversation_history))
        
        # Build compressed history
        compressed = []
        current_tokens = 0
        
        for item in history:
            q_tokens = estimate_tokens(item.question)
            a_tokens = estimate_tokens(item.answer)
            
            # If adding both would exceed limit, try just question
            if current_tokens + q_tokens + a_tokens > max_tokens:
                if current_tokens + q_tokens <= max_tokens:
                    compressed.append(f"User: {item.question}")
                    compressed.append("Assistant: [Response summarized]")
                    break
                else:
                    break
                    
            # Add the full Q&A pair
            compressed.append(f"User: {item.question}")
            compressed.append(f"Assistant: {item.answer}")
            current_tokens += q_tokens + a_tokens
            
            # Check if we've reached the limit
            if current_tokens >= max_tokens:
                break
        
        # Reverse back to chronological order
        compressed.reverse()
        return "\n".join(compressed)
