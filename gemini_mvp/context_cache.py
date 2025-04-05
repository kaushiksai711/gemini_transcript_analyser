# gemini_mvp/context_cache.py - ACTIVELY USED
# Used for caching context to reduce token usage and API calls

"""
Context Cache module for storing and retrieving context chunks.

This module implements caching logic for context chunks to reduce
token usage and improve response times.
"""

import os
import logging
import time
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class ContextCache:
    """
    Specialized caching system for context reuse.
    
    Features:
    - Semantic context caching for reusing similar contexts
    - Token usage tracking and optimization
    - Cache performance metrics
    """
    
    def __init__(self, cache_manager, config: Dict[str, Any] = None):
        """
        Initialize the context cache.
        
        Args:
            cache_manager: Cache manager instance
            config: Configuration options
        """
        self.cache_manager = cache_manager
        self.config = config or {}
        
        # Configure context similarity threshold
        self.context_similarity_threshold = self.config.get("context_similarity_threshold", 0.6)
        # Cache statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0,
            "tokens_used": 0,
            "cache_hit_ratio": 0.0,
            "semantic_matches": 0,
            "exact_matches": 0,
            "partial_cache_hits": 0
        }
        
        # Token tracking
        self.token_savings = defaultdict(int)
        
        # Initialize timestamp for calculating cache hit ratio
        self.last_stats_reset = time.time()
    
    def get_context(self, query: str, available_chunks: List, token_budget: int) -> Tuple[List, bool, int]:
        """
        Get relevant context for a query, with caching applied.
        
        Args:
            query: The query string
            available_chunks: List of available context chunks
            token_budget: Maximum tokens for context
            
        Returns:
            Tuple of (context_chunks, is_cache_hit, tokens_saved)
        """
        # Generate a cache key for this query
        cache_key = f"context:{self.cache_manager._generate_key(query)}"
        
        # Try to get from cache
        cached_context = self.cache_manager.get(cache_key, cache_type="context")
        
        if cached_context:
            self.stats["cache_hits"] += 1
            self.stats["exact_matches"] += 1
            
            # Calculate tokens saved by using cache - adapt to model capabilities
            # For Gemini 2.0 Flash Thinking model, we can handle much larger contexts efficiently
            # Detect if we're using the Gemini 2.0 Flash Thinking model
            model_name = self._detect_model_name()
            
            if "gemini-2.0-flash-thinking-exp-01-21" in model_name:
                # Enhanced token savings calculation for Gemini 2.0 Flash Thinking
                chars_per_token = 3.5  # More efficient tokenization
                processing_overhead = 1.2  # 20% overhead for Flash Thinking
            else:
                # Standard token estimation for other models
                chars_per_token = 4.0  # Standard tokenization rate
                processing_overhead = 1.1  # 10% overhead for standard models
            
            if isinstance(cached_context[0], dict) and "tokens" in cached_context[0]:
                # If tokens field is available in chunk dict, use it directly
                tokens_saved = sum(chunk.get("tokens", 100) for chunk in cached_context)
            else:
                # Estimate based on text length and model-specific tokenization rate
                tokens_saved = 0
                for chunk in cached_context:
                    if isinstance(chunk, dict) and "text" in chunk:
                        tokens_saved += len(chunk["text"]) // chars_per_token
                    elif hasattr(chunk, 'text'):
                        tokens_saved += len(chunk.text) // chars_per_token
                    else:
                        tokens_saved += len(str(chunk)) // chars_per_token
                
            # Add overhead for processing based on model
            tokens_saved = int(tokens_saved * processing_overhead)
            
            self.stats["tokens_saved"] += tokens_saved
            self.token_savings[query] = tokens_saved
            
            logger.info(f"Context cache hit for query: '{query[:50]}...' (saved ~{tokens_saved} tokens using {model_name})")
            return cached_context, True, tokens_saved
        
        # If no exact match, check for semantic similarity
        # Enhanced semantic similarity matching with query analysis
        
        # Extract key concepts from the query to improve matching
        key_concepts = self._extract_key_concepts(query)
        
        similar_key = None
        highest_sim = 0
        
        # Check for similar queries in cache
        for key in self.cache_manager.list_keys(cache_type="context"):
            if key.startswith("context:"):
                stored_query = key[8:]  # Remove "context:" prefix
                
                # Calculate semantic similarity with enhanced metrics
                similarity = self._calculate_semantic_similarity(query, stored_query, key_concepts)
                
                if similarity > highest_sim and similarity > self.context_similarity_threshold:
                    highest_sim = similarity
                    similar_key = key
        
        # If we found a similar context, use it
        if similar_key:
            similar_context = self.cache_manager.get(similar_key, cache_type="context")
            
            if similar_context:
                self.stats["cache_hits"] += 1
                self.stats["semantic_matches"] += 1
                
                # Calculate tokens saved with model-specific optimizations
                model_name = self._detect_model_name()
                
                if "gemini-2.0-flash-thinking-exp-01-21" in model_name:
                    # Enhanced token savings for advanced model
                    chars_per_token = 3.5
                    similarity_factor = (highest_sim + 1) / 2  # Boost similarity factor for better models
                else:
                    # Standard token estimation
                    chars_per_token = 4.0
                    similarity_factor = highest_sim
                
                # Calculate token savings with enhanced estimation
                tokens_saved = 0
                for chunk in similar_context:
                    if isinstance(chunk, dict) and "tokens" in chunk:
                        tokens_saved += chunk.get("tokens", 100)
                    elif isinstance(chunk, dict) and "text" in chunk:
                        tokens_saved += len(chunk["text"]) // chars_per_token
                    elif hasattr(chunk, 'text'):
                        tokens_saved += len(chunk.text) // chars_per_token
                    else:
                        tokens_saved += len(str(chunk)) // chars_per_token
                
                # Apply similarity-based discount but recognize context value
                tokens_saved = int(tokens_saved * similarity_factor)
                
                self.stats["tokens_saved"] += tokens_saved
                self.token_savings[query] = tokens_saved
                
                logger.info(f"Semantic context match for: '{query[:50]}...' (saved ~{tokens_saved} tokens, similarity: {highest_sim:.2f})")
                return similar_context, True, tokens_saved
        
        # If no cache hit, process the available chunks
        self.stats["cache_misses"] += 1
        
        # Select chunks within token budget
        # Note: This assumes the chunks have already been sorted by relevance
        selected_chunks = []
        tokens_used = 0
        
        for chunk in available_chunks:
            # Estimate token count for this chunk
            if hasattr(chunk, 'tokens'):
                chunk_tokens = chunk.tokens
            elif hasattr(chunk, 'text'):
                # Approximate 1 token per 4 chars for standard, 3.5 for advanced models
                model_name = self._detect_model_name()
                chars_per_token = 3.5 if "gemini-2.0-flash-thinking-exp-01-21" in model_name else 4.0
                chunk_tokens = len(chunk.text) // chars_per_token
            else:
                # Default estimate
                chunk_tokens = 100
            
            # Check if adding this chunk would exceed the token budget
            if tokens_used + chunk_tokens <= token_budget:
                selected_chunks.append(chunk)
                tokens_used += chunk_tokens
            else:
                # If this is the first chunk and we're already over budget,
                # include it anyway to ensure we have at least some context
                if not selected_chunks:
                    selected_chunks.append(chunk)
                break
        
        # Cache the selected chunks for future use
        if selected_chunks:
            self.cache_manager.set(cache_key, selected_chunks, cache_type="context")
            
        return selected_chunks, False, 0
        
    def _detect_model_name(self) -> str:
        """
        Detect the model name being used from the configuration.
        
        Returns:
            Model name string
        """
        # Try to get model name from config
        if self.config and "model" in self.config:
            return self.config.get("model", "")
        
        # If not in config, check if API client is accessible
        if hasattr(self, 'api_client') and hasattr(self.api_client, 'model_name'):
            return self.api_client.model_name
        
        # Default to standard model if unknown
        return "gemini-1.0-pro"
        
    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from a query to improve matching.
        
        Args:
            query: The query string
            
        Returns:
            List of key concepts
        """
        # Skip common words and only keep meaningful terms
        common_words = {
            "the", "and", "or", "if", "a", "an", "to", "is", "are", "was", "were", 
            "in", "on", "at", "by", "for", "with", "about", "that", "this",
            "what", "when", "where", "who", "how", "why", "which", "do", "does", "did"
        }
        
        # Extract terms that are likely to be meaningful
        words = query.lower().split()
        key_terms = [w for w in words if w not in common_words and len(w) > 3]
        
        # Also extract any quoted phrases as concepts
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        
        # Combine individual terms and phrases
        concepts = key_terms + quoted_phrases
        
        return concepts
        
    def _calculate_semantic_similarity(self, query1: str, query2: str, key_concepts: List[str] = None) -> float:
        """
        Calculate semantic similarity between two queries with enhanced matching.
        
        Args:
            query1: First query
            query2: Second query
            key_concepts: Optional key concepts extracted from query1
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap as base similarity
        q1_words = set(query1.lower().split())
        q2_words = set(query2.lower().split())
        
        if not q1_words or not q2_words:
            return 0.0
            
        overlap = len(q1_words.intersection(q2_words))
        total = len(q1_words.union(q2_words))
        base_similarity = overlap / total if total > 0 else 0
        
        # Enhanced similarity with concept matching
        concept_similarity = 0.0
        if key_concepts:
            # Check how many key concepts from query1 appear in query2
            query2_lower = query2.lower()
            matched_concepts = sum(1 for c in key_concepts if c.lower() in query2_lower)
            if len(key_concepts) > 0:
                concept_similarity = matched_concepts / len(key_concepts)
        
        # Combine base similarity with concept similarity (weighting concepts higher)
        combined_similarity = (base_similarity * 0.4) + (concept_similarity * 0.6)
        
        return combined_similarity

    def update_stats(self, query: str, context_chunks: List, is_cache_hit: bool, tokens_saved: int = 0) -> None:
        """
        Update cache statistics.
        
        Args:
            query: The query string
            context_chunks: Context chunks used
            is_cache_hit: Whether this was a cache hit
            tokens_saved: Tokens saved by using cache
        """
        if is_cache_hit:
            self.stats["cache_hits"] += 1
            self.stats["tokens_saved"] += tokens_saved
            self.token_savings[query] = tokens_saved
        else:
            self.stats["cache_misses"] += 1
            tokens_used = sum(chunk.get("tokens", 100) for chunk in context_chunks)
            self.stats["tokens_used"] += tokens_used
        
        # Calculate cache hit ratio
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            self.stats["cache_hit_ratio"] = self.stats["cache_hits"] / total_requests
    
    def clear_cache(self) -> None:
        """Clear the context cache."""
        self.cache_manager.clear(cache_type="context")
        
        # Reset statistics
        self.stats["cache_hits"] = 0
        self.stats["cache_misses"] = 0
        self.stats["tokens_saved"] = 0
        self.stats["tokens_used"] = 0
        self.stats["cache_hit_ratio"] = 0.0
        self.stats["semantic_matches"] = 0
        self.stats["exact_matches"] = 0
        self.stats["partial_cache_hits"] = 0
        
        # Reset token savings
        self.token_savings = defaultdict(int)
        
        # Reset timestamp
        self.last_stats_reset = time.time()
        
        logger.info("Context cache cleared and statistics reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.stats.copy()
        
        # Add token savings distribution
        stats["token_savings_per_query"] = dict(self.token_savings)
        
        # Add time since last reset
        stats["seconds_since_reset"] = time.time() - self.last_stats_reset
        
        return stats

    def optimize_for_batch(self, queries: List[str], available_chunks: List, total_token_budget: int) -> Dict[str, List]:
        """
        Optimize context selection for a batch of queries.
        
        Args:
            queries: List of query strings
            available_chunks: List of available context chunks
            total_token_budget: Total token budget for all queries
            
        Returns:
            Dictionary mapping query to context chunks
        """
        # Allocate token budget per query
        token_budget_per_query = total_token_budget // max(1, len(queries))
        
        # Prepare result mapping
        query_to_context = {}
        
        # Track overall token usage
        total_tokens_used = 0
        total_tokens_saved = 0
        
        # Process queries
        for query in queries:
            # Check if we're still under budget
            remaining_budget = total_token_budget - total_tokens_used
            if remaining_budget <= 0:
                logger.warning(f"Token budget exhausted, skipping query: {query}")
                query_to_context[query] = []
                continue
            
            # Get context for this query
            context, is_cache_hit, tokens_saved = self.get_context(
                query=query, 
                available_chunks=available_chunks,
                token_budget=min(token_budget_per_query, remaining_budget)
            )
            
            # Store in result map
            query_to_context[query] = context
            
            # Update token tracking
            if is_cache_hit:
                total_tokens_saved += tokens_saved
            else:
                total_tokens_used += sum(chunk.get("tokens", 100) for chunk in context)
        
        # Log optimization results
        logger.info(f"Optimized context for {len(queries)} queries")
        logger.info(f"Total tokens used: {total_tokens_used}, total tokens saved: {total_tokens_saved}")
        
        return query_to_context
