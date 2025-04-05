# gemini_mvp/long_context_handler.py - ACTIVELY USED
# Handles long context processing and optimization for large transcripts

"""
Long Context Handler module for managing large context windows.

This module optimizes context selection for large documents, implements
chunking strategies, and ensures efficient token usage.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
import time
import json
from tqdm import tqdm
import re

from .chunking import Chunk, ChunkManager
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)

@dataclass
class ContextWindow:
    """Represents a context window within a large document."""
    
    start_idx: int
    end_idx: int
    chunks: List[Chunk]
    importance_score: float = 0.0
    tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
            "importance_score": self.importance_score,
            "tokens": self.tokens
        }


class LongContextHandler:
    """
    Handles optimization strategies for long context.
    
    Features:
    - Context windowing across large transcripts
    - Semantic segmentation of transcripts
    - Context compression for token optimization
    - Strategic context selection for multiple questions
    """
    
    def __init__(
        self,
        chunk_manager: ChunkManager,
        cache_manager: CacheManager,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the long context handler.
        
        Args:
            chunk_manager: Chunk manager instance
            cache_manager: Cache manager instance
            config: Configuration options
        """
        self.chunk_manager = chunk_manager
        self.cache_manager = cache_manager
        self.config = config or {}
        
        # Context window configuration
        self.max_window_size = self.config.get("max_window_size", 30)  # Max chunks per window
        self.window_overlap = self.config.get("window_overlap", 5)     # Overlap between windows
        self.window_stride = max(1, self.max_window_size - self.window_overlap)
        
        # Token management
        self.max_context_tokens = self.config.get("max_context_tokens", 100000)
        self.token_buffer = self.config.get("token_buffer", 5000)      # Buffer for safety
        
        # Threshold for considering chunks similar
        self.chunk_similarity_threshold = self.config.get("chunk_similarity_threshold", 0.8)
        
        # Cache for window importance scores
        self.window_importance_cache = {}
        
        # Context windows for the current document
        self.context_windows: List[ContextWindow] = []
        
        # Tracking timestamp overlap
        self.timestamp_regex = re.compile(r'(\d{1,2}):(\d{2}):(\d{2})')
    
    def process_transcript(self, transcript_text: str) -> None:
        """
        Process a long transcript into optimized context windows.
        
        Args:
            transcript_text: The transcript text
        """
        logger.info("Processing transcript for long context handling")
        
        # Check if we've already processed this transcript
        transcript_hash = self.cache_manager._generate_key(transcript_text)
        cached_windows = self.cache_manager.get(
            f"context_windows_{transcript_hash}",
            cache_type="context"
        )
        
        if cached_windows:
            logger.info("Using cached context windows")
            self._reconstruct_windows(cached_windows)
            return
        
        # Process the transcript through chunk manager if needed
        if not self.chunk_manager.chunks:
            logger.info("Processing transcript through chunk manager")
            self.chunk_manager.process_transcript(transcript_text)
        
        # Create overlapping context windows
        self._create_context_windows()
        
        # Calculate importance scores for each window
        self._calculate_window_importance()
        
        # Cache the windows for future use
        window_dicts = [window.to_dict() for window in self.context_windows]
        self.cache_manager.set(
            f"context_windows_{transcript_hash}",
            window_dicts,
            cache_type="context",
            ttl=86400 * 7  # Store for 7 days
        )
        
        logger.info(f"Created {len(self.context_windows)} context windows")
    
    def _create_context_windows(self) -> None:
        """
        Create overlapping context windows from chunks.
        """
        chunks = self.chunk_manager.chunks
        
        if not chunks:
            logger.warning("No chunks available to create context windows")
            return
            
        # Create overlapping windows of chunks
        windows = []
        
        for i in range(0, len(chunks), self.window_stride):
            end_idx = min(i + self.max_window_size, len(chunks))
            window_chunks = chunks[i:end_idx]
            
            # Calculate approximate token count
            tokens = sum(len(chunk.text) // 4 for chunk in window_chunks)  # Approximate 4 chars per token
            
            window = ContextWindow(
                start_idx=i,
                end_idx=end_idx,
                chunks=window_chunks,
                tokens=tokens
            )
            
            windows.append(window)
            
            # If we've reached the end, break
            if end_idx == len(chunks):
                break
        
        self.context_windows = windows
    
    def _calculate_window_importance(self) -> None:
        """
        Calculate importance scores for each context window.
        """
        if not self.context_windows:
            return
            
        for window in self.context_windows:
            # Calculate various features that contribute to importance
            
            # 1. Presence of timestamps (more timestamps = more useful for referencing)
            timestamp_count = sum(
                len(self.timestamp_regex.findall(chunk.text))
                for chunk in window.chunks
            )
            
            # 2. Text length (longer windows may contain more information)
            text_length = sum(len(chunk.text) for chunk in window.chunks)
            
            # 3. Topic coherence (chunks with similar topics are more important together)
            topic_coherence = self._calculate_topic_coherence(window.chunks)
            
            # Combine into importance score (weights can be tuned)
            importance = (
                0.4 * (timestamp_count / max(1, self.max_window_size)) +
                0.3 * (text_length / 5000) +  # Normalize to a reasonable length
                0.3 * topic_coherence
            )
            
            window.importance_score = min(1.0, importance)  # Cap at 1.0
    
    def _calculate_topic_coherence(self, chunks: List[Chunk]) -> float:
        """
        Calculate topic coherence for a set of chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Coherence score (0-1)
        """
        if not chunks or len(chunks) <= 1:
            return 0.0
            
        # Simple approach: check for common keywords across chunks
        common_keywords = set()
        first = True
        
        for chunk in chunks:
            # Extract keywords (simple approach: take longer words)
            words = set(word.lower() for word in chunk.text.split() if len(word) > 5)
            
            if first:
                common_keywords = words
                first = False
            else:
                common_keywords &= words
        
        # Score based on number of common keywords
        return min(1.0, len(common_keywords) / 10)  # Cap at 1.0
    
    def _reconstruct_windows(self, window_dicts: List[Dict[str, Any]]) -> None:
        """
        Reconstruct context windows from cached dictionary representations.
        
        Args:
            window_dicts: List of window dictionaries
        """
        self.context_windows = []
        
        for window_dict in window_dicts:
            # Get the chunks by IDs
            chunk_ids = window_dict.get("chunk_ids", [])
            chunks = []
            
            for chunk_id in chunk_ids:
                # Find chunk with matching ID
                for chunk in self.chunk_manager.chunks:
                    if chunk.chunk_id == chunk_id:
                        chunks.append(chunk)
                        break
            
            # Create the window
            window = ContextWindow(
                start_idx=window_dict.get("start_idx", 0),
                end_idx=window_dict.get("end_idx", 0),
                chunks=chunks,
                importance_score=window_dict.get("importance_score", 0.0),
                tokens=window_dict.get("tokens", 0)
            )
            
            self.context_windows.append(window)
    
    def get_optimal_context(
        self, 
        query: str,
        query_embeddings: Optional[np.ndarray] = None,
        token_budget: Optional[int] = None
    ) -> List[Chunk]:
        """
        Get optimal context chunks for a query within token budget.
        
        Args:
            query: The query text
            query_embeddings: Query embeddings (if available)
            token_budget: Maximum tokens for context
            
        Returns:
            List of context chunks
        """
        if token_budget is None:
            token_budget = self.max_context_tokens - self.token_buffer
            
        if not self.context_windows:
            logger.warning("No context windows available")
            return self.chunk_manager.get_relevant_chunks(query, max_chunks=self.max_window_size)
        
        # Get scored windows for the query
        scored_windows = self._score_windows_for_query(query, query_embeddings)
        
        # Sort by score (descending)
        scored_windows.sort(key=lambda x: x[0], reverse=True)
        
        # Select windows until we hit token budget
        selected_chunks: Set[Chunk] = set()
        total_tokens = 0
        
        for score, window in scored_windows:
            # Check if adding this window would exceed token budget
            window_tokens = window.tokens
            
            if total_tokens + window_tokens > token_budget:
                # If this is the first window, take it anyway (partial context better than none)
                if not selected_chunks:
                    # Add chunks until we hit token budget
                    for chunk in window.chunks:
                        chunk_tokens = len(chunk.text) // 4  # Approximate 4 chars per token
                        if total_tokens + chunk_tokens <= token_budget:
                            selected_chunks.add(chunk)
                            total_tokens += chunk_tokens
                # Otherwise skip this window
                continue
            
            # Add all chunks from this window
            for chunk in window.chunks:
                selected_chunks.add(chunk)
            total_tokens += window_tokens
            
            # If we've reached token budget, stop
            if total_tokens >= token_budget:
                break
        
        # Convert set to list and sort by chunk index to maintain order
        result = list(selected_chunks)
        # Try to sort by id, start_time, or another attribute if available
        if result and hasattr(result[0], 'id'):
            result = sorted(result, key=lambda x: x.id)
        elif result and hasattr(result[0], 'start_time'):
            result = sorted(result, key=lambda x: x.start_time)
        
        logger.info(f"Selected {len(result)} chunks with ~{total_tokens} tokens for context")
        return result
    
    def _score_windows_for_query(
        self, 
        query: str,
        query_embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[float, ContextWindow]]:
        """
        Score context windows for relevance to a query.
        
        Args:
            query: The query text
            query_embeddings: Query embeddings (if available)
            
        Returns:
            List of (score, window) tuples
        """
        # Check cache first
        cache_key = f"window_scores_{self.cache_manager._generate_key(query)}"
        cached_scores = self.cache_manager.get(cache_key, cache_type="scores")
        
        if cached_scores:
            # Reconstruct the scored windows
            scored_windows = []
            for score, window_idx in cached_scores:
                if 0 <= window_idx < len(self.context_windows):
                    scored_windows.append((score, self.context_windows[window_idx]))
            return scored_windows
        
        # Calculate scores for each window
        scored_windows = []
        
        for window in self.context_windows:
            # Baseline score is the window's importance
            base_score = window.importance_score
            
            # Calculate relevance to query
            query_relevance = self._calculate_query_relevance(query, window.chunks)
            
            # Combine scores (weight can be tuned)
            score = 0.3 * base_score + 0.7 * query_relevance
            
            scored_windows.append((score, window))
        
        # Cache the scores
        cache_data = [(score, self.context_windows.index(window)) for score, window in scored_windows]
        self.cache_manager.set(cache_key, cache_data, cache_type="scores")
        
        return scored_windows
    
    def _calculate_query_relevance(self, query: str, chunks: List[Chunk]) -> float:
        """
        Calculate relevance of chunks to a query.
        
        Args:
            query: The query text
            chunks: List of chunks
            
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword matching for now (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        # Remove common stop words
        stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about", "like"}
        query_words -= stop_words
        
        # Count chunks that contain query words
        matching_chunks = 0
        
        for chunk in chunks:
            chunk_text = chunk.text.lower()
            if any(word in chunk_text for word in query_words):
                matching_chunks += 1
        
        # Calculate relevance as proportion of matching chunks
        return matching_chunks / max(1, len(chunks))
    
    def optimize_context_for_batch(
        self, 
        questions: List[str], 
        token_budget: Optional[int] = None
    ) -> List[Chunk]:
        """
        Optimize context selection for a batch of questions.
        
        Args:
            questions: List of questions
            token_budget: Maximum tokens for context
            
        Returns:
            List of optimized context chunks
        """
        if token_budget is None:
            token_budget = self.max_context_tokens - self.token_buffer
        
        if not questions:
            return []
            
        if len(questions) == 1:
            return self.get_optimal_context(questions[0], token_budget=token_budget)
            
        # Calculate scores for each window across all questions
        window_scores = [0.0] * len(self.context_windows)
        
        for question in questions:
            scored_windows = self._score_windows_for_query(question, None)
            
            # Aggregate scores
            for score, window in scored_windows:
                window_idx = self.context_windows.index(window)
                window_scores[window_idx] += score / len(questions)  # Normalize by question count
        
        # Create scored windows list
        scored_windows = [(score, self.context_windows[i]) for i, score in enumerate(window_scores)]
        
        # Sort by score (descending)
        scored_windows.sort(key=lambda x: x[0], reverse=True)
        
        # Select windows until we hit token budget (same logic as get_optimal_context)
        selected_chunks: Set[Chunk] = set()
        total_tokens = 0
        
        for score, window in scored_windows:
            # Check if adding this window would exceed token budget
            window_tokens = window.tokens
            
            if total_tokens + window_tokens > token_budget:
                # If this is the first window, take it anyway
                if not selected_chunks:
                    for chunk in window.chunks:
                        chunk_tokens = len(chunk.text) // 4
                        if total_tokens + chunk_tokens <= token_budget:
                            selected_chunks.add(chunk)
                            total_tokens += chunk_tokens
                continue
            
            # Add all chunks from this window
            for chunk in window.chunks:
                selected_chunks.add(chunk)
            total_tokens += window_tokens
            
            # If we've reached token budget, stop
            if total_tokens >= token_budget:
                break
        
        # Convert set to list and sort by chunk index to maintain order
        result = list(selected_chunks)
        # Try to sort by id, start_time, or another attribute if available
        if result and hasattr(result[0], 'id'):
            result = sorted(result, key=lambda x: x.id)
        elif result and hasattr(result[0], 'start_time'):
            result = sorted(result, key=lambda x: x.start_time)
        
        logger.info(f"Selected {len(result)} chunks with ~{total_tokens} tokens for batch context")
        return result
    
    def get_context_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current context.
        
        Returns:
            Dictionary with context statistics
        """
        if not self.context_windows:
            return {
                "windows": 0,
                "chunks": 0,
                "tokens": 0,
                "max_window_size": self.max_window_size,
                "window_overlap": self.window_overlap
            }
            
        total_chunks = sum(len(window.chunks) for window in self.context_windows)
        total_tokens = sum(window.tokens for window in self.context_windows)
        
        return {
            "windows": len(self.context_windows),
            "chunks": total_chunks,
            "unique_chunks": len(self.chunk_manager.chunks),
            "tokens": total_tokens,
            "avg_tokens_per_window": total_tokens / len(self.context_windows),
            "max_window_size": self.max_window_size,
            "window_overlap": self.window_overlap,
            "window_stride": self.window_stride
        }
