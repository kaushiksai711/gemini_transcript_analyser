# gemini_mvp/query_processor.py - ACTIVELY USED
# Used by TranscriptProcessor for processing individual queries

"""
Query Processor module for handling individual questions.

This module analyzes questions, extracts key information, and
prepares queries for optimal API interaction.
"""

import re
import time
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from .context_manager import ContextManager

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Handles question processing and dependency tracking.
    
    This class is responsible for:
    1. Classifying questions by type
    2. Detecting dependencies between questions
    3. Batching independent questions for efficient processing
    """
    
    def __init__(self, config: Dict[str, Any], context_manager: ContextManager):
        """
        Initialize the query processor.
        
        Args:
            config: Configuration for dependency tracking
            context_manager: Context manager instance
        """
        self.config = config
        self.context_manager = context_manager
        
        # Track query metadata for caching
        self.processed_queries: Dict[str, Dict[str, Any]] = {}
        
        # Reference patterns for direct dependency detection
        self.reference_patterns = [
            r"(previous|above|earlier)(\s+answer|\s+response|\s+question)?",
            r"(you\s+(just|recently)\s+(said|mentioned|noted))",
            r"(that|this|it|which|they|these|those)(\s+one)?$",
            r"follow[-\s]up",
            r"related\s+to\s+(that|this|it|which)"
        ]
        self.reference_regex = re.compile("|".join(self.reference_patterns), re.IGNORECASE)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query to extract information and detect dependencies.
        
        Args:
            query: The question to process
            
        Returns:
            Dictionary with query information
        """
        logger.info(f"Processing query: {query}")
        
        # Check cache first
        if query in self.processed_queries:
            logger.info("Using cached query information")
            return self.processed_queries[query]
        
        # Classify query
        query_type = self._classify_query(query)
        
        # Detect dependencies
        dependencies = self._detect_dependencies(query)
        
        # Create query info
        query_info = {
            "query_type": query_type,
            "dependencies": dependencies,
            "timestamp": time.time(),
            "entities": self._extract_entities(query)
        }
        
        # Cache the results
        self.processed_queries[query] = query_info
        
        return query_info
    
    def _classify_query(self, query: str) -> str:
        """
        Classify a query as factual, inferential, or opinion-based.
        
        Args:
            query: The question to classify
            
        Returns:
            Classification string
        """
        query_lower = query.lower()
        
        # Check for opinion patterns
        opinion_patterns = [
            "what do you think", "your opinion", "would you say",
            "do you believe", "how do you feel", "what's your take"
        ]
        for pattern in opinion_patterns:
            if pattern in query_lower:
                return "opinion"
        
        # Check for inferential patterns
        inferential_patterns = [
            "why", "how come", "what if", "explain why", "reason for",
            "implication", "infer", "interpret", "analyze", "understand the significance"
        ]
        for pattern in inferential_patterns:
            if pattern in query_lower:
                return "inferential"
        
        # Default to factual
        return "factual"
    
    def _detect_dependencies(self, query: str) -> Dict[str, Any]:
        """
        Detect dependencies on previous questions/answers.
        
        Args:
            query: The question to analyze
            
        Returns:
            Dependency information
        """
        # Get conversation history
        history = self.context_manager.conversation_history
        if not history:
            return {"has_dependencies": False}
        
        # Check for direct references
        direct_ref = self._check_direct_references(query)
        
        # Check for semantic similarity
        semantic_ref = self._check_semantic_similarity(query, history)
        
        # Check for entity/keyword overlap
        keyword_ref = self._check_keyword_overlap(query, history)
        
        # Combine signals
        confidence_blend = self.config["confidence_blend"]
        semantic_confidence = semantic_ref.get("confidence", 0)
        keyword_confidence = keyword_ref.get("confidence", 0)
        
        # Calculate combined confidence
        combined_confidence = (
            confidence_blend * semantic_confidence + 
            (1 - confidence_blend) * keyword_confidence
        )
        
        # Determine if there's a dependency
        has_dependency = (
            direct_ref["has_direct_reference"] or
            combined_confidence >= self.config["similarity_threshold"]
        )
        
        # Prepare result
        result = {
            "has_dependencies": has_dependency,
            "direct_reference": direct_ref,
            "semantic_similarity": semantic_ref,
            "keyword_overlap": keyword_ref,
            "combined_confidence": combined_confidence
        }
        
        # If there's a dependency, include the referenced items
        if has_dependency:
            referenced_indices = set()
            
            # Add directly referenced items
            if direct_ref["has_direct_reference"]:
                referenced_indices.update(direct_ref.get("referenced_indices", []))
            
            # Add semantically similar items
            if semantic_ref.get("confidence", 0) >= self.config["similarity_threshold"]:
                referenced_indices.update(semantic_ref.get("referenced_indices", []))
            
            # Add keyword overlap items if confidence is high enough
            if keyword_confidence >= self.config["keyword_threshold"]:
                referenced_indices.update(keyword_ref.get("referenced_indices", []))
            
            # Get the referenced Q&A pairs
            referenced_items = []
            for idx in sorted(referenced_indices):
                if idx < len(history):
                    item = history[idx]
                    referenced_items.append({
                        "question": item.question,
                        "answer": item.answer,
                        "index": idx
                    })
            
            result["referenced_items"] = referenced_items
        
        return result
    
    def _check_direct_references(self, query: str) -> Dict[str, Any]:
        """
        Check for direct references to previous answers.
        
        Args:
            query: The question to analyze
            
        Returns:
            Direct reference information
        """
        history = self.context_manager.conversation_history
        
        # Check for direct reference patterns
        match = self.reference_regex.search(query)
        has_direct_reference = match is not None
        
        result = {"has_direct_reference": has_direct_reference}
        
        if has_direct_reference:
            # Default to most recent Q&A pair
            result["referenced_indices"] = [len(history) - 1]
        
        return result
    
    def _check_semantic_similarity(
        self, 
        query: str, 
        history: List[Any]
    ) -> Dict[str, Any]:
        """
        Check for semantic similarity with previous questions.
        
        Args:
            query: The current question
            history: Conversation history
            
        Returns:
            Semantic similarity information
        """
        # If no embedding model, return low confidence
        if not hasattr(self.context_manager, 'embedding_model') or self.context_manager.embedding_model is None:
            return {"confidence": 0.0}
        
        # Get the embedding model
        embedding_model = self.context_manager.embedding_model
        
        # Encode the query
        query_embedding = embedding_model.encode(query)
        
        # Calculate similarity with previous questions
        max_similarity = 0.0
        max_index = -1
        similarities = []
        
        for i, item in enumerate(history):
            # Get or calculate question embedding
            question = item.question
            cache_key = f"question_embedding_{hash(question)}"
            question_embedding = self.context_manager.cache.get(cache_key)
            
            if question_embedding is None:
                question_embedding = embedding_model.encode(question)
                self.context_manager.cache.set(cache_key, question_embedding)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, question_embedding)
            similarities.append((i, similarity))
            
            # Track maximum similarity
            if similarity > max_similarity:
                max_similarity = similarity
                max_index = i
        
        # Sort similarities in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get indices of top similar questions
        threshold = self.config["similarity_threshold"]
        referenced_indices = [i for i, sim in similarities if sim >= threshold]
        
        return {
            "confidence": max_similarity,
            "most_similar_index": max_index,
            "referenced_indices": referenced_indices
        }
    
    def _check_keyword_overlap(
        self, 
        query: str, 
        history: List[Any]
    ) -> Dict[str, Any]:
        """
        Check for keyword/entity overlap with previous Q&A pairs.
        
        Args:
            query: The current question
            history: Conversation history
            
        Returns:
            Keyword overlap information
        """
        # Extract entities from query
        query_entities = set(self._extract_entities(query))
        
        # If no entities, use important keywords
        if not query_entities:
            query_words = self._extract_important_words(query)
            query_entities = set(query_words)
        
        # If still no entities, return low confidence
        if not query_entities:
            return {"confidence": 0.0}
        
        # Calculate overlap with previous Q&A pairs
        max_overlap = 0.0
        max_index = -1
        overlaps = []
        
        for i, item in enumerate(history):
            # Get entities from question and answer
            question_entities = set(self._extract_entities(item.question))
            answer_entities = set(self._extract_entities(item.answer))
            
            # Combine question and answer entities
            item_entities = question_entities.union(answer_entities)
            
            # If no entities, use important words
            if not item_entities:
                question_words = self._extract_important_words(item.question)
                answer_words = self._extract_important_words(item.answer)
                item_entities = set(question_words + answer_words)
            
            # Calculate Jaccard similarity
            if item_entities:
                overlap = len(query_entities.intersection(item_entities)) / len(query_entities.union(item_entities))
            else:
                overlap = 0.0
                
            overlaps.append((i, overlap))
            
            # Track maximum overlap
            if overlap > max_overlap:
                max_overlap = overlap
                max_index = i
        
        # Sort overlaps in descending order
        overlaps.sort(key=lambda x: x[1], reverse=True)
        
        # Get indices of top overlapping Q&A pairs
        threshold = self.config["keyword_threshold"]
        referenced_indices = [i for i, ovr in overlaps if ovr >= threshold]
        
        return {
            "confidence": max_overlap,
            "most_similar_index": max_index,
            "referenced_indices": referenced_indices
        }
    
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
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entities
        """
        # Simple implementation: extract capitalized words
        words = text.split()
        entities = [
            word.strip(".,?!\"'()[]{}") 
            for word in words 
            if word and word[0].isupper()
        ]
        return entities
    
    def _extract_important_words(self, text: str) -> List[str]:
        """
        Extract important words from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of important words
        """
        # Simple implementation: extract words with 4+ characters
        words = text.lower().split()
        
        # Remove stop words
        stop_words = {"the", "and", "but", "for", "nor", "yet", "from", "with", 
                      "this", "that", "these", "those", "they", "them", "their"}
        
        important = [
            word.strip(".,?!\"'()[]{}") 
            for word in words
            if len(word) >= 4 and word not in stop_words
        ]
        
        return important
    
    def batch_questions(self, questions: List[str]) -> List[List[str]]:
        """
        Group questions into batches based on dependencies.
        
        Args:
            questions: List of questions to batch
            
        Returns:
            List of question batches
        """
        logger.info(f"Batching {len(questions)} questions")
        
        # Process each question
        processed = []
        for q in questions:
            query_info = self.process_query(q)
            processed.append((q, query_info))
        
        # Group by dependencies
        batches = []
        current_batch = []
        
        for q, info in processed:
            if info.get("dependencies", {}).get("has_dependencies", False):
                # If this question has dependencies, finish the current batch
                # and start a new one with this question
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                
                # Add dependent question to its own batch
                batches.append([q])
            else:
                # Independent question, add to current batch
                current_batch.append(q)
        
        # Add any remaining questions
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Batched into {len(batches)} groups")
        return batches
