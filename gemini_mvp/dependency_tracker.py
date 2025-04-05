# gemini_mvp/dependency_tracker.py - ACTIVELY USED
# Used by BatchProcessor to track dependencies between questions

"""
Dependency Tracker module for analyzing question relationships.

This module identifies dependencies between questions, helps determine
optimal processing order, and enhances context selection.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy
from thefuzz import fuzz

logger = logging.getLogger(__name__)

class DependencyTracker:
    """
    Tracks dependencies between questions in multi-turn conversations.
    
    Uses a hybrid approach combining:
    1. Semantic similarity (using SBERT)
    2. Entity overlap detection
    3. Direct reference detection (pronouns, etc.)
    4. Previous context influence analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dependency tracker.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer(
                self.config.get("sentence_model", "all-MiniLM-L6-v2")
            )
            self.use_sentence_transformer = True
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            logger.warning("Falling back to simpler similarity methods")
            self.use_sentence_transformer = False
            
        # Initialize entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            self.use_spacy = False
            
        # History of previous questions and metadata
        self.question_history = []
        
        # Thresholds for similarity
        self.semantic_threshold = self.config.get("semantic_threshold", 0.65)
        self.entity_threshold = self.config.get("entity_threshold", 0.5)
        self.reference_threshold = self.config.get("reference_threshold", 0.7)
        
        # Regular expressions for direct references
        self.reference_patterns = [
            r"\b(it|they|them|those|these|this|that)\b",
            r"\b(the|this|that) (point|topic|issue|subject|concept|idea|problem|question)\b",
            r"\b(above|previous|earlier|prior|before|former|preceding|aforementioned)\b",
            r"\b(related to|regarding|concerning|about|refer to|referring to|mentioned)\b",
            r"\b(elaborate|expand|more|details|tell me more)\b"
        ]
        
    def add_to_history(self, question: str, answer: str, metadata: Dict[str, Any]) -> None:
        """
        Add a question-answer pair to history.
        
        Args:
            question: The question text
            answer: The answer text
            metadata: Additional metadata about the QA pair
        """
        self.question_history.append({
            "question": question,
            "answer": answer,
            "metadata": metadata,
            "entities": self._extract_entities(question + " " + answer) if self.use_spacy else [],
            "embedding": self._get_embedding(question) if self.use_sentence_transformer else None
        })
    
    def find_dependencies(self, current_question: str) -> List[Dict[str, Any]]:
        """
        Find dependencies between current question and question history.
        
        Args:
            current_question: The current question text
            
        Returns:
            List of dependency information with confidence scores
        """
        if not self.question_history:
            return []
            
        dependencies = []
        
        # Extract entities from current question
        current_entities = self._extract_entities(current_question) if self.use_spacy else []
        
        # Get embedding for current question
        current_embedding = self._get_embedding(current_question) if self.use_sentence_transformer else None
        
        # Check each previous question for dependencies
        for idx, prev_qa in enumerate(self.question_history):
            dependency_scores = {}
            
            # 1. Check for semantic similarity
            if self.use_sentence_transformer and current_embedding is not None and prev_qa["embedding"] is not None:
                sim_score = self._calculate_semantic_similarity(current_embedding, prev_qa["embedding"])
                dependency_scores["semantic"] = sim_score
            else:
                # Fallback to fuzzy matching
                sim_score = fuzz.token_sort_ratio(current_question, prev_qa["question"]) / 100
                dependency_scores["semantic"] = sim_score
            
            # 2. Check for entity overlap
            if self.use_spacy and current_entities and prev_qa["entities"]:
                entity_score = self._calculate_entity_overlap(current_entities, prev_qa["entities"])
                dependency_scores["entity"] = entity_score
            else:
                # Simple word overlap fallback
                entity_score = self._calculate_word_overlap(current_question, prev_qa["question"])
                dependency_scores["entity"] = entity_score
            
            # 3. Check for direct references
            reference_score = self._check_direct_references(current_question)
            dependency_scores["reference"] = reference_score
            
            # 4. Calculate recency score (more recent questions have higher influence)
            recency_score = 1.0 - (idx / max(1, len(self.question_history)))
            dependency_scores["recency"] = recency_score
            
            # Calculate composite score
            # Weights: semantic (0.4), entity (0.25), reference (0.25), recency (0.1)
            composite_score = (
                0.4 * dependency_scores.get("semantic", 0) +
                0.25 * dependency_scores.get("entity", 0) +
                0.25 * dependency_scores.get("reference", 0) +
                0.1 * dependency_scores.get("recency", 0)
            )
            
            # Only include dependencies with significant scores
            if composite_score >= self.reference_threshold:
                dependencies.append({
                    "question_idx": idx,
                    "question": prev_qa["question"],
                    "answer": prev_qa["answer"],
                    "composite_score": composite_score,
                    "detailed_scores": dependency_scores
                })
        
        # Sort by composite score (descending)
        dependencies.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return dependencies
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            return self.sentence_model.encode(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def _calculate_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        try:
            doc = self.nlp(text)
            # Extract named entities and noun chunks
            entities = [ent.text.lower() for ent in doc.ents]
            noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks 
                          if len(chunk.text) > 3 and chunk.text.lower() not in entities]
            
            return entities + noun_chunks
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _calculate_entity_overlap(self, entities1: List[str], entities2: List[str]) -> float:
        """
        Calculate entity overlap between two entity lists.
        
        Args:
            entities1: First entity list
            entities2: Second entity list
            
        Returns:
            Overlap score (0-1)
        """
        if not entities1 or not entities2:
            return 0
            
        # Exact matches
        exact_matches = set(entities1).intersection(set(entities2))
        
        # Fuzzy matches for entities that aren't exact matches
        fuzzy_matches = 0
        remaining_entities1 = [e for e in entities1 if e not in exact_matches]
        remaining_entities2 = [e for e in entities2 if e not in exact_matches]
        
        for e1 in remaining_entities1:
            for e2 in remaining_entities2:
                if fuzz.token_sort_ratio(e1, e2) > 80:  # threshold for fuzzy match
                    fuzzy_matches += 0.8  # count as partial match
                    break
        
        # Calculate overlap score
        overlap = len(exact_matches) + fuzzy_matches
        max_possible = max(len(entities1), len(entities2))
        
        return overlap / max_possible if max_possible > 0 else 0
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate simple word overlap between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score (0-1)
        """
        # Tokenize and remove common stop words
        stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "is", "are"}
        words1 = {w.lower() for w in re.findall(r'\b\w+\b', text1) if w.lower() not in stop_words}
        words2 = {w.lower() for w in re.findall(r'\b\w+\b', text2) if w.lower() not in stop_words}
        
        if not words1 or not words2:
            return 0
            
        overlap = len(words1.intersection(words2))
        max_possible = max(len(words1), len(words2))
        
        return overlap / max_possible
    
    def _check_direct_references(self, text: str) -> float:
        """
        Check for direct references in text.
        
        Args:
            text: Input text
            
        Returns:
            Reference score (0-1)
        """
        # Check for patterns indicating references to previous questions
        reference_count = 0
        
        for pattern in self.reference_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                reference_count += 1
        
        # Normalize score
        return min(1.0, reference_count / 3)  # Cap at 1.0
    
    def get_relevant_context(self, current_question: str, max_items: int = 3) -> List[Dict[str, Any]]:
        """
        Get most relevant context from question history.
        
        Args:
            current_question: Current question text
            max_items: Maximum number of items to return
            
        Returns:
            List of relevant context items
        """
        dependencies = self.find_dependencies(current_question)
        
        # Take top N dependencies
        return dependencies[:max_items]
    
    def clear_history(self) -> None:
        """Clear question history."""
        self.question_history = []
