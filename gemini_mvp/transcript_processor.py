# gemini_mvp/transcript_processor.py - ACTIVELY USED
# Core component for processing transcript data

"""
Transcript Processor module for handling document input.

This module parses transcripts, extracts metadata, and prepares
content for chunking and context selection.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import re
import yaml

from .chunking import ChunkManager
from .query_processor import QueryProcessor
from .context_manager import ContextManager
from .api_client import GeminiClient
from .caching import CacheManager
from .utils import load_config, setup_logging

logger = logging.getLogger(__name__)

class TranscriptProcessor:
    """
    Main class for processing video transcripts and handling questions.
    
    This class coordinates all components of the system including chunking,
    context management, query processing, and API integration.
    """
    
    def __init__(
        self, 
        transcript_path: Optional[str] = None, 
        transcript_text: Optional[str] = None,
        config_path: str = "config.yaml",
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the transcript processor.
        
        Args:
            transcript_path: Path to the transcript file (if loading from file)
            transcript_text: Raw transcript text (if providing directly)
            config_path: Path to configuration file
            api_key: Google API key (if not set in environment)
            cache_dir: Directory for persistent cache
            verbose: Enable verbose logging
        """
        # Setup logging
        setup_logging(verbose)
        logger.info("Initializing TranscriptProcessor")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup components
        self.cache_manager = CacheManager(
            memory_size=self.config["cache"]["memory_cache_size"],
            disk_dir=cache_dir or self.config["cache"]["disk_cache_dir"],
            ttl=self.config["cache"]["cache_ttl"]
        )
        
        self.api_client = GeminiClient(
            api_key=api_key,
            config=self.config["api"]
        )
        
        self.context_manager = ContextManager(
            config=self.config["context"],
            cache_manager=self.cache_manager
        )
        
        self.query_processor = QueryProcessor(
            config=self.config["dependency"],
            context_manager=self.context_manager
        )
        
        self.chunk_manager = ChunkManager(
            config=self.config["transcript"],
            cache_manager=self.cache_manager
        )
        
        # Process transcript if provided
        self.transcript_loaded = False
        if transcript_path:
            self.load_transcript_from_file(transcript_path)
        elif transcript_text:
            self.load_transcript_from_text(transcript_text)
    
    def load_transcript_from_file(self, file_path: str) -> bool:
        """
        Load and process a transcript from a file.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            bool: Success status
        """
        logger.info(f"Loading transcript from file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            return self.load_transcript_from_text(transcript_text)
        except Exception as e:
            logger.error(f"Failed to load transcript file: {e}")
            return False
    
    def load_transcript_from_text(self, text: str) -> bool:
        """
        Load and process a transcript from text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            bool: Success status
        """
        logger.info("Processing transcript text")
        try:
            self.chunk_manager.process_transcript(text)
            self.transcript_loaded = True
            logger.info(f"Transcript processed into {self.chunk_manager.chunk_count} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to process transcript: {e}")
            return False
    
    def ask(self, question: str, include_context: bool = True) -> Dict[str, Any]:
        """
        Process a question against the loaded transcript.
        
        Args:
            question: The question to answer
            include_context: Whether to include context from previous questions
            
        Returns:
            Dict containing the response and metadata
        """
        if not self.transcript_loaded:
            return {"error": "No transcript loaded", "success": False}
        
        logger.info(f"Processing question: {question}")
        
        # Step 1: Process the query and identify dependencies
        query_info = self.query_processor.process_query(question)
        
        # Add available chunks to query_info
        query_info["available_chunks"] = self.chunk_manager.chunks
        
        # Step 2: Retrieve relevant context chunks
        context_chunks = self.context_manager.get_context(
            query=question,
            query_info=query_info
        )
        
        # Step 3: Call Gemini API with context and question
        response = self.api_client.generate_response(
            question=question,
            context_chunks=context_chunks,
            query_info=query_info
        )
        
        # Step 4: Update conversation history
        self.context_manager.update_history(
            question=question,
            response=response,
            query_info=query_info
        )
        
        return response
    
    def batch_process(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of questions optimally.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        # Group questions by dependencies
        batched_questions = self.query_processor.batch_questions(questions)
        
        responses = []
        for batch in batched_questions:
            batch_responses = []
            for question in batch:
                response = self.ask(question)
                batch_responses.append(response)
            responses.extend(batch_responses)
        
        return responses
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        logger.info("Resetting conversation history")
        self.context_manager.reset_history()
    
    def get_quota_status(self) -> Dict[str, Any]:
        """
        Get the current API quota status.
        
        Returns:
            Dictionary with quota information
        """
        logger.info("Checking API quota status")
        return self.api_client.get_quota_status()
        
    def export_session(self, file_path: str) -> bool:
        """
        Export the current session to a file.
        
        Args:
            file_path: Path to save the session
            
        Returns:
            bool: Success status
        """
        try:
            session_data = {
                "history": [item.to_dict() for item in self.context_manager.conversation_history],
                "metadata": {
                    "chunk_count": self.chunk_manager.chunk_count,
                    "timestamp": self.context_manager.last_update_time
                }
            }
            print(session_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            print('done')
            logger.info(f"Session exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export session: {e}")
            return False
    
    def import_session(self, file_path: str) -> bool:
        """
        Import a previously saved session.
        
        Args:
            file_path: Path to the session file
            
        Returns:
            bool: Success status
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.context_manager.conversation_history = session_data["history"]
            self.context_manager.last_update_time = session_data["metadata"]["timestamp"]
            
            logger.info(f"Session imported from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return False
