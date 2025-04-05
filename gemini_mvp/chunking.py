# gemini_mvp/chunking.py - ACTIVELY USED
# Handles transcript chunking for efficient processing

"""
Chunking module for breaking down long documents.

This module implements strategies for dividing transcripts into
meaningful, processable chunks with appropriate boundaries.
"""

import re
import time
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
import numpy as np

from .caching import CacheManager
from .utils import extract_entities, timestamp_to_seconds, seconds_to_timestamp

logger = logging.getLogger(__name__)

class Chunk:
    """Represents a chunk of transcript text with metadata."""
    
    def __init__(
        self,
        text: str,
        chunk_id: int,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None
    ):
        """
        Initialize a chunk.
        
        Args:
            text: The chunk text content
            chunk_id: Unique identifier for the chunk
            start_time: Starting timestamp (if available)
            end_time: Ending timestamp (if available)
            entities: List of key entities in the chunk
            topics: List of topics in the chunk
        """
        self.text = text
        self.chunk_id = chunk_id
        self.start_time = start_time
        self.end_time = end_time
        self.entities = entities or []
        self.topics = topics or []
        self.creation_time = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "entities": self.entities,
            "topics": self.topics,
            "creation_time": self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create a chunk from dictionary representation."""
        chunk = cls(
            text=data["text"],
            chunk_id=data["chunk_id"],
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            entities=data.get("entities"),
            topics=data.get("topics")
        )
        chunk.creation_time = data.get("creation_time", time.time())
        return chunk
    
    def __str__(self) -> str:
        """String representation of the chunk."""
        time_info = ""
        if self.start_time and self.end_time:
            time_info = f" ({self.start_time} - {self.end_time})"
        return f"Chunk {self.chunk_id}{time_info}: {self.text[:50]}..."


class ChunkManager:
    """
    Manages transcript chunking and processing.
    
    This class is responsible for:
    1. Splitting transcripts into optimal chunks
    2. Extracting timestamps and entities
    3. Caching processed chunks
    """
    
    def __init__(self, config: Dict[str, Any], cache_manager: CacheManager):
        """
        Initialize the chunk manager.
        
        Args:
            config: Configuration for chunking
            cache_manager: Cache manager instance
        """
        self.config = config
        self.cache = cache_manager
        self.chunks: List[Chunk] = []
        self.chunk_count = 0
        self.timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)')
        self._entity_cache: Dict[int, List[str]] = {}
        
    def process_transcript(self, transcript_text: str) -> List[Chunk]:
        """
        Process a transcript into chunks with metadata.
        
        Args:
            transcript_text: The raw transcript text
            
        Returns:
            List of processed chunks
        """
        logger.info("Processing transcript into chunks")
        
        # Check cache first
        cache_key = f"transcript_chunks_{hash(transcript_text)}"
        cached_chunks = self.cache.get(cache_key)
        if cached_chunks:
            logger.info("Using cached chunks")
            self.chunks = [Chunk.from_dict(c) for c in cached_chunks]
            self.chunk_count = len(self.chunks)
            return self.chunks
        
        # Extract timestamps if present
        timestamps = self._extract_timestamps(transcript_text)
        
        # Split transcript into chunks based on content and structure
        chunks = self._split_into_semantic_chunks(transcript_text)
        
        # Assign timestamps to chunks
        if timestamps:
            chunks = self._assign_timestamps_to_chunks(chunks, timestamps)
        
        # Extract entities and topics for each chunk
        for chunk in chunks:
            chunk.entities = self._get_entities_for_chunk(chunk)
            chunk.topics = self._extract_topics_for_chunk(chunk)
            
        # Store processed chunks
        self.chunks = chunks
        self.chunk_count = len(chunks)
        
        # Cache the results
        self.cache.set(
            cache_key, 
            [chunk.to_dict() for chunk in chunks]
        )
        
        logger.info(f"Transcript processed into {len(chunks)} chunks")
        return chunks
    
    def _split_into_semantic_chunks(self, text: str) -> List[Chunk]:
        """
        Split transcript into semantic chunks based on content and structure.
        
        Args:
            text: Transcript text
            
        Returns:
            List of chunk objects
        """
        # Check if transcript has timestamp markers
        has_timestamps = bool(self.timestamp_pattern.search(text))
        
        if has_timestamps:
            # If timestamps exist, split by timestamp sections first
            timestamp_chunks = self._split_by_timestamps(text)
            
            # If we got a reasonable number of chunks, use them
            if len(timestamp_chunks) >= 3:
                return timestamp_chunks
        
        # Try topic-based segmentation for longer transcripts
        if len(text.split()) > 1000:
            topic_chunks = self._split_by_topics(text)
            if topic_chunks and len(topic_chunks) >= 2:
                return topic_chunks
        
        # Fallback to simple size-based chunking
        return self._split_into_size_chunks(text)
    
    def _split_by_topics(self, text: str) -> List[Chunk]:
        """
        Split transcript into topic-based chunks.
        
        Args:
            text: Transcript text
            
        Returns:
            List of chunk objects
        """
        import re
        
        # Try to identify paragraph or section breaks
        paragraphs = re.split(r'\n\s*\n', text)
        
        # If text doesn't have clear paragraph breaks, create them
        if len(paragraphs) < 3:
            # Split by sentences
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_pattern, text)
            
            # Group sentences into paragraphs (approximately 3-5 sentences per paragraph)
            paragraphs = []
            current_paragraph = []
            for sentence in sentences:
                current_paragraph.append(sentence)
                if len(current_paragraph) >= 4:  # Adjust for desired paragraph size
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            # Add any remaining sentences
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
        
        # Now group paragraphs into topic-based chunks
        chunks = []
        chunk_id = 0
        current_paragraphs = []
        
        # Get config values
        max_paragraphs_per_chunk = self.config.get("max_paragraphs_per_chunk", 5)
        min_paragraphs_per_chunk = self.config.get("min_paragraphs_per_chunk", 2)
        
        for paragraph in paragraphs:
            current_paragraphs.append(paragraph)
            
            # Create a chunk when we reach the desired number of paragraphs
            if len(current_paragraphs) >= max_paragraphs_per_chunk:
                chunk_text = '\n\n'.join(current_paragraphs)
                chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id))
                chunk_id += 1
                
                # Keep the last paragraph for overlap
                current_paragraphs = [current_paragraphs[-1]]
        
        # Add any remaining paragraphs as a chunk
        if len(current_paragraphs) >= min_paragraphs_per_chunk:
            chunk_text = '\n\n'.join(current_paragraphs)
            chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id))
        
        return chunks
    
    def _split_into_size_chunks(self, text: str) -> List[Chunk]:
        """
        Split transcript text into chunks of appropriate size with overlap.
        
        Args:
            text: Transcript text
            
        Returns:
            List of chunk objects
        """
        # Get config values
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]
        
        # Split text into words to ensure we don't break in the middle of words
        words = text.split()
        
        if len(words) <= chunk_size:
            # If text is shorter than chunk size, return as single chunk
            return [Chunk(text=text, chunk_id=0)]
        
        chunks = []
        chunk_id = 0
        
        # Calculate word positions for chunks
        pos = 0
        while pos < len(words):
            # Calculate end position of current chunk
            end_pos = min(pos + chunk_size, len(words))
            
            # Create chunk from words
            chunk_text = " ".join(words[pos:end_pos])
            chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id))
            
            # Move position forward, accounting for overlap
            pos += chunk_size - chunk_overlap
            chunk_id += 1
            
            # Ensure we're making progress
            if pos >= len(words) - chunk_overlap:
                break
        
        return chunks
    
    def _split_by_timestamps(self, text: str) -> List[Chunk]:
        """
        Split transcript by timestamp sections with topic-based chunking.
        
        Args:
            text: Transcript text with timestamps
            
        Returns:
            List of chunks
        """
        # Extract timestamp positions
        timestamps = self._extract_timestamps(text)
        
        if not timestamps:
            # Fallback to regular chunking if no timestamps found
            return self._split_into_chunks(text)
            
        # Get config values
        min_chunk_size = self.config.get("min_timestamp_sections", 3)
        max_chunk_size = self.config.get("max_timestamp_sections", 6)
        
        # Create sections based on timestamps
        sections = []
        for i, (timestamp, pos) in enumerate(timestamps):
            # Find the text for this section (until next timestamp or end)
            if i < len(timestamps) - 1:
                next_pos = timestamps[i+1][1]
                section_text = text[pos:next_pos]
            else:
                section_text = text[pos:]
                
            sections.append((timestamp, section_text))
        
        # Now group sections into semantically related chunks
        chunks = []
        chunk_id = 0
        current_sections = []
        current_text = ""
        current_start_time = None
        current_end_time = None
        
        for timestamp, section_text in sections:
            current_sections.append((timestamp, section_text))
            
            # Set start time if this is the first section
            if current_start_time is None:
                current_start_time = timestamp
                
            # Update end time to current timestamp
            current_end_time = timestamp
            
            # If we've reached the desired number of sections, create a chunk
            if len(current_sections) >= max_chunk_size or len(current_text) > 2000:
                # Join all sections
                current_text = "\n".join([s[1] for s in current_sections])
                
                # Create chunk
                chunk = Chunk(
                    text=current_text,
                    chunk_id=chunk_id,
                    start_time=current_start_time,
                    end_time=current_end_time
                )
                chunks.append(chunk)
                
                # Reset for next chunk, but keep overlap by retaining the last section
                if current_sections:
                    # Keep last section for overlap
                    current_sections = [current_sections[-1]]
                    current_text = current_sections[0][1]
                    current_start_time = current_sections[0][0]
                    current_end_time = current_sections[0][0]
                else:
                    current_sections = []
                    current_text = ""
                    current_start_time = None
                    current_end_time = None
                    
                chunk_id += 1
        
        # Add any remaining sections
        if current_sections and (len(current_sections) >= min_chunk_size or not chunks):
            current_text = "\n".join([s[1] for s in current_sections])
            chunk = Chunk(
                text=current_text,
                chunk_id=chunk_id,
                start_time=current_start_time,
                end_time=current_end_time
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_timestamps(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract timestamps from transcript text.
        
        Args:
            text: Transcript text
            
        Returns:
            List of (timestamp, character_position) tuples
        """
        timestamps = []
        for match in self.timestamp_pattern.finditer(text):
            timestamps.append((match.group(1), match.start()))
        return timestamps
    
    def _assign_timestamps_to_chunks(
        self, 
        chunks: List[Chunk], 
        timestamps: List[Tuple[str, int]]
    ) -> List[Chunk]:
        """
        Assign timestamps to chunks based on text positions.
        
        Args:
            chunks: List of chunks
            timestamps: List of (timestamp, position) tuples
            
        Returns:
            Chunks with timestamps assigned
        """
        if not timestamps:
            return chunks
        
        # Track current position in text
        current_pos = 0
        timestamp_idx = 0
        
        for i, chunk in enumerate(chunks):
            chunk_start_pos = current_pos
            chunk_end_pos = chunk_start_pos + len(chunk.text)
            
            # Find timestamps within this chunk
            chunk_timestamps = []
            while (timestamp_idx < len(timestamps) and 
                   timestamps[timestamp_idx][1] <= chunk_end_pos):
                chunk_timestamps.append(timestamps[timestamp_idx])
                timestamp_idx += 1
            
            # Assign start and end timestamps
            if chunk_timestamps:
                chunk.start_time = chunk_timestamps[0][0]
                chunk.end_time = chunk_timestamps[-1][0]
            elif i > 0 and chunks[i-1].end_time:
                # If no timestamps in this chunk, use previous chunk's end time
                chunk.start_time = chunks[i-1].end_time
            
            # Update current position
            current_pos = chunk_end_pos
        
        return chunks
    
    def _get_entities_for_chunk(self, chunk: Chunk) -> List[str]:
        """
        Extract key entities from a chunk.
        
        Args:
            chunk: The chunk to process
            
        Returns:
            List of entities
        """
        # Check entity cache first
        if chunk.chunk_id in self._entity_cache:
            return self._entity_cache[chunk.chunk_id]
        
        # Extract entities
        entities = extract_entities(chunk.text)
        
        # Cache entities
        self._entity_cache[chunk.chunk_id] = entities
        
        return entities
    
    def _extract_topics_for_chunk(self, chunk: Chunk) -> List[str]:
        """
        Extract main topics from a chunk.
        
        Args:
            chunk: Chunk to extract topics from
            
        Returns:
            List of main topics
        """
        from collections import Counter
        import re
        
        # Skip if no text
        if not chunk.text:
            return []
        
        # Common stop words to exclude
        stop_words = set([
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "who", "whom", "whose", "where", "when", "why", "how", "is", 
            "are", "was", "were", "be", "been", "being", "have", "has", "had", 
            "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "in", "on", "at", "to", "for", "with", "about"
        ])
        
        # Extract words, remove punctuation and stop words
        words = re.findall(r'\b\w{4,}\b', chunk.text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get the most common words
        return [word for word, _ in word_counts.most_common(5)]
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Chunk]:
        """
        Retrieve a chunk by its ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            The chunk or None if not found
        """
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_entity(self, entity: str) -> List[Chunk]:
        """
        Find chunks containing a specific entity.
        
        Args:
            entity: Entity to search for
            
        Returns:
            List of chunks containing the entity
        """
        return [
            chunk for chunk in self.chunks
            if entity in chunk.entities
        ]
    
    def get_chunks_by_time_range(
        self, 
        start_time: str, 
        end_time: str
    ) -> List[Chunk]:
        """
        Find chunks within a time range.
        
        Args:
            start_time: Start timestamp (HH:MM:SS)
            end_time: End timestamp (HH:MM:SS)
            
        Returns:
            List of chunks in the time range
        """
        start_seconds = timestamp_to_seconds(start_time)
        end_seconds = timestamp_to_seconds(end_time)
        
        result = []
        for chunk in self.chunks:
            if not chunk.start_time or not chunk.end_time:
                continue
                
            chunk_start = timestamp_to_seconds(chunk.start_time)
            chunk_end = timestamp_to_seconds(chunk.end_time)
            
            # Check for overlap
            if not (chunk_end < start_seconds or chunk_start > end_seconds):
                result.append(chunk)
                
        return result
