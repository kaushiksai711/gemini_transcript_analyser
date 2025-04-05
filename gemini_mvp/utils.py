# gemini_mvp/utils.py - ACTIVELY USED
# Utility functions used across multiple components

"""
Utilities module with helper functions.

This module provides common utilities used across the system, including
formatting, conversion, and helper functions.
"""

import os
import re
import yaml
import logging
import time
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce verbosity of some loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "transcript": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            
        },
        "context": {
            "weights": {
                "recency": 0.3,
                "relevance": 0.5,
                "entity_overlap": 0.2
            },
            "history_limit": 5,
            "max_context_chunks": 10
        },
        "cache": {
            "memory_cache_size": 100,
            "disk_cache_dir": ".cache",
            "cache_ttl": 3600
        },
        "api": {
            "model": "gemini-1.5-pro",
            "max_retries": 3,
            "retry_delay": 2,
            "temperature": 0.2,
            "max_output_tokens": 1024
        },
        "dependency": {
            "similarity_threshold": 0.65,
            "keyword_threshold": 0.4,
            "confidence_blend": 0.7
        }
    }
    
    # Try to load configuration file
    config = default_config
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Deep merge configurations
            config = deep_merge(default_config, file_config)
    except Exception as e:
        logging.warning(f"Failed to load configuration file: {e}")
        logging.info("Using default configuration")
    
    return config

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # If both values are dictionaries, merge them recursively
            result[key] = deep_merge(result[key], value)
        else:
            # Otherwise, just take the value from dict2
            result[key] = value
    
    return result

def extract_entities(text: str) -> List[str]:
    """
    Extract entities from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of entities
    """
    # Simple implementation: extract capitalized words and multi-word phrases
    entity_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    
    matches = re.findall(entity_pattern, text)
    
    # Filter out common non-entity capitalized words
    common_words = {"I", "The", "A", "An", "This", "That", "These", "Those", "It"}
    entities = [match for match in matches if match not in common_words]
    
    return entities

def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert a timestamp to seconds.
    
    Args:
        timestamp: Timestamp in HH:MM:SS or HH:MM:SS.MS format
        
    Returns:
        Timestamp in seconds
    """
    parts = timestamp.split(':')
    
    # Handle milliseconds
    seconds_parts = parts[2].split('.')
    seconds = float(seconds_parts[0])
    if len(seconds_parts) > 1:
        seconds += float('0.' + seconds_parts[1])
    
    # Add minutes and hours
    seconds += int(parts[1]) * 60
    seconds += int(parts[0]) * 3600
    
    return seconds

def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to timestamp.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Timestamp in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def format_response_as_json(response: Dict[str, Any]) -> str:
    """
    Format a response as JSON.
    
    Args:
        response: Response dictionary
        
    Returns:
        JSON string
    """
    return json.dumps(response, indent=2)

def format_response_as_yaml(response: Dict[str, Any]) -> str:
    """
    Format a response as YAML.
    
    Args:
        response: Response dictionary
        
    Returns:
        YAML string
    """
    return yaml.dump(response, sort_keys=False)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def generate_timestamp_str() -> str:
    """
    Generate a timestamp string for filenames.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def smart_truncate(text: str, max_length: int = 100) -> str:
    """
    Truncate text while preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
        
    # Find the last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."
