# gemini_mvp/cache_manager.py - ACTIVELY USED
# Manages both memory and disk cache for optimal performance

"""
Cache Manager module for handling different cache levels.

This module provides a unified interface for both memory and disk caching,
with configurable TTL, size limits, and eviction policies.
"""

import os
import json
import pickle
import hashlib
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Two-tier caching system for efficient data storage and retrieval.
    
    Features:
    - In-memory LRU cache for fast access
    - Persistent disk cache for session state
    - Configurable TTL (time-to-live) for cached items
    - Cache invalidation strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the cache manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Cache configuration
        self.memory_cache_size = self.config.get("memory_cache_size", 1000)
        self.disk_cache_dir = self.config.get("disk_cache_dir", "cache")
        self.default_ttl = self.config.get("default_ttl", 3600)  # 1 hour default
        self.chunk_ttl = self.config.get("chunk_ttl", 86400)  # 24 hours for chunks
        self.response_ttl = self.config.get("response_ttl", 43200)  # 12 hours for API responses
        
        # Ensure cache directory exists
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        # Initialize in-memory cache
        self._init_memory_cache()
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "total_requests": 0
        }
    
    def _init_memory_cache(self):
        """Initialize the memory cache with LRU functionality."""
        # Create separate dictionaries to store cache data
        self.chunk_cache = {}
        self.response_cache = {}
        self.embedding_cache = {}
        self.context_cache = {}
        
        # Track LRU order for each cache
        self.chunk_lru = []
        self.response_lru = []
        self.embedding_lru = []
        self.context_lru = []
        
    def _create_lru_cache(self):
        """Create an LRU cache with the configured size."""
        # This function is kept for backward compatibility
        # We're implementing LRU manually instead of using the decorator
        # which was causing recursion issues
        return None
        
    def _get_cache_for_type(self, cache_type: str):
        """
        Get the appropriate cache object for the given type.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            Cache object and LRU list
        """
        if cache_type == "chunk":
            return self.chunk_cache, self.chunk_lru
        elif cache_type == "response":
            return self.response_cache, self.response_lru
        elif cache_type == "embedding":
            return self.embedding_cache, self.embedding_lru
        elif cache_type == "context":
            return self.context_cache, self.context_lru
        else:
            # Default to response cache
            return self.response_cache, self.response_lru
            
    def _update_memory_cache(self, key: str, value: Any, cache_type: str):
        """
        Update the in-memory LRU cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to use
        """
        cache_dict, lru_list = self._get_cache_for_type(cache_type)
        
        # Add or update the key in the cache
        cache_dict[key] = value
        
        # Update LRU order - remove if exists and add to end (most recently used)
        if key in lru_list:
            lru_list.remove(key)
        lru_list.append(key)
        
        # Enforce cache size limit
        while len(lru_list) > self.memory_cache_size:
            # Remove least recently used
            oldest_key = lru_list.pop(0)
            if oldest_key in cache_dict:
                del cache_dict[oldest_key]
            
    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache to use
            
        Returns:
            Cached value or None if not found
        """
        self.stats["total_requests"] += 1
        
        # First try memory cache
        try:
            cache_dict, lru_list = self._get_cache_for_type(cache_type)
            
            # Check if key is in memory cache
            if key in cache_dict:
                # Update LRU status
                if key in lru_list:
                    lru_list.remove(key)
                lru_list.append(key)
                
                self.stats["memory_hits"] += 1
                return cache_dict[key]
                
        except Exception as e:
            logger.warning(f"Error accessing memory cache: {e}")
        
        self.stats["memory_misses"] += 1
        
        # If not in memory, try disk cache
        try:
            disk_path = self._get_disk_path(key, cache_type)
            if os.path.exists(disk_path):
                with open(disk_path, 'rb') as f:
                    # Load the cached item with metadata
                    cached_item = pickle.load(f)
                    
                    # Check if the item is expired
                    if self._is_expired(cached_item):
                        # Remove expired item
                        os.remove(disk_path)
                        self.stats["disk_misses"] += 1
                        return None
                    
                    # Valid item found on disk
                    self.stats["disk_hits"] += 1
                    
                    # Store in memory cache for faster future access
                    value = cached_item["value"]
                    self._update_memory_cache(key, value, cache_type)
                    
                    return value
        except Exception as e:
            logger.warning(f"Error accessing disk cache: {e}")
            
        self.stats["disk_misses"] += 1
        return None
    
    def set(self, key: str, value: Any, cache_type: str = "general", ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to use
            ttl: Time-to-live in seconds
        """
        if ttl is None:
            # Use default TTL based on cache type
            if cache_type == "chunk":
                ttl = self.chunk_ttl
            elif cache_type == "response":
                ttl = self.response_ttl
            else:
                ttl = self.default_ttl
        
        # Update memory cache
        self._update_memory_cache(key, value, cache_type)
        
        # Update disk cache
        try:
            disk_path = self._get_disk_path(key, cache_type)
            with open(disk_path, 'wb') as f:
                # Store with metadata
                cached_item = {
                    "value": value,
                    "timestamp": datetime.now(),
                    "ttl": ttl
                }
                pickle.dump(cached_item, f)
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {e}")
    
    def _generate_key(self, data: Any) -> str:
        """
        Generate a cache key for the given data.
        
        Args:
            data: Data to generate key for
            
        Returns:
            Cache key string
        """
        # Use different strategies depending on data type
        if isinstance(data, str):
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        elif isinstance(data, dict):
            return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
        elif isinstance(data, (list, tuple)):
            return hashlib.md5(str(data).encode('utf-8')).hexdigest()
        else:
            return hashlib.md5(pickle.dumps(data)).hexdigest()
            
    def _get_disk_path(self, key: str, cache_type: str) -> str:
        """
        Get the disk path for a cache key.
        
        Args:
            key: Cache key
            cache_type: Type of cache
            
        Returns:
            Path to the cache file
        """
        # Create subdirectory for cache type
        cache_dir = os.path.join(self.disk_cache_dir, cache_type)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate filename for the cache item
        return os.path.join(cache_dir, f"{key}.cache")
    
    def _is_expired(self, cached_item: Dict[str, Any]) -> bool:
        """
        Check if a cached item is expired.
        
        Args:
            cached_item: Cached item with metadata
            
        Returns:
            True if expired, False otherwise
        """
        timestamp = cached_item.get("timestamp")
        ttl = cached_item.get("ttl", self.default_ttl)
        
        if timestamp is None:
            return True
            
        # Check if the TTL has passed
        expiry_time = timestamp + timedelta(seconds=ttl)
        return datetime.now() > expiry_time
    
    def invalidate(self, key: str, cache_type: str = "general") -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache
        """
        # Remove from disk cache
        disk_path = self._get_disk_path(key, cache_type)
        if os.path.exists(disk_path):
            try:
                os.remove(disk_path)
            except Exception as e:
                logger.warning(f"Error removing from disk cache: {e}")
        
        # Remove from memory cache
        cache_dict, lru_list = self._get_cache_for_type(cache_type)
        if key in cache_dict:
            del cache_dict[key]
        if key in lru_list:
            lru_list.remove(key)
    
    def invalidate_pattern(self, pattern: str, cache_type: str = "general") -> int:
        """
        Remove items matching a pattern from the cache.
        
        Args:
            pattern: Pattern to match
            cache_type: Type of cache
            
        Returns:
            Number of items removed
        """
        count = 0
        cache_dir = os.path.join(self.disk_cache_dir, cache_type)
        
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if pattern in filename:
                    try:
                        os.remove(os.path.join(cache_dir, filename))
                        count += 1
                    except Exception as e:
                        logger.warning(f"Error removing from disk cache: {e}")
        
        # Remove from memory cache
        cache_dict, lru_list = self._get_cache_for_type(cache_type)
        for key in list(cache_dict.keys()):
            if pattern in key:
                del cache_dict[key]
                if key in lru_list:
                    lru_list.remove(key)
        
        return count
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            cache_type: Type of cache to clear, or None for all
        """
        # Clear memory cache
        self._init_memory_cache()
        
        # Clear disk cache
        try:
            if cache_type:
                cache_dir = os.path.join(self.disk_cache_dir, cache_type)
                if os.path.exists(cache_dir):
                    for filename in os.listdir(cache_dir):
                        os.remove(os.path.join(cache_dir, filename))
            else:
                # Clear all cache types
                if os.path.exists(self.disk_cache_dir):
                    for dirpath, _, filenames in os.walk(self.disk_cache_dir):
                        for filename in filenames:
                            os.remove(os.path.join(dirpath, filename))
        except Exception as e:
            logger.warning(f"Error clearing disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        # Calculate hit ratios
        total_hits = self.stats["memory_hits"] + self.stats["disk_hits"]
        hit_ratio = total_hits / max(1, self.stats["total_requests"])
        
        # Add to stats
        self.stats["hit_ratio"] = hit_ratio
        self.stats["memory_hit_ratio"] = self.stats["memory_hits"] / max(1, self.stats["total_requests"])
        self.stats["disk_hit_ratio"] = self.stats["disk_hits"] / max(1, self.stats["total_requests"])
        
        # Get cache size info
        self.stats["cache_size"] = self._get_cache_size()
        
        return self.stats
    
    def _get_cache_size(self) -> Dict[str, int]:
        """
        Get the size of the cache.
        
        Returns:
            Dictionary with cache size information
        """
        sizes = {}
        
        # Get disk cache size
        if os.path.exists(self.disk_cache_dir):
            total_size = 0
            
            for dirpath, _, filenames in os.walk(self.disk_cache_dir):
                cache_type = os.path.basename(dirpath)
                cache_size = 0
                
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    cache_size += os.path.getsize(file_path)
                
                sizes[cache_type] = cache_size
                total_size += cache_size
            
            sizes["total"] = total_size
        
        return sizes
    
    def preload_cache(self, data_items: List[Tuple[str, Any, str]], show_progress: bool = False) -> None:
        """
        Preload multiple items into the cache.
        
        Args:
            data_items: List of (key, value, cache_type) tuples
            show_progress: Whether to show progress
        """
        total = len(data_items)
        
        for i, (key, value, cache_type) in enumerate(data_items):
            if show_progress and i % 10 == 0:
                logger.info(f"Preloading cache: {i}/{total}")
                
            self.set(key, value, cache_type)
