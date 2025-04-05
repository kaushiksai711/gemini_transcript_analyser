# gemini_mvp/caching.py - NOT ACTIVELY USED (likely superseded by context_cache.py)
# Older caching implementation

"""
Caching module (deprecated).

This module contains older caching implementation, likely superseded
by context_cache.py and cache_manager.py.
"""

import os
import time
import pickle
import logging
from typing import Dict, Any, Optional, Union
from cachetools import LRUCache
from diskcache import Cache

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages two-tier caching system.
    
    This class provides:
    1. Fast in-memory LRU cache for frequently accessed items
    2. Persistent disk cache for longer-term storage
    3. Automatic cache invalidation based on TTL
    """
    
    def __init__(
        self, 
        memory_size: int = 100, 
        disk_dir: str = ".cache", 
        ttl: int = 3600
    ):
        """
        Initialize the cache manager.
        
        Args:
            memory_size: Size of the in-memory LRU cache
            disk_dir: Directory for the disk cache
            ttl: Time-to-live for cached items in seconds
        """
        logger.info(f"Initializing cache manager: memory_size={memory_size}, disk_dir={disk_dir}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(disk_dir, exist_ok=True)
        
        # Initialize caches
        self.memory_cache = LRUCache(maxsize=memory_size)
        self.disk_cache = Cache(disk_dir)
        self.ttl = ttl
        
        # Track cache statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "sets": 0,
            "invalidations": 0
        }
    
    def get(self, key: str) -> Any:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            
            # Check if expired
            if expiry > time.time():
                self.stats["memory_hits"] += 1
                return value
            else:
                # Remove expired item
                del self.memory_cache[key]
                self.stats["invalidations"] += 1
        
        self.stats["memory_misses"] += 1
        
        # Check disk cache
        if key in self.disk_cache:
            try:
                value, expiry = self.disk_cache[key]
                
                # Check if expired
                if expiry > time.time():
                    # Promote to memory cache
                    self.memory_cache[key] = (value, expiry)
                    self.stats["disk_hits"] += 1
                    return value
                else:
                    # Remove expired item
                    del self.disk_cache[key]
                    self.stats["invalidations"] += 1
            except (pickle.PickleError, TypeError) as e:
                logger.warning(f"Error unpickling cached item: {e}")
                # Remove corrupt item
                del self.disk_cache[key]
        
        self.stats["disk_misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store item in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Optional custom TTL in seconds
        """
        if value is None:
            return
            
        # Use custom TTL or default
        ttl_value = ttl if ttl is not None else self.ttl
        expiry = time.time() + ttl_value
        
        # Store in both caches
        try:
            self.memory_cache[key] = (value, expiry)
            
            # Only store picklable objects in disk cache
            if self._is_picklable(value):
                self.disk_cache[key] = (value, expiry)
            
            self.stats["sets"] += 1
        except Exception as e:
            logger.warning(f"Error caching item: {e}")
    
    def invalidate(self, key: str) -> None:
        """
        Remove item from cache.
        
        Args:
            key: Cache key to invalidate
        """
        # Remove from both caches
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if key in self.disk_cache:
            del self.disk_cache[key]
        
        self.stats["invalidations"] += 1
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        # Calculate hit rates
        memory_requests = self.stats["memory_hits"] + self.stats["memory_misses"]
        disk_requests = self.stats["disk_hits"] + self.stats["disk_misses"]
        
        memory_hit_rate = (
            self.stats["memory_hits"] / max(1, memory_requests)
        )
        
        disk_hit_rate = (
            self.stats["disk_hits"] / max(1, disk_requests)
        )
        
        total_hit_rate = (
            (self.stats["memory_hits"] + self.stats["disk_hits"]) / 
            max(1, memory_requests)
        )
        
        # Return stats with hit rates
        return {
            **self.stats,
            "memory_hit_rate": memory_hit_rate,
            "disk_hit_rate": disk_hit_rate,
            "total_hit_rate": total_hit_rate,
            "memory_size": len(self.memory_cache),
            "disk_size": len(self.disk_cache)
        }
    
    def _is_picklable(self, obj: Any) -> bool:
        """
        Check if an object can be pickled.
        
        Args:
            obj: Object to check
            
        Returns:
            True if object can be pickled
        """
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PickleError, TypeError):
            return False
