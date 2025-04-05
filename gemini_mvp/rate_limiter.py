# gemini_mvp/rate_limiter.py - ACTIVELY USED
# Used by API client to manage request rates and prevent throttling

"""
Rate Limiter module for API request throttling.

This module provides rate limiting functionality to prevent hitting API rate limits
and ensures request spacing for optimal API usage.
"""

import time
import logging
import threading
from typing import Dict, Any
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter for API requests.
    
    This class implements a token bucket algorithm for rate limiting,
    tracking both per-minute and daily request limits.
    """
    
    def __init__(self, rpm: int = 10, rpd: int = 1500):
        """
        Initialize the rate limiter.
        
        Args:
            rpm: Requests per minute limit (default: 10)
            rpd: Requests per day limit (default: 1500)
        """
        self.rpm = rpm
        self.rpd = rpd
        
        # Calculate minimum time between requests in seconds
        self.min_request_interval = 60.0 / max(1, rpm)  # Minimum time between requests
        
        # Initialize counters
        self.minute_request_times = deque(maxlen=rpm)
        self.day_request_times = deque(maxlen=rpd)
        
        # Last request timestamp
        self.last_request_time = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Rate limiter initialized with {rpm} RPM, {rpd} RPD, {self.min_request_interval:.2f}s between requests")
    
    def wait_for_token(self) -> float:
        """
        Wait for a rate limit token if needed.
        
        Returns:
            Time waited in seconds
        """
        with self.lock:
            now = time.time()
            
            # Wait at least the minimum interval since last request
            elapsed_since_last = now - self.last_request_time
            if elapsed_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed_since_last
                logger.debug(f"Waiting {wait_time:.2f}s to maintain minimum interval between requests")
                return wait_time
            
            # Check minute-based rate limit
            if len(self.minute_request_times) >= self.rpm:
                # Get the oldest request in our window
                oldest_time = self.minute_request_times[0]
                time_since_oldest = now - oldest_time
                
                # If we haven't waited a full minute since the oldest request
                if time_since_oldest < 60.0:
                    # Calculate how much longer to wait
                    wait_time = 60.0 - time_since_oldest
                    logger.info(f"Rate limit reached: {self.rpm} requests/minute. Waiting {wait_time:.2f}s")
                    return wait_time
            
            # Check day-based rate limit
            if len(self.day_request_times) >= self.rpd:
                # Get the oldest request in our day window
                oldest_time = self.day_request_times[0]
                time_since_oldest = now - oldest_time
                
                # If we haven't waited a full day
                if time_since_oldest < 86400.0:  # 24 hours in seconds
                    # Calculate how much longer to wait
                    wait_time = 86400.0 - time_since_oldest
                    logger.info(f"Daily rate limit reached: {self.rpd} requests/day. Waiting {wait_time:.2f}s")
                    return wait_time
            
            # Record this request
            self.minute_request_times.append(now)
            self.day_request_times.append(now)
            self.last_request_time = now
            
            return 0.0
    
    def get_quota_status(self) -> Dict[str, Any]:
        """
        Get current quota status.
        
        Returns:
            Dictionary with current quota information
        """
        now = time.time()
        
        # Calculate minute quota
        minute_requests = len(self.minute_request_times)
        minute_remaining = self.rpm - minute_requests
        
        # Calculate when the next token will be available
        next_token_time = 0
        if minute_requests >= self.rpm and self.minute_request_times:
            oldest = self.minute_request_times[0]
            next_token_time = oldest + 60.0
        
        # Calculate day quota
        day_requests = len(self.day_request_times)
        day_remaining = self.rpd - day_requests
        
        # Calculate wait time
        wait_time = 0
        if next_token_time > now:
            wait_time = next_token_time - now
        elif minute_requests > 0:
            # Ensure minimum spacing between requests
            elapsed_since_last = now - self.last_request_time
            if elapsed_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed_since_last
        
        return {
            "minute": {
                "limit": self.rpm,
                "used": minute_requests,
                "remaining": minute_remaining,
                "reset_in": 60.0 - (now - self.minute_request_times[0]) if self.minute_request_times else 0
            },
            "day": {
                "limit": self.rpd,
                "used": day_requests,
                "remaining": day_remaining,
                "reset_in": 86400.0 - (now - self.day_request_times[0]) if self.day_request_times else 0
            },
            "status": {
                "can_request": wait_time == 0,
                "wait_time": wait_time,
                "next_request_at": datetime.fromtimestamp(now + wait_time).isoformat() if wait_time > 0 else None,
                "min_interval": self.min_request_interval
            }
        }
