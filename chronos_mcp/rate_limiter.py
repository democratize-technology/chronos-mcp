"""
Rate limiting implementation for Chronos MCP server.

Uses token bucket algorithm for smooth rate limiting with burst capacity.
"""

import asyncio
import os
import time
from collections import defaultdict
from functools import wraps
from typing import Dict, Optional, Tuple

from .exceptions import ChronosError
from .logging_config import setup_logging

logger = setup_logging()


class RateLimitExceeded(ChronosError):
    """Raised when rate limit is exceeded"""

    def __init__(
        self,
        message: str,
        retry_after: float,
        limit: int,
        window: float,
        endpoint: str,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
        self.window = window
        self.endpoint = endpoint
        self.status_code = 429


class TokenBucket:
    """Token bucket implementation for rate limiting with burst capacity"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst capacity)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            Tuple of (success, wait_time_if_failed)
        """
        async with self.lock:
            # Refill tokens based on time elapsed
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # Calculate wait time until enough tokens are available
                tokens_needed = tokens - self.tokens
                # If refill rate is 0, tokens will never refill
                if self.refill_rate <= 0:
                    wait_time = float("inf")
                else:
                    wait_time = tokens_needed / self.refill_rate
                return False, wait_time

    async def get_available_tokens(self) -> float:
        """Get current available tokens"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            return min(self.capacity, self.tokens + elapsed * self.refill_rate)


class RateLimiter:
    """Rate limiter for MCP endpoints"""

    # Default rate limits (requests per minute)
    DEFAULT_LIMITS = {
        "accounts": 10,  # Account operations
        "calendars": 60,  # Calendar operations
        "events": 120,  # Event operations
        "bulk": 5,  # Bulk operations
        "tasks": 60,  # Task operations
        "journals": 60,  # Journal operations
        "search": 30,  # Search operations
        "rrule": 100,  # RRule operations
    }

    def __init__(self, enabled: Optional[bool] = None):
        """
        Initialize rate limiter.

        Args:
            enabled: Whether rate limiting is enabled. If None, checks env var.
        """
        if enabled is None:
            enabled = os.getenv("CHRONOS_RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.enabled = enabled

        # Load limits from environment or use defaults
        self.limits = {}
        for category, default in self.DEFAULT_LIMITS.items():
            env_key = f"CHRONOS_RATE_LIMIT_{category.upper()}"
            limit = int(os.getenv(env_key, str(default)))
            self.limits[category] = limit

        # Create token buckets for each category
        # Convert per-minute limits to per-second refill rates
        self.buckets: Dict[str, TokenBucket] = {}
        for category, limit in self.limits.items():
            # Capacity = limit (allows burst up to full limit)
            # Refill rate = limit/60 tokens per second
            self.buckets[category] = TokenBucket(
                capacity=limit, refill_rate=limit / 60.0
            )

        # Statistics tracking
        self.stats = defaultdict(lambda: {"allowed": 0, "rejected": 0})

        logger.info(
            f"Rate limiter initialized (enabled={self.enabled}, limits={self.limits})"
        )

    async def check_rate_limit(self, category: str) -> Tuple[bool, float]:
        """
        Check if request is within rate limit.

        Args:
            category: Rate limit category

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0.0

        if category not in self.buckets:
            logger.warning(f"Unknown rate limit category: {category}, allowing request")
            return True, 0.0

        bucket = self.buckets[category]
        allowed, wait_time = await bucket.consume()

        # Update statistics
        if allowed:
            self.stats[category]["allowed"] += 1
        else:
            self.stats[category]["rejected"] += 1

        return allowed, wait_time

    async def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get rate limiting statistics"""
        return dict(self.stats)

    async def reset_stats(self):
        """Reset statistics"""
        self.stats.clear()

    async def get_available_capacity(self, category: str) -> Optional[float]:
        """
        Get available capacity for a category.

        Args:
            category: Rate limit category

        Returns:
            Available tokens or None if category doesn't exist
        """
        if category not in self.buckets:
            return None
        return await self.buckets[category].get_available_tokens()


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(category: str):
    """
    Decorator to apply rate limiting to MCP tool endpoints.

    Args:
        category: Rate limit category (e.g., 'accounts', 'events')
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            allowed, retry_after = await limiter.check_rate_limit(category)

            if not allowed:
                limit = limiter.limits.get(category, 0)
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {category}. "
                    f"Limit: {limit} requests per minute. "
                    f"Please retry after {retry_after:.1f} seconds.",
                    retry_after=retry_after,
                    limit=limit,
                    window=60.0,
                    endpoint=category,
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def disable_rate_limiting():
    """Disable rate limiting globally (useful for tests)"""
    global _rate_limiter
    _rate_limiter = RateLimiter(enabled=False)


def enable_rate_limiting():
    """Enable rate limiting globally"""
    global _rate_limiter
    _rate_limiter = RateLimiter(enabled=True)


def reset_rate_limiter():
    """Reset the global rate limiter instance"""
    global _rate_limiter
    _rate_limiter = None
