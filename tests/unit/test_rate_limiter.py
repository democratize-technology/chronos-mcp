"""
Unit tests for rate limiting functionality
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chronos_mcp.rate_limiter import (
    RateLimitExceeded,
    RateLimiter,
    TokenBucket,
    disable_rate_limiting,
    enable_rate_limiting,
    get_rate_limiter,
    rate_limit,
    reset_rate_limiter,
)


class TestTokenBucket:
    """Test token bucket algorithm"""

    @pytest.mark.asyncio
    async def test_token_bucket_initialization(self):
        """Test token bucket initializes with correct capacity and refill rate"""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 5.0
        assert bucket.tokens == 10.0

    @pytest.mark.asyncio
    async def test_token_consumption_success(self):
        """Test successful token consumption"""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        success, wait_time = await bucket.consume(3)
        assert success is True
        assert wait_time == 0.0
        assert await bucket.get_available_tokens() < 10

    @pytest.mark.asyncio
    async def test_token_consumption_failure(self):
        """Test failed token consumption when insufficient tokens"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        # Consume all tokens
        await bucket.consume(5)
        # Try to consume more
        success, wait_time = await bucket.consume(3)
        assert success is False
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test tokens refill over time"""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens per second
        # Consume all tokens
        await bucket.consume(10)
        available = await bucket.get_available_tokens()
        assert available < 1

        # Wait for refill
        await asyncio.sleep(0.5)  # Should refill ~5 tokens
        available = await bucket.get_available_tokens()
        assert 4 <= available <= 6  # Allow some timing variance

    @pytest.mark.asyncio
    async def test_token_capacity_limit(self):
        """Test tokens don't exceed capacity"""
        bucket = TokenBucket(capacity=5, refill_rate=100.0)  # Fast refill
        await asyncio.sleep(0.1)  # Wait for potential overflow
        available = await bucket.get_available_tokens()
        assert available <= 5

    @pytest.mark.asyncio
    async def test_concurrent_token_consumption(self):
        """Test thread-safe concurrent token consumption"""
        bucket = TokenBucket(capacity=100, refill_rate=0)  # No refill for test

        async def consume_tokens():
            for _ in range(10):
                await bucket.consume(1)

        # Run 10 concurrent consumers
        tasks = [consume_tokens() for _ in range(10)]
        await asyncio.gather(*tasks)

        available = await bucket.get_available_tokens()
        assert available == 0  # All 100 tokens should be consumed


class TestRateLimiter:
    """Test RateLimiter class"""

    def test_rate_limiter_initialization_enabled(self):
        """Test rate limiter initializes as enabled by default"""
        with patch.dict(os.environ, {"CHRONOS_RATE_LIMIT_ENABLED": "true"}):
            limiter = RateLimiter()
            assert limiter.enabled is True
            assert "accounts" in limiter.limits
            assert "events" in limiter.limits

    def test_rate_limiter_initialization_disabled(self):
        """Test rate limiter can be disabled via environment"""
        with patch.dict(os.environ, {"CHRONOS_RATE_LIMIT_ENABLED": "false"}):
            limiter = RateLimiter()
            assert limiter.enabled is False

    def test_rate_limiter_custom_limits(self):
        """Test custom rate limits from environment variables"""
        with patch.dict(
            os.environ,
            {
                "CHRONOS_RATE_LIMIT_ACCOUNTS": "20",
                "CHRONOS_RATE_LIMIT_EVENTS": "200",
            },
        ):
            limiter = RateLimiter()
            assert limiter.limits["accounts"] == 20
            assert limiter.limits["events"] == 200

    @pytest.mark.asyncio
    async def test_check_rate_limit_when_disabled(self):
        """Test rate limiting passes through when disabled"""
        limiter = RateLimiter(enabled=False)
        allowed, wait_time = await limiter.check_rate_limit("accounts")
        assert allowed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_rate_limit_unknown_category(self):
        """Test unknown categories are allowed"""
        limiter = RateLimiter(enabled=True)
        allowed, wait_time = await limiter.check_rate_limit("unknown_category")
        assert allowed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_rate_limit_success(self):
        """Test successful rate limit check"""
        limiter = RateLimiter(enabled=True)
        # Set up a bucket with sufficient tokens
        limiter.buckets["accounts"] = TokenBucket(capacity=10, refill_rate=0)
        allowed, wait_time = await limiter.check_rate_limit("accounts")
        assert allowed is True
        assert wait_time == 0.0
        assert limiter.stats["accounts"]["allowed"] == 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_rejection(self):
        """Test rate limit rejection"""
        limiter = RateLimiter(enabled=True)
        # Set up a bucket with no tokens
        bucket = TokenBucket(capacity=1, refill_rate=0)
        await bucket.consume(1)  # Consume the only token
        limiter.buckets["accounts"] = bucket

        allowed, wait_time = await limiter.check_rate_limit("accounts")
        assert allowed is False
        assert wait_time > 0
        assert limiter.stats["accounts"]["rejected"] == 1

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test statistics are tracked correctly"""
        limiter = RateLimiter(enabled=True)
        # Make some successful requests
        for _ in range(3):
            await limiter.check_rate_limit("events")

        stats = await limiter.get_stats()
        assert stats["events"]["allowed"] >= 3

    @pytest.mark.asyncio
    async def test_reset_statistics(self):
        """Test statistics can be reset"""
        limiter = RateLimiter(enabled=True)
        await limiter.check_rate_limit("accounts")
        await limiter.reset_stats()
        stats = await limiter.get_stats()
        assert len(stats) == 0

    @pytest.mark.asyncio
    async def test_get_available_capacity(self):
        """Test getting available capacity for a category"""
        limiter = RateLimiter(enabled=True)
        capacity = await limiter.get_available_capacity("accounts")
        assert capacity > 0
        assert capacity <= limiter.limits["accounts"]

    @pytest.mark.asyncio
    async def test_get_available_capacity_unknown_category(self):
        """Test getting capacity for unknown category returns None"""
        limiter = RateLimiter(enabled=True)
        capacity = await limiter.get_available_capacity("unknown")
        assert capacity is None


class TestRateLimitDecorator:
    """Test rate_limit decorator"""

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_allows_request(self):
        """Test decorator allows requests within limit"""
        reset_rate_limiter()
        enable_rate_limiting()

        @rate_limit("accounts")
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_raises_exception(self):
        """Test decorator raises exception when limit exceeded"""
        reset_rate_limiter()
        enable_rate_limiting()

        # Create a limiter with no available tokens
        limiter = get_rate_limiter()
        bucket = TokenBucket(capacity=1, refill_rate=0)
        await bucket.consume(1)
        limiter.buckets["accounts"] = bucket

        @rate_limit("accounts")
        async def test_function():
            return "success"

        with pytest.raises(RateLimitExceeded) as exc_info:
            await test_function()

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.retry_after > 0
        assert exc_info.value.endpoint == "accounts"

    @pytest.mark.asyncio
    async def test_rate_limit_decorator_with_disabled_limiter(self):
        """Test decorator allows all requests when rate limiting is disabled"""
        reset_rate_limiter()
        disable_rate_limiting()

        @rate_limit("accounts")
        async def test_function():
            return "success"

        # Should work even with many rapid calls
        for _ in range(100):
            result = await test_function()
            assert result == "success"


class TestRateLimitException:
    """Test RateLimitExceeded exception"""

    def test_rate_limit_exceeded_exception_attributes(self):
        """Test exception has correct attributes"""
        exc = RateLimitExceeded(
            message="Test message",
            retry_after=5.0,
            limit=10,
            window=60.0,
            endpoint="test",
        )
        assert "Test message" in str(exc)
        assert exc.retry_after == 5.0
        assert exc.limit == 10
        assert exc.window == 60.0
        assert exc.endpoint == "test"
        assert exc.status_code == 429


class TestGlobalRateLimiter:
    """Test global rate limiter functions"""

    def test_get_rate_limiter_singleton(self):
        """Test get_rate_limiter returns singleton instance"""
        reset_rate_limiter()
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_disable_rate_limiting_global(self):
        """Test global disable function"""
        reset_rate_limiter()
        disable_rate_limiting()
        limiter = get_rate_limiter()
        assert limiter.enabled is False

    def test_enable_rate_limiting_global(self):
        """Test global enable function"""
        reset_rate_limiter()
        enable_rate_limiting()
        limiter = get_rate_limiter()
        assert limiter.enabled is True

    def test_reset_rate_limiter_creates_new_instance(self):
        """Test reset creates new instance"""
        reset_rate_limiter()
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.mark.asyncio
    async def test_burst_handling(self):
        """Test rate limiter handles burst traffic correctly"""
        reset_rate_limiter()
        enable_rate_limiting()
        limiter = get_rate_limiter()

        # Configure a small limit for testing
        limiter.buckets["test"] = TokenBucket(capacity=5, refill_rate=10.0)

        # Burst of requests up to capacity should succeed
        for i in range(5):
            allowed, _ = await limiter.check_rate_limit("test")
            assert allowed, f"Request {i+1} should be allowed"

        # Next request should fail
        allowed, wait_time = await limiter.check_rate_limit("test")
        assert not allowed
        assert wait_time > 0

        # Wait for refill and retry
        await asyncio.sleep(0.2)  # Wait for ~2 tokens
        allowed, _ = await limiter.check_rate_limit("test")
        assert allowed

    @pytest.mark.asyncio
    async def test_multiple_categories_independent(self):
        """Test different categories have independent limits"""
        reset_rate_limiter()
        enable_rate_limiting()
        limiter = get_rate_limiter()

        # Set up different limits
        limiter.buckets["cat1"] = TokenBucket(capacity=1, refill_rate=0)
        limiter.buckets["cat2"] = TokenBucket(capacity=10, refill_rate=0)

        # Exhaust cat1
        await limiter.check_rate_limit("cat1")
        allowed1, _ = await limiter.check_rate_limit("cat1")
        assert not allowed1

        # cat2 should still work
        allowed2, _ = await limiter.check_rate_limit("cat2")
        assert allowed2

    @pytest.mark.asyncio
    async def test_concurrent_requests_fairness(self):
        """Test concurrent requests are handled fairly"""
        reset_rate_limiter()
        enable_rate_limiting()
        limiter = get_rate_limiter()

        # Set up limited capacity
        limiter.buckets["concurrent"] = TokenBucket(capacity=10, refill_rate=0)

        async def make_request(request_id):
            allowed, _ = await limiter.check_rate_limit("concurrent")
            return (request_id, allowed)

        # Launch concurrent requests
        tasks = [make_request(i) for i in range(15)]
        results = await asyncio.gather(*tasks)

        # Exactly 10 should succeed
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]

        assert len(successful) == 10
        assert len(failed) == 5