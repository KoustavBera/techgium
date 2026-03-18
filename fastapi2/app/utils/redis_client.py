"""
Redis Client Utility
Singleton wrapper for Redis connection with screening-specific helpers.
Provides async get/set/expire operations and JSON serialization for screening data.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import redis; if unavailable, fall back to no-op mode
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("redis package not installed. Redis caching disabled — using in-memory fallback.")
    REDIS_AVAILABLE = False


class RedisClient:
    """
    Singleton Redis client with helpers for screening data.

    Environment variables:
        REDIS_URL  – full Redis URL (default: redis://localhost:6379/0)
    """

    _instance: Optional["RedisClient"] = None
    _client: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Connection ────────────────────────────────────────────────
    def connect(self) -> bool:
        """Establish connection to Redis. Returns True on success."""
        if not REDIS_AVAILABLE:
            logger.info("Redis package not available — caching disabled.")
            return False

        try:
            url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            logger.info(f"Connected to Redis at {url}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed ({e}). Caching disabled — using in-memory fallback.")
            self._client = None
            return False

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    # ── Primitive ops ────────────────────────────────────────────
    def get(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed for key={key}: {e}")
            return None

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> bool:
        if not self._client:
            return False
        try:
            self._client.setex(key, ttl_seconds, value)
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for key={key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._client:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed for key={key}: {e}")
            return False

    # ── JSON helpers ─────────────────────────────────────────────
    def get_json(self, key: str) -> Optional[Dict]:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, data: Dict, ttl_seconds: int = 3600) -> bool:
        try:
            return self.set(key, json.dumps(data, default=str), ttl_seconds)
        except Exception as e:
            logger.warning(f"Redis SET_JSON failed for key={key}: {e}")
            return False

    # ── Screening-specific helpers ───────────────────────────────
    SCREENING_PREFIX = "screening:"
    CONTEXT_PREFIX = "ctx:"
    DEFAULT_TTL = 3600  # 1 hour

    def cache_screening_context(self, patient_id: str, context_md: str, ttl: int = DEFAULT_TTL) -> bool:
        """Cache the rendered Markdown screening context for a patient."""
        key = f"{self.CONTEXT_PREFIX}{patient_id.lower()}"
        return self.set(key, context_md, ttl)

    def get_screening_context(self, patient_id: str) -> Optional[str]:
        """Retrieve cached screening context for a patient."""
        key = f"{self.CONTEXT_PREFIX}{patient_id.lower()}"
        return self.get(key)

    def invalidate_screening_context(self, patient_id: str) -> bool:
        """Remove cached context (e.g. after a new screening)."""
        key = f"{self.CONTEXT_PREFIX}{patient_id.lower()}"
        return self.delete(key)

    def cache_screening_result(self, screening_id: str, data: Dict, ttl: int = DEFAULT_TTL) -> bool:
        """Cache a full screening result dict."""
        key = f"{self.SCREENING_PREFIX}{screening_id}"
        return self.set_json(key, data, ttl)

    def get_screening_result(self, screening_id: str) -> Optional[Dict]:
        """Retrieve a cached screening result dict."""
        key = f"{self.SCREENING_PREFIX}{screening_id}"
        return self.get_json(key)


# Module-level singleton
redis_client = RedisClient()
