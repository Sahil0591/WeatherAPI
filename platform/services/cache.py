from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Optional, Protocol, Dict

log = logging.getLogger(__name__)

class Cache(Protocol):
    def get_json(self, key: str) -> Optional[Dict[str, Any]]: ...
    def set_json(self, key: str, value: Dict[str, Any], ttl_s: int) -> None: ...

class InMemoryCache:
    def __init__(self) -> None:
        # key -> (expires_at_monotonic, json_string)
        self._store: Dict[str, tuple[float, str]] = {}

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        item = self._store.get(key)
        if not item:
            return None
        expires_at, payload = item
        if now >= expires_at:
            # expired
            self._store.pop(key, None)
            return None
        try:
            return json.loads(payload)
        except Exception as e:
            log.warning("cache_deserialize_error", extra={"key": key, "error": str(e)})
            self._store.pop(key, None)
            return None

    def set_json(self, key: str, value: Dict[str, Any], ttl_s: int) -> None:
        try:
            payload = json.dumps(value, separators=(",", ":"))
        except Exception as e:
            log.warning("cache_serialize_error", extra={"key": key, "error": str(e)})
            return
        expires_at = time.monotonic() + max(0, int(ttl_s))
        self._store[key] = (expires_at, payload)

class RedisCache:
    def __init__(self, url: str) -> None:
        try:
            import redis  # type: ignore
        except Exception as e:
            raise RuntimeError(f"redis package not available: {e}") from e
        self._redis = redis.from_url(url, decode_responses=True)

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            log.warning("cache_deserialize_error", extra={"key": key, "error": str(e)})
            self._redis.delete(key)
            return None

    def set_json(self, key: str, value: Dict[str, Any], ttl_s: int) -> None:
        try:
            raw = json.dumps(value, separators=(",", ":"))
        except Exception as e:
            log.warning("cache_serialize_error", extra={"key": key, "error": str(e)})
            return
        ttl = max(0, int(ttl_s))
        if ttl > 0:
            self._redis.setex(key, ttl, raw)
        else:
            self._redis.set(key, raw)

def build_cache() -> Cache:
    """Factory: use Redis if REDIS_URL is set and redis-py is available; otherwise in-memory."""
    url = os.getenv("REDIS_URL")
    if not url:
        log.info("cache_init_inmemory")
        return InMemoryCache()
    try:
        cache = RedisCache(url)
        log.info("cache_init_redis", extra={"url": url})
        return cache
    except Exception as e:
        log.warning("cache_init_redis_failed_fallback_inmemory", extra={"error": str(e)})
        return InMemoryCache()
