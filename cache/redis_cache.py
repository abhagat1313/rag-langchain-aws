import redis
import json
import hashlib
from typing import Optional


class RedisCache:
    """
    Simple Redis cache wrapper.
    """

    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def _hash_key(self, raw_key: str) -> str:
        """
        Hash raw cache keys to keep them short and consistent.
        """
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        hashed_key = self._hash_key(key)
        value = self.client.get(hashed_key)
        return json.loads(value) if value else None

    def set(self, key: str, value: dict, ttl_seconds: int):
        hashed_key = self._hash_key(key)
        self.client.setex(
            hashed_key,
            ttl_seconds,
            json.dumps(value)
        )
