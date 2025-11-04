"""
Redis-based session manager for AutoML Pipeline
Provides automatic expiration, persistence, and scalability
"""

import os
import pickle
import redis
from typing import Any, Optional, Dict


class SessionManager:
    """Manages user sessions with Redis backend"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        session_ttl: int = 7200,  # 2 hours default
    ):
        """
        Initialize Redis session manager

        Args:
            host: Redis host (defaults to env REDIS_HOST or 'localhost')
            port: Redis port (defaults to env REDIS_PORT or 6379)
            db: Redis database number
            session_ttl: Session time-to-live in seconds (default: 2 hours)
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", 6379))
        self.db = db
        self.session_ttl = session_ttl

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=False,  # We'll handle serialization ourselves
            socket_connect_timeout=5,
            socket_keepalive=True,
        )

        # Test connection
        try:
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise

    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"

    def create(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Create a new session

        Args:
            session_id: Unique session identifier
            data: Session data dictionary
        """
        key = self._get_key(session_id)
        serialized = pickle.dumps(data)
        self.redis_client.setex(key, self.session_ttl, serialized)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None if not found
        """
        key = self._get_key(session_id)
        data = self.redis_client.get(key)
        if data is None:
            return None
        return pickle.loads(data)

    def update(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update existing session (resets TTL)

        Args:
            session_id: Session identifier
            data: Updated session data

        Returns:
            True if successful, False if session doesn't exist
        """
        key = self._get_key(session_id)
        if not self.redis_client.exists(key):
            return False

        serialized = pickle.dumps(data)
        self.redis_client.setex(key, self.session_ttl, serialized)
        return True

    def set_field(self, session_id: str, field: str, value: Any) -> bool:
        """
        Set a specific field in session data

        Args:
            session_id: Session identifier
            field: Field name to update
            value: Value to set

        Returns:
            True if successful, False if session doesn't exist
        """
        session_data = self.get(session_id)
        if session_data is None:
            return False

        session_data[field] = value
        return self.update(session_id, session_data)

    def get_field(self, session_id: str, field: str, default: Any = None) -> Any:
        """
        Get a specific field from session data

        Args:
            session_id: Session identifier
            field: Field name to retrieve
            default: Default value if field or session doesn't exist

        Returns:
            Field value or default
        """
        session_data = self.get(session_id)
        if session_data is None:
            return default
        return session_data.get(field, default)

    def delete(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if it didn't exist
        """
        key = self._get_key(session_id)
        return bool(self.redis_client.delete(key))

    def exists(self, session_id: str) -> bool:
        """
        Check if session exists

        Args:
            session_id: Session identifier

        Returns:
            True if session exists, False otherwise
        """
        key = self._get_key(session_id)
        return bool(self.redis_client.exists(key))

    def refresh_ttl(self, session_id: str) -> bool:
        """
        Refresh session TTL (reset expiration timer)

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False if session doesn't exist
        """
        key = self._get_key(session_id)
        return bool(self.redis_client.expire(key, self.session_ttl))

    def get_ttl(self, session_id: str) -> Optional[int]:
        """
        Get remaining TTL for session

        Args:
            session_id: Session identifier

        Returns:
            Remaining seconds or None if session doesn't exist
        """
        key = self._get_key(session_id)
        ttl = self.redis_client.ttl(key)
        return ttl if ttl > 0 else None

    def list_all_sessions(self) -> list[str]:
        """
        List all active session IDs

        Returns:
            List of session IDs
        """
        keys = self.redis_client.keys("session:*")
        return [key.decode("utf-8").replace("session:", "") for key in keys]

    def clear_all(self) -> int:
        """
        Delete all sessions (use with caution!)

        Returns:
            Number of sessions deleted
        """
        keys = self.redis_client.keys("session:*")
        if keys:
            return self.redis_client.delete(*keys)
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis and session statistics

        Returns:
            Dictionary with stats
        """
        info = self.redis_client.info()
        session_count = len(self.redis_client.keys("session:*"))

        return {
            "active_sessions": session_count,
            "redis_version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "total_connections_received": info.get("total_connections_received"),
            "uptime_in_seconds": info.get("uptime_in_seconds"),
        }

    def close(self):
        """Close Redis connection"""
        self.redis_client.close()


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get or create the global session manager instance

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
