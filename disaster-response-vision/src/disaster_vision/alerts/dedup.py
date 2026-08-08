"""In-memory alert deduplication with configurable time window."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DedupKey:
    """Key used to identify duplicate alert events."""

    detected_class: str
    source_path: str | None


class AlertDeduplicator:
    """Suppress repeated alerts for the same event within a time window."""

    def __init__(self, window_seconds: int = 300) -> None:
        self._window_seconds = window_seconds
        self._last_sent: dict[str, float] = {}

    def _hash_key(self, key: DedupKey) -> str:
        raw = f"{key.detected_class}|{key.source_path or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def should_send(self, key: DedupKey) -> bool:
        """Return True if an alert should be sent (not a duplicate within window)."""
        digest = self._hash_key(key)
        now = time.monotonic()
        last = self._last_sent.get(digest)
        if last is not None and (now - last) < self._window_seconds:
            logger.info(
                "Suppressing duplicate alert for class=%s source=%s",
                key.detected_class,
                key.source_path,
            )
            return False
        self._last_sent[digest] = now
        return True

    def clear(self) -> None:
        """Clear deduplication state (useful in tests)."""
        self._last_sent.clear()
