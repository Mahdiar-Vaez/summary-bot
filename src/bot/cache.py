"""Bounded TTL in-memory cache for storing full voice transcripts."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptEntry:
    """Cached transcript entry with speaker name and timestamp."""

    speaker_name: str
    text: str
    created_at: float


class TranscriptCache:
    """Bounded LRU + TTL cache for voice message transcripts."""

    def __init__(self, ttl_seconds: int = 3600, max_items: int = 500) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._store: OrderedDict[str, TranscriptEntry] = OrderedDict()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired_keys = [
            k
            for k, entry in self._store.items()
            if (now - entry.created_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            self._store.pop(k, None)

    def put(self, speaker_name: str, text: str) -> str:
        """Store transcript and return a compact 12-char cache ID."""
        self._evict_expired()
        while len(self._store) >= self.max_items:
            self._store.popitem(last=False)

        cache_id = uuid.uuid4().hex[:12]
        self._store[cache_id] = TranscriptEntry(
            speaker_name=speaker_name,
            text=text,
            created_at=time.monotonic(),
        )
        return cache_id

    def get(self, cache_id: str) -> TranscriptEntry | None:
        """Retrieve a transcript entry if present and not expired."""
        self._evict_expired()
        entry = self._store.get(cache_id)
        if entry is not None:
            self._store.move_to_end(cache_id)
        return entry


def chunk_text(text: str, max_len: int = 4000) -> list[str]:
    """Split text into safe Telegram message chunks (<= max_len characters)."""
    if not text:
        return [""]
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]
