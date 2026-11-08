"""Bounded TTL in-memory cache for storing full voice transcripts and summaries."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptEntry:
    """Cached transcript and summary entry with speaker name, language, and button state."""

    speaker_name: str
    text: str
    summary: str
    language: str
    created_at: float
    tts_used: bool = False
    transcript_used: bool = False


class TranscriptCache:
    """Bounded LRU + TTL cache for voice message transcripts, summaries, and deduplication."""

    def __init__(self, ttl_seconds: int = 3600, max_items: int = 500) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._store: OrderedDict[str, TranscriptEntry] = OrderedDict()
        self._inflight: set[str] = set()
        self._completed_voices: OrderedDict[str, float] = OrderedDict()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired_keys = [
            k
            for k, entry in self._store.items()
            if (now - entry.created_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            self._store.pop(k, None)

        expired_voices = [
            vk
            for vk, ts in self._completed_voices.items()
            if (now - ts) > self.ttl_seconds
        ]
        for vk in expired_voices:
            self._completed_voices.pop(vk, None)

    def try_acquire_voice(self, voice_key: str) -> bool:
        """Return True and lock voice_key if not currently in-flight or already summarized."""
        if not voice_key:
            return True
        self._evict_expired()
        if voice_key in self._inflight or voice_key in self._completed_voices:
            return False
        self._inflight.add(voice_key)
        return True

    def mark_voice_completed(self, voice_key: str) -> None:
        """Mark voice_key as completed and remove from in-flight set."""
        if not voice_key:
            return
        self._inflight.discard(voice_key)
        while len(self._completed_voices) >= self.max_items:
            self._completed_voices.popitem(last=False)
        self._completed_voices[voice_key] = time.monotonic()

    def release_voice(self, voice_key: str) -> None:
        """Release an in-flight lock without marking completed (e.g. on failure so retry is allowed)."""
        if not voice_key:
            return
        self._inflight.discard(voice_key)

    def put(
        self,
        speaker_name: str,
        text: str,
        summary: str = "",
        language: str = "fa",
    ) -> str:
        """Store transcript and summary and return a compact 12-char cache ID."""
        self._evict_expired()
        while len(self._store) >= self.max_items:
            self._store.popitem(last=False)

        cache_id = uuid.uuid4().hex[:12]
        self._store[cache_id] = TranscriptEntry(
            speaker_name=speaker_name,
            text=text,
            summary=summary or text,
            language=language,
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
