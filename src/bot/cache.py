"""Bounded TTL cache for transcripts, isolated per-chat history/sessions, and feedback persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_PER_CHAT = 250
MAX_SESSION_TURNS_PER_CHAT = 20
MAX_PERSISTED_FEEDBACKS = 500
CHAT_RATE_LIMIT_MAX_REQUESTS = 5
CHAT_RATE_LIMIT_WINDOW_SECONDS = 300


@dataclass(slots=True)
class TranscriptEntry:
    """Cached transcript and summary entry with speaker name, language, and button state."""

    speaker_name: str
    text: str
    summary: str
    language: str
    created_at: float
    chat_id: str | None = None
    tts_used: bool = False
    transcript_used: bool = False
    fb_id: str | None = None


@dataclass(slots=True)
class ChatMessageEntry:
    """Single recorded text/caption/voice message in a group, channel, or chat."""

    sender: str
    text: str
    created_at: float


@dataclass(slots=True)
class FeedbackEntry:
    """Represents a bot output that can receive a one-time thumbs up/down feedback."""

    fb_id: str
    output_type: str
    chat_id: str
    chat_title: str
    preview: str
    created_at: str
    language: str = "fa"
    rating: str | None = None
    voter_id: int | None = None
    voter_name: str | None = None
    voter_username: str | None = None
    voted_at: str | None = None
    linked_transcript_cache_id: str | None = None


class TranscriptCache:
    """Bounded LRU + TTL cache for transcripts, isolated per-chat sessions, and feedback."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_items: int = 500,
        max_messages_per_chat: int = MAX_CHAT_HISTORY_PER_CHAT,
        persistence_path: Path | str | None = None,
        feedback_path: Path | str | None = None,
        admin_config_path: Path | str | None = None,
        admin_chat_id: int | None = None,
        users_stats_path: Path | str | None = None,
        rate_limit_max_requests: int = CHAT_RATE_LIMIT_MAX_REQUESTS,
        rate_limit_window_seconds: int = CHAT_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.max_messages_per_chat = max_messages_per_chat
        self.rate_limit_max_requests = max(1, int(rate_limit_max_requests))
        self.rate_limit_window_seconds = max(1, int(rate_limit_window_seconds))
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.feedback_path = Path(feedback_path) if feedback_path else None
        self.admin_config_path = Path(admin_config_path) if admin_config_path else None
        self.users_stats_path = Path(users_stats_path) if users_stats_path else None
        self._admin_chat_id: int | None = admin_chat_id

        self._store: OrderedDict[str, TranscriptEntry] = OrderedDict()
        self._inflight: set[str] = set()
        self._completed_voices: OrderedDict[str, float] = OrderedDict()
        self._chat_history: OrderedDict[str, deque[ChatMessageEntry]] = OrderedDict()
        self._chat_sessions: OrderedDict[str, deque[tuple[str, str, str]]] = OrderedDict()
        self._chat_rate_windows: OrderedDict[str, deque[float]] = OrderedDict()
        self._prompt_message_ids: OrderedDict[str, int] = OrderedDict()
        self._vibe_pickers: OrderedDict[str, tuple[str, list[str]]] = OrderedDict()
        self._feedbacks: OrderedDict[str, FeedbackEntry] = OrderedDict()
        self._users_stats: dict[str, dict[str, Any]] = {}
        self._chats_stats: dict[str, dict[str, Any]] = {}

        self._load_chat_history()
        self._load_feedbacks()
        self._load_admin_config()
        self._load_users_stats()

    def _load_json_safe(self, path: Path | None) -> Any:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", path, exc)
            return None

    def _save_json_safe(
        self, path: Path | None, payload: Any, *, compact: bool = False
    ) -> None:
        if path is None:
            return
        tmp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":") if compact else None,
                indent=None if compact else 2,
            )
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save %s: %s", path, exc)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _load_chat_history(self) -> None:
        raw = self._load_json_safe(self.persistence_path)
        if not isinstance(raw, dict):
            return
        now = time.monotonic()
        items_list = list(raw.items())[-self.max_items :]
        for chat_key, items in items_list:
            if not isinstance(items, list):
                continue
            q: deque[ChatMessageEntry] = deque(maxlen=self.max_messages_per_chat)
            for item in items[-self.max_messages_per_chat :]:
                if isinstance(item, dict) and item.get("text"):
                    q.append(
                        ChatMessageEntry(
                            sender=str(item.get("sender") or "کاربر"),
                            text=str(item["text"]),
                            created_at=now,
                        )
                    )
            if q:
                self._chat_history[str(chat_key)] = q

    def _save_chat_history(self) -> None:
        payload = {
            chat_key: [{"sender": entry.sender, "text": entry.text} for entry in entries]
            for chat_key, entries in self._chat_history.items()
        }
        self._save_json_safe(self.persistence_path, payload, compact=True)

    def _load_feedbacks(self) -> None:
        raw = self._load_json_safe(self.feedback_path)
        if not isinstance(raw, list):
            return
        for item in raw[-MAX_PERSISTED_FEEDBACKS:]:
            if not isinstance(item, dict) or not item.get("fb_id"):
                continue
            try:
                entry = FeedbackEntry(**item)
                self._feedbacks[entry.fb_id] = entry
            except TypeError:
                continue

    def _save_feedbacks(self) -> None:
        payload = [asdict(entry) for entry in list(self._feedbacks.values())[-MAX_PERSISTED_FEEDBACKS:]]
        self._save_json_safe(self.feedback_path, payload)

    def _load_admin_config(self) -> None:
        if self._admin_chat_id is not None:
            return
        raw = self._load_json_safe(self.admin_config_path)
        if isinstance(raw, dict) and raw.get("admin_chat_id"):
            try:
                self._admin_chat_id = int(raw["admin_chat_id"])
            except (ValueError, TypeError):
                pass

    def get_admin_chat_id(self) -> int | None:
        """Return configured admin Telegram chat ID for live feedback delivery."""
        return self._admin_chat_id

    def set_admin_chat_id(self, chat_id: int) -> None:
        """Persist the owner's Telegram chat ID for feedback notifications."""
        self._admin_chat_id = int(chat_id)
        self._save_json_safe(self.admin_config_path, {"admin_chat_id": self._admin_chat_id})

    def create_feedback_target(
        self,
        output_type: str,
        chat_id: int | str,
        chat_title: str,
        preview: str,
        language: str = "fa",
        linked_transcript_cache_id: str | None = None,
    ) -> str:
        """Create and persist a feedback target, preferring to evict unvoted items first."""
        while len(self._feedbacks) >= MAX_PERSISTED_FEEDBACKS:
            unvoted_key = next(
                (k for k, v in self._feedbacks.items() if v.rating is None), None
            )
            if unvoted_key is not None:
                self._feedbacks.pop(unvoted_key, None)
            else:
                self._feedbacks.popitem(last=False)

        fb_id = uuid.uuid4().hex[:10]
        clean_preview = " ".join((preview or "").split())[:180]
        entry = FeedbackEntry(
            fb_id=fb_id,
            output_type=output_type,
            chat_id=str(chat_id),
            chat_title=(chat_title or str(chat_id)).strip(),
            preview=clean_preview,
            created_at=datetime.now(timezone.utc).isoformat(),
            language=language,
            linked_transcript_cache_id=linked_transcript_cache_id,
        )
        self._feedbacks[fb_id] = entry
        self._save_feedbacks()
        return fb_id

    def get_feedback_target(self, fb_id: str) -> FeedbackEntry | None:
        """Return the feedback entry by ID if present."""
        return self._feedbacks.get(fb_id)

    def record_feedback(
        self,
        fb_id: str,
        rating: str,
        voter_id: int | None = None,
        voter_name: str = "",
        voter_username: str | None = None,
    ) -> tuple[FeedbackEntry | None, bool]:
        """Record a rating ('up' or 'down') once. Returns (entry, already_voted)."""
        entry = self._feedbacks.get(fb_id)
        if entry is None:
            return None, False
        if entry.rating is not None:
            return entry, True

        entry.rating = "up" if rating == "up" else "down"
        entry.voter_id = voter_id
        entry.voter_name = (voter_name or "کاربر").strip()
        entry.voter_username = voter_username.lstrip("@") if voter_username else None
        entry.voted_at = datetime.now(timezone.utc).isoformat()
        self._save_feedbacks()
        return entry, False

    def get_feedback_stats(self) -> dict[str, Any]:
        """Return aggregated feedback statistics and recent rated items."""
        rated = [entry for entry in self._feedbacks.values() if entry.rating is not None]
        up_count = sum(1 for entry in rated if entry.rating == "up")
        down_count = sum(1 for entry in rated if entry.rating == "down")
        return {
            "total": len(rated),
            "up": up_count,
            "down": down_count,
            "recent": rated[-10:],
        }

    def _load_users_stats(self) -> None:
        raw = self._load_json_safe(self.users_stats_path)
        if not isinstance(raw, dict):
            return
        users = raw.get("users")
        chats = raw.get("chats")
        if isinstance(users, dict):
            self._users_stats = {str(k): v for k, v in users.items() if isinstance(v, dict)}
        if isinstance(chats, dict):
            self._chats_stats = {str(k): v for k, v in chats.items() if isinstance(v, dict)}

    def _save_users_stats(self) -> None:
        self._save_json_safe(
            self.users_stats_path,
            {"users": self._users_stats, "chats": self._chats_stats},
            compact=True,
        )

    @staticmethod
    def _prune_stats_dict(store: dict[str, dict[str, Any]], max_entries: int) -> None:
        if len(store) <= max_entries:
            return
        by_oldest = sorted(store.items(), key=lambda kv: str(kv[1].get("last_seen", "")))
        for key, _ in by_oldest[: len(store) - max_entries]:
            store.pop(key, None)

    def record_activity(
        self,
        *,
        user_id: int | None = None,
        user_name: str = "",
        username: str | None = None,
        chat_id: int | str | None = None,
        chat_title: str = "",
        chat_type: str = "private",
        is_direct: bool = True,
        now: datetime | None = None,
    ) -> None:
        """Record user and chat activity persistently for active-user analytics."""
        ts = (now or datetime.now(timezone.utc)).isoformat()
        changed = False

        if user_id is not None:
            ukey = str(user_id)
            clean_name = (user_name or "کاربر").strip()
            clean_handle = username.lstrip("@").strip() if username else None
            existing = self._users_stats.get(ukey)
            if existing is None:
                self._users_stats[ukey] = {
                    "user_id": int(user_id),
                    "name": clean_name,
                    "username": clean_handle,
                    "first_seen": ts,
                    "last_seen": ts,
                    "direct_interactions": 1 if is_direct else 0,
                    "total_events": 1,
                }
            else:
                existing["name"] = clean_name or existing.get("name", "کاربر")
                existing["username"] = clean_handle
                existing["last_seen"] = ts
                existing["total_events"] = int(existing.get("total_events", 0)) + 1
                if is_direct:
                    existing["direct_interactions"] = int(existing.get("direct_interactions", 0)) + 1
            self._prune_stats_dict(self._users_stats, 5000)
            changed = True

        if chat_id is not None:
            ckey = str(chat_id)
            clean_title = (chat_title or ckey).strip()
            existing_chat = self._chats_stats.get(ckey)
            if existing_chat is None:
                self._chats_stats[ckey] = {
                    "chat_id": ckey,
                    "title": clean_title,
                    "chat_type": chat_type or "private",
                    "first_seen": ts,
                    "last_seen": ts,
                    "total_events": 1,
                }
            else:
                if clean_title:
                    existing_chat["title"] = clean_title
                if chat_type:
                    existing_chat["chat_type"] = chat_type
                existing_chat["last_seen"] = ts
                existing_chat["total_events"] = int(existing_chat.get("total_events", 0)) + 1
            self._prune_stats_dict(self._chats_stats, 2000)
            changed = True

        if changed:
            self._save_users_stats()

    def get_active_users_stats(self, now: datetime | None = None) -> dict[str, Any]:
        """Compute total and active user/chat counts across 24h, 7d, and 30d windows."""
        ref = now or datetime.now(timezone.utc)
        cutoff_24h = ref - timedelta(hours=24)
        cutoff_7d = ref - timedelta(days=7)
        cutoff_30d = ref - timedelta(days=30)

        def _parse_ts(value: Any) -> datetime | None:
            if not isinstance(value, str) or not value:
                return None
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                return None

        active_24h = 0
        active_7d = 0
        active_30d = 0
        direct_users = 0
        direct_24h = 0
        recent_users: list[dict[str, Any]] = []

        for info in self._users_stats.values():
            last_dt = _parse_ts(info.get("last_seen"))
            has_direct = int(info.get("direct_interactions", 0)) > 0
            if has_direct:
                direct_users += 1
            if last_dt is not None:
                if last_dt >= cutoff_24h:
                    active_24h += 1
                    if has_direct:
                        direct_24h += 1
                if last_dt >= cutoff_7d:
                    active_7d += 1
                if last_dt >= cutoff_30d:
                    active_30d += 1
            recent_users.append(
                {
                    "user_id": info.get("user_id"),
                    "name": info.get("name", "کاربر"),
                    "username": info.get("username"),
                    "last_seen": info.get("last_seen", ""),
                    "direct_interactions": int(info.get("direct_interactions", 0)),
                    "total_events": int(info.get("total_events", 0)),
                }
            )

        recent_users.sort(key=lambda u: u.get("last_seen", ""), reverse=True)

        group_chats = 0
        private_chats = 0
        active_chats_7d = 0
        for cinfo in self._chats_stats.values():
            ctype = str(cinfo.get("chat_type", "private")).lower()
            if ctype == "private":
                private_chats += 1
            else:
                group_chats += 1
            last_chat_dt = _parse_ts(cinfo.get("last_seen"))
            if last_chat_dt is not None and last_chat_dt >= cutoff_7d:
                active_chats_7d += 1

        return {
            "total_users": len(self._users_stats),
            "direct_users": direct_users,
            "active_24h": active_24h,
            "direct_24h": direct_24h,
            "active_7d": active_7d,
            "active_30d": active_30d,
            "total_chats": len(self._chats_stats),
            "group_chats": group_chats,
            "private_chats": private_chats,
            "active_chats_7d": active_chats_7d,
            "recent_users": recent_users[:10],
        }

    def set_prompt_message_id(self, chat_id: int | str, message_id: int | None) -> None:
        if message_id is not None:
            key = str(chat_id)
            if key not in self._prompt_message_ids:
                while len(self._prompt_message_ids) >= max(self.max_items, 500):
                    self._prompt_message_ids.popitem(last=False)
            else:
                self._prompt_message_ids.move_to_end(key)
            self._prompt_message_ids[key] = int(message_id)

    def pop_prompt_message_id(self, chat_id: int | str) -> int | None:
        return self._prompt_message_ids.pop(str(chat_id), None)

    def get_recent_chat_senders(
        self, chat_id: int | str, *, max_senders: int = 8
    ) -> list[str]:
        """Return unique active sender labels for `chat_id`, ordered most-recent first."""
        entries = self._chat_history.get(str(chat_id))
        if not entries:
            return []
        senders: list[str] = []
        seen_lower: set[str] = set()
        for entry in reversed(entries):
            raw_sender = (entry.sender or "").strip()
            if raw_sender.endswith(" (ویس)"):
                raw_sender = raw_sender[: -len(" (ویس)")].strip()
            if not raw_sender or raw_sender in {"کاربر", "گوینده"}:
                continue
            key = raw_sender.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            senders.append(raw_sender)
            if len(senders) >= max(1, int(max_senders)):
                break
        return senders

    def create_vibe_picker(self, chat_id: int | str, targets: list[str]) -> str:
        """Create a short-ID inline picker mapping for `/vibe` member selection."""
        while len(self._vibe_pickers) >= max(self.max_items, 500):
            self._vibe_pickers.popitem(last=False)
        picker_id = uuid.uuid4().hex[:8]
        self._vibe_pickers[picker_id] = (str(chat_id), list(targets))
        return picker_id

    def get_vibe_picker(self, picker_id: str) -> tuple[str, list[str]] | None:
        """Return `(chat_id, targets)` for `picker_id` if still cached."""
        item = self._vibe_pickers.get(picker_id)
        if item is not None:
            self._vibe_pickers.move_to_end(picker_id)
        return item

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

    def add_chat_message(
        self,
        chat_id: int | str,
        sender: str,
        text: str,
        *,
        deduplicate_recent: bool = False,
    ) -> None:
        """Append a message strictly to `chat_id`'s isolated rolling buffer (>= 200 msgs)."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        clean_sender = (sender or "کاربر").strip()
        key = str(chat_id)
        if key not in self._chat_history:
            while len(self._chat_history) >= self.max_items:
                self._chat_history.popitem(last=False)
            self._chat_history[key] = deque(maxlen=self.max_messages_per_chat)
        else:
            self._chat_history.move_to_end(key)

        q = self._chat_history[key]
        if q and q[-1].text == cleaned and q[-1].sender == clean_sender:
            return
        if deduplicate_recent and any(
            entry.text == cleaned and entry.sender == clean_sender for entry in q
        ):
            return

        q.append(
            ChatMessageEntry(
                sender=clean_sender,
                text=cleaned,
                created_at=time.monotonic(),
            )
        )
        self._save_chat_history()

    def get_recent_chat_messages(
        self, chat_id: int | str, limit: int = MAX_CHAT_HISTORY_PER_CHAT
    ) -> list[ChatMessageEntry]:
        """Return up to `limit` (clamped to `max_messages_per_chat`) most recent messages strictly for `chat_id`."""
        clamped = max(1, min(int(limit), self.max_messages_per_chat))
        key = str(chat_id)
        history = self._chat_history.get(key)
        if not history:
            return []
        items = list(history)
        return items[-clamped:]

    def add_chat_turn(
        self, chat_id: int | str, user_name: str, question: str, answer: str
    ) -> None:
        """Store a conversational Q&A turn strictly inside `chat_id`'s isolated session."""
        key = str(chat_id)
        if key not in self._chat_sessions:
            while len(self._chat_sessions) >= self.max_items:
                self._chat_sessions.popitem(last=False)
            self._chat_sessions[key] = deque(maxlen=MAX_SESSION_TURNS_PER_CHAT)
        else:
            self._chat_sessions.move_to_end(key)

        self._chat_sessions[key].append((user_name, question, answer))

    def get_chat_session_turns(
        self, chat_id: int | str
    ) -> list[tuple[str, str, str]]:
        """Return isolated Q&A turns strictly for `chat_id`."""
        turns = self._chat_sessions.get(str(chat_id))
        return list(turns) if turns else []

    def clear_chat_session(self, chat_id: int | str) -> None:
        """Clear ONLY `chat_id`'s interactive Q&A session turns."""
        self._chat_sessions.pop(str(chat_id), None)

    def check_and_consume_chat_rate_limit(
        self, chat_id: int | str, *, now: float | None = None
    ) -> tuple[bool, int]:
        """Enforce a per-chat sliding window rate limit (default 5 requests / 300s).

        Returns `(allowed, retry_after_seconds)`.
        """
        current = time.monotonic() if now is None else float(now)
        key = str(chat_id)
        if key not in self._chat_rate_windows:
            while len(self._chat_rate_windows) >= max(self.max_items * 4, 2000):
                self._chat_rate_windows.popitem(last=False)
            self._chat_rate_windows[key] = deque()
        else:
            self._chat_rate_windows.move_to_end(key)
        window = self._chat_rate_windows[key]
        cutoff = current - self.rate_limit_window_seconds

        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self.rate_limit_max_requests:
            retry_after = max(1, math.ceil(self.rate_limit_window_seconds - (current - window[0])))
            return False, retry_after

        window.append(current)
        return True, 0

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
        """Release an in-flight lock without marking completed."""
        if not voice_key:
            return
        self._inflight.discard(voice_key)

    def put(
        self,
        speaker_name: str,
        text: str,
        summary: str = "",
        language: str = "fa",
        chat_id: int | str | None = None,
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
            chat_id=str(chat_id) if chat_id is not None else None,
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
