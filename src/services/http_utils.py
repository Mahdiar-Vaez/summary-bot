"""Shared HTTP JSON parsing, response extraction, and secret-redaction helpers."""

from __future__ import annotations

import base64
import re
from typing import Any

import httpx

from src.exceptions import ProviderError

_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL | re.IGNORECASE)
_QUERY_SECRET_RE = re.compile(r"([?&])(?:key|api_key|token|access_token)=[^&\s'\"]+", re.IGNORECASE)
_RAW_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])"
    r"(?:AQ\.[A-Za-z0-9_\-]{10,}|AIza[0-9A-Za-z\-_]{16,}|gsk_[0-9A-Za-z\-_]{16,}|sk-(?:or-v1-|proj-)?[0-9A-Za-z\-_]{12,})"
    r"(?![A-Za-z0-9_\-])"
)


def redact_secrets(text: str | None) -> str:
    """Mask query-string API keys and raw provider tokens inside error/log messages."""
    cleaned = str(text or "")
    cleaned = _QUERY_SECRET_RE.sub(r"\1[REDACTED_KEY]", cleaned)
    cleaned = _RAW_KEY_RE.sub("[REDACTED]", cleaned)
    return cleaned


def parse_json_response(response: httpx.Response, provider_name: str) -> Any:
    """Safely decode an HTTP response body as JSON or raise ProviderError."""
    try:
        return response.json()
    except (ValueError, TypeError) as exc:
        preview = redact_secrets(getattr(response, "text", "")[:200])
        raise ProviderError(f"{provider_name} invalid JSON response: {preview}") from exc


def extract_gemini_parts_text(data: Any, provider_name: str) -> str:
    """Extract non-thought text parts from a Gemini generateContent JSON payload."""
    if isinstance(data, dict) and data.get("error"):
        raise ProviderError(redact_secrets(f"{provider_name} API error: {data['error']}"))
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text_chunks = [
            p["text"]
            for p in parts
            if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str)
        ]
        return "".join(text_chunks).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            redact_secrets(f"{provider_name} unexpected response structure: {str(data)[:250]}")
        ) from exc


def extract_gemini_inline_audio(data: Any, provider_name: str) -> tuple[bytes, str]:
    """Extract (raw_audio_bytes, mime_type) from the first inlineData part in Gemini output."""
    if isinstance(data, dict) and data.get("error"):
        raise ProviderError(redact_secrets(f"{provider_name} API error: {data['error']}"))
    try:
        parts = data["candidates"][0]["content"]["parts"]
        inline_data = next(
            (
                p["inlineData"]
                for p in parts
                if isinstance(p, dict) and isinstance(p.get("inlineData"), dict)
            ),
            None,
        )
        if inline_data is None:
            raise KeyError("inlineData missing in response parts")
        b64_audio = inline_data["data"]
        mime_type = str(inline_data.get("mimeType") or "audio/wav")
        return base64.b64decode(b64_audio), mime_type
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError(
            redact_secrets(f"{provider_name} unexpected response: {str(data)[:250]}")
        ) from exc


def _flatten_message_content(content: Any) -> str:
    """Convert OpenAI string or multipart list content into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    pieces.append(text_val)
            elif isinstance(item, str):
                pieces.append(item)
        return "".join(pieces)
    return ""


def extract_openai_chat_text(data: Any, provider_name: str) -> str:
    """Extract text from OpenAI-compatible chat completions JSON and strip <think> reasoning."""
    if isinstance(data, dict) and data.get("error"):
        raise ProviderError(redact_secrets(f"{provider_name} API error: {data['error']}"))
    try:
        raw_content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            redact_secrets(f"{provider_name} unexpected response: {str(data)[:250]}")
        ) from exc

    flat = _flatten_message_content(raw_content)
    return _THINK_BLOCK_RE.sub("", flat).strip()
