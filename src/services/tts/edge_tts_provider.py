"""Microsoft Edge Neural TTS provider (supports natural Persian and English voices)."""

from __future__ import annotations

import asyncio
import os
import edge_tts

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.tts.base import TTSResult, clean_text_for_tts


class EdgeTTSProvider:
    """Synthesizes speech using free Microsoft Edge Neural TTS voices."""

    name = "edge-tts"

    def __init__(
        self,
        fa_voice: str = "fa-IR-DilaraNeural",
        en_voice: str = "en-US-AriaNeural",
        proxy: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.fa_voice = fa_voice
        self.en_voice = en_voice
        self.proxy = proxy
        self._timeout = timeout

    async def _collect_audio(self, communicate: edge_tts.Communicate) -> bytes:
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    async def synthesize(self, text: str, language: str) -> TTSResult:
        norm_lang = normalize_language(language, text)
        spoken_text = clean_text_for_tts(text)
        if not spoken_text:
            raise ProviderError("Cannot synthesize empty text")

        voice = self.fa_voice if norm_lang == "fa" else self.en_voice
        active_proxy = (
            self.proxy
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or None
        )
        try:
            communicate = edge_tts.Communicate(
                spoken_text, voice, rate="+5%", proxy=active_proxy
            )
            audio_bytes = await asyncio.wait_for(
                self._collect_audio(communicate), timeout=self._timeout
            )
        except TimeoutError as exc:
            raise ProviderError(f"Edge TTS timeout after {self._timeout}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Edge TTS error: {exc}") from exc

        if not audio_bytes:
            raise ProviderError("Edge TTS returned empty audio stream")

        return TTSResult(audio_bytes=audio_bytes, language=norm_lang, provider=self.name)
