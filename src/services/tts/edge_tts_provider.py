"""Microsoft Edge Neural TTS provider (supports natural Persian and English voices)."""

from __future__ import annotations

import edge_tts

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.tts.base import TTSResult, clean_text_for_tts


class EdgeTTSProvider:
    """Synthesizes speech using free Microsoft Edge Neural TTS voices."""

    name = "edge-tts"

    def __init__(
        self,
        fa_voice: str = "fa-IR-FaridNeural",
        en_voice: str = "en-US-AndrewNeural",
    ) -> None:
        self.fa_voice = fa_voice
        self.en_voice = en_voice

    async def synthesize(self, text: str, language: str) -> TTSResult:
        norm_lang = normalize_language(language, text)
        spoken_text = clean_text_for_tts(text)
        if not spoken_text:
            raise ProviderError("Cannot synthesize empty text")

        voice = self.fa_voice if norm_lang == "fa" else self.en_voice
        try:
            communicate = edge_tts.Communicate(spoken_text, voice)
            audio_chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    audio_chunks.append(chunk["data"])
            audio_bytes = b"".join(audio_chunks)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Edge TTS error: {exc}") from exc

        if not audio_bytes:
            raise ProviderError("Edge TTS returned empty audio stream")

        return TTSResult(audio_bytes=audio_bytes, language=norm_lang, provider=self.name)
