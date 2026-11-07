import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.tts.base import TTSResult, clean_text_for_tts
from src.services.tts.google_tts import GoogleTranslateTTSProvider
from src.services.tts.manager import TTSManager


def test_clean_text_for_tts_removes_markdown_and_emojis():
    raw = "📌 خلاصه صحبت‌های mr.vaez:\n\n• **زمان جلسه:** فردا ساعت ۱۰ صبح\n• آماده‌سازی گزارش"
    cleaned = clean_text_for_tts(raw)
    assert "📌" not in cleaned
    assert "**" not in cleaned
    assert "•" not in cleaned
    assert "زمان جلسه" in cleaned


@pytest.mark.asyncio
async def test_google_translate_tts_provider_synthesizes_audio():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tl=fa" in str(request.url)
        return httpx.Response(200, content=b"ID3-fake-mp3-bytes")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GoogleTranslateTTSProvider(client=client)
    res = await provider.synthesize("سلام فردا جلسه داریم", language="fa")
    assert res.audio_bytes == b"ID3-fake-mp3-bytes"
    assert res.language == "fa"
    assert res.provider == "google-tts"


@pytest.mark.asyncio
async def test_tts_manager_falls_back_when_first_provider_fails():
    class FailingTTS:
        name = "edge-tts"

        async def synthesize(self, text: str, language: str) -> TTSResult:
            raise ProviderError("Connection reset")

    class BackupTTS:
        name = "google-tts"

        async def synthesize(self, text: str, language: str) -> TTSResult:
            return TTSResult(audio_bytes=b"MP3DATA", language=language, provider=self.name)

    mgr = TTSManager([FailingTTS(), BackupTTS()])
    res = await mgr.synthesize("تست صوتی", language="fa")
    assert res.audio_bytes == b"MP3DATA"
    assert res.provider == "google-tts"


@pytest.mark.asyncio
async def test_tts_manager_raises_when_all_providers_fail():
    class FailingTTS:
        name = "failing-tts"

        async def synthesize(self, text: str, language: str) -> TTSResult:
            raise ProviderError("TTS unavailable")

    mgr = TTSManager([FailingTTS()])
    with pytest.raises(ProviderChainExhaustedError):
        await mgr.synthesize("تست", language="fa")
