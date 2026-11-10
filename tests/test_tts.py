import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.tts.base import TTSResult, clean_text_for_tts
from src.services.tts.gemini_tts import GeminiTTSProvider
from src.services.tts.google_tts import GoogleTranslateTTSProvider
from src.services.tts.manager import TTSManager


def test_clean_text_for_tts_removes_markdown_and_emojis():
    raw = "📌 خلاصه صحبت‌های mr.vaez:\n\n• **زمان جلسه:** فردا ساعت ۱۰ صبح\n• آماده‌سازی گزارش"
    cleaned = clean_text_for_tts(raw)
    assert "📌" not in cleaned
    assert "خلاصه صحبت‌های" not in cleaned
    assert "**" not in cleaned
    assert "•" not in cleaned
    assert "زمان جلسه" in cleaned


@pytest.mark.asyncio
async def test_gemini_tts_provider_synthesizes_natural_persian_audio():
    import base64
    import json

    fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt "

    def handler(request: httpx.Request) -> httpx.Response:
        assert "gemini-3.8-flash-tts:generateContent" in str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
        prompt_text = body["contents"][0]["parts"][0]["text"]
        assert "زمان جلسه" in prompt_text
        assert "خلاصه صحبت‌های" not in prompt_text
        assert "متن زیر را" not in prompt_text
        assert "بخوان" not in prompt_text
        assert prompt_text == "زمان جلسه: فردا ساعت ۱۰ صبح"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/wav",
                                        "data": base64.b64encode(fake_wav).decode("ascii"),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiTTSProvider(
        api_key="AQ.test_key",
        model="gemini-3.8-flash-tts",
        client=client,
    )
    res = await provider.synthesize(
        "📌 خلاصه صحبت‌های mr.vaez:\n• **زمان جلسه:** فردا ساعت ۱۰ صبح",
        language="fa",
    )
    assert res.audio_bytes == fake_wav
    assert res.language == "fa"
    assert res.provider == "gemini-tts:gemini-3.8-flash-tts"


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
