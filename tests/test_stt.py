from pathlib import Path
import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.stt.base import TranscriptionResult, normalize_language
from src.services.stt.groq_stt import GroqWhisperProvider
from src.services.stt.gemini_stt import GeminiAudioSTTProvider
from src.services.stt.cloudflare_stt import CloudflareWhisperProvider
from src.services.stt.manager import STTManager


@pytest.mark.asyncio
async def test_groq_whisper_provider_parses_verbose_json(tmp_path: Path):
    audio_file = tmp_path / "voice.ogg"
    audio_file.write_bytes(b"OggS-fake-audio")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer gsk_test"
        return httpx.Response(200, json={"text": "سلام بچه‌ها چطورید", "language": "persian"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqWhisperProvider(api_key="gsk_test", client=client)
    result = await provider.transcribe(audio_file)

    assert result.text == "سلام بچه‌ها چطورید"
    assert result.language == "fa"
    assert result.provider == "groq"


@pytest.mark.asyncio
async def test_cloudflare_whisper_provider_parses_result(tmp_path: Path):
    audio_file = tmp_path / "voice.ogg"
    audio_file.write_bytes(b"OggS-fake-audio")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer cf_token"
        return httpx.Response(200, json={"result": {"text": "Hello everyone this is a test"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CloudflareWhisperProvider(account_id="acc_123", api_token="cf_token", client=client)
    result = await provider.transcribe(audio_file)

    assert result.text == "Hello everyone this is a test"
    assert result.language == "en"
    assert result.provider == "cloudflare"


@pytest.mark.asyncio
async def test_stt_manager_falls_back_when_first_provider_fails_or_returns_empty(tmp_path: Path):
    audio_file = tmp_path / "voice.ogg"
    audio_file.write_bytes(b"OggS-fake-audio")

    class EmptyProvider:
        name = "empty_stt"
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            return TranscriptionResult(text="   ", language="fa", provider="empty_stt")

    class WorkingProvider:
        name = "backup_stt"
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            return TranscriptionResult(text="متن تست از بکاپ", language="fa", provider="backup_stt")

    manager = STTManager([EmptyProvider(), WorkingProvider()])
    result = await manager.transcribe(audio_file)
    assert result.text == "متن تست از بکاپ"
    assert result.provider == "backup_stt"


@pytest.mark.asyncio
async def test_stt_manager_raises_when_all_providers_fail(tmp_path: Path):
    audio_file = tmp_path / "voice.ogg"
    audio_file.write_bytes(b"OggS")

    class FailingProvider:
        name = "failing_stt"
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            raise ProviderError("HTTP 429 Too Many Requests")

    manager = STTManager([FailingProvider()])
    with pytest.raises(ProviderChainExhaustedError) as exc_info:
        await manager.transcribe(audio_file)
    assert "failing_stt" in str(exc_info.value)


def test_normalize_language_detects_persian_script_even_if_lang_missing():
    assert normalize_language("", "سلام خوبی چه خبر") == "fa"
    assert normalize_language("english", "Hello how are you doing today") == "en"


@pytest.mark.asyncio
async def test_gemini_audio_stt_provider_transcribes_inline_audio(tmp_path: Path):
    audio_file = tmp_path / "voice.oga"
    audio_file.write_bytes(b"OggS-fake-audio")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "سلام مهدیار، جلسه ساعت پنج برگزار میشه."}]
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiAudioSTTProvider(api_key="AIza_test", client=client)
    result = await provider.transcribe(audio_file)
    assert result.text == "سلام مهدیار، جلسه ساعت پنج برگزار میشه."
    assert result.language == "fa"
    assert result.provider == "gemini-stt"
