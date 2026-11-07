# Telegram Voice Summary Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an asynchronous Telegram bot that transcribes and summarizes Persian and English voice/audio messages (direct, forwarded, or replied-to) using a multi-provider cloud API fallback chain with speaker attribution.

**Architecture:** Lightweight Python `aiogram 3` gateway orchestrating a two-stage pipeline: an `STTManager` fallback chain (`Groq Whisper-large-v3` -> `Cloudflare Whisper`) and a `SummarizerManager` fallback chain (`Google Gemini 2.0 Flash` -> `Groq Llama 3.3 70B` -> `OpenRouter Free`), paired with a bounded TTL `TranscriptCache` for full-transcript retrieval via inline buttons.

**Tech Stack:** Python `3.12`, `aiogram >=3.10`, `httpx >=0.27`, `pydantic-settings >=2.0`, `pytest >=8.0`, `pytest-asyncio >=0.23`.

**Spec:** `docs/superpowers/specs/2026-11-07-telegram-voice-summary-design.md`

## Global Constraints
- Runtime: Python `>=3.11` using `aiogram >=3.10`, `httpx >=0.27`, `pydantic-settings >=2.0`.
- Standard Bot API file download limit: `20 MB` (`max_file_size_mb: int = 20`).
- Downloaded temporary audio files must always be deleted in a `finally` block.
- Both `STTManager` and `SummarizerManager` must iterate through providers in order and raise `ProviderChainExhaustedError` if all fail or return blank output.

## Review Focus
- Empty or whitespace-only transcription returned by an STT provider must be treated as a provider failure and trigger fallback to the next STT provider rather than passing blank text to the summarizer.
- Voice messages forwarded from users who hid their forward privacy (`MessageOriginHiddenUser`) or from channels/chats (`MessageOriginChannel`/`MessageOriginChat`) must still extract the display name accurately without raising `AttributeError`.
- Transcripts longer than Telegram's 4096-character limit must be split into safe chunks (`<= 4000` chars) when the user clicks `📜 مشاهده متن کامل ویس`.
- Expired or missing `cache_id` on inline callback query must answer the callback gracefully with a Persian alert message instead of crashing.
- Temporary audio files on disk must be removed even when `STTManager` or `SummarizerManager` raises `ProviderChainExhaustedError`.

---

### Task 1: Project Environment, Configuration, and Custom Exceptions

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/exceptions.py`
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `src.exceptions.VoiceBotError(Exception)`
  - `src.exceptions.ProviderError(VoiceBotError)`
  - `src.exceptions.ProviderChainExhaustedError(VoiceBotError)`
  - `src.exceptions.FileTooLargeError(VoiceBotError)`
  - `src.config.Settings(BaseSettings)` with fields `telegram_bot_token: str`, `groq_api_key: str | None`, `gemini_api_key: str | None`, `openrouter_api_key: str | None`, `cloudflare_account_id: str | None`, `cloudflare_api_token: str | None`, `max_file_size_mb: int = 20`, `transcript_cache_ttl_seconds: int = 3600`, `transcript_cache_max_items: int = 500`, `has_any_stt_provider() -> bool`, `has_any_llm_provider() -> bool`

- [ ] **Step 1: Create virtualenv, install dependencies, and write the failing test in `tests/test_config.py`**

```python
import pytest
from src.config import Settings
from src.exceptions import ProviderChainExhaustedError, FileTooLargeError


def test_settings_defaults_and_provider_checks():
    settings = Settings(
        telegram_bot_token="123:ABC",
        groq_api_key="gsk_test",
        _env_file=None,
    )
    assert settings.max_file_size_mb == 20
    assert settings.transcript_cache_ttl_seconds == 3600
    assert settings.has_any_stt_provider() is True
    assert settings.has_any_llm_provider() is True


def test_settings_without_providers_reports_false():
    settings = Settings(
        telegram_bot_token="123:ABC",
        groq_api_key=None,
        gemini_api_key=None,
        openrouter_api_key=None,
        cloudflare_account_id=None,
        cloudflare_api_token=None,
        _env_file=None,
    )
    assert settings.has_any_stt_provider() is False
    assert settings.has_any_llm_provider() is False


def test_provider_chain_exhausted_error_stores_errors():
    err = ProviderChainExhaustedError(["groq: 429", "cf: 500"])
    assert "groq: 429" in str(err)
    assert len(err.errors) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Implement `src/exceptions.py` and `src/config.py`**

Implement `Settings` using `pydantic_settings.BaseSettings` and `SettingsConfigDict(env_file=".env", extra="ignore")`, plus the exception hierarchy in `src/exceptions.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add .gitignore .env.example requirements.txt pyproject.toml src/__init__.py src/exceptions.py src/config.py tests/test_config.py
git commit -m "feat: add project configuration and exception classes"
```

---

### Task 2: STT Providers and Fallback Chain (`STTManager`)

**Files:**
- Create: `src/services/__init__.py`
- Create: `src/services/stt/__init__.py`
- Create: `src/services/stt/base.py`
- Create: `src/services/stt/groq_stt.py`
- Create: `src/services/stt/cloudflare_stt.py`
- Create: `src/services/stt/manager.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Consumes: `src.exceptions.ProviderError`, `src.exceptions.ProviderChainExhaustedError`
- Produces:
  - `src.services.stt.base.TranscriptionResult(text: str, language: str, provider: str)`
  - `src.services.stt.base.STTProvider(Protocol)` with `name: str` and `async def transcribe(self, audio_path: Path) -> TranscriptionResult`
  - `src.services.stt.groq_stt.GroqWhisperProvider(api_key: str, model: str = "whisper-large-v3", client: httpx.AsyncClient | None = None)`
  - `src.services.stt.cloudflare_stt.CloudflareWhisperProvider(account_id: str, api_token: str, client: httpx.AsyncClient | None = None)`
  - `src.services.stt.manager.STTManager(providers: Sequence[STTProvider])` with `async def transcribe(self, audio_path: Path) -> TranscriptionResult`

- [ ] **Step 1: Write the failing tests in `tests/test_stt.py`**

```python
from pathlib import Path
import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.stt.base import TranscriptionResult
from src.services.stt.groq_stt import GroqWhisperProvider
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_stt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services'`

- [ ] **Step 3: Implement STT base, Groq provider, Cloudflare provider, and `STTManager`**

Normalize detected language in `normalize_language(raw_lang: str, text: str) -> str`: if `raw_lang.lower()` in `("fa", "pes", "persian", "farsi")` or Persian Unicode characters (`\u0600`–`\u06FF`) make up `> 20%` of letters in `text`, return `"fa"`, else `"en"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_stt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/ tests/test_stt.py
git commit -m "feat: implement STT providers and fallback manager"
```

---

### Task 3: Summarizer Prompts, LLM Providers, and Fallback Chain (`SummarizerManager`)

**Files:**
- Create: `src/services/summarizer/__init__.py`
- Create: `src/services/summarizer/base.py`
- Create: `src/services/summarizer/prompts.py`
- Create: `src/services/summarizer/gemini_llm.py`
- Create: `src/services/summarizer/groq_llm.py`
- Create: `src/services/summarizer/openrouter_llm.py`
- Create: `src/services/summarizer/manager.py`
- Test: `tests/test_summarizer.py`

**Interfaces:**
- Consumes: `src.exceptions.ProviderError`, `src.exceptions.ProviderChainExhaustedError`
- Produces:
  - `src.services.summarizer.base.SummaryResult(summary: str, language: str, provider: str)`
  - `src.services.summarizer.base.SummarizerProvider(Protocol)` with `name: str` and `async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult`
  - `src.services.summarizer.prompts.build_summary_prompt(transcript: str, speaker_name: str, language: str) -> tuple[str, str]`
  - `src.services.summarizer.gemini_llm.GeminiSummarizerProvider(api_key: str, model: str = "gemini-2.0-flash", client: httpx.AsyncClient | None = None)`
  - `src.services.summarizer.groq_llm.GroqSummarizerProvider(api_key: str, model: str = "llama-3.3-70b-versatile", client: httpx.AsyncClient | None = None)`
  - `src.services.summarizer.openrouter_llm.OpenRouterSummarizerProvider(api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct:free", client: httpx.AsyncClient | None = None)`
  - `src.services.summarizer.manager.SummarizerManager(providers: Sequence[SummarizerProvider])` with `async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult`

- [ ] **Step 1: Write the failing tests in `tests/test_summarizer.py`**

```python
import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.summarizer.prompts import build_summary_prompt
from src.services.summarizer.gemini_llm import GeminiSummarizerProvider
from src.services.summarizer.groq_llm import GroqSummarizerProvider
from src.services.summarizer.openrouter_llm import OpenRouterSummarizerProvider
from src.services.summarizer.manager import SummarizerManager


def test_build_summary_prompt_persian_includes_speaker_attribution():
    system_prompt, user_prompt = build_summary_prompt(
        transcript="سلام، فردا ساعت ده جلسه داریم و باید گزارش فروش آماده بشه.",
        speaker_name="مهدیار",
        language="fa",
    )
    assert "مهدیار میگه که" in system_prompt
    assert "📌 خلاصه صحبت‌های مهدیار" in system_prompt
    assert "مهدیار" in user_prompt


def test_build_summary_prompt_english_includes_speaker_attribution():
    system_prompt, user_prompt = build_summary_prompt(
        transcript="Hey team, let's deploy the release tomorrow morning at 9am.",
        speaker_name="Alex",
        language="en",
    )
    assert "Alex says that" in system_prompt
    assert "📌 Summary of Alex's voice" in system_prompt


@pytest.mark.asyncio
async def test_summarizer_manager_falls_back_from_gemini_to_groq():
    def gemini_fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Quota exceeded"}})

    def groq_ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "📌 خلاصه صحبت‌های مهدیار:\nمهدیار میگه که فردا جلسه داریم."}}
                ]
            },
        )

    gemini = GeminiSummarizerProvider(
        api_key="AIza_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gemini_fail_handler)),
    )
    groq = GroqSummarizerProvider(
        api_key="gsk_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(groq_ok_handler)),
    )

    manager = SummarizerManager([gemini, groq])
    result = await manager.summarize(
        transcript="سلام فردا جلسه داریم",
        speaker_name="مهدیار",
        language="fa",
    )
    assert result.provider == "groq-llm"
    assert "مهدیار میگه که" in result.summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_summarizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.summarizer'`

- [ ] **Step 3: Implement prompts, Gemini/Groq/OpenRouter providers, and `SummarizerManager`**

Implement `build_summary_prompt`, `GeminiSummarizerProvider`, `GroqSummarizerProvider`, `OpenRouterSummarizerProvider`, and `SummarizerManager`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_summarizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/summarizer/ tests/test_summarizer.py
git commit -m "feat: implement LLM summarizer providers, prompts, and fallback chain"
```

---

### Task 4: Telegram Speaker Extractor, Transcript Cache, and Bot Handlers

**Files:**
- Create: `src/bot/__init__.py`
- Create: `src/bot/cache.py`
- Create: `src/bot/extractors.py`
- Create: `src/bot/keyboards.py`
- Create: `src/bot/handlers.py`
- Create: `src/main.py`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `README.md`
- Test: `tests/test_bot.py`

**Interfaces:**
- Consumes: `STTManager`, `SummarizerManager`, `Settings`, `ProviderChainExhaustedError`, `FileTooLargeError`
- Produces:
  - `src.bot.cache.TranscriptCache(ttl_seconds: int = 3600, max_items: int = 500)` with `put(speaker_name: str, text: str) -> str`, `get(cache_id: str) -> TranscriptEntry | None`, and `chunk_text(text: str, max_len: int = 4000) -> list[str]`
  - `src.bot.extractors.extract_voice_target(message: Any) -> tuple[Any, str] | None`
  - `src.bot.handlers.process_voice_request(message: Any, bot: Any, stt_manager: STTManager, summarizer_manager: SummarizerManager, cache: TranscriptCache, max_file_size_mb: int = 20) -> None`
  - `src.main.build_app(settings: Settings)`

- [ ] **Step 1: Write the failing tests in `tests/test_bot.py`**

```python
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.bot.cache import TranscriptCache, chunk_text
from src.bot.extractors import extract_voice_target
from src.bot.handlers import process_voice_request
from src.services.stt.base import TranscriptionResult
from src.services.summarizer.base import SummaryResult


def test_extract_voice_target_from_reply_and_hidden_forward():
    replied_msg = SimpleNamespace(
        voice=SimpleNamespace(file_id="v1", file_size=1024, duration=120),
        audio=None,
        forward_origin=SimpleNamespace(
            type="hidden_user",
            sender_user_name="مهدیار",
            sender_user=None,
            sender_chat=None,
            chat=None,
        ),
        from_user=SimpleNamespace(full_name="Ali", first_name="Ali"),
    )
    msg = SimpleNamespace(
        voice=None,
        audio=None,
        reply_to_message=replied_msg,
        from_user=SimpleNamespace(full_name="Reza", first_name="Reza"),
    )
    target = extract_voice_target(msg)
    assert target is not None
    media, speaker = target
    assert media.file_id == "v1"
    assert speaker == "مهدیار"


def test_transcript_cache_and_chunking():
    cache = TranscriptCache(ttl_seconds=60, max_items=10)
    cid = cache.put(speaker_name="مهدیار", text="A" * 8500)
    entry = cache.get(cid)
    assert entry is not None
    assert entry.speaker_name == "مهدیار"
    chunks = chunk_text(entry.text, max_len=4000)
    assert len(chunks) == 3
    assert all(len(c) <= 4000 for c in chunks)


@pytest.mark.asyncio
async def test_process_voice_request_cleans_up_temp_file_and_edits_message():
    status_edits = []

    class DummyStatusMsg:
        async def edit_text(self, text: str, reply_markup=None):
            status_edits.append((text, reply_markup))

    class DummyMessage:
        voice = SimpleNamespace(file_id="voice_123", file_size=500_000, duration=300)
        audio = None
        reply_to_message = None
        forward_origin = None
        from_user = SimpleNamespace(full_name="مهدیار", first_name="مهدیار")

        async def reply(self, text: str):
            return DummyStatusMsg()

    class DummyBot:
        async def get_file(self, file_id: str):
            return SimpleNamespace(file_path="voice/file_1.oga")

        async def download_file(self, file_path: str, destination: Path):
            Path(destination).write_bytes(b"OggS-voice-content")

    class DummySTT:
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            assert audio_path.exists()
            return TranscriptionResult(text="سلام من مهدیارم و می‌خوام پروژه رو استارت بزنیم", language="fa", provider="groq")

    class DummySummarizer:
        async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult:
            return SummaryResult(
                summary=f"📌 خلاصه صحبت‌های {speaker_name}:\n{speaker_name} میگه که پروژه رو استارت بزنیم.",
                language="fa",
                provider="gemini",
            )

    cache = TranscriptCache()
    await process_voice_request(
        message=DummyMessage(),
        bot=DummyBot(),
        stt_manager=DummySTT(),
        summarizer_manager=DummySummarizer(),
        cache=cache,
        max_file_size_mb=20,
    )

    assert len(status_edits) >= 2
    final_text, final_markup = status_edits[-1]
    assert "مهدیار میگه که" in final_text
    assert final_markup is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_bot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bot'`

- [ ] **Step 3: Implement `cache.py`, `extractors.py`, `keyboards.py`, `handlers.py`, `main.py`, and Docker files**

Implement all Telegram handlers, temporary file cleanup in `finally`, inline callback handler for full transcript display, `main.py` bootstrap, `Dockerfile`, `docker-compose.yml`, and Persian `README.md`.

- [ ] **Step 4: Run full test suite to verify everything passes**

Run: `.venv\Scripts\pytest.exe -v`
Expected: PASS (all tests across all files pass)

- [ ] **Step 5: Commit**

```bash
git add src/bot/ src/main.py Dockerfile docker-compose.yml README.md tests/test_bot.py
git commit -m "feat: implement Telegram bot handlers, speaker extractor, cache, and Docker deployment"
```
