# Telegram Voice Summary Bot — Design Document

## 1. Intent and Why
Users in Telegram frequently receive or encounter 5-to-10-minute voice messages (`voice` or `audio`) in private chats and groups. Listening to long voice messages is time-consuming. This project builds a fast, lightweight Telegram bot that accepts voice messages (directly, forwarded, or via reply in groups/private chats), transcribes them, detects their language (Persian or English), and returns a concise, speaker-attributed summary (e.g., *"📌 خلاصه صحبت‌های مهدیار: مهدیار میگه که..."*) along with an inline button to view the full transcript on demand.

The user's server acts strictly as a lightweight asynchronous orchestrator (gateway) that consumes external free-tier AI APIs with multi-provider automatic fallback chains, requiring zero local GPU or heavy CPU computation.

## 2. Goals, Non-Goals, and Anti-Goals
### Goals
- Support direct voice/audio messages in private chat, forwarded voices, and `/summary` (or bot mention) replies to voice messages in groups.
- Extract the actual speaker's display name from Telegram message metadata (`forward_origin`, `reply_to_message.from_user`, or `from_user`).
- Transcribe audio via a cloud STT `FallbackChain` (`Groq Whisper-large-v3` primary -> `Cloudflare Workers AI Whisper` / OpenAI-compatible STT secondary).
- Summarize the transcript via a cloud LLM `FallbackChain` (`Google Gemini 2.0 Flash` primary -> `Groq Llama-3.3-70b-versatile` secondary -> `OpenRouter Free` tertiary).
- Automatically detect Persian vs. English and format the summary naturally in the detected language, attributing statements to the speaker.
- Provide live status updates in Telegram (*"📥 در حال دریافت ویس..."* -> *"🎙️ در حال تبدیل گفتار به متن..."* -> *"🧠 در حال نگارش خلاصه..."*) and an inline button (*"📜 مشاهده متن کامل ویس"*) to retrieve the full transcript from an in-memory TTL cache.
- Provide production-ready `Dockerfile`, `docker-compose.yml`, and `.env.example` for instant server deployment.

### Non-Goals
- Running heavy local ML models (`faster-whisper`, `ollama`) on the server.
- Persistent database storage of user voice recordings (audio files are deleted immediately after processing; transcripts are kept only in an in-memory TTL cache for the inline button).

### Anti-Goals
- Failing visible requests when a single free API provider hits a `429 Rate Limit` or timeout (the fallback chain must transparently switch to the next configured provider).
- Blocking the asyncio event loop during network I/O or file operations.

## 3. Constraints
- **Runtime:** Python `>=3.11` using `aiogram >=3.10`, `httpx >=0.27`, `pydantic-settings >=2.0`.
- **Telegram Bot API File Size Limit:** Standard Bot API limits file downloads to `20 MB`. The bot checks `file_size` before downloading and returns a clear user-friendly message if exceeded (unless a custom Local Bot API URL is configured).
- **Audio Cleanup:** Downloaded temporary `.ogg`/`.oga`/`.mp3` files must always be removed in a `finally` block.
- **Zero Single-Provider Lock-in:** Both STT and Summarizer managers must accept an ordered list of providers and iterate through them on failure, raising `ProviderChainExhaustedError` only if all configured providers fail.

## 4. Approaches Considered
1. **Approach 1 (Chosen): Two-Stage Modular Pipeline with Independent Fallback Chains (STT + LLM)**
   - Stage 1 converts audio to text and detects language via `Groq Whisper-large-v3` (with fallback to Cloudflare/OpenAI-compatible STT).
   - Stage 2 summarizes the text with speaker attribution via `Gemini Flash` (with fallback to `Groq Llama-3.3-70B` and `OpenRouter`).
   - *Why chosen:* Fastest transcription (`Groq Whisper` processes 10 minutes of audio in ~3 seconds), independent failover per stage, and immediate availability of the verbatim transcript for the *"مشاهده متن کامل"* button.
2. **Approach 2: Direct Multimodal Audio-to-Summary (`Gemini Audio`)**
   - Uploads audio directly to Gemini in one prompt.
   - *Why rejected:* Slower audio upload/processing for long files, harder to provide an instant verbatim transcript alongside the summary, and fewer drop-in free multimodal fallbacks.

## 5. Architecture & Component Specifications

### 5.1 Configuration (`src/config.py`)
`Settings(BaseSettings)` loaded from environment / `.env`:
- `telegram_bot_token: str`
- `groq_api_key: str | None = None`
- `gemini_api_key: str | None = None`
- `openrouter_api_key: str | None = None`
- `cloudflare_account_id: str | None = None`
- `cloudflare_api_token: str | None = None`
- `max_file_size_mb: int = 20`
- `transcript_cache_ttl_seconds: int = 3600`
- ` transcript_cache_max_items: int = 500`

### 5.2 Speaker Extraction (`src/bot/extractors.py`)
- `extract_target_voice_and_speaker(message: Message) -> tuple[Voice | Audio, str] | None`:
  - Inspects `message.reply_to_message` first (if it contains `voice` or `audio`), otherwise inspects `message` itself.
  - Determines speaker name in priority order:
    1. `forward_origin` sender name (`MessageOriginUser.sender_user.full_name`, `MessageOriginHiddenUser.sender_user_name`, `MessageOriginChat.sender_chat.title`, `MessageOriginChannel.chat.title`).
    2. `from_user.full_name` (or `first_name`).
    3. Fallback default: `"گوینده"` (Speaker).

### 5.3 STT Pipeline (`src/services/stt/`)
- `TranscriptionResult( dataclass )`: `text: str`, `language: str`, `provider: str`.
- `STTProvider(Protocol)`: `async def transcribe(self, audio_path: Path) -> TranscriptionResult`.
- `GroqWhisperProvider`: Calls `https://api.groq.com/openai/v1/audio/transcriptions` with `model="whisper-large-v3"`, `response_format="verbose_json"` to obtain both `text` and detected `language`.
- `CloudflareWhisperProvider`: Calls Cloudflare Workers AI `@cf/openai/whisper` endpoint as fallback.
- `STTManager`: Takes `Sequence[STTProvider]`. Tries each provider in order; logs failures and falls back to the next. Raises `ProviderChainExhaustedError` if all fail or if resulting text is empty.

### 5.4 Summarizer Pipeline (`src/services/summarizer/`)
- `SummaryResult( dataclass )`: `summary: str`, `language: str`, `provider: str`.
- `SummarizerProvider(Protocol)`: `async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult`.
- `build_summary_prompt(transcript: str, speaker_name: str, language: str) -> tuple[str, str]`:
  - Returns `(system_prompt, user_prompt)`.
  - Detects whether `language` is Persian (`fa` / `persian` / Persian script detection) or English (`en` / other).
  - For Persian: instructs the model to correct minor STT phonetic errors, start naturally with `📌 خلاصه صحبت‌های {speaker_name}:` followed by `{speaker_name} میگه که...`, and list key points & action items in concise bullet points.
  - For English: starts with `📌 Summary of {speaker_name}'s voice:` followed by `{speaker_name} says that...` and concise bullet points.
- Providers:
  - `GeminiSummarizerProvider`: Calls Google Gemini REST API (`gemini-2.0-flash`).
  - `GroqSummarizerProvider`: Calls Groq Chat Completions API (`llama-3.3-70b-versatile`).
  - `OpenRouterSummarizerProvider`: Calls OpenRouter Chat Completions API (`meta-llama/llama-3.3-70b-instruct:free`).
- `SummarizerManager`: Iterates through configured `SummarizerProvider` instances in order; falls back transparently on network errors, HTTP 4xx/5xx, or empty outputs.

### 5.5 Transcript Cache & Telegram Handlers (`src/bot/`)
- `TranscriptCache`: Bounded in-memory TTL store mapping `cache_id: str` (UUID4 short token) -> `TranscriptEntry(speaker_name, text, created_at)` so clicking the inline button `📜 مشاهده متن کامل ویس` (`callback_data="transcript:<cache_id>"`) sends the full transcript (chunked at 4000 chars if long).
- `VoiceHandler`:
  - Handles private chat voice/audio messages directly.
  - Handles `/summary` or `/خلاصه` commands (or bot `@mention`) when replying to a voice/audio message in groups or private chats.
  - Enforces `max_file_size_mb`, downloads audio to a temporary file, runs `STTManager` -> `SummarizerManager`, edits the progress message with the final summary + inline keyboard, and removes the temporary audio file in `finally`.
