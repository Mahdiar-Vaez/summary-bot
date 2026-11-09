"""Application entrypoint and provider wiring for Telegram Voice Summary Bot."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from src.bot.cache import TranscriptCache
from src.bot.handlers import create_bot_router
from src.config import Settings
from src.services.stt.base import STTProvider
from src.services.stt.cloudflare_stt import CloudflareWhisperProvider
from src.services.stt.gemini_stt import GeminiAudioSTTProvider
from src.services.stt.groq_stt import GroqWhisperProvider
from src.services.stt.manager import STTManager
from src.services.summarizer.base import SummarizerProvider
from src.services.summarizer.gemini_llm import GeminiSummarizerProvider
from src.services.summarizer.groq_llm import GroqSummarizerProvider
from src.services.summarizer.manager import SummarizerManager
from src.services.summarizer.openrouter_llm import OpenRouterSummarizerProvider
from src.services.tts.base import TTSProvider
from src.services.tts.edge_tts_provider import EdgeTTSProvider
from src.services.tts.gemini_tts import GeminiTTSProvider
from src.services.tts.google_tts import GoogleTranslateTTSProvider
from src.services.tts.manager import TTSManager

GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
)

GEMINI_TTS_MODELS = (
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
)


def build_stt_manager(settings: Settings) -> STTManager:
    """Build STTManager fallback chain from configured credentials."""
    providers: list[STTProvider] = []
    if settings.groq_api_key and settings.groq_api_key.strip():
        providers.append(GroqWhisperProvider(api_key=settings.groq_api_key.strip()))
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        gkey = settings.gemini_api_key.strip()
        for model_name in GEMINI_FALLBACK_MODELS:
            providers.append(GeminiAudioSTTProvider(api_key=gkey, model=model_name))
    if (
        settings.cloudflare_account_id
        and settings.cloudflare_account_id.strip()
        and settings.cloudflare_api_token
        and settings.cloudflare_api_token.strip()
    ):
        providers.append(
            CloudflareWhisperProvider(
                account_id=settings.cloudflare_account_id.strip(),
                api_token=settings.cloudflare_api_token.strip(),
            )
        )
    return STTManager(providers)


def build_summarizer_manager(settings: Settings) -> SummarizerManager:
    """Build SummarizerManager fallback chain from configured credentials."""
    providers: list[SummarizerProvider] = []
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        gkey = settings.gemini_api_key.strip()
        for model_name in GEMINI_FALLBACK_MODELS:
            providers.append(GeminiSummarizerProvider(api_key=gkey, model=model_name))
    if settings.groq_api_key and settings.groq_api_key.strip():
        providers.append(GroqSummarizerProvider(api_key=settings.groq_api_key.strip()))
    if settings.openrouter_api_key and settings.openrouter_api_key.strip():
        providers.append(
            OpenRouterSummarizerProvider(api_key=settings.openrouter_api_key.strip())
        )
    return SummarizerManager(providers)


def build_tts_manager(settings: Settings | None = None) -> TTSManager:
    """Build TTSManager fallback chain (Gemini 3.8/3.1 Flash TTS -> Edge Neural TTS -> Google TTS)."""
    providers: list[TTSProvider] = []
    if settings is not None and settings.gemini_api_key and settings.gemini_api_key.strip():
        gkey = settings.gemini_api_key.strip()
        for tts_model in GEMINI_TTS_MODELS:
            providers.append(GeminiTTSProvider(api_key=gkey, model=tts_model))
    providers.append(EdgeTTSProvider())
    providers.append(GoogleTranslateTTSProvider())
    return TTSManager(providers)


def build_dispatcher(settings: Settings) -> Dispatcher:
    """Create and configure the aiogram Dispatcher."""
    stt_manager = build_stt_manager(settings)
    summarizer_manager = build_summarizer_manager(settings)
    tts_manager = build_tts_manager(settings)
    cache = TranscriptCache(
        ttl_seconds=settings.transcript_cache_ttl_seconds,
        max_items=settings.transcript_cache_max_items,
        persistence_path="data/chat_history.json",
    )
    dp = Dispatcher()
    dp.include_router(
        create_bot_router(
            stt_manager=stt_manager,
            summarizer_manager=summarizer_manager,
            cache=cache,
            tts_manager=tts_manager,
            max_file_size_mb=settings.max_file_size_mb,
        )
    )
    return dp


async def configure_bot_profile(bot: Bot) -> None:
    """Register bot commands, description, and short description on Telegram."""
    try:
        await bot.set_my_short_description(
            short_description=(
                "🎙️ خلاصه‌ساز هوشمند ویس (متنی و صوتی) + خلاصه‌ساز ۱۰ تا ۱۵۰ پیام آخر گروه و چنل 💬"
            )
        )
        await bot.set_my_description(
            description=(
                "🤖 دستیار هوشمند خلاصه‌ساز ویس و پیام‌های گروه و چنل (@vaez_summary_bot)\n\n"
                "✨ قابلیت‌های ربات:\n"
                "۱. 🎙️ خلاصه‌سازی فوق‌سریع ویس و فایل‌های صوتی (فارسی و انگلیسی) به زبان خودمونی و روان\n"
                "۲. 💬 خلاصه‌سازی پیام‌های گروه و چنل (از ۱۰ تا ۱۰۰ و حداکثر ۱۵۰ پیام آخر به انتخاب شما!) با دستور /chat_summary\n"
                "۳. 🔊 تبدیل خلاصه‌ی ویس‌ها به فایل صوتی طبیعی (شنیدن خلاصه به صورت ویس)\n"
                "۴. 📜 دریافت متن کامل پیاده‌شده‌ی ویس‌ها\n\n"
                "🚀 کافیست ویس بفرستید، در گروه روی ویس ریپلای بزنید، یا دستور /chat_summary را امتحان کنید!"
            )
        )
        await bot.set_my_commands(
            [
                BotCommand(
                    command="summary",
                    description="خلاصه‌سازی ویس (با ریپلای) یا انتخاب تعداد پیام",
                ),
                BotCommand(
                    command="chat_summary",
                    description="انتخاب تعداد پیام برای خلاصه‌سازی از کیبورد پایین",
                ),
                BotCommand(
                    command="summary_10",
                    description="خلاصه‌سازی ۱۰ پیام آخر گروه یا چنل",
                ),
                BotCommand(
                    command="summary_50",
                    description="خلاصه‌سازی ۵۰ پیام آخر گروه یا چنل",
                ),
                BotCommand(
                    command="summary_100",
                    description="خلاصه‌سازی ۱۰۰ پیام آخر گروه یا چنل",
                ),
                BotCommand(
                    command="summary_150",
                    description="خلاصه‌سازی ۱۵۰ پیام آخر (حداکثر)",
                ),
                BotCommand(
                    command="start",
                    description="راهنما و معرفی قابلیت‌های ربات",
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not update Telegram bot profile metadata: %s", exc)


async def main() -> None:
    """Run the Telegram bot using long polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    settings = Settings()
    if not settings.has_any_stt_provider():
        raise RuntimeError(
            "No STT provider configured! Please set GEMINI_API_KEY or GROQ_API_KEY in your .env file."
        )
    if not settings.has_any_llm_provider():
        raise RuntimeError(
            "No LLM provider configured! Please set GEMINI_API_KEY or GROQ_API_KEY in your .env file."
        )

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await configure_bot_profile(bot)
    dp = build_dispatcher(settings)
    logging.info("Starting Telegram Voice Summary Bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
