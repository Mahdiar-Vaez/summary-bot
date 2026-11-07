"""Application entrypoint and provider wiring for Telegram Voice Summary Bot."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from src.bot.cache import TranscriptCache
from src.bot.handlers import create_bot_router
from src.config import Settings
from src.services.stt.base import STTProvider
from src.services.stt.cloudflare_stt import CloudflareWhisperProvider
from src.services.stt.groq_stt import GroqWhisperProvider
from src.services.stt.manager import STTManager
from src.services.summarizer.base import SummarizerProvider
from src.services.summarizer.gemini_llm import GeminiSummarizerProvider
from src.services.summarizer.groq_llm import GroqSummarizerProvider
from src.services.summarizer.manager import SummarizerManager
from src.services.summarizer.openrouter_llm import OpenRouterSummarizerProvider


def build_stt_manager(settings: Settings) -> STTManager:
    """Build STTManager fallback chain from configured credentials."""
    providers: list[STTProvider] = []
    if settings.groq_api_key and settings.groq_api_key.strip():
        providers.append(GroqWhisperProvider(api_key=settings.groq_api_key.strip()))
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
        providers.append(
            GeminiSummarizerProvider(api_key=settings.gemini_api_key.strip())
        )
    if settings.groq_api_key and settings.groq_api_key.strip():
        providers.append(GroqSummarizerProvider(api_key=settings.groq_api_key.strip()))
    if settings.openrouter_api_key and settings.openrouter_api_key.strip():
        providers.append(
            OpenRouterSummarizerProvider(api_key=settings.openrouter_api_key.strip())
        )
    return SummarizerManager(providers)


def build_dispatcher(settings: Settings) -> Dispatcher:
    """Create and configure the aiogram Dispatcher."""
    stt_manager = build_stt_manager(settings)
    summarizer_manager = build_summarizer_manager(settings)
    cache = TranscriptCache(
        ttl_seconds=settings.transcript_cache_ttl_seconds,
        max_items=settings.transcript_cache_max_items,
    )
    dp = Dispatcher()
    dp.include_router(
        create_bot_router(
            stt_manager=stt_manager,
            summarizer_manager=summarizer_manager,
            cache=cache,
            max_file_size_mb=settings.max_file_size_mb,
        )
    )
    return dp


async def main() -> None:
    """Run the Telegram bot using long polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    settings = Settings()
    if not settings.has_any_stt_provider():
        raise RuntimeError(
            "No STT provider configured! Please set GROQ_API_KEY in your .env file."
        )
    if not settings.has_any_llm_provider():
        raise RuntimeError(
            "No LLM provider configured! Please set GEMINI_API_KEY or GROQ_API_KEY in your .env file."
        )

    bot = Bot(token=settings.telegram_bot_token)
    dp = build_dispatcher(settings)
    logging.info("Starting Telegram Voice Summary Bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
