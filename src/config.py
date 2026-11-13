"""Application configuration loaded from environment variables or .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Bot and AI provider configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: str
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    fallback_llm_api_key: str | None = None
    fallback_llm_base_url: str = "https://discovery-api.intern-ai.org.cn/v1"
    fallback_llm_models: str = "deepseek-v4.1-flash-1017,qwen3.8-27b"
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None

    max_file_size_mb: int = 20
    transcript_cache_ttl_seconds: int = 3600
    transcript_cache_max_items: int = 500
    chat_rate_limit_max_requests: int = 5
    chat_rate_limit_window_seconds: int = 300
    admin_chat_id: int | None = None

    def has_any_stt_provider(self) -> bool:
        """Return True if at least one STT provider has valid credentials."""
        has_groq = bool(self.groq_api_key and self.groq_api_key.strip())
        has_gemini = bool(self.gemini_api_key and self.gemini_api_key.strip())
        has_cf = bool(
            self.cloudflare_account_id
            and self.cloudflare_account_id.strip()
            and self.cloudflare_api_token
            and self.cloudflare_api_token.strip()
        )
        return has_groq or has_gemini or has_cf

    def has_any_llm_provider(self) -> bool:
        """Return True if at least one LLM summarizer provider has valid credentials."""
        has_fallback = bool(
            self.fallback_llm_api_key
            and self.fallback_llm_api_key.strip()
            and self.fallback_llm_base_url
            and self.fallback_llm_base_url.strip()
        )
        return has_fallback or any(
            bool(key and key.strip())
            for key in (
                self.gemini_api_key,
                self.groq_api_key,
                self.openrouter_api_key,
            )
        )
