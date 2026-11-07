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
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None

    max_file_size_mb: int = 20
    transcript_cache_ttl_seconds: int = 3600
    transcript_cache_max_items: int = 500

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
        return any(
            bool(key and key.strip())
            for key in (self.gemini_api_key, self.groq_api_key, self.openrouter_api_key)
        )
