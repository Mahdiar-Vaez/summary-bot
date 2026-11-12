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


def test_settings_with_only_gemini_key_has_stt_and_llm():
    settings = Settings(
        telegram_bot_token="123:ABC",
        gemini_api_key="AIza_test",
        _env_file=None,
    )
    assert settings.has_any_stt_provider() is True
    assert settings.has_any_llm_provider() is True


def test_settings_without_providers_reports_false():
    settings = Settings(
        telegram_bot_token="123:ABC",
        groq_api_key=None,
        gemini_api_key=None,
        openrouter_api_key=None,
        fallback_llm_api_key=None,
        cloudflare_account_id=None,
        cloudflare_api_token=None,
        _env_file=None,
    )
    assert settings.has_any_stt_provider() is False
    assert settings.has_any_llm_provider() is False


def test_settings_with_only_fallback_llm_key_has_llm():
    settings = Settings(
        telegram_bot_token="123:ABC",
        gemini_api_key=None,
        groq_api_key=None,
        openrouter_api_key=None,
        fallback_llm_api_key="sk-test-123",
        _env_file=None,
    )
    assert settings.has_any_llm_provider() is True


def test_provider_chain_exhausted_error_stores_errors():
    err = ProviderChainExhaustedError(["groq: 429", "cf: 500"])
    assert "groq: 429" in str(err)
    assert len(err.errors) == 2
