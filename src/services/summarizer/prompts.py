"""Prompt templates for Persian and English voice message summarization."""

from __future__ import annotations

from src.services.stt.base import normalize_language


def build_summary_prompt(
    transcript: str, speaker_name: str, language: str
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) with speaker name ONLY in the top header."""
    speaker = (speaker_name or "گوینده").strip()
    norm_lang = normalize_language(language, transcript)

    if norm_lang == "fa":
        system_prompt = (
            "تو یک دستیار هوشمند خلاصه‌ساز پیام‌های صوتی تلگرام هستی.\n"
            "وظیفه تو این است که متن پیاده‌شده‌ی ویس را بخوانی، خطاهای املایی یا شنیداری احتمالی را در ذهن خود اصلاح کنی "
            "و یک خلاصه‌ی بسیار روان، دقیق و کاربردی به زبان فارسی ارائه دهی.\n\n"
            "قوانین اکید قالب خروجی:\n"
            f"۱. خط اول دقیقاً و فقط همین عبارت باشد: «📌 خلاصه صحبت‌های {speaker}:»\n"
            "۲. در متن خلاصه به هیچ وجه نام گوینده را نیاور و از عباراتی مثل «فلانی میگه که...» یا «گوینده میگه...» استفاده نکن؛ "
            "بلکه مستقیماً اصل موضوع و محتوای صحبت‌ها را بنویس.\n"
            "۳. اگر ویس شامل جزئیات، تصمیمات یا کارهای مهم است، آن‌ها را در قالب بولت‌پوینت‌های کوتاه و شفاف (•) فهرست کن.\n"
            "۴. از زیاده‌گویی پرهیز کن تا کاربر در چند ثانیه کل منظور ویس را متوجه شود."
        )
        user_prompt = (
            f"متن پیاده‌شده از ویس:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"لطفاً خلاصه این ویس را فقط با تیتر بالای «📌 خلاصه صحبت‌های {speaker}:» و بدون ذکر نام در داخل متن بنویس."
        )
    else:
        system_prompt = (
            "You are an intelligent Telegram voice message summarizer.\n"
            "Read the transcribed voice message, mentally fix any minor speech-to-text transcription errors, "
            "and write a clear, natural, and concise summary in English.\n\n"
            "Strict output format rules:\n"
            f"1. Start the first line with: '📌 Summary of {speaker}'s voice:'\n"
            "2. Do NOT mention the speaker's name in the summary body and do NOT write phrases like 'X says that...'. "
            "Go straight into summarizing the content and key points.\n"
            "3. List key points, decisions, or action items as short bullet points (•).\n"
            "4. Keep it concise and easy to scan in seconds."
        )
        user_prompt = (
            f"Voice transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"Please provide the summary starting with '📌 Summary of {speaker}'s voice:' without repeating the speaker's name in the body."
        )

    return system_prompt, user_prompt
