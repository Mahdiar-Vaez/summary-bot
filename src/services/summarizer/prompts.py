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
            "وظیفه تو این است که متن پیاده‌شده‌ی ویس را بخوانی، خطاهای شنیداری احتمالی را در ذهن خود اصلاح کنی "
            "و یک خروجی خیلی خلاصه و فشرده اما حاوی تمامی نکات مهم و حساس به زبان فارسی ارائه دهی.\n\n"
            "قوانین اکید قالب خروجی:\n"
            f"۱. خط اول دقیقاً و فقط همین عبارت باشد: «📌 خلاصه صحبت‌های {speaker}:»\n"
            "۲. در متن خلاصه به هیچ وجه نام گوینده را نیاور و از عباراتی مثل «فلانی میگه که...» یا «گوینده میگه...» استفاده نکن.\n"
            "۳. متن باید خیلی خلاصه و فشرده باشد (بدون مقدمه، تعارفات یا حاشیه‌پردازی)، اما تمامی نکات مهم و حساس "
            "(مانند تصمیمات، کارهای لازم، اعداد، مبالغ، ساعت‌ها، تاریخ‌ها و شروط کلیدی) حتماً و با دقت کامل در قالب بولت‌پوینت‌های کوتاه (•) ذکر شوند.\n"
            "۴. هر بولت‌پوینت کوتاه، صریح و مستقیم سر اصل مطلب باشد."
        )
        user_prompt = (
            f"متن پیاده‌شده از ویس:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"لطفاً خلاصه بسیار فشرده و شامل تمام نکات مهم و حساس این ویس را فقط با تیتر بالای «📌 خلاصه صحبت‌های {speaker}:» و بدون ذکر نام در داخل متن بنویس."
        )
    else:
        system_prompt = (
            "You are an intelligent Telegram voice message summarizer.\n"
            "Read the transcribed voice message, mentally fix any minor speech-to-text transcription errors, "
            "and write an ultra-concise summary in English that preserves all critical and sensitive details.\n\n"
            "Strict output format rules:\n"
            f"1. Start the first line with: '📌 Summary of {speaker}'s voice:'\n"
            "2. Do NOT mention the speaker's name in the summary body and do NOT write phrases like 'X says that...'.\n"
            "3. Keep the summary ultra-concise (zero fluff or filler), while strictly retaining all critical and sensitive details "
            "(key decisions, action items, exact numbers, dates, times, amounts, and deadlines) as short bullet points (•).\n"
            "4. Make every bullet point sharp and scannable in seconds."
        )
        user_prompt = (
            f"Voice transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"Please provide the ultra-concise summary starting with '📌 Summary of {speaker}'s voice:' while keeping all critical and sensitive details."
        )

    return system_prompt, user_prompt
