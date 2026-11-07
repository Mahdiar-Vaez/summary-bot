"""Prompt templates for Persian and English voice message summarization."""

from __future__ import annotations

from src.services.stt.base import normalize_language


def build_summary_prompt(
    transcript: str, speaker_name: str, language: str
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) tailored to speaker_name and detected language."""
    speaker = (speaker_name or "گوینده").strip()
    norm_lang = normalize_language(language, transcript)

    if norm_lang == "fa":
        system_prompt = (
            "تو یک دستیار هوشمند خلاصه‌ساز پیام‌های صوتی تلگرام هستی.\n"
            "وظیفه تو این است که متن پیاده‌شده‌ی ویس را بخوانی، خطاهای املایی یا شنیداری احتمالی را در ذهن خود اصلاح کنی "
            f"و یک خلاصه‌ی بسیار روان، صمیمی، دقیق و کاربردی به زبان فارسی ارائه دهی.\n\n"
            "قوانین قالب خروجی:\n"
            f"۱. خط اول دقیقاً با این عنوان شروع شود: «📌 خلاصه صحبت‌های {speaker}:»\n"
            f"۲. پاراگراف اول را به صورت طبیعی و محاوره‌ای با عبارتی مثل «{speaker} میگه که...» شروع کن و اصل حرفش را در ۱ تا ۲ جمله بگو.\n"
            "۳. اگر ویس شامل جزئیات، تصمیمات یا کارهای مهم است، آن‌ها را در قالب بولت‌پوینت‌های کوتاه و شفاف (•) فهرست کن.\n"
            "۴. از زیاده‌گویی پرهیز کن تا کاربر بدون نیاز به گوش دادن ویس، در چند ثانیه کل منظور را بفهمد."
        )
        user_prompt = (
            f"نام گوینده: {speaker}\n"
            f"متن پیاده‌شده از ویس:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"لطفاً خلاصه صحبت‌های {speaker} را طبق قالب خواسته‌شده بنویس."
        )
    else:
        system_prompt = (
            "You are an intelligent Telegram voice message summarizer.\n"
            "Read the transcribed voice message, mentally fix any minor speech-to-text transcription errors, "
            "and write a clear, natural, and concise summary in English.\n\n"
            "Output format rules:\n"
            f"1. Start the first line with: '📌 Summary of {speaker}'s voice:'\n"
            f"2. Begin the summary naturally with '{speaker} says that...' capturing the core message in 1-2 sentences.\n"
            "3. List key points, decisions, or action items as short bullet points (•).\n"
            "4. Keep it concise and easy to scan in seconds."
        )
        user_prompt = (
            f"Speaker name: {speaker}\n"
            f"Voice transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"Please provide the summary of what {speaker} said."
        )

    return system_prompt, user_prompt
