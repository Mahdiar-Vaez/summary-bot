"""Prompt templates for Persian and English voice and chat history summarization."""

from __future__ import annotations

from src.services.stt.base import normalize_language

CHAT_HISTORY_SPEAKER_PREFIX = "__CHAT_HISTORY__:"


def build_summary_prompt(
    transcript: str, speaker_name: str, language: str
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for either a voice message or recent chat messages."""
    speaker = (speaker_name or "گوینده").strip()
    norm_lang = normalize_language(language, transcript)

    if speaker.startswith(CHAT_HISTORY_SPEAKER_PREFIX):
        count_label = speaker.split(":", 1)[1].strip() or "چند"
        if norm_lang == "fa":
            system_prompt = (
                "تو یک دستیار هوشمند خلاصه‌ساز گفتگوهای گروه و کانال تلگرام هستی.\n"
                "وظیفه تو این است که پیام‌های اخیر گفتگو را بخوانی و یک جمع‌بندی **خیلی خیلی خلاصه، کوتاه، ساده و خفن** به زبان کاملاً خودمونی و محاوره‌ای (نه کتابی و رسمی) ارائه دهی.\n\n"
                "قوانین اکید و حیاتی قالب خروجی:\n"
                f"۱. خط اول دقیقاً و فقط همین عبارت باشد: «📌 خلاصه {count_label} پیام آخر گفتگو:»\n"
                "۲. متن باید خیلی خیلی خلاصه و کوتاه باشد (بدون هیچ مقدمه، حاشیه یا تعارف اضافه)، اما **تمام مطالب، موضوعات مطرح‌شده، تصمیم‌ها، زمان‌ها، ساعت‌ها، اعداد، نام افراد کلیدی و نکات مهم** باید به صورت کامل و تیتروار با بولت‌پوینت (•) در آن شرح داده شوند تا هیچ نکته مهمی جا نیفتد.\n"
                "۳. برای خوانایی عالی در تلگرام، تیتر هر نکته یا کلمات کلیدی مهم (مثل **نام افراد**، **ساعت‌ها** و **موضوع اصلی**) را حتماً بین دو ستاره `**...**` (بولد مارک‌داون) قرار بده.\n"
                "۴. کلمات بی‌معنی، نامربوط، پیام‌های هرز و سلام و احوال‌پرسی‌های معمول را غربال و حذف کن تا خلاصه کاملاً تمیز، مفید و متمرکز بر اصل موضوعات باشد.\n"
                "۵. لحن متن کاملاً خودمونی، روان و جذاب باشد."
            )
            user_prompt = (
                f"لیست {count_label} پیام آخر گفتگو:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
                f"لطفاً خلاصه خیلی کوتاه، خودمونی و جامع این پیام‌ها را (با حذف کلمات بی‌معنی و فرمت مارک‌داون `**بولد**` برای نکات و اسامی مهم) با تیتر «📌 خلاصه {count_label} پیام آخر گفتگو:» بنویس."
            )
        else:
            system_prompt = (
                "You are an intelligent Telegram group and channel chat summarizer.\n"
                "Read the recent messages and write an ultra-concise, casual, easy-to-scan summary in English that covers all topics and critical details.\n\n"
                "Strict output format rules:\n"
                f"1. Start the first line with: '📌 Summary of Last {count_label} Messages:'\n"
                "2. Keep the summary ultra-concise and super short (casual conversational tone, zero fluff), while covering all discussed topics, key decisions, names, dates, times, numbers, and action items as short bullet points (•).\n"
                "3. Use Markdown bold (`**...**`) around key names, times, and topic labels for clean Telegram formatting.\n"
                "4. Filter out meaningless noise, filler words, and small talk; focus purely on substance without altering critical facts.\n"
                "5. Ensure the final summary is clean and coherent."
            )
            user_prompt = (
                f"Recent {count_label} chat messages:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
                f"Please provide the super short, casual summary starting with '📌 Summary of Last {count_label} Messages:' covering all topics and important details."
            )
        return system_prompt, user_prompt

    if norm_lang == "fa":
        system_prompt = (
            "تو یک دستیار هوشمند خلاصه‌ساز پیام‌های صوتی تلگرام هستی.\n"
            "وظیفه تو این است که متن پیاده‌شده‌ی ویس را بخوانی، خطاهای شنیداری احتمالی را در ذهن خود اصلاح کنی "
            "و یک خروجی خیلی خیلی خلاصه و کوتاه به زبان کاملاً خودمونی و محاوره‌ای (نه کتابی و رسمی) ارائه دهی.\n\n"
            "قوانین اکید و حیاتی قالب خروجی:\n"
            f"۱. خط اول دقیقاً و فقط همین عبارت باشد: «📌 خلاصه صحبت‌های {speaker}:»\n"
            "۲. در متن خلاصه به هیچ وجه نام گوینده را نیاور و از عباراتی مثل «فلانی میگه که...» یا «گوینده میگه...» استفاده نکن.\n"
            "۳. لحن متن باید کاملاً خودمونی و محاوره‌ای (نه کتابی و رسمی) باشد؛ انگار داری خیلی سریع به دوستت می‌گی طرف چی گفت.\n"
            "۴. متن باید خیلی خیلی خلاصه و کوتاه باشد (در کوتاه‌ترین حالت ممکن، بدون هیچ مقدمه، تعارف یا توضیح اضافه).\n"
            "۵. با وجود کوتاه بودن، هیچ‌کدام از جزئیات و نکات مهم و حساس مثل نام افراد، زمان کاری، ساعت‌ها، روزها، تاریخ‌ها، مبالغ، اعداد و کارهای کلیدی نباید گم یا حذف شوند! همه را خیلی کوتاه و تیتروار با بولت‌پوینت (•) بنویس و کلمات کلیدی مهم (مثل **نام افراد**، **ساعت‌ها**، **مبالغ** یا **موضوع**) را بین دو ستاره `**...**` بولد کن.\n"
            "۶. کلمات بی‌معنی، نامربوط، تپق‌ها، تکیه‌کلام‌ها و نویزهای گفتاری وسط جمله (مانند «مثلاً مثلاً»، «چیز»، «اممم»، «ینی»، «عرضم به حضورت»، «چیزه») را بررسی و کاملاً حذف کن، به شرطی که به جزئیات، اسامی و مفاهیم اصلی پیام آسیبی نزند و متن خلاصه کاملاً تمیز، معنادار و بدون کلمات زائد باشد."
        )
        user_prompt = (
            f"متن پیاده‌شده از ویس:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"لطفاً خلاصه خیلی خیلی کوتاه و خودمونی این ویس را (با بررسی و حذف کلمات بی‌معنی و تپق‌های وسط جمله بدون آسیب به جزئیات مهم مثل نام افراد و زمان کاری، و بولد کردن نکات کلیدی با `**...**`) فقط با تیتر بالای «📌 خلاصه صحبت‌های {speaker}:» بنویس."
        )
    else:
        system_prompt = (
            "You are an intelligent Telegram voice message summarizer.\n"
            "Read the transcribed voice message, mentally fix any minor speech-to-text transcription errors, "
            "and write an ultra-concise, casual, conversational summary in English (not stiff or overly formal) that preserves all critical and sensitive details.\n\n"
            "Strict output format rules:\n"
            f"1. Start the first line with: '📌 Summary of {speaker}'s voice:'\n"
            "2. Do NOT mention the speaker's name in the summary body and do NOT write phrases like 'X says that...'.\n"
            "3. Keep the summary ultra-concise and super short (zero fluff or filler, casual conversational tone), while strictly retaining all critical and sensitive details "
            "(names of people, working hours, exact numbers, dates, times, amounts, and action items) as very short bullet points (•) with Markdown bold (`**...**`) on key terms.\n"
            "4. Every bullet point must be telegraphically brief and easy to scan in 2 seconds.\n"
            "5. Carefully filter out and remove meaningless filler words, verbal tics, stammering, and irrelevant speech disfluencies (e.g., 'um', 'uh', 'you know', 'like', stumbled repetitions) from the middle of sentences, as long as it does not affect any critical details, names, or facts."
        )
        user_prompt = (
            f"Voice transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"Please provide the super short, casual summary starting with '📌 Summary of {speaker}'s voice:' (filtering out meaningless filler words while strictly keeping all names, working hours, and critical details)."
        )

    return system_prompt, user_prompt
