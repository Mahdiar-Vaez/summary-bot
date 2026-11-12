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


MAX_CHAT_QA_HISTORY_CHARS = 48_000


def _trim_history_to_recent_chars(chat_history: str, max_chars: int = MAX_CHAT_QA_HISTORY_CHARS) -> str:
    """Keep the most recent lines up to `max_chars` so 200+ typical messages fit safely."""
    clean = (chat_history or "").strip()
    if len(clean) <= max_chars:
        return clean
    tail = clean[-max_chars:]
    newline_idx = tail.find("\n")
    if 0 <= newline_idx < len(tail) - 1:
        tail = tail[newline_idx + 1 :]
    return tail


def build_chat_qa_prompt(
    chat_history: str,
    session_turns: list[tuple[str, str]],
    question: str,
    asker_name: str,
    mode: str = "chat",
    language: str = "fa",
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for interactive per-chat Q&A and fun personality analysis."""
    clean_question = (question or "").strip()
    asker = (asker_name or "کاربر").strip()
    trimmed_history = _trim_history_to_recent_chars(chat_history)
    norm_lang = normalize_language(language, f"{clean_question}\n{trimmed_history}")

    history_block = trimmed_history or (
        "(هنوز پیام متنی یا ویسی در این گفتگو ثبت نشده است.)"
        if norm_lang == "fa"
        else "(No chat messages or transcripts recorded in this chat yet.)"
    )
    turns_lines: list[str] = []
    for turn in session_turns[-10:]:
        if len(turn) == 3:
            u_name, q, a = turn
            turns_lines.append(
                f"{u_name}: {q}\nربات: {a}" if norm_lang == "fa" else f"{u_name}: {q}\nBot: {a}"
            )
        elif len(turn) == 2:
            q, a = turn
            turns_lines.append(
                f"کاربر: {q}\nربات: {a}" if norm_lang == "fa" else f"User: {q}\nBot: {a}"
            )
    session_block = "\n---\n".join(turns_lines) if turns_lines else (
        "(شروع گفتگوی جدید در همین چت)" if norm_lang == "fa" else "(Start of a fresh session)"
    )

    if norm_lang == "fa":
        personality_hint = (
            "کاربر الان درخواست تحلیل شخصیت و وایب داده است! "
            "اگر در سوال اسم شخص خاصی (مثل مهدی یا سحر) آمده، تمرکز اصلی را روی تحلیل همان شخص بگذار؛ "
            "و اگر اسم خاصی نیامده، برای اعضای فعال همین گفتگو بر اساس لحن، شوخی‌ها، دغدغه‌ها و سبک پیام‌هایشان، "
            "تحلیل شخصیت و تیپ رفتاری بامزه، دقیق، فان و محترمانه (با ایموجی و بولت‌پوینت) بنویس."
            if mode == "personality"
            else (
                "کاربر ممکن است درباره صحبت‌های یک شخص خاص (مثل «سحر چی گفته؟» یا «مهدی راجع به فلان موضوع چی گفت؟»)، "
                "تیپ شخصیت افراد بر اساس پیام‌هایشان، یا جمع‌بندی بحث‌های همین چت سوال بپرسد:\n"
                "   - اگر درباره حرف‌های شخص خاصی پرسید، تمام پیام‌ها و ویس‌های او را در تاریخچه همین چت بررسی کن و خلاصه دقیق، روشن و خودمونی صحبت‌هایش را بگو.\n"
                "   - اگر درباره شخصیت کسی پرسید، با استناد به سبک پیام‌های واقعی‌اش در همین چت، تحلیل فان و محترمانه بده.\n"
                "   - در تشخیص نام افراد منعطف باش (مثلاً نام کوچک، نام کامل، آیدی `@username` یا املای فارسی/انگلیسی را تطبیق بده)."
            )
        )
        system_prompt = (
            "تو هم‌صحبت هوشمند، باحال، شوخ‌طبع و دقیقِ یک گروه یا چت تلگرام هستی.\n"
            "قوانین حیاتی:\n"
            "۱. **ایزوله‌سازی کامل سشن:** تو فقط و فقط به پیام‌ها و سشن گفتگوی **همین چت** دسترسی داری. هیچ‌وقت اطلاعاتی از بیرون یا چت‌های دیگر حدس نزن و اگر کسی یا موضوعی در تاریخچه ثبت‌شده همین چت نبود، روراست و بامزه بگو در پیام‌های اخیر ثبت‌شده اثری از آن نبود.\n"
            f"۲. {personality_hint}\n"
            "۳. لحن تو باید کاملاً خودمونی، بانمک، روان و جمع‌وجور باشد (نه کتابی و نه طولانی و خسته‌کننده).\n"
            "۴. کلمات کلیدی، اسم افراد و نکات مهم را برای نمایش زیبا در تلگرام بین `**...**` بولد کن."
        )
        user_prompt = (
            f"تاریخچه پیام‌های ثبت‌شده در همین چت:\n\"\"\"\n{history_block}\n\"\"\"\n\n"
            f"ادامه گفتگوی قبلی در همین سشن:\n\"\"\"\n{session_block}\n\"\"\"\n\n"
            f"پیام/سوال جدید از «{asker}»:\n{clean_question}"
        )
    else:
        personality_hint = (
            "The user explicitly asked for a fun personality & vibe analysis in this specific chat! "
            "If a specific person is named in the question, focus primarily on analyzing that person; "
            "otherwise analyze each active speaker's tone, habits, and topics from this chat's history with witty, playful, respectful humor."
            if mode == "personality"
            else (
                "The user may ask what a specific person said (e.g., 'What did Sahar say?'), someone's personality type based on the chats, or general Q&A. "
                "Match speakers flexibly by first name, full name, or `@username`, cite their actual messages from this chat accurately, and chat naturally with sharp, friendly wit."
            )
        )
        system_prompt = (
            "You are a witty, playful, sharp, and friendly AI companion inside a Telegram chat.\n"
            "Critical rules:\n"
            "1. **Strict Session Isolation:** ONLY use context from THIS chat's history and session turns. Never fabricate facts that didn't happen here.\n"
            f"2. {personality_hint}\n"
            "3. Keep answers punchy, conversational, fun, and concise.\n"
            "4. Use Markdown bold (`**...**`) around names and key highlights."
        )
        user_prompt = (
            f"Recorded messages in this chat:\n\"\"\"\n{history_block}\n\"\"\"\n\n"
            f"Recent Q&A session turns:\n\"\"\"\n{session_block}\n\"\"\"\n\n"
            f"New message from {asker}:\n{clean_question}"
        )

    return system_prompt, user_prompt
