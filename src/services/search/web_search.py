"""Lightweight async web search service (Bing RSS, Google News RSS, Wikipedia, TGJU)."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from src.services.summarizer.prompts import sanitize_sensitive_text

logger = logging.getLogger(__name__)

_SEARCH_DIRECTIVE_RE = re.compile(r"\[\[\s*SEARCH\s*:\s*([^\n\]]+?)\s*\]\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE)

_PRESEARCH_KEYWORDS = (
    "قیمت",
    "نرخ",
    "طلا",
    "سکه",
    "دلار",
    "یورو",
    "تتر",
    "ارز",
    "بیت کوین",
    "بیت‌کوین",
    "اتریوم",
    "کریپتو",
    "بورس",
    "شاخص",
    "انس",
    "آب و هوا",
    "آب‌وهوا",
    "هواشناسی",
    "هوا چطور",
    "اخبار",
    "خبر",
    "امروز",
    "الان چنده",
    "چند شده",
    "لحظه‌ای",
    "لحظه ای",
    "نتیجه بازی",
    "بازی کی",
    "لیگ",
    "سرچ کن",
    "جستجو کن",
    "جست‌وجو کن",
    "گوگل کن",
    "price",
    "gold",
    "dollar",
    "usd",
    "btc",
    "bitcoin",
    "crypto",
    "exchange rate",
    "weather",
    "forecast",
    "news",
    "latest",
    "today",
    "current",
    "score",
    "search ",
)

_IR_MARKET_TRIGGERS = (
    ("geram18", ("طلا", "۱۸ عیار", "18 عیار", "گرم طلا", "gold"), "طلای ۱۸ عیار"),
    ("sekee", ("سکه", "امامی", "coin"), "سکه امامی"),
    ("price_dollar_rl", ("دلار", "ارز", "dollar", "usd"), "دلار آزاد"),
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def should_presearch_query(question: str) -> bool:
    """Return True when the user's question clearly asks for live prices, weather, news, or web facts."""
    normalized = " ".join((question or "").strip().lower().split())
    if not normalized:
        return False
    return any(kw in normalized for kw in _PRESEARCH_KEYWORDS)


def extract_auto_search_query(text: str) -> str | None:
    """Extract query from a `[[SEARCH: ...]]` directive emitted by an LLM, if present."""
    match = _SEARCH_DIRECTIVE_RE.search(text or "")
    if not match:
        return None
    query = " ".join(match.group(1).split()).strip()
    return query[:160] if query else None


def strip_search_directives(text: str) -> str:
    """Remove any internal `[[SEARCH: ...]]` directives so they never reach Telegram users."""
    cleaned = _SEARCH_DIRECTIVE_RE.sub("", text or "")
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def _clean_html_snippet(raw: str, max_len: int = 220) -> str:
    text = html.unescape(raw or "")
    text = _HTML_TAG_RE.sub(" ", text)
    text = " ".join(text.split()).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def _format_toman_and_rial(raw_rial: str) -> str:
    digits = re.sub(r"[^\d]", "", (raw_rial or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")))
    if not digits:
        return raw_rial
    rial_val = int(digits)
    toman_val = rial_val // 10
    return f"{toman_val:,} تومان ({rial_val:,} ریال)"


class WebSearchService:
    """Fetches concise live web snippets from reachable HTTP RSS/JSON sources."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout: float = 8.0,
    ) -> None:
        self._client = client
        self._timeout = timeout

    async def _get(self, url: str, *, bypass_proxy: bool = False) -> httpx.Response:
        headers = {"User-Agent": _USER_AGENT}
        if self._client is not None:
            return await self._client.get(url, headers=headers)
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            trust_env=not bypass_proxy,
        ) as client:
            return await client.get(url, headers=headers)

    async def _fetch_rss_items(
        self, url: str, source_label: str, *, limit: int = 3
    ) -> list[str]:
        try:
            resp = await self._get(url)
            if resp.status_code != 200 or not resp.text:
                return []
            xml_text = _DOCTYPE_RE.sub("", resp.text)
            root = ET.fromstring(xml_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s RSS fetch/parse skipped: %s", source_label, exc)
            return []

        items: list[str] = []
        for item in root.findall(".//item"):
            title = _clean_html_snippet(item.findtext("title") or "", max_len=140)
            desc = _clean_html_snippet(item.findtext("description") or "", max_len=200)
            pub_date = " ".join((item.findtext("pubDate") or "").split())[:35]
            if not title:
                continue
            line = f"• [{source_label}] {title}"
            if desc and desc.lower() != title.lower():
                line += f": {desc}"
            if pub_date:
                line += f" ({pub_date})"
            items.append(line)
            if len(items) >= limit:
                break
        return items

    async def _fetch_wikipedia_items(
        self, query: str, lang: str, *, limit: int = 2
    ) -> list[str]:
        wiki_lang = "fa" if lang == "fa" else "en"
        encoded = urllib.parse.quote(query)
        url = (
            f"https://{wiki_lang}.wikipedia.org/w/api.php"
            f"?action=query&list=search&format=json&utf8=1&srlimit={limit}&srsearch={encoded}"
        )
        try:
            resp = await self._get(url)
            if resp.status_code != 200:
                return []
            payload = resp.json()
            hits = payload.get("query", {}).get("search", [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Wikipedia search skipped: %s", exc)
            return []

        results: list[str] = []
        for hit in hits[:limit]:
            if not isinstance(hit, dict):
                continue
            title = _clean_html_snippet(str(hit.get("title") or ""), max_len=100)
            snippet = _clean_html_snippet(str(hit.get("snippet") or ""), max_len=220)
            if title and snippet:
                results.append(f"• [ویکی‌پدیا] {title}: {snippet}")
        return results

    async def _fetch_tgju_market(self, query: str) -> list[str]:
        lowered = (query or "").lower()
        selected = [
            (slug, label)
            for slug, triggers, label in _IR_MARKET_TRIGGERS
            if any(t in lowered for t in triggers)
        ]
        if not selected:
            return []

        async def _fetch_slug(slug: str, label: str) -> str | None:
            url = f"https://api.tgju.org/v1/market/indicator/summary-table-data/{slug}"
            try:
                resp = await self._get(url, bypass_proxy=True)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                rows = data.get("data")
                if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
                    return None
                row = rows[0]
                raw_price = str(row[0]) if len(row) > 0 else ""
                if not raw_price:
                    return None
                g_date = _clean_html_snippet(str(row[6]), max_len=20) if len(row) > 6 else ""
                j_date = _clean_html_snippet(str(row[7]), max_len=20) if len(row) > 7 else ""
                date_part = " | ".join(p for p in (j_date, g_date) if p)
                formatted_price = _format_toman_and_rial(raw_price)
                suffix = f" [تاریخ ثبت TGJU: {date_part}]" if date_part else ""
                return f"• [نرخ TGJU] {label}: {formatted_price}{suffix}"
            except Exception as exc:  # noqa: BLE001
                logger.debug("TGJU fetch skipped for %s: %s", slug, exc)
                return None

        rows = await asyncio.gather(*(_fetch_slug(s, l) for s, l in selected[:3]))
        return [r for r in rows if r]

    async def search(self, query: str, language: str = "fa") -> str:
        """Run concurrent lightweight web searches and return sanitized summary bullets."""
        clean_query = sanitize_sensitive_text(" ".join((query or "").split()))[:160].strip()
        if not clean_query:
            return ""

        encoded = urllib.parse.quote(clean_query)
        bing_url = f"https://www.bing.com/search?format=rss&q={encoded}"
        if language == "fa":
            gnews_url = f"https://news.google.com/rss/search?hl=fa&gl=IR&ceid=IR:fa&q={encoded}"
        else:
            gnews_url = f"https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q={encoded}"

        tgju_task = self._fetch_tgju_market(clean_query)
        bing_task = self._fetch_rss_items(bing_url, "Bing", limit=4)
        gnews_task = self._fetch_rss_items(gnews_url, "Google News", limit=3)

        tgju_items, bing_items, gnews_items = await asyncio.gather(
            tgju_task, bing_task, gnews_task
        )

        combined = [*tgju_items, *bing_items, *gnews_items]
        if len(combined) < 2:
            wiki_items = await self._fetch_wikipedia_items(clean_query, language, limit=2)
            combined.extend(wiki_items)

        if not combined:
            return ""

        joined = sanitize_sensitive_text("\n".join(combined))
        if len(joined) > 2200:
            joined = joined[:2199].rstrip() + "…"
        return joined
