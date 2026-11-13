"""Markdown-to-Telegram-HTML converter and safe chunk splitter."""

from __future__ import annotations

import html
import re

_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*")
_BOLD_UNDER_RE = re.compile(r"(?<![A-Za-z0-9_@])__(.+?)__(?![A-Za-z0-9_])")
_HEADING_RE = re.compile(r"(?m)^#{1,4}\s+(.+)$")
_PYTHON_DUNDERS = frozenset({
    "init", "main", "name", "doc", "file", "package", "spec", "loader",
    "cached", "builtins", "all", "dict", "class", "slots", "weakref",
    "module", "annotations", "qualname", "str", "repr", "len", "bool",
    "hash", "eq", "ne", "lt", "le", "gt", "ge", "iter", "next", "call",
    "getitem", "setitem", "delitem", "contains", "enter", "exit",
    "aenter", "aexit", "aiter", "anext", "await", "new", "del",
    "getattr", "setattr", "delattr", "getattribute", "dir", "format",
    "bytes", "int", "float", "complex", "index", "add", "sub", "mul",
    "truediv", "floordiv", "mod", "pow", "and", "or", "xor", "radd",
    "pycache", "version", "author", "future", "debug", "wrapped",
})


def _strip_existing_bold(text: str) -> str:
    return text.replace("<b>", "").replace("</b>", "")


def format_telegram_html(text: str) -> str:
    """Convert Markdown formatting (**bold**, __bold__, `code`, # headers) into safe Telegram HTML."""
    if not text:
        return ""

    sanitized = text.replace("\x00", "")
    escaped = html.escape(sanitized, quote=False)

    # Extract inline `code` first so bold/heading regexes can never cross code boundaries.
    code_tokens: list[str] = []

    def _stash_code(match: re.Match[str]) -> str:
        idx = len(code_tokens)
        code_tokens.append(f"<code>{match.group(1)}</code>")
        return f"\x00C{idx}\x00"

    escaped = _INLINE_CODE_RE.sub(_stash_code, escaped)

    # Convert headings first (before inline bolds), then **bold**, then __bold__.
    bold_tokens: list[str] = []

    def _stash_bold(inner: str) -> str:
        idx = len(bold_tokens)
        bold_tokens.append(f"<b>{inner}</b>")
        return f"\x00B{idx}\x00"

    escaped = _HEADING_RE.sub(
        lambda m: _stash_bold(_strip_existing_bold(m.group(1))), escaped
    )
    escaped = _BOLD_STAR_RE.sub(lambda m: _stash_bold(m.group(1)), escaped)
    escaped = _BOLD_UNDER_RE.sub(
        lambda m: (
            m.group(0)
            if m.group(1) in _PYTHON_DUNDERS
            else _stash_bold(m.group(1))
        ),
        escaped,
    )

    # Remove any leftover stray double-asterisks
    escaped = escaped.replace("**", "")

    # Ensure top 📌 header line is bolded if not already bold-wrapped
    lines = escaped.splitlines()
    if lines and "📌" in lines[0] and "\x00B" not in lines[0]:
        lines[0] = f"<b>{lines[0].strip()}</b>"
    escaped = "\n".join(lines)

    for idx, b_html in enumerate(bold_tokens):
        escaped = escaped.replace(f"\x00B{idx}\x00", b_html)
    for idx, c_html in enumerate(code_tokens):
        escaped = escaped.replace(f"\x00C{idx}\x00", c_html)

    return escaped


def _find_split_pos(text: str, limit: int) -> int:
    if len(text) <= limit:
        return len(text)
    nl = text.rfind("\n", 0, limit + 1)
    if nl >= limit // 2:
        return nl + 1
    sp = text.rfind(" ", 0, limit + 1)
    if sp >= limit // 2:
        return sp + 1
    return limit


def split_telegram_html(
    raw_text: str, max_len: int = 4000, *, escape_only: bool = False
) -> list[str]:
    """Split raw text and render each chunk safely within Telegram's 4096-char HTML limit."""
    renderer = (lambda s: html.escape(s, quote=False)) if escape_only else format_telegram_html
    if not raw_text:
        return [""]

    chunks: list[str] = []
    remaining = raw_text
    raw_window = max(200, min(max_len - 400, 3500))

    while remaining:
        if len(renderer(remaining)) <= max_len:
            chunks.append(renderer(remaining))
            break

        split_at = _find_split_pos(remaining, raw_window)
        candidate_raw = remaining[:split_at]
        rendered = renderer(candidate_raw)

        # Shrink proportionally if dense `<>&` escaping or markup pushes chunk past max_len
        while len(rendered) > max_len and len(candidate_raw) > 1:
            new_len = max(1, (len(candidate_raw) * (max_len - 32)) // len(rendered))
            if new_len >= len(candidate_raw):
                new_len = len(candidate_raw) - 1
            split_at = _find_split_pos(remaining, new_len)
            candidate_raw = remaining[:split_at]
            rendered = renderer(candidate_raw)

        chunks.append(rendered)
        remaining = remaining[split_at:].lstrip("\n")

    return chunks or [""]
