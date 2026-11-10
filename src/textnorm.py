from __future__ import annotations

import re
import unicodedata

_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})


def normalize_text(text: str) -> str:
    """Normalize PDF text for quote matching."""
    if not text:
        return ""

    out = unicodedata.normalize("NFKC", str(text))
    for src, dst in _LIGATURES.items():
        out = out.replace(src, dst)
    out = out.translate(_DASHES)

    # Join words broken across lines: "simu-\nlation" -> "simulation".
    out = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", out)
    # Other hyphenated line breaks keep a single hyphen.
    out = re.sub(r"-\s*\n\s*", "-", out)

    out = re.sub(r"\s+", " ", out).strip()
    return out.casefold()


def contains_quote(haystack: str, quote: str) -> bool:
    q = normalize_text(quote)
    if not q:
        return False
    return q in normalize_text(haystack)
