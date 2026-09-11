"""Answer normalization and multiple-choice letter parsing."""

from __future__ import annotations

import re

NUMBER_WORDS = {w: str(i) for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}
LETTERS = "ABCDEFGH"


def normalize_short(text: str) -> str:
    """Lowercase, strip punctuation, keep the first meaningful token; map number words to digits."""
    t = text.strip().lower()
    t = re.sub(r"^(answer|the answer is|it is|it's|there are|there is)[:\s]+", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t).strip()
    if not t:
        return ""
    tok = t.split()[0]
    return NUMBER_WORDS.get(tok, tok)


def parse_letters(text: str, n_options: int) -> str:
    """Extract chosen option letters ('A, C' / 'AC' / 'Options A and C' / 'none') -> 'AC'."""
    valid = LETTERS[:n_options]
    t = text.strip()
    if re.search(r"\bnone\b", t, re.IGNORECASE) and not re.search(r"\b[A-H]\b", t):
        return ""
    found = re.findall(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", t.upper())
    letters = sorted({c for c in found if c in valid})
    if not letters:  # fallback: compact strings like "AC"
        m = re.fullmatch(r"[A-H]+", t.upper().replace(",", "").replace(" ", ""))
        if m:
            letters = sorted({c for c in m.group(0) if c in valid})
    return "".join(letters)
