"""
PII detection and redaction, applied to span inputs and outputs before they
leave the process. Patterns cover what Indian and global apps commonly
handle: emails, phone numbers, payment cards (Luhn-checked), PAN, Aadhaar,
IP addresses and API keys/tokens.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bAKIA[0-9A-Z]{16}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("phone", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3,5}\)?[ -]?)\d{3,4}[ -]?\d{3,4}(?!\w)")),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def find(text: str) -> List[Dict[str, Any]]:
    """Every PII match in `text`: [{"type", "start", "end"}], earliest first, without overlaps."""
    hits: List[Dict[str, Any]] = []
    taken: List[Tuple[int, int]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            if kind == "card":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 19 and _luhn(digits)):
                    continue
            hits.append({"type": kind, "start": s, "end": e})
            taken.append((s, e))
    return sorted(hits, key=lambda h: h["start"])


def redact(text: str) -> Tuple[str, List[str]]:
    """`text` with each PII match replaced by [TYPE], and the types found."""
    hits = find(text)
    out, last = [], 0
    for h in hits:
        out.append(text[last:h["start"]])
        out.append(f"[{h['type'].upper()}]")
        last = h["end"]
    out.append(text[last:])
    return "".join(out), [h["type"] for h in hits]


def redact_value(value: Any) -> Tuple[Any, List[str]]:
    """Redact strings anywhere inside dicts/lists (chat messages, tool args)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        found: List[str] = []
        out = {}
        for k, v in value.items():
            out[k], f = redact_value(v)
            found += f
        return out, found
    if isinstance(value, (list, tuple)):
        found = []
        items = []
        for v in value:
            r, f = redact_value(v)
            items.append(r)
            found += f
        return items, found
    return value, []
