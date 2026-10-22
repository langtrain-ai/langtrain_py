"""
Prompt-injection detection: a fast heuristic score for inputs that try to
override instructions, leak the system prompt or smuggle new roles in.

It flags likely attacks for review; it is not a guarantee. Pair it with
guardrails on what the model is allowed to do.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_SIGNALS = [
    ("override", 0.5, re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,20}\b(instructions?|rules?|prompts?|messages?)\b", re.I | re.S)),
    ("system_prompt_leak", 0.4, re.compile(r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}\b(system prompt|hidden (instructions|prompt)|your (instructions|rules|prompt))\b", re.I | re.S)),
    ("role_hijack", 0.35, re.compile(r"\b(you are now|act as|pretend to be|from now on you)\b.{0,40}\b(dan|developer mode|jailbroken|unrestricted|no (rules|limits|filter))", re.I | re.S)),
    ("fake_role_tag", 0.3, re.compile(r"(^|\n)\s*(system|assistant)\s*:|<\|?(system|im_start)\|?>|\[/?INST\]", re.I)),
    ("exfiltration", 0.3, re.compile(r"\b(send|post|upload|exfiltrate)\b.{0,40}\b(https?://|webhook|api key|password|credentials)", re.I | re.S)),
    ("encoded_payload", 0.2, re.compile(r"\b(base64|rot13|hex)[- ]?(decode|encoded)\b", re.I)),
]


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(v) for v in value)
    return ""


def scan(value: Any) -> Dict[str, Any]:
    """{"score": 0..1, "flagged": bool, "signals": [...]} for a prompt, message list or dict."""
    text = _text(value)
    signals: List[str] = []
    score = 0.0
    for name, weight, pattern in _SIGNALS:
        if pattern.search(text):
            signals.append(name)
            score += weight
    score = min(1.0, score)
    return {"score": round(score, 2), "flagged": score >= 0.5, "signals": signals}
