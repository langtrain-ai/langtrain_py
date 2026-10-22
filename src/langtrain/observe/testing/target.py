"""
The model a test suite talks to.

Target          any OpenAI-compatible chat endpoint: Ollama (the default,
                http://localhost:11434/v1), vLLM, Langtrain-served models,
                OpenAI, OpenRouter, ...
FunctionTarget  a Python function, for custom stacks and tests.

Both return a Reply with the text, any tool calls, time to first token,
latency and token counts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible API


@dataclass
class Reply:
    text: str
    ttft_ms: float
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_estimated: bool = False

    @property
    def tokens_per_sec(self) -> float:
        gen_ms = self.latency_ms - self.ttft_ms
        return round((self.output_tokens or 0) / (gen_ms / 1000), 2) if gen_ms > 0 and self.output_tokens else 0.0


class Target:
    """An OpenAI-compatible chat model."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 300):
        self.model = model
        self.base_url = (base_url or os.environ.get("LANGTRAIN_TARGET_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("LANGTRAIN_TARGET_KEY") or "not-needed"
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"Target({self.model!r} @ {self.base_url})"

    @property
    def name(self) -> str:
        return self.model

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None) -> Reply:
        import requests

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: Dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()

        if tools:  # tool calls are simplest to read from a non-streamed reply
            body["tools"] = tools
            resp = requests.post(url, headers=self._headers(), data=json.dumps(body), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.perf_counter() - started) * 1000
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            calls = []
            for c in msg.get("tool_calls") or []:
                fn = c.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"_raw": args}
                calls.append({"name": fn.get("name"), "arguments": args or {}})
            usage = data.get("usage") or {}
            return Reply(text=msg.get("content") or "", ttft_ms=latency, latency_ms=latency, tool_calls=calls,
                         input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))

        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        ttft: Optional[float] = None
        parts: List[str] = []
        usage: Dict[str, Any] = {}
        with requests.post(url, headers=self._headers(), data=json.dumps(body), timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000
                        parts.append(piece)
        latency = (time.perf_counter() - started) * 1000
        text = "".join(parts)
        out = usage.get("completion_tokens")
        return Reply(text=text, ttft_ms=round(ttft if ttft is not None else latency, 2), latency_ms=round(latency, 2),
                     input_tokens=usage.get("prompt_tokens"),
                     output_tokens=out if out is not None else max(1, len(text) // 4) if text else 0,
                     tokens_estimated=out is None)


class FunctionTarget:
    """Wrap fn(prompt, system=..., temperature=..., max_tokens=..., tools=...) -> str | dict."""

    def __init__(self, fn: Callable[..., Any], name: str = "function"):
        self.fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def chat(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None) -> Reply:
        started = time.perf_counter()
        out = self.fn(prompt, system=system, temperature=temperature, max_tokens=max_tokens, tools=tools)
        latency = round((time.perf_counter() - started) * 1000, 3)
        if isinstance(out, dict):
            text = out.get("text", "")
            return Reply(text=text, ttft_ms=latency, latency_ms=latency, tool_calls=out.get("tool_calls") or [],
                         output_tokens=out.get("output_tokens", max(1, len(text) // 4) if text else 0),
                         input_tokens=out.get("input_tokens"), tokens_estimated="output_tokens" not in out)
        text = str(out)
        return Reply(text=text, ttft_ms=latency, latency_ms=latency,
                     output_tokens=max(1, len(text) // 4) if text else 0, tokens_estimated=True)
