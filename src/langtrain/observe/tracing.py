"""
Spans and traces.

A trace is one request or task; each step in it (a model call, a tool call,
a retrieval, an agent step) is a span with a parent. Spans nest through
contextvars, so `span()` and `@observe` work across functions, threads that
copy context, and asyncio tasks.

    from langtrain import observe

    observe.init()                                  # local JSONL traces

    @observe.observe(kind="agent")
    def answer(question):
        with observe.span("search", kind="retrieval") as s:
            docs = search(question)
            s.set(documents=docs)
        with observe.span("generate", kind="llm", model="my-model") as s:
            reply = llm(question, docs)
            s.set(input=question, output=reply)
            s.record_usage(input_tokens=812, output_tokens=96)
        return reply
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import pricing, redact, security

KINDS = ("llm", "agent", "tool", "retrieval", "embedding", "chain", "guardrail", "task")

_current: "contextvars.ContextVar[Optional[Span]]" = contextvars.ContextVar("langtrain_span", default=None)


def _now_ns() -> int:
    return time.time_ns()


@dataclass
class Span:
    name: str
    kind: str = "task"
    trace_id: str = ""
    span_id: str = ""
    parent_id: Optional[str] = None
    start_ns: int = 0
    end_ns: Optional[int] = None
    status: str = "ok"
    error: Optional[Dict[str, str]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    # ── recording ────────────────────────────────────────────────────────────

    def set(self, **attributes: Any) -> "Span":
        """Attach attributes: input, output, model, documents, user_id, session_id, tags, ..."""
        self.attributes.update(attributes)
        return self

    def record_usage(self, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                     model: Optional[str] = None, cost: Optional[float] = None, cached: Optional[bool] = None) -> "Span":
        if model:
            self.attributes["model"] = model
        if input_tokens is not None:
            self.attributes["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            self.attributes["output_tokens"] = int(output_tokens)
        if cost is not None:
            self.attributes["cost"] = float(cost)
        if cached is not None:
            self.attributes["cache_hit"] = bool(cached)
        return self

    def add_event(self, name: str, **attributes: Any) -> "Span":
        self.events.append({"name": name, "at_ns": _now_ns(), "attributes": attributes})
        return self

    def record_error(self, exc: BaseException) -> "Span":
        self.status = "error"
        self.error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status_code:
            self.attributes["http_status"] = int(status_code)
        return self

    # ── reading ──────────────────────────────────────────────────────────────

    @property
    def latency_ms(self) -> Optional[float]:
        return None if self.end_ns is None else round((self.end_ns - self.start_ns) / 1e6, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id,
            "name": self.name, "kind": self.kind, "start_ns": self.start_ns, "end_ns": self.end_ns,
            "latency_ms": self.latency_ms, "status": self.status, "error": self.error,
            "attributes": self.attributes, "events": self.events,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        return cls(**{k: d.get(k) for k in ("name", "kind", "trace_id", "span_id", "parent_id", "start_ns", "end_ns", "status", "error")},
                   attributes=d.get("attributes") or {}, events=d.get("events") or [])


class Tracer:
    """Creates spans and hands finished ones to exporters."""

    def __init__(self, exporters: Optional[List[Any]] = None, sample_rate: float = 1.0,
                 redact_pii: bool = True, detect_injection: bool = True, project: Optional[str] = None):
        self.exporters = list(exporters or [])
        self.sample_rate = sample_rate
        self.redact_pii = redact_pii
        self.detect_injection = detect_injection
        self.project = project
        self._sampled: Dict[str, bool] = {}

    @contextmanager
    def span(self, name: str, kind: str = "task", **attributes: Any) -> Iterator[Span]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        parent = _current.get()
        s = Span(
            name=name, kind=kind,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start_ns=_now_ns(), attributes=dict(attributes),
        )
        if self.project and not parent:
            s.attributes.setdefault("project", self.project)
        token = _current.set(s)
        try:
            yield s
        except BaseException as exc:  # record, then re-raise unchanged
            s.record_error(exc)
            raise
        finally:
            _current.reset(token)
            s.end_ns = _now_ns()
            self._finish(s)

    def _finish(self, s: Span) -> None:
        if s.trace_id not in self._sampled:
            self._sampled[s.trace_id] = random.random() < self.sample_rate
        keep = self._sampled[s.trace_id]
        if s.parent_id is None:
            self._sampled.pop(s.trace_id, None)
        if not keep:
            return
        a = s.attributes
        if a.get("cost") is None:
            c = pricing.cost(a.get("model"), a.get("input_tokens"), a.get("output_tokens"))
            if c is not None:
                a["cost"] = c
        if self.detect_injection and s.kind in ("llm", "agent") and a.get("input") is not None:
            verdict = security.scan(a["input"])
            if verdict["signals"]:
                a["security"] = verdict
        if self.redact_pii:
            found: List[str] = []
            for key in ("input", "output", "documents", "messages", "arguments", "result"):
                if key in a:
                    a[key], f = redact.redact_value(a[key])
                    found += f
            if s.error:
                s.error["message"], f = redact.redact(s.error["message"])
                found += f
            if found:
                a["pii_redacted"] = sorted(set(found))
        for exporter in self.exporters:
            try:
                exporter.export(s)
            except Exception:  # an exporter must never break the app
                pass

    def observe(self, name: Optional[str] = None, kind: str = "task", capture_io: bool = True) -> Callable:
        """Decorator: trace each call of a function (sync or async) as a span."""
        def wrap(fn: Callable) -> Callable:
            span_name = name or fn.__qualname__

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def awrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(span_name, kind=kind) as s:
                        if capture_io:
                            s.set(input=_io(args, kwargs))
                        result = await fn(*args, **kwargs)
                        if capture_io:
                            s.set(output=_short(result))
                        return result
                return awrapper

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, kind=kind) as s:
                    if capture_io:
                        s.set(input=_io(args, kwargs))
                    result = fn(*args, **kwargs)
                    if capture_io:
                        s.set(output=_short(result))
                    return result
            return wrapper
        return wrap

    def flush(self) -> None:
        for exporter in self.exporters:
            flush = getattr(exporter, "flush", None)
            if flush:
                flush()


def current_span() -> Optional[Span]:
    return _current.get()


def _short(value: Any, limit: int = 4000) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:limit] if isinstance(value, str) else value
    if isinstance(value, (list, tuple)):
        return [_short(v, limit) for v in list(value)[:50]]
    if isinstance(value, dict):
        return {str(k): _short(v, limit) for k, v in list(value.items())[:50]}
    return repr(value)[:limit]


def _io(args: tuple, kwargs: dict) -> Any:
    if len(args) == 1 and not kwargs:
        return _short(args[0])
    return _short({"args": list(args), "kwargs": kwargs})
