"""
Where finished spans go.

LocalExporter    JSONL files in ~/.langtrain/traces (Free). Read back by the
                 `langtrain-observe` CLI and observe.load_spans().
CloudExporter    Batched to the Langtrain API, searchable in the dashboard
                 (Pro; the server rejects other plans).
OTLPExporter     Any OpenTelemetry collector over OTLP/HTTP JSON, with the
                 GenAI semantic-convention attributes (Free).
ConsoleExporter  A one-line summary per finished trace.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .tracing import Span

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".langtrain", "traces")


class LocalExporter:
    def __init__(self, directory: Optional[str] = None):
        self.directory = directory or os.environ.get("LANGTRAIN_TRACES_DIR") or DEFAULT_DIR
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        day = datetime.fromtimestamp(span.start_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
        line = json.dumps(span.to_dict(), default=str)
        with self._lock:
            os.makedirs(self.directory, exist_ok=True)
            with open(os.path.join(self.directory, f"traces-{day}.jsonl"), "a", encoding="utf-8") as f:
                f.write(line + "\n")


class ConsoleExporter:
    def export(self, span: Span) -> None:
        if span.parent_id is not None:
            return
        a = span.attributes
        cost = f"  cost={a['cost']:.4f}" if isinstance(a.get("cost"), (int, float)) else ""
        print(f"[langtrain] {span.status:5} {span.name}  {span.latency_ms:.0f}ms  trace={span.trace_id[:12]}{cost}")


class _Batcher:
    """Collects spans on a background thread and sends them in batches."""

    def __init__(self, batch_size: int = 100, interval: float = 5.0):
        self.batch_size = batch_size
        self.interval = interval
        self._q: "queue.Queue[Span]" = queue.Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="langtrain-observe", daemon=True)
        self._thread.start()
        atexit.register(self.flush)

    def export(self, span: Span) -> None:
        try:
            self._q.put_nowait(span)
        except queue.Full:
            pass  # drop rather than block the app

    def _drain(self) -> List[Span]:
        batch: List[Span] = []
        while len(batch) < self.batch_size:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._send_all()

    def _send_all(self) -> None:
        while True:
            batch = self._drain()
            if not batch:
                return
            try:
                self.send(batch)
            except Exception:
                return

    def flush(self) -> None:
        self._send_all()

    def send(self, batch: List[Span]) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class CloudExporter(_Batcher):
    """Send spans to Langtrain. Needs a Pro plan; checked once at start-up."""

    def __init__(self, client: Any = None, batch_size: int = 100, interval: float = 5.0):
        from langtrain.client import LangtrainClient

        from . import entitlements

        self.client = client or LangtrainClient()
        entitlements.require("cloud_traces", self.client)
        self.disabled_reason: Optional[str] = None
        super().__init__(batch_size, interval)

    def send(self, batch: List[Span]) -> None:
        if self.disabled_reason:
            return
        from langtrain.client import LangtrainError

        try:
            self.client._post("/api/v1/observe/traces", {"spans": [s.to_dict() for s in batch]})
        except LangtrainError as e:
            if e.status_code in (401, 402, 403):
                self.disabled_reason = str(e)  # plan or key changed: stop, don't retry forever
            else:
                raise


class OTLPExporter(_Batcher):
    """OpenTelemetry over OTLP/HTTP JSON, e.g. endpoint="http://localhost:4318/v1/traces"."""

    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None,
                 service_name: str = "langtrain-app", batch_size: int = 100, interval: float = 5.0):
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.service_name = service_name
        super().__init__(batch_size, interval)

    def send(self, batch: List[Span]) -> None:
        import requests

        body = to_otlp(batch, self.service_name)
        resp = requests.post(self.endpoint, data=json.dumps(body, default=str), headers=self.headers, timeout=10)
        resp.raise_for_status()


def _otel_value(v: Any) -> Dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": v if isinstance(v, str) else json.dumps(v, default=str)}


_GENAI_KEYS = {
    "model": "gen_ai.request.model",
    "provider": "gen_ai.system",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
}


def to_otlp(spans: List[Span], service_name: str = "langtrain-app") -> Dict[str, Any]:
    """Spans as an OTLP ExportTraceServiceRequest (JSON encoding)."""
    out = []
    for s in spans:
        attrs = [{"key": "langtrain.kind", "value": _otel_value(s.kind)}]
        for k, v in s.attributes.items():
            attrs.append({"key": _GENAI_KEYS.get(k, f"langtrain.{k}"), "value": _otel_value(v)})
        item: Dict[str, Any] = {
            "traceId": s.trace_id.ljust(32, "0")[:32],
            "spanId": s.span_id.ljust(16, "0")[:16],
            "name": s.name,
            "kind": 3 if s.kind in ("llm", "embedding", "tool") else 1,  # CLIENT, INTERNAL
            "startTimeUnixNano": str(s.start_ns),
            "endTimeUnixNano": str(s.end_ns or s.start_ns),
            "attributes": attrs,
            "status": {"code": 2, "message": (s.error or {}).get("message", "")} if s.status == "error" else {"code": 1},
        }
        if s.parent_id:
            item["parentSpanId"] = s.parent_id.ljust(16, "0")[:16]
        out.append(item)
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service_name}}]},
        "scopeSpans": [{"scope": {"name": "langtrain.observe"}, "spans": out}],
    }]}


def load_spans(directory: Optional[str] = None, since_hours: Optional[float] = None, limit: Optional[int] = None) -> List[Span]:
    """Spans written by LocalExporter, oldest first."""
    directory = directory or os.environ.get("LANGTRAIN_TRACES_DIR") or DEFAULT_DIR
    if not os.path.isdir(directory):
        return []
    cutoff = time.time_ns() - int(since_hours * 3600e9) if since_hours else None
    spans: List[Span] = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("traces-") and name.endswith(".jsonl")):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            for line in f:
                try:
                    s = Span.from_dict(json.loads(line))
                except (ValueError, TypeError):
                    continue
                if cutoff is None or (s.start_ns or 0) >= cutoff:
                    spans.append(s)
    spans.sort(key=lambda s: s.start_ns or 0)
    return spans[-limit:] if limit else spans
