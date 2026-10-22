"""
langtrain.observe: tracing, evaluation and monitoring for LLM apps and agents.

    from langtrain import observe

    observe.init(project="support-bot")          # local traces (Free)
    observe.init(cloud=True)                     # + Langtrain dashboard (Pro)

    @observe.observe(kind="agent")
    def handle(question): ...

    client = observe.wrap_openai(OpenAI())       # every call traced

    observe.summary(since_hours=24)              # tokens, cost, latency, errors
    observe.recommendations()                    # Pro

Capture, cost, latency, errors, tools, evals, prompt versions, PII redaction,
injection detection and OpenTelemetry export are Free. Analysis features are
Pro; see observe.capabilities.CAPABILITIES or `langtrain-observe features`.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from . import analysis, capabilities, entitlements, evals, prompts, redact, security
from .entitlements import ProFeatureError
from .exporters import CloudExporter, ConsoleExporter, LocalExporter, OTLPExporter, load_spans
from .integrations import wrap_openai
from .pricing import set_price
from .tracing import Span, Tracer, current_span

_tracer: Optional[Tracer] = None
_cloud: Optional[CloudExporter] = None


def _default_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer([LocalExporter()])
    return _tracer


def init(project: Optional[str] = None, cloud: bool = False, local: bool = True, console: bool = False,
         otlp_endpoint: Optional[str] = None, otlp_headers: Optional[Dict[str, str]] = None,
         sample_rate: float = 1.0, redact_pii: bool = True, detect_injection: bool = True,
         traces_dir: Optional[str] = None, client: Any = None) -> Tracer:
    """Configure tracing. cloud=True sends traces to Langtrain and needs a Pro plan."""
    global _tracer, _cloud
    exporters: List[Any] = []
    if local:
        exporters.append(LocalExporter(traces_dir))
    if console:
        exporters.append(ConsoleExporter())
    if otlp_endpoint:
        exporters.append(OTLPExporter(otlp_endpoint, otlp_headers, service_name=project or "langtrain-app"))
    _cloud = CloudExporter(client) if cloud else None
    if _cloud:
        exporters.append(_cloud)
    _tracer = Tracer(exporters, sample_rate=sample_rate, redact_pii=redact_pii,
                     detect_injection=detect_injection, project=project)
    return _tracer


def span(name: str, kind: str = "task", **attributes: Any):
    """Context manager: `with observe.span("search", kind="retrieval") as s: ...`"""
    return _default_tracer().span(name, kind, **attributes)


def observe(name: Optional[str] = None, kind: str = "task", capture_io: bool = True) -> Callable:
    """Decorator: trace each call of a function as a span."""
    def wrap(fn: Callable) -> Callable:
        traced: Dict[str, Callable] = {}

        # Resolve the tracer at call time, so init() after decoration still applies.
        import functools
        import inspect

        def get() -> Callable:
            t = _default_tracer()
            key = str(id(t))
            if key not in traced:
                traced.clear()
                traced[key] = t.observe(name or fn.__qualname__, kind, capture_io)(fn)
            return traced[key]

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a: Any, **k: Any) -> Any:
                return await get()(*a, **k)
            return aw

        @functools.wraps(fn)
        def w(*a: Any, **k: Any) -> Any:
            return get()(*a, **k)
        return w
    return wrap


def flush() -> None:
    """Send any spans still waiting in batch exporters."""
    if _tracer:
        _tracer.flush()


def _spans(spans: Optional[List[Span]], since_hours: Optional[float]) -> List[Span]:
    return spans if spans is not None else load_spans(since_hours=since_hours)


# ── Free ─────────────────────────────────────────────────────────────────────

def summary(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    """Traffic, tokens, cost, latency and errors for local traces."""
    return analysis.summarize(_spans(spans, since_hours))


def traces(limit: int = 20, since_hours: Optional[float] = None, spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
    return analysis.trace_summaries(_spans(spans, since_hours), limit)


def errors(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
    return analysis.error_list(_spans(spans, since_hours))


def tools(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    return analysis.tool_stats(_spans(spans, since_hours))


def trace(trace_id: str) -> List[Span]:
    """Every span of one trace (a prefix of the id is enough), in start order."""
    return [s for s in load_spans() if s.trace_id.startswith(trace_id)]


evaluate = evals.evaluate


# ── Pro ──────────────────────────────────────────────────────────────────────

def _pro(capability: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    entitlements.require(capability)
    return fn(*args, **kwargs)


def models(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    """Per-model latency, errors, tokens and cost (Pro)."""
    return _pro("model_performance", analysis.model_stats, _spans(spans, since_hours))


def cost_per_success(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    return _pro("cost_per_success", analysis.cost_per_success, _spans(spans, since_hours))


def rate_limits(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    return _pro("api_rate_limits", analysis.rate_limits, _spans(spans, since_hours))


def caching(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    return _pro("caching_analytics", analysis.caching, _spans(spans, since_hours))


def rag(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None, threshold: float = 0.5) -> Dict[str, Any]:
    """Groundedness of answers in their retrieved context, and likely hallucinations (Pro)."""
    entitlements.require("hallucination_detection")
    return _pro("rag_observability", analysis.rag, _spans(spans, since_hours), threshold)


def drift(baseline_hours: float = 168, current_hours: float = 24) -> Dict[str, Any]:
    """Compare the last `current_hours` with the window before it (Pro)."""
    entitlements.require("drift_detection")
    now = time.time_ns()
    cut = now - int(current_hours * 3600e9)
    spans = load_spans(since_hours=baseline_hours + current_hours)
    return analysis.drift([s for s in spans if s.start_ns < cut], [s for s in spans if s.start_ns >= cut])


def anomalies(metric: str = "latency_ms", z: float = 3.0, since_hours: Optional[float] = 24,
              spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
    return _pro("anomaly_detection", analysis.anomalies, _spans(spans, since_hours), metric, z)


def root_cause(since_hours: Optional[float] = 24, spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
    return _pro("root_cause_analysis", analysis.root_cause, _spans(spans, since_hours))


def recommendations(since_hours: Optional[float] = 168, spans: Optional[List[Span]] = None) -> List[Dict[str, str]]:
    return _pro("optimization_recommendations", analysis.recommendations, _spans(spans, since_hours))


def sla(p95_ms: float, max_error_rate: float = 0.01, since_hours: Optional[float] = 24,
        spans: Optional[List[Span]] = None) -> Dict[str, Any]:
    """Check latency and error objectives (Enterprise)."""
    return _pro("sla_monitoring", analysis.sla, _spans(spans, since_hours), p95_ms, max_error_rate)


def regression(name: str, cases: Sequence[Dict[str, Any]], fn: Callable[[Any], Any],
               scorers: Sequence[evals.Scorer], tolerance: float = 0.02) -> Dict[str, Any]:
    """Run an eval and compare it with the last saved run of the same name (Pro). Fails CI via result["passed"]."""
    entitlements.require("regression_testing")
    baseline = evals.latest(name)
    report = evals.evaluate(cases, fn, scorers, name=name)
    if baseline is None:
        return {"baseline": None, "candidate": name, "passed": True, "note": "First run saved as the baseline."}
    return evals.compare_reports(baseline, report, tolerance)


def ab_test(variants: Dict[str, Callable[[Any], Any]], cases: Sequence[Dict[str, Any]],
            scorers: Sequence[evals.Scorer], name: str = "ab") -> Dict[str, Any]:
    """Compare prompt or model variants on the same cases and pick a winner (Pro)."""
    entitlements.require("prompt_ab_testing")
    return evals.ab(variants, cases, scorers, name)


def compare_models(variants: Dict[str, Callable[[Any], Any]], cases: Sequence[Dict[str, Any]],
                   scorers: Sequence[evals.Scorer]) -> Dict[str, Any]:
    """A leaderboard of models on the same cases (Pro)."""
    entitlements.require("model_comparison")
    return evals.ab(variants, cases, scorers, name="leaderboard")


def replay(span_or_trace: Union[Span, str], fn: Callable[[Any], Any]) -> Dict[str, Any]:
    """Re-run a recorded llm call's input through `fn` (another model or prompt) and show both outputs (Pro)."""
    entitlements.require("trace_replay")
    if isinstance(span_or_trace, Span):
        target = span_or_trace
    else:
        llm = [s for s in trace(span_or_trace) if s.kind == "llm"]
        if not llm:
            raise ValueError(f"No llm span in trace {span_or_trace!r}")
        target = llm[-1]
    started = time.perf_counter()
    new_output = fn(target.attributes.get("input"))
    return {"trace_id": target.trace_id, "span": target.name, "input": target.attributes.get("input"),
            "original_output": target.attributes.get("output"), "replayed_output": new_output,
            "changed": new_output != target.attributes.get("output"),
            "replay_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "original_latency_ms": target.latency_ms}


def feedback(trace_id: str, score: float, comment: str = "", user_id: Optional[str] = None) -> Dict[str, Any]:
    """Attach a user rating (0..1) to a trace: saved locally, and in Langtrain when cloud is on (Pro)."""
    entitlements.require("production_feedback")
    if not 0 <= score <= 1:
        raise ValueError("score must be between 0 and 1")
    entry = {"trace_id": trace_id, "score": score, "comment": comment, "user_id": user_id, "at": time.time()}
    path = os.path.join(os.path.expanduser("~"), ".langtrain", "feedback.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if _cloud is not None:
        _cloud.client._post("/api/v1/observe/feedback", entry)
    return entry


from . import testing  # noqa: E402  (suites call observe.span)

__all__ = [
    "init", "span", "observe", "flush", "current_span", "wrap_openai", "set_price",
    "summary", "traces", "trace", "errors", "tools", "evaluate",
    "models", "cost_per_success", "rate_limits", "caching", "rag", "drift", "anomalies",
    "root_cause", "recommendations", "sla", "regression", "ab_test", "compare_models", "replay", "feedback",
    "Span", "Tracer", "LocalExporter", "CloudExporter", "OTLPExporter", "ConsoleExporter", "load_spans",
    "ProFeatureError", "analysis", "capabilities", "entitlements", "evals", "prompts", "redact", "security", "testing",
]
