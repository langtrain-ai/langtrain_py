"""
Analytics over spans: plain computations with no plan checks, so the
Langtrain server runs the same code on stored traces that this package runs
on local ones. Plan checks live in the public functions in
langtrain.observe and in the server's endpoints.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .tracing import Span


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 3)


def _lat(spans: Iterable[Span]) -> Dict[str, Optional[float]]:
    vals = [s.latency_ms for s in spans if s.latency_ms is not None]
    return {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "p99": percentile(vals, 99)}


def _num(a: Dict[str, Any], key: str) -> float:
    v = a.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def group_traces(spans: Iterable[Span]) -> Dict[str, List[Span]]:
    out: Dict[str, List[Span]] = defaultdict(list)
    for s in spans:
        out[s.trace_id].append(s)
    return out


def root_of(trace: List[Span]) -> Span:
    roots = [s for s in trace if s.parent_id is None]
    return roots[0] if roots else min(trace, key=lambda s: s.start_ns or 0)


# ── Free: usage, latency, errors, tools ─────────────────────────────────────

def summarize(spans: List[Span]) -> Dict[str, Any]:
    """Totals for a window of spans: traffic, tokens, cost, latency and errors."""
    traces = group_traces(spans)
    llm = [s for s in spans if s.kind == "llm"]
    errors = [s for s in spans if s.status == "error"]
    starts = [s.start_ns for s in spans if s.start_ns]
    minutes = max((max(starts) - min(starts)) / 60e9, 1 / 60) if starts else None
    return {
        "traces": len(traces),
        "spans": len(spans),
        "llm_calls": len(llm),
        "tool_calls": sum(1 for s in spans if s.kind == "tool"),
        "input_tokens": int(sum(_num(s.attributes, "input_tokens") for s in spans)),
        "output_tokens": int(sum(_num(s.attributes, "output_tokens") for s in spans)),
        "cost": round(sum(_num(s.attributes, "cost") for s in spans), 6),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(spans), 4) if spans else 0.0,
        "latency_ms": {"llm": _lat(llm), "traces": _lat(root_of(t) for t in traces.values())},
        "llm_calls_per_min": round(len(llm) / minutes, 2) if minutes else 0.0,
    }


def trace_summaries(spans: List[Span], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    for tid, trace in group_traces(spans).items():
        root = root_of(trace)
        rows.append({
            "trace_id": tid, "name": root.name, "start_ns": root.start_ns, "latency_ms": root.latency_ms,
            "status": "error" if any(s.status == "error" for s in trace) else "ok",
            "spans": len(trace),
            "tokens": int(sum(_num(s.attributes, "input_tokens") + _num(s.attributes, "output_tokens") for s in trace)),
            "cost": round(sum(_num(s.attributes, "cost") for s in trace), 6),
        })
    rows.sort(key=lambda r: r["start_ns"] or 0, reverse=True)
    return rows[:limit] if limit else rows


def error_list(spans: List[Span]) -> List[Dict[str, Any]]:
    return [{"trace_id": s.trace_id, "span": s.name, "kind": s.kind, "model": s.attributes.get("model"),
             "error": (s.error or {}).get("type"), "message": (s.error or {}).get("message")}
            for s in spans if s.status == "error"]


def tool_stats(spans: List[Span]) -> Dict[str, Dict[str, Any]]:
    by: Dict[str, List[Span]] = defaultdict(list)
    for s in spans:
        if s.kind == "tool":
            by[s.name].append(s)
    return {name: {"calls": len(ss), "errors": sum(1 for s in ss if s.status == "error"), "latency_ms": _lat(ss)}
            for name, ss in sorted(by.items())}


# ── Pro: models, success cost, rate limits, caching ──────────────────────────

def model_stats(spans: List[Span]) -> Dict[str, Dict[str, Any]]:
    """Per model: calls, error rate, latency, tokens and cost (model performance & comparison)."""
    by: Dict[str, List[Span]] = defaultdict(list)
    for s in spans:
        if s.kind == "llm":
            by[s.attributes.get("model") or "unknown"].append(s)
    out = {}
    for model, ss in by.items():
        cost = sum(_num(s.attributes, "cost") for s in ss)
        errs = sum(1 for s in ss if s.status == "error")
        out[model] = {
            "calls": len(ss), "errors": errs, "error_rate": round(errs / len(ss), 4),
            "latency_ms": _lat(ss),
            "input_tokens": int(sum(_num(s.attributes, "input_tokens") for s in ss)),
            "output_tokens": int(sum(_num(s.attributes, "output_tokens") for s in ss)),
            "cost": round(cost, 6), "cost_per_call": round(cost / len(ss), 6),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["calls"]))


def cost_per_success(spans: List[Span]) -> Dict[str, Any]:
    """Total spend divided by traces that succeeded (no errors, and not marked success=False)."""
    traces = group_traces(spans)
    ok = [t for t in traces.values()
          if all(s.status == "ok" for s in t) and root_of(t).attributes.get("success", True) is not False]
    total = sum(_num(s.attributes, "cost") for s in spans)
    return {"cost": round(total, 6), "traces": len(traces), "successful": len(ok),
            "success_rate": round(len(ok) / len(traces), 4) if traces else 0.0,
            "cost_per_success": round(total / len(ok), 6) if ok else None}


def rate_limits(spans: List[Span]) -> Dict[str, Any]:
    llm = [s for s in spans if s.kind == "llm"]
    per_min = Counter(int((s.start_ns or 0) // 60e9) for s in llm)
    hits = [s for s in spans if s.attributes.get("http_status") == 429]
    return {"calls": len(llm), "peak_calls_per_min": max(per_min.values()) if per_min else 0,
            "rate_limited": len(hits), "rate_limited_models": dict(Counter(s.attributes.get("model") or "unknown" for s in hits))}


def caching(spans: List[Span]) -> Dict[str, Any]:
    llm = [s for s in spans if s.kind == "llm" and "cache_hit" in s.attributes]
    hits = [s for s in llm if s.attributes["cache_hit"]]
    misses = [s for s in llm if not s.attributes["cache_hit"]]
    avg_cost: Dict[str, float] = {}
    for model, ss in group_by(misses, lambda s: s.attributes.get("model") or "unknown").items():
        avg_cost[model] = sum(_num(s.attributes, "cost") for s in ss) / len(ss)
    saved = sum(avg_cost.get(s.attributes.get("model") or "unknown", 0.0) for s in hits)
    return {"lookups": len(llm), "hits": len(hits), "hit_rate": round(len(hits) / len(llm), 4) if llm else 0.0,
            "estimated_cost_saved": round(saved, 6),
            "latency_ms": {"hit": _lat(hits), "miss": _lat(misses)}}


def group_by(spans: Iterable[Span], key) -> Dict[Any, List[Span]]:
    out: Dict[Any, List[Span]] = defaultdict(list)
    for s in spans:
        out[key(s)].append(s)
    return out


# ── Pro: RAG and hallucination ───────────────────────────────────────────────

_STOP = set("""the and for are but not you all any can had her was one our out has his how its may new now old see two who
boy did get him let put say she too use that with this from they will would there their what about which when your
have been were than them then into more some such only also just like make over very most must should could""".split())


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP]


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return " ".join(_text(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return " ".join(_text(x) for x in v)
    return ""


def groundedness(answer: Any, context: Any) -> Optional[float]:
    """Share of the answer's content words that appear in the context (0..1). A heuristic, not a verdict."""
    words = _words(_text(answer))
    if not words:
        return None
    ctx = set(_words(_text(context)))
    if not ctx:
        return 0.0
    return round(sum(1 for w in words if w in ctx) / len(words), 4)


def rag(spans: List[Span], threshold: float = 0.5) -> Dict[str, Any]:
    """For traces with retrieval then generation: groundedness, and the answers likely to be hallucinated."""
    rows = []
    for tid, trace in group_traces(spans).items():
        docs = [s.attributes.get("documents") or s.attributes.get("output") for s in trace if s.kind == "retrieval"]
        answers = [s.attributes.get("output") for s in trace if s.kind == "llm" and s.attributes.get("output")]
        if not docs or not answers:
            continue
        g = groundedness(answers[-1], docs)
        rows.append({"trace_id": tid, "groundedness": g, "documents": sum(len(d) if isinstance(d, list) else 1 for d in docs if d)})
    scored = [r["groundedness"] for r in rows if r["groundedness"] is not None]
    return {"rag_traces": len(rows), "avg_groundedness": round(sum(scored) / len(scored), 4) if scored else None,
            "possible_hallucinations": [r for r in rows if r["groundedness"] is not None and r["groundedness"] < threshold]}


# ── Pro: drift, anomalies, root cause, recommendations ───────────────────────

def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> Optional[float]:
    """Population stability index of `actual` against `expected`. <0.1 stable, 0.1-0.25 moderate, >0.25 drift."""
    e = [v for v in expected if v is not None]
    a = [v for v in actual if v is not None]
    if len(e) < 2 or len(a) < 2:
        return None
    edges = sorted(set(percentile(e, 100 * i / bins) for i in range(1, bins)))

    def dist(vals: List[float]) -> List[float]:
        counts = [0] * (len(edges) + 1)
        for v in vals:
            counts[sum(1 for edge in edges if v > edge)] += 1
        return [max(c / len(vals), 1e-4) for c in counts]

    pe, pa = dist(e), dist(a)
    return round(sum((x - y) * math.log(x / y) for x, y in zip(pa, pe)), 4)


def _metric(s: Span, metric: str) -> Optional[float]:
    if metric == "latency_ms":
        return s.latency_ms
    v = s.attributes.get(metric)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def drift(baseline: List[Span], current: List[Span], metrics: Sequence[str] = ("latency_ms", "output_tokens", "cost")) -> Dict[str, Any]:
    out = {}
    for m in metrics:
        value = psi([_metric(s, m) for s in baseline if s.kind == "llm"], [_metric(s, m) for s in current if s.kind == "llm"])
        out[m] = {"psi": value, "drift": None if value is None else ("high" if value > 0.25 else "moderate" if value > 0.1 else "none")}
    return out


def anomalies(spans: List[Span], metric: str = "latency_ms", z: float = 3.0) -> List[Dict[str, Any]]:
    """LLM spans whose `metric` is more than `z` standard deviations from the mean."""
    pts = [(s, _metric(s, metric)) for s in spans if s.kind == "llm"]
    vals = [v for _, v in pts if v is not None]
    if len(vals) < 5:
        return []
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    if sd == 0:
        return []
    return [{"trace_id": s.trace_id, "span": s.name, "model": s.attributes.get("model"), metric: v, "z": round((v - mean) / sd, 2)}
            for s, v in pts if v is not None and abs(v - mean) / sd > z]


def root_cause(spans: List[Span]) -> List[Dict[str, Any]]:
    """Failures grouped by where they happen and why, biggest group first."""
    groups: Dict[tuple, List[Span]] = defaultdict(list)
    for s in spans:
        if s.status == "error":
            where = s.attributes.get("model") if s.kind == "llm" else s.name
            groups[(s.kind, where, (s.error or {}).get("type"))].append(s)
    total = sum(len(v) for v in groups.values()) or 1
    rows = [{"kind": k[0], "where": k[1], "error": k[2], "count": len(v), "share": round(len(v) / total, 4),
             "example_message": (v[0].error or {}).get("message"), "example_traces": [s.trace_id for s in v[:3]]}
            for k, v in groups.items()]
    return sorted(rows, key=lambda r: -r["count"])


def recommendations(spans: List[Span]) -> List[Dict[str, str]]:
    """Rule-based suggestions to cut cost, latency and failures, from what the traces show."""
    recs: List[Dict[str, str]] = []
    models = model_stats(spans)
    summary = summarize(spans)
    if models:
        top_cost = max(models.items(), key=lambda kv: kv[1]["cost"])
        cheapest = min((kv for kv in models.items() if kv[1]["error_rate"] <= 0.05 and kv[1]["calls"] >= 5),
                       key=lambda kv: kv[1]["cost_per_call"], default=None)
        if cheapest and top_cost[0] != cheapest[0] and top_cost[1]["cost_per_call"] > 2 * cheapest[1]["cost_per_call"] > 0:
            recs.append({"area": "cost", "action": f"Route simpler requests from {top_cost[0]} to {cheapest[0]}: "
                         f"{top_cost[1]['cost_per_call'] / cheapest[1]['cost_per_call']:.1f}x cheaper per call with a low error rate. Check quality with an A/B test first."})
    cache = caching(spans)
    if not cache["lookups"] and summary["llm_calls"] >= 50:
        recs.append({"area": "cost", "action": "No cache data recorded. If users repeat questions, add a response cache and record cache_hit on llm spans."})
    elif cache["lookups"] and cache["hit_rate"] < 0.1:
        recs.append({"area": "cost", "action": f"Cache hit rate is {cache['hit_rate']:.0%}. Normalise prompts (whitespace, casing) before the cache lookup."})
    out_tok = [_num(s.attributes, "output_tokens") for s in spans if s.kind == "llm" and s.attributes.get("output_tokens")]
    if out_tok and percentile(out_tok, 95) and percentile(out_tok, 95) > 1500:
        recs.append({"area": "latency", "action": f"p95 output is {percentile(out_tok, 95):.0f} tokens. Set max_tokens or ask for shorter answers to cut latency and cost."})
    rl = rate_limits(spans)
    if rl["rate_limited"]:
        recs.append({"area": "reliability", "action": f"{rl['rate_limited']} calls hit rate limits (429). Add retries with backoff or spread load across models."})
    rc = root_cause(spans)
    if rc and rc[0]["share"] >= 0.5 and rc[0]["count"] >= 3:
        recs.append({"area": "reliability", "action": f"{rc[0]['share']:.0%} of failures are {rc[0]['error']} in {rc[0]['where']}. Fix that first."})
    rag_report = rag(spans)
    if rag_report["avg_groundedness"] is not None and rag_report["avg_groundedness"] < 0.5:
        recs.append({"area": "quality", "action": f"Answers average {rag_report['avg_groundedness']:.0%} groundedness in retrieved context. Improve retrieval or tell the model to answer only from context."})
    return recs


def sla(spans: List[Span], p95_ms: float, max_error_rate: float) -> Dict[str, Any]:
    traces = group_traces(spans)
    p95 = percentile([root_of(t).latency_ms for t in traces.values()], 95)
    failed = sum(1 for t in traces.values() if any(s.status == "error" for s in t))
    err = round(failed / len(traces), 4) if traces else 0.0
    return {"p95_ms": p95, "p95_target_ms": p95_ms, "error_rate": err, "error_rate_target": max_error_rate,
            "latency_ok": p95 is None or p95 <= p95_ms, "errors_ok": err <= max_error_rate,
            "met": (p95 is None or p95 <= p95_ms) and err <= max_error_rate}
