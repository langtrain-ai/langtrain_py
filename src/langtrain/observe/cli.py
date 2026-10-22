"""
langtrain-observe: inspect local traces from the terminal.

    langtrain-observe features            what each plan includes
    langtrain-observe stats --since 24    tokens, cost, latency, errors
    langtrain-observe traces              recent traces
    langtrain-observe show <trace-id>     one trace as a tree
    langtrain-observe recommend           ways to cut cost and latency (Pro)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from rich import box
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from . import analysis, capabilities, entitlements, prompts
from .entitlements import ProFeatureError
from .exporters import OTLPExporter, load_spans, to_otlp

console = Console()


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.2f}s" if v >= 1000 else (f"{v:.0f}ms" if v >= 10 else f"{v:.2f}ms")


def _print_json(obj: Any) -> None:
    console.print_json(json.dumps(obj, default=str))


def cmd_features(args: argparse.Namespace) -> None:
    ent = entitlements.current()
    table = Table(title=f"Langtrain observability · your plan: {ent.plan.title()}", box=box.SIMPLE_HEAD)
    for col in ("Capability", "Tier", "Status", "Access"):
        table.add_column(col)
    for cap in capabilities.CAPABILITIES:
        if cap.status == "planned":
            access = "[dim]planned[/dim]"
        elif ent.allows(cap.id):
            access = "[green]✓ included[/green]"
        else:
            access = f"[yellow]🔒 {cap.tier.title()}[/yellow]"
        table.add_row(cap.name, cap.tier.title(), cap.status, access)
    console.print(table)
    if ent.plan == "free":
        console.print(f"Unlock Pro features at {entitlements.PRICING_URL}")


def cmd_plan(args: argparse.Namespace) -> None:
    ent = entitlements.current()
    console.print(f"Plan: [bold]{ent.plan.title()}[/bold]")


def cmd_stats(args: argparse.Namespace) -> None:
    s = analysis.summarize(load_spans(since_hours=args.since))
    if args.json:
        return _print_json(s)
    t = Table(title=f"Last {args.since:g}h", box=box.SIMPLE_HEAD, show_header=False)
    t.add_column(); t.add_column(justify="right")
    lat = s["latency_ms"]
    for k, v in [("Traces", s["traces"]), ("LLM calls", s["llm_calls"]), ("Tool calls", s["tool_calls"]),
                 ("Input tokens", f"{s['input_tokens']:,}"), ("Output tokens", f"{s['output_tokens']:,}"),
                 ("Cost", f"{s['cost']:.4f}"), ("Errors", f"{s['errors']} ({s['error_rate']:.1%})"),
                 ("LLM latency p50 / p95", f"{_fmt_ms(lat['llm']['p50'])} / {_fmt_ms(lat['llm']['p95'])}"),
                 ("Trace latency p95", _fmt_ms(lat["traces"]["p95"])), ("LLM calls / min", s["llm_calls_per_min"])]:
        t.add_row(k, str(v))
    console.print(t)


def cmd_traces(args: argparse.Namespace) -> None:
    rows = analysis.trace_summaries(load_spans(since_hours=args.since), args.limit)
    if args.json:
        return _print_json(rows)
    t = Table(box=box.SIMPLE_HEAD)
    for col in ("Trace", "Name", "Status", "Spans", "Latency", "Tokens", "Cost"):
        t.add_column(col)
    for r in rows:
        status = "[green]ok[/green]" if r["status"] == "ok" else "[red]error[/red]"
        t.add_row(r["trace_id"][:12], r["name"], status, str(r["spans"]), _fmt_ms(r["latency_ms"]), f"{r['tokens']:,}", f"{r['cost']:.4f}")
    console.print(t if rows else "No traces yet. Call observe.init() in your app, or use @observe.observe.")


def cmd_show(args: argparse.Namespace) -> None:
    spans = [s for s in load_spans() if s.trace_id.startswith(args.trace_id)]
    if not spans:
        console.print(f"No trace starting with {args.trace_id!r}")
        sys.exit(1)
    by_parent: dict = {}
    for s in spans:
        by_parent.setdefault(s.parent_id, []).append(s)

    def label(s) -> str:
        a = s.attributes
        bits = [f"[bold]{s.name}[/bold]", f"[dim]{s.kind}[/dim]", _fmt_ms(s.latency_ms)]
        if a.get("model"):
            bits.append(str(a["model"]))
        if a.get("input_tokens") or a.get("output_tokens"):
            bits.append(f"{a.get('input_tokens', 0)}→{a.get('output_tokens', 0)} tok")
        if s.status == "error":
            bits.append(f"[red]{(s.error or {}).get('type')}: {(s.error or {}).get('message')}[/red]")
        if a.get("security", {}).get("flagged"):
            bits.append("[yellow]⚠ possible prompt injection[/yellow]")
        if a.get("pii_redacted"):
            bits.append(f"[dim]PII redacted: {', '.join(a['pii_redacted'])}[/dim]")
        return "  ".join(bits)

    roots = by_parent.get(None) or [spans[0]]
    for root in roots:
        tree = Tree(label(root))

        def add(node: Tree, span) -> None:
            for child in sorted(by_parent.get(span.span_id, []), key=lambda c: c.start_ns or 0):
                add(node.add(label(child)), child)
        add(tree, root)
        console.print(tree)


def cmd_errors(args: argparse.Namespace) -> None:
    _print_json(analysis.error_list(load_spans(since_hours=args.since)))


def cmd_tools(args: argparse.Namespace) -> None:
    _print_json(analysis.tool_stats(load_spans(since_hours=args.since)))


def _pro(capability: str, fn, *a: Any) -> None:
    entitlements.require(capability)
    _print_json(fn(*a))


def cmd_otlp(args: argparse.Namespace) -> None:
    spans = load_spans(since_hours=args.since)
    if args.dry_run:
        return _print_json(to_otlp(spans[:5]))
    exporter = OTLPExporter(args.endpoint)
    for s in spans:
        exporter.export(s)
    exporter.flush()
    console.print(f"Sent {len(spans)} spans to {args.endpoint}")


def cmd_prompts(args: argparse.Namespace) -> None:
    _print_json(prompts.list_prompts())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="langtrain-observe", description="Inspect Langtrain traces on this machine.")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_: str, since: Optional[float] = 24) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        if since is not None:
            sp.add_argument("--since", type=float, default=since, help="hours of traces to include")
        return sp

    add("features", cmd_features, "capabilities and what your plan includes", since=None)
    add("plan", cmd_plan, "your Langtrain plan", since=None)
    add("stats", cmd_stats, "tokens, cost, latency and errors").add_argument("--json", action="store_true")
    tr = add("traces", cmd_traces, "recent traces", since=None)
    tr.add_argument("--since", type=float, default=None)
    tr.add_argument("-n", "--limit", type=int, default=20)
    tr.add_argument("--json", action="store_true")
    add("show", cmd_show, "one trace as a tree", since=None).add_argument("trace_id")
    add("errors", cmd_errors, "failed spans")
    add("tools", cmd_tools, "calls, failures and latency per tool")
    add("models", lambda a: _pro("model_performance", analysis.model_stats, load_spans(since_hours=a.since)), "per-model performance (Pro)")
    add("cost", lambda a: _pro("cost_per_success", analysis.cost_per_success, load_spans(since_hours=a.since)), "cost per successful task (Pro)")
    add("rag", lambda a: _pro("rag_observability", analysis.rag, load_spans(since_hours=a.since)), "groundedness and likely hallucinations (Pro)")
    add("cache", lambda a: _pro("caching_analytics", analysis.caching, load_spans(since_hours=a.since)), "cache hit rate and savings (Pro)")
    add("rate-limits", lambda a: _pro("api_rate_limits", analysis.rate_limits, load_spans(since_hours=a.since)), "calls per minute and 429s (Pro)")
    an = add("anomalies", lambda a: _pro("anomaly_detection", analysis.anomalies, load_spans(since_hours=a.since), a.metric, a.z), "outliers in latency, tokens or cost (Pro)")
    an.add_argument("--metric", default="latency_ms")
    an.add_argument("--z", type=float, default=3.0)
    add("rca", lambda a: _pro("root_cause_analysis", analysis.root_cause, load_spans(since_hours=a.since)), "failures grouped by cause (Pro)")
    add("recommend", lambda a: _pro("optimization_recommendations", analysis.recommendations, load_spans(since_hours=a.since)), "ways to cut cost and latency (Pro)", since=168)
    dr = add("drift", lambda a: _pro("drift_detection", _drift, a.baseline, a.since), "drift against an earlier window (Pro)")
    dr.add_argument("--baseline", type=float, default=168, help="hours before the current window")
    sl = add("sla", lambda a: _pro("sla_monitoring", analysis.sla, load_spans(since_hours=a.since), a.p95, a.error_rate), "check latency and error objectives (Enterprise)")
    sl.add_argument("--p95", type=float, required=True, help="p95 latency target in ms")
    sl.add_argument("--error-rate", type=float, default=0.01)
    ot = add("export-otlp", cmd_otlp, "send traces to an OpenTelemetry collector")
    ot.add_argument("endpoint", nargs="?", default="http://localhost:4318/v1/traces")
    ot.add_argument("--dry-run", action="store_true", help="print the OTLP payload instead of sending")
    add("prompts", cmd_prompts, "saved prompt versions", since=None)
    return p


def _drift(baseline_hours: float, current_hours: float) -> Any:
    import time
    cut = time.time_ns() - int(current_hours * 3600e9)
    spans = load_spans(since_hours=baseline_hours + current_hours)
    return analysis.drift([s for s in spans if s.start_ns < cut], [s for s in spans if s.start_ns >= cut])


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ProFeatureError as e:
        console.print(f"[yellow]🔒 {e}[/yellow]")
        sys.exit(2)


if __name__ == "__main__":
    main()
