"""
Evaluation: run cases through a function and score the outputs.

    from langtrain.observe import evals

    report = evals.evaluate(
        cases=[{"input": "2+2?", "expected": "4"}],
        fn=my_app,
        scorers=[evals.contains(), evals.max_length(200)],
        name="smoke",
    )
    print(report.means, report.pass_rate)

A scorer is any function (output, case) -> float in [0, 1]. Reports are
saved to ~/.langtrain/evals so later runs can be compared against them.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import analysis

Scorer = Callable[[Any, Dict[str, Any]], float]
def _evals_dir() -> str:
    return os.environ.get("LANGTRAIN_EVALS_DIR") or os.path.join(os.path.expanduser("~"), ".langtrain", "evals")


def _named(name: str, fn: Scorer) -> Scorer:
    fn.__name__ = name
    return fn


def exact_match(ignore_case: bool = True) -> Scorer:
    def score(output: Any, case: Dict[str, Any]) -> float:
        a, b = str(output).strip(), str(case.get("expected", "")).strip()
        return float(a.lower() == b.lower() if ignore_case else a == b)
    return _named("exact_match", score)


def contains(ignore_case: bool = True) -> Scorer:
    def score(output: Any, case: Dict[str, Any]) -> float:
        a, b = str(output), str(case.get("expected", ""))
        return float((b.lower() in a.lower()) if ignore_case else (b in a))
    return _named("contains", score)


def regex(pattern: str) -> Scorer:
    rx = re.compile(pattern)
    return _named(f"regex:{pattern}", lambda output, case: float(bool(rx.search(str(output)))))


def json_valid() -> Scorer:
    def score(output: Any, case: Dict[str, Any]) -> float:
        try:
            json.loads(output if isinstance(output, str) else json.dumps(output))
            return 1.0
        except (TypeError, ValueError):
            return 0.0
    return _named("json_valid", score)


def max_length(chars: int) -> Scorer:
    return _named(f"max_length:{chars}", lambda output, case: float(len(str(output)) <= chars))


def grounded(threshold: float = 0.5) -> Scorer:
    """1 when the answer is grounded in case["context"] (the hallucination check), else 0."""
    def score(output: Any, case: Dict[str, Any]) -> float:
        g = analysis.groundedness(output, case.get("context"))
        return float(g is not None and g >= threshold)
    return _named("grounded", score)


@dataclass
class EvalReport:
    name: str
    created_at: float
    rows: List[Dict[str, Any]] = field(default_factory=list)
    means: Dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        with open(path) as f:
            return cls(**{**json.load(f), "path": path})


def evaluate(cases: Sequence[Dict[str, Any]], fn: Callable[[Any], Any], scorers: Sequence[Scorer],
             name: str = "eval", pass_threshold: float = 1.0, save: bool = True) -> EvalReport:
    """Run each case's input through `fn`, score it, and summarise. A case passes when every score ≥ pass_threshold."""
    rows = []
    for i, case in enumerate(cases):
        started = time.perf_counter()
        try:
            output, error = fn(case.get("input")), None
        except Exception as exc:
            output, error = None, f"{type(exc).__name__}: {exc}"
        scores = {s.__name__: (0.0 if error else float(s(output, case))) for s in scorers}
        rows.append({"case": i, "input": case.get("input"), "expected": case.get("expected"), "output": output,
                     "error": error, "scores": scores, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                     "passed": error is None and all(v >= pass_threshold for v in scores.values())})
    means = {s.__name__: round(sum(r["scores"][s.__name__] for r in rows) / len(rows), 4) for s in scorers} if rows else {}
    report = EvalReport(name=name, created_at=time.time(), rows=rows, means=means,
                        pass_rate=round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else 0.0)
    if save:
        directory = _evals_dir()
        os.makedirs(directory, exist_ok=True)
        report.path = os.path.join(directory, f"{name}-{int(report.created_at * 1000)}.json")
        with open(report.path, "w") as f:
            json.dump({k: v for k, v in report.to_dict().items() if k != "path"}, f, default=str)
    return report


def latest(name: str) -> Optional[EvalReport]:
    """The most recent saved report called `name`."""
    directory = _evals_dir()
    if not os.path.isdir(directory):
        return None
    files = sorted((f for f in os.listdir(directory) if f.startswith(f"{name}-") and f.endswith(".json")),
                   key=lambda f: int(f[len(name) + 1:-5]) if f[len(name) + 1:-5].isdigit() else 0)
    return EvalReport.load(os.path.join(directory, files[-1])) if files else None


def compare_reports(baseline: EvalReport, candidate: EvalReport, tolerance: float = 0.02) -> Dict[str, Any]:
    """Regression check: which scores dropped by more than `tolerance`."""
    deltas = {k: round(candidate.means.get(k, 0.0) - v, 4) for k, v in baseline.means.items()}
    regressions = {k: d for k, d in deltas.items() if d < -tolerance}
    return {"baseline": baseline.name, "candidate": candidate.name, "deltas": deltas,
            "pass_rate_delta": round(candidate.pass_rate - baseline.pass_rate, 4),
            "regressions": regressions, "passed": not regressions}


def ab(variants: Dict[str, Callable[[Any], Any]], cases: Sequence[Dict[str, Any]], scorers: Sequence[Scorer],
       name: str = "ab") -> Dict[str, Any]:
    """Run every variant on the same cases; rank by mean score, then pass rate, then latency."""
    reports = {v: evaluate(cases, fn, scorers, name=f"{name}-{v}", save=False) for v, fn in variants.items()}

    def rank_key(item):
        r = item[1]
        mean = sum(r.means.values()) / len(r.means) if r.means else 0.0
        lat = sum(row["latency_ms"] for row in r.rows) / len(r.rows) if r.rows else 0.0
        return (-mean, -r.pass_rate, lat)

    ranked = sorted(reports.items(), key=rank_key)
    return {"winner": ranked[0][0] if ranked else None,
            "results": {v: {"means": r.means, "pass_rate": r.pass_rate,
                            "avg_latency_ms": round(sum(x["latency_ms"] for x in r.rows) / len(r.rows), 2) if r.rows else None}
                        for v, r in ranked}}
