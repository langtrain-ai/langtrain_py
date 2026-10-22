"""
Model test suites for any OpenAI-compatible model (Ollama by default),
ported from langtrain_harness: benchmarks, evals, long-context retrieval,
consistency, function calling, secret and PII leakage, red-teaming, load
testing and a blinded arena with Elo. See suites.py.
"""

from .suites import (  # noqa: F401
    arena,
    benchmark,
    consistency,
    elo_ratings,
    eval_suite,
    leakage,
    llm_judge,
    load_test,
    needle,
    pii_leakage,
    red_team,
    tool_calling,
)
from .target import FunctionTarget, Reply, Target  # noqa: F401
from . import datasets  # noqa: F401

__all__ = ["Target", "FunctionTarget", "Reply", "datasets", "benchmark", "eval_suite", "leakage", "pii_leakage",
           "needle", "consistency", "tool_calling", "red_team", "load_test", "arena", "elo_ratings", "llm_judge"]
