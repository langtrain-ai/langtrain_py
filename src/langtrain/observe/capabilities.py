"""
The observability capabilities Langtrain offers, with the plan each needs.

This list is the one place a capability's tier is decided. The server keeps
the same ids (langtrain-server: app/core/observability.py) and is the real
gate for anything that runs in Langtrain's cloud; the checks in this package
only explain what a plan includes.

tier:    "free" works for everyone; "pro" needs a Pro (or Enterprise) plan;
         "enterprise" needs Enterprise.
status:  "available" ships in this package; "cloud" is served by the Langtrain
         API; "planned" is on the roadmap and not usable yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

TIERS = ("free", "pro", "enterprise")


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    tier: str
    status: str
    summary: str


CAPABILITIES: List[Capability] = [
    # Capture: free, so every app can be traced from day one.
    Capability("llm_observability", "LLM Observability", "free", "available", "Every model call as a span: input, output, model, tokens, latency."),
    Capability("agent_observability", "Agent Observability", "free", "available", "Agent runs as traces of nested steps."),
    Capability("token_cost_analytics", "Token Usage & Cost Analytics", "free", "available", "Tokens and cost per call, model and trace."),
    Capability("llm_evaluation", "LLM Evaluation & Benchmarking", "free", "available", "Score outputs with built-in or custom scorers."),
    Capability("prompt_management", "Prompt Management & Versioning", "free", "available", "Versioned prompt templates, stored locally."),
    Capability("latency_throughput", "Latency & Throughput Monitoring", "free", "available", "p50/p95/p99 latency and calls per minute."),
    Capability("error_tracking", "Error Tracking & Debugging", "free", "available", "Failed spans with the exception and where it happened."),
    Capability("agent_workflow_tracing", "Agent Workflow Tracing", "free", "available", "The tree of steps an agent took, in order."),
    Capability("tool_call_monitoring", "Tool Call Monitoring", "free", "available", "Calls, failures and latency per tool."),
    Capability("privacy_pii", "Privacy & PII Detection", "free", "available", "Emails, phones, cards, PAN, Aadhaar and keys redacted before export."),
    Capability("security_injection", "Security & Prompt Injection Detection", "free", "available", "Flags inputs that try to override instructions."),
    Capability("opentelemetry", "OpenTelemetry Integration", "free", "available", "Export spans over OTLP/HTTP with GenAI attributes."),
    # Analysis: Pro.
    Capability("model_performance", "Model Performance Monitoring", "pro", "available", "Latency, errors and cost per model over time."),
    Capability("hallucination_detection", "Hallucination Detection", "pro", "available", "Answers checked against their retrieved context."),
    Capability("rag_observability", "RAG Observability", "pro", "available", "Retrieval spans with context relevance and groundedness."),
    Capability("model_comparison", "Model Comparison & Leaderboards", "pro", "available", "Models ranked on the same cases and live traffic."),
    Capability("cost_per_success", "Cost per Successful Task", "pro", "available", "Spend divided by traces that succeeded."),
    Capability("production_feedback", "Production Feedback Collection", "pro", "available", "User ratings attached to traces."),
    Capability("regression_testing", "Regression Testing", "pro", "available", "Fail a build when scores drop against a baseline."),
    Capability("trace_replay", "Trace Replay & Debugging", "pro", "available", "Re-run a recorded call against another model or prompt."),
    Capability("prompt_ab_testing", "Prompt A/B Testing", "pro", "available", "Compare prompt or model variants on the same cases."),
    Capability("drift_detection", "Model Drift Detection", "pro", "available", "PSI drift on latency, tokens and scores."),
    Capability("anomaly_detection", "Anomaly Detection & Alerts", "pro", "available", "Spikes in latency, cost or errors."),
    Capability("api_rate_limits", "API Usage & Rate Limit Monitoring", "pro", "available", "Calls per minute and rate-limit (429) hits."),
    Capability("caching_analytics", "Caching Analytics", "pro", "available", "Cache hit rate and the cost it saved."),
    Capability("root_cause_analysis", "Automated Root Cause Analysis", "pro", "available", "Failures grouped by model, tool and error type."),
    Capability("optimization_recommendations", "AI-Powered Optimization Recommendations", "pro", "available", "Concrete changes to cut cost and latency."),
    Capability("cloud_traces", "Cloud Trace Storage & Dashboards", "pro", "cloud", "Traces stored in Langtrain and searchable in the dashboard."),
    Capability("safety_guardrails", "AI Safety & Guardrails", "pro", "cloud", "Guardrail rules run on inputs and outputs (Langtrain API)."),
    Capability("training_experiments", "Training Experiment Tracking", "pro", "cloud", "Runs, metrics and comparisons (Langtrain API)."),
    Capability("dataset_validation", "Dataset Quality & Validation", "free", "available", "DatasetIntelligence checks a dataset before training."),
    Capability("finetune_observability", "Fine-Tuning Observability", "pro", "planned", "Loss, gradients and evals of a fine-tune, live."),
    Capability("gpu_monitoring", "GPU & Infrastructure Monitoring", "pro", "planned", "GPU utilisation, memory and cost per run."),
    Capability("model_routing", "Model Routing & Optimization", "pro", "planned", "Send each request to the cheapest model that passes."),
    Capability("deployment_monitoring", "Deployment Monitoring", "pro", "planned", "Health and traffic of each deployed endpoint."),
    Capability("model_versioning", "Model Versioning & Rollbacks", "pro", "planned", "Pin, promote and roll back model versions."),
    Capability("multimodal_observability", "Multimodal Observability", "pro", "planned", "Image and audio inputs and outputs in traces."),
    Capability("synthetic_load_testing", "Synthetic Testing & Load Testing", "pro", "planned", "Generated test cases and load runs."),
    Capability("human_review", "Human-in-the-Loop Evaluation", "pro", "planned", "A review queue for people to score outputs."),
    Capability("continuous_evaluation", "Continuous Evaluation & Improvement Loops", "pro", "planned", "Scheduled evals on fresh traffic that feed retraining."),
    Capability("sla_monitoring", "SLA & Reliability Monitoring", "enterprise", "available", "Latency and error objectives checked over a window."),
]

BY_ID: Dict[str, Capability] = {c.id: c for c in CAPABILITIES}


def get(capability_id: str) -> Capability:
    try:
        return BY_ID[capability_id]
    except KeyError:
        raise KeyError(f"Unknown capability {capability_id!r}") from None


def tier_allows(plan: str, tier: str) -> bool:
    """True when `plan` includes capabilities of `tier` (enterprise ⊃ pro ⊃ free)."""
    rank = {t: i for i, t in enumerate(TIERS)}
    return rank.get(plan, 0) >= rank[tier]
