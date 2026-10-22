# Langtrain Observability

`langtrain.observe` traces LLM apps and agents, evaluates their outputs and
monitors them in production. It ships in the `langtrain-ai` Python package,
the `langtrain-observe` CLI, and the Langtrain API (`/api/v1/observe`).

## Architecture

```
 your app ──► langtrain.observe (SDK) ──────────────► exporters
             spans · @observe · wrap_openai          ├─ LocalExporter    ~/.langtrain/traces/*.jsonl   Free
             PII redaction · injection scan          ├─ OTLPExporter     any OpenTelemetry collector   Free
             cost from your price book               └─ CloudExporter    POST /api/v1/observe/traces   Pro
                                                                          │
 langtrain-observe CLI ◄── load_spans() ◄── local JSONL                    ▼
 observe.summary(), observe.recommendations(), ...                 langtrain-server
                                                                   ├─ observe_spans / observe_feedback (Postgres)
                                                                   ├─ plan gate: app/core/observability.py (402)
                                                                   └─ langtrain.observe.analysis on stored spans
                                                                          │
                                                                   dashboard (planned) ◄── /api/v1/observe/*
```

One package, three surfaces:

- **SDK** (`from langtrain import observe`): capture and analysis in your process.
- **CLI** (`langtrain-observe`): the same analyses on local traces, in the terminal.
- **API** (`/api/v1/observe`): stored traces and the same analyses for the dashboard.

`langtrain.observe.capabilities` is the single list of capabilities and plans.
The server imports it, and so does the SDK, so a capability's tier is decided
in one place. `langtrain.observe.analysis` is plain computation with no plan
checks; the server runs it on stored spans and the SDK on local ones, so every
surface reports the same numbers.

## Plans and enforcement

- **Free**: capture, cost, latency, errors, tools, evals, prompt versions, PII
  redaction, injection detection, OpenTelemetry export, dataset validation.
  Everything runs locally.
- **Pro**: analysis (model performance, RAG and hallucination checks, regression
  and A/B tests, replay, drift, anomalies, root cause, recommendations, caching
  and rate limits, feedback) and cloud trace storage.
- **Enterprise**: SLA and reliability monitoring, plus everything in Pro.

The server is the real gate. `langtrain-ai` is MIT-licensed, so checks inside
the SDK can be edited by anyone; they exist to explain plans and show upgrade
prompts. Everything stored or computed in Langtrain's cloud is enforced by
`app/core/observability.py`, which answers **402 Payment Required** with the
capability and the plan it needs.

A plan's flags appear in the API key validation response as
`features["observe.<capability>"]`, so other clients (CLI, dashboard,
desktop app) read the same answer.

## Capabilities

Status: **available** ships now; **cloud** is served by the Langtrain API;
**planned** is designed but not built yet.

| # | Capability | Plan | Status | SDK / CLI |
|---|---|---|---|---|
| 1 | LLM Observability | Free | available | `observe.span(kind="llm")`, `wrap_openai` |
| 2 | Agent Observability | Free | available | `@observe.observe(kind="agent")` |
| 3 | Token Usage & Cost Analytics | Free | available | `observe.summary()`, `set_price()` · `stats` |
| 4 | LLM Evaluation & Benchmarking | Free | available | `observe.evaluate()`, `testing.eval_suite/needle/consistency/benchmark` · `test eval` |
| 5 | Prompt Management & Versioning | Free | available | `prompts.save/render/get` · `prompts` |
| 6 | Model Performance Monitoring | Pro | available | `observe.models()` · `models` |
| 7 | Latency & Throughput Monitoring | Free | available | `observe.summary()` · `stats` |
| 8 | Error Tracking & Debugging | Free | available | `observe.errors()` · `errors`, `show` |
| 9 | Agent Workflow Tracing | Free | available | nested spans · `show <trace>` |
| 10 | Tool Call Monitoring | Free | available | `kind="tool"` · `tools`; `testing.tool_calling()` |
| 11 | Hallucination Detection | Pro | available | `observe.rag()`, `evals.grounded()` · `rag` |
| 12 | RAG Observability | Pro | available | `kind="retrieval"` spans · `rag` |
| 13 | AI Safety & Guardrails | Pro | available | `testing.red_team()` · `test redteam`; rules: existing `/api/v1/guardrails` |
| 14 | Fine-Tuning Observability | Pro | planned | |
| 15 | Training Experiment Tracking | Pro | cloud | existing `/api/v1/experiments` |
| 16 | Dataset Quality & Validation | Free | available | `DatasetIntelligence.analyze()` |
| 17 | GPU & Infrastructure Monitoring | Pro | planned | |
| 18 | Model Comparison & Leaderboards | Pro | available | `observe.compare_models()`, `testing.arena()` (Elo) · `test arena` |
| 19 | Model Routing & Optimization | Pro | planned | |
| 20 | Cost per Successful Task | Pro | available | `observe.cost_per_success()` · `cost` |
| 21 | Production Feedback Collection | Pro | available | `observe.feedback()` |
| 22 | Regression Testing | Pro | available | `observe.regression()` |
| 23 | Trace Replay & Debugging | Pro | available | `observe.replay()` |
| 24 | Prompt A/B Testing | Pro | available | `observe.ab_test()` |
| 25 | Model Drift Detection | Pro | available | `observe.drift()` · `drift` |
| 26 | Anomaly Detection & Alerts | Pro | available | `observe.anomalies()` · `anomalies` (alerts: planned) |
| 27 | Deployment Monitoring | Pro | planned | |
| 28 | Model Versioning & Rollbacks | Pro | planned | |
| 29 | Multimodal Observability | Pro | planned | |
| 30 | Synthetic Testing & Load Testing | Pro | available | `testing.load_test()` + built-in suites · `test load` |
| 31 | Privacy & PII Detection | Free | available | on by default (`redact_pii=True`); `testing.pii_leakage()` |
| 32 | Security & Prompt Injection Detection | Free | available | on by default · flagged in `show`; `testing.leakage()` |
| 33 | Human-in-the-Loop Evaluation | Pro | planned | |
| 34 | OpenTelemetry Integration | Free | available | `init(otlp_endpoint=...)` · `export-otlp` |
| 35 | API Usage & Rate Limit Monitoring | Pro | available | `observe.rate_limits()` · `rate-limits` |
| 36 | Caching Analytics | Pro | available | `record_usage(cached=...)` · `cache` |
| 37 | SLA & Reliability Monitoring | Enterprise | available | `observe.sla()` · `sla` |
| 38 | Automated Root Cause Analysis | Pro | available | `observe.root_cause()` · `rca` |
| 39 | AI-Powered Optimization Recommendations | Pro | available | `observe.recommendations()` · `recommend` |
| 40 | Continuous Evaluation & Improvement Loops | Pro | planned | |
| + | Cloud Trace Storage & Dashboards | Pro | cloud | `init(cloud=True)` · `/api/v1/observe/traces` |

Today: 31 available, 2 served by the API, 8 planned.

### Honest limits of what ships now

- **Hallucination detection** uses groundedness: the share of an answer's
  content words found in its retrieved context. It catches answers that
  ignore the context; it cannot judge whether a grounded answer is correct.
  An LLM-judge scorer is the next step.
- **Recommendations** are rules over your traces (expensive model with a cheap
  alternative, no cache, long outputs, rate limits, a dominant failure, low
  groundedness), not model-generated advice.
- **Prompt injection detection** is a pattern score that flags likely attacks
  for review; it is not a guarantee.
- **Cost** needs your prices (`observe.set_price` or `~/.langtrain/prices.json`).
  No vendor prices are built in, because stale defaults would report wrong costs.
- **Anomaly alerts** are detected on demand; sending alerts (email, Slack,
  webhook) needs the planned scheduler.

## Testing models (`langtrain.observe.testing`)

Test suites for any OpenAI-compatible model: Ollama on localhost by default,
or vLLM, Langtrain-served models, OpenAI, OpenRouter (`Target(model,
base_url=..., api_key=...)`), or a plain function (`FunctionTarget`). Every
call is traced, and each report is saved in `~/.langtrain/tests`.

```bash
langtrain-observe test eval --model llama3.1           # accuracy by category
langtrain-observe test benchmark --model llama3.1      # TTFT and tokens/sec
langtrain-observe test load --model llama3.1 --concurrency 8 --duration 30   # Pro
langtrain-observe test arena --model llama3.1 --against qwen3 --judge gpt-oss  # Pro
```

| Suite | What it measures | Plan |
|---|---|---|
| `benchmark` | time to first token, tokens/sec, latency for short/medium/long prompts | Free |
| `eval_suite` | accuracy on 46 questions: math, knowledge, coding, instructions, factuality traps | Free |
| `needle` | long-context retrieval across context lengths and depths | Free |
| `consistency` | agreement across repeated runs of one prompt | Free |
| `tool_calling` | right function, right arguments | Free |
| `leakage` | a planted secret against 10 exfiltration techniques (incl. base64) | Free |
| `pii_leakage` | a fake customer record (PAN, Aadhaar, account, ...) against the same attacks | Free |
| `red_team` | refusal rate on 24 prompts across 8 harm categories | Pro |
| `load_test` | throughput, latency percentiles and errors under concurrency | Pro |
| `arena` | blinded pairwise comparison with Elo; a person or a judge model votes | Pro |

These come from **langtrain_harness**, the earlier Next.js testing app for
Ollama. Ported with their datasets; the harness answer grader marked answers
ending in a full stop ("408.") as wrong, which is fixed here. Not ported yet,
because they are UI tools for the planned dashboard: the attention visualizer,
the hardware monitor, and the vision, embeddings, entity-matching and
capability testers.

## API (`/api/v1/observe`)

| Route | Plan | Purpose |
|---|---|---|
| `GET /capabilities` | any | capabilities and what the caller's plan includes |
| `POST /traces` | Pro | ingest up to 1,000 spans; retries are idempotent |
| `GET /traces`, `GET /traces/{id}` | Pro | search and open traces |
| `GET /metrics` | Pro | summary, tools and errors for a window |
| `POST /feedback` | Pro | rating for a trace |
| `GET /analysis/{models,cost-per-success,rate-limits,caching,rag,root-cause,recommendations,drift,anomalies}` | Pro | analyses on stored spans |
| `GET /analysis/sla` | Enterprise | latency and error objectives |

Data: `observe_spans` (one row per span, unique per organization and span id)
and `observe_feedback`; migration `20261022_observe_tables`.

## Roadmap for the planned capabilities

1. **Dashboard** in langtrain-web on `/api/v1/observe`: traces, trace tree,
   metrics, analyses, and locked cards with an upgrade prompt for Free plans.
2. **Alerts and continuous evaluation**: a scheduler that runs anomaly, drift,
   SLA and eval checks and notifies by email, Slack or webhook.
3. **Training and infrastructure**: fine-tuning observability and GPU monitoring
   from the training workers' telemetry; deployment monitoring, model versions
   and rollbacks for served endpoints.
4. **Routing** based on model comparison results; **human review** queue;
   **multimodal** spans; **synthetic and load testing**.
