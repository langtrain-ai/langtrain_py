"""
Automatic tracing for OpenAI-compatible clients: OpenAI, Langtrain-served
models, vLLM, Ollama, OpenRouter and anything else that speaks the same API.

    from openai import OpenAI
    from langtrain import observe

    observe.init()
    client = observe.wrap_openai(OpenAI())
    client.chat.completions.create(model="...", messages=[...])   # traced

Each call becomes an llm span with the model, messages, reply, token usage
and latency. Streaming calls are traced without token counts unless the
server sends usage in the final chunk.
"""

from __future__ import annotations

import functools
from typing import Any, Optional


def _usage(resp: Any) -> tuple:
    usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
    if usage is None:
        return None, None
    get = (lambda k: usage.get(k)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    return get("prompt_tokens") or get("input_tokens"), get("completion_tokens") or get("output_tokens")


def _reply(resp: Any) -> Optional[str]:
    try:
        choice = resp.choices[0] if not isinstance(resp, dict) else resp["choices"][0]
        msg = choice.message if not isinstance(choice, dict) else choice["message"]
        return msg.content if not isinstance(msg, dict) else msg.get("content")
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def wrap_openai(client: Any, tracer: Any = None, provider: str = "openai-compatible") -> Any:
    """Patch client.chat.completions.create (sync) to record a span per call. Returns the same client."""
    from . import _default_tracer

    completions = client.chat.completions
    original = completions.create
    if getattr(original, "_langtrain_wrapped", False):
        return client

    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        t = tracer or _default_tracer()
        model = kwargs.get("model")
        with t.span("chat.completions", kind="llm", model=model, provider=provider) as span:
            span.set(input=kwargs.get("messages"))
            for key in ("temperature", "max_tokens", "top_p"):
                if key in kwargs:
                    span.attributes[key] = kwargs[key]
            resp = original(*args, **kwargs)
            if kwargs.get("stream"):
                return resp
            inp, out = _usage(resp)
            span.record_usage(input_tokens=inp, output_tokens=out,
                              model=getattr(resp, "model", None) or (resp.get("model") if isinstance(resp, dict) else None) or model)
            span.set(output=_reply(resp))
            return resp

    create._langtrain_wrapped = True  # type: ignore[attr-defined]
    completions.create = create
    return client
