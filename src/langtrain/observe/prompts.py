"""
Versioned prompt templates, stored in ~/.langtrain/prompts.json.

    from langtrain.observe import prompts

    v = prompts.save("support-reply", "You are a support agent for {company}. Answer: {question}")
    text = prompts.render("support-reply", company="Acme", question="Where is my order?")
    old = prompts.render("support-reply", 1, company="Acme", question="...")

Saving identical text returns the existing version instead of a new one.
Record the version on spans (prompt=name, prompt_version=v) to compare
versions in traces.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

_lock = threading.Lock()


def _path() -> str:
    return os.environ.get("LANGTRAIN_PROMPTS_FILE") or os.path.join(os.path.expanduser("~"), ".langtrain", "prompts.json")


def _load() -> Dict[str, List[Dict[str, Any]]]:
    try:
        with open(_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _store(data: Dict[str, List[Dict[str, Any]]]) -> None:
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save(name: str, template: str, note: str = "") -> int:
    """Store a new version of `name` and return its number (1, 2, ...)."""
    digest = hashlib.sha256(template.encode()).hexdigest()[:12]
    with _lock:
        data = _load()
        versions = data.setdefault(name, [])
        for v in versions:
            if v["hash"] == digest:
                return v["version"]
        version = len(versions) + 1
        versions.append({"version": version, "hash": digest, "template": template, "note": note, "created_at": time.time()})
        _store(data)
    return version


def get(name: str, version: Optional[int] = None) -> Dict[str, Any]:
    versions = _load().get(name)
    if not versions:
        raise KeyError(f"No prompt called {name!r}")
    if version is None:
        return versions[-1]
    for v in versions:
        if v["version"] == version:
            return v
    raise KeyError(f"{name!r} has no version {version}")


def render(prompt: str, version: Optional[int] = None, /, **variables: Any) -> str:
    """Fill a template. The prompt name and version are positional, so any variable name works."""
    return get(prompt, version)["template"].format(**variables)


def list_prompts() -> Dict[str, int]:
    """Each prompt name with its latest version number."""
    return {name: versions[-1]["version"] for name, versions in _load().items() if versions}
