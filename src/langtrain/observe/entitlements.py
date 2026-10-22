"""
Which observability capabilities the current account's plan includes.

The plan comes from the Langtrain API (LangtrainClient.me()), cached for an
hour in ~/.langtrain/entitlements.json. With no API key, or offline, the
account is treated as Free. These checks explain what a plan includes; the
server enforces it for everything that runs in Langtrain's cloud.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import capabilities

PRICING_URL = "https://langtrain.xyz/pricing"
_CACHE_TTL = 3600
_override: Optional["Entitlements"] = None


class ProFeatureError(PermissionError):
    """Raised when a capability needs a higher plan than the account has."""

    def __init__(self, capability: capabilities.Capability, plan: str):
        self.capability = capability
        self.plan = plan
        need = "Enterprise" if capability.tier == "enterprise" else "Pro"
        super().__init__(
            f"{capability.name} is a Langtrain {need} feature (your plan: {plan.title()}). "
            f"Upgrade at {PRICING_URL}"
        )


@dataclass
class Entitlements:
    plan: str = "free"
    features: Dict[str, Any] = field(default_factory=dict)

    def allows(self, capability_id: str) -> bool:
        cap = capabilities.get(capability_id)
        if cap.status == "planned":
            return False
        if self.features.get(f"observe.{cap.id}") is True:
            return True
        return capabilities.tier_allows(self.plan, cap.tier)

    def require(self, capability_id: str) -> None:
        cap = capabilities.get(capability_id)
        if cap.status == "planned":
            raise NotImplementedError(f"{cap.name} is on the roadmap and not available yet.")
        if not self.allows(capability_id):
            raise ProFeatureError(cap, self.plan)


def _cache_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".langtrain", "entitlements.json")


def _read_cache(key_hint: str) -> Optional[Entitlements]:
    try:
        with open(_cache_path()) as f:
            data = json.load(f)
        if data.get("key") == key_hint and time.time() - data.get("at", 0) < _CACHE_TTL:
            return Entitlements(plan=data.get("plan", "free"), features=data.get("features") or {})
    except (OSError, ValueError):
        pass
    return None


def _write_cache(key_hint: str, ent: Entitlements) -> None:
    try:
        path = _cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"key": key_hint, "at": time.time(), "plan": ent.plan, "features": ent.features}, f)
    except OSError:
        pass


def current(client: Any = None) -> Entitlements:
    """The plan of the account behind LANGTRAIN_API_KEY (or `client`); Free when unknown."""
    if _override is not None:
        return _override
    api_key = getattr(client, "api_key", None) or os.environ.get("LANGTRAIN_API_KEY")
    if not api_key:
        return Entitlements()
    key_hint = api_key[-6:]  # identifies the cache entry without storing the key
    cached = _read_cache(key_hint)
    if cached:
        return cached
    try:
        if client is None:
            from langtrain.client import LangtrainClient
            client = LangtrainClient(api_key=api_key)
        account = client.me()
        ent = Entitlements(plan=account.get("plan") or "free", features=account.get("features") or {})
    except Exception:
        return Entitlements()  # offline or invalid key: Free, not cached
    _write_cache(key_hint, ent)
    return ent


def require(capability_id: str, client: Any = None) -> None:
    current(client).require(capability_id)


def set_entitlements(ent: Optional[Entitlements]) -> None:
    """Use `ent` instead of asking the API (tests, air-gapped installs). None restores lookup."""
    global _override
    _override = ent
