"""
Cost per call, from token counts and a price book you control.

No vendor prices are built in: they change often and a stale default would
report wrong costs. Set the prices you pay with set_price(), or put them in
~/.langtrain/prices.json as {"model": [input_per_1m, output_per_1m]}.
Self-hosted models default to no cost; add a price to account for GPU time.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

_prices: Dict[str, Tuple[float, float]] = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    path = os.path.join(os.path.expanduser("~"), ".langtrain", "prices.json")
    try:
        with open(path) as f:
            for model, (inp, out) in json.load(f).items():
                _prices.setdefault(model, (float(inp), float(out)))
    except (OSError, ValueError, TypeError):
        pass


def set_price(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Price for `model` in your currency per 1M input and output tokens."""
    _prices[model] = (float(input_per_1m), float(output_per_1m))


def cost(model: Optional[str], input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Cost of one call, or None when the model has no price."""
    _load()
    if not model or model not in _prices:
        return None
    inp, out = _prices[model]
    return round(((input_tokens or 0) * inp + (output_tokens or 0) * out) / 1_000_000, 8)
