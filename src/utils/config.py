"""
YAML config loader with ``{variable}`` interpolation:

    input:   output/preprocessed/rve.xdmf
    jobname: "{input.stem}"                   # -> "rve"
    output:  "output/simulation/{jobname}"    # -> "output/simulation/rve"

``{key}`` is another value, ``{key.attr}`` a Path attribute (stem, name,
parent, ...), ``{a.b.c}`` a nested key; chains resolve in up to 5 passes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r'\{([^}]+)\}')


def _resolve_expr(expr: str, ctx: dict) -> str:
    """One ``{expr}`` token: ``key``, ``key.attr`` (Path attribute) or ``a.b.c``; unresolved stays as is."""
    parts = expr.split(".")
    val: Any = ctx

    for part in parts:
        if isinstance(val, dict):
            if part not in val:
                return "{" + expr + "}"
            val = val[part]
        elif isinstance(val, (str, Path)):
            p = Path(str(val))
            if not hasattr(p, part):
                return "{" + expr + "}"
            val = getattr(p, part)
        else:
            if not hasattr(val, part):
                return "{" + expr + "}"
            val = getattr(val, part)

    return str(val)


def _resolve_str(s: str, ctx: dict) -> str:
    return _VAR.sub(lambda m: _resolve_expr(m.group(1), ctx), s)


def _walk(node: Any, ctx: dict) -> Any:
    if isinstance(node, str):
        return _resolve_str(node, ctx)
    if isinstance(node, dict):
        return {k: _walk(v, ctx) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, ctx) for v in node]
    return node


def field_write_mode(raw: Any) -> str:
    """
    ``write_fields`` value -> "all", "increments" (skips the step-0 zero field)
    or "none"; booleans accepted. Field output dominates large runs (~0.4 GB
    per increment at 152^3).
    """
    if isinstance(raw, bool):
        return "all" if raw else "none"
    val = str(raw).strip().lower()
    if val in ("true", "all"):
        return "all"
    if val in ("false", "none", "off"):
        return "none"
    if val == "increments":
        return "increments"
    raise ValueError(f"write_fields: expected true/false/increments, got {raw!r}")


def load_config(path: str | Path) -> dict:
    """YAML config with every ``{variable}`` reference resolved."""
    with open(path) as f:
        cfg: dict = yaml.safe_load(f)

    for _ in range(5):
        cfg = _walk(cfg, cfg)   # type: ignore[assignment]

    return cfg
