"""
YAML configuration loader with {variable} interpolation.

Supports referencing other keys and Path attributes:

    input:   output/preprocessed/rve.xdmf
    jobname: "{input.stem}"          # → "rve"
    output:  "output/simulation/{jobname}"   # → "output/simulation/rve"

Resolution rules
----------------
- {key}          replace with the value of ``key`` in the same config
- {key.attr}     resolve ``key`` first, wrap in Path, apply ``.attr``
                 (stem, name, parent, suffix, …)
- {a.b.c}        walk nested dicts  a → b → c
- Chained references are resolved in up to 5 passes, so
  {c} where c = "{b}" where b = "{a}" resolves correctly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r'\{([^}]+)\}')


def _resolve_expr(expr: str, ctx: dict) -> str:
    """
    Resolve a single {expr} token against ctx.

    Supports:
      key           → ctx[key]
      key.attr      → getattr(Path(ctx[key]), attr)
      a.b.c         → ctx[a][b][c]  (nested dict walk)
    """
    parts = expr.split(".")
    val: Any = ctx

    for part in parts:
        if isinstance(val, dict):
            if part not in val:
                return "{" + expr + "}"   # leave unresolved
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


def load_config(path: str | Path) -> dict:
    """
    Load a YAML file and resolve all {variable} references.

    Parameters
    ----------
    path : path to the .yaml / .yml file

    Returns
    -------
    dict with all string values interpolated
    """
    with open(path) as f:
        cfg: dict = yaml.safe_load(f)

    # Up to 5 passes — handles chains like {c} → {b} → {a} → literal
    for _ in range(5):
        cfg = _walk(cfg, cfg)   # type: ignore[assignment]

    return cfg
