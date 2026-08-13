"""
Thin wiring layer for the mechanical (elasticity) problem: pick a
formulation, build C(x), pick a reference medium, solve, return the fields.

Scope of this first pass, matching what's actually built on the operators/
and solvers/ stacks so far:
- formulation="lippmann_schwinger" only; "displacement" raises
  NotImplementedError -- solvers/elliptic/vector/displacement_based.py
  doesn't exist yet.
- Reference-medium averaging is a plain arithmetic mean of the materials'
  Lame parameters (matches what notebooks/lin-elastic_strain.ipynb already
  does by hand). This is a placeholder for materialmodels/averaging.py's
  VoxelAveraging ABC, which doesn't exist yet either -- swap it in here
  once built, this function's signature shouldn't need to change.
- Builds a LippmannSchwingerSolver (ElasticitySolver) and calls .solve() --
  not the plain solve_lippmann_schwinger function -- so the ABC actually
  gets exercised by the one real caller it exists to serve, instead of
  sitting unused until displacement_based.py shows up.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple

import jax.numpy as jnp

from mat_models.elastic import assemble_C_field
from operators.green import GreenOperatorBasic, GreenOperatorWillot
from solvers.elliptic.vector.base import ElasticitySolution
from solvers.elliptic.vector.lippmann_schwinger import LippmannSchwingerSolver


def solve_mechanics(
    n:           Tuple[int, ...],
    L:           Tuple[float, ...],
    phase:       jnp.ndarray,
    materials:   list,
    eps_bar:     jnp.ndarray,
    formulation: str = "lippmann_schwinger",
    scheme:      str = "rotated",
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-6,
    maxiter:     int = 1000,
) -> ElasticitySolution:
    """
    Solve the mechanical equilibrium problem on a phase-labelled periodic
    voxel grid under a prescribed macroscopic strain.

    Parameters
    ----------
    n, L        : grid shape and physical domain size
    phase       : (Nv,) int      phase index per voxel (0-based)
    materials   : list           each implements .stiffness_tensor(), .lam, .mu
                  (see mat_models.elastic.LinearElasticIsotropic)
    eps_bar     : (3, 3)         prescribed macroscopic strain
    formulation : "lippmann_schwinger" (only option implemented so far)
    scheme      : "standard" (GreenOperatorBasic) or "rotated" (GreenOperatorWillot)
    stress_goal : (3, 3, Nv) or None, passed through to the solver
    toler_lin, maxiter : CG tolerance / iteration cap

    Returns
    -------
    ElasticitySolution(eps, sigma, delta, converged) -- a NamedTuple, so it
    still unpacks as a plain 4-tuple for existing callers.
    """
    C_field = assemble_C_field(materials, phase)

    # Reference medium: arithmetic mean of the materials' Lame parameters --
    # see module docstring for why this isn't materialmodels/averaging.py yet.
    lam0 = sum(m.lam for m in materials) / len(materials)
    mu0 = sum(m.mu for m in materials) / len(materials)

    if scheme == "standard":
        green_op = GreenOperatorBasic(n, L, lam0, mu0)
    elif scheme == "rotated":
        dx = tuple(Li / ni for Li, ni in zip(L, n))
        green_op = GreenOperatorWillot(n, L, lam0, mu0, dx)
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected 'standard' or 'rotated'")

    if formulation == "lippmann_schwinger":
        solver = LippmannSchwingerSolver(n, green_op, toler_lin, maxiter)
        return solver.solve(C_field, eps_bar, stress_goal)
    elif formulation == "displacement":
        raise NotImplementedError(
            "formulation='displacement' needs solvers/elliptic/vector/displacement_based.py, "
            "which doesn't exist yet -- use formulation='lippmann_schwinger'"
        )
    else:
        raise ValueError(
            f"unknown formulation {formulation!r}, expected 'lippmann_schwinger' or 'displacement'"
        )
