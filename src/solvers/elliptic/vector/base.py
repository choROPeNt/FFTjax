"""
Abstract elasticity-solver interface for solvers/elliptic/vector/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax.numpy as jnp


class ElasticitySolution(NamedTuple):
    """
    Result of one elasticity solve. A NamedTuple (not a dataclass) so it is
    automatically a JAX pytree -- safe to pass through jax.jit/grad/jvp, same
    convention as solvers.types.SolveState (see that class's docstring for
    why: a plain dataclass isn't a pytree without extra registration, which
    would silently break autodiff through the solve).
    """

    eps:       jnp.ndarray  # (3, 3, Nv)  local strain
    sigma:     jnp.ndarray  # (3, 3, Nv)  local stress
    delta:     jnp.ndarray  # (3, 3, Nv)  strain correction from the CG solve
    converged: jnp.ndarray  # bool array
    eps_bar:   jnp.ndarray | None = None  # (3, 3) macroscopic strain with any
                                           # stress-controlled entries filled in
                                           # by the solve -- None (default) for
                                           # solvers with no mixed-BC concept
                                           # (e.g. LippmannSchwingerSolver);
                                           # populated by DisplacementBasedSolver


class ElasticitySolver(ABC):
    """
    Common interface for elasticity solvers (Lippmann-Schwinger,
    displacement-based, ...). Formulation-specific setup (reference-medium
    Green's operator, boundary conditions, preconditioner, ...) belongs in
    each concrete solver's ``__init__``; ``solve`` takes only what's common
    to every formulation.
    """

    @abstractmethod
    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        """
        ``stress_goal``'s shape and meaning are formulation-specific -- see
        each concrete solver: ``LippmannSchwingerSolver`` takes a per-voxel
        ``(3, 3, Nv)`` target stress field (None = zero, pure strain BC);
        ``DisplacementBasedSolver`` takes a macroscopic ``(3, 3)`` target,
        used only on the entries its ``control`` marks stress-controlled.
        """
        ...
