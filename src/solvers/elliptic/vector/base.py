"""
Abstract elasticity-solver interface for solvers/elliptic/vector/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp

from solvers.solution import ElasticitySolution


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
        each concrete solver: ``LippmannSchwingerSolver`` (pure strain BC
        only, no ``control``) takes a per-voxel ``(3, 3, Nv)`` target stress
        field (None = zero); ``DisplacementBasedSolver``,
        ``FourierGalerkinSolver``, and ``LippmannSchwingerMixedBCSolver``
        take a macroscopic ``(3, 3)`` target, used only on the entries their
        ``control`` marks stress-controlled.
        """
        ...
