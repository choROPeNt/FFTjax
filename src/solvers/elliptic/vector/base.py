"""Abstract elasticity-solver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp

from solvers.solution import ElasticitySolution


class ElasticitySolver(ABC):
    """Elasticity solver; formulation-specific setup goes in ``__init__``."""

    @abstractmethod
    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        """
        Solve for one macroscopic load.

        ``stress_goal`` differs by solver: (3, 3, Nv) field for
        LippmannSchwingerSolver, (3, 3) macroscopic target (stress-controlled
        entries only) for the mixed-BC solvers.
        """
        ...
