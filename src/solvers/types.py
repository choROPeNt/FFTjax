
from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"


from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp


class SolveState(NamedTuple):
    """
    Evolving state for one variational NW-CG elastic solve.

    A NamedTuple so it is automatically a JAX pytree — safe to pass
    through jax.jit, lax.while_loop, lax.scan, etc.

    Precomputed operators (G_glob, C_field) are intentionally excluded;
    they are static and should be passed as separate arguments to the solver.
    """

    # ── per-voxel fields ──────────────────────────────────────────────
    strain_loc:    jnp.ndarray  # (3, 3, Nv)      local strain  ε
    stress_loc:    jnp.ndarray  # (3, 3, Nv)      local stress  σ
    tangent_glob:  jnp.ndarray  # (3, 3, 3, 3, Nv) material tangent ∂σ/∂ε
                                #   equals C_field for linear elastic,
                                #   updated each Newton step for nonlinear

    # ── macroscopic averages ──────────────────────────────────────────
    strain_ave:    jnp.ndarray  # (3, 3)   〈ε〉
    stress_ave:    jnp.ndarray  # (3, 3)   〈σ〉

    # ── current increment bookkeeping ─────────────────────────────────
    strain_ave_inc_goal:  jnp.ndarray  # (3, 3)      Δ〈ε〉 per time increment
    stress_ave_inc_goal:  jnp.ndarray  # (3, 3)      Δ〈σ〉 per time increment
    Deltastrain_loc:      jnp.ndarray  # (3, 3, Nv)  initial trial ε increment
    Deltastress_loc:      jnp.ndarray  # (3, 3, Nv)  initial trial σ target
    stress_loc_goal:      jnp.ndarray  # (3, 3, Nv)  target σ for current increment

    # ── Newton correction (output of the inner CG solve) ──────────────
    deltastrain_loc:  jnp.ndarray  # (3, 3, Nv)  Δε from last CG solve

    # ── scalar state ──────────────────────────────────────────────────
    time:     float  # current pseudo-time
    dtime:    float  # current time step size
    kinc:     int    # increment counter within the current step
    kstep:    int    # load-step index (1-based, matches rsteps)
    iter_nw:  int    # Newton iterations consumed in current increment
    iter_cg:  int    # CG iterations consumed in last inner solve
    info:     int    # CG exit flag — 0 = converged, nonzero = failure
    bb0n:     float  # ‖r₀‖ at Newton iter 0; used for adaptive CG tolerance
                     #   tol = max(toler_lin * bb0n / ‖rₙ‖, toler_lin)
    pnewdt:   float  # time-step reduction request; 1.0 = no reduction


@dataclass
class SolverSettings:
    """
    Static configuration for the variational NW-CG elastic solver.

    Passed once at setup; nothing here changes during iteration.
    Load steps are accumulated by calling add_load_step().
    """

    # ── domain ────────────────────────────────────────────────────────
    ndim:  int
    n:     tuple[int, ...]    # voxel counts per dimension, e.g. (nx, ny, nz)
    L:     tuple[float, ...]  # physical domain lengths per dimension

    # ── tolerances ────────────────────────────────────────────────────
    toler_lin: float = 1e-7   # CG relative residual tolerance
    toler_nw:  float = 1e-7   # Newton convergence: |Δε|∞ / |〈ε〉|

    # ── iteration caps ────────────────────────────────────────────────
    maxiter_cg: int = 1000
    maxiter_nw: int = 6

    # ── load steps ────────────────────────────────────────────────────
    # control[i]         : (3, 3) int mask — 1 = stress-controlled, 0 = strain
    # strain_ave_goal[i] : (3, 3) target macroscopic strain for step i
    # stress_ave_goal[i] : (3, 3) target macroscopic stress for step i
    # timer[i]           : (dt_init, t_end, dt_min, dt_max)
    control:          list[jnp.ndarray]               = field(default_factory=list)
    strain_ave_goal:  list[jnp.ndarray]               = field(default_factory=list)
    stress_ave_goal:  list[jnp.ndarray]               = field(default_factory=list)
    timer:            list[tuple[float, float, float, float]] = field(default_factory=list)

    # ── output ────────────────────────────────────────────────────────
    jobname: str = ""
    output:  str = ""

    # ── derived ───────────────────────────────────────────────────────
    @property
    def nvoxels(self) -> int:
        from math import prod
        return prod(self.n)

    @property
    def nsteps(self) -> int:
        return len(self.timer)

    @property
    def rsteps(self) -> range:
        """1-based step range matching the reference solver convention."""
        return range(1, self.nsteps + 1)

    def add_load_step(
        self,
        control:         jnp.ndarray,
        strain_ave_goal: jnp.ndarray,
        stress_ave_goal: jnp.ndarray,
        timer:           tuple[float, float, float, float],
    ) -> None:
        """Append one load step."""
        self.control.append(control)
        self.strain_ave_goal.append(strain_ave_goal)
        self.stress_ave_goal.append(stress_ave_goal)
        self.timer.append(timer)
