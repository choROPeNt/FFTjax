"""State and settings of the variational Newton-CG elastic solver."""

from __future__ import annotations

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp


class SolveState(NamedTuple):
    """
    Evolving state of one Newton-CG elastic solve (a pytree). Static operators
    (G, C_field) are passed separately.

    Attributes
    ----------
    strain_loc, stress_loc : (3, 3, Nv) local strain and stress
    tangent_glob           : (3, 3, 3, 3, Nv) tangent d sigma / d eps
    strain_ave, stress_ave : (3, 3) macroscopic averages
    strain_ave_inc_goal, stress_ave_inc_goal : (3, 3) increment targets
    Deltastrain_loc        : (3, 3, Nv) trial strain increment
    Deltastress_loc        : (3, 3, Nv) trial stress target
    stress_loc_goal        : (3, 3, Nv) stress target of the increment
    deltastrain_loc        : (3, 3, Nv) last CG correction
    time, dtime            : pseudo-time and step size
    kinc, kstep            : increment counter and 1-based load step
    iter_nw, iter_cg       : Newton and last CG iteration counts
    info                   : CG exit flag, 0 = converged
    bb0n                   : initial residual norm, for the adaptive CG tolerance
    pnewdt                 : step-size reduction request, 1.0 = none
    """

    strain_loc:    jnp.ndarray
    stress_loc:    jnp.ndarray
    tangent_glob:  jnp.ndarray

    strain_ave:    jnp.ndarray
    stress_ave:    jnp.ndarray

    strain_ave_inc_goal:  jnp.ndarray
    stress_ave_inc_goal:  jnp.ndarray
    Deltastrain_loc:      jnp.ndarray
    Deltastress_loc:      jnp.ndarray
    stress_loc_goal:      jnp.ndarray

    deltastrain_loc:  jnp.ndarray

    time:     float
    dtime:    float
    kinc:     int
    kstep:    int
    iter_nw:  int
    iter_cg:  int
    info:     int
    bb0n:     float
    pnewdt:   float


@dataclass
class SolverSettings:
    """
    Static settings of the Newton-CG elastic solver; add load steps with ``add_load_step``.

    Per load step i: ``control[i]`` (3, 3) mask (1 = stress-controlled),
    ``strain_ave_goal[i]``/``stress_ave_goal[i]`` (3, 3) targets and
    ``timer[i]`` = (dt_init, t_end, dt_min, dt_max).
    """

    ndim:  int
    n:     tuple[int, ...]
    L:     tuple[float, ...]

    toler_lin: float = 1e-7
    toler_nw:  float = 1e-7

    maxiter_cg: int = 1000
    maxiter_nw: int = 6

    control:          list[jnp.ndarray]               = field(default_factory=list)
    strain_ave_goal:  list[jnp.ndarray]               = field(default_factory=list)
    stress_ave_goal:  list[jnp.ndarray]               = field(default_factory=list)
    timer:            list[tuple[float, float, float, float]] = field(default_factory=list)

    jobname: str = ""
    output:  str = ""

    @property
    def nvoxels(self) -> int:
        from math import prod
        return prod(self.n)

    @property
    def nsteps(self) -> int:
        return len(self.timer)

    @property
    def rsteps(self) -> range:
        """1-based step range (reference-solver convention)."""
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
