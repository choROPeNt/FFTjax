"""
Thin wiring layer for the fracture problem: staggered mechanics + AT2
phase-field damage on a phase-labelled periodic voxel grid, for one
prescribed macroscopic strain (one time increment).

Scope of this first pass, matching what's actually built on the
materialmodels/ and solvers/ stacks so far:
- formulation="lippmann_schwinger" only, mirroring problems/mechanics.py.
- Amor driving-force split, AT2 degradation, hybrid irreversibility,
  homogeneous Gc -- the two single-notch-plate benchmarks' scheme
  (Schneider & Kaestner 2025, doi:10.1111/ffe.14553).
- Damage/history (d, H) are carried across increments by the *caller*
  (this function takes d_init/H_init and returns d/H, it doesn't own the
  time-stepping loop) -- same division of responsibility as solve_mechanics
  not owning load-stepping either.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple

import jax.numpy as jnp

from materialmodels.assembly import assemble_C_field
from materialmodels.phasefield.degradation import k_res_field
from materialmodels.phasefield.driving_force import lame_field
from operators.green import build_freq_grid, build_reference_green_operator
from solvers.coupling.staggered import StaggeredFractureSolution, solve_staggered_mechanics_phasefield


def solve_fracture(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    eps_bar:      jnp.ndarray,
    l0:           float,
    Gc:           float,
    d_init:       jnp.ndarray,
    H_init:       jnp.ndarray,
    formulation:  str = "lippmann_schwinger",
    scheme:       str = "rotated",
    toler_lin:    float = 1e-6,
    maxiter_cg:   int = 1000,
    toler_helm:   float = 1e-4,
    maxiter_helm: int = 300,
    eta:          float = 0.0,
    dt:           float = 1.0,
    d_thres:      float = 0.95,
    k_res:        float | jnp.ndarray | None = None,
    toler_st_abs: float = 1e-2,
    toler_st_rel: float = 1e-3,
    maxiter_st:   int = 200,
) -> StaggeredFractureSolution:
    """
    Solve one staggered mechanics<->phase-field increment under a
    prescribed macroscopic strain.

    Parameters
    ----------
    n, L        : grid shape and physical domain size
    phase       : (Nv,) int      phase index per voxel (0-based)
    materials   : list           each implements .stiffness_tensor(), .lam, .mu
                  (see materialmodels.elastic.isotropic.LinearElasticIsotropic)
    eps_bar     : (3, 3)         prescribed macroscopic strain for this increment
    l0, Gc      : phase-field length scale and critical energy release rate
    d_init      : (Nv,)          damage carried in from the previous increment
    H_init      : (Nv,)          history variable carried in from the previous increment
    formulation : "lippmann_schwinger" (only option implemented so far)
    scheme      : "standard" (GreenOperatorBasic) or "rotated" (GreenOperatorWillot)
    toler_lin, maxiter_cg     : mechanical CG tolerance / iteration cap
    toler_helm, maxiter_helm  : damage CG tolerance / iteration cap
    eta, dt                  : viscous regularisation of the damage equation
    d_thres                  : hybrid irreversibility threshold
    k_res                    : AT2 residual stiffness. None (default) gathers
                                each material's own .k_res per-phase (see
                                materialmodels.phasefield.degradation.k_res_field
                                -- e.g. LinearElasticIsotropic(..., k_res=1.0)
                                for a damage-immune phase). Pass a scalar or an
                                explicit (Nv,) array to override the per-phase
                                gather entirely (e.g. a smoothly-varying field).
    toler_st_abs, toler_st_rel, maxiter_st : staggered fixed-point convergence

    Returns
    -------
    StaggeredFractureSolution(eps, sigma, delta, d, H, converged_mech,
    converged_helm, converged_staggered, iter_staggered, err_abs, err_rel)
    """
    if formulation != "lippmann_schwinger":
        raise NotImplementedError(
            f"formulation={formulation!r} not implemented -- "
            "use formulation='lippmann_schwinger'"
        )

    C_field = assemble_C_field(materials, phase)
    lam_vox, mu_vox = lame_field(materials, phase)
    green_op = build_reference_green_operator(n, L, materials, scheme=scheme)
    xi_flat = build_freq_grid(n, L)
    k_res_arr = k_res_field(materials, phase) if k_res is None else k_res

    return solve_staggered_mechanics_phasefield(
        n, C_field, green_op, eps_bar,
        xi_flat, l0, Gc, lam_vox, mu_vox,
        d_init, H_init,
        toler_lin=toler_lin, maxiter_cg=maxiter_cg,
        toler_helm=toler_helm, maxiter_helm=maxiter_helm,
        eta=eta, dt=dt,
        d_thres=d_thres, k_res=k_res_arr,
        toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
    )
