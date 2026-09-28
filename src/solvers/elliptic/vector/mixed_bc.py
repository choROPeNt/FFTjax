"""
Shared mixed strain/stress macroscopic-BC infrastructure for solvers/elliptic/
vector/ -- the ``control``/``sv``/``sm`` convention (see ``_active_pairs``'s
docstring) was previously duplicated verbatim between displacement_based.py
and problems/mechanics.py's ``solve_displacement_based_nonlinear``; this
module is the one place it lives now, plus the DC-bin ("zero frequency")
mixed-BC scheme shared by solve_lippmann_schwinger's/solve_fourier_galerkin's
mixed-BC entry points (Lucarini & Segurado 2019a; Kabel et al 2016; see
notes/controll.md for the derivation this implements).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from math import prod
from typing import Protocol, Tuple, runtime_checkable

import jax.numpy as jnp

from operators.base import LinearOperator
from operators.general_functions import ddot42
from operators.projection import Gamma0Operator
from solvers.krylov.cg import cg_solve

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))


@runtime_checkable
class _HasG(Protocol):
    """Structural type for a LinearOperator with a cached, DC-zero 4th-order
    tensor ``.G`` (3, 3, 3, 3, Nv) -- GreenOperatorBasic/Willot
    (operators.green) and GalerkinProjector (operators.galerkin) both
    satisfy this (both are also LinearOperator subclasses, just via
    structural rather than nominal typing here -- a Protocol can't inherit
    from a plain ABC). The generic LinearOperator ABC doesn't declare ``.G``
    at all (not every LinearOperator caches a dense tensor -- e.g.
    operators.base's _Composed doesn't), so a plain LinearOperator type hint
    would go unchecked at the one place (patch_dc_identity's
    ``elastic_op.G`` access) that actually needs it to exist."""
    G: jnp.ndarray


def _active_pairs(control: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, int], ...]:
    """Upper-triangular (i, j) index pairs, i<=j, where control[i][j] == 1."""
    return tuple(
        (i, j)
        for i in range(3)
        for j in range(i, 3)
        if control[i][j]
    )


def sv2sm(sv: jnp.ndarray, pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """(K,) stacked active-pair values -> (3, 3) symmetric matrix, zero off ``pairs``."""
    sm = jnp.zeros((3, 3), dtype=sv.dtype)
    for k, (i, j) in enumerate(pairs):
        sm = sm.at[i, j].set(sv[k])
        sm = sm.at[j, i].set(sv[k])
    return sm


def sm2sv(sm: jnp.ndarray, pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """(3, 3) -> (K,) stacked values at ``pairs`` only; K=0 (empty) if pairs is empty."""
    if not pairs:
        return jnp.zeros((0,), dtype=sm.dtype)
    return jnp.stack([sm[i, j] for i, j in pairs])


def _isotropic_stiffness(lam, mu) -> jnp.ndarray:
    """(3,3,3,3) isotropic stiffness tensor from Lame parameters -- same formula
    as materialmodels.elastic.isotropic.LinearElasticIsotropic.elastic_stiffness_tensor,
    duplicated here (rather than imported) to keep solvers/ free of a
    materialmodels/ dependency; used only to build the small reference-stiffness
    system for the Lippmann-Schwinger outer mixed-BC correction."""
    d = jnp.eye(3)
    return (lam * jnp.einsum('ij,kl->ijkl', d, d)
            + mu * (jnp.einsum('ik,jl->ijkl', d, d)
                    + jnp.einsum('il,jk->ijkl', d, d)))


def patch_dc_identity(G: jnp.ndarray, control: Tuple[Tuple[int, ...], ...]) -> jnp.ndarray:
    """
    Replace a reference-medium/projector tensor's value at the zero frequency
    (DC bin, always flat voxel index 0 for any grid operators.green.
    build_freq_grid produces -- fftfreq's own convention plus row-major
    meshgrid('ij') raveling) with a control-masked symmetric identity, leaving
    every other frequency (including Nyquist) untouched.

    ``G`` must already be exactly zero at the DC bin (true of both
    operators.green.build_green_operator and operators.galerkin.
    build_galerkin_projection_operator, via their shared
    ``jnp.where(safe[...], G, 0.0)`` construction) -- this function does not
    check that, it simply overwrites index 0.

    G*_ijkl(DC) = control[i][j] * 0.5*(delta_ik delta_jl + delta_il delta_jk)

    i.e. the identity (on symmetric tensors) for stress-controlled (i, j)
    pairs -- letting that component of a target stress pass straight through
    the projector unprojected, the mechanism that turns it into a genuine
    solved-for macroscopic degree of freedom -- and 0 for strain-controlled
    pairs, reproducing today's "eps_bar fixed, zero-mean correction only"
    behaviour exactly when ``control`` is all-zero (see
    solve_mixed_bc_dc_identity's docstring for why this degenerates bit-for-
    bit to solve_lippmann_schwinger's own algebra in that case).

    Major symmetry (G*_ijkl == G*_klij) is preserved automatically: control
    is a symmetric (3,3) mask by convention, and the identity tensor above is
    itself invariant under swapping (i,j)<->(k,l), so wherever it is nonzero
    control[i][j] == control[k][l] already.

    Parameters
    ----------
    G       : (3, 3, 3, 3, Nv)
    control : (3, 3) 0/1 mask, 1 = stress-controlled, 0 = strain-controlled

    Returns
    -------
    G_patched : (3, 3, 3, 3, Nv)
    """
    control_arr = jnp.asarray(control, dtype=G.dtype)
    d = jnp.eye(3, dtype=G.dtype)
    isym = 0.5 * (jnp.einsum('ik,jl->ijkl', d, d) + jnp.einsum('il,jk->ijkl', d, d))
    dc_patch = control_arr[:, :, None, None] * isym
    return G.at[:, :, :, :, 0].set(dc_patch)


class _PatchedOperator(LinearOperator):
    """Thin LinearOperator around an already DC-patched tensor -- same calling
    convention as GreenOperatorBasic/GalerkinProjector (``ddot42(self.G, x)``).
    Self-adjoint: patch_dc_identity preserves major symmetry (see its own
    docstring), which is what makes ``.T`` -> self valid here exactly as it is
    for the unpatched operators."""

    def __init__(self, G: jnp.ndarray):
        self.G = G

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return ddot42(self.G, x)

    @property
    def T(self) -> LinearOperator:
        return self


def solve_mixed_bc_dc_identity(
    n:                Tuple[int, ...],
    C_field:          jnp.ndarray,
    elastic_op:       _HasG,
    control:          Tuple[Tuple[int, ...], ...],
    eps_bar:          jnp.ndarray,
    macro_stress_goal: jnp.ndarray,
    toler_lin:        float = 1e-4,
    maxiter:          int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single-CG-solve mixed strain/stress macroscopic BC via the DC-bin identity
    patch (patch_dc_identity): Kabel et al (2016) when ``elastic_op`` is a
    GreenOperatorBasic/Willot (reference-medium Lippmann-Schwinger), Lucarini &
    Segurado (2019a) when it's a GalerkinProjector (no reference medium) --
    the same code either way, only which operator is passed in differs. See
    notes/controll.md for the derivation.

    Solves G*(C:eps - macro_stress_goal_bcast) = 0 for the FULL strain field
    directly (not a correction around a fixed uniform baseline the way
    solve_lippmann_schwinger's pure-strain path does it): strain-controlled
    directions are still pinned via ``eps0``, stress-controlled directions are
    solved for jointly with the fluctuation, through the same single CG system.

    Degenerates bit-for-bit to solve_lippmann_schwinger's own eps0/bb/eps/
    sigma algebra when ``control`` is all-zero: patch_dc_identity is then a
    strict no-op (control-masked identity is zero everywhere), so
    Gamma0Operator(n, _PatchedOperator(elastic_op.G)) computes exactly what
    Gamma0Operator(n, elastic_op) would.

    Parameters
    ----------
    n           : grid shape (nx, ny, nz)
    C_field     : (3, 3, 3, 3, Nv)  per-voxel stiffness (or tangent)
    elastic_op  : LinearOperator with a cached (3,3,3,3,Nv) ``.G`` attribute,
                  zero at DC -- GreenOperatorBasic/Willot or GalerkinProjector
    control     : (3, 3) 0/1 mask, 1 = stress-controlled, 0 = strain-controlled
    eps_bar     : (3, 3)  prescribed macroscopic strain; entries where
                  ``control == 1`` are ignored (solved for instead)
    macro_stress_goal : (3, 3)  target macroscopic stress; only entries where
                  ``control == 1`` are used
    toler_lin, maxiter : CG tolerance / iteration cap

    Returns
    -------
    eps        : (3, 3, Nv)
    sigma      : (3, 3, Nv)
    delta      : (3, 3, Nv)   strain correction from the last CG solve
    eps_bar_out: (3, 3)       macroscopic strain, stress-controlled entries filled in
    converged  : bool array
    """
    Nv = prod(n)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

    patched_op = _PatchedOperator(patch_dc_identity(elastic_op.G, control))
    gamma0 = Gamma0Operator(n, patched_op)

    def A_op(v_flat):
        v = v_flat.reshape(3, 3, Nv)
        Cv = ddot42(C_field, v)
        return gamma0(Cv).reshape(-1)

    eps0 = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
    sg = jnp.ones((3, 3, Nv)) * macro_stress_goal[:, :, None]
    sigma0 = ddot42(C_field, eps0)
    bb = -gamma0(sigma0 - sg).reshape(-1)

    x0 = jnp.zeros_like(bb)
    delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)

    delta = delta_flat.reshape(3, 3, Nv)
    eps = eps0 + delta
    sigma = ddot42(C_field, eps)
    eps_bar_out = jnp.mean(eps, axis=-1)

    return eps, sigma, delta, eps_bar_out, converged
