"""
Mixed strain/stress macroscopic BC: the ``control``/``sv``/``sm`` helpers and
the DC-bin identity scheme (Kabel et al 2016; Lucarini & Segurado 2019a;
derivation in notes/controll.md).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from math import prod
from typing import Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from operators.base import LinearOperator
from operators.fft_distributed import (
    choose_device_count, pfft3d_flat, pifft3d_flat, shard_x_slabs, x_slab_shard_map, x_slab_spec,
)
from operators.general_functions import ddot42
from operators.projection import Gamma0Operator
from solvers.krylov.cg import cg_solve, cg_solve_pmap

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))


@runtime_checkable
class _HasG(Protocol):
    """Operator with a cached, DC-zero ``.G`` (3, 3, 3, 3, Nv): GreenOperatorBasic/
    Willot or GalerkinProjector. A Protocol because LinearOperator itself has no ``.G``."""
    G: jnp.ndarray


def _active_pairs(control: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, int], ...]:
    """Stress-controlled pairs (i, j), i <= j, where control[i][j] == 1."""
    return tuple(
        (i, j)
        for i in range(3)
        for j in range(i, 3)
        if control[i][j]
    )


def sv2sm(sv: jnp.ndarray, pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """(K,) values at ``pairs`` -> symmetric (3, 3), zero elsewhere."""
    sm = jnp.zeros((3, 3), dtype=sv.dtype)
    for k, (i, j) in enumerate(pairs):
        sm = sm.at[i, j].set(sv[k])
        sm = sm.at[j, i].set(sv[k])
    return sm


def sm2sv(sm: jnp.ndarray, pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """
    (3, 3) -> (K,) values at ``pairs``: sm[i, i] on the diagonal, sm[i, j] + sm[j, i] off it.

    The adjoint of sv2sm, not its inverse: sv2sm writes a shear unknown into
    both (i, j) and (j, i), so its conjugate stress is the sum. Taking
    sm[i, j] alone makes the bordered displacement operator non-symmetric,
    which CG does not tolerate.
    """
    if not pairs:
        return jnp.zeros((0,), dtype=sm.dtype)
    return jnp.stack([sm[i, j] if i == j else sm[i, j] + sm[j, i] for i, j in pairs])


def _isotropic_stiffness(lam, mu) -> jnp.ndarray:
    """(3, 3, 3, 3) isotropic stiffness from Lame parameters.

    Duplicates LinearElasticIsotropic.elastic_stiffness_tensor on purpose:
    solvers/ must not depend on materialmodels/.
    """
    d = jnp.eye(3)
    return (lam * jnp.einsum('ij,kl->ijkl', d, d)
            + mu * (jnp.einsum('ik,jl->ijkl', d, d)
                    + jnp.einsum('il,jk->ijkl', d, d)))


def patch_dc_identity(G: jnp.ndarray, control: Tuple[Tuple[int, ...], ...]) -> jnp.ndarray:
    """
    Set the DC bin (flat index 0) of ``G`` to the control-masked symmetric identity:

        G*_ijkl(DC) = control[i][j] * 0.5 (d_ik d_jl + d_il d_jk)

    Stress-controlled components pass the projector unprojected and become
    solved-for macroscopic unknowns; all-zero ``control`` is a no-op. Major
    symmetry is kept.

    ``G`` must already be zero at DC; this is not checked.

    Parameters
    ----------
    G       : (3, 3, 3, 3, Nv)
    control : (3, 3) 0/1 mask, 1 = stress-controlled

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
    """LinearOperator ``ddot42(G, x)`` around a DC-patched ``G``; self-adjoint."""

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
    n_devices:      int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Mixed strain/stress macroscopic BC in a single CG solve (DC-bin identity).

    Kabel et al (2016) for a Green operator, Lucarini & Segurado (2019a) for a
    Galerkin projector; same code. Solves G*(C:eps - sigma_goal) = 0 for the
    full strain: strain-controlled components are fixed in ``eps0``,
    stress-controlled ones are solved jointly with the fluctuation. All-zero
    ``control`` reproduces solve_lippmann_schwinger exactly.

    Parameters
    ----------
    n                 : grid shape (nx, ny, nz)
    C_field           : (3, 3, 3, 3, Nv) stiffness or tangent
    elastic_op        : operator with a DC-zero ``.G`` (see _HasG)
    control           : (3, 3) 0/1 mask, 1 = stress-controlled
    eps_bar           : (3, 3) macroscopic strain; stress-controlled entries ignored
    macro_stress_goal : (3, 3) target stress; only stress-controlled entries used
    toler_lin, maxiter: CG tolerance and iteration cap
    n_devices         : device cap for domain decomposition (None = all)

    Returns
    -------
    eps, sigma, delta : (3, 3, Nv) strain, stress, strain correction
    eps_bar_out       : (3, 3) macroscopic strain, stress-controlled entries solved
    converged         : bool
    """
    resolved_n_devices = choose_device_count(n, n_devices)
    if resolved_n_devices > 1:
        return _solve_mixed_bc_dc_identity_sharded(
            n, C_field, elastic_op, control, eps_bar, macro_stress_goal,
            toler_lin, maxiter, resolved_n_devices,
        )

    Nv = prod(n)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

    patched_op = _PatchedOperator(patch_dc_identity(elastic_op.G, control))
    gamma0 = Gamma0Operator(n, patched_op, n_devices=1)

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


def _solve_mixed_bc_dc_identity_sharded(
    n:                 Tuple[int, ...],
    C_field:           jnp.ndarray,
    elastic_op:        _HasG,
    control:           Tuple[Tuple[int, ...], ...],
    eps_bar:           jnp.ndarray,
    macro_stress_goal: jnp.ndarray,
    toler_lin:         float,
    maxiter:           int,
    n_devices:         int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    ``n_devices > 1`` body of solve_mixed_bc_dc_identity: the whole CG under one
    ``jax.shard_map``, fields kept sharded.

    The DC patch is baked into ``G`` before sharding, so unlike the
    displacement solver no device-0 masking is needed; the mean strain is a
    cross-device psum.
    """
    Nv = prod(n)
    n_local = (n[0] // n_devices, n[1], n[2])
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

    patched_G = patch_dc_identity(elastic_op.G, control)
    C_sharded = shard_x_slabs(C_field, n_devices)
    G_sharded = shard_x_slabs(patched_G, n_devices)

    def solve_local(C_local, G_local):
        Nv_local = C_local.shape[-1]

        def fft_(x):
            return pfft3d_flat(x, n_local, "i")

        def ifft_(x):
            return pifft3d_flat(x, n_local, "i")

        def gamma0_local(x_local):
            return ifft_(ddot42(G_local, fft_(x_local)))

        def A_op(v_flat):
            v = v_flat.reshape(3, 3, Nv_local)
            Cv = ddot42(C_local, v)
            return gamma0_local(Cv).reshape(-1)

        eps0_local = jnp.ones((3, 3, Nv_local)) * (eps_bar * (1.0 - control_arr))[:, :, None]
        sg_local = jnp.ones((3, 3, Nv_local)) * macro_stress_goal[:, :, None]
        sigma0_local = ddot42(C_local, eps0_local)
        bb = -gamma0_local(sigma0_local - sg_local).reshape(-1)

        x0 = jnp.zeros_like(bb)
        delta_flat, converged = cg_solve_pmap(A_op, bb, x0, toler_lin, maxiter, axis_name="i")

        delta_local = delta_flat.reshape(3, 3, Nv_local)
        eps_local = eps0_local + delta_local
        sigma_local = ddot42(C_local, eps_local)
        eps_bar_out_local = jax.lax.psum(jnp.sum(eps_local, axis=-1), "i") / Nv
        return eps_local, sigma_local, delta_local, eps_bar_out_local, converged

    return x_slab_shard_map(
        solve_local, n_devices,
        in_specs=(x_slab_spec(5), x_slab_spec(5)),
        out_specs=(x_slab_spec(3), x_slab_spec(3), x_slab_spec(3), P(), P()),
    )(C_sharded, G_sharded)
