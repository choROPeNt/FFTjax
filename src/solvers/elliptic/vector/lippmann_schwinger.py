"""Lippmann-Schwinger (reference-medium) elastic solves and their mixed-BC variants."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple
from math import prod

import jax
import jax.numpy as jnp

from operators.base import LinearOperator
from jax.sharding import PartitionSpec as P

from operators.fft_distributed import (
    choose_device_count, pfft3d_flat, pifft3d_flat, shard_x_slabs, x_slab_mesh, x_slab_spec,
)
from operators.general_functions import ddot42
from operators.green import GreenOperatorBasic
from operators.projection import Gamma0Operator
from solvers.elliptic.vector.base import ElasticitySolver
from solvers.elliptic.vector.mixed_bc import (
    _HasG, _ZERO_CONTROL, _active_pairs, sm2sv, solve_mixed_bc_dc_identity, sv2sm,
)
from solvers.krylov.cg import cg_solve, cg_solve_pmap
from solvers.solution import ElasticitySolution


def solve_lippmann_schwinger(
    n:           Tuple[int, ...],
    C_field:     jnp.ndarray,
    green_op:    LinearOperator,
    eps_bar:     jnp.ndarray,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-4,
    maxiter:     int = 1000,
    n_devices: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Krylov Lippmann-Schwinger CG solve (pure strain BC):

        A(d_eps) = Gamma0(C : d_eps),   b = -Gamma0(C : eps0 - sigma_goal),   eps0 = eps_bar

    Shards the whole CG across devices when more than one is available.

    Parameters
    ----------
    n           : grid shape (nx, ny, nz)
    C_field     : (3, 3, 3, 3, Nv) stiffness or tangent
    green_op    : GreenOperatorBasic or GreenOperatorWillot for this grid
    eps_bar     : (3, 3) macroscopic strain
    stress_goal : (3, 3, Nv) target stress field (None = 0)
    toler_lin, maxiter : CG tolerance and iteration cap
    n_devices   : device cap for domain decomposition (None = all)

    Returns
    -------
    eps, sigma, delta : (3, 3, Nv) strain, stress, strain correction
    converged         : bool
    """
    Nv = prod(n)
    resolved_n_devices = choose_device_count(n, n_devices)
    if resolved_n_devices > 1:
        return _solve_lippmann_schwinger_sharded(
            n, C_field, green_op, eps_bar, stress_goal, toler_lin, maxiter, resolved_n_devices,
        )

    gamma0 = Gamma0Operator(n, green_op, n_devices=1)

    def A_op(v_flat):
        v = v_flat.reshape(3, 3, Nv)
        Cv = ddot42(C_field, v)
        return gamma0(Cv).reshape(-1)

    sg     = jnp.zeros((3, 3, Nv)) if stress_goal is None else stress_goal
    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = ddot42(C_field, eps0)
    bb     = -gamma0(sigma0 - sg).reshape(-1)

    x0 = jnp.zeros_like(bb)
    delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)

    delta = delta_flat.reshape(3, 3, Nv)
    eps   = eps0 + delta
    sigma = ddot42(C_field, eps)

    return eps, sigma, delta, converged


def _solve_lippmann_schwinger_sharded(
    n:           Tuple[int, ...],
    C_field:     jnp.ndarray,
    green_op:    LinearOperator,
    eps_bar:     jnp.ndarray,
    stress_goal: jnp.ndarray | None,
    toler_lin:   float,
    maxiter:     int,
    n_devices:   int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``n_devices > 1`` body of solve_lippmann_schwinger: one ``shard_map``, fields kept sharded."""
    n_local = (n[0] // n_devices, n[1], n[2])

    C_sharded = shard_x_slabs(C_field, n_devices)
    G_sharded = shard_x_slabs(green_op.G, n_devices)
    sg = jnp.zeros((3, 3, prod(n))) if stress_goal is None else stress_goal
    sg_sharded = shard_x_slabs(sg, n_devices)

    def solve_local(C_local, G_local, sg_local):
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

        eps0_local = jnp.ones((3, 3, Nv_local)) * eps_bar[:, :, None]
        sigma0_local = ddot42(C_local, eps0_local)
        bb = -gamma0_local(sigma0_local - sg_local).reshape(-1)

        x0 = jnp.zeros_like(bb)
        delta_flat, converged = cg_solve_pmap(A_op, bb, x0, toler_lin, maxiter, axis_name="i")

        delta_local = delta_flat.reshape(3, 3, Nv_local)
        eps_local = eps0_local + delta_local
        sigma_local = ddot42(C_local, eps_local)
        return eps_local, sigma_local, delta_local, converged

    return jax.shard_map(
        solve_local, mesh=x_slab_mesh(n_devices),
        in_specs=(x_slab_spec(5), x_slab_spec(5), x_slab_spec(3)),
        out_specs=(x_slab_spec(3), x_slab_spec(3), x_slab_spec(3), P()),
    )(C_sharded, G_sharded, sg_sharded)


class LippmannSchwingerSolver(ElasticitySolver):
    """ElasticitySolver for ``solve_lippmann_schwinger`` (pure strain BC)."""

    def __init__(
        self,
        n:           Tuple[int, ...],
        green_op:    LinearOperator,
        toler_lin:   float = 1e-4,
        maxiter:     int = 1000,
        n_devices: int | None = None,
    ):
        self.n = n
        self.green_op = green_op
        self.toler_lin = toler_lin
        self.maxiter = maxiter
        self.n_devices = n_devices

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        eps, sigma, delta, converged = solve_lippmann_schwinger(
            self.n, C_field, self.green_op, eps_bar, stress_goal, self.toler_lin, self.maxiter,
            n_devices=self.n_devices,
        )
        return ElasticitySolution(eps, sigma, delta, converged)


def solve_lippmann_schwinger_dc(
    n:                 Tuple[int, ...],
    C_field:           jnp.ndarray,
    green_op:          _HasG,
    eps_bar:           jnp.ndarray,
    control:           Tuple[Tuple[int, ...], ...] | None = None,
    macro_stress_goal: jnp.ndarray | None = None,
    toler_lin:         float = 1e-4,
    maxiter:           int = 1000,
    n_devices:       int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Mixed BC in one CG solve via the DC-bin identity (Kabel et al 2016); see
    ``solve_mixed_bc_dc_identity``. A cross-check for the primary route,
    ``solve_lippmann_schwinger_mixed_bc``.
    """
    control = control if control is not None else _ZERO_CONTROL
    macro_stress_goal = jnp.zeros((3, 3)) if macro_stress_goal is None else macro_stress_goal
    return solve_mixed_bc_dc_identity(
        n, C_field, green_op, control, eps_bar, macro_stress_goal, toler_lin, maxiter,
        n_devices=n_devices,
    )


def solve_lippmann_schwinger_mixed_bc(
    n:                 Tuple[int, ...],
    C_field:           jnp.ndarray,
    green_op:          GreenOperatorBasic,
    control:           Tuple[Tuple[int, ...], ...],
    eps_bar_guess:     jnp.ndarray,
    macro_stress_goal: jnp.ndarray,
    toler_lin:         float = 1e-4,
    maxiter:           int = 1000,
    toler_outer:       float = 1e-6,
    maxiter_outer:     int = 50,
    C0:                jnp.ndarray | None = None,
    n_devices:       int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Mixed strain/stress BC by outer correction of ``eps_bar`` (Michel et al 1999).

    Repeats ``solve_lippmann_schwinger`` and corrects the stress-controlled
    entries with ``M dx = sigma_goal - <sigma>``, where M is ``C0`` restricted
    to those entries. The step is undamped: a poor ``C0`` can diverge rather
    than converge slowly. All-zero ``control`` is a single plain solve.

    Parameters
    ----------
    n, C_field, green_op : as solve_lippmann_schwinger
    control              : (3, 3) 0/1 mask, 1 = stress-controlled
    eps_bar_guess        : (3, 3) start; strain-controlled entries stay fixed
    macro_stress_goal    : (3, 3) target, stress-controlled entries only
    toler_lin, maxiter   : inner CG tolerance and iteration cap
    toler_outer, maxiter_outer : outer relative tolerance and iteration cap
    C0                   : (3, 3, 3, 3) stiffness for M; default the
                           anisotropic voxel mean of C_field
    n_devices            : device cap for domain decomposition (None = all)

    Returns
    -------
    eps, sigma, delta : from the last inner solve
    eps_bar_out       : (3, 3) macroscopic strain, stress-controlled entries solved
    converged         : inner CG and outer residual both converged
    n_iter_outer      : outer iterations run
    """
    pairs = _active_pairs(control)

    if not pairs:
        eps, sigma, delta, converged = solve_lippmann_schwinger(
            n, C_field, green_op, eps_bar_guess, toler_lin=toler_lin, maxiter=maxiter,
            n_devices=n_devices,
        )
        return eps, sigma, delta, eps_bar_guess, converged, 1

    # The anisotropic voxel mean, not green_op's isotropized reference medium:
    # with that, this undamped step diverged (eps_bar ~1e21) on a carbon/epoxy RVE.
    C0_ref = jnp.mean(C_field, axis=-1) if C0 is None else C0
    basis = jnp.eye(len(pairs), dtype=eps_bar_guess.dtype)
    M = jnp.stack(
        [sm2sv(jnp.einsum("ijkl,kl->ij", C0_ref, sv2sm(basis[k], pairs)), pairs)
         for k in range(len(pairs))],
        axis=1,
    )

    eps_bar = eps_bar_guess
    converged_outer = False

    for it in range(1, maxiter_outer + 1):
        eps, sigma, delta, converged = solve_lippmann_schwinger(
            n, C_field, green_op, eps_bar, toler_lin=toler_lin, maxiter=maxiter,
            n_devices=n_devices,
        )
        sigma_mean = jnp.mean(sigma, axis=-1)
        resid = sm2sv(macro_stress_goal - sigma_mean, pairs)
        # Relative to the current mean stress, not the target: a traction-free
        # target has norm 0, which would make the tolerance unreachable.
        scale = jnp.linalg.norm(sigma_mean) + 1e-30
        if float(jnp.linalg.norm(resid)) < toler_outer * float(scale):
            converged_outer = True
            break
        eps_bar = eps_bar + sv2sm(jnp.linalg.solve(M, resid), pairs)

    return eps, sigma, delta, eps_bar, jnp.logical_and(converged, converged_outer), it


class LippmannSchwingerMixedBCSolver(ElasticitySolver):
    """ElasticitySolver for ``solve_lippmann_schwinger_mixed_bc``; mixed BC via ``control``."""

    def __init__(
        self,
        n:             Tuple[int, ...],
        green_op:      GreenOperatorBasic,
        control:       Tuple[Tuple[int, ...], ...] | None = None,
        toler_lin:     float = 1e-4,
        maxiter:       int = 1000,
        toler_outer:   float = 1e-6,
        maxiter_outer: int = 50,
        C0:            jnp.ndarray | None = None,
        n_devices:   int | None = None,
    ):
        self.n = n
        self.green_op = green_op
        self.control = control if control is not None else _ZERO_CONTROL
        self.toler_lin = toler_lin
        self.maxiter = maxiter
        self.toler_outer = toler_outer
        self.maxiter_outer = maxiter_outer
        self.C0 = C0
        self.n_devices = n_devices

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        sg = jnp.zeros((3, 3)) if stress_goal is None else stress_goal
        eps, sigma, delta, eps_bar_out, converged, _n_iter_outer = solve_lippmann_schwinger_mixed_bc(
            self.n, C_field, self.green_op, self.control, eps_bar, sg,
            toler_lin=self.toler_lin, maxiter=self.maxiter,
            toler_outer=self.toler_outer, maxiter_outer=self.maxiter_outer, C0=self.C0,
            n_devices=self.n_devices,
        )
        return ElasticitySolution(eps, sigma, delta, converged, eps_bar=eps_bar_out)
