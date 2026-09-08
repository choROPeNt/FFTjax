"""
In-Elastic Solve (J2 Elastoplasticity) -- Quad RVE with a Plastic Matrix

A minimal walkthrough of FFTjax's rate-independent J2 (von Mises) plasticity
material model (materialmodels.inelastic.plasticity_j2.J2Plasticity, linear
isotropic hardening, autodiff-derived tangent) on a real two-phase composite
RVE: an elastic glass fiber embedded in a plastic epoxy matrix, same
square-packed geometry as lin_elastic_mixed_bc.py.

Because the matrix's tangent stiffness depends on the unknown strain and its
own history, this can't go through the plain linear solvers the way the
elastic examples do. Instead it uses the Newton-outer/CG-inner driver
problems.mechanics.solve_displacement_based_nonlinear: at each Newton
iteration it evaluates the actual (nonlinear) stress and tangent at the
current strain guess via a per-voxel local_update callable, and uses CG only
for the linearized correction.

The macroscopic shear strain is ramped from 0 up past the matrix's yield
strain, unloaded through zero into reversed shear, then reloaded -- carrying
(eps_p, alpha) forward from one step to the next -- tracing out a
hysteresis loop in the homogenized (volume-averaged) shear response even
though only the matrix phase is plastic.

Saves its plot to docs/static/img/in-elastic_J2.png for the Examples page.
Re-run this script and copy its printed output into
docs/docs/documentation/examples/inelastic-j2.md if the example ever changes.

Run from the repo root: python examples/inelastic_j2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.inelastic.plasticity_j2 import J2Plasticity
from operators.green import build_freq_grid
from problems.mechanics import solve_displacement_based_nonlinear

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# Composite RVE: square-packed 2-fiber geometry, same as lin_elastic_mixed_bc.py's,
# a slightly coarser grid so the load-stepped Newton-CG sweep stays fast for a demo.
phase_np, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=24, nz=1,
)
Nv = int(np.prod(n))
phase = jnp.array(phase_np.reshape(-1))
xi_flat = build_freq_grid(n, L)

print("grid n :", n)
print("total voxels Nv:", Nv)
print("fiber volume fraction (actual):", phi_act)

# Materials: elastic glass fiber, J2-plastic epoxy matrix. local_update is the
# per-voxel combinator the nonlinear driver needs: fiber gets a plain linear
# sigma = C:eps (constant tangent, no state); matrix gets
# J2Plasticity.stress_and_tangent_field. Combined by phase via jnp.where.
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
matrix = J2Plasticity(E=3.76e3, nu=0.39, sigma_y0=50.0, H=1.0e3, name="epoxy matrix (plastic)")
C_fiber = fiber.stiffness_tensor()

print(fiber)
print(matrix)


def local_update(eps_field, state):
    eps_p_field, alpha_field = state
    sigma_pl, C_pl, (eps_p_new, alpha_new) = matrix.stress_and_tangent_field(
        eps_field, eps_p_field, alpha_field
    )
    sigma_el = jnp.einsum("ijkl,klm->ijm", C_fiber, eps_field)
    C_el = jnp.broadcast_to(C_fiber[..., None], C_pl.shape)

    is_matrix = (phase == 0)
    sigma = jnp.where(is_matrix, sigma_pl, sigma_el)
    C_tan = jnp.where(is_matrix, C_pl, C_el)
    eps_p_out = jnp.where(is_matrix, eps_p_new, eps_p_field)
    alpha_out = jnp.where(is_matrix, alpha_new, alpha_field)
    return sigma, C_tan, (eps_p_out, alpha_out)


# Load / unload / reload cycle in macroscopic shear strain, including the
# virgin gamma=0 state as step 0.
gamma_max = 0.03
n_load, n_unload, n_reload = 10, 15, 15
gammas_load = np.linspace(0.0, gamma_max, n_load + 1)[1:]
gammas_unload = np.linspace(gamma_max, -gamma_max, n_unload + 1)[1:]
gammas_reload = np.linspace(-gamma_max, gamma_max, n_reload + 1)[1:]
gammas_applied = np.concatenate([[0.0], gammas_load, gammas_unload, gammas_reload])
i_load_end = 1 + len(gammas_load)
i_unload_end = 1 + len(gammas_load) + len(gammas_unload)

state = (jnp.zeros((3, 3, Nv)), jnp.zeros(Nv))
eps_prev_full = jnp.zeros((3, 3, Nv))
tau_avg_path, n_iters_path = [], []
for step, gamma in enumerate(gammas_applied):
    eps_bar_step = jnp.array([[0., gamma / 2, 0.], [gamma / 2, 0., 0.], [0., 0., 0.]])
    if step == 0:
        eps_step, sigma_step = jnp.zeros((3, 3, Nv)), jnp.zeros((3, 3, Nv))
    else:
        eps0_step = jnp.ones((3, 3, Nv)) * eps_bar_step[:, :, None]
        # warm start: previous step's converged strain minus the new baseline --
        # keeps Newton converging even past yield, whichever direction gamma moves.
        delta_init = eps_prev_full - eps0_step
        eps_step, sigma_step, state, converged, n_iter = solve_displacement_based_nonlinear(
            n, xi_flat, eps_bar_step, local_update, state,
            toler_lin=1e-7, maxiter_lin=2000, toler_nr=1e-7, maxiter_nr=50,
            delta_init=delta_init,
        )
        assert converged, f"Newton did not converge at gamma={gamma:.4f}"
        n_iters_path.append(n_iter)
    eps_prev_full = eps_step
    tau_avg_path.append(float(jnp.mean(sigma_step[0, 1])))

_, alpha_final = state
tau_avg_path = np.array(tau_avg_path)

print()
print(f"Newton iterations per solved step: min={min(n_iters_path)}, max={max(n_iters_path)}")
print("converged: True (all", len(gammas_applied) - 1, "solved steps)")

matrix_alpha = alpha_final[phase == 0]
n_plastic = int(jnp.sum(matrix_alpha > 1e-12))
print(f"plastic matrix voxels: {n_plastic}/{matrix_alpha.shape[0]}")
print(f"max accumulated plastic strain: {float(jnp.max(alpha_final)):.4e}")

# Visualize: hysteresis loop (homogenized shear response) and the spatial
# pattern of accumulated plastic strain at the end of the cycle.
strain_p_grid = np.array(alpha_final).reshape(n)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

axes[0].plot(gammas_applied[:i_load_end], tau_avg_path[:i_load_end],
             'o-', ms=4, color="C0", label="load")
axes[0].plot(gammas_applied[i_load_end - 1:i_unload_end], tau_avg_path[i_load_end - 1:i_unload_end],
             's--', ms=4, color="C1", label="unload")
axes[0].plot(gammas_applied[i_unload_end - 1:], tau_avg_path[i_unload_end - 1:],
             '^:', ms=4, color="C2", label="reload")
axes[0].axhline(0, color='gray', lw=0.5)
axes[0].axvline(0, color='gray', lw=0.5)
axes[0].set_xlabel(r"applied shear strain $\bar\gamma_{12}$")
axes[0].set_ylabel(r"homogenized shear stress $\langle\sigma_{12}\rangle$ [MPa]")
axes[0].set_title("RVE hysteresis")
axes[0].legend()

im = axes[1].imshow(strain_p_grid[:, :, 0].T, origin="lower", cmap="plasma")
axes[1].set_title("Accumulated plastic strain")
axes[1].set_xlabel("voxel x")
axes[1].set_ylabel("voxel y")
fig.colorbar(im, ax=axes[1], fraction=0.046)

fig.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "docs" / "static" / "img" / "in-elastic_J2_random-rve.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
