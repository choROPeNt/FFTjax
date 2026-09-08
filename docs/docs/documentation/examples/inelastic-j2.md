# In-Elastic Solve (J2 Elastoplasticity)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choROPeNt/FFTjax/blob/main/notebooks/in-elastic_J2.ipynb)

FFTjax's rate-independent J2 (von Mises) plasticity material model
(`materialmodels.inelastic.plasticity_j2.J2Plasticity`, linear isotropic hardening),
demonstrated on the same two-phase composite RVE as the
[Linear-Elastic examples](./lin-elastic-strain.md): an elastic glass fiber embedded in a
plastic epoxy matrix.

Unlike the linear elastic models, this material is **stateful** — stress and tangent depend
on $(\boldsymbol{\varepsilon}, \boldsymbol{\varepsilon}_p, \alpha)$, not just $\boldsymbol{\varepsilon}$ — and its
tangent stiffness

$$
\mathbb{C}^\text{tan}_{ijkl} = \frac{\partial \sigma_{ij}}{\partial \varepsilon_{kl}}
$$

is obtained by **automatic differentiation** of the closed-form radial-return stress update
(`jax.jacfwd`), not a hand-derived consistent elastoplastic tangent. Because that tangent
depends on the unknown strain, this can't go through the plain linear solvers the way the
elastic examples do — it uses a Newton-outer/CG-inner driver instead,
`problems.mechanics.solve_displacement_based_nonlinear`: at each Newton iteration it
evaluates the actual (nonlinear) stress and tangent at the current strain guess via a
per-voxel `local_update` callable, and uses CG only for the linearized correction.

The macroscopic shear strain is ramped from 0 up past the matrix's yield strain, unloaded
through zero into reversed shear, then reloaded — carrying `(eps_p, alpha)` forward from one
step to the next — tracing out a **hysteresis loop** in the homogenized (volume-averaged)
shear response, even though only the matrix phase is plastic.

```python
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.inelastic.plasticity_j2 import J2Plasticity
from operators.green import build_freq_grid
from problems.mechanics import solve_displacement_based_nonlinear

# Composite RVE: square-packed 2-fiber geometry, same as the Linear-Elastic examples.
phase_np, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=24, nz=1,
)
Nv = int(np.prod(n))
phase = jnp.array(phase_np.reshape(-1))
xi_flat = build_freq_grid(n, L)

# Materials: elastic glass fiber, J2-plastic epoxy matrix.
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
matrix = J2Plasticity(E=3.76e3, nu=0.39, sigma_y0=50.0, H=1.0e3, name="epoxy matrix (plastic)")
C_fiber = fiber.stiffness_tensor()

# local_update is the per-voxel combinator the nonlinear driver needs: fiber gets a
# plain linear sigma = C:eps (constant tangent, no state); matrix gets
# J2Plasticity.stress_and_tangent_field, which returns the autodiff tangent above.
# Combined by phase via jnp.where.
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

# Load / unload / reload cycle in macroscopic shear strain, warm-starting each
# step from the previous converged strain.
gamma_max = 0.03
gammas_load = np.linspace(0.0, gamma_max, 11)[1:]
gammas_unload = np.linspace(gamma_max, -gamma_max, 16)[1:]
gammas_reload = np.linspace(-gamma_max, gamma_max, 16)[1:]
gammas_applied = np.concatenate([[0.0], gammas_load, gammas_unload, gammas_reload])

state = (jnp.zeros((3, 3, Nv)), jnp.zeros(Nv))
eps_prev_full = jnp.zeros((3, 3, Nv))
tau_avg_path = []
for step, gamma in enumerate(gammas_applied):
    eps_bar_step = jnp.array([[0., gamma / 2, 0.], [gamma / 2, 0., 0.], [0., 0., 0.]])
    if step == 0:
        eps_step, sigma_step = jnp.zeros((3, 3, Nv)), jnp.zeros((3, 3, Nv))
    else:
        eps0_step = jnp.ones((3, 3, Nv)) * eps_bar_step[:, :, None]
        delta_init = eps_prev_full - eps0_step  # warm start
        eps_step, sigma_step, state, converged, n_iter = solve_displacement_based_nonlinear(
            n, xi_flat, eps_bar_step, local_update, state,
            toler_lin=1e-7, maxiter_lin=2000, toler_nr=1e-7, maxiter_nr=50,
            delta_init=delta_init,
        )
    eps_prev_full = eps_step
    tau_avg_path.append(float(jnp.mean(sigma_step[0, 1])))

_, alpha_final = state
```

```text title="Output"
JAX backend: cpu
Devices: [CpuDevice(id=0)]
grid n : (89, 89, 1)
total voxels Nv: 7921
fiber volume fraction (actual): 0.5010730968312082
LinearElasticIsotropic (glass fiber): E=7e+04, nu=0.2, lam=1.94e+04, mu=2.92e+04
J2Plasticity (epoxy matrix (plastic)): E=3.76e+03, nu=0.39, sigma_y0=50, H=1e+03

Newton iterations per solved step: min=3, max=12
converged: True (all 40 solved steps)
plastic matrix voxels: 3524/3952
max accumulated plastic strain: 1.8632e-01
```

![In-Elastic Solve (J2 elastoplasticity)](/img/in-elastic_J2.png)

The unloading branch (left) has the elastic modulus's slope, not the plastic tangent's —
correct, since unloading from a plastic state is purely elastic until the reversed yield
surface is reached. The reload branch sits offset above the load branch, the expected
permanent-set signature of isotropic hardening. The accumulated plastic strain map (right)
shows yielding concentrated at the fiber/matrix interface and along the diagonals between
periodic neighboring fibers — the low-stiffness matrix regions where the stress locally rises
above the far-field average — while the fiber itself (dark blue disk) stays purely elastic
throughout, as expected.

:::note[Reproducing]
This page's code, output, and plot are generated by
[`examples/inelastic_j2.py`](https://github.com/choROPeNt/FFTjax/blob/main/examples/inelastic_j2.py):

```bash
python examples/inelastic_j2.py
```

If the example changes, re-run the script and update the pasted output/image above — there's no
build-time execution here, since Docusaurus can't run Python.

For the full interactive version — including the virgin `strain_p = 0` field, XDMF/HDF5 export
via `IncrementalWriter` for the whole load/unload/reload history, and the theoretical background
(yield function, closed-form radial return) — see
[`notebooks/in-elastic_J2.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/in-elastic_J2.ipynb),
linked via the Colab badge above.
:::

## Next steps

- See `test/test_problems_mechanics_nonlinear.py` for the validation of
  `solve_displacement_based_nonlinear` itself, including reproducing
  `solve_displacement_based`'s own answer to ~1e-13 when given a plain linear `local_update`.
- A nonlinear hardening law (e.g. Voce) would need a local scalar Newton solve for the plastic
  multiplier instead of the closed-form radial return used here — not built yet.
