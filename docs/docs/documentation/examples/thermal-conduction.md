# Steady-State Thermal Conduction Homogenization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choROPeNt/FFTjax/blob/main/notebooks/thermal/thermal_conduction.ipynb)

A walkthrough of FFTjax's thermal conduction solver (`problems.thermal.solve_thermal`) — the
scalar (heat conduction) analogue of the [Linear-Elastic Solve](./lin-elastic-strain.md), mirrored
structurally throughout: build $K(\mathbf{x})$ instead of $\mathbb{C}(\mathbf{x})$, prescribe a
macroscopic temperature gradient instead of a macroscopic strain, solve for a periodic fluctuation
instead of a periodic displacement.

$$
\nabla \cdot \mathbf{q}(\mathbf{x}) = 0, \qquad
\mathbf{q} = -K(\mathbf{x})\,\nabla T, \qquad
\nabla T = \overline{\nabla T} + \nabla T'(\mathbf{x})
$$

where $\overline{\nabla T}$ is the prescribed macroscopic gradient and $T'$ is an unknown periodic
fluctuation (mean zero — a gauge choice, only $\nabla T'$ is physically meaningful). One difference
from the elastic case worth flagging up front: a gradient/flux pair is a plain 3-vector, not a
symmetric 2nd-order tensor like strain/stress — there's no Voigt-6 form for it
(`post.fields.macroscopic_thermal_response` reports plain `(grad_T_bar, flux_bar)` vectors, not a
6-component `to_voigt` pair). Full derivation: [Thermal Conduction](../theorie/thermal.md).

## Two-phase laminate — closed-form validation

Before trusting the solver on a real microstructure, check it against a case with a known answer.
A laminate (layers of two materials, normal to $x$) has a textbook effective conductivity: the
**harmonic mean** $k_\text{eff} = \big(\tfrac{v_1}{k_1} + \tfrac{v_2}{k_2}\big)^{-1}$ for a gradient
**normal** to the layers (a *series* thermal circuit), and the **arithmetic mean**
$k_\text{eff} = v_1 k_1 + v_2 k_2$ for a gradient **in-plane** (a *parallel* circuit).

```python
import utils.precision  # side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from materialmodels.thermal.isotropic import ThermalConductivityIsotropic
from problems.thermal import solve_thermal

n = (64, 8, 8)
L = (1.0, 1.0, 1.0)   # any consistent length unit

# Layers normal to x, equal volume fraction.
phase_np = np.zeros(n, dtype=int)
phase_np[n[0] // 2:, :, :] = 1
phase = jnp.array(phase_np.reshape(-1))

# W/(m*K) -- representative values for epoxy resin and E-glass fibre (reused for the
# composite RVE further down, so the whole page shares one physically grounded pair).
k_matrix, k_fiber = 0.20, 1.05
materials = [ThermalConductivityIsotropic(k_matrix, "phase 0"), ThermalConductivityIsotropic(k_fiber, "phase 1")]

vf = 0.5
k_eff_series   = 1.0 / (vf / k_matrix + vf / k_fiber)
k_eff_parallel = vf * k_matrix + vf * k_fiber

# gradient along x -- normal to the layers, a series circuit
grad_T_bar_x = jnp.array([1.0, 0.0, 0.0])
results = solve_thermal(n, L, phase, materials, grad_T_bar_x, toler_lin=1e-10, maxiter=2000)
sol = results[0].solution
flux_bar = jnp.mean(sol.flux, axis=-1)
k_eff_x = -float(flux_bar[0]) / float(grad_T_bar_x[0])

# gradient along y -- in-plane, a parallel circuit
grad_T_bar_y = jnp.array([0.0, 1.0, 0.0])
results = solve_thermal(n, L, phase, materials, grad_T_bar_y, toler_lin=1e-10, maxiter=2000)
sol = results[0].solution
flux_bar = jnp.mean(sol.flux, axis=-1)
k_eff_y = -float(flux_bar[1]) / float(grad_T_bar_y[1])
```

```text title="Output"
analytical k_eff:  series (harmonic mean) = 0.3360   parallel (arithmetic mean) = 0.6250
converged: True
k_eff (numeric, series)   = 0.336000   (analytical 0.336000)
k_eff (numeric, parallel) = 0.625000   (analytical 0.625000)

Both bounds match to within CG tolerance.
```

## Mixed gradient/flux boundary conditions

`control` marks which directions are flux-controlled (1) rather than gradient-controlled (0) — the
scalar analogue of `solve_mechanics`'s mixed strain/stress `control`. Driving the gradient along
$x$ while prescribing a nonzero target flux on $y$ and $z$ is solved for jointly, and the solved
macroscopic gradient on those directions comes back in `grad_T_bar_out`.

```python
control = (0, 1, 1)                        # y, z flux-controlled; x gradient-controlled
flux_goal = jnp.array([0.0, -0.7, 0.3])    # target flux on y, z

results = solve_thermal(n, L, phase, materials, grad_T_bar_x,
                         control=control, flux_goal=flux_goal, toler_lin=1e-10, maxiter=3000)
sol = results[0].solution
flux_bar = jnp.mean(sol.flux, axis=-1)
```

```text title="Output"
flux_bar        = [-0.336 -0.7    0.3  ]
target flux_goal= [ 0.  -0.7  0.3]  (y, z should match)
grad_T_bar_out   = [ 1.    1.12 -0.48]  (x stays 1.0, y/z solved for)

Flux-controlled directions hit their targets; the gradient-controlled entry is untouched.
```

## A real composite microstructure

Same square-packed fiber RVE as the [Linear-Elastic Solve](./lin-elastic-strain.md) — a matrix
phase with circular fiber cross-sections on a square lattice. No closed-form bound applies to a
real (non-laminate) microstructure, but the effective conductivity along any direction must still
fall between the series and parallel bounds above — a real material can never do better than an
idealized parallel circuit or worse than an idealized series one.

```python
from generation.rve import make_square_composite_rve
from post.fields import macroscopic_thermal_response, vector_field_to_grid

phi, r_fiber, dx = 0.5, 0.005, 0.0002
phase_np, n, L, phi_act = make_square_composite_rve(
    phi=phi, r_fiber=r_fiber, dx=dx, N_min=32, nz=1,
)
phase = jnp.array(phase_np.reshape(-1))

materials = [ThermalConductivityIsotropic(k_matrix, "matrix"), ThermalConductivityIsotropic(k_fiber, "fibre")]

grad_T_bar = jnp.array([1.0, 0.0, 0.0])
results = solve_thermal(n, L, phase, materials, grad_T_bar, toler_lin=1e-8)
sol = results[0].solution

grad_bar = jnp.mean(sol.grad_T, axis=-1)
flux_bar = jnp.mean(sol.flux, axis=-1)
resp = macroscopic_thermal_response(grad_bar, flux_bar)

k_eff_x = -float(flux_bar[0]) / float(grad_T_bar[0])

# Local flux magnitude |q(x)| -- the thermal analogue of the von Mises stress concentration
# the Linear-Elastic example shows.
flux_grid = vector_field_to_grid(sol.flux, n)
flux_mag = np.linalg.norm(flux_grid, axis=-1)[:, :, 0]
```

```text title="Output"
grid n : (89, 89, 1)   fibre volume fraction (actual): 0.5011
converged: True
macroscopic response: {'grad_T_bar': array([1., 0., 0.]), 'flux_bar': array([-0.41086, -0., 0.]), 'flux_magnitude': 0.41086}

k_eff_x (this RVE) = 0.4109 W/(m*K)
  series/parallel bounds : [0.3360, 0.6250]  (satisfied)
```

![Fibre cross-section](/img/examples/thermal_fibre_rve.png)
![Local flux magnitude](/img/examples/thermal_flux_magnitude.png)

Flux concentrates in the higher-conductivity fiber and funnels through the narrow matrix
ligaments between periodic neighbors, with low-flux "shadow" pockets forming where the fiber
blocks the gradient direction — the same qualitative redistribution pattern the elastic examples
show for stress, just for a scalar field instead of a tensor one.

## Anisotropic fiber conductivity: along vs. across the fiber axis

Real fibers are often not isotropic — carbon fiber in particular can have a conductivity along its
own axis many times its transverse value (E-glass, used above, happens to be close to isotropic,
which is why the plain `ThermalConductivityIsotropic` fiber already gave the right transverse
answer). `materialmodels.thermal.transverse_isotropic.ThermalConductivityTransverseIsotropic` —
mirroring the elastic side's `TransverseIsotropic` — carries the two independent constants a
fiber's conductivity actually needs: `k_L` along its axis, `k_T` across it.

This RVE's fibers run along $z$ (`nz=1`), so both regimes can be probed on the *same* geometry,
just by changing which direction the gradient points. Along the fiber axis the extruded
microstructure doesn't vary with $z$, so the trivial ansatz $T'=0$ already satisfies
$\nabla\cdot\mathbf{q}=0$ exactly — making the along-fiber effective conductivity an *exact*
closed-form volume-fraction-weighted arithmetic mean, a stronger check than the CG-tolerance-limited
laminate bounds above.

```python
from materialmodels.thermal.transverse_isotropic import ThermalConductivityTransverseIsotropic

k_L_fiber, k_T_fiber = 20.0, 1.5   # illustrative (carbon-fibre-like anisotropy), not measured data

materials_aniso = [
    ThermalConductivityIsotropic(k_matrix, "matrix"),
    ThermalConductivityTransverseIsotropic(k_L=k_L_fiber, k_T=k_T_fiber,
                                            fiber_dir=[0.0, 0.0, 1.0], name="fibre (anisotropic)"),
]

# transverse: gradient in-plane, same direction as the composite section above
results_T = solve_thermal(n, L, phase, materials_aniso, grad_T_bar, toler_lin=1e-10, maxiter=3000)
flux_T = jnp.mean(results_T[0].solution.flux, axis=-1)
k_eff_T = -float(flux_T[0]) / float(grad_T_bar[0])

# along-fibre: gradient along z
grad_T_bar_z = jnp.array([0.0, 0.0, 1.0])
results_L = solve_thermal(n, L, phase, materials_aniso, grad_T_bar_z, toler_lin=1e-10, maxiter=3000)
sol_L = results_L[0].solution
flux_L = jnp.mean(sol_L.flux, axis=-1)
k_eff_L = -float(flux_L[2]) / float(grad_T_bar_z[2])

k_eff_z_analytical = phi_act * k_L_fiber + (1 - phi_act) * k_matrix
```

```text title="Output"
transverse  k_eff (x-y plane, uses k_T=1.5)  : 0.4554
along-fibre k_eff (z, uses k_L=20.0)          : 10.1212
anisotropy ratio k_eff_L / k_eff_T                   : 22.23

k_eff_L (numeric)    = 10.121247
k_eff_L (analytical) = 10.121247   (Vf*k_L + (1-Vf)*k_matrix)
max|T_prime| along z = 0.00e+00   (expect ~0 -- trivial solution)

PASSED: along-fibre conduction matches the exact closed form.
```

:::note[Reproducing]
This page's code and output are generated by
[`notebooks/thermal/thermal_conduction.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/thermal/thermal_conduction.ipynb)
(linked via the Colab badge above) — there is no standalone `examples/` script for this case, since
it would just be a redundant duplicate of the notebook.

If the notebook changes, re-run it and update the pasted output/images above — there's no
build-time execution here, since Docusaurus can't run Python.
:::

## Next steps

- Swap in a mix of fiber orientations or a third phase — `materialmodels.assembly.assemble_K_field`
  is generic over any `ConductivityModel`, no different from the elastic side's `assemble_C_field`.
- Sweep the gradient direction over `uniaxial_x/y/z` to assemble a full effective conductivity
  tensor from several single-direction solves, the thermal analogue of the load-case sweep
  `problems/utils/loadcases.py` already provides on the elastic side.
- Give the anisotropic fiber a per-voxel orientation field instead of one global `fiber_dir` —
  `ThermalConductivityTransverseIsotropic.conductivity_field_oriented` (and `assemble_K_field`'s
  automatic detection of it) already support that; this page's RVE happens to be single-orientation.
- `solve_thermal`'s `writer=` argument writes `gradient`/`flux`/`temperature_fluctuation` fields to
  XDMF/HDF5 via the same `utils.io.xdmf_writer.IncrementalWriter` every other solve in this project
  uses — open the result in ParaView exactly like an elastic or fracture run.
