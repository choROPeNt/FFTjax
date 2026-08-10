# Mixed Strain/Stress Boundary Conditions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choROPeNt/FFTjax/blob/main/notebooks/lin-elastic_mixed-BC.ipynb)

A more realistic walkthrough of FFTjax's mechanical solver: instead of prescribing the full
macroscopic strain tensor (as in [Linear-Elastic Solve](./lin-elastic-strain.md)), this example
prescribes a **mixed** boundary condition — displacement-controlled tension along x with free
(traction-free) lateral surfaces along y and z, the condition an actual tensile-test specimen is
under (axial extension imposed at the grips, lateral surfaces free to contract via Poisson's
effect). The point is to measure the composite's effective Poisson's ratios, which a fully
strain-constrained case cannot show.

This uses `solvers.mechanical.displacement_nw_cg.ddisp_nw_cg`, the displacement-based solver —
required here because the strain-based solver's cheaper mixed-BC variant
(`dstrain_nw_cg_mixed`) is only valid for **homogeneous** materials (the mixed-BC CG operator loses
the symmetry a one-shot solve needs otherwise); our fibre/matrix composite is heterogeneous, so this
is the correct choice.

```python
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from operators.green import build_freq_grid
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field
from solvers.mechanical.displacement_nw_cg import ddisp_nw_cg
from post.fields import field_to_grid, compute_displacement

# Composite RVE: square-packed 2-fibre geometry, same as the Linear-Elastic example.
phase_np, N, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, spacing=0.0002, N_min=32, nz=10,
)
Nv = int(np.prod(n))

# Materials: glass fibre in an epoxy matrix -- a common, high-contrast (~23x) composite.
matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")

phase = jnp.array(phase_np.reshape(-1))  # 0 = matrix, 1 = fiber
C_field = assemble_C_field([matrix, fiber], phase)

# Mixed boundary conditions and solve:
# - control[0][0] = 0 (strain-controlled): eps_bar[0, 0] = 1e-3 sets the axial tension directly.
# - control[1][1] = control[2][2] = 1 (stress-controlled): stress_goal is zero there, i.e. free
#   lateral surfaces -- the solver finds whatever lateral strain makes sigma22 = sigma33 = 0.
# - Shear components stay strain-controlled at zero (no shear loading).
L_mm = tuple(float(Li) for Li in L)
dx = tuple(Li / ni for Li, ni in zip(L_mm, n))
xi_flat = build_freq_grid(n, L_mm)

eps_bar = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
control = (
    (0, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)
stress_goal = jnp.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

eps, sigma, delta, eps_bar_out, converged = ddisp_nw_cg(
    n, C_field, xi_flat, eps_bar, control, stress_goal, toler_lin=1e-6, maxiter=2000,
)

nu_xy = float(-eps_bar_out[1, 1] / eps_bar_out[0, 0])
nu_xz = float(-eps_bar_out[2, 2] / eps_bar_out[0, 0])
```

```text title="Output"
JAX backend: cpu
Devices: [CpuDevice(id=0)]
grid n : (89, 89, 10)
domain L [mm]: (0.01772453850905516, 0.01772453850905516, 0.002)
fiber volume fraction (actual): 0.5010730968312082
LinearElasticIsotropic (epoxy matrix): E=3e+03, nu=0.35, lam=2.59e+03, mu=1.11e+03
LinearElasticIsotropic (glass fiber): E=7e+04, nu=0.2, lam=1.94e+04, mu=2.92e+04
converged     : True
eps_bar (solved):
[[ 1.00000000e-03  0.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00 -5.17366217e-04  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00 -5.13418124e-05]]

sigma11 (avg) : 7.09444930382397 MPa
sigma22 (avg) : -2.1981055482330628e-08 MPa  (target: 0, free surface)
sigma33 (avg) : -4.814807193507261e-09 MPa  (target: 0, free surface)

PASSED -- free lateral surfaces converged to sigma22 = sigma33 = 0.

effective nu_xy = -eps22/eps11 : 0.5173662173836436
effective nu_xz = -eps33/eps11 : 0.051341812371163746
```

![Fiber phase and resulting displacement field](/img/lin_elastic_mixed_bc.png)

The free lateral surfaces let the composite contract under axial load — exactly what a real
tensile specimen does (Poisson's effect) — which the constrained uniaxial-*strain* case in
[Linear-Elastic Solve](./lin-elastic-strain.md) cannot show, since it locks `eps22 = eps33 = 0` by
construction. The two effective Poisson ratios also differ from each other: the square-packed
fibres (aligned along Z) make this composite's in-plane (xy) and through-thickness (xz) responses
genuinely anisotropic, not the single-value isotropic Poisson's ratio either constituent has on its
own. `nu_xy ≈ 0.52` slightly exceeding 0.5 is not an error — that ceiling is a thermodynamic bound
for *isotropic* materials only, and doesn't apply to this anisotropic effective medium; it's a
documented effect in fibre-composite micromechanics at this volume fraction and stiffness contrast.

:::note[Reproducing]
This page's code, output, and plot are generated by
[`examples/lin_elastic_mixed_bc.py`](https://github.com/choROPeNt/FFTjax/blob/main/examples/lin_elastic_mixed_bc.py):

```bash
python examples/lin_elastic_mixed_bc.py
```

If the example changes, re-run the script and update the pasted output/image above — there's no
build-time execution here, since Docusaurus can't run Python.

For the full interactive version — with per-field strain/stress visualization and `.xdmf`/`.h5`
export for ParaView — see
[`notebooks/lin-elastic_mixed-BC.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/lin-elastic_mixed-BC.ipynb),
linked via the Colab badge above.
:::

## Next steps

- For a **homogeneous** material, `solvers.mechanical.strain_nw_cg.dstrain_nw_cg_mixed` solves the
  same kind of mixed BC more cheaply, reusing the fixed reference-medium Green's operator instead of
  the true heterogeneous stiffness — not valid here since it loses symmetry for heterogeneous
  materials.
- See `solvers.damage.anderson` and the [Phase-Field Fracture example](./phase-field.md) for
  coupling this kind of mechanical solve to damage evolution.
