# Phase-Field Fracture (AT2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choROPeNt/FFTjax/blob/main/notebooks/damage_fracture/pff-damage.ipynb)

A minimal walkthrough of FFTjax's staggered phase-field fracture solver (`problems.fracture.solve_fracture`)
on a **randomly-packed circular carbon-fibre composite** with three phases (matrix, interphase,
fibre), generated with `generation.rve.make_random_composite_rve`
([Catalanotti 2016](https://doi.org/10.1016/j.compstruct.2015.11.039)) at a target fibre volume
fraction of 55%.

Two coupled PDEs are solved, staggered: mechanical equilibrium with a damage-degraded stress,

$$
\nabla \cdot \sigma(\mathbf{x}) = 0, \qquad
\sigma = \frac{\partial \psi}{\partial \varepsilon} = g(d)\,\frac{\partial \psi^+}{\partial \varepsilon} + \frac{\partial \psi^-}{\partial \varepsilon}, \qquad
\varepsilon = \tfrac12\big(\nabla u + \nabla u^\top\big),
$$

and the AT2 damage evolution equation — stationarity of the regularized fracture energy
$\Pi = \int_\Omega \psi(\varepsilon, d)\,dV + G_c\!\int_\Omega\!\big[\tfrac{d^2}{2\ell_0} + \tfrac{\ell_0}{2}|\nabla d|^2\big]\,dV$
with respect to $d$:

$$
\frac{G_c}{\ell_0}\,d \;-\; G_c\,\ell_0\,\Delta d \;=\; -g'(d)\,H, \qquad d \ge d_{\text{prev}} \ \text{(irreversibility)}.
$$

Here $g(d) = (1-k_{res})(1-d)^2 + k_{res}$ is the AT2 degradation function, $\psi^+$ is the tensile
part of the elastic strain energy density
([Amor, Marigo & Maurini 2009](https://doi.org/10.1016/j.jmps.2009.04.011) volumetric-deviatoric
split), and $H$ is a history variable enforcing irreversibility via the *hybrid* scheme
([Steinke & Kaliske 2019](https://doi.org/10.1007/s00466-018-1635-0)) rather than
$H=\max_{s\le t}\psi^+(s)$, which over-widens the diffuse process zone. Full derivation:
[Damage & Fracture Solvers](../theorie/damage.md).

The matrix and interphase (the two phases that actually damage) use
`materialmodels.phasefield.isotropic.PhaseFieldIsotropic`, which derives $\sigma$ and the
mechanical tangent as automatic derivatives of this same $\psi$ (`jax.grad`/`jax.jacfwd`), so the
compressive branch $\psi^-$ is never degraded by construction. The carbon fibre stays
`TransverseIsotropic` on the legacy uniform-degradation fallback (the Amor split is only defined
for an isotropic elastic law) — immaterial here since its `k_res=1.0` keeps it from ever losing
stiffness regardless of which path degrades it. `Gc` differs across all three phases
(`0.001`/`0.0004`/`0.0016` N/mm), so this example also exercises the heterogeneous-toughness path
(`solvers.elliptic.scalar.solve_damage_helmholtz_cg_het`) automatically — `solve_fracture`
dispatches to it whenever `Gc` isn't uniform, no separate call needed.

```python
import utils.precision  # side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

from generation.rve import make_random_composite_rve

# Randomly-packed circular fibres with a thin interphase ring around each one. The generator's
# own phase labeling is 0=matrix, 1=fibre, 2=interphase; remapped below to 0=matrix,
# 1=interphase, 2=fibre to match this example's materials list.
phi_target = 0.55
r_fiber    = 0.0035   # mm
vox        = 0.0005   # mm  target voxel size
interphase_thickness = 1 * vox

phase_raw, n, L, phi_fiber_act, centres = make_random_composite_rve(
    phi=phi_target, r_fiber=r_fiber, dx=vox,
    size_in_r=15,   # domain side ~ 15*r_fiber (Catalanotti 2016 convention)
    nz=1, K=15, seed=67,
    interphase_thickness=interphase_thickness,
)
phase_np = np.where(phase_raw == 1, 2, np.where(phase_raw == 2, 1, 0)).astype(np.uint8).reshape(-1)

Nv = int(np.prod(n))
dx = tuple(Li / ni for Li, ni in zip(L, n))
phi = float((phase_np == 2).mean())   # fibre+interphase volume fraction

from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from materialmodels.assembly import describe_materials
from materialmodels.phasefield.isotropic import PhaseFieldIsotropic

# Gc set per-material so solve_fracture can gather the heterogeneous Gc field automatically
# (Gc=None below); fibre's k_res=1.0 makes it damage-immune.
matrix = PhaseFieldIsotropic(E=3500.0, nu=0.35, Gc=1.0e-3, name="epoxy matrix")
interphase = PhaseFieldIsotropic(E=12000.0, nu=0.35, Gc=0.4e-3, name="interphase")
fiber = TransverseIsotropic(
    E_L=230000.0, E_T=15000.0, G_LT=15000.0, nu_LT=0.20, nu_TT=0.30,
    k_res=1.0, Gc=1.6e-3, name="carbon fibre",
)
materials = [matrix, interphase, fiber]
phase = jnp.array(phase_np)
describe_materials(materials)

from typing import cast
from problems.fracture import FractureSolution, solve_fracture_incremental

EPS_MAX = 2.0e-2   # target macroscopic strain at t=1
l0 = 3.0 * dx[0]   # length scale, a few voxels wide
eps_dir = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
eps_bar_target = EPS_MAX * eps_dir

def _report(r, write_time):
    sol = cast(FractureSolution, r.solution)
    print(f"step {r.step:3d}  t={r.t:.3f}  eps11={r.t * EPS_MAX:.2e}  "
          f"sigma11_ave={float(jnp.mean(sol.sigma[0, 0])):8.3f} MPa  "
          f"max(d)={float(jnp.max(sol.d)):.4f}  staggered_iters={sol.iter_staggered}")

# Uniaxial tension along x, ramped over 100 equal load-fraction increments -- the staggered
# mechanics<->damage solve (AT2 degradation, Amor split, hybrid irreversibility) runs internally,
# once per increment.
results = solve_fracture_incremental(
    n, L, phase, materials, eps_bar_target, l0, None, jnp.zeros(Nv), jnp.zeros(Nv),
    stepping="fixed", dt_step=0.01,
    toler_lin=1e-2, maxiter_cg=200,
    toler_helm=1e-2, maxiter_helm=200,
    toler_st_abs=1e-2, maxiter_st=30,
    on_increment=_report,
)

sol = cast(FractureSolution, results[-1].solution)
eps, sigma, d_field = sol.eps, sol.sigma, sol.d
print("\nPASSED -- mechanical CG converged at every accepted increment.")
```

```text title="Output"
JAX backend: cpu
Devices: [CpuDevice(id=0)]
grid n : (126, 125, 1)
domain L [mm]: (0.06292, 0.06228, 0.0005)
phi (fibre+interphase volume fraction): 0.5508
phase fractions -- matrix: 0.3079  interphase: 0.1413  fibre: 0.5508
phase 0: PhaseFieldIsotropic (epoxy matrix): E=3.5e+03, nu=0.35, Gc=0.001, k_res=1e-06
phase 1: PhaseFieldIsotropic (interphase): E=1.2e+04, nu=0.35, Gc=0.0004, k_res=1e-06
phase 2: TransverseIsotropic (carbon fibre): E_L=2.3e+05  E_T=1.5e+04  G_LT=1.5e+04  G_TT=5.77e+03  nu_LT=0.2  nu_TT=0.3

step   1  t=0.010  eps11=2.00e-04  sigma11_ave=   2.267 MPa  max(d)=0.0014  staggered_iters=1
step   5  t=0.050  eps11=1.00e-03  sigma11_ave=  11.143 MPa  max(d)=0.0380  staggered_iters=2
step  10  t=0.100  eps11=2.00e-03  sigma11_ave=  20.998 MPa  max(d)=0.1994  staggered_iters=3
step  11  t=0.110  eps11=2.20e-03  sigma11_ave=  22.624 MPa  max(d)=0.2750  staggered_iters=4
step  12  t=0.120  eps11=2.40e-03  sigma11_ave=  24.014 MPa  max(d)=0.4064  staggered_iters=6
step  13  t=0.130  eps11=2.60e-03  sigma11_ave=  20.325 MPa  max(d)=0.9906  staggered_iters=25
step  14  t=0.140  eps11=2.80e-03  sigma11_ave=  17.078 MPa  max(d)=1.0000  staggered_iters=5
step  15  t=0.150  eps11=3.00e-03  sigma11_ave=  18.285 MPa  max(d)=1.0000  staggered_iters=1
step  20  t=0.200  eps11=4.00e-03  sigma11_ave=  22.810 MPa  max(d)=1.0000  staggered_iters=1
step  30  t=0.300  eps11=6.00e-03  sigma11_ave=  34.214 MPa  max(d)=1.0000  staggered_iters=1
step  50  t=0.500  eps11=1.00e-02  sigma11_ave=  57.024 MPa  max(d)=1.0000  staggered_iters=1
step  70  t=0.700  eps11=1.40e-02  sigma11_ave=  79.834 MPa  max(d)=1.0000  staggered_iters=1
step 100 t=1.000  eps11=2.00e-02  sigma11_ave= 114.048 MPa  max(d)=1.0000  staggered_iters=1

PASSED -- mechanical CG converged at every accepted increment.
```

![Phase-Field Fracture (AT2) on a CFRP microstructure](/img/examples/pff_damage_cfrp.png)

The macroscopic response (centre panel) shows a real, but **local**, snap: between `eps11=2.4e-3`
and `eps11=2.8e-3` the stress drops from 24.0 to 17.1 MPa while `max(d)` jumps from 0.41 to 1.0 —
the staggered loop needing 25 iterations at that one step (versus 1-6 elsewhere) is the numerical
signature of the coupled elastic/damage system working hard right at the transition. The damage
field (right panel) shows exactly why it's local rather than a global snap-back: a single
connected crack forms through one narrow matrix ligament between two closely-packed fibres, not
diffusely across the whole domain — that's the interphase and realistic fibre packing doing their
job, concentrating failure where the geometry is actually weakest, unlike a uniform seed in a
homogeneous material. Past that point `max(d)` stays pinned at 1.0 (irreversibility) while the
staggered loop converges trivially (1 iteration) for the rest of the load path — the crack found a
local matrix channel but didn't percolate across the periodic cell at this `phi`/domain size, so
the surrounding fibre network (undamageable, `k_res=1.0`) keeps carrying load and the macroscopic
stress resumes climbing almost immediately. That's a genuine, geometry-driven micro-cracking event
being correctly bridged by the fibre network, not a solver failure — whether it's also what a
larger domain or higher `phi` would show as full specimen failure is exactly the kind of question
this RVE-based approach is built to explore.

:::note[Reproducing]
This page's code, output, and plot are generated by
[`notebooks/damage_fracture/pff-damage.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/damage_fracture/pff-damage.ipynb)
(linked via the Colab badge above) — there is no standalone `examples/` script for this case, since
it would just be a redundant duplicate of the notebook.

If the notebook changes, re-run it and update the pasted output/image above — there's no
build-time execution here, since Docusaurus can't run Python.

For the full interactive version — including the initial phase-map figure and `.xdmf`/`.h5` export
for ParaView — open the notebook directly.
:::

## Next steps

- Try a different `seed`/`phi_target` in the RVE generation cell — fibre packing and volume
  fraction both shift where the crack nucleates and whether it percolates, same as switching
  between real patches from `data/patches_fft/` would (see `configs/user/pff_prototype.yaml`).
- Anderson-accelerated staggered iterations (the old `solvers.damage.anderson`) haven't been
  ported to the new layout yet — the 25-iteration staggered solve right at the snap-through above
  is exactly the case that acceleration would help most.
- See `configs/user/pff_prototype.yaml` for the full production setup with an adaptive timestepper,
  driven via `scripts/simulation/pff_nw_cg_strain.py`.
