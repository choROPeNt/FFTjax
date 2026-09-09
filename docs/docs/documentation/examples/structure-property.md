# Structure-Property: Transverse Modulus vs. Fibre Volume Fraction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choROPeNt/FFTjax/blob/main/notebooks/structure-property_phi-sweep.ipynb)

A minimal structure-property study: sweep the fibre volume fraction φ of a random-fibre RVE
(`generation.rve.make_random_composite_rve`,
[Catalanotti 2016](https://doi.org/10.1016/j.compstruct.2015.11.039)), solve for the effective
transverse Young's modulus E(φ) at each φ via `learning.extractors.effective_modulus` (a mixed
strain/stress boundary-condition displacement solve — free lateral surfaces, the same real
tensile-test condition as [Mixed Strain/Stress Boundary Conditions](./lin-elastic-mixed-bc.md)),
then fit a Gaussian-process surrogate E(φ) with `learning.surrogates.GPSurrogate`
([GPJax](https://docs.jaxgaussianprocesses.com/) underneath, Matérn-5/2 kernel).

The GP is refit **inside** the sweep loop, `N_UPDATES` times as data accumulates, rather than
once at the end — at each update it's asked one "simple answer" question: the modulus at a
fixed, never-simulated query φ. The printed output below shows that cheap answer sharpening as
more real FFT solves feed into it, without ever running a new solve at the query point itself.

```python
import jax.numpy as jnp
import numpy as np

from generation.rve import make_random_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from learning.extractors import effective_modulus
from learning.surrogates import GPSurrogate

r_fiber   = 0.005   # mm
vox       = 0.001   # mm -- 5 voxels per fibre radius
size_in_r = 10       # domain side ~ 10*r_fiber (Catalanotti 2016 convention)
seed      = 42       # fixed packing realisation across the sweep

E_matrix, nu_matrix = 3500.0, 0.35    # epoxy matrix
E_fiber,  nu_fiber  = 70000.0, 0.20   # glass fibre
eps0 = 1.0e-3   # small uniaxial tensile strain probe (xx)

N_UPDATES = 5      # how many times the GP gets refit as the sweep progresses
PHI_QUERY = 0.35   # never simulated -- the GP's "simple answer" target

phis        = np.linspace(0.10, 0.60, 12)
phi_grid    = np.linspace(phis.min(), phis.max(), 200)
snapshot_at = sorted(set(np.linspace(3, len(phis), N_UPDATES).round().astype(int)))

phi_obs, E_obs, nu_obs = [], [], []
gp_snapshots = []   # per update: n_obs, mean/std over phi_grid, mean/std at PHI_QUERY

for k, phi in enumerate(phis, start=1):
    phase_np, n, L, phi_act, _ = make_random_composite_rve(
        phi=phi, r_fiber=r_fiber, dx=vox, size_in_r=size_in_r, nz=1, K=15, seed=seed,
    )
    phase = jnp.array(phase_np.reshape(-1))
    materials = [
        LinearElasticIsotropic(E=E_matrix, nu=nu_matrix, name="epoxy matrix"),
        LinearElasticIsotropic(E=E_fiber, nu=nu_fiber, name="glass fibre"),
    ]

    # Mixed-BC uniaxial tension (xx strain-controlled, yy/zz free) -- the true
    # engineering modulus, not the stiffer constrained coefficient a pure-strain
    # BC would give.
    E_val, nu_val, converged = effective_modulus(
        phase, materials, n, L, component=(0, 0), eps0=eps0, toler_lin=1e-6, maxiter=200,
    )
    phi_obs.append(phi_act)
    E_obs.append(E_val)
    nu_obs.append(nu_val[1])

    if k in snapshot_at:
        # Refit the GP on every observation seen so far, then ask it for both
        # a dense curve (for the plot) and the single PHI_QUERY answer, in one
        # predict() call.
        X_query = np.concatenate([phi_grid, [PHI_QUERY]])
        surrogate = GPSurrogate(num_iters=300).fit(phi_obs, E_obs, verbose=False)
        mean_all, var_all = surrogate.predict(X_query)
        std_all = np.sqrt(var_all)
        mean_q, std_q = float(mean_all[-1]), float(std_all[-1])
```

```text title="Output"
JAX backend: cpu
Devices: [CpuDevice(id=0)]
[ 1/12] phi_act=0.100  grid=(151, 157, 1)  E=  4443.1 MPa  nu=0.4493  converged=True
[ 2/12] phi_act=0.146  grid=(125, 130, 1)  E=  4845.7 MPa  nu=0.4613  converged=True
[ 3/12] phi_act=0.191  grid=(109, 114, 1)  E=  5353.0 MPa  nu=0.4674  converged=True
          -> GP update (3 obs): E(phi=0.35) = 4892.8 +/- 760.5 MPa (95%, never simulated)
[ 4/12] phi_act=0.236  grid=(98, 102, 1)  E=  5826.6 MPa  nu=0.4573  converged=True
[ 5/12] phi_act=0.282  grid=(90, 94, 1)  E=  6427.5 MPa  nu=0.4561  converged=True
          -> GP update (5 obs): E(phi=0.35) = 6972.7 +/- 526.7 MPa (95%, never simulated)
[ 6/12] phi_act=0.328  grid=(84, 87, 1)  E=  6942.1 MPa  nu=0.4396  converged=True
[ 7/12] phi_act=0.373  grid=(78, 82, 1)  E=  7622.6 MPa  nu=0.4224  converged=True
[ 8/12] phi_act=0.419  grid=(74, 77, 1)  E=  8389.3 MPa  nu=0.4244  converged=True
          -> GP update (8 obs): E(phi=0.35) = 7316.5 +/- 127.9 MPa (95%, never simulated)
[ 9/12] phi_act=0.465  grid=(70, 73, 1)  E=  9521.1 MPa  nu=0.3937  converged=True
[10/12] phi_act=0.510  grid=(67, 70, 1)  E= 10677.3 MPa  nu=0.3773  converged=True
          -> GP update (10 obs): E(phi=0.35) = 7256.4 +/- 188.6 MPa (95%, never simulated)
[11/12] phi_act=0.555  grid=(64, 67, 1)  E= 12232.7 MPa  nu=0.3498  converged=True
[12/12] phi_act=0.604  grid=(62, 64, 1)  E= 14275.4 MPa  nu=0.3299  converged=True
          -> GP update (12 obs): E(phi=0.35) = 7256.1 +/- 268.3 MPa (95%, never simulated)

PASSED -- every sweep point's mixed-BC solve converged.
```

![Structure-Property: Transverse Modulus vs. Fibre Volume Fraction](/img/structure-property_gp_convergence.png)

Left: every GP snapshot's mean curve over the sweep range (darker = more observations), the
final snapshot's ±2σ band, and the actual FFT solves it was fit on. Right: the GP's answer at
the never-simulated `PHI_QUERY`, plotted against how many real solves it had seen at that
point — a cheap, uncertainty-aware answer that gets more confident with a handful of real FFT
solves, with no solve ever run at the query point itself. The prediction swings widely on 3
observations, then stabilizes to within a few percent by 8.

:::note[Reproducing]
This page's code, output, and plot are generated by
[`examples/structure_property_phi_sweep.py`](https://github.com/choROPeNt/FFTjax/blob/main/examples/structure_property_phi_sweep.py):

```bash
python examples/structure_property_phi_sweep.py
```

If the example changes, re-run the script and update the pasted output/image above — there's no
build-time execution here, since Docusaurus can't run Python.

For the full interactive version — with the tunable `N_UPDATES`/`PHI_QUERY` parameters and the
effective Poisson's ratio also tracked per point — see
[`notebooks/structure-property_phi-sweep.ipynb`](https://github.com/choROPeNt/FFTjax/blob/main/notebooks/structure-property_phi-sweep.ipynb),
linked via the Colab badge above.
:::

## Next steps

- Use the GP's posterior variance to pick the *next* φ to actually simulate (uncertainty
  sampling) instead of an evenly-spaced sweep — exactly `scripts/active_learning.py`'s
  active-learning loop, applied here to φ instead of a VAE latent code.
- Add a second sweep dimension (e.g. `r_fiber`, or the matrix/fibre modulus ratio) --
  `GPSurrogate` accepts a multi-dimensional `X` unchanged, same as `scripts/active_learning.py`'s
  4-D latent-space GP.
- Swap the isotropic glass fibre for `TransverseIsotropic` carbon fibre (as in the
  [Phase-Field Fracture](./phase-field.md) example's RVE) and compare the longitudinal vs.
  transverse modulus sweep.
