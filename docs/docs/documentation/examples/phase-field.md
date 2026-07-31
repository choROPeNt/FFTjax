# Phase-Field Fracture (AT2)

A minimal walkthrough of FFTjax's staggered phase-field fracture solver: the strain-based
Newton-CG elastic solver (`solvers.mechanical.strain_nw_cg.solve_elastic`) coupled with the AT2
Helmholtz damage solver (`solvers.damage.pff_damage.solve_helmholtz_cg`) on a homogeneous,
isotropic cube with a small pre-damaged spherical seed at its centre.

The seed (`d = 0.9` inside a sphere at ~1.5% volume fraction) acts as a stress concentrator, like a
small void or pre-crack. Under increasing macroscopic tensile strain, each load step runs a
staggered loop: solve the mechanical problem with the current damage-degraded stiffness, compute
the tensile (Amor-split) driving force, update the irreversible history variable, then solve the
Helmholtz equation for the updated damage field — repeating until the damage field stops changing.

:::note[Why the seed starts at d = 0.9, not d = 1]
A fully open seed (`d = 1`) is a near-void — about 10⁶× stiffness contrast against the `k_res = 1e-6`
residual stiffness in `degradation()`. The basic (non-accelerated) Lippmann-Schwinger CG scheme
behind `solve_elastic` does not converge at that contrast within a practical iteration budget
(verified: still unconverged after 2000+ CG iterations, with both the `standard` and `rotated`
Green's-operator schemes). Seeding at `d = 0.9` instead (~100× contrast) converges in tens of CG
iterations and still reaches full failure under load.
:::

```python
import jax.numpy as jnp
import numpy as np

from operators.green import build_freq_grid, build_green_operator
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field, lame_from_C_field, strain_energy_amor_split
from solvers.mechanical.strain_nw_cg import solve_elastic
from solvers.damage.pff_damage import degradation, update_history, solve_helmholtz_cg

# Grid and material: a 32^3 voxel unit cube, single-phase brittle isotropic solid.
n = (32, 32, 32)
L = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))
dx = tuple(Li / ni for Li, ni in zip(L, n))

material = LinearElasticIsotropic(E=20e3, nu=0.2, name="brittle solid")
phase = jnp.zeros(Nv, dtype=int)
C_field = assemble_C_field([material], phase)
lam_vox, mu_vox = lame_from_C_field(C_field)

xi_flat = build_freq_grid(n, L)
G_glob = build_green_operator(xi_flat, material.lam, material.mu)

# Pre-damaged spherical seed, Vf ~ 1.5%, d = 0.9 inside (see sphere_seed() in the full script).
d_init = 0.9 * sphere_seed(n, L, vf=0.015)   # 1 inside the sphere, 0 outside

# AT2 parameters.
l0 = 3.0 * dx[0]   # length scale, a few voxels wide
Gc = 2.0e-2        # critical energy release rate

d_field, H_field = d_init, jnp.zeros(Nv)
eps_dir = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

for eps_scale in np.linspace(2.0e-4, 2.4e-3, 8):
    eps_bar = float(eps_scale) * eps_dir
    d_st = d_field
    for _ in range(50):                                        # staggered loop
        d_prev = d_st
        g = degradation(d_st)
        C_eff = g[None, None, None, None, :] * C_field
        eps, sigma, delta, conv = solve_elastic(
            n, C_eff, G_glob, eps_bar, toler_lin=1e-6, maxiter=3000,
        )
        psi_pos, _ = strain_energy_amor_split(eps, lam_vox, mu_vox)
        H_field = update_history(H_field, psi_pos)
        d_st, conv_helm = solve_helmholtz_cg(
            H_field, xi_flat, n, l0, Gc, d_prev, toler_cg=1e-4, maxiter=300,
        )
        if float(jnp.max(jnp.abs(d_st - d_prev))) < 1e-3:
            break
    d_field = d_st
```

```text title="Output"
JAX backend: cpu
Devices: [CpuDevice(id=0)]
LinearElasticIsotropic (brittle solid): E=2e+04, nu=0.2, lam=5.56e+03, mu=8.33e+03
seed voxels: 480 (Vf=0.0146)
step 1/8  eps11=2.00e-04  sigma11_ave=  4.2725 MPa  max(d)=0.9000  staggered_iters= 2
step 2/8  eps11=5.14e-04  sigma11_ave= 10.4737 MPa  max(d)=0.9000  staggered_iters= 2
step 3/8  eps11=8.29e-04  sigma11_ave= 15.4583 MPa  max(d)=0.9000  staggered_iters= 3
step 4/8  eps11=1.14e-03  sigma11_ave= 18.8560 MPa  max(d)=0.9000  staggered_iters= 4
step 5/8  eps11=1.46e-03  sigma11_ave= 20.5955 MPa  max(d)=0.9000  staggered_iters= 5
step 6/8  eps11=1.77e-03  sigma11_ave= 20.6374 MPa  max(d)=0.9000  staggered_iters= 9
step 7/8  eps11=2.09e-03  sigma11_ave=  0.0975 MPa  max(d)=0.9945  staggered_iters=39
step 8/8  eps11=2.40e-03  sigma11_ave=  0.0588 MPa  max(d)=0.9959  staggered_iters= 4

PASSED -- mechanical CG converged at every step, damage stayed in [0, 1], and only grew (irreversibility held).
```

![Macroscopic stress-strain response and final damage field](/img/pff_damage.png)

The macroscopic response is the real signature of brittle AT2 fracture: the average stress rises
elastically, plateaus as the seed drives local damage growth, then **collapses catastrophically**
between `eps11=1.77e-3` and `eps11=2.09e-3` — a classic snap-through, matching the sudden,
unstable crack-propagation behaviour phase-field fracture models are built to capture.

:::caution[Reading the damage field]
The right-hand plot shows damage spread across nearly the whole cross-section rather than a
localized crack emanating from the seed. That's correct for this setup, not a bug: at only ~1.5%
volume fraction, the seed isn't a strong enough geometric stress concentrator to spatially localize
failure in a domain this small under uniform periodic loading — once the surrounding matrix's own
(nearly uniform) driving force exceeds the critical energy release rate, it fails almost everywhere
at once (a "global snap-back"). Producing a visibly localized crack path instead would need either
a much larger domain (so the seed's stress concentration decays before reaching the periodic
images) or an explicit notch/pre-crack geometry with its own tips, rather than a small isolated
seed.
:::

:::note[Reproducing]
This page's code, output, and plot are generated by
[`examples/pff_damage.py`](https://github.com/choROPeNt/FFTjax/blob/main/examples/pff_damage.py):

```bash
python examples/pff_damage.py
```

If the example changes, re-run the script and update the pasted output/image above — there's no
build-time execution here, since Docusaurus can't run Python.
:::

## Next steps

- See `solvers.damage.anderson` for Anderson-accelerated staggered iterations — the plain
  fixed-point staggered loop used here can need many iterations near the snap-through point (see
  `staggered_iters=39` at the failure step above).
- See `solvers.damage.pff_damage.solve_helmholtz_cg_het` for spatially varying fracture toughness
  `Gc(x)` (e.g. a stiffer fibre phase with much higher toughness than the matrix).
- For a case with actual material heterogeneity (not just a damage seed), see the
  [Benchmark](../benchmark.mdx) page's two-phase composite.
