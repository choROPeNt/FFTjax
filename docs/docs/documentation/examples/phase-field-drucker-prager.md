# Phase-Field Fracture of a Plastic Matrix

The companion to [Phase-Field Fracture (AT2)](phase-field), with the brittle solid swapped for a
Drucker-Prager elastoplastic one coupled to AT2 damage through the **effective stress**
(`materialmodels.phasefield.drucker_prager.PhaseFieldDruckerPrager`). See
[Damage & Fracture](../theorie/damage#fracture-of-an-elastoplastic-matrix) for the formulation.

Same geometry as the brittle example — a homogeneous cube with a small pre-damaged spherical seed
at its centre, pulled in uniaxial tension — run three ways on one axis:

| case | meaning |
| --- | --- |
| `brittle` | `PhaseFieldIsotropic`: no plasticity at all |
| `plastic, beta_p = 0` | a brittle crack in a plastic solid — yielding blunts the crack by capping the elastic energy it can store |
| `plastic, beta_p = 2` | ductile — accumulated plastic work *also* drives the crack |

```python
import jax.numpy as jnp
import numpy as np

from materialmodels.phasefield.drucker_prager import PhaseFieldDruckerPrager
from problems.fracture import solve_fracture

n, L = (20, 20, 20), (1.0, 1.0, 1.0)
Nv = int(np.prod(n))
l0 = 4.0 * L[0] / n[0]

material = PhaseFieldDruckerPrager(
    E=3.76e3, nu=0.39,          # epoxy elastic constants
    Gc=2.0e-2,                  # critical energy release rate
    sigma_y0=56.1, H=1.0e3,     # linear isotropic hardening
    a_f=0.13, a_g=0.05,         # friction / dilatancy (non-associated)
    beta_p=0.0,                 # 0 = brittle-in-plastic, > 0 = ductile
)

# formulation="displacement" is REQUIRED: a stateful phase-field material makes
# the mechanical sub-problem nonlinear, so solve_fracture switches to its Newton
# sub-solve, and the lippmann_schwinger reference-medium form has no nonlinear path.
sol = solve_fracture(
    n, L, phase, [material], eps_bar, l0, None, d, H,
    formulation="displacement",
    toler_nr=1e-7, maxiter_nr=40,     # Newton (mechanical)
    state_init=state,                 # (eps_p, alpha, W_p) from the previous increment
)
d, H, state = sol.d, sol.H, (sol.eps_p, sol.alpha, sol.W_p)
```

Run it with:

```bash
python examples/pff_dp_damage.py
```

## Result

![Phase-field fracture of a Drucker-Prager matrix](/img/pff_dp_damage.png)

```
  brittle                peak=   8.234 MPa   final=   0.128 MPa   d_max=0.9942   alpha_max=0.000e+00
  plastic, beta_p=0      peak=   8.234 MPa   final=   0.630 MPa   d_max=0.9970   alpha_max=3.581e-01
  plastic, beta_p=2      peak=   8.234 MPa   final=   0.305 MPa   d_max=0.9987   alpha_max=7.189e-02
```

## Reading it: where the curves differ

All three share the **same peak**, 8.234 MPa at $\bar\varepsilon_{11} = 0.2\%$. That is not a
null result — it is two things at once.

First, it is a **check**: while everything is still elastic, the two plastic models must reduce
exactly to the brittle one, and they agree to four significant figures.

Second, it says which regime this configuration is in. The seed cracks the cube at ~8 MPa while
$\sigma_{y0}$ is 56.1 MPa, so fracture wins the race against yield and nothing has yielded at the
peak (`alpha_max` is still 0 there). Plasticity therefore acts entirely on the **post-peak**
branch — and in opposite directions:

- `beta_p = 0` softens **more slowly** than brittle (final 0.630 vs 0.128 MPa). Plastic
  dissipation around the crack tip holds residual load that the brittle solid simply sheds.
- `beta_p = 2` softens **faster again** (0.305 MPa). Plastic work now feeds the driving force, so
  the crack grows on its own dissipation.

Whether plasticity moves the *peak* is a competition between $G_c$ and $\sigma_{y0}$, not a
property of the model. Raise $G_c$ (tougher) or lower $\sigma_{y0}$ (weaker) until yield precedes
cracking and the peak shifts too. This configuration deliberately shows the other regime — the
one a stiff, tough matrix with a pre-existing flaw is actually in.

:::caution[The `beta_p = 2` curve stops at 0.4%]
The example truncates each curve at the first increment whose mechanical (Newton) solve does not
converge, rather than plotting it. Past the softening branch the solve stops converging and
returns a small *negative* macroscopic stress under applied tension — not an equilibrium state,
so there is nothing behind the number.

The cause is not established. Severance, the $\sim 10^6$ stiffness contrast, the preconditioner
choice, and plastic-strain-driven compressive strutting were each reproduced in isolation and
each **ruled out**. The tangent's non-symmetry (up to ~30% once a voxel is both damaged and
plastic — see the [theory page](../theorie/damage#what-this-costs-numerically)) is a plausible
contributor but has not been shown to be the cause. Treat results past the softening branch as
unverified; `problems.fracture` folds the mechanical failure into the increment's convergence
flag, so a load stepper in `automatic` mode cuts back on it.
:::

:::note[`alpha` inside a crack is not a material strain]
`alpha_max` reads 0.358 for `beta_p = 0` — far above anything physical. A fully cracked voxel
keeps its *effective* stress on the yield surface, so its plastic strain goes on accumulating
while it transmits almost nothing. This is inherent to effective-stress ductile phase-field. Mask
on damage before homogenizing any plastic strain.
:::

## Config-driven equivalent

`configs/simulation/fracture_drucker_prager_example.yaml` runs the same physics on a real
microstructure through `scripts/simulation/solve_fracture.py`:

```yaml
materials:
  - model: phasefield_drucker_prager
    E: 3.76e3
    nu: 0.39
    sigma_y0: 56.1
    H: 1.0e3
    a_f: 0.13
    a_g: 0.05
    Gc: 6.4e-4
    beta_p: 0.0
    name: epoxy matrix (Drucker-Prager + phase field)

formulation: displacement     # required
toler_nr: 1.0e-7
maxiter_nr: 40
stepping:
  mode: automatic             # cuts back on a failed mechanical solve
```

It writes `strain_p` (= $\alpha$) and `plastic_work` (= $W_p$) per increment alongside the usual
damage and stress fields, plus the `eps_p` tensor behind
`write_plastic_strain_tensor: true` — the same field names
`scripts/simulation/solve_inelastic.py` uses, so `scripts/postprocessing/homogenize.py` picks
them up unchanged.
