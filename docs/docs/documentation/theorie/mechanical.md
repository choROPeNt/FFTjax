# Mechanical Solvers

FFTjax provides three interchangeable elastic formulations (`solvers/elliptic/vector/`), a shared
scheme for mixed strain/stress macroscopic boundary conditions, and a Newton-CG extension to
nonlinear (plastic) local constitutive laws. `problems/mechanics.py` is the thin wiring layer that
picks a formulation, assembles the per-voxel stiffness, and calls the corresponding solver.

## Three formulations, one problem

All three solve the same periodic equilibrium problem
$\nabla\cdot(\mathbb{C}(\mathbf{x}):\varepsilon) = 0$ under a prescribed macroscopic strain
$\bar\varepsilon$; they differ in which frequency-space operator (see
[Frequency-Space Operators](operators)) carries the CG solve and whether a reference medium is
involved. All three are exposed as `ElasticitySolver` subclasses (`LippmannSchwingerSolver`,
`FourierGalerkinSolver`, `DisplacementBasedSolver`) with a common `.solve(C_field, eps_bar,
stress_goal) -> ElasticitySolution` interface, selected in `problems.mechanics.solve_mechanics` via
`formulation="lippmann_schwinger" | "fourier_galerkin" | "displacement"`.

### Lippmann-Schwinger

`lippmann_schwinger.py` (Moulinec-Suquet / Vondřejc et al. 2014). The unknown is the periodic
strain fluctuation $\Delta\varepsilon$, driven through a fixed reference-medium Green's operator
$\hat\Gamma_0$:

$$
A(\Delta\varepsilon) = \Gamma_0(\mathbb{C}:\Delta\varepsilon), \qquad
b = -\Gamma_0(\mathbb{C}:\varepsilon_0 - \sigma_{\text{goal}})
$$

$\hat\Gamma_0$'s even-power-of-$\boldsymbol\xi$ construction (see
[Frequency-Space Operators](operators)) is what makes $A$ provably symmetric positive-definite for
arbitrarily heterogeneous $\mathbb{C}(\mathbf{x})$ — no extra preconditioning needed beyond the
reference medium itself.

### Fourier-Galerkin

`fourier_galerkin.py` (Vondřejc et al. 2014). Uses the reference-medium-free Galerkin projector
$\hat G_s$ instead of $\hat\Gamma_0$ — no reference medium anywhere in the solve. Otherwise
structurally identical to the mixed-BC Lippmann-Schwinger route (see [Mixed strain/stress boundary
conditions](#mixed-strainstress-boundary-conditions) below): both dispatch to the same shared
DC-bin-identity solve, only the operator passed in differs.

### Displacement-based

`displacement_based.py`. The unknown is the periodic displacement fluctuation $\hat u$ itself,
with $\hat\varepsilon = \mathrm{sym}(i\boldsymbol\xi\otimes\hat u)$ and the **true** (possibly
heterogeneous) tangent $\mathbb{C}(\mathbf{x})$ applied directly — no reference-medium
approximation. Because $\boldsymbol\xi$ appears to an odd power here, the Nyquist bin must be
zeroed (`nyquist_safe_xi`, see [Frequency-Space Operators](operators)). Convergence for realistic
(high-contrast) composites needs a frequency-domain preconditioner
$(\boldsymbol\xi\cdot\mathbb{C}_0\cdot\boldsymbol\xi)^{-1}$ built from a reference stiffness
$\mathbb{C}_0$ (default: the voxel-mean of $\mathbb{C}$).

## Mixed strain/stress boundary conditions

All three formulations can prescribe macroscopic strain on some tensor components and macroscopic
stress on others (e.g. a free-lateral-surface uniaxial-stress test), selected via a `(3,3)`
`control` mask (`solvers/elliptic/vector/mixed_bc.py`). Two independent schemes exist:

- **DC-bin identity patch** — a single CG solve, shared by all three formulations. The operator's
  tensor `G` is exactly zero at the DC (zero-frequency) bin by construction; overwriting that one
  entry with a control-masked identity lets a stress-controlled component's target pass straight
  through unprojected, turning it into a genuine solved-for degree of freedom, while leaving every
  other frequency (including Nyquist) untouched. This degenerates bit-for-bit to the plain
  pure-strain solve when `control` is all-zero. `solve_mixed_bc_dc_identity` implements this once,
  shared by both **Lippmann-Schwinger** (Kabel et al. 2016, `solve_lippmann_schwinger_dc`) and
  **Fourier-Galerkin** (Lucarini & Segurado 2019a) — only which operator is passed in differs. The
  **displacement-based** solver uses the same DC-bin idea directly: the macroscopic-strain
  correction on stress-controlled directions is embedded live in the CG unknown's own DC-frequency
  mode, solved jointly with the displacement fluctuation through one bordered (fluctuation +
  macroscopic-correction) system.
- **Outer iterative correction** (Michel et al. 1999) — `solve_lippmann_schwinger_mixed_bc`, the
  primary mixed-BC route for Lippmann-Schwinger since it leaves the base (unmodified,
  well-tested) reference-medium solve untouched. Repeatedly calls the plain solve, correcting
  $\bar\varepsilon$ between calls via a small $(K{\le}6)$-dimensional Newton-like step built from
  the active stress-controlled directions' sensitivity matrix $M = \langle\mathbb{C}_0\rangle$.
  For a homogeneous material this converges in one outer iteration (the sensitivity is then
  exact); for a heterogeneous one it needs several. This is an **undamped** fixed-point step: a
  poor $\mathbb{C}_0$ can make it diverge rather than merely converge slowly. In particular,
  $\mathbb{C}_0$ defaults to the true (anisotropic) voxel-mean $\langle\mathbb{C}\rangle$, *not*
  the reference-medium's own isotropized $(\lambda_0,\mu_0)$ — isotropizing away a genuinely
  anisotropic material's directional stiffness (e.g. a transversely isotropic fiber with
  $E_L/E_T\sim15\times$) makes $M$ a poor enough Jacobian approximation for this undamped
  correction to blow up geometrically instead of converging, verified on a carbon-fiber/epoxy RVE
  under mixed BC.

## Material models

`materialmodels/elastic/` implements `ConstitutiveModel`'s `elastic_stiffness_tensor()`:

- **`LinearElasticIsotropic`** — the standard $(\lambda,\mu)$ isotropic tensor from $(E,\nu)$.
- **`TransverseIsotropic`** — a fiber-reinforced material with 5 independent constants
  ($E_L, E_T, G_{LT}, \nu_{LT}$, plus one of $\nu_{TT}$/$G_{TT}$ — not independent for an
  isotropic transverse plane). The reference fiber axis is local $Z$; `fiber_dir` rotates the
  reference-frame stiffness into the global frame via `materialmodels.tensors.rotate_tensor4`, and
  accepts either one direction or a per-voxel orientation field (e.g. from TexGen VTU import), in
  which case `elastic_stiffness_tensor()` returns an already-rotated `(3,3,3,3,Nv)` field that
  `materialmodels.assembly.assemble_C_field` mixes transparently with constant-tensor materials in
  the same phase list.

Both carry `k_res` (AT2 residual stiffness) and `Gc` (critical energy release rate) — read only by
fracture problems (see [Damage & Fracture Solvers](damage)); elasticity-only solves ignore them.

## Nonlinear extension: J2 and Drucker-Prager plasticity

`problems.mechanics.solve_displacement_based_nonlinear` generalizes the displacement-based solve
from a one-shot linear CG solve into a genuine outer Newton loop for any nonlinear local
constitutive law exposing a `stress_and_tangent_field(eps, ...)` method — currently
`materialmodels.inelastic.plasticity_j2.J2Plasticity` and
`materialmodels.inelastic.plasticity_drucker_prager.DruckerPrager`. Both derive their consistent
tangent by forward-mode autodiff (`jax.jacfwd`) on a single closed-form `stress()` return mapping,
rather than a hand-derived formula — see `notes/AUTODIFF_CONSTITUTIVE.md` for why, and the
recurring gotchas both files guard against (strain symmetrization inside `stress()`; a
`jnp.where`-discarded branch must still evaluate to something finite, or it poisons the *kept*
branch's gradient).

**J2 (von Mises), linear isotropic hardening** — closed-form radial return (no local Newton
solve, since the yield function is linear in the plastic multiplier):

$$
\sigma_{\text{trial}} = \lambda\,\mathrm{tr}(\varepsilon-\varepsilon_p^n)\mathbf I + 2\mu(\varepsilon-\varepsilon_p^n),
\quad q_{\text{trial}} = \sqrt{\tfrac32}\lVert s_{\text{trial}}\rVert,
\quad \Delta\gamma = \frac{\max(q_{\text{trial}} - \sigma_{y0} - H\alpha^n,\,0)}{3\mu+H}
$$

**Drucker-Prager** — a pressure-sensitive, optionally non-associated generalization
(drop-in replacement: same state convention, same call surface; $a_f{=}a_g{=}0$ reduces to J2
exactly). The yield surface $f = q + 3a_f p - \sigma_y(\alpha)$ is a cone in principal-stress
space, so a return mapping needs **two** branches — a smooth-cone return (J2's radial return plus
a volumetric correction) and an apex/vertex return for trial states that would otherwise overshoot
into $q<0$ (a region J2's cylinder never reaches). At the sharp vertex the apex return sets the
deviator to exactly zero, which makes the consistent tangent's deviatoric block — and hence the CG
operator inside the outer Newton loop — singular; a rounded hyperbolic tip
($f=\sqrt{q^2+a_{\text{tip}}^2}+3a_fp-\sigma_y$, solved for by a bracketed safeguarded Newton
iteration on the single scalar unknown $q$) trades an exact vertex for a small but strictly
positive deviatoric stiffness, keeping the linearized CG solve well-posed under confined,
pressure-sensitive loading. Non-associated flow ($a_g \ne a_f$) makes the tangent non-major-
symmetric by construction — expected, not a defect — which puts CG formally outside its usual
assumptions; in practice the asymmetry is a modest perturbation of a strongly SPD operator and
Newton still converges.

**Outer Newton-CG loop.** At each Newton iteration $k$, the residual is the *true* nonlinear
divergence of $\sigma(\varepsilon_k)$ (not a linearized approximation), while the inner CG solve
linearizes against the current consistent tangent $\mathbb{C}_{\text{algo}}(\varepsilon_k)$ — true
Newton, not a fixed-point/secant scheme. Mixed strain/stress BC folds in the same bordered-system
trick as the linear displacement-based solver (the macroscopic strain correction on
stress-controlled directions is its own Newton unknown, accumulated across iterations alongside
the fluctuation). One invariant is load-bearing for path dependence: the plastic state
$(\varepsilon_p,\alpha)$ must stay **frozen** at the last converged increment throughout every
Newton iteration of the current one — advancing it per iteration lets plastic flow ratchet up once
per iteration and the solve can converge to a garbage fixed point under a large enough confined
load, with the inner CG reporting success the entire time.
