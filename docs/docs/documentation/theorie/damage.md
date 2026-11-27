# Damage & Fracture Solvers

FFTjax implements variational phase-field (AT2) fracture as a staggered solve between the
mechanical equilibrium problem (see [Mechanical Solvers](mechanical)) and a Helmholtz-type damage
sub-problem, both solved by matrix-free CG in Fourier space. `problems/fracture.py` owns both the
wiring (materials → per-voxel fields) and the staggered loop itself — kept out of `solvers/`, which
stays numerics-only with no knowledge of "which degradation law" or "which energy split".

## Variational formulation

$$
\Pi(u,d) = \int_\Omega g(d)\,\psi_e(\varepsilon(u))\,d\Omega
+ \int_\Omega G_c\Big(\frac{d^2}{2\ell} + \frac{\ell}{2}|\nabla d|^2\Big)\,d\Omega
$$

where $d\in[0,1]$ is the phase-field damage variable, $g(d) = (1-k)(1-d)^2 + k$ is the quadratic
AT2 degradation function with a small residual stiffness $k\sim10^{-6}$ (`k_res`, per-material) for
numerical conditioning, $G_c$ is the critical energy release rate (may be homogeneous or spatially
heterogeneous), and $\ell$ is the length scale. Only the tensile part $\psi_e^+$ of the strain
energy drives damage growth (Amor et al. 2009 volumetric-deviatoric split,
`materialmodels/phasefield/splits.py`):

$$
\psi^+ = \tfrac12 K\langle\mathrm{tr}\,\varepsilon\rangle_+^2 + \mu\,\varepsilon_{\text{dev}}\!:\!\varepsilon_{\text{dev}},
\qquad \psi^- = \tfrac12 K\langle\mathrm{tr}\,\varepsilon\rangle_-^2, \qquad K=\lambda+\tfrac23\mu
$$

All deviatoric energy is assigned to $\psi^+$ — a crack shouldn't heal or grow under compression,
so only the volumetric part distinguishes tension from compression; $\psi^++\psi^-$ reproduces the
ordinary elastic energy exactly.

**Two ways to apply degradation.** `degrade_stiffness_field` (`materialmodels/phasefield/
degradation.py`) applies $g(d)$ uniformly to an already-assembled stiffness field,
$\mathbb{C}_{\text{eff}} = g(d)\mathbb{C}$ — simple, but this does *not* reproduce the Amor split
at the stress level, since the compressive volumetric branch gets degraded too.
`PhaseFieldIsotropic` (`materialmodels/phasefield/isotropic.py`) instead derives stress and tangent
by automatic differentiation of one energy density,
$\psi(\varepsilon,d) = g(d)\,\psi^+(\varepsilon) + \psi^-(\varepsilon)$ — $\sigma =
\partial\psi/\partial\varepsilon$ (`jax.grad`), $\mathbb{C}_{\text{tan}} =
\partial\sigma/\partial\varepsilon$ (`jax.jacfwd` on that) — so the compressive branch is
*structurally* undegraded (it never sees $g(d)$) rather than depending on two independently
maintained formulas staying in sync. `materialmodels.assembly.assemble_pff_local_update` picks
whichever a material provides (duck-typed on a `psi_split` attribute), so a `materials` list can
freely mix both kinds.

Crack irreversibility is enforced through a monotone history variable. The hybrid formulation
(Steinke & Kaliske 2019, adopted by Schneider & Kästner 2025, doi:10.1111/ffe.14553) only locks in
monotonicity once a point has effectively cracked ($d\ge d_{\text{thres}}$, default 0.95):

$$
H = \begin{cases} \psi^+ & d_{\text{prev}} < d_{\text{thres}} \quad\text{(unrestricted)}\\
\max(H_{\text{prev}}, \psi^+) & d_{\text{prev}} \ge d_{\text{thres}} \quad\text{(irreversible)}
\end{cases}
$$

so the pre-crack process zone can still relax instead of being over-widened by premature history
locking.

## Staggered solution scheme

`problems.fracture.solve_fracture` alternates the two Euler-Lagrange sub-problems of $\Pi(u,d)$
until the coupled system converges ($\max|d_{\text{new}}-d_{\text{old}}|$ below an absolute or
relative tolerance):

1. **Elastic step** — the degraded tangent $\mathbb{C}_{\text{eff}}$ is evaluated at the
   *previous* staggered iteration's strain $\varepsilon_{\text{prev}}$, not the current (still
   unknown) one — a frozen-tangent scheme. A material with a nonlinear-in-$\varepsilon$ degraded
   stress (`PhaseFieldIsotropic`'s Amor split) would otherwise need its own nonlinear Newton-CG
   sub-solve at every staggered iteration; freezing the tangent keeps the mechanical step a plain
   linear CG solve (Lippmann-Schwinger, Fourier-Galerkin, or displacement-based — the same three
   formulations from [Mechanical Solvers](mechanical), including mixed strain/stress BC), and the
   loop still converges to the fully self-consistent solution at staggered convergence, where
   $\varepsilon_{\text{prev}} = \varepsilon$.
2. **Phase-field step** — a Helmholtz-type equation solved by preconditioned CG entirely in
   Fourier space (`solvers/elliptic/scalar.py`).

## Homogeneous vs. heterogeneous damage solve

Stationarity of $\Pi$ with respect to $d$, using that $-g'(d) = (1-k)(2-2d)$ is *affine* in $d$
(what makes this a closed-form linear CG solve rather than a per-voxel Newton iteration):

**Homogeneous $G_c$** (`solve_damage_helmholtz_cg`) — a single scalar diffusion coefficient:

$$
\Big(\frac{G_c}{\ell} + 2(1-k)H\Big)d - G_c\ell\,\Delta d = 2(1-k)H
$$

**Heterogeneous $G_c(\mathbf{x})$** (`solve_damage_helmholtz_cg_het`) — genuinely a different
equation, not the same one at more generality: the variational derivative of the
$G_c(\mathbf{x})\frac{\ell}{2}|\nabla d|^2$ term is $\nabla\!\cdot\!(G_c(\mathbf{x})\nabla d)$, not
$G_c(\mathbf{x})\Delta d$ (product rule — an extra $\nabla G_c\cdot\nabla d$ term wherever $G_c$
varies, including at a sharp phase interface):

$$
\Big(\frac{G_c(\mathbf{x})}{\ell} + 2(1-k)H\Big)d - \ell\,\nabla\!\cdot\!(G_c(\mathbf{x})\nabla d) = 2(1-k)H
$$

Plugging a per-voxel $G_c(\mathbf{x})$ pointwise into the homogeneous operator's $G_c\Delta d$ term
would silently solve the wrong equation. The heterogeneous solve costs 3 FFT round trips per CG
matvec (gradient, real-space multiply, divergence) instead of 1, and needs `nyquist_safe_xi` (its
gradient/divergence are odd powers of $\boldsymbol\xi$, unlike the homogeneous operator's even-power
Laplacian). `problems.fracture.solve_fracture` dispatches between the two automatically — a
per-voxel $G_c$ that happens to be uniform collapses to the cheaper homogeneous solver.

An optional viscous term $\eta/\Delta t$ adds resistance to rapid damage growth: without it,
damage snap-through is instantaneous and spatially diffuse; with $\eta>0$ the crack grows
gradually from the notch tip. Irreversibility is enforced post-solve, $d = \max(d_{\text{prev}},
d_{\text{CG}})$, and at $k_{\text{res}}=1$ (a damage-immune phase) both driving terms vanish
exactly, matching $g'(d)\equiv0$.

## Accelerating the staggered fixed point

Plain (Picard) staggering is a fixed-point iteration $x_{k+1} = G(x_k)$ that converges slowly —
and can stall — near the brittle snap-through, where the elastic and damage sub-problems become
strongly coupled. `solvers/damage/anderson.py` provides two drop-in accelerators that reuse the
same $(x_k, G(x_k))$ pairs the staggered loop already produces:

- **Anderson mixing** (Walker & Ni 2011, Type-II) solves an *unconstrained* least-squares problem
  on windowed differences of the residual $f_k = G(x_k)-x_k$, then mixes the next iterate as
  $x_{k+1} = x_k + \beta f_k - (\Delta X + \beta\Delta F)\gamma$, damped by $\beta \in (0,1]$ and
  Tikhonov-regularized against near-collinear residuals late in the snap-through.
- **NGMRES** (Nonlinear GMRES) instead solves a *constrained* least-squares problem directly on
  the residuals, $\alpha = \arg\min\lVert F\alpha\rVert^2$ subject to $\mathbf{1}^\top\alpha=1$,
  and combines past map values as $x_{k+1} = G\alpha$. This matches DAMASK's outer
  `-snes_type ngmres` solver (Schneider & Kästner 2025, doi:10.1111/ffe.14553); the sum-to-1
  constraint stabilizes the solve when residuals become nearly collinear near snap-through and
  guarantees exact recovery of any fixed point $x^* = G(x^*)$.

Both accelerators cost one staggered sweep per outer iteration — identical to plain Picard — with
no extra function evaluations, and share the same `reset()`/`step()` API so either can be dropped
into the staggered loop interchangeably. `problems/fracture.py` selects one via the
`accelerator: anderson | ngmres | none` config key (`none` is the default and is bit-for-bit the
un-accelerated loop; depth 1 is also exactly Picard).

### What they buy, measured

On a linear fixed point with a known solution, sweeps to a $10^{-10}$ residual — note the speedup
*grows* with difficulty, which is what matters, since a staggered loop is cheap when it is easy
(2 sweeps) and ruinous when it is hard (44):

| spectral radius | Picard | Anderson (depth 5) | NGMRES (depth 5) |
| --- | --- | --- | --- |
| 0.97 | 610 | 130 (4.7×) | 130 (4.7×) |
| 0.995 | 3699 | 600 (6.2×) | 600 (6.2×) |

Through the real coupled solver, one increment pre-severance: Picard 11 sweeps, either
accelerator 4 sweeps (2.75×), reaching the same state to $6\times10^{-5}$ MPa in stress and
$2.8\times10^{-6}$ in damage — a shorter path to the same fixed point.

### Three details that are not in the textbook version

- **Clamping.** The iterate is a *damage* field, and an extrapolated step is a linear combination
  with possibly negative weights, so it can leave $[0,1]$ or move damage backwards. Every iterate
  is clamped to $[0,1]$ and to an irreversibility floor (the increment's starting damage).
  Without the floor, acceleration can un-crack voxels.
- **The safeguard is retrospective.** Anderson has no descent guarantee, so an accelerated step
  is judged by the residual the *next* sweep actually produces: if
  $\lVert f_{k+1}\rVert > \texttt{safeguard\_factor}\,\lVert f_k\rVert$ the history is flushed
  and the step falls back to Picard. This costs nothing, since that sweep happens anyway. A
  *prospective* test on step length is wrong — an accelerated step is supposed to travel further
  than the Picard step, and such a rule rejects precisely the steps that work (measured: 497
  rejections in 499 steps, zero speedup).
- **Convergence is measured on the Picard residual** $\lVert G(d)-d\rVert$, not on the
  accelerated step $\lVert d_{k+1}-d_k\rVert$. They coincide without an accelerator, but an
  extrapolated step is deliberately larger, so testing the step would make a converging run look
  divergent.

:::note[Reset once per load increment]
The fixed point moves with the applied strain, so history from the previous increment
extrapolates toward the previous solution. `problems/fracture.py` calls `reset(floor=d_init)` at
the start of every increment.
:::

---

## Fracture of an elastoplastic matrix

The formulation above assumes the bulk is elastic up to failure. A polymer matrix is not: it
yields — in a markedly pressure-sensitive way — well before it cracks, so a brittle matrix
over-predicts its strength and mis-places the crack.
`materialmodels/phasefield/drucker_prager.py` couples AT2 damage to the Drucker-Prager
elastoplastic model of [Mechanical Solvers](mechanical) through the **effective stress**.

### Effective-stress coupling

The plastic return mapping runs in *undegraded* stress space, exactly as it does without
damage, and $d$ then scales what the material transmits:

$$
\boldsymbol\sigma_{\text{eff}} = \mathbb{C} : (\boldsymbol\varepsilon - \boldsymbol\varepsilon_p),
\qquad
\boldsymbol\sigma = g(d)\left(\mathbf{s}_{\text{eff}} + \langle p_{\text{eff}}\rangle_+\mathbf{I}\right)
 + \langle p_{\text{eff}}\rangle_-\mathbf{I}
$$

with $\mathbf{s}_{\text{eff}}$ and $p_{\text{eff}}$ the deviatoric and mean parts of
$\boldsymbol\sigma_{\text{eff}}$. This is Amor's split applied at the stress level: everything
degrades *except* the compressive hydrostatic branch, so a fully cracked voxel still carries
compression rather than interpenetrating. The yield surface never sees $d$, which is what lets
the return mapping — cone, apex, tip-rounded and all of their guards — be reused unchanged.

Degrading the yield stress as well ($\sigma_y \to g(d)\sigma_y$, so a cracking voxel also loses
strength) is the fuller coupling and is **not** implemented.

### Driving force

$$
\mathcal{H} = \psi_e^+(\boldsymbol\varepsilon - \boldsymbol\varepsilon_p)
            + \beta_p \, W_p ,
\qquad
W_p = \int \boldsymbol\sigma_{\text{eff}} : \mathrm{d}\boldsymbol\varepsilon_p
$$

Two parts, with $\beta_p$ selecting the physics:

- $\beta_p = 0$ — a **brittle crack in a plastic solid**. Plasticity affects fracture only by
  capping the elastic energy that can be stored, which blunts the crack. Note $\psi_e^+$ is
  evaluated on the *elastic* strain: plastic strain stores no energy, and using the total
  strain would let plastic flow drive cracking through the back door.
- $\beta_p > 0$ — **ductile** fracture: accumulated plastic work also drives the crack, so a
  voxel that has yielded heavily cracks sooner. $\beta_p$ trades off directly against $G_c$, so
  re-calibrate $G_c$ if you change it.

$W_p$ is accumulated as $\boldsymbol\sigma_{\text{eff}}:\mathrm{d}\boldsymbol\varepsilon_p$,
which is exact for associated *and* non-associated flow — the closed form
$\int\sigma_y\,\mathrm{d}\alpha$ is not, since
$\boldsymbol\sigma:\mathrm{d}\boldsymbol\varepsilon_p =
\mathrm{d}\lambda\,(\sigma_y + 3(a_g-a_f)p)$ and the two agree only at $a_g = a_f$.

$\mathcal{H}$ enters the damage sub-problem exactly where $\psi^+$ did, so the Helmholtz solve
is structurally unchanged.

### What this costs numerically

The mechanical sub-problem is now genuinely nonlinear and path-dependent, so the staggered loop
switches from one linear CG solve to a Newton iteration
(`solve_displacement_based_nonlinear`); `problems/fracture.py` selects this automatically when a
coupled material is present, which also means `formulation: displacement` is required and the
run can no longer be `vmap`ped over several RVE realizations.

Two consequences worth knowing before trusting a result:

- **The tangent is non-symmetric** wherever a voxel is both damaged and plastic, and
  increasingly so with $d$. For *associated* flow, where the Drucker-Prager tangent alone is
  exactly symmetric, the relative major asymmetry $|C_{ijkl}-C_{klij}|/|C|$ measures
  $3\times10^{-16}$, $5\times10^{-2}$, $2\times10^{-1}$ and $3\times10^{-1}$ at
  $d = 0,\,0.1,\,0.5,\,0.9$, while a purely elastic state stays symmetric to $10^{-16}$ at any
  $d$. The Amor operator $g\mathbf{P}_+ + \mathbf{P}_-$ commutes with the isotropic elastic
  tangent but not with the elastoplastic one's $\mathbf{s}\otimes\mathbf{s}$ structure. CG is
  applied to it unsymmetrized, which is formally outside CG's assumptions.
- **Results past the softening branch are unverified.** Newton converges cleanly through the
  load, the peak and the softening. Once the transmitted stress is three to four orders below
  peak it stops converging and can return a small *negative* macroscopic stress under applied
  tension. The cause is not established — severance, the $\sim10^6$ stiffness contrast, the
  preconditioner choice and plastic-strain-driven compressive strutting were each tested and
  ruled out. The mechanical failure is folded into the increment's convergence flag, so the
  load stepper in `automatic` mode cuts back on it rather than accepting a non-equilibrium
  state.

- **Plastic strain inside a crack is not meaningful.** A fully cracked voxel keeps its effective
  stress on the yield surface, so $\boldsymbol\varepsilon_p$ and $\alpha$ go on growing while it
  transmits almost nothing. This is inherent to effective-stress ductile phase-field; mask on
  damage before homogenizing a plastic strain.
