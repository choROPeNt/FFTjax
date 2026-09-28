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
