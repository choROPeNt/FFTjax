# 💥 Damage & Fracture Solvers

FFTjax implements variational phase-field (AT2) fracture (`solvers/damage/`) as a staggered
solve between the mechanical equilibrium problem (see [Mechanical Solvers](mechanical)) and a
Helmholtz-type damage sub-problem, both solved by matrix-free CG in Fourier space.

## Variational formulation

The solver is based on an energy minimization principle. For elasticity coupled to phase-field
fracture, the total energy functional reads

$$
\Pi(u, d) =
\int_\Omega g(d)\,\psi_e(\varepsilon(u)) \, d\Omega
+
\int_\Omega G_c \left(
\frac{d^2}{2\ell} + \frac{\ell}{2} |\nabla d|^2
\right) d\Omega
$$

where

- $u$ is the displacement field
- $d$ is the phase-field variable ($d=0$ intact, $d=1$ fully broken)
- $g(d) = (1-k)(1-d)^2 + k$ is the quadratic degradation function, with a small residual
  stiffness $k \sim 10^{-6}$ kept for numerical conditioning rather than $g(1)=0$ exactly
- $G_c$ is the fracture toughness (spatially heterogeneous fields are supported)
- $\ell$ is the length scale parameter

Only the tensile part of the strain energy $\psi_e^+$ drives damage growth (Miehe spectral split
or Amor volumetric/deviatoric split, `mat_models/elastic.py`), and crack irreversibility is
enforced through a monotone history variable $H = \max(H_{\text{prev}}, \psi_e^+)$. A hybrid
variant (Steinke & Kaliske 2019, as adopted by Schneider & Kästner 2025,
doi:10.1111/ffe.14553) only locks in that monotonicity once a point has effectively cracked
($d \ge d_{\text{thres}}$, default 0.95); below the threshold the driving force stays
unrestricted so the pre-crack process zone can still relax instead of being over-widened by
premature history locking.

---

## Staggered solution scheme

A variational staggered scheme alternates between the two Euler-Lagrange sub-problems of
$\Pi(u,d)$:

1. **Elastic step** — minimize $\Pi(u, d^{n})$ with respect to $u$ at fixed damage, i.e. solve
   the mechanical equilibrium problem with the degraded stiffness $g(d^n)\,\mathbb{C}$ using the
   Newton-CG solvers described in [Mechanical Solvers](mechanical):

$$
A_u \, u = b_u
$$

2. **Phase-field step** — minimize $\Pi(u^{n+1}, d)$ with respect to $d$ at fixed displacement.
   This is a Helmholtz-type equation, solved by preconditioned CG entirely in Fourier space
   (`solve_helmholtz_cg` for homogeneous $G_c$, `solve_helmholtz_cg_het` for spatially varying
   $G_c$):

$$
A_d \, d = b_d
$$

Both $A_u$ and $A_d$ are matrix-free spectral operators, so each sub-problem reuses the same
FFT-based CG machinery as the mechanical solvers rather than needing a separate discretization.
The two fields are updated alternately, $d^n \to u^{n+1} \to d^{n+1} \to \dots$, until the
coupled system converges.

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
into the staggered loop interchangeably.
