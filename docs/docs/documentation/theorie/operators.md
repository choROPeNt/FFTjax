# Frequency-Space Operators

Every solver in FFTjax is built from a small set of frequency-space linear operators
(`operators/`), each implementing a common `LinearOperator` interface: `__call__` (apply),
`.T` (adjoint), and `@` (compose without evaluating either side, e.g. `A @ B` is itself a
`LinearOperator`). This page covers what those operators are and the one recurring numerical
gotcha (Nyquist handling) that several of them share; how each is *used* inside a solve is
covered on [Mechanical Solvers](mechanical), [Damage & Fracture Solvers](damage) and
[Thermal Conduction](thermal).

## Green's operator (Lippmann-Schwinger)

The reference-medium Green's operator $\hat{\Gamma}_0(\boldsymbol{\xi})$ for an isotropic
reference medium $(\lambda_0, \mu_0)$ (`operators/green.py`):

$$
K^{\text{eff}}_{ik} = \frac{\delta_{ik} - c\,\hat n_i \hat n_k}{\mu_0}, \qquad
\hat\Gamma_{0,ijkl} = \tfrac14\big(K^{\text{eff}}_{ik}\hat n_j \hat n_l + \text{3 sym. terms}\big),
\qquad c = \frac{\lambda_0+\mu_0}{\lambda_0+2\mu_0}
$$

built from the frequency **direction** $\hat n = \boldsymbol\xi/|\boldsymbol\xi|$, not
$\boldsymbol\xi$ itself: $\hat\Gamma_0 = \mathrm{sym}(\nabla)\!:\!G_0\!:\!\mathrm{sym}(\nabla)$ is
exactly degree-0 homogeneous in $\boldsymbol\xi$ for an isotropic reference medium, so the
$|\boldsymbol\xi|^2$ from the two gradients cancels the $1/|\boldsymbol\xi|^2$ in $G_0$ exactly.
Using $\boldsymbol\xi$ instead of $\hat n$ would silently reintroduce a spurious dependence on
domain length $L$ (since $|\boldsymbol\xi| \sim 1/L$ but direction does not). $\hat\Gamma_0$ is
major-symmetric and zero at the DC (zero-frequency) mode by construction, and is self-adjoint
(`GreenOperatorBasic.T` returns `self`).

**Discretization schemes:**

- `'standard'` (`GreenOperatorBasic`) — continuous DFT frequencies $\xi_j = 2\pi k_j/L_j$
  (Moulinec-Suquet). Prone to Gibbs oscillations and a 45° anisotropy bias at high stiffness
  contrast.
- `'rotated'` (`GreenOperatorWillot`, Willot 2015) — the finite-difference effective frequency
  $\xi_j^{\text{eff}} = (2/h_j)\sin(\xi_j h_j/2)$, equivalent to linear hexahedral elements with
  reduced integration. Eliminates the diagonal crack-propagation bias and improves CG convergence
  under high contrast; requires the voxel spacing $h_j = L_j/n_j$.

`build_reference_green_operator(n, L, materials, scheme=...)` builds the reference medium as a
Voigt-isotropization of the materials' mean stiffness tensor
(`materialmodels.tensors.isotropic_equivalent_lame`) — exact for isotropic materials, an
isotropizing approximation for anisotropic ones (e.g. `TransverseIsotropic`).

## Galerkin projector (reference-medium-free)

`operators/galerkin.py` implements the Vondřejc et al. (2014) Fourier-Galerkin scheme's symmetric
compatibility projector $\hat G_s$, which projects an arbitrary symmetric 2nd-order tensor field
onto its compatible (curl-free) part — purely geometric, no reference medium, no materials:

$$
\hat G_{s,ijkl} = \tfrac12\big(\delta_{ik}\hat n_j \hat n_l + \delta_{il}\hat n_j \hat n_k
+ \delta_{jk}\hat n_i \hat n_l + \delta_{jl}\hat n_i \hat n_k\big) - \hat n_i \hat n_j \hat n_k \hat n_l
$$

Major- and minor-symmetric, **idempotent** ($\hat G_s\!:\!\hat G_s = \hat G_s$, a true
projector), fixes any compatible field $\mathrm{sym}(\hat n \otimes v)$ exactly, annihilates any
$\hat n$-orthogonal field, and has trace 3 (a rank-3 projector on the 6-dimensional symmetric-
tensor space). $\hat\Gamma_0$ and $\hat G_s$ are genuinely different operators, not one a special
case of the other, at any choice of $(\lambda_0, \mu_0)$ — $\hat\Gamma_0$'s ansatz cannot
reproduce $\hat G_s$'s coefficients. Same `'standard'`/`'rotated'` scheme dispatch and zero-at-DC
convention as the Green's operator; both use only even powers of $\boldsymbol\xi$
($\xi\xi$, $\xi\xi\xi\xi$), so neither needs the Nyquist correction below.

## Applying an operator: Γ₀ as an FFT round trip

`Gamma0Operator` (`operators/projection.py`) applies *any* `LinearOperator` with a cached Fourier-
space tensor `.G` (a Green's operator or a Galerkin projector) to a real-space per-voxel tensor
field: `x -> ifftn(G(fftn(x)))`. It is self-adjoint whenever the wrapped operator is (forward/
inverse DFT are adjoints of each other on real fields). On more than one available device it
transparently domain-decomposes the FFT round trip across `jax.local_device_count()` devices
(slab decomposition) — a pure performance/memory-scaling detail that changes nothing about the
operator's math; every solver behaves identically on one device or many.

## The Nyquist gotcha

$\hat\Gamma_0$ and $\hat G_s$ only ever use **even** powers of $\boldsymbol\xi$
($\xi_i\xi_j$, $\xi_i\xi_j\xi_k\xi_l$), which is exactly what makes them provably symmetric
positive-definite (well-posed for CG) with no special handling at the grid's Nyquist frequency.
Any operator built from a plain **odd**-power gradient or divergence — the displacement-based
elastic solve, the heterogeneous-$G_c$ damage solve, steady-state thermal conduction — does not
have this luxury: for an even-length grid dimension, the Nyquist bin has no distinct negative-
frequency partner, so its FFT coefficient must be real for a real-input transform to stay
Hermitian-symmetric. Multiplying it by an odd (purely imaginary) power of $\boldsymbol\xi$ breaks
that symmetry and corrupts the real-space result. `operators.green.nyquist_safe_xi` zeros the
Nyquist component of $\boldsymbol\xi$ along every even dimension before such an operator ever uses
it — the standard fix in FFT-Galerkin homogenization, applied identically wherever a raw
gradient/divergence appears in this project.
